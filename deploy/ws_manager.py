"""
ws_manager.py — Resilient WebSocket Connection Manager
=======================================================
Provides a robust, reusable WS connection layer with:

  - Exponential backoff with jitter on reconnect (5s → 10s → 20s → max 60s)
  - Backoff resets after first successful message received
  - Connection health metrics (uptime, disconnect count, msgs/min, last_msg_age)
  - Zombie connection detection (no data received for configurable threshold)
  - Structured logging with connection lifecycle events
  - Graceful shutdown support

Usage:
    manager = WebSocketManager(
        name="clob",
        url="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        on_message=handle_msg,
        on_connect=subscribe_tokens,
        zombie_timeout=45.0,
    )
    manager.start()  # starts background thread
    ...
    print(manager.health())  # {"status": "connected", "uptime_s": 3600, ...}
    manager.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

try:
    from websockets.exceptions import InvalidStatusCode
except ImportError:
    InvalidStatusCode = None  # type: ignore[misc,assignment]

log = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class WSConfig:
    """WebSocket connection configuration."""
    # Reconnect backoff
    initial_backoff: float = 5.0       # first retry delay (seconds)
    max_backoff: float = 60.0          # cap on retry delay
    backoff_factor: float = 2.0        # exponential multiplier
    jitter_range: float = 0.3          # ±30% randomization on delays

    # Ping/pong (RFC 6455 keepalive)
    # Set to None to let the SERVER control ping/pong (recommended for Polymarket).
    # The websockets lib still responds to server pings even with None.
    ping_interval: Optional[float] = 20.0
    ping_timeout: Optional[float] = 10.0
    close_timeout: float = 5.0

    # Zombie detection
    zombie_timeout: float = 45.0       # no message for N seconds → force reconnect
    zombie_check_interval: float = 5.0 # how often to check for zombie state

    # Health logging
    health_log_interval: float = 300.0 # log health summary every N seconds (5 min)


# ── Health Metrics ─────────────────────────────────────────────────────────────

@dataclass
class WSMetrics:
    """Thread-safe connection health metrics."""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Counters
    total_connects: int = 0
    total_disconnects: int = 0
    total_messages: int = 0
    total_errors: int = 0
    zombie_kills: int = 0

    # Timestamps
    first_connect_at: float = 0.0
    current_connect_at: float = 0.0
    last_message_at: float = 0.0
    last_disconnect_at: float = 0.0
    last_error_at: float = 0.0

    # Rolling message rate (last 60s window)
    _msg_timestamps: list[float] = field(default_factory=list, repr=False)

    def record_connect(self) -> None:
        with self._lock:
            self.total_connects += 1
            now = time.time()
            self.current_connect_at = now
            if self.first_connect_at == 0:
                self.first_connect_at = now

    def record_disconnect(self, error: bool = False) -> None:
        with self._lock:
            self.total_disconnects += 1
            self.last_disconnect_at = time.time()
            if error:
                self.total_errors += 1
                self.last_error_at = time.time()

    def record_message(self) -> None:
        with self._lock:
            self.total_messages += 1
            now = time.time()
            self.last_message_at = now
            self._msg_timestamps.append(now)
            # Prune older than 60s
            cutoff = now - 60.0
            self._msg_timestamps = [t for t in self._msg_timestamps if t > cutoff]

    def record_zombie_kill(self) -> None:
        with self._lock:
            self.zombie_kills += 1

    def get_health(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            uptime = (now - self.current_connect_at) if self.current_connect_at else 0
            last_msg_age = (now - self.last_message_at) if self.last_message_at else -1
            # Messages per minute (last 60s window)
            cutoff = now - 60.0
            recent = [t for t in self._msg_timestamps if t > cutoff]
            msgs_per_min = len(recent)

            return {
                "total_connects": self.total_connects,
                "total_disconnects": self.total_disconnects,
                "total_messages": self.total_messages,
                "total_errors": self.total_errors,
                "zombie_kills": self.zombie_kills,
                "current_uptime_s": round(uptime, 1),
                "last_msg_age_s": round(last_msg_age, 1),
                "msgs_per_min": msgs_per_min,
            }


# ── Backoff Calculator ─────────────────────────────────────────────────────────

class BackoffCalculator:
    """Exponential backoff with jitter, resets on success."""

    def __init__(self, config: WSConfig):
        self._config = config
        self._attempt = 0

    def next_delay(self) -> float:
        """Calculate next retry delay with jitter."""
        base = min(
            self._config.initial_backoff * (self._config.backoff_factor ** self._attempt),
            self._config.max_backoff,
        )
        jitter = base * self._config.jitter_range * (2 * random.random() - 1)
        self._attempt += 1
        return max(0.5, base + jitter)

    def reset(self) -> None:
        """Reset backoff after successful connection + first message."""
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt


# ── WebSocket Manager ──────────────────────────────────────────────────────────

class WebSocketManager:
    """Manages a single resilient WebSocket connection in a background thread.

    Parameters
    ----------
    name : str
        Human-readable name for logging (e.g. "clob", "binance-spot").
    url : str
        WebSocket URL to connect to.
    on_message : callable
        Async callback: ``async def handler(msg: dict | list) -> None``
        Called for every parsed JSON message received.
    on_connect : callable, optional
        Async callback: ``async def on_connect(ws) -> None``
        Called after each successful connection (use to send subscriptions).
    on_disconnect : callable, optional
        Sync callback: ``def on_disconnect(error: Exception | None) -> None``
        Called when connection drops.
    config : WSConfig, optional
        Connection configuration. Uses defaults if not provided.
    """

    def __init__(
        self,
        name: str,
        url: str,
        on_message: Callable[[Any], Awaitable[None]],
        on_connect: Optional[Callable[[Any], Awaitable[None]]] = None,
        on_disconnect: Optional[Callable[[Optional[Exception]], None]] = None,
        config: Optional[WSConfig] = None,
    ):
        self.name = name
        self.url = url
        self._on_message = on_message
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self.config = config or WSConfig()
        self.metrics = WSMetrics()
        self._backoff = BackoffCalculator(self.config)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ws: Optional[Any] = None  # current websocket connection
        self._ws_lock = threading.Lock()

    def start(self) -> None:
        """Start the WS connection in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            log.warning("[%s] WS manager already running", self.name)
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=f"ws-{self.name}",
        )
        self._thread.start()
        log.info("[%s] WS manager started", self.name)

    def stop(self) -> None:
        """Signal the WS manager to stop and wait for thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            log.info("[%s] WS manager stopped", self.name)

    @property
    def is_connected(self) -> bool:
        with self._ws_lock:
            return self._ws is not None

    def health(self) -> dict[str, Any]:
        """Return current health metrics."""
        h = self.metrics.get_health()
        h["name"] = self.name
        h["connected"] = self.is_connected
        h["backoff_attempt"] = self._backoff.attempt
        return h

    async def send(self, data: dict | list | str) -> bool:
        """Send data through the current WS connection. Returns True on success."""
        with self._ws_lock:
            ws = self._ws
        if ws is None:
            log.warning("[%s] send() called but not connected", self.name)
            return False
        try:
            payload = json.dumps(data) if not isinstance(data, str) else data
            await ws.send(payload)
            return True
        except Exception as e:
            log.warning("[%s] send() failed: %s", self.name, e)
            return False

    def send_sync(self, data: dict | list | str) -> bool:
        """Thread-safe synchronous send (queues into the event loop)."""
        with self._ws_lock:
            ws = self._ws
        if ws is None:
            return False
        try:
            payload = json.dumps(data) if not isinstance(data, str) else data
            # Use the running loop from the daemon thread
            loop = self._loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(ws.send(payload), loop).result(timeout=5)
                return True
        except Exception as e:
            log.warning("[%s] send_sync() failed: %s", self.name, e)
        return False

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Thread entry: run the async event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as e:
            log.error("[%s] Event loop crashed: %s", self.name, e)
        finally:
            self._loop.close()

    async def _connect_loop(self) -> None:
        """Main reconnect loop with exponential backoff."""
        last_health_log = 0.0

        while not self._stop_event.is_set():
            error: Optional[Exception] = None
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                    close_timeout=self.config.close_timeout,
                ) as ws:
                    with self._ws_lock:
                        self._ws = ws
                    self.metrics.record_connect()
                    self._backoff.reset()
                    log.info(
                        "[%s] Connected (attempt #%d, total connects: %d)",
                        self.name,
                        self._backoff.attempt + 1,
                        self.metrics.total_connects,
                    )

                    # Fire on_connect callback (e.g., send subscriptions)
                    if self._on_connect:
                        try:
                            await self._on_connect(ws)
                        except Exception as e:
                            log.error("[%s] on_connect callback failed: %s", self.name, e)

                    # Main message loop with zombie detection
                    await self._message_loop(ws)

            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as e:
                error = e
                log.warning(
                    "[%s] Connection closed: code=%s reason=%s",
                    self.name,
                    getattr(e, "code", "?"),
                    getattr(e, "reason", str(e)),
                )
            except OSError as e:
                error = e
                log.warning("[%s] Network error: %s", self.name, e)
            except Exception as e:
                error = e
                if InvalidStatusCode and isinstance(e, InvalidStatusCode):
                    log.warning(
                        "[%s] Connection rejected: HTTP %s",
                        self.name,
                        getattr(e, "status_code", "?"),
                    )
                else:
                    log.warning("[%s] Unexpected error: %s (%s)", self.name, type(e).__name__, e)
            finally:
                with self._ws_lock:
                    self._ws = None
                self.metrics.record_disconnect(error=error is not None)

                if self._on_disconnect:
                    try:
                        self._on_disconnect(error)
                    except Exception:
                        pass

            if self._stop_event.is_set():
                break

            # Exponential backoff before reconnect
            delay = self._backoff.next_delay()
            log.info(
                "[%s] Reconnecting in %.1fs (attempt #%d, disconnects: %d, errors: %d)",
                self.name,
                delay,
                self._backoff.attempt,
                self.metrics.total_disconnects,
                self.metrics.total_errors,
            )
            # Use stop_event wait so we can interrupt during backoff
            self._stop_event.wait(timeout=delay)

    async def _message_loop(self, ws) -> None:
        """Process messages with zombie detection and periodic health logging."""
        last_health_log = time.time()

        # Start zombie detector as background task
        zombie_task = asyncio.create_task(self._zombie_watchdog(ws))

        try:
            async for raw in ws:
                if self._stop_event.is_set():
                    return

                self.metrics.record_message()

                # Parse and dispatch
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    log.debug("[%s] Non-JSON message: %s", self.name, raw[:100] if raw else "")
                    continue

                try:
                    await self._on_message(msg)
                except Exception as e:
                    log.warning("[%s] on_message callback error: %s", self.name, e)

                # Periodic health log
                now = time.time()
                if now - last_health_log >= self.config.health_log_interval:
                    h = self.health()
                    log.info(
                        "[%s] Health: uptime=%ds msgs=%d rate=%d/min disconnects=%d errors=%d zombies=%d",
                        self.name,
                        h["current_uptime_s"],
                        h["total_messages"],
                        h["msgs_per_min"],
                        h["total_disconnects"],
                        h["total_errors"],
                        h["zombie_kills"],
                    )
                    last_health_log = now
        finally:
            zombie_task.cancel()
            try:
                await zombie_task
            except asyncio.CancelledError:
                pass

        # If async for exits cleanly, the connection was closed by the server
        log.info("[%s] Server closed connection gracefully", self.name)

    async def _zombie_watchdog(self, ws) -> None:
        """Periodically check if the connection is zombie (connected but no data).

        ACTIVE zombie detection: before killing, sends a protocol-level ping
        and waits for pong. If pong comes back, the connection is alive — just
        no *data* messages (common on illiquid/one-sided markets). Only kills
        if ping also fails (true zombie: connection silently dead).

        STALE DATA guard: if ping/pong succeeds but no real data messages
        arrive for too many consecutive checks, force reconnect.
        The Polymarket WS can keep TCP alive via ping/pong but silently stop
        sending book/price_change events — a reconnect fixes this.

        KEY FIX: We do NOT reset last_message_at on ping/pong success.
        Instead we track message count to detect actual data flow resumption.
        Previous bug: resetting last_message_at created a 120s blind window
        each cycle, so consecutive_quiet never accumulated past 1.
        """
        MAX_QUIET_BEFORE_RECONNECT = 3   # 3 checks ≈ 6 min max
        STALE_CHECK_INTERVAL = 120.0     # seconds between stale-data checks

        # Wait for initial messages before starting checks
        await asyncio.sleep(self.config.zombie_timeout)

        consecutive_quiet = 0
        last_seen_count = self.metrics.total_messages

        while True:
            await asyncio.sleep(STALE_CHECK_INTERVAL)
            if self.metrics.last_message_at == 0:
                continue  # Haven't received first message yet

            # Check if new data messages arrived since last check
            current_count = self.metrics.total_messages
            if current_count > last_seen_count:
                # Real data flowing — reset quiet counter
                consecutive_quiet = 0
                last_seen_count = current_count
                continue

            age = time.time() - self.metrics.last_message_at
            if age <= self.config.zombie_timeout:
                continue

            # No new data for zombie_timeout+ seconds — active probe
            try:
                pong = await ws.ping()
                await asyncio.wait_for(pong, timeout=10.0)
                # Pong received — connection alive but no data messages
                consecutive_quiet += 1
                if consecutive_quiet >= MAX_QUIET_BEFORE_RECONNECT:
                    log.warning(
                        "[%s] No data for %d consecutive checks (~%.0fs total) despite ping/pong OK. "
                        "Force reconnecting to restore data flow.",
                        self.name,
                        consecutive_quiet,
                        age,
                    )
                    self.metrics.record_zombie_kill()
                    await ws.close(code=4001, reason="stale-data-reconnect")
                    return
                log.info(
                    "[%s] No data for %.0fs (0 new msgs in %ds), ping/pong OK — stale data (%d/%d)",
                    self.name,
                    age,
                    int(STALE_CHECK_INTERVAL),
                    consecutive_quiet,
                    MAX_QUIET_BEFORE_RECONNECT,
                )
                # DO NOT reset last_message_at here — that was the root bug.
                # The next check in STALE_CHECK_INTERVAL will re-evaluate.
                continue
            except (asyncio.TimeoutError, Exception) as e:
                log.warning(
                    "[%s] Zombie confirmed — no message for %.0fs AND ping failed (%s). Force closing.",
                    self.name,
                    age,
                    e,
                )
                self.metrics.record_zombie_kill()
                await ws.close(code=4000, reason="zombie-timeout")
                return
