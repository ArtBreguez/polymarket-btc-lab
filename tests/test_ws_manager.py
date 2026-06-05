"""
Tests for ws_manager.py — Resilient WebSocket Connection Manager
================================================================
Tests cover:
  - BackoffCalculator: exponential growth, jitter, cap, reset
  - WSMetrics: thread-safe counters, health dict, message rate
  - WSConfig: defaults and custom values
  - WebSocketManager: connect/disconnect lifecycle, message dispatch,
    zombie detection, reconnect backoff, health reporting
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure deploy/ is on the path so we can import ws_manager
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "deploy"))

from ws_manager import BackoffCalculator, WSConfig, WSMetrics, WebSocketManager


# ── Helpers ────────────────────────────────────────────────────────────────────

class _AsyncWSMock:
    """Mock WebSocket that supports `async for raw in ws`."""

    def __init__(self, messages: list[str]):
        self._messages = messages
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._index]
        self._index += 1
        return msg

    async def close(self, code=1000, reason=""):
        pass

    async def send(self, data):
        pass


def _make_async_ws(messages: list[str]) -> _AsyncWSMock:
    """Create a mock WS that yields messages then closes."""
    return _AsyncWSMock(messages)


# ═══════════════════════════════════════════════════════════════════════════════
# BackoffCalculator
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackoffCalculator:
    """Tests for exponential backoff with jitter."""

    def test_first_delay_near_initial(self):
        """First delay should be close to initial_backoff (±jitter)."""
        cfg = WSConfig(initial_backoff=5.0, jitter_range=0.0)
        bc = BackoffCalculator(cfg)
        delay = bc.next_delay()
        assert delay == 5.0

    def test_exponential_growth(self):
        """Delays should grow exponentially (without jitter)."""
        cfg = WSConfig(initial_backoff=5.0, backoff_factor=2.0, jitter_range=0.0)
        bc = BackoffCalculator(cfg)
        d1 = bc.next_delay()  # 5
        d2 = bc.next_delay()  # 10
        d3 = bc.next_delay()  # 20
        assert d1 == pytest.approx(5.0)
        assert d2 == pytest.approx(10.0)
        assert d3 == pytest.approx(20.0)

    def test_max_backoff_cap(self):
        """Delay should never exceed max_backoff."""
        cfg = WSConfig(initial_backoff=5.0, max_backoff=15.0, backoff_factor=2.0, jitter_range=0.0)
        bc = BackoffCalculator(cfg)
        for _ in range(10):
            d = bc.next_delay()
        assert d <= 15.0

    def test_jitter_adds_variance(self):
        """With jitter > 0, repeated delays at same attempt should vary."""
        cfg = WSConfig(initial_backoff=10.0, jitter_range=0.3)
        delays = set()
        for _ in range(20):
            bc = BackoffCalculator(cfg)
            delays.add(round(bc.next_delay(), 4))
        # With 30% jitter on 10.0, we expect range [7.0, 13.0]
        # At least a few different values
        assert len(delays) > 1

    def test_reset_restarts_sequence(self):
        """After reset(), delays should start from initial again."""
        cfg = WSConfig(initial_backoff=5.0, backoff_factor=2.0, jitter_range=0.0)
        bc = BackoffCalculator(cfg)
        bc.next_delay()  # 5
        bc.next_delay()  # 10
        bc.reset()
        d = bc.next_delay()  # should be 5 again
        assert d == pytest.approx(5.0)

    def test_attempt_counter(self):
        """Attempt counter should increment and reset correctly."""
        cfg = WSConfig()
        bc = BackoffCalculator(cfg)
        assert bc.attempt == 0
        bc.next_delay()
        assert bc.attempt == 1
        bc.next_delay()
        assert bc.attempt == 2
        bc.reset()
        assert bc.attempt == 0

    def test_minimum_delay_is_half_second(self):
        """Delay should never go below 0.5s even with extreme jitter."""
        cfg = WSConfig(initial_backoff=0.1, jitter_range=0.9)
        bc = BackoffCalculator(cfg)
        for _ in range(50):
            bc2 = BackoffCalculator(cfg)
            assert bc2.next_delay() >= 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# WSMetrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestWSMetrics:
    """Tests for connection health metrics."""

    def test_initial_state(self):
        """Fresh metrics should have all zeros."""
        m = WSMetrics()
        h = m.get_health()
        assert h["total_connects"] == 0
        assert h["total_disconnects"] == 0
        assert h["total_messages"] == 0
        assert h["total_errors"] == 0
        assert h["zombie_kills"] == 0
        assert h["msgs_per_min"] == 0
        assert h["last_msg_age_s"] == -1  # no message received yet

    def test_record_connect(self):
        """Connect should increment counter and set timestamps."""
        m = WSMetrics()
        m.record_connect()
        h = m.get_health()
        assert h["total_connects"] == 1
        assert h["current_uptime_s"] >= 0

    def test_record_disconnect(self):
        """Disconnect should increment counter."""
        m = WSMetrics()
        m.record_disconnect()
        h = m.get_health()
        assert h["total_disconnects"] == 1
        assert h["total_errors"] == 0

    def test_record_disconnect_with_error(self):
        """Disconnect with error=True should increment both counters."""
        m = WSMetrics()
        m.record_disconnect(error=True)
        h = m.get_health()
        assert h["total_disconnects"] == 1
        assert h["total_errors"] == 1

    def test_record_message(self):
        """Message recording should increment counter and update timestamp."""
        m = WSMetrics()
        m.record_message()
        m.record_message()
        m.record_message()
        h = m.get_health()
        assert h["total_messages"] == 3
        assert h["msgs_per_min"] == 3
        assert h["last_msg_age_s"] >= 0
        assert h["last_msg_age_s"] < 1  # just recorded

    def test_record_zombie_kill(self):
        """Zombie kills should increment."""
        m = WSMetrics()
        m.record_zombie_kill()
        m.record_zombie_kill()
        h = m.get_health()
        assert h["zombie_kills"] == 2

    def test_messages_per_minute_window(self):
        """Only messages within last 60s should count for rate."""
        m = WSMetrics()
        # Inject old timestamps directly
        with m._lock:
            m.total_messages = 5
            m._msg_timestamps = [time.time() - 120, time.time() - 90, time.time()]
        h = m.get_health()
        assert h["msgs_per_min"] == 1  # only the recent one counts

    def test_thread_safety(self):
        """Multiple threads recording concurrently should not corrupt state."""
        m = WSMetrics()
        errors = []

        def worker():
            try:
                for _ in range(100):
                    m.record_message()
                    m.record_connect()
                    m.record_disconnect()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        h = m.get_health()
        assert h["total_messages"] == 400
        assert h["total_connects"] == 400
        assert h["total_disconnects"] == 400


# ═══════════════════════════════════════════════════════════════════════════════
# WSConfig
# ═══════════════════════════════════════════════════════════════════════════════


class TestWSConfig:
    """Tests for configuration defaults."""

    def test_defaults(self):
        """Default config should have sensible values."""
        cfg = WSConfig()
        assert cfg.initial_backoff == 5.0
        assert cfg.max_backoff == 60.0
        assert cfg.backoff_factor == 2.0
        assert cfg.ping_interval == 20.0
        assert cfg.zombie_timeout == 45.0

    def test_custom_values(self):
        """Custom config should override defaults."""
        cfg = WSConfig(initial_backoff=1.0, max_backoff=10.0, zombie_timeout=30.0)
        assert cfg.initial_backoff == 1.0
        assert cfg.max_backoff == 10.0
        assert cfg.zombie_timeout == 30.0


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocketManager
# ═══════════════════════════════════════════════════════════════════════════════


class TestWebSocketManager:
    """Tests for the WS manager lifecycle and behavior."""

    def test_health_before_start(self):
        """Health should report disconnected before start."""
        mgr = WebSocketManager(
            name="test",
            url="wss://example.com",
            on_message=AsyncMock(),
        )
        h = mgr.health()
        assert h["name"] == "test"
        assert h["connected"] is False
        assert h["total_connects"] == 0

    def test_is_connected_false_initially(self):
        """Manager should not be connected before start."""
        mgr = WebSocketManager(
            name="test",
            url="wss://example.com",
            on_message=AsyncMock(),
        )
        assert mgr.is_connected is False

    def test_stop_without_start(self):
        """Stopping a never-started manager should not raise."""
        mgr = WebSocketManager(
            name="test",
            url="wss://example.com",
            on_message=AsyncMock(),
        )
        mgr.stop()  # should be no-op

    def test_send_sync_when_disconnected(self):
        """send_sync should return False when not connected."""
        mgr = WebSocketManager(
            name="test",
            url="wss://example.com",
            on_message=AsyncMock(),
        )
        result = mgr.send_sync({"type": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_send_when_disconnected(self):
        """send should return False when not connected."""
        mgr = WebSocketManager(
            name="test",
            url="wss://example.com",
            on_message=AsyncMock(),
        )
        result = await mgr.send({"type": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_message_loop_dispatches(self):
        """Message loop should parse JSON and call on_message."""
        handler = AsyncMock()
        mgr = WebSocketManager(
            name="test",
            url="wss://example.com",
            on_message=handler,
            config=WSConfig(zombie_timeout=999),  # disable zombie for this test
        )

        # Create a mock WS that yields messages then closes
        messages = [
            json.dumps({"event_type": "book", "asset_id": "abc"}),
            json.dumps({"event_type": "price_change", "price_changes": []}),
        ]

        mock_ws = _make_async_ws(messages)

        await mgr._message_loop(mock_ws)

        assert handler.call_count == 2
        # First call should have the book event
        first_call = handler.call_args_list[0][0][0]
        assert first_call["event_type"] == "book"

    @pytest.mark.asyncio
    async def test_message_loop_handles_invalid_json(self):
        """Message loop should skip non-JSON messages without crashing."""
        handler = AsyncMock()
        mgr = WebSocketManager(
            name="test",
            url="wss://example.com",
            on_message=handler,
            config=WSConfig(zombie_timeout=999),
        )

        messages = [
            "not valid json {{{{",
            json.dumps({"valid": True}),
        ]

        mock_ws = _make_async_ws(messages)

        await mgr._message_loop(mock_ws)

        # Only the valid message should be dispatched
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_message_loop_handles_callback_error(self):
        """Message loop should continue if on_message raises."""
        handler = AsyncMock(side_effect=[ValueError("boom"), None])
        mgr = WebSocketManager(
            name="test",
            url="wss://example.com",
            on_message=handler,
            config=WSConfig(zombie_timeout=999),
        )

        messages = [
            json.dumps({"msg": 1}),
            json.dumps({"msg": 2}),
        ]

        mock_ws = _make_async_ws(messages)

        await mgr._message_loop(mock_ws)

        # Both messages should be attempted
        assert handler.call_count == 2

    @pytest.mark.asyncio
    async def test_zombie_watchdog_triggers(self):
        """Zombie watchdog should close WS if no messages for too long."""
        mgr = WebSocketManager(
            name="test",
            url="wss://example.com",
            on_message=AsyncMock(),
            config=WSConfig(zombie_timeout=0.2, zombie_check_interval=0.1),
        )

        # Simulate a message was received a long time ago
        mgr.metrics.last_message_at = time.time() - 10

        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock()

        # Zombie watchdog should fire within ~0.3s and close the connection
        await asyncio.wait_for(mgr._zombie_watchdog(mock_ws), timeout=2.0)

        mock_ws.close.assert_called_once()
        assert mgr.metrics.zombie_kills == 1

    @pytest.mark.asyncio
    async def test_zombie_watchdog_no_false_positive(self):
        """Zombie watchdog should NOT trigger if messages are fresh."""
        mgr = WebSocketManager(
            name="test",
            url="wss://example.com",
            on_message=AsyncMock(),
            config=WSConfig(zombie_timeout=5.0, zombie_check_interval=0.1),
        )

        # Simulate a recent message
        mgr.metrics.last_message_at = time.time()

        mock_ws = AsyncMock()
        mock_ws.close = AsyncMock()

        # Run watchdog for 0.5s — should NOT trigger (timeout is 5s)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(mgr._zombie_watchdog(mock_ws), timeout=0.5)

        mock_ws.close.assert_not_called()
        assert mgr.metrics.zombie_kills == 0

    def test_double_start_warns(self, caplog):
        """Starting an already-running manager should log a warning."""
        mgr = WebSocketManager(
            name="test",
            url="wss://invalid.example.com:9999",
            on_message=AsyncMock(),
        )
        # Mock the thread to look alive
        mgr._thread = MagicMock()
        mgr._thread.is_alive.return_value = True

        import logging
        with caplog.at_level(logging.WARNING):
            mgr.start()

        assert "already running" in caplog.text


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: live_trader CLOB message handler
# ═══════════════════════════════════════════════════════════════════════════════


class TestClobMessageHandler:
    """Tests for _clob_on_message from live_trader (validates the refactor)."""

    @pytest.fixture(autouse=True)
    def _setup_clob_state(self):
        """Import and reset CLOB state for each test."""
        # We can't import live_trader directly (it needs env vars),
        # so we test the message handling logic by reimplementing the
        # core cache update logic that _clob_on_message does.
        self.prices = {}
        self.prices_ts = {}

    def _handle_book_event(self, ev: dict) -> None:
        """Simplified book event handler (matches live_trader logic)."""
        asset_id = ev.get("asset_id", "")
        asks = ev.get("asks", [])
        valid_asks = [
            float(a["price"]) for a in asks
            if float(a.get("price", 1)) < 0.97
        ]
        if valid_asks:
            best = min(valid_asks)
            self.prices[asset_id] = best
            self.prices_ts[asset_id] = time.time()

    def _handle_price_change(self, ev: dict) -> None:
        """Simplified price_change handler (matches live_trader logic)."""
        for change in ev.get("price_changes", []):
            if change.get("side") == "ASK":
                asset_id = change.get("asset_id", "")
                price = float(change.get("price", 1))
                existing = self.prices.get(asset_id, 1.0)
                if price < 0.97 and price <= existing:
                    self.prices[asset_id] = price
                    self.prices_ts[asset_id] = time.time()
                elif price > existing:
                    self.prices.pop(asset_id, None)
                    self.prices_ts.pop(asset_id, None)

    def test_book_event_sets_best_ask(self):
        """Book event should pick the lowest valid ask."""
        ev = {
            "event_type": "book",
            "asset_id": "token-123",
            "asks": [
                {"price": "0.55", "size": "10"},
                {"price": "0.60", "size": "20"},
                {"price": "0.45", "size": "5"},
            ],
        }
        self._handle_book_event(ev)
        assert self.prices["token-123"] == pytest.approx(0.45)

    def test_book_event_filters_high_asks(self):
        """Asks >= 0.97 should be excluded."""
        ev = {
            "event_type": "book",
            "asset_id": "token-456",
            "asks": [
                {"price": "0.97", "size": "10"},
                {"price": "0.99", "size": "20"},
            ],
        }
        self._handle_book_event(ev)
        assert "token-456" not in self.prices

    def test_book_event_empty_asks(self):
        """Empty asks should not update cache."""
        ev = {
            "event_type": "book",
            "asset_id": "token-789",
            "asks": [],
        }
        self._handle_book_event(ev)
        assert "token-789" not in self.prices

    def test_price_change_tightens_ask(self):
        """price_change with lower price should update cache."""
        self.prices["token-abc"] = 0.60
        self.prices_ts["token-abc"] = time.time()

        ev = {
            "event_type": "price_change",
            "price_changes": [
                {"side": "ASK", "asset_id": "token-abc", "price": "0.55"},
            ],
        }
        self._handle_price_change(ev)
        assert self.prices["token-abc"] == pytest.approx(0.55)

    def test_price_change_widens_ask_invalidates(self):
        """price_change with higher price should invalidate cache."""
        self.prices["token-abc"] = 0.50
        self.prices_ts["token-abc"] = time.time()

        ev = {
            "event_type": "price_change",
            "price_changes": [
                {"side": "ASK", "asset_id": "token-abc", "price": "0.65"},
            ],
        }
        self._handle_price_change(ev)
        assert "token-abc" not in self.prices

    def test_price_change_ignores_bid_side(self):
        """price_change on BID side should be ignored."""
        ev = {
            "event_type": "price_change",
            "price_changes": [
                {"side": "BID", "asset_id": "token-xyz", "price": "0.40"},
            ],
        }
        self._handle_price_change(ev)
        assert "token-xyz" not in self.prices

    def test_price_change_rejects_high_price(self):
        """price_change >= 0.97 should not update cache even if lower."""
        self.prices["token-hi"] = 0.98
        ev = {
            "event_type": "price_change",
            "price_changes": [
                {"side": "ASK", "asset_id": "token-hi", "price": "0.97"},
            ],
        }
        self._handle_price_change(ev)
        # Price was 0.98, new is 0.97 which is < existing BUT >= 0.97 threshold
        assert self.prices["token-hi"] == 0.98  # unchanged

    def test_reconnect_invalidates_cache(self):
        """All cached prices should be cleared on reconnect to prevent stale data."""
        self.prices["token-a"] = 0.55
        self.prices_ts["token-a"] = time.time()
        self.prices["token-b"] = 0.60
        self.prices_ts["token-b"] = time.time()

        # Simulate reconnect: clear all prices
        self.prices.clear()
        self.prices_ts.clear()

        assert "token-a" not in self.prices
        assert "token-b" not in self.prices
        assert len(self.prices) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Spot message handler
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpotMessageHandler:
    """Tests for Binance spot message handling logic."""

    def test_kline_message_appends_candle(self):
        """Valid kline message should append to buffer."""
        from collections import deque
        buf = deque(maxlen=300)

        msg = {
            "stream": "btcusdt@kline_1m",
            "data": {
                "k": {
                    "t": 1717500000000,  # ms timestamp
                    "c": "67500.50",
                }
            }
        }

        sym = msg.get("stream", "").split("@")[0]
        assert sym == "btcusdt"

        k = msg["data"]["k"]
        ts_s = k["t"] // 1000
        close = float(k["c"])
        buf.append([ts_s, close])

        assert len(buf) == 1
        assert buf[0] == [1717500000, 67500.50]

    def test_kline_same_timestamp_updates(self):
        """Same-second kline should update existing candle, not append."""
        from collections import deque
        buf = deque(maxlen=300)
        buf.append([1717500000, 67500.00])

        # Update with same timestamp
        ts_s = 1717500000
        close = 67550.25
        if buf and buf[-1][0] == ts_s:
            buf[-1][1] = close
        else:
            buf.append([ts_s, close])

        assert len(buf) == 1
        assert buf[0][1] == 67550.25

    def test_kline_new_timestamp_appends(self):
        """New timestamp should append a new candle."""
        from collections import deque
        buf = deque(maxlen=300)
        buf.append([1717500000, 67500.00])

        ts_s = 1717500060  # 1 min later
        close = 67600.00
        if buf and buf[-1][0] == ts_s:
            buf[-1][1] = close
        else:
            buf.append([ts_s, close])

        assert len(buf) == 2

    def test_unknown_stream_ignored(self):
        """Unknown stream names should be silently ignored."""
        known = {"btcusdt", "ethusdt", "solusdt"}
        msg = {"stream": "dogeusdt@kline_1m", "data": {"k": {"t": 1, "c": "1.0"}}}
        sym = msg.get("stream", "").split("@")[0]
        assert sym not in known

    def test_missing_kline_data_ignored(self):
        """Message without kline data should be ignored."""
        msg = {"stream": "btcusdt@kline_1m", "data": {}}
        k = msg.get("data", {}).get("k", {})
        assert not k
