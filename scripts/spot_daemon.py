"""
Binance Spot Price Daemon
=========================
Connects via WebSocket to Binance combined stream for BTC/ETH/SOL 1m klines.
Maintains a rolling 75-minute buffer of close prices per symbol.
Writes atomically to /tmp/spot_buffer.json every tick.

Run as background process (managed by cron or systemd):
  uv run python scripts/spot_daemon.py

Buffer schema:
  {
    "updated_at": 1234567890,
    "btcusdt": [[ts_s, close], ...],   # ascending, ~75 entries
    "ethusdt": [[ts_s, close], ...],
    "solusdt": [[ts_s, close], ...],
  }

Each entry is the CLOSE price of the 1m candle starting at ts_s.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

import websockets

# ── Config ─────────────────────────────────────────────────────────────────────
BUFFER_PATH   = Path("/tmp/spot_buffer.json")
SYMBOLS       = ["btcusdt", "ethusdt", "solusdt"]
KEEP_MINUTES  = 75          # how many 1m candles to keep per symbol
WS_URL        = (
    "wss://stream.binance.com:9443/stream?streams="
    + "/".join(f"{s}@kline_1m" for s in SYMBOLS)
)
RECONNECT_DELAY = 5         # seconds before reconnect on error
WRITE_INTERVAL  = 1         # write buffer to disk at most every N seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spot_daemon")

# ── State ──────────────────────────────────────────────────────────────────────
# deque of [ts_s, close_price] — newest at right
buffers: dict[str, deque] = {s: deque(maxlen=KEEP_MINUTES) for s in SYMBOLS}
last_write = 0.0


def write_buffer() -> None:
    """Atomically write buffer to disk."""
    global last_write
    now = time.time()
    if now - last_write < WRITE_INTERVAL:
        return
    payload = {"updated_at": int(now)}
    for sym, dq in buffers.items():
        payload[sym] = list(dq)  # list of [ts_s, close]
    tmp = Path(str(BUFFER_PATH) + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(BUFFER_PATH)
    last_write = now


def process_kline(msg: dict) -> None:
    """Extract symbol + kline data and update buffer."""
    stream = msg.get("stream", "")
    symbol = stream.split("@")[0]  # e.g. "btcusdt"
    if symbol not in buffers:
        return
    k = msg.get("data", {}).get("k", {})
    if not k:
        return
    ts_s  = k["t"] // 1000         # candle open time in seconds
    close = float(k["c"])
    is_closed = k.get("x", False)  # True = candle finalized

    dq = buffers[symbol]

    # Update in-place if same candle, else append
    if dq and dq[-1][0] == ts_s:
        dq[-1][1] = close
    else:
        dq.append([ts_s, close])

    write_buffer()

    if is_closed:
        log.debug("%s candle closed @ %s: %.2f", symbol.upper(), ts_s, close)


async def run_stream() -> None:
    """Main WebSocket loop with auto-reconnect."""
    log.info("Connecting to Binance stream: %s", WS_URL[:80] + "...")
    while True:
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                log.info("Connected. Streaming BTC/ETH/SOL klines...")
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        process_kline(msg)
                    except Exception as e:
                        log.warning("parse error: %s", e)
        except (websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                OSError) as e:
            log.warning("WS disconnected: %s — reconnecting in %ds", e, RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)
        except Exception as e:
            log.error("Unexpected error: %s — reconnecting in %ds", e, RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)


def handle_signal(sig, frame):
    log.info("Signal %s received — shutting down", sig)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Seed buffer from REST on startup so first trade isn't blind
    try:
        import requests
        log.info("Seeding buffer from Binance REST API...")
        for sym in SYMBOLS:
            r = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": sym.upper(), "interval": "1m", "limit": KEEP_MINUTES},
                timeout=10,
            )
            if r.ok:
                for k in r.json():
                    ts_s  = k[0] // 1000
                    close = float(k[4])
                    buffers[sym].append([ts_s, close])
                log.info("  %s: seeded %d candles", sym.upper(), len(buffers[sym]))
    except Exception as e:
        log.warning("Seed failed (will fill from stream): %s", e)

    write_buffer()
    log.info("Buffer written to %s", BUFFER_PATH)

    asyncio.run(run_stream())
