"""
live_trader.py — BTC Directional Live Trader v3
================================================
Runs the v8 ML model on Polymarket BTC 5-minute markets.

Strategy:
  - Observe first 3 min of each 5m slot (order flow from CLOB)
  - Predict UP or DOWN at t=170–240s
  - Enter only if confidence > 60% AND edge (model_prob - ask_price) >= 10%
  - Flat stake: $1.50 USDC per trade
  - Settle 60s after slot end

Credentials (Fly secrets):
  POLY_PRIVATE_KEY      — EOA private key
  POLY_SAFE_ADDRESS     — Proxy wallet (0x362095...)
  MM_BUILDER_KEY        — Builder API key
  MM_BUILDER_SECRET     — Builder API secret
  MM_BUILDER_PASSPHRASE — Builder API passphrase

Model: downloaded from HuggingFace on startup (artbreguez/polymarket-btc-model).
Spot data: Binance WebSocket stream (btcusdt @kline_1m) running as a background
           thread, writing to /tmp/spot_buffer.json. BTC buffer = 300 candles (5h)
           to support btc_pre_4h_ret feature.

Feature parity: build_features() matches train_v8_modal.py tick_features_v8() exactly:
  - 6x30s sub-windows (btc_up_w0..w5) + per-window zscores
  - multi-scale up_ratio zscore (5/10/20 slots)
  - time-weighted order flow, VWAP trend, volume-weighted momentum
  - lag outcomes + lag streak from _slot_history ring buffer
  - BTC-only spot: inslot_ret/vol, pre_5m/15m/30m/1h/4h_ret/vol, dist_1k/5k/10k
  - OB features filled with neutral defaults (not available live)
"""

import gc
import json
import logging
import math
import os
import pickle
import queue as _queue_mod
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import asyncio
import numpy as np
import pandas as pd
from clob_features import get_accumulator, FEATURE_NAMES as CLOB_FEATURE_NAMES, OB_PC_FEATURE_NAMES
from clob_feature_logger import log_clob_features, resolve_clob_features
from data_quality_gate import DataQualityGate
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
# In DRY_RUN mode credentials are optional — checked at startup
PRIVATE_KEY    = os.environ.get("POLY_PRIVATE_KEY", "")
PROXY_WALLET   = os.environ.get("POLY_SAFE_ADDRESS", "")
BUILDER_KEY    = os.environ.get("MM_BUILDER_KEY", "")
BUILDER_SECRET = os.environ.get("MM_BUILDER_SECRET", "")
BUILDER_PASS   = os.environ.get("MM_BUILDER_PASSPHRASE", "")

GAMMA_HOST      = "https://gamma-api.polymarket.com"
DATA_API        = "https://data-api.polymarket.com"
CLOB_URL        = "https://clob.polymarket.com"
CLOB_WS_URL     = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
# Binance WS/REST são bloqueados pelo Fly AMS (HTTP 451 geo-block).
# Bybit WS não tem geo-block e tem o mesmo stream de kline 1m.
BINANCE_WS      = "wss://stream.bybit.com/v5/public/spot"   # Bybit (sem geo-block)
BINANCE_REST    = "https://api.bybit.com/v5/market"          # Bybit REST fallback

HTTP_TIMEOUT    = 3    # seconds — tight budget for a 10s loop
MAX_TRADE_PAGES = 8    # max 8 pages/token × 2 tokens = 16 calls max (~4000 trades/token)
                       # Keeps fetch_inslot_trades well under 30s even on slow days.
                       # NOTE: data-api returns trades in random order (not chronological),
                       # so we must page through all pages to collect inslot trades.

SLOT_DURATION   = 300
OBSERVE_SECS    = 60   # v23: matches model training (was 180 for v22, caused data mismatch)
ENTER_WINDOW    = (170, 240)  # Keep same window: data-api lag ~120s means ticks from t=0-60 arrive at t~180
SETTLE_GRACE    = 60
MIN_CONFIDENCE  = 0.55   # v26: lowered from 60% — backtest shows 55% profitable with edge>7%
MIN_EDGE        = 0.07   # v26: kept at 7% — backtest confirms 7%+ edge is profitable
MIN_EDGE_MID    = 0.05          # edge vs market mid (catches stale ask)
TAKER_FEE       = 0.02          # Polymarket taker fee default ~2% (updated dynamically)

# Fee rate cache — fetched from CLOB at startup and refreshed periodically
_fee_rate_cache: float = TAKER_FEE
_fee_rate_ts: float = 0.0
STAKE_USDC      = 1.50          # capped at $1.50 per trade
BUFFER_STALE    = 120

# ── Auto-sizing shares ────────────────────────────────────────────────────────
# When AUTO_SHARES=true, shares scale linearly with balance between MIN/MAX.
# When AUTO_SHARES=false (default), FIXED_SHARES is used.
AUTO_SHARES     = os.environ.get("AUTO_SHARES", "false").lower() in ("true", "1", "yes")
FIXED_SHARES    = float(os.environ.get("FIXED_SHARES", "8"))
AUTO_SHARES_MIN = float(os.environ.get("AUTO_SHARES_MIN", "5"))
AUTO_SHARES_MAX = float(os.environ.get("AUTO_SHARES_MAX", "40"))
# Balance anchors: at BAL_FLOOR shares=MIN, at BAL_CEIL shares=MAX, linear between
AUTO_SHARES_BAL_FLOOR = float(os.environ.get("AUTO_SHARES_BAL_FLOOR", "20"))
AUTO_SHARES_BAL_CEIL  = float(os.environ.get("AUTO_SHARES_BAL_CEIL", "700"))

MODEL_PATH  = Path("/tmp/champion.pkl")
HF_REPO     = "artbreguez/polymarket-btc-model"
HF_TOKEN    = os.environ.get("HF_TOKEN")
TRADES_FILE = Path("/tmp/live_trades.json")
SPOT_BUFFER = Path("/tmp/spot_buffer.json")

# ── Dry Run mode ───────────────────────────────────────────────────────────────
# DRY_RUN=true  → full pipeline runs (features, model, price) but NO real orders.
#                 Trades are recorded as "dry_open" and settled via Gamma outcome.
# DRY_VIRTUAL_BALANCE=100 → simulated USDC wallet balance for sizing.
DRY_RUN             = os.environ.get("DRY_RUN", "false").lower() in ("true", "1", "yes")
DRY_VIRTUAL_BALANCE = float(os.environ.get("DRY_VIRTUAL_BALANCE", "100.0"))
_dry_virtual_balance = DRY_VIRTUAL_BALANCE   # mutable — updated on each dry settle

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("live_trader")

# Shared HTTP session — connection keep-alive, no TCP handshake per call
_http = requests.Session()
_http.headers.update({"User-Agent": "polymarket-btc-trader/2.0"})


# ── Spot daemon (background thread) ───────────────────────────────────────────
from ws_manager import WebSocketManager, WSConfig

_spot_buffers: dict[str, deque] = {
    "btcusdt": deque(maxlen=300),  # 5h of 1m candles — needed for btc_pre_4h_ret
}
_spot_last_write = 0.0

def _write_spot_buffer():
    global _spot_last_write
    now = time.time()
    if now - _spot_last_write < 1.0:
        return
    payload = {"updated_at": int(now)}
    for sym, dq in _spot_buffers.items():
        payload[sym] = list(dq)
    tmp = Path(str(SPOT_BUFFER) + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(SPOT_BUFFER)
    _spot_last_write = now

def _seed_spot_buffers():
    """Seed spot buffers via Bybit REST on startup."""
    for sym in _spot_buffers:
        limit = 300 if sym == "btcusdt" else 75
        try:
            r = _http.get(f"{BINANCE_REST}/kline",
                          params={"category": "spot", "symbol": sym.upper(),
                                  "interval": "1", "limit": limit},
                          timeout=HTTP_TIMEOUT)
            if r.ok:
                candles = r.json().get("result", {}).get("list", [])
                for k in candles:
                    # Bybit list: [startMs, open, high, low, close, volume, turnover]
                    _spot_buffers[sym].append([int(k[0]) // 1000, float(k[4]), float(k[2]), float(k[3]), float(k[5])])
                log.info("Spot seed %s: %d candles (Bybit)", sym.upper(), len(_spot_buffers[sym]))
        except Exception as e:
            log.warning("Spot seed %s failed: %s", sym, e)
    _write_spot_buffer()


async def _spot_on_message(msg: dict) -> None:
    """Handle Bybit kline WS message.

    Bybit format:
      {"topic": "kline.1.BTCUSDT", "data": [{"start": ms, "open": "...", "close": "...",
       "high": "...", "low": "...", "volume": "...", "confirm": bool, ...}]}
    Fields map directly to what we used from Binance.
    """
    if not isinstance(msg, dict):
        return
    # Subscribe ACK / other control messages
    topic = msg.get("topic", "")
    if not topic.startswith("kline"):
        return
    data_list = msg.get("data", [])
    if not data_list:
        return
    k = data_list[0]
    # Infer symbol: "kline.1.BTCUSDT" → "btcusdt"
    sym = topic.split(".")[-1].lower()   # "btcusdt"
    if sym not in _spot_buffers:
        return
    ts_s = k["start"] // 1000
    close = float(k["close"])
    high = float(k["high"])
    low = float(k["low"])
    volume = float(k["volume"])
    dq = _spot_buffers[sym]
    if dq and dq[-1][0] == ts_s:
        dq[-1][1] = close
        if len(dq[-1]) >= 4:
            dq[-1][2] = max(dq[-1][2], high)
            dq[-1][3] = min(dq[-1][3], low)
        else:
            dq[-1].extend([high, low])
        # Update volume (always latest cumulative from WS)
        if len(dq[-1]) >= 5:
            dq[-1][4] = volume
        else:
            dq[-1].append(volume)
    else:
        dq.append([ts_s, close, high, low, volume])
    _write_spot_buffer()


async def _spot_on_connect(ws) -> None:
    """Subscribe to Bybit kline stream on connect."""
    await ws.send(json.dumps({
        "op": "subscribe",
        "args": ["kline.1.BTCUSDT"]
    }))
    log.info("Bybit WS subscribed kline.1.BTCUSDT")


def _spot_rest_poll():
    """Fallback: poll Bybit REST API for latest klines when WS is disconnected."""
    while True:
        try:
            for sym in _spot_buffers:
                # Bybit REST: /v5/market/kline?category=spot&symbol=BTCUSDT&interval=1&limit=5
                r = _http.get(f"{BINANCE_REST}/kline",
                              params={"category": "spot", "symbol": sym.upper(),
                                      "interval": "1", "limit": 5},
                              timeout=HTTP_TIMEOUT + 2)
                if r.ok:
                    data = r.json()
                    # Bybit returns {"result": {"list": [[startMs, open, high, low, close, vol, turnover], ...]}}
                    candles = data.get("result", {}).get("list", [])
                    for k in candles:
                        ts_s  = int(k[0]) // 1000
                        close = float(k[4])
                        high  = float(k[2])
                        low   = float(k[3])
                        volume= float(k[5])
                        dq = _spot_buffers[sym]
                        if dq and dq[-1][0] == ts_s:
                            dq[-1][1] = close
                            if len(dq[-1]) >= 4:
                                dq[-1][2] = max(dq[-1][2], high)
                                dq[-1][3] = min(dq[-1][3], low)
                            else:
                                dq[-1].extend([high, low])
                            if len(dq[-1]) >= 5:
                                dq[-1][4] = volume
                            else:
                                dq[-1].append(volume)
                        elif not dq or ts_s > dq[-1][0]:
                            dq.append([ts_s, close, high, low, volume])
                    _write_spot_buffer()
        except Exception as e:
            log.warning("Spot REST poll error: %s", e)
        time.sleep(30)


_spot_ws_manager: WebSocketManager | None = None

def start_spot_daemon():
    global _spot_ws_manager

    # Seed spot buffers via REST before WS connects
    _seed_spot_buffers()

    _spot_ws_manager = WebSocketManager(
        name="bybit-spot",
        url=BINANCE_WS,
        on_message=_spot_on_message,
        on_connect=_spot_on_connect,
        config=WSConfig(
            ping_interval=20.0,
            ping_timeout=10.0,
            zombie_timeout=60.0,       # Bybit sends klines every ~2s
            health_log_interval=300.0,
        ),
    )
    _spot_ws_manager.start()
    log.info("Spot daemon started (ws_manager)")

    # Start REST fallback poller for when WS is geo-blocked (HTTP 451)
    rest_thread = threading.Thread(target=_spot_rest_poll, daemon=True, name="spot-rest-poll")
    rest_thread.start()
    log.info("Spot REST fallback poller started (every 30s)")

    # Wait up to 10s for initial seed
    for _ in range(10):
        if SPOT_BUFFER.exists():
            return
        time.sleep(1)
    log.warning("Spot buffer not ready after 10s — continuing anyway")


# ── CLOB WebSocket daemon (background thread) ──────────────────────────────────
_clob_prices: dict[str, float] = {}
_clob_price_ts: dict[str, float] = {}   # token_id → timestamp of last price update
_clob_prices_lock = threading.Lock()
_clob_subscribed: set[str] = set()
_token_slot: dict[str, int] = {}   # token_id → slot_ts for pruning
_subscribe_queue: _queue_mod.Queue = _queue_mod.Queue()
_clob_ws_manager: WebSocketManager | None = None
_clob_last_keepalive: float = 0.0


def clob_subscribe(token_ids: list[str], slot_ts: int = 0):
    """Queue token_ids for the CLOB WS daemon to subscribe to."""
    for tid in token_ids:
        _subscribe_queue.put((tid, slot_ts))


PRICE_MAX_AGE = 15.0   # seconds — reject WS price older than this

def get_ask_price_ws(token_id: str) -> float | None:
    """Return best ask from WS cache only if fresh (<15s). None triggers HTTP fallback."""
    with _clob_prices_lock:
        price = _clob_prices.get(token_id)
        ts    = _clob_price_ts.get(token_id, 0)
    if price is None:
        return None
    age = time.time() - ts
    if age > PRICE_MAX_AGE:
        log.info("  WS ask for %s is %.1fs old (stale) — falling back to /book", token_id[:12], age)
        return None
    return price


def _clob_prune_stale() -> None:
    """Remove tokens from subscriptions that belong to old slots."""
    now_ts = int(time.time())
    cur = (now_ts // SLOT_DURATION) * SLOT_DURATION
    active_cutoff = cur - SLOT_DURATION
    with _clob_prices_lock:
        stale = {tid for tid in _clob_subscribed
                 if _token_slot.get(tid, 0) < active_cutoff}
        for tid in stale:
            _clob_subscribed.discard(tid)
            _clob_prices.pop(tid, None)
            _token_slot.pop(tid, None)
    if stale:
        log.info("CLOB WS pruned %d stale tokens", len(stale))


def _clob_drain_and_subscribe(ws_manager: WebSocketManager) -> None:
    """Drain the subscribe queue and send new subscriptions via WS manager."""
    pending: list[tuple[str, int]] = []
    while True:
        try:
            pending.append(_subscribe_queue.get_nowait())
        except _queue_mod.Empty:
            break
    if not pending:
        return
    new_tokens = [t for t, _ in pending if t not in _clob_subscribed]
    if new_tokens:
        _clob_subscribed.update(new_tokens)
        for t, s in pending:
            if s:
                _token_slot[t] = s
        ws_manager.send_sync({"type": "Market", "assets_ids": new_tokens})
        log.info("CLOB WS subscribed %d new tokens", len(new_tokens))


async def _clob_on_connect(ws) -> None:
    """Re-subscribe existing tokens + drain queue on (re)connect."""
    global _clob_last_keepalive

    _clob_prune_stale()

    # Invalidate ALL cached prices on reconnect — the server will send a fresh
    # book snapshot after we re-subscribe, which will repopulate the cache.
    # Without this, stale prices from before the disconnect (~4-6s gap) could
    # still pass the PRICE_MAX_AGE check and be used for trading decisions.
    with _clob_prices_lock:
        stale_count = len(_clob_prices)
        _clob_prices.clear()
        _clob_price_ts.clear()
    if stale_count:
        log.info("CLOB WS reconnect — invalidated %d cached prices (forcing HTTP fallback until fresh data)", stale_count)

    with _clob_prices_lock:
        existing = list(_clob_subscribed)
    if existing:
        await ws.send(json.dumps({"type": "Market", "assets_ids": existing}))
        log.info("CLOB WS re-subscribed %d tokens on connect", len(existing))

        # Hydrate accumulator via REST /book on reconnect.
        # The CLOB WS sends 1 book snapshot per subscribe, but it arrives a few
        # seconds after connect. If the entry window fires during that gap the
        # accumulator is empty. Pre-seeding via REST guarantees data immediately.
        _acc = get_accumulator()
        hydrated = 0
        for tid in existing:
            try:
                r = _http.get(f"{CLOB_URL}/book", params={"token_id": tid}, timeout=3)
                if r.ok:
                    book = r.json()
                    book["event_type"] = "book"
                    book["asset_id"] = tid
                    _acc.feed_event(tid, book)
                    hydrated += 1
            except Exception:
                pass
        if hydrated:
            log.info("CLOB WS reconnect — hydrated accumulator for %d tokens via REST", hydrated)

    _clob_last_keepalive = time.time()

    # Drain any queued subscription requests that arrived before connect
    pending: list[tuple[str, int]] = []
    while True:
        try:
            pending.append(_subscribe_queue.get_nowait())
        except _queue_mod.Empty:
            break
    if pending:
        new_tokens = [t for t, _ in pending if t not in _clob_subscribed]
        if new_tokens:
            _clob_subscribed.update(new_tokens)
            for t, s in pending:
                if s:
                    _token_slot[t] = s
            await ws.send(json.dumps({"type": "Market", "assets_ids": new_tokens}))
            log.info("CLOB WS subscribed %d new tokens on connect", len(new_tokens))


async def _clob_on_message(msg) -> None:
    """Handle CLOB WS message — updates _clob_prices cache + feeds CLOB feature accumulator."""
    global _clob_last_keepalive

    # Server may send a list or a single dict
    events = [msg] if isinstance(msg, dict) else msg
    if not isinstance(events, list):
        return

    _acc = get_accumulator()
    for ev in events:
        etype = ev.get("event_type")
        # Feed into CLOB feature accumulator for real-time microstructure features
        if etype == "book":
            asset_id = ev.get("asset_id", "")
            if asset_id:
                _acc.feed_event(asset_id, ev)
        elif etype == "price_change":
            # Feed to all referenced asset_ids
            for pc in ev.get("price_changes", []):
                aid = pc.get("asset_id", "")
                if aid:
                    _acc.feed_event(aid, ev)
        if etype == "book":
            asset_id = ev.get("asset_id", "")
            asks = ev.get("asks", [])
            valid_asks = [
                float(a["price"])
                for a in asks
                if float(a.get("price", 1)) < 0.97
            ]
            if valid_asks:
                best = min(valid_asks)
                with _clob_prices_lock:
                    _clob_prices[asset_id] = best
                    _clob_price_ts[asset_id] = time.time()
        elif etype == "price_change":
            for change in ev.get("price_changes", []):
                asset_id = change.get("asset_id", "")
                if not asset_id:
                    continue
                # Polymarket sends best_ask directly on every price_change — use it.
                # Much more reliable than filtering by side (WS uses "SELL"/"BUY",
                # not "ASK"/"BID", so the old side=="ASK" filter never matched).
                raw_best_ask = change.get("best_ask")
                if raw_best_ask is not None:
                    try:
                        best_ask = float(raw_best_ask)
                        if 0 < best_ask < 0.97:
                            with _clob_prices_lock:
                                _clob_prices[asset_id] = best_ask
                                _clob_price_ts[asset_id] = time.time()
                    except (ValueError, TypeError):
                        pass

    # Drain pending subscriptions from external threads
    if _clob_ws_manager:
        pending: list[tuple[str, int]] = []
        while True:
            try:
                pending.append(_subscribe_queue.get_nowait())
            except _queue_mod.Empty:
                break
        if pending:
            new_tokens = [t for t, _ in pending if t not in _clob_subscribed]
            if new_tokens:
                _clob_subscribed.update(new_tokens)
                for t, s in pending:
                    if s:
                        _token_slot[t] = s
                # Use async send — we're already in the event loop
                await _clob_ws_manager.send({"type": "Market", "assets_ids": new_tokens})
                log.info("CLOB WS subscribed %d new tokens", len(new_tokens))

    # NOTE: We no longer send periodic re-subscribe as "keepalive".
    # The websockets library already sends RFC 6455 PING frames every 20s
    # (ping_interval=20 in WSConfig), which is the correct keepalive mechanism.
    # Re-subscribing every 30s was likely causing unnecessary server-side load
    # and possibly triggering rate-limit disconnects (code=1006).


def start_clob_daemon():
    """Start the CLOB WS daemon thread and pre-subscribe to current + next slot tokens."""
    global _clob_ws_manager

    _clob_ws_manager = WebSocketManager(
        name="clob",
        url=CLOB_WS_URL,
        on_message=_clob_on_message,
        on_connect=_clob_on_connect,
        config=WSConfig(
            # CRITICAL: ping_interval=None lets the Polymarket server control
            # ping/pong. Their server sends pings every ~30s; the websockets lib
            # automatically responds with pong even with ping_interval=None.
            # Client-initiated pings (ping_interval=20) cause "double-ping"
            # conflict → server drops connection every ~60s (code 1006).
            # Ref: https://github.com/Polymarket/py-clob-client/issues/82
            ping_interval=None,
            ping_timeout=None,
            close_timeout=5.0,
            zombie_timeout=60.0,       # BTC 5min markets have constant book activity. 60s silence = stale connection.
            health_log_interval=300.0,  # log health every 5 min
        ),
    )
    _clob_ws_manager.start()
    log.info("CLOB WS daemon started (ws_manager)")

    # Pre-subscribe to current + next slot only (4 tokens max).
    # Subscribing to too many tokens (8-16) causes Polymarket to drop the WS
    # connection every ~5min (code=1006) due to excessive message volume.
    # The pruner + on-demand subscribe in fetch_market handles future slots.
    now = int(time.time())
    cur_slot = (now // SLOT_DURATION) * SLOT_DURATION
    for i in range(2):
        slot_ts = cur_slot + i * SLOT_DURATION
        mkt = fetch_market(slot_ts)
        if mkt:
            clob_subscribe([mkt["yes_token"], mkt["no_token"]], slot_ts=slot_ts)
            log.info("CLOB WS pre-subscribed slot=%d", slot_ts)


# ── Auto-sizing shares logic ───────────────────────────────────────────────────
def compute_shares(balance_usdc: float, ask_price: float) -> float:
    """Compute number of shares for a trade.

    When AUTO_SHARES is enabled, scales linearly between AUTO_SHARES_MIN and
    AUTO_SHARES_MAX based on balance (floor/ceil anchors). Also ensures the
    trade cost doesn't exceed 10% of balance (risk cap).

    When AUTO_SHARES is disabled, returns FIXED_SHARES.

    Always clamps to [AUTO_SHARES_MIN, AUTO_SHARES_MAX] and rounds down to int.
    """
    if not AUTO_SHARES:
        return float(max(FIXED_SHARES, AUTO_SHARES_MIN))

    floor = AUTO_SHARES_BAL_FLOOR
    ceil  = AUTO_SHARES_BAL_CEIL
    lo    = AUTO_SHARES_MIN
    hi    = AUTO_SHARES_MAX

    if balance_usdc <= floor:
        shares = lo
    elif balance_usdc >= ceil:
        shares = hi
    else:
        # Linear interpolation
        t = (balance_usdc - floor) / (ceil - floor)
        shares = lo + t * (hi - lo)

    # Risk cap: never spend more than 10% of balance on a single trade
    max_cost = balance_usdc * 0.10
    max_shares_by_cost = max_cost / (ask_price + 1e-8)
    shares = min(shares, max_shares_by_cost)

    # Clamp to MAX and floor to integer. Note: we do NOT clamp up to MIN here
    # because the risk cap must take priority. If risk cap says 3 shares but
    # MIN is 5, we return MIN (5) only because CLOB rejects < 5 — but the
    # balance check downstream will catch that the cost exceeds budget.
    shares = min(hi, shares)
    shares = max(lo, shares)  # CLOB min 5 shares — order would be rejected below this
    return float(int(shares))


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    """Download champion.pkl from HuggingFace (always fresh), then load it."""
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
        log.info("Removed stale champion cache — downloading fresh from HF...")
    log.info("Downloading champion model from HuggingFace (%s)...", HF_REPO)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="champion.pkl",
            repo_type="model",
            token=HF_TOKEN,
            local_dir="/tmp",
        )
        # hf_hub_download saves to /tmp/champion.pkl (or subdir) — normalise
        import shutil
        if Path(path) != MODEL_PATH:
            shutil.copy(path, MODEL_PATH)
        log.info("Champion model downloaded: %s", MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to download champion model from HF: {e}") from e
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    log.info("Model loaded: %d features, WF AUC=%.3f, ensemble=%s",
             len(bundle["features"]), bundle.get("wf_auc", 0),
             bundle.get("ensemble", False))
    return bundle["model"], bundle["features"]


def predict_proba(model_obj, X: "pd.DataFrame") -> float:
    """Unified predict — handles both plain sklearn model and ensemble dict."""
    if isinstance(model_obj, dict) and model_obj.get("ensemble"):
        lgb_prob = model_obj["lgb"].predict_proba(X)[0][1]
        lr_prob  = model_obj["lr"].predict_proba(X)[0][1]
        w = model_obj.get("lgb_weight", 0.65)
        return float(lgb_prob * w + lr_prob * (1 - w))
    else:
        # Plain sklearn / calibrated model (v5/v6 and v7 non-ensemble)
        m = model_obj["lgb"] if isinstance(model_obj, dict) else model_obj
        return float(m.predict_proba(X)[0][1])


# ── Spot features from buffer ──────────────────────────────────────────────────
def _window_feats(prices, label, wname):
    if len(prices) < 2:
        return {f"{label}_{wname}_ret": 0.0,
                f"{label}_{wname}_vol": 0.0,
                f"{label}_{wname}_mom": 0.0}
    px = np.array(prices, dtype=np.float64)
    ret = float((px[-1] - px[0]) / (px[0] + 1e-8))
    vol = float(np.std(np.diff(px) / (px[:-1] + 1e-8)))
    mid = px[len(px) // 2]
    mom = float((px[-1] - mid) / (mid + 1e-8))
    return {f"{label}_{wname}_ret": ret,
            f"{label}_{wname}_vol": vol,
            f"{label}_{wname}_mom": mom}

def build_spot_features(slot_ts: int) -> dict:
    """Build BTC spot features matching train_v8_modal.py exactly.

    Windows (BTC only — v8 is BTC-only):
      btc_inslot_ret / btc_inslot_vol  — [slot_ts, slot_ts+180s]
      btc_pre_5m_ret / _vol            — [slot_ts-300,  slot_ts]
      btc_pre_15m_ret / _vol           — [slot_ts-900,  slot_ts]
      btc_pre_30m_ret / _vol           — [slot_ts-1800, slot_ts]
      btc_pre_1h_ret  / _vol           — [slot_ts-3600, slot_ts]
      btc_pre_4h_ret  / _vol           — [slot_ts-14400,slot_ts]
      btc_dist_1k / _5k / _10k        — round-number proximity
    """
    feat: dict = {}
    zeros = {
        "btc_inslot_ret": 0.0, "btc_inslot_vol": 0.0,
        "btc_pre_5m_ret": 0.0,  "btc_pre_5m_vol": 0.0,
        "btc_pre_15m_ret": 0.0, "btc_pre_15m_vol": 0.0,
        "btc_pre_30m_ret": 0.0, "btc_pre_30m_vol": 0.0,
        "btc_pre_1h_ret": 0.0,  "btc_pre_1h_vol": 0.0,
        "btc_pre_4h_ret": 0.0,  "btc_pre_4h_vol": 0.0,
        "btc_dist_1k": 0.5,
    }
    if not SPOT_BUFFER.exists():
        log.warning("spot_buffer missing — filling zeros")
        return zeros

    try:
        buf = json.loads(SPOT_BUFFER.read_text())
    except Exception as e:
        log.warning("spot_buffer read error: %s", e)
        return zeros

    age = int(time.time()) - buf.get("updated_at", 0)
    if age > BUFFER_STALE:
        log.warning("spot_buffer is %ds stale — using zeros", age)
        return zeros

    candles = buf.get("btcusdt", [])
    if not candles:
        return zeros

    ts_arr = np.array([c[0] for c in candles], dtype=np.int64)
    px_arr = np.array([c[1] for c in candles], dtype=np.float64)

    def _seg(lo: int, hi: int) -> np.ndarray:
        mask = (ts_arr >= lo) & (ts_arr < hi)
        return px_arr[mask]

    def _ret_vol(seg: np.ndarray) -> tuple[float, float]:
        if len(seg) < 2:
            return 0.0, 0.0
        ret = float((seg[-1] - seg[0]) / (seg[0] + 1e-8))
        vol = float(np.std(seg) / (np.mean(seg) + 1e-8))
        return ret, vol

    # Inslot: BTC price return during [slot_ts, slot_ts+OBSERVE_SECS)
    # Use searchsorted to find nearest candle at start and end (matches training)
    idx_start = int(np.searchsorted(ts_arr, slot_ts, side="left"))
    idx_end   = int(np.searchsorted(ts_arr, slot_ts + OBSERVE_SECS, side="right")) - 1
    idx_start = max(0, min(idx_start, len(px_arr) - 1))
    idx_end   = max(0, min(idx_end, len(px_arr) - 1))
    if idx_end > idx_start and px_arr[idx_start] > 0:
        feat["btc_inslot_ret"] = float(px_arr[idx_end] / px_arr[idx_start] - 1)
        seg_inslot = px_arr[idx_start:idx_end + 1]
        feat["btc_inslot_vol"] = float(np.std(seg_inslot) / (np.mean(seg_inslot) + 1e-8)) if len(seg_inslot) > 1 else 0.0
    else:
        feat["btc_inslot_ret"] = 0.0
        feat["btc_inslot_vol"] = 0.0

    # Pre-slot windows — MUST use px at observation end (slot_ts + OBS_SECS)
    # to match training: spot_at(obs_end_ts) vs spot_at(slot_ts - window)
    obs_end_ts = slot_ts + OBSERVE_SECS
    idx_obs = np.searchsorted(ts_arr, obs_end_ts, side="right") - 1
    px_obs_end = float(px_arr[idx_obs]) if idx_obs >= 0 else 0.0

    for w_s, lbl in [(300, "5m"), (900, "15m"), (1800, "30m"), (3600, "1h")]:
        idx_h = np.searchsorted(ts_arr, slot_ts - w_s, side="right") - 1
        px_h = float(px_arr[idx_h]) if idx_h >= 0 else 0.0
        if px_h > 0 and px_obs_end > 0:
            feat[f"btc_pre_{lbl}_ret"] = float(px_obs_end / px_h - 1)
        else:
            feat[f"btc_pre_{lbl}_ret"] = 0.0

    # Round-number proximity (use obs-end price to match training)
    if px_obs_end > 0:
        px_k = px_obs_end / 1000
        feat["btc_dist_1k"]  = float(min(px_k - math.floor(px_k), math.ceil(px_k) - px_k))
    else:
        feat["btc_dist_1k"] = 0.5

    # btc_pre_1h_4h_ratio removed (warm-up 4h dependency)

    # ── v26 NEW spot features ─────────────────────────────────────────────────
    # btc_inslot_range: (high - low) / px during [slot_ts, slot_ts+OBS]
    if idx_end > idx_start and px_obs_end > 0:
        hi_arr = np.array([c[2] for c in candles], dtype=np.float64) if len(candles[0]) > 2 else px_arr
        lo_arr = np.array([c[3] for c in candles], dtype=np.float64) if len(candles[0]) > 3 else px_arr
        inslot_hi = hi_arr[idx_start:idx_end + 1]
        inslot_lo = lo_arr[idx_start:idx_end + 1]
        feat["btc_inslot_range"] = float((inslot_hi.max() - inslot_lo.min()) / px_obs_end) if len(inslot_hi) > 0 else 0.0
    else:
        feat["btc_inslot_range"] = 0.0

    # btc_vol_1h / btc_vol_4h: std of 1-min returns over 1h / 4h
    def _volatility(lo_ts: int, hi_ts: int) -> float:
        mask = (ts_arr >= lo_ts) & (ts_arr < hi_ts)
        seg = px_arr[mask]
        if len(seg) >= 3:
            rets = np.diff(seg) / seg[:-1]
            return float(np.std(rets))
        return 0.0

    feat["btc_vol_1h"] = _volatility(slot_ts - 3600, slot_ts)
    # btc_vol_4h removed (warm-up 4h dependency)
    # NOTA: normalização por vol_1h foi testada no v30 mas resultou em AUC=0.778
    # abaixo do champion v29 (AUC=0.792) que usa retornos brutos.
    # Mantido bruto para paridade com v29. Será revisado quando houver parquet [0,168s).

    # btc_spot_vol_ratio: recent 5m volume / avg hourly 5m volume
    # Uses actual Binance candle volume (matching training formula)
    vol_arr = np.array([c[4] if len(c) > 4 else 0.0 for c in candles], dtype=np.float64)
    mask_5m = (ts_arr >= slot_ts - 300) & (ts_arr < slot_ts)
    mask_55m = (ts_arr >= slot_ts - 3600) & (ts_arr < slot_ts - 300)
    vol_5m = float(vol_arr[mask_5m].sum()) if mask_5m.any() else 0.0
    vol_55m = float(vol_arr[mask_55m].sum()) if mask_55m.any() else 0.0
    avg_5m_per_hour = vol_55m / 11 if vol_55m > 0 else 1e-9
    feat["btc_spot_vol_ratio"] = float(vol_5m / (avg_5m_per_hour + 1e-9))

    return feat


# ── Market helpers ─────────────────────────────────────────────────────────────
_market_cache: dict[int, dict] = {}   # slot_ts → market dict, avoids hammering Gamma

def fetch_market(slot_ts: int) -> dict | None:
    if slot_ts in _market_cache:
        return _market_cache[slot_ts]
    slug = f"btc-updown-5m-{slot_ts}"
    try:
        r = _http.get(f"{GAMMA_HOST}/markets/slug/{slug}", timeout=HTTP_TIMEOUT)
        if not r.ok:
            return None
        m = r.json()
        if not m or m.get("closed") or m.get("archived"):
            return None
        token_ids = m.get("clobTokenIds", "[]")
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if len(token_ids) < 2:
            return None
        end_str = m.get("endDate", "") or m.get("endDateIso", "")
        end_ts = int(datetime.fromisoformat(
            end_str.replace("Z", "+00:00")).timestamp()) if end_str else slot_ts + SLOT_DURATION
        result = {
            "condition_id": m.get("conditionId", ""),
            "yes_token":    token_ids[0],
            "no_token":     token_ids[1],
            "end_ts":       end_ts,
            "slot_ts":      slot_ts,
        }
        # Pre-subscribe WS daemon so prices are ready by entry window
        clob_subscribe([token_ids[0], token_ids[1]], slot_ts=slot_ts)
        _market_cache[slot_ts] = result
        return result
    except Exception as e:
        log.warning("fetch_market slot=%d: %s", slot_ts, e)
        return None

def get_ask_price(token_id: str) -> float:
    """Get best ask (cost to buy) — WS cache first, then REST /book with retry.

    IMPORTANT: do NOT fall back to /price — that endpoint returns the last traded
    price, not the current best ask. It can be stale by minutes and causes entries
    at prices far from the real market (e.g. $0.87 when mid is $0.50).
    /book always reflects the live order book state.
    """
    # Fast path: WS cache (book snapshot + price_change events, <15s)
    ws_price = get_ask_price_ws(token_id)
    if ws_price is not None:
        return ws_price

    # Fallback: REST /book with retry — parse best ask from live order book
    for attempt in range(3):
        try:
            r = _http.get(f"{CLOB_URL}/book",
                          params={"token_id": token_id}, timeout=HTTP_TIMEOUT + 2)
            if not r.ok:
                log.warning("  get_ask_price /book HTTP %d (attempt %d/3)", r.status_code, attempt + 1)
                time.sleep(0.5 * (attempt + 1))
                continue
            book = r.json()
            asks = book.get("asks", [])
            if not asks:
                # Empty book = market one-sided, no sellers. Don't retry — it won't change.
                log.info("  get_ask_price /book empty (no asks) — market one-sided")
                return 0.0
            best_asks = [float(a["price"]) for a in asks
                         if float(a.get("price", 1)) < 0.97]
            if best_asks:
                best_ask = min(best_asks)
                log.info("  get_ask_price /book fallback: %.3f (attempt %d)", best_ask, attempt + 1)
                return best_ask
            else:
                # All asks >= 0.97 = market already resolved, no edge possible
                log.info("  get_ask_price /book all asks >= 0.97 — market decided")
                return 0.0
        except Exception as e:
            log.warning("  get_ask_price /book exception (attempt %d/3): %s", attempt + 1, e)
            time.sleep(0.5 * (attempt + 1))

    log.warning("  get_ask_price FAILED all 3 attempts for %s — returning 0.0", token_id[:16])
    return 0.0  # outside [0.38, 0.90] filter — trade will be skipped


def get_market_mid(slot_ts: int) -> tuple[float, float] | tuple[None, None]:
    """
    Get current market mid prices (UP, DOWN) from CLOB.

    Uses GET /midpoint (server-authoritative) with fallback to GET /book.
    Returns (up_mid, down_mid) or (None, None) on failure.
    """
    slug = f"btc-updown-5m-{slot_ts}"
    try:
        r = _http.get(f"{GAMMA_HOST}/markets/slug/{slug}", timeout=HTTP_TIMEOUT)
        if not r.ok:
            return None, None
        m = r.json()
        ctids = m.get("clobTokenIds", "[]")
        if isinstance(ctids, str):
            ctids = json.loads(ctids)
        if not ctids or len(ctids) < 2:
            return None, None

        up_token, dn_token = ctids[0], ctids[1]
        mids: list[float | None] = []
        for tid in [up_token, dn_token]:
            mid = None
            # Fast path: GET /midpoint (server-authoritative, no parsing needed)
            try:
                rm = _http.get(f"{CLOB_URL}/midpoint",
                               params={"token_id": tid}, timeout=HTTP_TIMEOUT)
                if rm.ok:
                    data = rm.json()
                    mp = float(data.get("mid", 0))
                    if 0 < mp < 1:
                        mid = mp
            except Exception:
                pass
            # Fallback: parse from /book
            if mid is None:
                try:
                    rb = _http.get(f"{CLOB_URL}/book",
                                   params={"token_id": tid}, timeout=HTTP_TIMEOUT + 2)
                    if rb.ok:
                        book = rb.json()
                        asks = [float(a["price"]) for a in book.get("asks", [])
                                if 0 < float(a.get("price", 0)) < 0.99]
                        bids = [float(b["price"]) for b in book.get("bids", [])
                                if 0 < float(b.get("price", 0)) < 0.99]
                        if asks and bids:
                            mid = (min(asks) + max(bids)) / 2.0
                        elif asks:
                            mid = min(asks)
                        elif bids:
                            mid = max(bids)
                except Exception:
                    pass
            mids.append(mid)

        up_mid, dn_mid = mids[0], mids[1]
        if up_mid is not None and dn_mid is not None:
            return up_mid, dn_mid
        # If only one side available, derive the other (binary market: up + dn ≈ 1)
        if up_mid is not None:
            return up_mid, 1.0 - up_mid
        if dn_mid is not None:
            return 1.0 - dn_mid, dn_mid
    except Exception:
        pass
    return None, None

def _fetch_token_trades(token_id: str, outcome_label: str, slot_ts: int, expected_slug: str) -> list[dict]:
    """Fetch trades for a single token. Used by fetch_inslot_trades for parallel fetching."""
    trades = []
    for page in range(MAX_TRADE_PAGES):
        offset = page * 500
        try:
            r = _http.get(f"{DATA_API}/trades",
                params={"asset": token_id, "limit": 500, "offset": offset,
                        "_t": int(time.time())},  # cache-bust Cloudflare CDN (5min TTL)
                timeout=HTTP_TIMEOUT)
            if not r.ok:
                break
            batch = r.json()
            if not batch:
                break
            for t in batch:
                trade_slug = t.get("slug", "")
                if trade_slug and trade_slug != expected_slug:
                    continue
                ts = int(t.get("timestamp", 0))
                if ts > 1e12:
                    ts //= 1000
                t_sec = ts - slot_ts
                if 0 <= t_sec < OBSERVE_SECS:
                    _price = float(t.get("price", 0) or 0)
                    _size  = float(t.get("size", 0) or 0)
                    trades.append({
                        "outcome":   t.get("outcome", outcome_label),
                        "side":      t.get("side", "BUY"),
                        "price":     _price,
                        "size":      _size,
                        "size_usdc": _price * _size,
                        "t_sec":     t_sec,
                    })
        except Exception:
            break
    return trades


def fetch_inslot_trades(yes_token: str, no_token: str, slot_ts: int) -> list[dict]:
    """Fetch inslot trades from data-api. Lag is ~90-120s so t=0-60s trades available at t=180s.

    IMPORTANT: The data-api returns trades for ALL markets using the same token ID,
    not just our BTC 5-min market. We filter by slug to exclude contamination from
    ETH, SOL, BNB, daily BTC, and other markets that share the same token.

    OPTIMIZATION: Fetches yes and no tokens in PARALLEL using ThreadPoolExecutor,
    cutting latency roughly in half (from ~4-6s sequential to ~2-3s).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    expected_slug = f"btc-updown-5m-{slot_ts}"

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="fetch-trades") as executor:
        fut_yes = executor.submit(_fetch_token_trades, yes_token, "Up", slot_ts, expected_slug)
        fut_no  = executor.submit(_fetch_token_trades, no_token, "Down", slot_ts, expected_slug)

        all_trades = []
        for fut in as_completed([fut_yes, fut_no]):
            try:
                all_trades.extend(fut.result(timeout=HTTP_TIMEOUT * MAX_TRADE_PAGES + 5))
            except Exception as e:
                log.warning("Parallel trade fetch error: %s", e)

    return all_trades



# ── OB snapshot cache for temporal features ──────────────────────────────
# Stores the first OB snapshot of each slot so we can compute drift/momentum
# between the "open" (first poll ~t=170s) and "close" (current poll) snapshots.
_ob_open_cache: dict[str, dict] = {}   # token_id -> {mid, imbalance, imb_w_values, ts}
_ob_last_slot: int = 0                 # track slot changes to clear cache


def _fetch_ob_snapshot(up_token_id: str) -> dict | None:
    """Fetch raw OB metrics from CLOB REST. Returns dict or None on failure.

    Retries up to 3 times with 0.5s delay to handle:
      - Transient network errors
      - Empty asks at slot start (~10s window where new market has no asks yet)

    Does NOT retry HTTP 404 (market closed/not found) — that's permanent.
    Does NOT retry asks=0 when bids>90 (market fully resolved ~99% one side) —
    that means no liquidity to trade and the bot would skip anyway.
    """
    OB_RETRIES = 3
    OB_RETRY_DELAY = 0.5   # seconds between retries

    for attempt in range(1, OB_RETRIES + 1):
        try:
            r = _http.get(
                f"{CLOB_URL}/book",
                params={"token_id": up_token_id},
                timeout=5,
            )
            if not r.ok:
                log.warning("OB snapshot HTTP %d for %s (attempt %d/%d)",
                            r.status_code, up_token_id[:20], attempt, OB_RETRIES)
                if r.status_code == 404:
                    return None   # permanent — market doesn't exist
                if attempt < OB_RETRIES:
                    time.sleep(OB_RETRY_DELAY)
                    continue
                return None

            book = r.json()
            asks = book.get("asks", [])
            bids = book.get("bids", [])

            if not bids:
                # No bids at all — highly unusual, retry
                log.warning("OB snapshot no bids for %s (attempt %d/%d)",
                            up_token_id[:20], attempt, OB_RETRIES)
                if attempt < OB_RETRIES:
                    time.sleep(OB_RETRY_DELAY)
                    continue
                return None

            if not asks:
                # No asks: either market is ~99% resolved (asks=0, bids=99) or
                # it's the first few seconds of a new slot.
                # Retry once — if still empty after retry, it's a resolved market.
                log.warning("OB snapshot asks=0 for %s bids=%d (attempt %d/%d)",
                            up_token_id[:20], len(bids), attempt, OB_RETRIES)
                if attempt < OB_RETRIES:
                    time.sleep(OB_RETRY_DELAY)
                    continue
                # After retries: give up — market is too one-sided to trade
                return None

            asks_sorted = sorted(asks, key=lambda x: float(x["price"]))
            bids_sorted = sorted(bids, key=lambda x: float(x["price"]), reverse=True)
            best_ask = float(asks_sorted[0]["price"])
            best_bid = float(bids_sorted[0]["price"])
            best_ask_sz = float(asks_sorted[0]["size"])
            best_bid_sz = float(bids_sorted[0]["size"])
            mid = (best_ask + best_bid) / 2
            spread = best_ask - best_bid
            imbalance = float((best_bid_sz - best_ask_sz) / (best_bid_sz + best_ask_sz + 1e-8))
            if attempt > 1:
                log.info("OB snapshot OK after %d attempts for %s", attempt, up_token_id[:20])
            return {
                "asks_sorted": asks_sorted, "bids_sorted": bids_sorted,
                "asks": asks, "bids": bids,
                "best_ask": best_ask, "best_bid": best_bid,
                "best_ask_sz": best_ask_sz, "best_bid_sz": best_bid_sz,
                "mid": mid, "spread": spread, "imbalance": imbalance,
                "ts": time.time(),
            }
        except Exception as e:
            log.warning("OB snapshot fetch failed for %s: %s (attempt %d/%d)",
                        up_token_id[:20], e, attempt, OB_RETRIES)
            if attempt < OB_RETRIES:
                time.sleep(OB_RETRY_DELAY)

    return None


def _build_ob_features(up_token_id: str) -> dict:
    """
    Fetch the current order book from CLOB REST and compute OB features.
    Matches training (pmdata poly_l2 book snapshots) as closely as possible.

    Uses two snapshots for temporal features:
      - "open" snapshot: cached from first poll of the entry window
      - "close" snapshot: current poll
    This lets us compute real ob_mid_drift, ob_imb_momentum, and windowed
    imbalance (ob_imb_w0/w1/w2) — top-5 features that were previously 0.0.

    FALLBACK: If close snapshot fails (book empty/one-sided late in slot),
    uses open snapshot for static features and zeros for temporal features.
    This prevents 9 OB features from going to zero simultaneously.

    Returns dict with features. On failure returns empty dict (model uses 0.0 fallback).
    """
    snap = _fetch_ob_snapshot(up_token_id)

    # ── Cache open snapshot (first poll of entry window) ──────────────
    if up_token_id not in _ob_open_cache:
        if snap is not None:
            _ob_open_cache[up_token_id] = {
                "mid": snap["mid"],
                "spread": snap["spread"],
                "imbalance": snap["imbalance"],
                "total_depth": float(sum(float(b["size"]) for b in snap["bids"]) +
                                     sum(float(a["size"]) for a in snap["asks"])),
                "ts": snap["ts"],
            }

    open_snap = _ob_open_cache.get(up_token_id)

    # If BOTH snapshots failed, we have nothing — return empty
    if snap is None and open_snap is None:
        return {}

    # If close snapshot failed but we have open_snap, use it as fallback
    # (static features from open, temporal features = 0)
    if snap is None:
        log.debug("OB close snapshot failed for %s — using open_snap fallback", up_token_id[:20])
        return {
            # Static OB features from open snapshot
            "ob_mid":           float(open_snap["mid"]),
            "ob_spread":        float(open_snap.get("spread", 0.02)),
            "ob_imbalance":     float(open_snap["imbalance"]),
            "ob_depth_ratio":   1.0,  # neutral
            "ob_bid_depth_5c":  0.5,
            "ob_ask_depth_5c":  0.5,
            "ob_total_depth":   float(open_snap.get("total_depth", 0)),
            "ob_weighted_imb":  float(open_snap["imbalance"]),  # approx
            # Temporal features = 0 (no change from open to close)
            "ob_mid_drift":     0.0,    # removed from model — kept as 0 for compatibility
            "ob_imbalance_end": float(open_snap["imbalance"]),
            "ob_spread_end":    float(open_snap.get("spread", 0.02)),
            "ob_depth_change":  0.0,
            "ob_imb_momentum":  0.0,
            # ob_imb_w1/w2 removed from model
        }

    # ── Temporal features from open vs close ──────────────────────────
    # ob_mid_drift removed from model (timing gap ~72s live vs treino)
    ob_imb_momentum = float(snap["imbalance"] - open_snap["imbalance"])
    # ob_imb_w1/w2 removed from model (quase sempre 0.0 live)

    # Use close snapshot for all other features
    book = snap
    asks = book["asks"]
    bids = book["bids"]
    asks_sorted = book["asks_sorted"]
    bids_sorted = book["bids_sorted"]
    best_ask = book["best_ask"]
    best_bid = book["best_bid"]
    best_ask_sz = book["best_ask_sz"]
    best_bid_sz = book["best_bid_sz"]
    mid    = book["mid"]
    spread = book["spread"]
    imbalance = book["imbalance"]

    try:
        # Depth within 5c of mid, normalized by total depth
        total_bid = sum(float(b["size"]) for b in bids) + 1e-8
        total_ask = sum(float(a["size"]) for a in asks) + 1e-8
        bid_depth_5c = sum(float(b["size"]) for b in bids
                           if float(b["price"]) >= mid - 0.05) / total_bid
        ask_depth_5c = sum(float(a["size"]) for a in asks
                           if float(a["price"]) <= mid + 0.05) / total_ask
        depth_ratio  = float(bid_depth_5c / (ask_depth_5c + 1e-8))

        # Total depth (raw sum, matches training)
        total_depth = float(sum(float(b["size"]) for b in bids) +
                            sum(float(a["size"]) for a in asks))

        # Weighted imbalance: exp-weighted by proximity to mid (matches training)
        bp = np.array([float(b["price"]) for b in bids])
        bs = np.array([float(b["size"]) for b in bids])
        ap = np.array([float(a["price"]) for a in asks])
        as_ = np.array([float(a["size"]) for a in asks])
        bid_wt = float(np.sum(bs * np.exp(-10 * np.abs(bp - mid))))
        ask_wt = float(np.sum(as_ * np.exp(-10 * np.abs(ap - mid))))
        weighted_imb = float((bid_wt - ask_wt) / (bid_wt + ask_wt + 1e-8))

        return {
            "ob_mid":           float(open_snap["mid"]),
            "ob_spread":        float(open_snap.get("spread", spread)),
            "ob_imbalance":     float(open_snap.get("imbalance", imbalance)),
            "ob_depth_ratio":   float(depth_ratio),
            "ob_bid_depth_5c":  float(bid_depth_5c),
            "ob_ask_depth_5c":  float(ask_depth_5c),
            "ob_total_depth":   float(total_depth),
            "ob_weighted_imb":  float(weighted_imb),
            # ob_mid_drift removed; ob_imb_w1/w2 removed
            "ob_imbalance_end": float(imbalance),
            "ob_spread_end":    float(spread),
            "ob_depth_change":  float(total_depth - open_snap.get("total_depth", total_depth)),
            "ob_imb_momentum":  ob_imb_momentum,
        }
    except Exception as e:
        log.warning("OB fetch failed for %s: %s", up_token_id[:20], e)
        return {}


def build_features(ticks: list[dict], slot_ts: int, features: list[str],
                   up_token_id: str = "") -> dict | None:
    """
    Build real-time features matching train_v8_modal.py tick_features_v8() exactly.

    Tick features computed from inslot trades [t=0, t=180s):
      - 6x30s sub-windows (btc_up_w0..w5)
      - per-sub-window zscores vs _slot_history (btc_up_w0_zscore..w5_zscore)
      - multi-scale up_ratio zscore (5/10/20 slots: btc_up_ratio_zscore_5s/10s/20s)
      - time-weighted order flow (btc_tw_up_ratio)
      - VWAP trend (btc_vwap_trend)
      - volume-weighted momentum (btc_vwmom)
      - tick acceleration (btc_tick_accel)
      - lag outcomes (lag_1/2/3_outcome, lag_streak) from _slot_history
      - OB features: filled with 0.5/0 (no live OB available in real-time)

    Spot features from Binance buffer (build_spot_features):
      btc_inslot_ret/vol, btc_pre_5m/15m/30m/1h/4h_ret/vol, btc_dist_1k/5k/10k

    Historical context from _slot_history ring buffer (last 20 slots).
    """
    OBS = OBSERVE_SECS  # 180s

    # ── Time features ──────────────────────────────────────────────────────────
    dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    feat: dict = {
        # v21: only hour_x_up_ratio uses hour (computed below); calendar features pruned
    }

    # ── Order flow from inslot ticks ───────────────────────────────────────────
    if ticks:
        n      = len(ticks)
        vol_up = sum(t["size_usdc"] for t in ticks if t.get("outcome") == "Up")
        vol_dn = sum(t["size_usdc"] for t in ticks if t.get("outcome") == "Down")
        total  = vol_up + vol_dn + 1e-8

        up_tks = [t for t in ticks if t.get("outcome") == "Up"]
        dn_tks = [t for t in ticks if t.get("outcome") == "Down"]

        def _ur_w(subset: list[dict]) -> float:
            vu = sum(t["size_usdc"] for t in subset if t.get("outcome") == "Up")
            tt = sum(t["size_usdc"] for t in subset) + 1e-8
            return float(vu / tt)

        # 2x30s sub-windows (w0/w1 only — w2-w5 always 0.5 with OBS=60s, removed)
        sw: dict = {}
        for i in range(2):
            t0_w, t1_w = i * 30, (i + 1) * 30
            sub = [t for t in ticks if t0_w <= t["t_sec"] < t1_w]
            sw[f"btc_up_w{i}"] = _ur_w(sub) if sub else 0.5

        # Momentum: w1 - w0 (only real windows)
        btc_momentum = float(sw["btc_up_w1"] - sw["btc_up_w0"])

        # Stability: std of 2 real windows
        up_ratio_stability = float(np.std([sw["btc_up_w0"], sw["btc_up_w1"]]))

        # Size disparity: avg trade size Up vs Down
        avg_up_sz  = float(sum(t["size_usdc"] for t in up_tks) / (len(up_tks) + 1e-8))
        avg_dn_sz  = float(sum(t["size_usdc"] for t in dn_tks) / (len(dn_tks) + 1e-8))
        size_disparity = float(avg_up_sz - avg_dn_sz)

        cur_up_ratio = float(vol_up / total)

        feat.update({
            "btc_up_ratio":           cur_up_ratio,
            "btc_n_ticks":            float(n),
            "btc_buy_ratio":          float(sum(t["size_usdc"] for t in ticks if t.get("side") == "BUY") / total),
            "btc_momentum":           btc_momentum,
            "btc_size_disparity":     size_disparity,
            "btc_up_ratio_stability": up_ratio_stability,
            **sw,
        })
    else:
        # No ticks — neutral fill
        cur_up_ratio = 0.5
        feat.update({
            "btc_up_ratio": 0.5, "btc_n_ticks": 0.0,
            "btc_buy_ratio": 0.5, "btc_momentum": 0.0,
            "btc_size_disparity": 0.0, "btc_up_ratio_stability": 0.0,
            "btc_up_w0": 0.5, "btc_up_w1": 0.5,
        })

    # ── Historical zscore features (from ring buffer) ──────────────────────────
    # _slot_history: list of {"slot_ts", "up_ratio", "target", "sw": [6 floats]}
    # populated by _update_slot_history() after each prediction
    hist = _slot_history  # module-level ring buffer

    # Temporal interaction features (v17): hour modulates CLOB signal
    feat["hour_x_up_ratio"] = cur_up_ratio * (hour / 24.0)
    # hour_cos: cyclical hour encoding (cos component)
    feat["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
    # hour_sin: cyclical hour encoding (sin component) — v23 uses this
    feat["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
    # dow_sin/cos: day-of-week cyclical encoding — v28 parity
    dow = datetime.fromtimestamp(slot_ts, tz=timezone.utc).weekday()
    feat["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
    feat["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))
    # btc_up_ratio_stability: std of 6 sub-windows (v23 feature)
    feat["btc_up_ratio_stability"] = feat.get("btc_up_ratio_stability", 0.0)
    # time-weighted up ratio (recency bias) — MUST match training formula:
    # Training uses exponential decay exp(-0.02*(obs_secs-t)) weighted by size_usdc.
    if ticks:
        t_arr = np.array([t["t_sec"] for t in ticks], dtype=np.float64)
        sz_arr = np.array([t.get("size_usdc", 1.0) for t in ticks], dtype=np.float64)
        up_arr = np.array([1.0 if t.get("outcome") == "Up" else 0.0 for t in ticks])
        w_exp = np.exp(-0.02 * (OBSERVE_SECS - t_arr))
        numerator = np.sum(up_arr * sz_arr * w_exp)
        denominator = np.sum(sz_arr * w_exp) + 1e-9
        tw_up_ratio = float(numerator / denominator)
    else:
        tw_up_ratio = 0.5
    feat["btc_tw_up_ratio"] = tw_up_ratio
    feat["hour_x_tw_ur"] = tw_up_ratio * (hour / 24.0)

    # prev_slot_up_ratio — continuous lag signals (v21 uses 1,2,3,4,5)
    # Staleness guard: if time gap between slots is too large (e.g. downtime),
    # fill with 0.5 (matches training behavior)
    SLOT_DURATION = 300  # 5 minutes
    n_hist = len(hist)
    for lag in [1, 2, 3, 4, 5]:
        if n_hist >= lag:
            h = hist[-lag]
            # Check staleness: if gap > lag * SLOT_DURATION * 3, consider stale
            time_gap = slot_ts - h.get("slot_ts", 0)
            if time_gap > lag * SLOT_DURATION * 3:
                feat[f"prev_slot_up_ratio_{lag}"]  = 0.5
                feat[f"prev_slot_n_ticks_{lag}"]   = 0.0
                feat[f"prev_slot_vol_{lag}"]       = 0.0
                feat[f"lag_{lag}_outcome"]         = 0.5
            else:
                feat[f"prev_slot_up_ratio_{lag}"]  = float(h.get("up_ratio") or 0.5)
                feat[f"prev_slot_n_ticks_{lag}"]   = float(h.get("n_ticks") or 0.0)
                feat[f"prev_slot_vol_{lag}"]       = float(h.get("vol_total") or 0.0)
                feat[f"lag_{lag}_outcome"]         = float(h.get("target") or 0.5)
        else:
            feat[f"prev_slot_up_ratio_{lag}"]  = 0.5
            feat[f"prev_slot_n_ticks_{lag}"]   = 0.0
            feat[f"prev_slot_vol_{lag}"]       = 0.0
            feat[f"lag_{lag}_outcome"]         = 0.5

    # lag_streak: consecutive same-direction outcomes — v28 parity
    lag_streak = 0
    streak_dir = None
    for lag in [1, 2, 3, 4, 5]:
        if n_hist >= lag:
            h = hist[-lag]
            time_gap = slot_ts - h.get("slot_ts", 0)
            if time_gap > lag * SLOT_DURATION * 3:
                break  # stale — stop streak
            t_val = h.get("target")
            if t_val is None:
                break
            if lag == 1:
                streak_dir = int(t_val); lag_streak = 1
            elif int(t_val) == streak_dir:
                lag_streak += 1
            else:
                break
    feat["lag_streak"] = float(lag_streak)

    # ── Multi-scale up_ratio zscore (5/20 slots) ─────────────────────────────
    # Matches training: zscore of current up_ratio vs recent history mean/std.
    def _hist_ur(n: int) -> list[float]:
        return [h["up_ratio"] for h in hist[-n:]] if hist else []

    hist_vals_20 = _hist_ur(20)
    if len(hist_vals_20) >= 3:
        mu20 = float(np.mean(hist_vals_20))
        sd20 = float(np.std(hist_vals_20)) + 1e-6
        # Guard: if sd is too small (e.g. seeded history with identical values),
        # use a minimum std to prevent zscore explosion
        if sd20 < 0.01:
            sd20 = 0.01
        feat["btc_up_ratio_zscore_20s"] = float(np.clip((cur_up_ratio - mu20) / sd20, -5.0, 5.0))
        prev_ur_1 = feat.get("prev_slot_up_ratio_1", 0.5)
        feat["lag_ur_zscore_20"] = float(np.clip((prev_ur_1 - mu20) / sd20, -5.0, 5.0))
    else:
        feat["btc_up_ratio_zscore_20s"] = 0.0
        feat["lag_ur_zscore_20"] = 0.0

    hist_vals_5 = _hist_ur(5)
    if len(hist_vals_5) >= 2:
        mu5 = float(np.mean(hist_vals_5))
        sd5 = float(np.std(hist_vals_5)) + 1e-6
        if sd5 < 0.01:
            sd5 = 0.01
        feat["btc_up_ratio_zscore_5s"] = float(np.clip((cur_up_ratio - mu5) / sd5, -5.0, 5.0))
        # v26: lag_ur_zscore_5 uses prev_slot_up_ratio_1 as current value
        prev_ur_1 = feat.get("prev_slot_up_ratio_1", 0.5)
        feat["lag_ur_zscore_5"] = float(np.clip((prev_ur_1 - mu5) / sd5, -5.0, 5.0))
    else:
        feat["btc_up_ratio_zscore_5s"] = 0.0
        feat["lag_ur_zscore_5"] = 0.0

    # ── OB features: fetch real order book via CLOB REST ──────────────────────
    # Identical computation to training (pmdata poly_l2 book snapshots).
    # Called once per slot at ~t=150s (after tick observation window ends).
    # up_token_id must be passed in; on failure fills with None (market excluded).
    feat.update(_build_ob_features(up_token_id))

    # ── Spot features ──────────────────────────────────────────────────────────
    feat.update(build_spot_features(slot_ts))

    # ── Cross-domain interaction features (OB x Spot) ────────────────────────
    # ob_mid_drift removed → x_drift_x_ret5m and x_ob_drift_x_inslot removed
    feat["x_imb_x_ur"]         = feat.get("ob_imbalance", 0.0) * feat.get("btc_up_ratio", 0.5)
    feat["x_depth_x_momentum"] = feat.get("ob_depth_ratio", 1.0) * feat.get("btc_momentum", 0.0)
    feat["x_imb_end_x_ret"]    = feat.get("ob_imbalance_end", 0.0) * feat.get("btc_inslot_ret", 0.0)
    feat["x_imb_x_inslot"]     = feat.get("ob_imbalance", 0.0) * feat.get("btc_inslot_ret", 0.0)
    feat["x_spread_x_vol"]     = feat.get("ob_spread", 0.02) * feat.get("btc_vol_1h", 0.0)
    feat["x_depth_x_vol"]      = feat.get("ob_depth_ratio", 1.0) * feat.get("btc_vol_1h", 0.0)

    # CLOB WS features — only the 5 OK ones (clob_spread/mid/ask_pressure)
    # clob_imb_*/depth_trend/activity_rate removed (❌ quase sempre 0.0 live)
    if up_token_id:
        _acc = get_accumulator()

        # clob_* WS features — janela [slot_ts, agora)
        # obs_secs dinâmico = tempo decorrido desde slot_ts + margem de 5s.
        # Isso garante que TODOS os eventos acumulados no slot (book WS + hydration REST)
        # entrem na janela, independente de quando chegaram.
        # BUG ANTERIOR: obs_secs=60 fixo fazia cutoff_wall = slot_ts+60; eventos
        # chegados em t>60s (hydration REST, price_change tardios) ficavam fora → zeros.
        _clob_obs_secs = max(60, int(time.time() - slot_ts) + 5)
        clob_feats = _acc.get_features(
            up_token_id,
            slot_ts=slot_ts,
            obs_secs=_clob_obs_secs,
            window_secs=float(_clob_obs_secs),  # janela = slot inteiro até agora
        )
        CLOB_KEEP = {"clob_spread_mean", "clob_spread_trend",
                     "clob_mid_volatility", "clob_ask_pressure"}
        for k, v in clob_feats.items():
            if k in CLOB_KEEP:
                feat[k] = v

        # ob_pc_* features — price_change events desde slot_ts (t=[0,168s))
        pc_feats = _acc.get_ob_pc_features(up_token_id, slot_ts, cutoff_secs=168)
        feat["ob_pc_up_ratio"]    = pc_feats.get("ob_pc_up_ratio", 0.5)
        feat["ob_pc_count"]       = pc_feats.get("ob_pc_count", 0.0)
        feat["ob_fill_imbalance"] = pc_feats.get("ob_fill_imbalance", 0.0)

        # ob_imb_w0 — mean imbalance de book reais em t=[0,60s)
        imb_windows = _acc.get_windowed_imbalance(
            up_token_id, slot_ts,
            windows=[(0, 60), (60, 120), (120, 168)]
        )
        feat["ob_imb_w0"] = imb_windows.get("ob_imb_w0", 0.0)

    # ── Final: fill any remaining model features with context-aware defaults ────
    _neutral_defaults = {
        "ob_mid":          0.5,
        "ob_ask_depth_5c": 0.5,
        "ob_bid_depth_5c": 0.5,
        "ob_depth_ratio":  1.0,
        "ob_total_depth":  1000.0,
        "ob_spread":       0.02,
    }
    for f in features:
        if f not in feat:
            feat[f] = _neutral_defaults.get(f, 0.0)
    return feat






def _fetch_seed_up_ratio(slot_ts: int, market_data: dict) -> float:
    """Fetch real up_ratio for a past slot from data-api trades.

    Uses a single page of trades (500) per token, filtered by slug.
    Returns 0.5 on failure (neutral fallback).
    """
    slug = f"btc-updown-5m-{slot_ts}"
    token_ids = market_data.get("clobTokenIds", "[]")
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)
    if len(token_ids) < 2:
        return 0.5

    vol_up = 0.0
    vol_dn = 0.0
    for token_id in token_ids[:2]:
        try:
            r = _http.get(f"{DATA_API}/trades",
                          params={"asset": token_id, "limit": 500},
                          timeout=5)
            if not r.ok:
                continue
            for t in r.json():
                if not isinstance(t, dict):
                    continue
                if t.get("slug", "") != slug:
                    continue
                ts = int(t.get("timestamp", 0))
                if ts > 1e12:
                    ts //= 1000
                if not (0 <= ts - slot_ts < OBSERVE_SECS):
                    continue
                outcome = t.get("outcome", "")
                price = float(t.get("price", 0) or 0)
                size = float(t.get("size", 0) or 0)
                usdc = price * size
                if outcome == "Up":
                    vol_up += usdc
                elif outcome == "Down":
                    vol_dn += usdc
        except Exception:
            continue

    total = vol_up + vol_dn
    if total > 0:
        return vol_up / total
    return 0.5


def _seed_slot_history():
    """
    Pre-populate _slot_history with the last _HIST_MAX resolved BTC 5min slots
    so zscore and lag features are warm from the very first prediction.

    Fetches from Gamma API (public, no auth). For each resolved slot:
      - up_ratio from outcomePrices (proxy: UP price ≈ market-implied up_ratio)
      - target from resolution (1=UP, 0=DOWN)
      - sw filled with [up_ratio]*6 (no per-window data available historically)
      - pre_ret from Binance spot buffer if available, else 0.0

    Falls back gracefully if API is down or returns no data.
    """
    log.info("Seeding slot history from recent resolved markets...")
    try:
        now = int(time.time())
        # Walk backwards through the last 30 slots (2h30min) to find _HIST_MAX resolved ones
        seeded = 0
        entries = []
        for i in range(1, 35):
            slot_ts = ((now // SLOT_DURATION) - i) * SLOT_DURATION
            slug = f"btc-updown-5m-{slot_ts}"
            try:
                r = _http.get(f"{GAMMA_HOST}/markets/slug/{slug}", timeout=5)
                if not r.ok:
                    continue
                m = r.json()
                # Need a resolved market
                op = m.get("outcomePrices", "[]")
                if isinstance(op, str):
                    op = json.loads(op)
                if not op or len(op) < 2:
                    continue
                up_price = float(op[0])
                dn_price = float(op[1])
                # Only use if clearly resolved (one side >= 0.99)
                if up_price >= 0.99:
                    target = 1
                elif dn_price >= 0.99:
                    target = 0
                else:
                    continue  # not resolved yet
                # Fetch REAL up_ratio from tick data for this slot.
                # This avoids zscore explosion from constant seed values.
                up_ratio = _fetch_seed_up_ratio(slot_ts, m)
                sw = [up_ratio] * 6  # approximate per-window with overall ratio

                # pre_ret from spot buffer if warm
                pre_ret = 0.0
                if SPOT_BUFFER.exists():
                    try:
                        buf = json.loads(SPOT_BUFFER.read_text())
                        candles = buf.get("btcusdt", [])
                        if candles:
                            ts_arr = [c[0] for c in candles]
                            px_arr = [c[1] for c in candles]
                            # find 5m window before slot
                            seg = [px_arr[j] for j, ts in enumerate(ts_arr)
                                   if slot_ts - 300 <= ts < slot_ts]
                            if len(seg) >= 2:
                                pre_ret = float((seg[-1] - seg[0]) / (seg[0] + 1e-8))
                    except Exception:
                        pass

                entries.append({
                    "slot_ts":  slot_ts,
                    "up_ratio": up_ratio,
                    "sw":       sw,
                    "pre_ret":  pre_ret,
                    "target":   target,
                })
                seeded += 1
                if seeded >= _HIST_MAX:
                    break
            except Exception:
                continue

        # Insert in chronological order (oldest first)
        for entry in reversed(entries):
            _slot_history.append(entry)

        log.info("Slot history seeded: %d entries (oldest=%s, newest=%s)",
                 len(_slot_history),
                 str(_slot_history[0]["slot_ts"]) if _slot_history else "—",
                 str(_slot_history[-1]["slot_ts"]) if _slot_history else "—")
    except Exception as e:
        log.warning("_seed_slot_history failed: %s — starting with empty history", e)



# Each entry: {slot_ts, up_ratio, sw: [6 floats], pre_ret: float, target: int|None}
# 'target' is filled in by settle_trades() or left None until settlement.
_slot_history: list[dict] = []
_HIST_MAX = 25  # keep last 25 slots (enough for 20-slot zscore window)

def _update_slot_history(slot_ts: int, up_ratio: float, sw: list[float],
                          pre_ret: float = 0.0, target: int | None = None):
    """Append or update a slot entry in the ring buffer."""
    global _slot_history
    # Update if already present (e.g. when target resolves)
    for entry in _slot_history:
        if entry["slot_ts"] == slot_ts:
            if target is not None:
                entry["target"] = target
            return
    _slot_history.append({
        "slot_ts":   slot_ts,
        "up_ratio":  up_ratio,
        "sw":        sw,
        "pre_ret":   pre_ret,
        "target":    target,
        "n_ticks":   0.0,    # filled by _update_slot_history callers
        "vol_total": 0.0,    # filled by _update_slot_history callers
    })
    if len(_slot_history) > _HIST_MAX:
        _slot_history = _slot_history[-_HIST_MAX:]


# ── Trades log ─────────────────────────────────────────────────────────────────
def load_trades() -> list[dict]:
    if TRADES_FILE.exists():
        try:
            return json.loads(TRADES_FILE.read_text())
        except Exception:
            pass
    return []


def rebuild_trades_from_chain(proxy_wallet: str) -> list[dict]:
    """
    Reconstruct trade history from Polymarket activity API on startup.
    Prevents data loss across Fly.io restarts (/tmp is ephemeral).
    Only covers today's 5m BTC slots.
    """
    try:
        r = _http.get(f"{DATA_API}/activity",
                         params={"user": proxy_wallet, "limit": 100},
                         timeout=10)  # rebuild is not in critical path — can wait
        if not r.ok:
            return []
        activity = r.json()
        if not isinstance(activity, list):
            return []

        # Group trades and redeems by slug
        trades_by_slug: dict[str, dict] = {}
        redeems_by_slug: dict[str, float] = {}
        now = int(time.time())
        cutoff = now - 86400  # last 24h

        for item in activity:
            slug = item.get("slug", "")
            if "btc-updown-5m" not in slug:
                continue
            ts = item.get("timestamp", 0)
            if ts < cutoff:
                continue
            try:
                slot_ts = int(slug.split("-")[-1])
            except Exception:
                continue

            if item["type"] == "TRADE":
                if slug not in trades_by_slug:
                    trades_by_slug[slug] = {
                        "slot_ts":    slot_ts,
                        "direction":  "UP" if item.get("outcome") == "Up" else "DOWN",
                        "entry_price": float(item.get("price", 0)),
                        "shares":     0.0,
                        "actual_cost": 0.0,
                        "stake_usdc": STAKE_USDC,
                        "token_id":   item.get("asset", ""),
                        "order_id":   item.get("transactionHash", "")[:40],
                        "entered_at": ts,
                        "confidence": 0.0,
                        "true_edge":  0.0,
                        "source":     "chain_rebuild",
                    }
                trades_by_slug[slug]["shares"]     += float(item.get("size", 0))
                trades_by_slug[slug]["actual_cost"] += float(item.get("usdcSize", 0))
            elif item["type"] == "REDEEM":
                redeems_by_slug[slug] = float(item.get("usdcSize", 0))

        # Determine status for each slot
        rebuilt = []
        for slug, trade in trades_by_slug.items():
            slot_ts = trade["slot_ts"]
            redeemed = redeems_by_slug.get(slug, 0)
            slot_end = slot_ts + SLOT_DURATION + SETTLE_GRACE
            if redeemed > 0:
                cost = trade["actual_cost"]
                pnl  = round(redeemed - cost, 4)
                trade.update({"status": "settled", "result": "WIN" if pnl > 0 else "LOSS",
                               "pnl_usdc": pnl, "settled_at": slot_end,
                               "actual": trade["direction"]})
            elif now > slot_end:
                # Past end + grace with no redeem = LOSS
                trade.update({"status": "settled", "result": "LOSS",
                               "pnl_usdc": round(-trade["actual_cost"], 4),
                               "settled_at": slot_end, "actual": "DOWN" if trade["direction"] == "UP" else "UP"})
            else:
                trade["status"] = "open"
            rebuilt.append(trade)

        rebuilt.sort(key=lambda t: t["slot_ts"])
        return rebuilt
    except Exception as e:
        log.warning("rebuild_trades_from_chain failed: %s", e)
        return []

def save_trades(trades: list[dict]):
    TRADES_FILE.write_text(json.dumps(trades, indent=2))


# ── Settlement ──────────────────────────────────────────────────────────────────
def settle_trades(trades: list[dict]) -> bool:
    global _dry_virtual_balance
    now     = int(time.time())
    updated = False
    for trade in trades:
        if trade.get("status") not in ("open", "dry_open"):
            continue
        if now < trade["slot_ts"] + SLOT_DURATION + SETTLE_GRACE:
            continue
        slug = f"btc-updown-5m-{trade['slot_ts']}"
        resolution = None
        try:
            r = _http.get(f"{GAMMA_HOST}/markets/slug/{slug}", timeout=HTTP_TIMEOUT)
            if r.ok:
                data = r.json()
                resolution = data.get("resolution")
                if resolution is None:
                    op = data.get("outcomePrices", "[]")
                    if isinstance(op, str):
                        op = json.loads(op)
                    if op and len(op) >= 2:
                        if float(op[0]) >= 0.99:
                            resolution = 1.0
                        elif float(op[1]) >= 0.99:
                            resolution = 0.0
        except Exception:
            pass
        if resolution is None:
            continue
        try:
            resolution = float(resolution)
        except (TypeError, ValueError):
            log.warning("  Unresolvable resolution value: %r — skipping", resolution)
            continue
        actual     = "UP" if resolution >= 0.5 else "DOWN"
        target_int = 1 if actual == "UP" else 0
        # Backfill resolved target into slot history (improves future lag features)
        _update_slot_history(trade["slot_ts"], up_ratio=0.5, sw=[0.5]*6, target=target_int)
        direction  = trade["direction"]
        entry      = trade["entry_price"]
        # Use actual_cost if stored (bumped-to-min-5 case), else derive from stake
        cost       = trade.get("actual_cost") or trade.get("stake_usdc") or STAKE_USDC
        shares_out = trade.get("shares") if trade.get("shares") else (cost / (entry + 1e-8))
        if direction == actual:
            # Deduct taker fee from gross proceeds
            pnl    = round(shares_out * (1.0 - TAKER_FEE) - cost, 4)
            result = "WIN"
        else:
            pnl, result = round(-cost, 4), "LOSS"
        trade.update({"status": "settled", "actual": actual,
                      "result": result, "pnl_usdc": pnl, "settled_at": now})
        updated = True
        # Resolve CLOB feature log with ground truth (for future training)
        resolve_clob_features(trade["slot_ts"], float(target_int))
        # Dry run: update virtual balance
        if trade.get("dry"):
            _dry_virtual_balance += pnl
            log.info("DRY SETTLED slot=%d | pred=%s actual=%s | %s | entry=$%.3f shares=%.0f cost=$%.2f P&L $%+.2f | virtual_bal=$%.2f",
                     trade["slot_ts"], direction, actual, result,
                     entry, shares_out, cost, pnl, _dry_virtual_balance)
        else:
            log.info("SETTLED slot=%d | pred=%s actual=%s | %s | entry=$%.3f shares=%.0f cost=$%.2f P&L $%+.2f",
                     trade["slot_ts"], direction, actual, result,
                     entry, shares_out, cost, pnl)
    return updated


def _backfill_history_targets():
    """Resolve targets for _slot_history entries that still have target=None.
    This covers slots that were skipped (low conf / no edge) and thus never
    went through settle_trades(). Without this, lag_N_outcome stays at 0.5
    for all skipped slots indefinitely.
    Only queries Gamma for slots that closed >60s ago (SETTLE_GRACE).
    Caches successes by updating in-place — idempotent.
    """
    now = int(time.time())
    backfilled = 0
    for entry in _slot_history:
        if backfilled >= 5:
            break  # max 5 Gamma calls per loop iteration to avoid blocking entry window
        if entry.get("target") is not None:
            continue
        slot_ts = entry["slot_ts"]
        if now < slot_ts + SLOT_DURATION + SETTLE_GRACE:
            continue  # not resolved yet
        slug = f"btc-updown-5m-{slot_ts}"
        try:
            r = _http.get(f"{GAMMA_HOST}/markets/slug/{slug}", timeout=5)
            if not r.ok:
                continue
            data = r.json()
            resolution = data.get("resolution")
            if resolution is None:
                op = data.get("outcomePrices", "[]")
                if isinstance(op, str):
                    op = json.loads(op)
                if op and len(op) >= 2:
                    if float(op[0]) >= 0.99:
                        resolution = 1.0
                    elif float(op[1]) >= 0.99:
                        resolution = 0.0
            if resolution is not None:
                entry["target"] = 1 if float(resolution) >= 0.5 else 0
                backfilled += 1
        except Exception:
            pass


# ── Main loop ───────────────────────────────────────────────────────────────────
def run(client, model, features):
    global _dry_virtual_balance
    gate = DataQualityGate(features_list=features, warmup_slots=3, min_subwindows_with_ticks=1)
    log.info("Live trader started | stake=$%.2f | min_conf=%.0f%% | min_edge=%.0f%%",
             STAKE_USDC, MIN_CONFIDENCE*100, MIN_EDGE*100)
    log.info("DataQualityGate active — warming up for %d slots", gate.warmup_slots)

    if DRY_RUN:
        log.info("DRY RUN active | virtual_balance=$%.2f | min_conf=%.0f%% | min_edge=%.0f%%",
                 _dry_virtual_balance, MIN_CONFIDENCE * 100, MIN_EDGE * 100)
    else:
        # Print balance
        try:
            bal  = client.get_balance_allowance(asset_type="COLLATERAL")
            usdc = float(bal.balance) / 1e6
            log.info("Wallet balance: $%.2f USDC", usdc)
        except Exception as e:
            log.warning("Balance check failed: %s", e)

    # Rebuild trade history from on-chain activity (survives restarts)
    if not TRADES_FILE.exists() and not DRY_RUN:
        log.info("Rebuilding trade history from chain...")
        rebuilt = rebuild_trades_from_chain(PROXY_WALLET)
        if rebuilt:
            save_trades(rebuilt)
            settled = [t for t in rebuilt if t.get("status") == "settled"]
            wins    = [t for t in settled if t.get("result") == "WIN"]
            pnl     = sum(t.get("pnl_usdc", 0) for t in settled)
            log.info("Rebuilt %d trades from chain: W%d/L%d P&L=$%.2f",
                     len(rebuilt), len(wins), len(settled)-len(wins), pnl)
        else:
            log.info("No prior trades found on chain")

    # Pre-populate _slot_history from recent resolved markets so zscore/lag
    # features are warm from the first prediction instead of needing ~1h40min.
    _seed_slot_history()

    _loop_count = 0
    while True:
      try:
        now    = int(time.time())
        _loop_count += 1
        trades = load_trades()

        # ── Debug: WS health every 30 loops (~5 min) ─────────────────
        if _loop_count % 30 == 1:
            if _spot_ws_manager:
                sh = _spot_ws_manager.health()
                log.info("WS-HEALTH [binance-spot] connected=%s uptime=%ds msgs=%d rate=%d/min disconnects=%d zombies=%d",
                         sh["connected"], sh["current_uptime_s"], sh["total_messages"],
                         sh["msgs_per_min"], sh["total_disconnects"], sh["zombie_kills"])
            if _clob_ws_manager:
                ch = _clob_ws_manager.health()
                log.info("WS-HEALTH [clob] connected=%s uptime=%ds msgs=%d rate=%d/min disconnects=%d zombies=%d",
                         ch["connected"], ch["current_uptime_s"], ch["total_messages"],
                         ch["msgs_per_min"], ch["total_disconnects"], ch["zombie_kills"])
            # Spot buffer freshness
            if SPOT_BUFFER.exists():
                try:
                    buf_data = json.loads(SPOT_BUFFER.read_text())
                    buf_age = now - buf_data.get("updated_at", 0)
                    btc_len = len(buf_data.get("btcusdt", []))
                    # Check 4h lookback coverage (need ~240 candles for btc_pre_4h_ret)
                    import numpy as _np
                    _candles = buf_data.get("btcusdt", [])
                    _ts_arr = [c[0] for c in _candles] if _candles else []
                    _now_ts = now
                    _covered_1h  = sum(1 for t in _ts_arr if _now_ts - 3600 <= t <= _now_ts)
                    _covered_4h  = sum(1 for t in _ts_arr if _now_ts - 14400 <= t <= _now_ts)
                    _buf_warn = " ⚠️ LOW_4H" if _covered_4h < 180 else ""
                    log.info("SPOT-BUFFER age=%ds btc_candles=%d | 1h=%d 4h=%d candles%s",
                             buf_age, btc_len, _covered_1h, _covered_4h, _buf_warn)
                except Exception as e:
                    log.warning("SPOT-BUFFER read error: %s", e)
            else:
                log.warning("SPOT-BUFFER missing!")

        # 1. Settle
        if settle_trades(trades):
            save_trades(trades)
            _print_summary(trades)

        # 1b. Backfill targets for skipped/no-trade slots in _slot_history
        # settle_trades() only fills targets for slots with open trades.
        # Slots that were skipped (low confidence / no edge) never get a target,
        # causing lag_N_outcome to stay at 0.5 indefinitely — degrading lag features.
        _backfill_history_targets()

        # 2. Enter — only block re-entry for open/settled/error/dry_open, NOT skipped
        already = {t["slot_ts"] for t in trades
                   if t.get("status") in ("open", "settled", "error", "dry_open")}
        cur_slot = (now // SLOT_DURATION) * SLOT_DURATION

        # Clear OB open snapshot cache when slot changes (new observation window)
        global _ob_last_slot
        if cur_slot != _ob_last_slot:
            _ob_open_cache.clear()
            # Reset CLOB accumulator buffers (só reancora slot_ts, não limpa eventos)
            _acc = get_accumulator()
            for tid in list(_clob_subscribed):
                _acc.reset_token(tid, slot_ts=cur_slot)
            _ob_last_slot = cur_slot

            # Pre-subscribe ao slot ATUAL e PRÓXIMO logo no início do slot (t~0s).
            # Sem isso, o subscribe só ocorre em fetch_market() na entry window (t~170s),
            # e o WS leva ~60s para enviar o primeiro book snapshot → CLOB empty em t=170-230s.
            # Pre-subscribendo agora garantimos dados desde t~30s, prontos para t=170s.
            for offset in [0, SLOT_DURATION]:
                _pre_slot = cur_slot + offset
                _pre_mkt = fetch_market(_pre_slot)
                if _pre_mkt:
                    clob_subscribe([_pre_mkt["yes_token"], _pre_mkt["no_token"]], slot_ts=_pre_slot)
                    log.info("CLOB WS pre-subscribed slot=%d (t=%ds)", _pre_slot, int(now - _pre_slot))

        # ── OB early snapshot: fetch "open" OB at t~60s (well before entry window) ──
        # This gives temporal separation between open (t~60s) and close (t~170s+)
        # snapshots, enabling real ob_mid_drift, ob_imb_momentum, and windowed
        # imbalance. Training data had snapshots ~180s apart; we aim for ~110s gap.
        t_in_slot = now - cur_slot
        if 50 <= t_in_slot <= 90 and cur_slot not in already:
            # Fetch market to get token IDs for OB pre-fetch
            mkt = fetch_market(cur_slot)
            if mkt and mkt["yes_token"] not in _ob_open_cache:
                snap = _fetch_ob_snapshot(mkt["yes_token"])
                if snap:
                    _ob_open_cache[mkt["yes_token"]] = {
                        "mid": snap["mid"],
                        "spread": snap["spread"],
                        "imbalance": snap["imbalance"],
                        "total_depth": float(sum(float(b["size"]) for b in snap["bids"]) +
                                             sum(float(a["size"]) for a in snap["asks"])),
                        "ts": snap["ts"],
                    }
                    log.info("OB early snapshot cached for slot=%d (t=%ds, mid=%.4f, imb=%.3f)",
                             cur_slot, int(t_in_slot), snap["mid"], snap["imbalance"])

        for slot_ts in [cur_slot]:  # prev-slot check removed: t_elapsed would be 300-599, outside ENTER_WINDOW [170,240]
            t_elapsed = now - slot_ts
            if not (ENTER_WINDOW[0] <= t_elapsed <= ENTER_WINDOW[1]):
                continue
            if slot_ts in already:
                continue

            log.info("Entry window: slot=%d t=%ds", slot_ts, t_elapsed)
            market = fetch_market(slot_ts)
            if not market:
                log.info("  Market not found")
                continue

            ticks = fetch_inslot_trades(market["yes_token"], market["no_token"], slot_ts)
            log.info("  Fetched %d inslot ticks (data-api lag ~120s)", len(ticks))

            # ── GATE 0: Cold start protection ──────────────────────────
            if not gate.is_warm():
                gate.record_slot_observed()
                continue

            # ── GATE 1: Data completeness ──────────────────────────────
            ok, reason = gate.check_data_completeness(ticks, SPOT_BUFFER, slot_ts)
            if not ok:
                log.info("  Skip — DATA GATE: %s", reason)
                continue

            t_feat_start = time.time()
            feat = build_features(ticks, slot_ts, features,
                                   up_token_id=market["yes_token"])
            t_feat_ms = (time.time() - t_feat_start) * 1000
            if feat is None:
                log.warning("  build_features returned None (took %.0fms)", t_feat_ms)
                continue
            nz = sum(1 for f in features if feat.get(f, 0.0) != 0.0)
            log.info("  build_features OK (%.0fms) — %d/%d features non-zero",
                     t_feat_ms, nz, len(features))
            if nz < len(features):
                zero_feats = [f for f in features if feat.get(f, 0.0) == 0.0]
                log.info("  zero features: %s", zero_feats)

            # ── GATE 2: Feature sanity ─────────────────────────────────
            ok, reason = gate.check_feature_sanity(feat)
            if not ok:
                log.warning("  Skip — FEATURE GATE: %s", reason)
                continue

            # Push this slot into history ring buffer (target unknown until settlement)
            _update_slot_history(
                slot_ts=slot_ts,
                up_ratio=feat.get("btc_up_ratio", 0.5),
                sw=[feat.get(f"btc_up_w{i}", 0.5) for i in range(6)],
                pre_ret=feat.get("btc_pre_5m_ret", 0.0),
            )
            # Fill in n_ticks and vol_total for prev_slot_* lag features
            if _slot_history and _slot_history[-1]["slot_ts"] == slot_ts:
                _slot_history[-1]["n_ticks"]   = float(feat.get("btc_n_ticks", 0.0))
                _slot_history[-1]["vol_total"]  = float(
                    feat.get("btc_vol_up", 0.0) + feat.get("btc_vol_dn", 0.0)
                )

            log.info("  btc_up_ratio=%.3f momentum=%.3f tw=%.3f n_ticks=%d",
                     feat.get("btc_up_ratio", 0.5), feat.get("btc_momentum", 0),
                     feat.get("btc_tw_up_ratio", 0.5), int(feat.get("btc_n_ticks", 0)))
            # Log CLOB real-time features
            _cf = {k: feat.get(k, 0.0) for k in CLOB_FEATURE_NAMES}
            clob_nonzero = sum(1 for v in _cf.values() if v != 0.0)
            if clob_nonzero > 0:
                log.info("  CLOB features: %d/%d non-zero | imb=%.3f vel=%.4f act=%.1f/s",
                         clob_nonzero, len(CLOB_FEATURE_NAMES),
                         _cf.get("clob_imb_mean", 0), _cf.get("clob_mid_velocity", 0),
                         _cf.get("clob_activity_rate", 0))
            else:
                log.info("  CLOB features: no data yet (accumulator empty)")

            X = pd.DataFrame([[feat.get(f, 0.0) for f in features]], columns=features)
            prob_up = predict_proba(model, X)
            direction  = "UP" if prob_up >= 0.5 else "DOWN"
            confidence = prob_up if direction == "UP" else 1.0 - prob_up
            log.info("  Prediction: %s  conf=%.1f%%  prob_up=%.4f", direction, confidence*100, prob_up)
            # Log all model features (sorted by |value| desc) for diagnosis
            _feat_vals = [(f, feat.get(f, 0.0)) for f in features]
            _feat_vals.sort(key=lambda x: abs(x[1]), reverse=True)
            _fv = " | ".join(f"{f}={v:.4f}" for f, v in _feat_vals)
            log.info("  FEATS (%d): %s", len(features), _fv)

            # ── Persist full feature snapshot to JSONL for audit/debug ──────
            # /tmp/all_features_log.jsonl — uma linha por avaliação de slot
            try:
                import time as _time
                _feat_row = {
                    "slot_ts": slot_ts,
                    "t_in_slot": t_elapsed,
                    "logged_at": round(_time.time(), 2),
                    "prob_up": round(prob_up, 4),
                    "direction": direction,
                    "confidence": round(confidence, 4),
                    **{f: round(feat.get(f, 0.0), 6) for f in features},
                }
                with open("/tmp/all_features_log.jsonl", "a") as _fh:
                    _fh.write(json.dumps(_feat_row) + "\n")
            except Exception as _e:
                log.debug("feature log write failed: %s", _e)

            # Log CLOB features for future v25 training
            clob_feats = {k: feat.get(k, 0.0) for k in CLOB_FEATURE_NAMES}
            log_clob_features(
                slot_ts=slot_ts,
                token_id=market["yes_token"],
                clob_features=clob_feats,
                model_prob=prob_up,
                t_in_slot=t_elapsed,
            )

            # ── GATE 3: Prediction sanity ──────────────────────────────
            ok, reason = gate.check_prediction_sanity(prob_up, feat)
            if not ok:
                log.warning("  Skip — PREDICTION GATE: %s", reason)
                continue

            if confidence < MIN_CONFIDENCE:
                log.info("  Skip — conf %.1f%% < %.0f%%", confidence*100, MIN_CONFIDENCE*100)
                trades.append({"slot_ts": slot_ts, "direction": direction,
                                "confidence": round(confidence,4), "status": "skipped",
                                "reason": f"conf {confidence:.2%} < {MIN_CONFIDENCE:.0%}",
                                "entered_at": now})
                save_trades(trades)
                continue

            # ── GATE 4: CLOB confirmation — skip if real-time order flow contradicts ──
            # If model says UP but CLOB buyers are absent (imb strongly negative),
            # or model says DOWN but CLOB is strongly bid-heavy, likely a false signal.
            # NOTE: This gate is SOFT — only fires on strong contradiction (>0.3).
            # Once we train v25 with CLOB features, this manual gate becomes unnecessary.
            CLOB_CONTRADICTION_THRESHOLD = 0.3
            clob_imb = feat.get("clob_imb_mean", 0.0)
            if clob_imb != 0.0:  # 0.0 means no data (CLOB accumulator empty)
                if direction == "UP" and clob_imb < -CLOB_CONTRADICTION_THRESHOLD:
                    log.info("  Skip — CLOB CONTRADICTION: model=UP but clob_imb=%.3f (sellers dominate)", clob_imb)
                    trades.append({"slot_ts": slot_ts, "direction": direction,
                                    "confidence": round(confidence, 4), "status": "skipped",
                                    "reason": f"clob_contradiction imb={clob_imb:.3f}",
                                    "entered_at": now})
                    save_trades(trades)
                    continue
                elif direction == "DOWN" and clob_imb > CLOB_CONTRADICTION_THRESHOLD:
                    log.info("  Skip — CLOB CONTRADICTION: model=DOWN but clob_imb=%.3f (buyers dominate)", clob_imb)
                    trades.append({"slot_ts": slot_ts, "direction": direction,
                                    "confidence": round(confidence, 4), "status": "skipped",
                                    "reason": f"clob_contradiction imb={clob_imb:.3f}",
                                    "entered_at": now})
                    save_trades(trades)
                    continue

            token_id   = market["yes_token"] if direction == "UP" else market["no_token"]

            # Fetch ask price AND market mid in PARALLEL — saves ~1-5s
            from concurrent.futures import ThreadPoolExecutor
            t_price_start = time.time()
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="price-fetch") as executor:
                fut_ask = executor.submit(get_ask_price, token_id)
                fut_mid = executor.submit(get_market_mid, slot_ts)
                ask_price = fut_ask.result(timeout=20)
                up_mid, down_mid = fut_mid.result(timeout=20)
            t_price_ms = (time.time() - t_price_start) * 1000
            log.info("  get_ask_price+mid: ask=$%.3f (%.0fms parallel)", ask_price, t_price_ms)
            model_prob = prob_up if direction == "UP" else (1.0 - prob_up)

            # Log ask price source and freshness
            with _clob_prices_lock:
                ws_ts = _clob_price_ts.get(token_id, 0)
            ask_age = time.time() - ws_ts if ws_ts else -1
            ask_src = f"WS {ask_age:.1f}s" if ws_ts else "HTTP"

            # Reject extreme ask prices — token is illiquid, already one-sided, or no edge
            # v23: tightened to [0.42, 0.65] from [0.38, 0.90]
            #   - Below $0.42: market strongly against us (29% WR historically)
            #   - Above $0.65: risk/reward terrible (need 66%+ WR to breakeven)
            #     At ask=$0.70, risk $0.70 to win $0.28 (2.5:1 against)
            #     $0.70-$0.80 bucket lost $29 on 18 trades despite 50% WR
            if not (0.42 <= ask_price <= 0.65):
                log.info("  Skip — ask $%.3f outside [0.42, 0.65] | model=%s conf=%.1f%% prob_up=%.4f",
                         ask_price, direction, confidence*100, prob_up)
                trades.append({"slot_ts": slot_ts, "direction": direction,
                                "confidence": round(confidence, 4),
                                "entry_price": round(ask_price, 4), "status": "skipped",
                                "reason": f"ask ${ask_price:.3f} outside valid range [0.42, 0.65]",
                                "entered_at": now})
                save_trades(trades)
                # ask == 0.0 means market decided/one-sided — no point retrying this slot
                if ask_price == 0.0:
                    already.add(slot_ts)
                    log.info("  Market decided/one-sided — blocking re-entry for slot %d", slot_ts)
                continue
            edge_vs_ask = model_prob - ask_price

            # Market mid already fetched in parallel above
            market_mid  = up_mid if direction == "UP" else down_mid
            edge_vs_mid = (model_prob - market_mid) if market_mid else None

            log.info("  BUY %s | ask=$%.3f [%s] edge_ask=%.1f%% | market_mid=$%.3f edge_mid=%.1f%%",
                     direction, ask_price, ask_src, edge_vs_ask * 100,
                     market_mid or 0, (edge_vs_mid or 0) * 100)

            # Sanity check: ask must not diverge >0.35 from market mid.
            # BTC 5min markets move fast late in slot — midpoint reflects
            # near-resolution while book still has tradeable liquidity.
            # 0.35 allows normal late-slot movement while catching true stale prices.
            if market_mid is not None and abs(ask_price - market_mid) > 0.35:
                log.warning("  Skip — ask $%.3f diverges %.2f from mid $%.3f (stale/bad price)",
                            ask_price, abs(ask_price - market_mid), market_mid)
                trades.append({"slot_ts": slot_ts, "direction": direction,
                                "confidence": round(confidence, 4),
                                "entry_price": round(ask_price, 4), "status": "skipped",
                                "reason": f"ask ${ask_price:.3f} diverges from mid ${market_mid:.3f} by >{abs(ask_price-market_mid):.2f}",
                                "entered_at": now})
                save_trades(trades)
                continue

            # Require edge vs ask >= 10% AND edge vs market mid >= 5%
            if edge_vs_ask < MIN_EDGE:
                log.info("  Skip — edge_ask %.1f%% < %.0f%%", edge_vs_ask * 100, MIN_EDGE * 100)
                trades.append({"slot_ts": slot_ts, "direction": direction,
                                "confidence": round(confidence, 4),
                                "entry_price": round(ask_price, 4), "status": "skipped",
                                "reason": f"edge_ask {edge_vs_ask:.2%} < {MIN_EDGE:.0%}",
                                "edge_vs_ask": round(edge_vs_ask, 4),
                                "edge_vs_mid": round(edge_vs_mid, 4) if edge_vs_mid else None,
                                "entered_at": now})
                save_trades(trades)
                continue

            if edge_vs_mid is not None and edge_vs_mid < MIN_EDGE_MID:
                log.info("  Skip — edge_mid %.1f%% < %.0f%% (market already priced in)",
                         edge_vs_mid * 100, MIN_EDGE_MID * 100)
                trades.append({"slot_ts": slot_ts, "direction": direction,
                                "confidence": round(confidence, 4),
                                "entry_price": round(ask_price, 4), "status": "skipped",
                                "reason": f"edge_mid {edge_vs_mid:.2%} < {MIN_EDGE_MID:.0%}",
                                "edge_vs_ask": round(edge_vs_ask, 4),
                                "edge_vs_mid": round(edge_vs_mid, 4),
                                "entered_at": now})
                save_trades(trades)
                continue

            true_edge = edge_vs_ask

            # ── Compute shares + actual cost (CLOB min 5 shares) ───────────
            # Fetch balance first — needed for both auto-sizing and balance check
            if DRY_RUN:
                usdc = _dry_virtual_balance
            else:
                try:
                    bal  = client.get_balance_allowance(asset_type="COLLATERAL")
                    usdc = float(bal.balance) / 1e6
                except Exception as e:
                    log.warning("  Balance check FAILED: %s — using FIXED_SHARES without confirmation", e)
                    usdc = None

            if usdc is not None:
                shares = compute_shares(usdc, ask_price)
            else:
                # Balance fetch failed — use fixed shares and SKIP trade (no balance = no trade)
                log.warning("  Balance unknown — skipping trade (cannot verify funds)")
                continue

            # order_price = ask + 1 tick for taker-aggressive fill (see placement block below)
            TICK = 0.01
            order_price = min(round(ask_price + TICK, 2), 0.96)
            actual_cost = round(shares * order_price, 4)
            sizing_mode = "auto" if AUTO_SHARES else "fixed"
            if AUTO_SHARES and usdc is not None:
                _floor, _ceil = AUTO_SHARES_BAL_FLOOR, AUTO_SHARES_BAL_CEIL
                _t = max(0.0, min(1.0, (usdc - _floor) / max(_ceil - _floor, 1)))
                _shares_by_interp = AUTO_SHARES_MIN + _t * (AUTO_SHARES_MAX - AUTO_SHARES_MIN)
                _max_cost = usdc * 0.10
                _shares_by_risk  = _max_cost / (ask_price + 1e-8)
                log.info("  %d shares @ $%.3f — cost $%.2f [auto bal=$%.2f t=%.2f interp=%.1f risk_cap=%.1f final=%d]",
                         int(shares), ask_price, actual_cost, usdc,
                         _t, _shares_by_interp, _shares_by_risk, int(shares))
            else:
                log.info("  %d shares @ $%.3f — cost $%.2f [%s, bal=$%.2f]",
                         int(shares), ask_price, actual_cost, sizing_mode, usdc or 0)

            # ── Balance check against real order cost ──────────────────────
            if usdc is not None:
                required = actual_cost * 1.05  # 5% buffer for fees/slippage
                if usdc < required:
                    log.error("  Insufficient balance $%.2f < required $%.2f — skipping",
                              usdc, required)
                    continue

            # ── Place order OR simulate (dry run) ──────────────────────────
            if DRY_RUN:
                # Simulate fill at ask_price (as if taker order filled instantly)
                dry_cost = round(shares * ask_price, 4)
                _dry_virtual_balance -= dry_cost
                log.info("  [DRY] BUY %s %.0f shares @ $%.3f — cost $%.2f | virtual_bal=$%.2f | edge_ask=%.1f%% edge_mid=%.1f%%",
                         direction, shares, ask_price, dry_cost, _dry_virtual_balance,
                         edge_vs_ask * 100, (edge_vs_mid or 0) * 100)
                trades.append({
                    "slot_ts":     slot_ts,
                    "direction":   direction,
                    "confidence":  round(confidence, 4),
                    "entry_price": round(ask_price, 4),
                    "shares":      shares,
                    "actual_cost": dry_cost,
                    "stake_usdc":  STAKE_USDC,
                    "token_id":    token_id,
                    "order_id":    f"DRY-{slot_ts}",
                    "status":      "dry_open",
                    "entered_at":  now,
                    "true_edge":   round(true_edge, 4),
                    "dry":         True,
                })
                save_trades(trades)
                continue

            # ── Place limit order at ask + 1 tick (taker-aggressive) ──────────
            # Polymarket tick size is 0.01. Posting exactly at the ask often races
            # against the market maker pulling the offer — we end up unfilled and
            # then the market moves away. Adding 1 tick makes us the best bid
            # (crossing the spread slightly) and dramatically improves fill rate.
            # We validated edge at ask_price already; the 1-tick premium (max $0.05
            # on a 5-share order) is well within the 7%+ edge we require.
            log.info("  Placing LIMIT BUY %s — %.2f shares @ $%.3f (+1 tick from ask $%.3f) | edge_ask=%.1f%% edge_mid=%.1f%%",
                     direction, shares, order_price, ask_price,
                     edge_vs_ask * 100, (edge_vs_mid or 0) * 100)
            try:
                result = client.place_limit_order(
                    token_id=token_id, side="BUY", price=order_price, size=shares
                )
                order_id = getattr(result, "order_id", None) or str(result)
                log.info("  Order placed: %s", str(order_id)[:20])

                # Wait up to 20s for fill with progressive polling (fast start, slower later)
                # Pattern: 1s, 2s, 3s, 4s, 5s, 5s = 20s total (vs old 5s x 6 = 30s)
                filled = False
                poll_errors = 0
                poll_delays = [1, 2, 3, 4, 5, 5]
                for delay in poll_delays:
                    time.sleep(delay)
                    try:
                        o = client.get_order(order_id=str(order_id))
                        status = str(getattr(o, "status", "")).upper()
                        matched = float(getattr(o, "size_matched", 0) or 0)
                        original = float(getattr(o, "original_size", shares) or shares)
                        if status in ("FILLED", "MATCHED") or matched >= original * 0.99:
                            filled = True
                            log.info("  Order FILLED: %s shares=%.2f cost=$%.2f",
                                     str(order_id)[:20], shares, actual_cost)
                            break
                        poll_errors = 0  # reset on success
                    except Exception as e:
                        poll_errors += 1
                        err_str = str(e)
                        # "did not match expected shape" usually means order was instantly filled
                        # and the endpoint returned a closed-order object instead of OpenOrder
                        if "expected shape" in err_str or "did not match" in err_str:
                            log.info("  get_order shape mismatch (likely instant fill) (%d/3)", poll_errors)
                        else:
                            log.warning("  get_order poll failed (%d/3): %s", poll_errors, e)
                        if poll_errors >= 3:
                            # Cannot confirm fill status — do NOT cancel; treat as open
                            if "expected shape" in err_str or "did not match" in err_str:
                                log.info("  get_order shape mismatch 3x — order likely instantly filled, recording as open")
                            else:
                                log.error("  get_order failed 3x — recording as open to avoid cancelling a filled order")
                            filled = True  # conservatively assume filled
                            break

                if not filled:
                    log.info("  Order unfilled after 20s — cancelling")
                    cancel_ok = False
                    try:
                        client.cancel_order(order_id=str(order_id))
                        cancel_ok = True
                    except Exception as e:
                        log.warning("  Cancel failed: %s — checking for partial fill", e)

                    # Check one more time for any partial fill before giving up
                    partial_shares = 0.0
                    try:
                        o = client.get_order(order_id=str(order_id))
                        partial_shares = float(getattr(o, "size_matched", 0) or 0)
                    except Exception:
                        pass

                    if partial_shares > 0:
                        # Partial fill — record as open position
                        partial_cost = round(partial_shares * ask_price, 4)
                        log.info("  Partial fill detected: %.2f shares @ $%.3f = $%.2f",
                                 partial_shares, ask_price, partial_cost)
                        trades.append({"slot_ts": slot_ts, "direction": direction,
                                       "confidence": round(confidence, 4),
                                       "entry_price": round(ask_price, 4),
                                       "shares": partial_shares,
                                       "actual_cost": partial_cost,
                                       "stake_usdc": STAKE_USDC,
                                       "token_id": token_id,
                                       "order_id": str(order_id)[:40],
                                       "status": "open",
                                       "entered_at": now,
                                       "true_edge": round(true_edge, 4),
                                       "note": "partial_fill"})
                    elif not cancel_ok:
                        # Cancel failed and no partial — ghost order, must track
                        log.error("  Ghost order %s — cancel failed and no partial fill. Manual reconciliation needed.",
                                  str(order_id)[:20])
                        trades.append({"slot_ts": slot_ts, "direction": direction,
                                       "confidence": round(confidence, 4),
                                       "entry_price": round(ask_price, 4),
                                       "shares": shares,
                                       "actual_cost": actual_cost,
                                       "token_id": token_id,
                                       "order_id": str(order_id)[:40],
                                       "status": "error_cancel_failed",
                                       "entered_at": now})
                    else:
                        trades.append({"slot_ts": slot_ts, "direction": direction,
                                       "confidence": round(confidence, 4),
                                       "entry_price": round(ask_price, 4), "status": "skipped",
                                       "reason": "limit order unfilled in 20s — cancelled",
                                       "entered_at": now})
                    save_trades(trades)
                    continue
                trades.append({
                    "slot_ts":     slot_ts,
                    "direction":   direction,
                    "confidence":  round(confidence, 4),
                    "entry_price": round(ask_price, 4),
                    "shares":      shares,
                    "actual_cost": actual_cost,
                    "stake_usdc":  STAKE_USDC,
                    "token_id":    token_id,
                    "order_id":    str(order_id)[:40],
                    "status":      "open",
                    "entered_at":  now,
                    "true_edge":   round(true_edge, 4),
                })
                save_trades(trades)
            except Exception as e:
                log.error("  Order FAILED: %s", e)
                trades.append({"slot_ts": slot_ts, "direction": direction,
                                "status": "error", "reason": str(e), "entered_at": now})
                save_trades(trades)

        time.sleep(10)  # tight loop — 70s entry window needs quick checks
      except Exception as e:
          log.error("Main loop unhandled exception: %s", e, exc_info=True)
          time.sleep(5)  # brief pause before retrying to avoid tight crash loop


def _print_summary(trades):
    settled = [t for t in trades if t.get("status") == "settled"]
    wins    = [t for t in settled if t.get("result") == "WIN"]
    pnl     = sum(t.get("pnl_usdc", 0) for t in settled)
    wr      = len(wins)/len(settled) if settled else 0
    log.info("── settled=%d W%d/L%d WR=%.0f%% P&L=$%.2f open=%d",
             len(settled), len(wins), len(settled)-len(wins), wr*100, pnl,
             sum(1 for t in trades if t.get("status")=="open"))


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from polymarket import SecureClient
    from polymarket.auth import BuilderApiKey

    # ── Startup diagnostics ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("BTC Live Trader starting up")
    log.info("=" * 60)
    log.info("Python: %s", __import__('sys').version.split()[0])
    log.info("Platform: %s", __import__('platform').platform())

    import importlib.metadata as _meta
    for _pkg in ["websockets", "lightgbm", "scikit-learn", "pandas", "numpy"]:
        try:
            log.info("  %s==%s", _pkg, _meta.version(_pkg))
        except Exception:
            log.warning("  %s: NOT INSTALLED", _pkg)

    # Log config (secrets masked)
    log.info("Config: GAMMA_HOST=%s", GAMMA_HOST)
    log.info("Config: CLOB_URL=%s", CLOB_URL)
    log.info("Config: CLOB_WS_URL=%s", CLOB_WS_URL)
    log.info("Config: BINANCE_WS=%s...", BINANCE_WS[:50])
    log.info("Config: HF_REPO=%s", HF_REPO)
    log.info("Config: PROXY_WALLET=%s", PROXY_WALLET)
    log.info("Config: MIN_CONFIDENCE=%.0f%% MIN_EDGE=%.0f%% MIN_EDGE_MID=%.0f%%",
             MIN_CONFIDENCE * 100, MIN_EDGE * 100, MIN_EDGE_MID * 100)
    log.info("Config: ENTER_WINDOW=%s SLOT_DURATION=%ds OBS_SECS=%ds", ENTER_WINDOW, SLOT_DURATION, OBSERVE_SECS)
    log.info("Config: ASK_RANGE=[0.42, 0.65]")
    if AUTO_SHARES:
        log.info("Config: AUTO_SHARES=ON min=%d max=%d bal_floor=$%.0f bal_ceil=$%.0f",
                 AUTO_SHARES_MIN, AUTO_SHARES_MAX, AUTO_SHARES_BAL_FLOOR, AUTO_SHARES_BAL_CEIL)
    else:
        log.info("Config: AUTO_SHARES=OFF fixed_shares=%d", FIXED_SHARES)
    log.info("=" * 60)

    if DRY_RUN:
        log.info("*** DRY RUN MODE — no real orders will be placed ***")
        log.info("    Virtual balance: $%.2f USDC", DRY_VIRTUAL_BALANCE)
        log.info("=" * 60)

    # Start spot daemon in background
    start_spot_daemon()

    # Load model
    model, features = load_model()

    if DRY_RUN:
        # Dummy client — never used for orders in dry run
        client = None
    else:
        if not PRIVATE_KEY or not PROXY_WALLET:
            raise RuntimeError("DRY_RUN=false but POLY_PRIVATE_KEY / POLY_SAFE_ADDRESS not set")
        # Init client (same pattern as maker_mm.py)
        client = SecureClient.create(
            private_key=PRIVATE_KEY,
            wallet=PROXY_WALLET,
            api_key=BuilderApiKey(
                key=BUILDER_KEY,
                secret=BUILDER_SECRET,
                passphrase=BUILDER_PASS,
            ),
        )
        log.info("Client initialized | wallet=%s", PROXY_WALLET)

    # Start CLOB WS daemon (after client init so fetch_market can run)
    start_clob_daemon()

    run(client, model, features)
