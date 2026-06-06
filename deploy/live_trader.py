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
from data_quality_gate import DataQualityGate
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
PRIVATE_KEY    = os.environ["POLY_PRIVATE_KEY"]
PROXY_WALLET   = os.environ["POLY_SAFE_ADDRESS"]
BUILDER_KEY    = os.environ["MM_BUILDER_KEY"]
BUILDER_SECRET = os.environ["MM_BUILDER_SECRET"]
BUILDER_PASS   = os.environ["MM_BUILDER_PASSPHRASE"]

GAMMA_HOST      = "https://gamma-api.polymarket.com"
DATA_API        = "https://data-api.polymarket.com"
CLOB_URL        = "https://clob.polymarket.com"
CLOB_WS_URL     = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
BINANCE_WS      = "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1m/ethusdt@kline_1m/solusdt@kline_1m"
BINANCE_REST    = "https://api.binance.com/api/v3"

HTTP_TIMEOUT    = 3    # seconds — tight budget for a 10s loop
MAX_TRADE_PAGES = 8    # max 8 pages/token × 2 tokens = 16 calls max (~4000 trades/token)
                       # Keeps fetch_inslot_trades well under 30s even on slow days.
                       # NOTE: data-api returns trades in random order (not chronological),
                       # so we must page through all pages to collect inslot trades.

SLOT_DURATION   = 300
OBSERVE_SECS    = 180
ENTER_WINDOW    = (170, 240)
SETTLE_GRACE    = 60
MIN_CONFIDENCE  = 0.60
MIN_EDGE        = 0.10
MIN_EDGE_MID    = 0.05          # edge vs market mid (catches stale ask)
TAKER_FEE       = 0.02          # Polymarket taker fee ~2%
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
    "ethusdt": deque(maxlen=75),   # kept for future use; not used by v8
    "solusdt": deque(maxlen=75),   # kept for future use; not used by v8
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
    for sym in _spot_buffers:
        limit = 300 if sym == "btcusdt" else 75  # BTC needs 4h+ history
        try:
            r = _http.get(f"{BINANCE_REST}/klines",
                             params={"symbol": sym.upper(), "interval": "1m", "limit": limit},
                             timeout=HTTP_TIMEOUT)
            if r.ok:
                for k in r.json():
                    _spot_buffers[sym].append([k[0] // 1000, float(k[4])])
                log.info("Spot seed %s: %d candles", sym.upper(), len(_spot_buffers[sym]))
        except Exception as e:
            log.warning("Spot seed %s failed: %s", sym, e)
    _write_spot_buffer()


async def _spot_on_message(msg: dict) -> None:
    """Handle Binance kline WS message."""
    if isinstance(msg, dict):
        sym = msg.get("stream", "").split("@")[0]
        if sym not in _spot_buffers:
            return
        k = msg.get("data", {}).get("k", {})
        if not k:
            return
        ts_s = k["t"] // 1000
        close = float(k["c"])
        dq = _spot_buffers[sym]
        if dq and dq[-1][0] == ts_s:
            dq[-1][1] = close
        else:
            dq.append([ts_s, close])
        _write_spot_buffer()


async def _spot_on_connect(ws) -> None:
    """Seed buffers on first connect (runs in the WS thread's event loop)."""
    # Seeding uses REST, safe to call from async context
    pass


def _spot_rest_poll():
    """Fallback: poll Binance REST API for latest klines when WS is blocked.

    Runs in a background thread, polls every 30s. This handles the case where
    Binance WS returns HTTP 451 (geo-blocked) from certain cloud regions.
    """
    while True:
        try:
            for sym in _spot_buffers:
                limit = 5  # just the latest candles, seed already loaded history
                r = _http.get(f"{BINANCE_REST}/klines",
                              params={"symbol": sym.upper(), "interval": "1m", "limit": limit},
                              timeout=HTTP_TIMEOUT + 2)
                if r.ok:
                    for k in r.json():
                        ts_s = k[0] // 1000
                        close = float(k[4])
                        dq = _spot_buffers[sym]
                        if dq and dq[-1][0] == ts_s:
                            dq[-1][1] = close
                        elif not dq or ts_s > dq[-1][0]:
                            dq.append([ts_s, close])
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
        name="binance-spot",
        url=BINANCE_WS,
        on_message=_spot_on_message,
        config=WSConfig(
            ping_interval=20.0,
            ping_timeout=10.0,
            zombie_timeout=60.0,       # Binance sends klines every ~2s
            health_log_interval=300.0,  # log health every 5 min
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
    """Handle CLOB WS message — updates _clob_prices cache."""
    global _clob_last_keepalive

    # Server may send a list or a single dict
    events = [msg] if isinstance(msg, dict) else msg
    if not isinstance(events, list):
        return

    for ev in events:
        etype = ev.get("event_type")
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
                if change.get("side") == "ASK":
                    asset_id = change.get("asset_id", "")
                    price = float(change.get("price", 1))
                    with _clob_prices_lock:
                        # price_change fires on ANY book level change,
                        # not just top-of-book. Two cases:
                        # (a) price <= existing: tighter ask or same level — update.
                        # (b) price > existing: could mean best offer was pulled
                        #     and book worsened. Invalidate cache so HTTP
                        #     fallback is used until next book snapshot.
                        existing = _clob_prices.get(asset_id, 1.0)
                        if price < 0.97 and price <= existing:
                            _clob_prices[asset_id] = price
                            _clob_price_ts[asset_id] = time.time()
                        elif price > existing:
                            # Best ask rose — invalidate to force HTTP fallback
                            _clob_prices.pop(asset_id, None)
                            _clob_price_ts.pop(asset_id, None)

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
            zombie_timeout=120.0,      # Raised from 60s: one-sided markets can go minutes without data msgs. Active ping/pong probe in ws_manager prevents true zombies.
            health_log_interval=300.0,  # log health every 5 min
        ),
    )
    _clob_ws_manager.start()
    log.info("CLOB WS daemon started (ws_manager)")

    # Pre-subscribe to current + next 3 slots on startup (15 min window, avoids mid-session sends)
    now = int(time.time())
    cur_slot = (now // SLOT_DURATION) * SLOT_DURATION
    for i in range(4):
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
    """Download champion.pkl from HuggingFace if not cached, then load it."""
    if not MODEL_PATH.exists():
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
        "btc_dist_1k": 0.5, "btc_dist_5k": 0.5, "btc_dist_10k": 0.5,
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

    # Inslot: [slot_ts, slot_ts+OBSERVE_SECS)
    seg_inslot = _seg(slot_ts, slot_ts + OBSERVE_SECS)
    feat["btc_inslot_ret"], feat["btc_inslot_vol"] = _ret_vol(seg_inslot)

    # Pre-slot windows — MUST use px at observation end (slot_ts + OBS_SECS)
    # to match training: spot_at(obs_end_ts) vs spot_at(slot_ts - window)
    obs_end_ts = slot_ts + OBSERVE_SECS
    idx_obs = np.searchsorted(ts_arr, obs_end_ts, side="right") - 1
    px_obs_end = float(px_arr[idx_obs]) if idx_obs >= 0 else 0.0

    for w_s, lbl in [(300, "5m"), (900, "15m"), (1800, "30m"), (3600, "1h"), (14400, "4h")]:
        idx_h = np.searchsorted(ts_arr, slot_ts - w_s, side="right") - 1
        px_h = float(px_arr[idx_h]) if idx_h >= 0 else 0.0
        if px_h > 0 and px_obs_end > 0:
            feat[f"btc_pre_{lbl}_ret"] = float(px_obs_end / px_h - 1)
        else:
            feat[f"btc_pre_{lbl}_ret"] = 0.0
        # Pre-window volatility: stddev/mean of prices in the window
        seg_pre = _seg(slot_ts - w_s, slot_ts)
        _, pre_vol = _ret_vol(seg_pre)
        feat[f"btc_pre_{lbl}_vol"] = pre_vol

    # Round-number proximity (use obs-end price to match training)
    if px_obs_end > 0:
        px_k = px_obs_end / 1000
        feat["btc_dist_1k"]  = float(min(px_k - math.floor(px_k), math.ceil(px_k) - px_k))
        feat["btc_dist_5k"]  = float(abs(px_obs_end % 5000) / 5000)
        feat["btc_dist_10k"] = float(abs(px_obs_end % 10000) / 10000)
    else:
        feat["btc_dist_1k"] = feat["btc_dist_5k"] = feat["btc_dist_10k"] = 0.5

    # 1h/4h ratio — (px_now - px_1h_ago) / (px_now - px_4h_ago)
    # MUST use px at observation end (slot_ts + OBS_SECS) to match training
    px_now = px_obs_end  # use observation-end price, not slot-start
    idx_1h = np.searchsorted(ts_arr, slot_ts - 3600, side="right") - 1
    idx_4h = np.searchsorted(ts_arr, slot_ts - 14400, side="right") - 1
    px_1h_ago = float(px_arr[idx_1h]) if idx_1h >= 0 else 0.0
    px_4h_ago = float(px_arr[idx_4h]) if idx_4h >= 0 else 0.0
    if px_now > 0 and px_1h_ago > 0 and px_4h_ago > 0 and abs(px_now - px_4h_ago) > 1:
        feat["btc_pre_1h_4h_ratio"] = (px_now - px_1h_ago) / (px_now - px_4h_ago + 1e-9)
    else:
        feat["btc_pre_1h_4h_ratio"] = 0.0  # cold buffer or no meaningful 4h move

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
    Get current market mid prices (UP, DOWN) from the CLOB order book.

    BUG FIX (2026-06-04): Previously used Gamma outcomePrices, but those start
    at ~0.50/0.50 for every new BTC 5-min market and rarely update during the
    active slot.  Real book midpoints diverge wildly (e.g. book UP mid=0.12 while
    Gamma says 0.50), causing the ask-vs-mid divergence check to block almost
    every trade.  Now we compute mid = (best_bid + best_ask) / 2 directly from
    the CLOB /book endpoint, which is always live.

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
        mids = []
        for tid in [up_token, dn_token]:
            try:
                rb = _http.get(f"{CLOB_URL}/book",
                               params={"token_id": tid}, timeout=HTTP_TIMEOUT + 2)
                if not rb.ok:
                    mids.append(None)
                    continue
                book = rb.json()
                asks = [float(a["price"]) for a in book.get("asks", [])
                        if 0 < float(a.get("price", 0)) < 0.99]
                bids = [float(b["price"]) for b in book.get("bids", [])
                        if 0 < float(b.get("price", 0)) < 0.99]
                if asks and bids:
                    mids.append((min(asks) + max(bids)) / 2.0)
                elif asks:
                    mids.append(min(asks))       # no bids, use best ask as proxy
                elif bids:
                    mids.append(max(bids))       # no asks, use best bid as proxy
                else:
                    mids.append(None)
            except Exception:
                mids.append(None)

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
    """Fetch raw OB metrics from CLOB REST. Returns dict or None on failure."""
    try:
        r = _http.get(
            f"{CLOB_URL}/book",
            params={"token_id": up_token_id},
            timeout=5,
        )
        if not r.ok:
            return None
        book = r.json()
        asks = book.get("asks", [])
        bids = book.get("bids", [])
        if not asks or not bids:
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
        return {
            "asks_sorted": asks_sorted, "bids_sorted": bids_sorted,
            "asks": asks, "bids": bids,
            "best_ask": best_ask, "best_bid": best_bid,
            "best_ask_sz": best_ask_sz, "best_bid_sz": best_bid_sz,
            "mid": mid, "spread": spread, "imbalance": imbalance,
            "ts": time.time(),
        }
    except Exception as e:
        log.warning("OB snapshot fetch failed for %s: %s", up_token_id[:20], e)
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

    Returns dict with features. On failure returns empty dict (model uses 0.0 fallback).
    """
    snap = _fetch_ob_snapshot(up_token_id)
    if snap is None:
        return {}

    # ── Cache open snapshot (first poll of entry window) ──────────────
    if up_token_id not in _ob_open_cache:
        _ob_open_cache[up_token_id] = {
            "mid": snap["mid"],
            "imbalance": snap["imbalance"],
            "total_depth": float(sum(float(b["size"]) for b in snap["bids"]) +
                                 sum(float(a["size"]) for a in snap["asks"])),
            "ts": snap["ts"],
        }

    open_snap = _ob_open_cache[up_token_id]

    # ── Temporal features from open vs close ──────────────────────────
    ob_mid_drift   = float(snap["mid"] - open_snap["mid"])
    ob_imb_momentum = float(snap["imbalance"] - open_snap["imbalance"])

    # Windowed imbalance: split the time range into 3 windows
    # open=w0, interpolated midpoint=w1, close=w2
    ob_imb_w0 = float(open_snap["imbalance"])
    ob_imb_w2 = float(snap["imbalance"])
    ob_imb_w1 = float((ob_imb_w0 + ob_imb_w2) / 2.0)  # interpolated middle

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
            "ob_mid":           float(mid),
            "ob_spread":        float(spread),
            "ob_imbalance":     float(imbalance),
            "ob_depth_ratio":   float(depth_ratio),
            "ob_bid_depth_5c":  float(bid_depth_5c),
            "ob_ask_depth_5c":  float(ask_depth_5c),
            "ob_total_depth":   float(total_depth),
            "ob_weighted_imb":  float(weighted_imb),
            # Temporal features — real values from open vs close snapshots
            "ob_mid_drift":     ob_mid_drift,
            "ob_imbalance_end": float(imbalance),
            "ob_spread_end":    float(spread),
            "ob_depth_change":  float(total_depth - open_snap.get("total_depth", total_depth)),
            "ob_imb_momentum":  ob_imb_momentum,
            # Windowed imbalance — open/mid/close
            "ob_imb_w0":        ob_imb_w0,
            "ob_imb_w1":        ob_imb_w1,
            "ob_imb_w2":        ob_imb_w2,
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
        vwap_up = sum(t["price"] * t["size_usdc"] for t in up_tks) / (vol_up + 1e-8) if up_tks else 0.5
        vwap_dn = sum(t["price"] * t["size_usdc"] for t in dn_tks) / (vol_dn + 1e-8) if dn_tks else 0.5

        def _ur_w(subset: list[dict]) -> float:
            vu = sum(t["size_usdc"] for t in subset if t.get("outcome") == "Up")
            tt = sum(t["size_usdc"] for t in subset) + 1e-8
            return float(vu / tt)

        # 6x30s sub-windows
        sw: dict = {}
        for i in range(6):
            t0_w, t1_w = i * 30, (i + 1) * 30
            sub = [t for t in ticks if t0_w <= t["t_sec"] < t1_w]
            sw[f"btc_up_w{i}"] = _ur_w(sub) if sub else 0.5

        # Momentum: mean(last 3 windows) - mean(first 3 windows)
        w_vals = [sw[f"btc_up_w{i}"] for i in range(6)]
        btc_momentum = float(np.mean(w_vals[3:]) - np.mean(w_vals[:3]))

        # ── v9 features ────────────────────────────────────────────────────────
        # Signal consistency across 6 windows (needed by btc_signal_conviction)
        w_vals_list = [sw[f"btc_up_w{i}"] for i in range(6)]
        up_ratio_stability = float(np.std(w_vals_list))

        # Size disparity: avg trade size Up vs Down
        up_tks_v9   = [t for t in ticks if t.get("outcome") == "Up"]
        dn_tks_v9   = [t for t in ticks if t.get("outcome") == "Down"]
        avg_up_sz   = float(sum(t["size_usdc"] for t in up_tks_v9) / (len(up_tks_v9) + 1e-8))
        avg_dn_sz   = float(sum(t["size_usdc"] for t in dn_tks_v9) / (len(dn_tks_v9) + 1e-8))
        size_disparity = float(avg_up_sz - avg_dn_sz)

        feat.update({
            "btc_up_ratio":    float(vol_up / total),
            "btc_n_ticks":     float(n),
            "btc_vol_up":      float(vol_up),
            "btc_vol_dn":      float(vol_dn),
            "btc_vwap_up":     float(vwap_up),
            "btc_vwap_dn":     float(vwap_dn),
            "btc_vwap_spread": float(vwap_up - vwap_dn),
            "btc_buy_ratio":   float(sum(t["size_usdc"] for t in ticks if t.get("side") == "BUY") / (total + 1e-8)),
            "btc_momentum":    btc_momentum,
            # v9
            "btc_size_disparity":     size_disparity,
            # v10 interaction features
            "btc_signal_conviction":  float((vol_up / total) * (1.0 - up_ratio_stability)),
            **sw,
        })
        cur_up_ratio = float(vol_up / total)
    else:
        # No ticks — neutral fill (only v21 features)
        feat.update({
            "btc_up_ratio": 0.5,
            "btc_n_ticks": 0.0,
            "btc_vol_up": 0.0, "btc_vol_dn": 0.0,
            "btc_vwap_up": 0.5, "btc_vwap_dn": 0.5, "btc_vwap_spread": 0.0,
            "btc_buy_ratio": 0.5, "btc_momentum": 0.0,
            "btc_size_disparity": 0.0,
            "btc_signal_conviction": 0.0,
            **{f"btc_up_w{i}": 0.5 for i in range(6)},
        })
        cur_up_ratio = 0.5

    # ── Historical zscore features (from ring buffer) ──────────────────────────
    # _slot_history: list of {"slot_ts", "up_ratio", "target", "sw": [6 floats]}
    # populated by _update_slot_history() after each prediction
    hist = _slot_history  # module-level ring buffer

    # Temporal interaction features (v17): hour modulates CLOB signal
    feat["hour_x_up_ratio"] = cur_up_ratio * (hour / 24.0)
    # hour_cos: cyclical hour encoding (cos component)
    feat["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
    # time-weighted up ratio (recency bias)
    if ticks:
        weights = np.array([t["t_sec"] + 1 for t in ticks], dtype=np.float64)
        weights /= weights.sum() + 1e-8
        tw_vals = np.array([1.0 if t.get("outcome") == "Up" else 0.0 for t in ticks])
        tw_up_ratio = float(np.dot(weights, tw_vals))
    else:
        tw_up_ratio = 0.5
    feat["btc_tw_up_ratio"] = tw_up_ratio
    feat["hour_x_tw_ur"] = tw_up_ratio * (hour / 24.0)

    # prev_slot_up_ratio — continuous lag signals (v21 uses 1,2,3,4,5)
    n_hist = len(hist)
    for lag in [1, 2, 3, 4, 5]:
        if n_hist >= lag:
            h = hist[-lag]
            feat[f"prev_slot_up_ratio_{lag}"]  = float(h.get("up_ratio", 0.5))
        else:
            feat[f"prev_slot_up_ratio_{lag}"]  = 0.5

    # ── Multi-scale up_ratio zscore (5/20 slots) ─────────────────────────────
    # Matches training: zscore of current up_ratio vs recent history mean/std.
    def _hist_ur(n: int) -> list[float]:
        return [h["up_ratio"] for h in hist[-n:]] if hist else []

    hist_vals_20 = _hist_ur(20)
    if len(hist_vals_20) >= 3:
        mu20 = float(np.mean(hist_vals_20))
        sd20 = float(np.std(hist_vals_20)) + 1e-6
        feat["btc_up_ratio_zscore_20s"] = (cur_up_ratio - mu20) / sd20
    else:
        feat["btc_up_ratio_zscore_20s"] = 0.0

    hist_vals_5 = _hist_ur(5)
    if len(hist_vals_5) >= 2:
        mu5 = float(np.mean(hist_vals_5))
        sd5 = float(np.std(hist_vals_5)) + 1e-6
        feat["btc_up_ratio_zscore_5s"] = (cur_up_ratio - mu5) / sd5
    else:
        feat["btc_up_ratio_zscore_5s"] = 0.0

    # ── OB features: fetch real order book via CLOB REST ──────────────────────
    # Identical computation to training (pmdata poly_l2 book snapshots).
    # Called once per slot at ~t=150s (after tick observation window ends).
    # up_token_id must be passed in; on failure fills with None (market excluded).
    feat.update(_build_ob_features(up_token_id))

    # ── Cross-domain interaction features (OB x CLOB) ─────────────────────────
    # v21 uses: x_imb_x_ur, x_depth_x_momentum, x_ob_drift_x_inslot
    feat["x_imb_x_ur"]          = feat.get("ob_imbalance", 0.0) * feat.get("btc_up_ratio", 0.5)
    feat["x_depth_x_momentum"]  = feat.get("ob_depth_ratio", 1.0) * feat.get("btc_momentum", 0.0)
    feat["x_ob_drift_x_inslot"] = feat.get("ob_mid_drift", 0.0) * feat.get("btc_inslot_ret", 0.0)

    # ── Spot features ──────────────────────────────────────────────────────────
    feat.update(build_spot_features(slot_ts))

    # ── Final: fill any remaining model features with context-aware defaults ────
    # ob_mid should be ~0.5 (binary market midpoint), not 0.0
    # ob_ask_depth_5c should be ~0.5 (neutral), not 0.0
    _neutral_defaults = {
        "ob_mid": 0.5,
        "ob_ask_depth_5c": 0.5,
        "ob_bid_depth_5c": 0.5,
        "ob_depth_ratio": 1.0,
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
    now     = int(time.time())
    updated = False
    for trade in trades:
        if trade.get("status") != "open":
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
        log.info("SETTLED slot=%d | %s→%s | %s | P&L $%.2f",
                 trade["slot_ts"], direction, actual, result, pnl)
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
    gate = DataQualityGate(features_list=features, warmup_slots=3)
    log.info("Live trader started | stake=$%.2f | min_conf=%.0f%% | min_edge=%.0f%%",
             STAKE_USDC, MIN_CONFIDENCE*100, MIN_EDGE*100)
    log.info("DataQualityGate active — warming up for %d slots", gate.warmup_slots)

    # Print balance
    try:
        bal  = client.get_balance_allowance(asset_type="COLLATERAL")
        usdc = float(bal.balance) / 1e6
        log.info("Wallet balance: $%.2f USDC", usdc)
    except Exception as e:
        log.warning("Balance check failed: %s", e)

    # Rebuild trade history from on-chain activity (survives restarts)
    if not TRADES_FILE.exists():
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
                    log.info("SPOT-BUFFER age=%ds btc_candles=%d", buf_age, btc_len)
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

        # 2. Enter — only block re-entry for open/settled/error, NOT skipped
        already = {t["slot_ts"] for t in trades
                   if t.get("status") in ("open", "settled", "error")}
        cur_slot = (now // SLOT_DURATION) * SLOT_DURATION

        # Clear OB open snapshot cache when slot changes (new observation window)
        global _ob_last_slot
        if cur_slot != _ob_last_slot:
            _ob_open_cache.clear()
            _ob_last_slot = cur_slot

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

            X = pd.DataFrame([[feat.get(f, 0.0) for f in features]], columns=features)
            prob_up = predict_proba(model, X)
            direction  = "UP" if prob_up >= 0.5 else "DOWN"
            confidence = prob_up if direction == "UP" else 1.0 - prob_up
            log.info("  Prediction: %s  conf=%.1f%%", direction, confidence*100)

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
            if not (0.38 <= ask_price <= 0.90):
                log.info("  Skip — ask $%.3f outside [0.38, 0.90] — market already one-sided", ask_price)
                trades.append({"slot_ts": slot_ts, "direction": direction,
                                "confidence": round(confidence, 4),
                                "entry_price": round(ask_price, 4), "status": "skipped",
                                "reason": f"ask ${ask_price:.3f} outside valid range [0.38, 0.90]",
                                "entered_at": now})
                save_trades(trades)
                continue
            edge_vs_ask = model_prob - ask_price

            # Market mid already fetched in parallel above
            market_mid  = up_mid if direction == "UP" else down_mid
            edge_vs_mid = (model_prob - market_mid) if market_mid else None

            log.info("  BUY %s | ask=$%.3f [%s] edge_ask=%.1f%% | market_mid=$%.3f edge_mid=%.1f%%",
                     direction, ask_price, ask_src, edge_vs_ask * 100,
                     market_mid or 0, (edge_vs_mid or 0) * 100)

            # Sanity check: ask must not diverge >0.20 from market mid.
            # A stale/deep-book ask can be e.g. $0.87 while mid is $0.50 — that
            # would pass the edge_ask check (model_prob=0.97, edge=0.10) but the
            # trade is a guaranteed overpay. Real best-ask is always within ~0.10
            # of mid in an active Polymarket binary.
            if market_mid is not None and abs(ask_price - market_mid) > 0.20:
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

            actual_cost = round(shares * ask_price, 4)
            sizing_mode = "auto" if AUTO_SHARES else "fixed"
            log.info("  %d shares @ $%.3f — cost $%.2f [%s, bal=$%.2f]",
                     int(shares), ask_price, actual_cost, sizing_mode, usdc or 0)

            # ── Balance check against real order cost ──────────────────────
            if usdc is not None:
                required = actual_cost * 1.05  # 5% buffer for fees/slippage
                if usdc < required:
                    log.error("  Insufficient balance $%.2f < required $%.2f — skipping",
                              usdc, required)
                    continue

            # ── Place limit order at the validated ask price ──────────────────
            # Use limit (not market) to guarantee we pay the price we validated.
            # Market orders can sweep stale/extreme prices in the book.
            log.info("  Placing LIMIT BUY %s — %.2f shares @ $%.3f | edge_ask=%.1f%% edge_mid=%.1f%%",
                     direction, shares, ask_price,
                     edge_vs_ask * 100, (edge_vs_mid or 0) * 100)
            try:
                result = client.place_limit_order(
                    token_id=token_id, side="BUY", price=ask_price, size=shares
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
                        log.warning("  get_order poll failed (%d/3): %s", poll_errors, e)
                        if poll_errors >= 3:
                            # Cannot confirm fill status — do NOT cancel; treat as open
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
    log.info("Config: ENTER_WINDOW=%s SLOT_DURATION=%ds", ENTER_WINDOW, SLOT_DURATION)
    if AUTO_SHARES:
        log.info("Config: AUTO_SHARES=ON min=%d max=%d bal_floor=$%.0f bal_ceil=$%.0f",
                 AUTO_SHARES_MIN, AUTO_SHARES_MAX, AUTO_SHARES_BAL_FLOOR, AUTO_SHARES_BAL_CEIL)
    else:
        log.info("Config: AUTO_SHARES=OFF fixed_shares=%d", FIXED_SHARES)
    log.info("=" * 60)

    # Start spot daemon in background
    start_spot_daemon()

    # Load model
    model, features = load_model()

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
