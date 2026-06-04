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

def _spot_daemon_thread():
    """Background thread: WS stream → buffer → /tmp/spot_buffer.json"""
    import websockets

    async def _run():
        _seed_spot_buffers()
        while True:
            try:
                async with websockets.connect(BINANCE_WS, ping_interval=20, ping_timeout=10) as ws:
                    log.info("Spot daemon connected")
                    async for raw in ws:
                        msg = json.loads(raw)
                        sym = msg.get("stream", "").split("@")[0]
                        if sym not in _spot_buffers:
                            continue
                        k = msg.get("data", {}).get("k", {})
                        if not k:
                            continue
                        ts_s = k["t"] // 1000
                        close = float(k["c"])
                        dq = _spot_buffers[sym]
                        if dq and dq[-1][0] == ts_s:
                            dq[-1][1] = close
                        else:
                            dq.append([ts_s, close])
                        _write_spot_buffer()
            except Exception as e:
                log.warning("Spot WS error: %s — reconnecting in 5s", e)
                await asyncio.sleep(5)

    asyncio.run(_run())

def start_spot_daemon():
    t = threading.Thread(target=_spot_daemon_thread, daemon=True, name="spot-daemon")
    t.start()
    log.info("Spot daemon thread started")
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


def _clob_daemon_thread():
    """Background thread: CLOB WS → _clob_prices dict."""
    import websockets

    async def _clob_run():
        while True:
            try:
                # No ping_interval — Polymarket WS server doesn't support WS-level pings
                # and will drop the connection if it receives an unexpected frame.
                async with websockets.connect(
                    CLOB_WS_URL, ping_interval=None, close_timeout=5
                ) as ws:
                    log.info("CLOB WS daemon connected")
                    last_send = time.time()

                    # Clear stale prices on reconnect — book snapshots arrive fresh below
                    with _clob_prices_lock:
                        _clob_prices.clear()
                        _clob_price_ts.clear()

                    # Prune stale tokens — only keep current + next 3 slots
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
                        existing = list(_clob_subscribed)
                    if stale:
                        log.info("CLOB WS pruned %d stale tokens", len(stale))
                    if existing:
                        await ws.send(json.dumps({"type": "Market", "assets_ids": existing}))
                        log.info("CLOB WS re-subscribed %d tokens", len(existing))
                        last_send = time.time()

                    # Drain any queued subscription requests that arrived before connect
                    pending: list[tuple[str,int]] = []
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
                            log.info("CLOB WS subscribed %d new tokens", len(new_tokens))
                            last_send = time.time()

                    while True:
                        # Check for new subscription requests (non-blocking)
                        pending = []
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
                                log.info("CLOB WS subscribed %d new tokens", len(new_tokens))
                                last_send = time.time()

                        # Keepalive: re-subscribe active tokens every 8 min to prevent
                        # server-side idle timeout (~11 min observed on Polymarket WS)
                        if time.time() - last_send >= 480:
                            with _clob_prices_lock:
                                active = list(_clob_subscribed)
                            if active:
                                await ws.send(json.dumps({"type": "Market", "assets_ids": active}))
                                log.info("CLOB WS keepalive — re-subscribed %d tokens", len(active))
                            last_send = time.time()

                        # Wait for next WS message with a short timeout so we can poll queue
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        try:
                            events = json.loads(raw)
                        except Exception:
                            continue

                        # Server may send a list or a single dict
                        if isinstance(events, dict):
                            events = [events]

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
            except Exception as e:
                log.warning("CLOB WS error: %s — reconnecting in 5s", e)
                await asyncio.sleep(5)

    asyncio.run(_clob_run())


def start_clob_daemon():
    """Start the CLOB WS daemon thread and pre-subscribe to current + next slot tokens."""
    t = threading.Thread(target=_clob_daemon_thread, daemon=True, name="clob-daemon")
    t.start()
    log.info("CLOB WS daemon thread started")

    # Pre-subscribe to current + next 3 slots on startup (15 min window, avoids mid-session sends)
    now = int(time.time())
    cur_slot = (now // SLOT_DURATION) * SLOT_DURATION
    for i in range(4):
        slot_ts = cur_slot + i * SLOT_DURATION
        mkt = fetch_market(slot_ts)
        if mkt:
            clob_subscribe([mkt["yes_token"], mkt["no_token"]], slot_ts=slot_ts)
            log.info("CLOB WS pre-subscribed slot=%d", slot_ts)


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
                local_dir_use_symlinks=False,
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
        log.warning("spot_buffer is %ds stale", age)

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
        feat[f"btc_pre_{lbl}_vol"] = 0.0  # vol not used in v18 top features

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
                log.warning("  get_ask_price /book empty asks (attempt %d/3)", attempt + 1)
                time.sleep(0.5 * (attempt + 1))
                continue
            best_asks = [float(a["price"]) for a in asks
                         if float(a.get("price", 1)) < 0.97]
            if best_asks:
                best_ask = min(best_asks)
                log.info("  get_ask_price /book fallback: %.3f (attempt %d)", best_ask, attempt + 1)
                return best_ask
            else:
                log.warning("  get_ask_price /book all asks >= 0.97 (attempt %d/3)", attempt + 1)
                time.sleep(0.5 * (attempt + 1))
                continue
        except Exception as e:
            log.warning("  get_ask_price /book exception (attempt %d/3): %s", attempt + 1, e)
            time.sleep(0.5 * (attempt + 1))

    log.warning("  get_ask_price FAILED all 3 attempts for %s — returning 0.0", token_id[:16])
    return 0.0  # outside [0.38, 0.90] filter — trade will be skipped


def get_market_mid(slot_ts: int) -> tuple[float, float] | tuple[None, None]:
    """
    Get current market mid prices (UP, DOWN) from Gamma outcomePrices.
    More reliable than ask for edge validation — reflects true market consensus.
    Returns (up_mid, down_mid) or (None, None) on failure.
    """
    slug = f"btc-updown-5m-{slot_ts}"
    try:
        r = _http.get(f"{GAMMA_HOST}/markets/slug/{slug}", timeout=HTTP_TIMEOUT)
        if not r.ok:
            return None, None
        m = r.json()
        op = m.get("outcomePrices", "[]")
        if isinstance(op, str):
            op = json.loads(op)
        if op and len(op) >= 2:
            up_mid   = float(op[0])
            down_mid = float(op[1])
            if 0 < up_mid < 1 and 0 < down_mid < 1:
                return up_mid, down_mid
    except Exception:
        pass
    return None, None

def fetch_inslot_trades(yes_token: str, no_token: str, slot_ts: int) -> list[dict]:
    """Fetch inslot trades from data-api. Lag is ~90-120s so t=0-60s trades available at t=180s."""
    all_trades = []
    for token_id, outcome_label in [(yes_token, "Up"), (no_token, "Down")]:
        for page in range(MAX_TRADE_PAGES):
            offset = page * 500
            try:
                r = _http.get(f"{DATA_API}/trades",
                    params={"asset": token_id, "limit": 500, "offset": offset},
                    timeout=HTTP_TIMEOUT)
                if not r.ok:
                    break
                batch = r.json()
                if not batch:
                    break
                # Filter to inslot window [slot_ts, slot_ts+OBS_SECS)
                for t in batch:
                    ts = int(t.get("timestamp", 0))
                    if ts > 1e12:
                        ts //= 1000
                    t_sec = ts - slot_ts
                    if 0 <= t_sec < OBSERVE_SECS:
                        _price = float(t.get("price", 0) or 0)
                        _size  = float(t.get("size", 0) or 0)
                        all_trades.append({
                            "outcome":   t.get("outcome", outcome_label),
                            "side":      t.get("side", "BUY"),
                            "price":     _price,
                            "size":      _size,
                            "size_usdc": _price * _size,  # dollar volume — matches training features.py
                            "t_sec":     t_sec,
                        })
                # NOTE: data-api returns trades in random order, NOT chronological.
                # Do NOT break early based on min_ts — it would skip inslot trades
                # that appear on later pages. Page until empty or max pages.
            except Exception:
                break
    return all_trades


def _build_ob_features(up_token_id: str) -> dict:
    """
    Fetch the current order book from CLOB REST and compute OB features.
    Identical computation to training (pmdata poly_l2 book snapshots).
    Returns 8 features. On failure returns empty dict (model uses 0.0 fallback).

    Features (all computable from CLOB REST /book?token_id=...):
      ob_mid           — (best_ask + best_bid) / 2
      ob_spread        — best_ask - best_bid
      ob_imbalance     — (best_bid_size - best_ask_size) / (bid + ask)
      ob_depth_ratio   — bid_depth_5c / ask_depth_5c
      ob_bid_depth_5c  — bid size within 5c of mid, normalized by total bid
      ob_ask_depth_5c  — ask size within 5c of mid, normalized by total ask
      ob_mid_drift     — 0.0 at live time (only one snapshot available)
      ob_imbalance_end — same as ob_imbalance (single snapshot)
    """
    try:
        r = _http.get(
            f"{CLOB_URL}/book",
            params={"token_id": up_token_id},
            timeout=5,
        )
        if not r.ok:
            return {}
        book = r.json()

        asks = book.get("asks", [])
        bids = book.get("bids", [])
        if not asks or not bids:
            return {}

        # Sort: asks ascending by price, bids descending
        asks_sorted = sorted(asks, key=lambda x: float(x["price"]))
        bids_sorted = sorted(bids, key=lambda x: float(x["price"]), reverse=True)

        best_ask = float(asks_sorted[0]["price"])
        best_bid = float(bids_sorted[0]["price"])
        best_ask_sz = float(asks_sorted[0]["size"])
        best_bid_sz = float(bids_sorted[0]["size"])
        mid    = (best_ask + best_bid) / 2
        spread = best_ask - best_bid

        # Imbalance at best level
        imbalance = float((best_bid_sz - best_ask_sz) / (best_bid_sz + best_ask_sz + 1e-8))

        # Depth within 5c of mid, normalized by total depth
        total_bid = sum(float(b["size"]) for b in bids) + 1e-8
        total_ask = sum(float(a["size"]) for a in asks) + 1e-8
        bid_depth_5c = sum(float(b["size"]) for b in bids
                           if float(b["price"]) >= mid - 0.05) / total_bid
        ask_depth_5c = sum(float(a["size"]) for a in asks
                           if float(a["price"]) <= mid + 0.05) / total_ask
        depth_ratio  = float(bid_depth_5c / (ask_depth_5c + 1e-8))

        return {
            "ob_mid":           float(mid),
            "ob_spread":        float(spread),
            "ob_imbalance":     float(imbalance),
            "ob_depth_ratio":   float(depth_ratio),
            "ob_bid_depth_5c":  float(bid_depth_5c),
            "ob_ask_depth_5c":  float(ask_depth_5c),
            "ob_mid_drift":     0.0,   # single snapshot — no drift measurable live
            "ob_imbalance_end": float(imbalance),  # same snapshot
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
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin":  math.sin(2 * math.pi * dt.weekday() / 7),
        "dow_cos":  math.cos(2 * math.pi * dt.weekday() / 7),
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

        # Time-weighted order flow — MUST match training formula exactly:
        # w = exp(-0.02 * (OBS - t_sec))  → more weight on recent ticks
        # tw_up = weighted_avg(is_up * size_usdc) / weighted_avg(size_usdc)
        # Both time AND volume weighted (training uses size_usdc as secondary weight)
        if n > 0:
            t_secs   = np.array([t["t_sec"] for t in ticks], dtype=np.float64)
            is_up_v  = np.array([1.0 if t.get("outcome") == "Up" else 0.0 for t in ticks])
            vol_v    = np.array([t.get("size_usdc", 0.0) for t in ticks], dtype=np.float64)
            w_exp    = np.exp(-0.02 * (OBS - t_secs))
            denom    = float(np.average(vol_v, weights=w_exp) + 1e-9)
            tw_up    = float(np.average(is_up_v * vol_v, weights=w_exp) / denom)
        else:
            tw_up = 0.5

        # VWAP trend: early half vs late half
        half = OBS / 2
        early = [t for t in ticks if t["t_sec"] < half]
        late  = [t for t in ticks if t["t_sec"] >= half]
        def _vwap_up_half(grp: list[dict]) -> float:
            up = [t for t in grp if t.get("outcome") == "Up"]
            if not up: return 0.5
            return float(sum(t["price"] * t["size_usdc"] for t in up) /
                         (sum(t["size_usdc"] for t in up) + 1e-8))
        vwap_trend = float(_vwap_up_half(late) - _vwap_up_half(early))

        # Volume-weighted momentum across 6 windows
        vol_by_w = np.array([
            sum(t["size_usdc"] for t in ticks if i*30 <= t["t_sec"] < (i+1)*30)
            for i in range(6)
        ], dtype=np.float64)
        ur_by_w  = np.array([sw[f"btc_up_w{i}"] for i in range(6)], dtype=np.float64)
        tw_vol   = vol_by_w.sum() + 1e-8
        vwmom    = float(np.dot(vol_by_w / tw_vol, ur_by_w - 0.5))

        # Tick acceleration: last 30s vs first 30s
        first30 = sum(1 for t in ticks if t["t_sec"] < 30)
        last30  = sum(1 for t in ticks if t["t_sec"] >= OBS - 30)
        tick_accel = float((last30 - first30) / (first30 + 1e-8))

        # ── v9 features ────────────────────────────────────────────────────────
        # Signal consistency across 6 windows
        w_vals_list = [sw[f"btc_up_w{i}"] for i in range(6)]
        up_ratio_stability = float(np.std(w_vals_list))

        # Volume acceleration: last 90s vs first 90s
        vol_first90 = sum(t["size_usdc"] for t in ticks if t["t_sec"] < 90)
        vol_last90  = sum(t["size_usdc"] for t in ticks if t["t_sec"] >= 90)
        vol_accel   = float(vol_last90 / (vol_first90 + 1e-8))

        # Size disparity: avg trade size Up vs Down
        up_tks_v9   = [t for t in ticks if t.get("outcome") == "Up"]
        dn_tks_v9   = [t for t in ticks if t.get("outcome") == "Down"]
        avg_up_sz   = float(sum(t["size_usdc"] for t in up_tks_v9) / (len(up_tks_v9) + 1e-8))
        avg_dn_sz   = float(sum(t["size_usdc"] for t in dn_tks_v9) / (len(dn_tks_v9) + 1e-8))
        size_disparity = float(avg_up_sz - avg_dn_sz)

        feat.update({
            "btc_n_ticks":     float(n),
            "btc_vol_up":      float(vol_up),
            "btc_vol_dn":      float(vol_dn),
            "btc_vol_ratio":   float(vol_up / (vol_dn + 1e-8)),
            "btc_up_ratio":    float(vol_up / total),
            "btc_vwap_up":     float(vwap_up),
            "btc_vwap_dn":     float(vwap_dn),
            "btc_vwap_spread": float(vwap_up - vwap_dn),
            "btc_buy_ratio":   float(sum(t["size_usdc"] for t in ticks if t.get("side") == "BUY") / (total + 1e-8)),
            "btc_avg_size":    float(total / n),
            "btc_momentum":    btc_momentum,
            "btc_tw_up_ratio": tw_up,
            "btc_vwap_trend":  vwap_trend,
            "btc_vwmom":       vwmom,
            "btc_tick_accel":  tick_accel,
            # v9
            "btc_up_ratio_stability": up_ratio_stability,
            "btc_vol_accel":          vol_accel,
            "btc_size_disparity":     size_disparity,
            # v10 interaction features
            "btc_signal_conviction":  float((vol_up / total) * (1.0 - up_ratio_stability)),
            "btc_momentum_vol_sync":  float(btc_momentum * vol_accel),
            **sw,
        })
        cur_up_ratio = float(vol_up / total)
    else:
        # No ticks — neutral fill
        feat.update({
            "btc_n_ticks": 0.0, "btc_vol_up": 0.0, "btc_vol_dn": 0.0,
            "btc_vol_ratio": 1.0, "btc_up_ratio": 0.5,
            "btc_vwap_up": 0.5, "btc_vwap_dn": 0.5, "btc_vwap_spread": 0.0,
            "btc_buy_ratio": 0.5, "btc_avg_size": 0.0, "btc_momentum": 0.0,
            "btc_tw_up_ratio": 0.5, "btc_vwap_trend": 0.0,
            "btc_vwmom": 0.0, "btc_tick_accel": 0.0,
            "btc_up_ratio_stability": 0.0, "btc_vol_accel": 1.0, "btc_size_disparity": 0.0,
            "btc_signal_conviction": 0.0, "btc_momentum_vol_sync": 0.0,
            **{f"btc_up_w{i}": 0.5 for i in range(6)},
        })
        cur_up_ratio = 0.5

    # ── Historical zscore features (from ring buffer) ──────────────────────────
    # _slot_history: list of {"slot_ts", "up_ratio", "target", "sw": [6 floats]}
    # populated by _update_slot_history() after each prediction
    hist = _slot_history  # module-level ring buffer

    # Temporal interaction features (v17): hour modulates CLOB signal
    feat["hour_x_up_ratio"] = cur_up_ratio * (hour / 24.0)
    feat["hour_x_tw_ur"]    = feat.get("btc_tw_up_ratio", 0.5) * (hour / 24.0)

    # Multi-scale up_ratio zscore (5 / 10 / 20 slots lookback)
    for win, lbl in [(5, "5s"), (10, "10s"), (20, "20s")]:
        past = [h["up_ratio"] for h in hist[-win:]] if hist else []
        if len(past) >= 3:
            mu, sd = float(np.mean(past)), float(np.std(past))
            feat[f"btc_up_ratio_zscore_{lbl}"]    = float((cur_up_ratio - mu) / (sd + 1e-8))
            feat[f"btc_up_ratio_hist_mean_{lbl}"] = mu
        else:
            feat[f"btc_up_ratio_zscore_{lbl}"]    = 0.0
            feat[f"btc_up_ratio_hist_mean_{lbl}"] = 0.5

    # Per-sub-window zscore vs last 20 slots — use overall up_ratio stats
    # Training uses mu20/sd20 from up_ratio history for btc_up_w5_zscore
    past_ur_20 = [h["up_ratio"] for h in hist[-20:]] if hist else []
    if len(past_ur_20) >= 3:
        mu20_ur = float(np.mean(past_ur_20))
        sd20_ur = float(np.std(past_ur_20)) + 1e-6
        feat["btc_up_w5_zscore"] = float((feat.get("btc_up_w5", 0.5) - mu20_ur) / sd20_ur)
    else:
        feat["btc_up_w5_zscore"] = 0.0
    # Other window zscores (not in v18 top features, but keep for compatibility)
    cur_sws = [feat.get(f"btc_up_w{i}", 0.5) for i in range(6)]
    for i in range(6):
        if i == 5:
            continue  # already computed above
        past_sw = [h["sw"][i] for h in hist[-20:] if "sw" in h] if hist else []
        if len(past_sw) >= 5:
            mu, sd = float(np.mean(past_sw)), float(np.std(past_sw))
            feat[f"btc_up_w{i}_zscore"] = float((cur_sws[i] - mu) / (sd + 1e-8))
        else:
            feat[f"btc_up_w{i}_zscore"] = 0.0

    # Realized vol (std of pre-slot returns over last 5/10 slots)
    for win, lbl in [(5, "5s"), (10, "10s")]:
        past_rets = [h.get("pre_ret", 0.0) for h in hist[-win:]] if hist else []
        feat[f"btc_realized_vol_{lbl}"] = float(np.std(past_rets)) if len(past_rets) >= 3 else 0.0

    # Lag outcomes (extended to 5 lags for v17+)
    n_hist = len(hist)
    for lag in [1, 2, 3, 4, 5]:
        if n_hist >= lag and hist[-lag].get("target") is not None:
            feat[f"lag_{lag}_outcome"] = float(hist[-lag]["target"])
        else:
            feat[f"lag_{lag}_outcome"] = 0.5

    # prev_slot_up_ratio/n_ticks/vol — continuous lag signals (v16/v17 features)
    for lag in [1, 2, 3, 4, 5]:
        if n_hist >= lag:
            h = hist[-lag]
            feat[f"prev_slot_up_ratio_{lag}"]  = float(h.get("up_ratio", 0.5))
            feat[f"prev_slot_n_ticks_{lag}"]   = float(h.get("n_ticks", 0.0))
            feat[f"prev_slot_vol_{lag}"]        = float(h.get("vol_total", 0.0))
        else:
            feat[f"prev_slot_up_ratio_{lag}"]  = 0.5
            feat[f"prev_slot_n_ticks_{lag}"]   = 0.0
            feat[f"prev_slot_vol_{lag}"]        = 0.0

    # Lag streak
    streak = 0
    if n_hist >= 1 and hist[-1].get("target") is not None:
        last_val = hist[-1]["target"]
        for back in range(1, min(n_hist + 1, 6)):
            v = hist[-back].get("target")
            if v == last_val and v is not None:
                streak += 1
            else:
                break
    feat["lag_streak"] = float(streak)

    # ── OB features: fetch real order book via CLOB REST ──────────────────────
    # Identical computation to training (pmdata poly_l2 book snapshots).
    # Called once per slot at ~t=150s (after tick observation window ends).
    # up_token_id must be passed in; on failure fills with None (market excluded).
    feat.update(_build_ob_features(up_token_id))

    # ── Spot features ──────────────────────────────────────────────────────────
    feat.update(build_spot_features(slot_ts))

    # ── Final: fill any remaining model features with 0 ───────────────────────
    for f in features:
        feat.setdefault(f, 0.0)
    return feat






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
                # Use neutral 0.5 for up_ratio seed — outcomePrices is 0/1 after
                # resolution and completely misrepresents tick-based up_ratio
                # (which ranges 0.2–0.8 during the slot). Starting neutral is
                # better than feeding extreme out-of-distribution values to
                # prev_slot_up_ratio_* features.
                up_ratio = 0.5
                sw = [0.5] * 6

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

    while True:
      try:
        now    = int(time.time())
        trades = load_trades()

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

            feat = build_features(ticks, slot_ts, features,
                                   up_token_id=market["yes_token"])
            if feat is None:
                continue

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
            ask_price  = get_ask_price(token_id)
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

            # Also validate against market mid (outcomePrices) — catches stale ask prices
            up_mid, down_mid = get_market_mid(slot_ts)
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
            # Do this BEFORE balance check so we can validate against real cost.
            shares = round(STAKE_USDC / ask_price, 2)
            actual_cost = round(shares * ask_price, 4)
            if shares < 5.0:
                # CLOB minimum is 5 shares — always bump, never skip.
                # Edge check already validated the trade is worth taking.
                shares = 5.0
                actual_cost = round(shares * ask_price, 4)
                log.info("  Bumped to min 5 shares — cost $%.2f (stake was $%.2f)",
                         actual_cost, STAKE_USDC)

            # ── Balance check against real order cost (not just STAKE_USDC) ──
            try:
                bal  = client.get_balance_allowance(asset_type="COLLATERAL")
                usdc = float(bal.balance) / 1e6
                required = actual_cost * 1.05  # 5% buffer for fees/slippage
                if usdc < required:
                    log.error("  Insufficient balance $%.2f < required $%.2f — skipping",
                              usdc, required)
                    continue
            except Exception as e:
                log.warning("  Balance check FAILED: %s — PROCEEDING WITHOUT CONFIRMATION", e)

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

                # Wait up to 30s for fill, then cancel if unfilled
                filled = False
                poll_errors = 0
                for _ in range(6):
                    time.sleep(5)
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
                    log.info("  Order unfilled after 30s — cancelling")
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
                                       "reason": "limit order unfilled in 30s — cancelled",
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
