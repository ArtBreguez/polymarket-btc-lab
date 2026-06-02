"""
BTC 5m Directional Paper Trader — v2 (fixed tick fetch)
=========================================================
Strategy:
  - Each slot is 5 minutes: btc-updown-5m-{unix_timestamp}
  - At t=180s (after observing 3 min of order flow): predict UP or DOWN
  - If model confidence > 60% AND edge >= 10%: record paper bet ($5 USDC)
  - After slot closes + 30s grace: settle (check resolution, compute P&L)

Model: btc_model_v2_research.pkl
  - 73 features: order flow (3 windows) + spot BTC/ETH/SOL + time
  - Trained on 7 cryptos x 616 slots = 3,900 samples
  - WF AUC: 0.853, WF Acc: 77.0%

Fix (2026-06-01):
  - data-api ?asset= filter is silently ignored — returns the global trade stream.
  - Correct approach: filter client-side by trade["asset"] matching yes/no token,
    AND use prices-history from clob for UP-token price series (price_first/last/trend/vol).
  - Also fixed: pagination break condition (was stopping too early).

Spot data: reads from /tmp/spot_buffer.json written by spot_daemon.py (WebSocket).
  Zero network calls for spot features — just a file read.

Run:  uv run python scripts/paper_trader.py
      (designed to be called every ~1min via cron, idempotent)
      Requires spot_daemon.py to be running as a background process.

Records:  artifacts/paper_trades.json
"""

import json
import logging
import math
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
ARTIFACTS       = Path("/home/ubuntu/polymarket-btc-lab/artifacts")
TRADES_FILE     = ARTIFACTS / "paper_trades.json"
MODEL_PATH      = ARTIFACTS / "btc_model_v3b_spot.pkl"
GAMMA_HOST      = "https://gamma-api.polymarket.com"
CLOB_HOST       = "https://clob.polymarket.com"
DATA_API        = "https://data-api.polymarket.com"
SPOT_BUFFER     = Path("/tmp/spot_buffer.json")
SLOT_DURATION   = 300           # 5 minutes
OBSERVE_SECS    = 180           # enter after first 3 min
ENTER_WINDOW    = (182, 240)    # t-seconds where we allow entry (must be > OBSERVE_SECS=180)
SETTLE_GRACE    = 60            # settle this many seconds after slot end
MIN_CONFIDENCE  = 0.60          # only bet if model says > 60%
MIN_EDGE        = 0.10          # require at least 10% edge over market price
STAKE_USDC      = 5.0           # flat stake per trade (paper only)
BUFFER_STALE_SECS = 120         # warn if buffer older than this
# Note: data-api has ~4-5 min lag, so order flow features are NOT available at t=180s.
# v3 model uses only real-time features: price_history (CLOB) + spot (Binance WS) + time.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("paper_trader_v2")

# ── Load model ─────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
model    = bundle["model"]
FEATURES = bundle["features"]
log.info("Model v2 loaded: %d features, window=%s, WF AUC=%.3f",
         len(FEATURES), bundle.get("window", "?"), bundle.get("wf_auc", 0))


# ── Spot features from local WS buffer ────────────────────────────────────────
def _window_feats(prices: list[float], label: str, wname: str) -> dict:
    """Compute ret/vol/mom for a slice of prices."""
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
    """
    Build spot BTC/ETH/SOL features from the WS buffer written by spot_daemon.py.
    All computed from buffer — zero network calls.
    """
    feat: dict = {}
    label_map = {"btcusdt": "btc", "ethusdt": "eth", "solusdt": "sol"}

    if not SPOT_BUFFER.exists():
        log.warning("spot_buffer.json not found — is spot_daemon.py running?")
        for label in label_map.values():
            for wname in ["inslot_3m", "pre_3m", "pre_15m", "pre_1h"]:
                feat.update({f"{label}_{wname}_ret": 0.0,
                              f"{label}_{wname}_vol": 0.0,
                              f"{label}_{wname}_mom": 0.0})
            feat[f"{label}_pct_of_1h_range"] = 0.5
        return feat

    try:
        buf = json.loads(SPOT_BUFFER.read_text())
    except Exception as e:
        log.warning("Failed to read spot buffer: %s", e)
        return feat

    age = int(time.time()) - buf.get("updated_at", 0)
    if age > BUFFER_STALE_SECS:
        log.warning("spot buffer is %ds old — daemon may be down", age)

    for sym, label in label_map.items():
        candles = buf.get(sym, [])  # list of [ts_s, close]
        ts_arr = np.array([c[0] for c in candles], dtype=np.int64)
        px_arr = np.array([c[1] for c in candles], dtype=np.float64)

        def slice_px(lo: int, hi: int) -> list[float]:
            mask = (ts_arr >= lo) & (ts_arr < hi)
            return px_arr[mask].tolist()

        windows = {
            "inslot_3m": (slot_ts,        slot_ts + OBSERVE_SECS),
            "pre_3m":    (slot_ts - 180,   slot_ts),
            "pre_15m":   (slot_ts - 900,   slot_ts),
            "pre_1h":    (slot_ts - 3600,  slot_ts),
        }
        for wname, (lo, hi) in windows.items():
            feat.update(_window_feats(slice_px(lo, hi), label, wname))

        px_1h = slice_px(slot_ts - 3600, slot_ts)
        if len(px_1h) > 1:
            lo1h = min(px_1h); hi1h = max(px_1h)
            rng  = hi1h - lo1h + 1e-8
            feat[f"{label}_pct_of_1h_range"] = float((px_1h[-1] - lo1h) / rng)
        else:
            feat[f"{label}_pct_of_1h_range"] = 0.5

    return feat


# ── Trades store ───────────────────────────────────────────────────────────────
def load_trades() -> list[dict]:
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []

def save_trades(trades: list[dict]) -> None:
    TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


# ── Market discovery ───────────────────────────────────────────────────────────
def get_current_slots() -> list[int]:
    """Return the current and next slot timestamps."""
    now = int(time.time())
    cur = (now // SLOT_DURATION) * SLOT_DURATION
    return [cur, cur + SLOT_DURATION]

def fetch_market(slot_ts: int) -> dict | None:
    """Fetch market metadata for a given slot from Gamma API."""
    slug = f"btc-updown-5m-{slot_ts}"
    try:
        r = requests.get(f"{GAMMA_HOST}/markets/slug/{slug}", timeout=8)
        if not r.ok:
            return None
        m = r.json()
        if not m or m.get("archived"):
            return None
        # Skip markets that are closed (resolved) — outcomePrices will be 1/0
        op = m.get("outcomePrices", "[]")
        if isinstance(op, str):
            op = json.loads(op)
        if op and len(op) >= 2:
            try:
                if float(op[0]) >= 0.99 or float(op[1]) >= 0.99:
                    return None  # already resolved
            except (ValueError, TypeError):
                pass
        token_ids = m.get("clobTokenIds", "[]")
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if len(token_ids) < 2:
            return None
        end_str = m.get("endDate", "") or m.get("endDateIso", "")
        if end_str:
            end_ts = int(datetime.fromisoformat(
                end_str.replace("Z", "+00:00")).timestamp())
        else:
            end_ts = slot_ts + SLOT_DURATION
        return {
            "condition_id": m.get("conditionId", ""),
            "question":     m.get("question", "")[:60],
            "yes_token":    token_ids[0],
            "no_token":     token_ids[1],
            "end_ts":       end_ts,
            "slot_ts":      slot_ts,
            "resolution":   m.get("resolution"),
        }
    except Exception as e:
        log.warning("fetch_market slot=%d: %s", slot_ts, e)
        return None


# ── UP-token price series from CLOB prices-history ────────────────────────────
def fetch_price_history(yes_token: str, slot_ts: int) -> list[float]:
    """
    Fetch the UP token price series during [slot_ts, slot_ts+OBSERVE_SECS).
    Uses clob.polymarket.com/prices-history (startTs/endTs, fidelity=1).
    Returns list of price floats sorted ascending by time.
    Falls back to [] on any error.
    """
    try:
        r = requests.get(
            f"{CLOB_HOST}/prices-history",
            params={
                "market":   yes_token,
                "startTs":  slot_ts,
                "endTs":    slot_ts + OBSERVE_SECS,
                "fidelity": 1,
            },
            timeout=8,
        )
        if not r.ok:
            return []
        data = r.json()
        history = data.get("history", [])
        # Filter to inslot window and sort ascending
        prices = [
            float(h["p"])
            for h in history
            if slot_ts <= int(h.get("t", 0)) <= slot_ts + OBSERVE_SECS
        ]
        return prices
    except Exception as e:
        log.warning("fetch_price_history: %s", e)
        return []


# ── Order flow feature computation (FIXED) ────────────────────────────────────
def fetch_3min_features(yes_token: str, no_token: str, slot_ts: int) -> dict | None:
    """
    Build real-time features for the v3b model (spot + time only).

    v3b uses ONLY features available in real-time at t=180s:
      - spot BTC/ETH/SOL from local WS buffer (zero network calls)
      - time features (hour_sin/cos, dow_sin/cos, hour)

    Dropped: price_history (CLOB token prices have near-zero correlation with outcome,
    and created strong DOWN bias in v3). Order flow also dropped (data-api 4-5min lag).

    Signal comes from btc/eth/sol inslot_3m_ret (corr ~0.37 with outcome).
    """
    now = int(time.time())
    t_elapsed = now - slot_ts
    if t_elapsed < OBSERVE_SECS:
        log.info("  Slot at t=%ds — waiting for 3-min mark", t_elapsed)
        return None

    # ── Time features ─────────────────────────────────────────────────────────
    dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    dow  = dt.weekday()

    feat = {
        "hour":     hour,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin":  math.sin(2 * math.pi * dow / 7),
        "dow_cos":  math.cos(2 * math.pi * dow / 7),
    }

    # ── Spot features (from local WS buffer — no network call) ────────────────
    log.info("  Reading spot prices from buffer...")
    spot_feat = build_spot_features(slot_ts)
    feat.update(spot_feat)

    # Verify we have all required features
    missing = [f for f in FEATURES if f not in feat]
    if missing:
        log.warning("  Missing features: %s — filling with 0", missing[:5])
        for f in missing:
            feat[f] = 0.0

    log.info(
        "  Features: btc_inslot_ret=%.5f eth_inslot_ret=%.5f sol_inslot_ret=%.5f"
        " btc_inslot_mom=%.5f hour=%.1f",
        feat.get("btc_inslot_3m_ret", 0), feat.get("eth_inslot_3m_ret", 0),
        feat.get("sol_inslot_3m_ret", 0), feat.get("btc_inslot_3m_mom", 0), hour,
    )

    return feat


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(feat: dict) -> tuple[str, float]:
    """Return (direction, confidence). direction='UP' or 'DOWN'."""
    X = np.array([[feat.get(f, 0.0) for f in FEATURES]])
    prob_up = float(model.predict_proba(X)[0][1])
    if prob_up >= 0.5:
        return "UP", prob_up
    else:
        return "DOWN", 1.0 - prob_up


# ── Entry price ────────────────────────────────────────────────────────────────
def get_market_price(token_id: str) -> float | None:
    """
    Get best ask price for a token using CLOB (not gamma orderbook or data-api).
    Returns None if price cannot be fetched reliably.
    """
    # Primary: CLOB /price endpoint (BUY side = ask)
    try:
        r = requests.get(
            f"{CLOB_HOST}/price",
            params={"token_id": token_id, "side": "BUY"},
            timeout=5,
        )
        if r.ok:
            price_str = r.json().get("price", "0")
            price = float(price_str)
            if 0.01 < price < 0.99:
                return price
    except Exception:
        pass

    # Fallback: CLOB /midpoint
    try:
        r = requests.get(
            f"{CLOB_HOST}/midpoint",
            params={"token_id": token_id},
            timeout=5,
        )
        if r.ok:
            mid_str = r.json().get("mid", "0")
            mid = float(mid_str)
            if 0.01 < mid < 0.99:
                return mid
    except Exception:
        pass

    # Final fallback: gamma outcomePrices
    # Requires fetching the full market — only use if CLOB is down
    return None


# ── Settlement ─────────────────────────────────────────────────────────────────
def settle_trades(trades: list[dict]) -> tuple[list[dict], bool]:
    """Settle any open trades whose slot has ended."""
    now = int(time.time())
    updated = False
    for trade in trades:
        if trade.get("status") != "open":
            continue
        slot_end = trade["slot_ts"] + SLOT_DURATION
        if now < slot_end + SETTLE_GRACE:
            continue  # too early

        slug = f"btc-updown-5m-{trade['slot_ts']}"
        resolution = None
        try:
            r = requests.get(f"{GAMMA_HOST}/markets/slug/{slug}", timeout=8)
            if r.ok:
                data = r.json()
                resolution = data.get("resolution")
                if resolution is None:
                    op = data.get("outcomePrices", "[]")
                    if isinstance(op, str):
                        op = json.loads(op)
                    if op and len(op) >= 2:
                        up_price = float(op[0])
                        dn_price = float(op[1])
                        if up_price >= 0.99:
                            resolution = 1.0
                        elif dn_price >= 0.99:
                            resolution = 0.0
        except Exception:
            pass

        if resolution is None:
            log.info("  Slot %d: resolution not yet available", trade["slot_ts"])
            continue

        try:
            res = float(resolution)
        except (ValueError, TypeError):
            log.warning("  Unexpected resolution value: %r", resolution)
            continue

        actual    = "UP" if res >= 0.5 else "DOWN"
        direction = trade["direction"]
        entry_price = trade["entry_price"]

        tokens = STAKE_USDC / entry_price
        if direction == actual:
            pnl    = tokens * (1.0 - 0.02) - STAKE_USDC  # 2% taker fee
            result = "WIN"
        else:
            pnl    = -STAKE_USDC
            result = "LOSS"

        trade["status"]     = "settled"
        trade["actual"]     = actual
        trade["result"]     = result
        trade["pnl_usdc"]   = round(pnl, 4)
        trade["settled_at"] = now
        updated = True

        log.info(
            "  SETTLED slot=%d | pred=%s actual=%s | %s | P&L $%.2f",
            trade["slot_ts"], direction, actual, result, pnl,
        )

    return trades, updated


# ── Summary ────────────────────────────────────────────────────────────────────
def _print_summary(trades: list[dict]) -> None:
    settled = [t for t in trades if t.get("status") == "settled"]
    v2 = [t for t in settled if t.get("model") in ("v2", "v3")]
    wins = sum(1 for t in v2 if t.get("result") == "WIN")
    losses = sum(1 for t in v2 if t.get("result") == "LOSS")
    pnl = sum(t.get("pnl_usdc", 0) for t in v2)
    log.info(
        "  Summary (v2/v3): %dW/%dL  P&L=$%.2f  WinRate=%.0f%%",
        wins, losses, pnl, wins / (wins + losses) * 100 if (wins + losses) > 0 else 0,
    )


# ── Main ────────────────────────────────────────────────────────────────────────
def run() -> None:
    now = int(time.time())
    trades = load_trades()

    # Step 1: Settle expired open trades
    trades, settled_any = settle_trades(trades)
    if settled_any:
        save_trades(trades)
        _print_summary(trades)

    # Step 2: Check slots for new entry opportunities
    # Only block re-entry for trades that are open/settled (not skipped)
    already_entered = {
        t["slot_ts"] for t in trades
        if t.get("status") in ("open", "settled", "error")
    }
    entered_new = False

    for slot_ts in get_current_slots():
        t_elapsed = now - slot_ts
        if not (ENTER_WINDOW[0] <= t_elapsed <= ENTER_WINDOW[1]):
            continue
        if slot_ts in already_entered:
            continue

        log.info("Entry window: slot=%d t=%ds — fetching market...", slot_ts, t_elapsed)
        market = fetch_market(slot_ts)
        if not market:
            log.info("  Market not found or already closed")
            continue

        yes_token = market["yes_token"]
        no_token  = market["no_token"]

        log.info("  Market: %s", market["question"])
        log.info("  Building features...")
        feat = fetch_3min_features(yes_token, no_token, slot_ts)
        if feat is None:
            log.info("  Features not ready — skip")
            trades.append({
                "slot_ts":    slot_ts,
                "status":     "skipped",
                "reason":     "features not ready",
                "entered_at": now,
            })
            save_trades(trades)
            continue

        direction, confidence = predict(feat)
        log.info("  Prediction: %s  confidence=%.1f%%", direction, confidence * 100)

        if confidence < MIN_CONFIDENCE:
            log.info("  Skip — confidence %.1f%% < %.0f%%", confidence * 100, MIN_CONFIDENCE * 100)
            trades.append({
                "slot_ts":    slot_ts,
                "direction":  direction,
                "confidence": round(confidence, 4),
                "entry_price": None,
                "status":     "skipped",
                "reason":     f"confidence {confidence*100:.2f}% < {MIN_CONFIDENCE*100:.0f}%",
                "entered_at": now,
                "price_trend": round(feat.get("price_trend", 0), 4),
                "price_last":  round(feat.get("price_last", 0), 4),
            })
            save_trades(trades)
            continue

        # Pick which token to price (the one we'd bet on)
        bet_token = yes_token if direction == "UP" else no_token
        ask_price = get_market_price(bet_token)

        if ask_price is None:
            log.warning("  Could not fetch ask price — skip")
            trades.append({
                "slot_ts":    slot_ts,
                "direction":  direction,
                "confidence": round(confidence, 4),
                "entry_price": None,
                "status":     "skipped",
                "reason":     "price fetch failed",
                "entered_at": now,
            })
            save_trades(trades)
            continue

        true_edge = confidence - ask_price
        log.info("  Ask=%.3f  edge=%.1f%%", ask_price, true_edge * 100)

        if true_edge < MIN_EDGE:
            log.info("  Skip — edge %.1f%% < %.0f%%", true_edge * 100, MIN_EDGE * 100)
            trades.append({
                "slot_ts":    slot_ts,
                "direction":  direction,
                "confidence": round(confidence, 4),
                "entry_price": ask_price,
                "status":     "skipped",
                "reason":     f"edge {true_edge*100:.2f}% < {MIN_EDGE*100:.0f}% — no edge vs market",
                "entered_at": now,
                "price_trend": round(feat.get("price_trend", 0), 4),
                "price_last":  round(feat.get("price_last", 0), 4),
                "true_edge":  round(true_edge, 4),
            })
            save_trades(trades)
            continue

        # Record trade
        trade_record: dict = {
            "slot_ts":      slot_ts,
            "direction":    direction,
            "confidence":   round(confidence, 4),
            "entry_price":  ask_price,
            "stake_usdc":   STAKE_USDC,
            "token_id":     bet_token,
            "status":       "open",
            "entered_at":   now,
            "model":        "v3b",
            # Diagnostic features
            "btc_inslot_3m_ret": round(feat.get("btc_inslot_3m_ret", 0), 6),
            "eth_inslot_3m_ret": round(feat.get("eth_inslot_3m_ret", 0), 6),
            "sol_inslot_3m_ret": round(feat.get("sol_inslot_3m_ret", 0), 6),
            "btc_inslot_3m_mom": round(feat.get("btc_inslot_3m_mom", 0), 6),
            "true_edge":    round(true_edge, 4),
        }

        trades.append(trade_record)
        save_trades(trades)
        entered_new = True

        log.info(
            "  ENTERED %s @ $%.3f  edge=%.1f%%  stake=$%.2f",
            direction, ask_price, true_edge * 100, STAKE_USDC,
        )

    if not entered_new and not settled_any:
        log.debug("No action this tick")


if __name__ == "__main__":
    run()
