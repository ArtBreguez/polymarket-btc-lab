"""
BTC 5m Directional Paper Trader — v2
=====================================
Strategy:
  - Each slot is 5 minutes: btc-updown-5m-{unix_timestamp}
  - At t=180s (after observing 3 min of order flow): predict UP or DOWN
  - If model confidence > 60% AND edge >= 10%: record paper bet ($5 USDC)
  - After slot closes + 30s grace: settle (check resolution, compute P&L)

Model: btc_model_v2_research.pkl
  - 73 features: order flow (3 windows) + spot BTC/ETH/SOL + time
  - Trained on 7 cryptos x 616 slots = 3,900 samples
  - WF AUC: 0.853, WF Acc: 77.0%

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
MODEL_PATH      = ARTIFACTS / "btc_model_v2_research.pkl"
GAMMA_HOST      = "https://gamma-api.polymarket.com"
DATA_API        = "https://data-api.polymarket.com"
SPOT_BUFFER     = Path("/tmp/spot_buffer.json")
SLOT_DURATION   = 300           # 5 minutes
OBSERVE_SECS    = 180           # enter after first 3 min
ENTER_WINDOW    = (170, 240)    # t-seconds where we allow entry
SETTLE_GRACE    = 60            # settle this many seconds after slot end
MIN_CONFIDENCE  = 0.60          # only bet if model says > 60%
MIN_EDGE        = 0.10          # require at least 10% edge over market price
STAKE_USDC      = 5.0           # flat stake per trade (paper only)
BUFFER_STALE_SECS = 120         # warn if buffer older than this

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
    Only computes features that appeared in the model's top-20 importance:
      - inslot_3m: ret/vol/mom  (all 3 symbols)
      - pre_3m:    ret/vol/mom  (all 3 symbols)
      - pre_15m:   ret/vol/mom  (all 3 symbols)
      - pre_1h:    ret/vol/mom  (all 3 symbols)
      - pct_of_1h_range         (all 3 symbols)
    All computed from buffer — zero network calls.
    """
    feat: dict = {}
    label_map = {"btcusdt": "btc", "ethusdt": "eth", "solusdt": "sol"}

    # Load buffer (written atomically by spot_daemon.py)
    if not SPOT_BUFFER.exists():
        log.warning("spot_buffer.json not found — is spot_daemon.py running?")
        # Fill zeros for all spot features
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

        # Build arrays (candles already sorted ascending from daemon)
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

        # pct_of_1h_range
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
        if not m or m.get("closed") or m.get("archived"):
            return None
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


# ── Order flow feature computation ────────────────────────────────────────────
def fetch_3min_features(yes_token: str, no_token: str, slot_ts: int) -> dict | None:
    """
    Fetch live trades for both tokens, filter to [0, 180s), compute the
    73 features expected by model v2.
    """
    now = int(time.time())
    t_elapsed = now - slot_ts
    if t_elapsed < OBSERVE_SECS:
        log.info("  Slot at t=%ds — waiting for 3-min mark", t_elapsed)
        return None

    def fetch_token_trades(token_id: str) -> list[dict]:
        """Paginate data-api (reverse-chron) until intra-slot trades found."""
        all_trades: list[dict] = []
        offset = 0
        while offset <= 2000:
            try:
                r = requests.get(
                    f"{DATA_API}/trades",
                    params={"asset": token_id, "limit": 500, "offset": offset},
                    timeout=10,
                )
                batch = r.json() if r.ok else []
                if isinstance(batch, dict):
                    batch = batch.get("data", [])
                if not batch:
                    break
                all_trades.extend(batch)
                min_ts = min(int(t.get("timestamp", 0)) for t in batch)
                if min_ts > 1e12:
                    min_ts //= 1000
                min_t_sec = min_ts - slot_ts
                if min_t_sec < -600:
                    break
                if min_t_sec >= 0:
                    break
                offset += 500
            except Exception as e:
                log.warning("  fetch_token_trades %s: %s", token_id[:12], e)
                break
        return all_trades

    raw_yes = fetch_token_trades(yes_token)
    raw_no  = fetch_token_trades(no_token)

    def parse_tick(t: dict, outcome: str) -> dict | None:
        try:
            ts = int(t.get("timestamp", 0) or 0)
            if ts > 1e12:
                ts = ts // 1000
            t_sec = ts - slot_ts
            if t_sec < 0 or t_sec >= OBSERVE_SECS:
                return None
            return {
                "t_sec":   t_sec,
                "price":   float(t.get("price", 0) or 0),
                "size":    float(t.get("size", 0) or t.get("amount", 0) or 0),
                "side":    str(t.get("side", "") or ""),
                "outcome": outcome,
            }
        except Exception:
            return None

    ticks = []
    for t in raw_yes:
        p = parse_tick(t, "Up")
        if p:
            ticks.append(p)
    for t in raw_no:
        p = parse_tick(t, "Down")
        if p:
            ticks.append(p)

    n = len(ticks)
    log.info("  Ticks in [0,180s): %d  (yes=%d no=%d)", n, len(raw_yes), len(raw_no))

    if n == 0:
        log.warning("  No ticks in window — skip")
        return None

    # ── Order flow aggregates ─────────────────────────────────────────────────
    vol_up  = sum(t["size"] for t in ticks if t["outcome"] == "Up")
    vol_dn  = sum(t["size"] for t in ticks if t["outcome"] == "Down")
    total   = vol_up + vol_dn
    n_buy   = sum(1 for t in ticks if t["side"].upper() == "BUY")
    n_up    = sum(1 for t in ticks if t["outcome"] == "Up")
    n_dn    = sum(1 for t in ticks if t["outcome"] == "Down")

    up_ratio  = vol_up / (total + 1e-8)
    vwap_up   = sum(t["price"] * t["size"] for t in ticks if t["outcome"] == "Up") / (vol_up + 1e-8)
    vwap_dn   = sum(t["price"] * t["size"] for t in ticks if t["outcome"] == "Down") / (vol_dn + 1e-8)
    buy_ratio = n_buy / (n + 1e-8)
    avg_size  = total / n

    # 3 sub-windows: w1=[0,60s), w2=[60,120s), w3=[120,180s)
    def window_stats(lo: int, hi: int) -> tuple[float, float, int]:
        w = [t for t in ticks if lo <= t["t_sec"] < hi]
        vu = sum(t["size"] for t in w if t["outcome"] == "Up")
        vd = sum(t["size"] for t in w if t["outcome"] == "Down")
        wt = vu + vd
        return vu / (wt + 1e-8), wt, len(w)

    ur1, t1, n1 = window_stats(0, 60)
    ur2, t2, n2 = window_stats(60, 120)
    ur3, t3, n3 = window_stats(120, 180)

    px = [t["price"] for t in sorted(ticks, key=lambda x: x["t_sec"])]
    dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    dow  = dt.weekday()

    order_flow_feat = {
        "n_ticks":        float(n),
        "total_vol":      total,
        "vol_up":         vol_up,
        "vol_dn":         vol_dn,
        "up_ratio":       up_ratio,
        "vwap_up":        vwap_up,
        "vwap_dn":        vwap_dn,
        "vwap_diff":      vwap_up - vwap_dn,
        "buy_ratio":      buy_ratio,
        "avg_size":       avg_size,
        "n_up":           float(n_up),
        "n_dn":           float(n_dn),
        "up_ratio_w1":    ur1, "vol_w1": t1, "n_w1": float(n1),
        "up_ratio_w2":    ur2, "vol_w2": t2, "n_w2": float(n2),
        "up_ratio_w3":    ur3, "vol_w3": t3, "n_w3": float(n3),
        "momentum_early": ur2 - ur1,
        "momentum_late":  ur3 - ur2,
        "acceleration":   (ur3 - ur2) - (ur2 - ur1),
        "imbalance":      (vol_up - vol_dn) / (total + 1e-8),
        "price_first":    float(px[0]) if px else 0.5,
        "price_last":     float(px[-1]) if px else 0.5,
        "price_trend":    float(px[-1] - px[0]) if len(px) > 1 else 0.0,
        "price_vol":      float(np.std(px)) if len(px) > 1 else 0.0,
        "hour":           hour,
        "hour_sin":       math.sin(2 * math.pi * hour / 24),
        "hour_cos":       math.cos(2 * math.pi * hour / 24),
        "dow_sin":        math.sin(2 * math.pi * dow / 7),
        "dow_cos":        math.cos(2 * math.pi * dow / 7),
    }

    # ── Spot features (from local WS buffer — no network call) ────────────────
    log.info("  Reading spot prices from buffer...")
    spot_feat = build_spot_features(slot_ts)
    order_flow_feat.update(spot_feat)

    # Verify we have all required features
    missing = [f for f in FEATURES if f not in order_flow_feat]
    if missing:
        log.warning("  Missing features: %s — filling with 0", missing[:5])
        for f in missing:
            order_flow_feat[f] = 0.0

    return order_flow_feat


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
def get_market_price(token_id: str) -> float:
    """Get best ask price for a token (what we'd pay to buy it)."""
    try:
        r = requests.get(f"{GAMMA_HOST}/orderbook/{token_id}", timeout=5)
        if r.ok:
            ob = r.json()
            asks = ob.get("asks") or ob.get("ask", [])
            if asks:
                best = min(float(a.get("price", 1.0)) for a in asks if a.get("price"))
                return best
    except Exception:
        pass
    # Fallback: use last trade price
    try:
        r = requests.get(f"{DATA_API}/trades", params={"asset": token_id, "limit": 1}, timeout=5)
        trades = r.json() if r.ok else []
        if isinstance(trades, dict):
            trades = trades.get("data", [])
        if trades:
            return float(trades[0].get("price", 0.5) or 0.5)
    except Exception:
        pass
    return 0.5  # fallback


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
            pnl    = tokens * 1.0 - STAKE_USDC
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
    already_entered = {t["slot_ts"] for t in trades}
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

        feat = fetch_3min_features(market["yes_token"], market["no_token"], slot_ts)
        if feat is None:
            continue

        direction, confidence = predict(feat)
        log.info("  Prediction: %s  confidence=%.1f%%", direction, confidence * 100)

        if confidence < MIN_CONFIDENCE:
            log.info("  Skip — confidence %.1f%% < %.0f%% threshold",
                     confidence * 100, MIN_CONFIDENCE * 100)
            trades.append({
                "slot_ts":     slot_ts,
                "direction":   direction,
                "confidence":  round(confidence, 4),
                "entry_price": None,
                "status":      "skipped",
                "reason":      f"confidence {confidence:.2%} < {MIN_CONFIDENCE:.0%}",
                "entered_at":  now,
                "model":       "v2",
                "n_ticks":     int(feat.get("n_ticks", 0)),
                "up_ratio":    round(feat.get("up_ratio", 0.5), 4),
                "momentum_early": round(feat.get("momentum_early", 0), 4),
                "momentum_late":  round(feat.get("momentum_late", 0), 4),
            })
            save_trades(trades)
            continue

        # Get entry price and compute true edge
        token_id    = market["yes_token"] if direction == "UP" else market["no_token"]
        entry_price = get_market_price(token_id)
        prob_up     = float(model.predict_proba(
            np.array([[feat.get(f, 0.0) for f in FEATURES]])
        )[0][1])
        model_prob  = prob_up if direction == "UP" else (1.0 - prob_up)
        true_edge   = model_prob - entry_price

        log.info(
            "  Entry candidate: BUY %s @ $%.3f | model=%.1f%% | edge=%.1f%%",
            direction, entry_price, model_prob * 100, true_edge * 100,
        )

        if true_edge < MIN_EDGE:
            log.info(
                "  Skip — true edge %.1f%% < %.0f%% (market already prices this in)",
                true_edge * 100, MIN_EDGE * 100,
            )
            trades.append({
                "slot_ts":     slot_ts,
                "direction":   direction,
                "confidence":  round(confidence, 4),
                "entry_price": round(entry_price, 4),
                "status":      "skipped",
                "reason":      f"edge {true_edge:.2%} < {MIN_EDGE:.0%} — no edge vs market",
                "entered_at":  now,
                "model":       "v2",
                "n_ticks":     int(feat.get("n_ticks", 0)),
                "up_ratio":    round(feat.get("up_ratio", 0.5), 4),
                "true_edge":   round(true_edge, 4),
            })
            save_trades(trades)
            continue

        # Record paper trade
        trade = {
            "slot_ts":        slot_ts,
            "direction":      direction,
            "confidence":     round(confidence, 4),
            "entry_price":    round(entry_price, 4),
            "stake_usdc":     STAKE_USDC,
            "token_id":       token_id,
            "status":         "open",
            "entered_at":     now,
            "model":          "v2",
            "n_ticks":        int(feat.get("n_ticks", 0)),
            "up_ratio":       round(feat.get("up_ratio", 0.5), 4),
            "momentum_early": round(feat.get("momentum_early", 0), 4),
            "momentum_late":  round(feat.get("momentum_late", 0), 4),
            "true_edge":      round(true_edge, 4),
            "up_ratio_w1":    round(feat.get("up_ratio_w1", 0.5), 4),
            "up_ratio_w2":    round(feat.get("up_ratio_w2", 0.5), 4),
            "up_ratio_w3":    round(feat.get("up_ratio_w3", 0.5), 4),
            "btc_pre_3m_ret": round(feat.get("btc_pre_3m_ret", 0.0), 6),
            "btc_inslot_3m_ret": round(feat.get("btc_inslot_3m_ret", 0.0), 6),
        }
        trades.append(trade)
        save_trades(trades)
        entered_new = True
        log.info("  Recorded paper trade (v2) → %s", TRADES_FILE)

    _print_summary(trades)


def _print_summary(trades: list[dict]) -> None:
    settled = [t for t in trades if t.get("status") == "settled"]
    open_   = [t for t in trades if t.get("status") == "open"]
    skipped = [t for t in trades if t.get("status") == "skipped"]
    wins    = [t for t in settled if t.get("result") == "WIN"]
    losses  = [t for t in settled if t.get("result") == "LOSS"]
    total_pnl = sum(t.get("pnl_usdc", 0) for t in settled)
    win_rate  = len(wins) / len(settled) if settled else 0

    # v2 trades only
    v2_settled = [t for t in settled if t.get("model") == "v2"]
    v2_pnl     = sum(t.get("pnl_usdc", 0) for t in v2_settled)
    v2_wins    = sum(1 for t in v2_settled if t.get("result") == "WIN")

    log.info(
        "── Summary: settled=%d (W%d/L%d, win=%.0f%%) open=%d skipped=%d | total P&L $%.2f",
        len(settled), len(wins), len(losses), win_rate * 100,
        len(open_), len(skipped), total_pnl,
    )
    if v2_settled:
        log.info(
            "── v2 model only: settled=%d wins=%d P&L $%.2f",
            len(v2_settled), v2_wins, v2_pnl,
        )


if __name__ == "__main__":
    run()
