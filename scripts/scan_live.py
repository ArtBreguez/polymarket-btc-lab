"""
STEP 6: Test scan_live.py against real Polymarket API.
Updated to use v4_clean model and features.
"""
import pickle
import time
import json
import requests
import numpy as np
from datetime import datetime, timezone

ARTIFACTS = "artifacts"
GAMMA_HOST = "https://gamma-api.polymarket.com"
DATA_API   = "https://data-api.polymarket.com"
BINANCE    = "https://api.binance.com"

# ── Load model ────────────────────────────────────────────────
print("Loading v4_clean model...")
with open(f"{ARTIFACTS}/btc_model_v4_clean.pkl", "rb") as f:
    bundle = pickle.load(f)
model    = bundle["model"]
features = bundle["features"]
print(f"Model loaded: {len(features)} features")

# ── Find active BTC 5m markets ────────────────────────────────
def find_active_btc_5m_markets():
    now = int(time.time())
    slot_sec = 300
    current_slot = (now // slot_sec) * slot_sec
    markets = []
    for slot in [current_slot, current_slot + slot_sec]:
        slug = f"btc-updown-5m-{slot}"
        try:
            r = requests.get(f"{GAMMA_HOST}/markets/slug/{slug}", timeout=8)
            if not r.ok or not r.json():
                continue
            m = r.json()
            if m.get("closed") or m.get("archived"):
                continue
            token_ids = m.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            if len(token_ids) < 2:
                continue
            end_ts = int(datetime.fromisoformat(
                m["endDate"].replace("Z", "+00:00")).timestamp())
            if end_ts < now + 60:
                continue
            markets.append({
                "condition_id": m.get("conditionId", ""),
                "question": m.get("question", "")[:60],
                "yes_token": token_ids[0],
                "no_token":  token_ids[1],
                "end_ts":    end_ts,
                "slot":      slot,
            })
        except Exception as e:
            print(f"  Slot {slot}: {e}")
    return markets

# ── Fetch features from live trades ──────────────────────────
def fetch_live_features(condition_id, yes_token, slot_ts):
    """Compute the 33 v3 features + temporal window features from live trades."""
    feat = {}

    # 1. Fetch trades
    try:
        r = requests.get(f"{DATA_API}/trades",
                         params={"asset": yes_token, "limit": 500}, timeout=10)
        trades_raw = r.json() if r.ok else []
        if isinstance(trades_raw, dict):
            trades_raw = trades_raw.get("data", [])
    except Exception as e:
        print(f"    trades fetch error: {e}")
        trades_raw = []

    now = int(time.time())
    slot_end_ts = slot_ts + 300

    # Filter to within slot
    trades = []
    for t in trades_raw:
        ts = int(t.get("timestamp", 0))
        if ts > 1e10:
            ts = ts // 1000
        if slot_ts <= ts <= slot_end_ts:
            trades.append({
                "ts": ts,
                "price": float(t.get("price", 0) or 0),
                "size":  float(t.get("size", 0) or 0),
                "side":  t.get("side", ""),
                "outcome": t.get("outcome", ""),
            })

    trades.sort(key=lambda x: x["ts"])
    n = len(trades)

    if n == 0:
        # No trades yet — return near-uniform features
        print(f"    No trades in slot yet — using neutral features")
        feat = {f: 0.0 for f in features}
        feat["up_down_volume_ratio"] = 1.0
        feat["hour_of_day_sin"] = np.sin(2 * np.pi * datetime.utcnow().hour / 24)
        feat["hour_of_day_cos"] = np.cos(2 * np.pi * datetime.utcnow().hour / 24)
        return feat

    prices  = [t["price"] for t in trades]
    sizes   = [t["size"]  for t in trades]
    is_up   = [t["outcome"] == "Up"   for t in trades]
    is_down = [t["outcome"] == "Down" for t in trades]
    is_buy  = [t["side"] == "BUY"     for t in trades]

    vol_up   = sum(s for s, u in zip(sizes, is_up)   if u)
    vol_down = sum(s for s, d in zip(sizes, is_down) if d)
    buy_vol  = sum(s for s, b in zip(sizes, is_buy)  if b)
    sell_vol = sum(s for s, b in zip(sizes, is_buy)  if not b)
    total    = vol_up + vol_down

    # v3-style price features
    feat["n_ticks"]            = float(n)
    feat["n_trades"]           = float(n)
    feat["first_price"]        = prices[0]
    feat["last_price"]         = prices[-1]
    feat["price_mean"]         = float(np.mean(prices))
    feat["price_std"]          = float(np.std(prices)) if n > 1 else 0.0
    feat["price_min"]          = float(np.min(prices))
    feat["price_max"]          = float(np.max(prices))
    feat["price_momentum"]     = prices[-1] - prices[0] if n > 1 else 0.0
    feat["price_at_25pct"]     = float(np.percentile(prices, 25))
    feat["price_at_50pct"]     = float(np.percentile(prices, 50))
    feat["price_at_75pct"]     = float(np.percentile(prices, 75))

    feat["total_volume_usdc"]  = float(sum(sizes))
    feat["buy_volume_usdc"]    = buy_vol
    feat["sell_volume_usdc"]   = sell_vol
    feat["buy_sell_imbalance"] = (buy_vol - sell_vol) / (total + 1e-8)
    feat["up_volume_usdc"]     = vol_up
    feat["down_volume_usdc"]   = vol_down
    feat["up_down_volume_ratio"] = vol_up / (vol_down + 1e-8)
    feat["n_buy_trades"]       = float(sum(is_buy))
    feat["n_sell_trades"]      = float(n - sum(is_buy))
    feat["avg_trade_size"]     = float(np.mean(sizes))

    vwap_up = sum(t["price"] * t["size"] for t in trades if t["outcome"] == "Up") / (vol_up + 1e-8)
    vwap_dn = sum(t["price"] * t["size"] for t in trades if t["outcome"] == "Down") / (vol_down + 1e-8)
    feat["vwap_up"]   = vwap_up
    feat["vwap_down"] = vwap_dn

    # Temporal window features
    t_secs = [(t["ts"] - slot_ts) for t in trades]
    def wfeat(wl, wh, label):
        w = [t for t, ts in zip(trades, t_secs) if wl <= ts < wh]
        wn = len(w)
        wvu = sum(t["size"] for t in w if t["outcome"] == "Up")
        wvd = sum(t["size"] for t in w if t["outcome"] == "Down")
        wtot = wvu + wvd
        ur = wvu / (wtot + 1e-8)
        vwu = sum(t["price"]*t["size"] for t in w if t["outcome"]=="Up")/(wvu+1e-8)
        vwd = sum(t["price"]*t["size"] for t in w if t["outcome"]=="Down")/(wvd+1e-8)
        feat[f"{label}_vol_up"]    = wvu
        feat[f"{label}_vol_down"]  = wvd
        feat[f"{label}_up_ratio"]  = ur
        feat[f"{label}_n_ticks"]   = float(wn)
        feat[f"{label}_n_up"]      = float(sum(1 for t in w if t["outcome"]=="Up"))
        feat[f"{label}_n_down"]    = float(sum(1 for t in w if t["outcome"]=="Down"))
        feat[f"{label}_total_vol"] = wtot
        feat[f"{label}_vwap_up"]   = vwu
        feat[f"{label}_vwap_down"] = vwd

    wfeat(0, 90, "early")
    wfeat(90, 210, "mid")
    wfeat(210, 300, "late")
    total_n = feat["early_n_ticks"] + feat["mid_n_ticks"] + feat["late_n_ticks"]
    feat["total_temporal_n_ticks"] = total_n
    feat["momentum_vol_ratio"]   = feat["early_up_ratio"] / (feat["late_up_ratio"] + 1e-8)
    feat["late_surge"]           = feat["late_n_ticks"] / (total_n + 1e-8)
    feat["early_mid_vol_diff"]   = feat["early_up_ratio"] - feat["mid_up_ratio"]
    feat["mid_late_vol_diff"]    = feat["mid_up_ratio"] - feat["late_up_ratio"]
    feat["vol_trend"]            = feat["late_total_vol"] / (feat["early_total_vol"] + 1e-8)

    # Spot price from Binance
    try:
        rb = requests.get(f"{BINANCE}/api/v3/ticker/price?symbol=BTCUSDT", timeout=3)
        spot = float(rb.json()["price"]) if rb.ok else 0.0
        feat["spot_price_start"] = spot
        feat["spot_price_end"]   = spot
        feat["spot_return"]      = 0.0
        feat["spot_volatility"]  = 0.0
        feat["spot_price_mean"]  = spot
    except Exception:
        for k in ["spot_price_start","spot_price_end","spot_return","spot_volatility","spot_price_mean"]:
            feat[k] = 0.0

    # Time features
    dt_now = datetime.now(timezone.utc)
    h = dt_now.hour + dt_now.minute / 60.0
    dow = dt_now.weekday()
    feat["hour_of_day_sin"] = np.sin(2 * np.pi * h / 24)
    feat["hour_of_day_cos"] = np.cos(2 * np.pi * h / 24)
    feat["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    feat["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)

    return feat

# ── Run scan ──────────────────────────────────────────────────
print("\nScanning active BTC 5m markets...")
markets = find_active_btc_5m_markets()
print(f"Found {len(markets)} active market(s)")

if not markets:
    print("No active markets right now. Try again during US trading hours (9AM-4PM ET).")
else:
    for m in markets:
        print(f"\n--- {m['question']} ---")
        now = int(time.time())
        secs_left = m["end_ts"] - now
        print(f"  Ends in: {secs_left}s  |  slot_ts={m['slot']}")

        feat_dict = fetch_live_features(m["condition_id"], m["yes_token"], m["slot"])

        # Build feature vector in correct order
        X = np.array([[feat_dict.get(f, 0.0) for f in features]])

        prob_up = model.predict_proba(X)[0][1]
        prob_dn = 1 - prob_up
        edge_up = prob_up - 0.5
        edge_dn = prob_dn - 0.5

        direction = "UP" if prob_up > 0.5 else "DOWN"
        confidence = max(prob_up, prob_dn)
        edge = abs(edge_up)

        print(f"  Prediction: {direction}  ({confidence*100:.1f}% confidence)")
        print(f"  P(UP)={prob_up:.3f}  P(DOWN)={prob_dn:.3f}")
        print(f"  Edge: {edge*100:+.1f}%")
        print(f"  Trades in slot: {int(feat_dict.get('n_ticks', 0))}")
        print(f"  up_vol=${feat_dict.get('up_volume_usdc',0):.2f}  down_vol=${feat_dict.get('down_volume_usdc',0):.2f}")
        print(f"  late_up_ratio={feat_dict.get('late_up_ratio',0):.3f}")

        if edge >= 0.15:
            print(f"  *** SIGNAL: BUY {direction} @ market price ***")
        else:
            print(f"  (No trade — edge {edge*100:.1f}% < 15% threshold)")

print("\nScan complete.")
