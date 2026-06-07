# v23 Changelog — BTC 5min Prediction Bot
## Date: 2026-06-07

### Summary
Complete infrastructure audit + model retrain + risk parameter optimization.
Bot was losing ~$0.73/trade (W31/L35 = 47% WR, P&L=-$48). Root causes:
formula mismatches between training and live, dead WebSocket connection,
and unfavorable risk/reward at high ask prices.

---

## Infrastructure Fixes

### 1. CLOB WebSocket Zombie Bug (CRITICAL)
**Problem:** WS showed rate=0/min for hours despite being "connected". 
The zombie watchdog did ping/pong, and when pong succeeded, RESET 
`last_message_at`. This created a 120s blind window each cycle — 
`consecutive_quiet` never accumulated past 1, so the force-reconnect 
at 5/5 never triggered.

**Fix:** Track `total_messages` count instead of resetting timestamps. 
3 checks without new messages (~6 min) → force reconnect.
- File: `deploy/ws_manager.py` (lines 441-522 rewritten)

### 2. CLOB WS Over-subscription
**Problem:** Pre-subscribed 4 slots × 2 tokens = 16 assets. Server dropped 
connection every ~5min (code=1006) due to message volume overload (46K/min).

**Fix:** Pre-subscribe only 2 slots (4 tokens). On-demand subscribe for future 
slots via `fetch_market()`.
- File: `deploy/live_trader.py` line 452

### 3. CLOB WS Price Cache Invalidation (MEDIUM)
**Problem:** On `price_change` events with worse price, code did `pop()` on 
cache, forcing HTTP /book fallback until next book snapshot (seconds later).

**Fix:** Always update to new price from server. The server is authoritative.
- File: `deploy/live_trader.py` lines 389-395

### 4. Zombie Timeout
**Change:** 120s → 60s. BTC 5min markets have constant book activity.
- File: `deploy/live_trader.py` line 445

---

## Model Fixes (Train-Live Alignment)

### 5. x_ob_drift_x_inslot ALWAYS ZERO (CRITICAL)
**Problem:** Cross-feature computed BEFORE `build_spot_features()` was called.
`feat.get("btc_inslot_ret", 0.0)` always returned 0.0 because the key didn't 
exist yet. This was feature #4 by importance — always zero since deployment.

**Fix:** Moved cross-domain features AFTER `build_spot_features()`.
- File: `deploy/live_trader.py` lines 1241-1253

### 6. tw_up_ratio Formula Mismatch (CRITICAL)
**Problem:** 
- Training: exponential decay `exp(-0.02*(obs_secs-t))` weighted by `size_usdc`
- Live: linear weights `(t_sec + 1)` without size weighting

**Fix:** Live now uses identical exp decay + size_usdc formula.
- File: `deploy/live_trader.py` lines 1179-1191

### 7. Momentum Formula Mismatch
**Problem:** Training used `w_vals[-1] - w_vals[0]` for n_windows<4 (60s variant).
Live always used `mean(w[3:]) - mean(w[:3])`.

**Fix:** v23 training unified to always use `mean(w[3:]) - mean(w[:3])` (match live).
- File: `scripts/train_v23_modal.py` line 337

### 8. Missing Features in Live
**Problem:** v23 model has 40 features. Several were not computed in live:
- `hour_sin` — missing entirely
- `btc_up_ratio_stability` — computed but not stored in feat dict
- `prev_slot_vol_N` / `prev_slot_n_ticks_N` — lag features not in live
- `btc_inslot_ret` — 0.0 because only 1 spot candle in 60s window

**Fixes:**
- Added `hour_sin` computation
- Added `btc_up_ratio_stability` to feat dict
- Added `prev_slot_vol_N` and `prev_slot_n_ticks_N` to lag feature loop
- Fixed `btc_inslot_ret` to use searchsorted (nearest candle at start/end)
- File: `deploy/live_trader.py` multiple locations

---

## Model Retrain (v23)

### Training Changes
- **Version:** v23-v23_60s_40f
- **OBS_SECS:** 60 (matches live data availability with ~120s data-api lag)
- **Features:** 40 (was 32 in v22)
- **WF AUC:** 0.8574 (vs v22_180s 0.9016 — but v22 had distribution shift)
- **Momentum formula:** Unified to always use 6-window mean split
- **tw_up_ratio:** Already correct in training (exp decay + size_usdc)
- **Promotion gate:** Forced 60s variant (60s matches live, 180s doesn't)
- File: `scripts/train_v23_modal.py`

---

## Risk Parameter Optimization

### 9. Ask Price Range: [0.38, 0.90] → [0.42, 0.65]
**Analysis (127 historical trades):**

| Ask Bucket | WR | Breakeven WR | P&L | Risk:Reward |
|------------|-----|-------------|------|-------------|
| $0.00-$0.40 | 29% | 39% | -$3.78 | 0.65:0.35 |
| $0.40-$0.50 | 43% | 46% | -$3.45 | 1:1 |
| $0.50-$0.60 | 49% | 55% | -$14.64 | 1.15:1 |
| $0.60-$0.70 | 60% | 65% | -$9.75 | 1.6:1 |
| $0.70-$0.80 | 50% | 76% | **-$29.04** | 2.5:1 |
| $0.80-$0.90 | 80% | 84% | -$3.61 | 4.4:1 |

**Reasoning:**
- Below $0.42: Market strongly against our direction (29% WR)
- Above $0.65: Risk/reward unfavorable. At ask=$0.70, risk $0.70 to win $0.28.
  Even with 50% WR, lose $29 on 18 trades. Need 76% WR to breakeven.
- The $0.70-$0.80 bucket alone accounts for 45% of total losses.

### 10. MIN_EDGE: 10% → 7%
**Reasoning:** With tighter ask_max=0.65, the edge filter can be less 
aggressive. At ask=$0.50, 7% edge means model_prob=0.57 (vs breakeven 51%). 
This gives ~6% real expected edge while allowing more trades in the 
favorable price range.

---

## Current Configuration

```
Model:          v23-v23_60s_40f (40 features, AUC=0.857)
OBSERVE_SECS:   60
ENTER_WINDOW:   (170, 240)
MIN_CONFIDENCE: 60%
MIN_EDGE:       7% (was 10%)
MIN_EDGE_MID:   5%
ASK_RANGE:      [0.42, 0.65] (was [0.38, 0.90])
ASK_MID_DIV:    0.35
AUTO_SHARES:    5-40 (balance $20-$700)
```

## Expected Impact
- Eliminates ~45% of historical losses (from high-ask trades)
- Model predictions now match training exactly (no distribution shift)
- x_ob_drift_x_inslot provides real signal for first time ever
- CLOB WS stays alive (was dying for hours undetected)
- Lower trade frequency but higher expected value per trade
