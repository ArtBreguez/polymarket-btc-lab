# 02 — Feature Engineering

> Last updated: 2026-06-04 (v18 champion)
> Owner: ML team

---

## Table of Contents

1. [Feature Catalog](#1-feature-catalog)
2. [Feature Selection](#2-feature-selection)
3. [Feature Hall of Fame](#3-feature-hall-of-fame)
4. [Feature Graveyard](#4-feature-graveyard)
5. [Feature Parity: Training vs Live Trader](#5-feature-parity-training-vs-live-trader)
6. [Sanity Check Probes](#6-sanity-check-probes)

---

## 1. Feature Catalog

All features are computed per 5-minute slot from CLOB tick data observed during [0, 180s) and Binance spot candles. v18 computes 56+ raw features, then selects the top 30 by LightGBM importance.

### 1.1 CLOB Flow Features

Core order-flow signals from the Polymarket CLOB during the observation window.

| Feature | Formula | Neutral | Range | Since |
|---------|---------|---------|-------|-------|
| btc_up_ratio | sum(size_usdc where outcome=Up) / sum(size_usdc) | 0.5 | [0, 1] | v4 |
| btc_n_ticks | count of ticks in [0, 180s) | 100.0 | [0, inf) | v4 |
| btc_buy_ratio | sum(size_usdc where side=BUY) / sum(size_usdc) | 0.5 | [0, 1] | v6 |
| btc_tw_up_ratio | exp(-0.02*(180-t)) time-weighted up_ratio | 0.5 | [0, 1] | v7 |
| btc_vwap_up | sum(price*size_usdc) / sum(size_usdc) for Up ticks | 0.5 | [0, 1] | v8 |
| btc_vwap_dn | sum(price*size_usdc) / sum(size_usdc) for Down ticks | 0.5 | [0, 1] | v8 |
| btc_vwap_spread | vwap_up - vwap_dn | 0.0 | [-1, 1] | v8 |
| btc_vwap_trend | vwap_up - 0.5 | 0.0 | [-0.5, 0.5] | v7 |
| btc_momentum | mean(w3,w4,w5) - mean(w0,w1,w2) | 0.0 | [-1, 1] | v6 |
| btc_size_disparity | avg_trade_size(Up) - avg_trade_size(Down) | 0.0 | (-inf, inf) | v9 |
| btc_up_ratio_stability | std([w0..w5]) | 0.0 | [0, 0.5] | v9 |
| btc_vol_accel | vol_last_90s / vol_first_90s | 1.0 | [0, inf) | v9 |
| btc_tick_accel | (ticks_last_30s - ticks_first_30s) / ticks_first_30s | 0.0 | (-1, inf) | v8 |
| btc_vwmom | dot(vol_per_window / total_vol, up_ratio_per_window - 0.5) | 0.0 | [-0.5, 0.5] | v8 |

### 1.2 Sub-Window Features

The 180s observation window is split into 6 consecutive 30-second windows.

| Feature | Formula | Neutral | Range | Since |
|---------|---------|---------|-------|-------|
| btc_up_w0 | up_ratio in [0, 30s) | 0.5 | [0, 1] | v8 |
| btc_up_w1 | up_ratio in [30, 60s) | 0.5 | [0, 1] | v8 |
| btc_up_w2 | up_ratio in [60, 90s) | 0.5 | [0, 1] | v8 |
| btc_up_w3 | up_ratio in [90, 120s) | 0.5 | [0, 1] | v8 |
| btc_up_w4 | up_ratio in [120, 150s) | 0.5 | [0, 1] | v8 |
| btc_up_w5 | up_ratio in [150, 180s) | 0.5 | [0, 1] | v8 |

### 1.3 Z-Score Features

Cross-slot standardization using a lookback ring buffer of recent slots.

| Feature | Formula | Neutral | Range | Since |
|---------|---------|---------|-------|-------|
| btc_up_ratio_zscore_5s | (up_ratio - mean_5) / (std_5 + 1e-6) | 0.0 | (-inf, inf) | v8 |
| btc_up_ratio_zscore_10s | (up_ratio - mean_10) / (std_10 + 1e-6) | 0.0 | (-inf, inf) | v8 |
| btc_up_ratio_zscore_20s | (up_ratio - mean_20) / (std_20 + 1e-6) | 0.0 | (-inf, inf) | v8 |
| btc_up_w5_zscore | (up_w5 - mean_20_ur) / (std_20_ur + 1e-6) | 0.0 | (-inf, inf) | v8 |
| btc_up_w0_zscore .. w4_zscore | (up_wN - mean_20_wN) / (std_20_wN + 1e-6) | 0.0 | (-inf, inf) | v8 |

Note on btc_up_w5_zscore: Training uses the overall up_ratio mean/std from 20-slot lookback (mu20, sd20) — NOT per-window w5 stats. This was a parity bug; see section 5.

### 1.4 Spot (Binance) Features

BTC spot price context from Binance 1m candles. Pre-slot returns use the price at observation end (slot_ts + 180s) as the numerator, and the price at (slot_ts - window) as the denominator.

| Feature | Formula | Neutral | Range | Since |
|---------|---------|---------|-------|-------|
| btc_inslot_ret | px[slot_ts+180s] / px[slot_ts] - 1 | 0.0 | (-0.05, 0.05) | v8 |
| btc_pre_5m_ret | px[obs_end] / px[slot_ts-300] - 1 | 0.0 | (-0.05, 0.05) | v8 |
| btc_pre_30m_ret | px[obs_end] / px[slot_ts-1800] - 1 | 0.0 | (-0.1, 0.1) | v8 |
| btc_pre_1h_ret | px[obs_end] / px[slot_ts-3600] - 1 | 0.0 | (-0.15, 0.15) | v8 |
| btc_pre_4h_ret | px[obs_end] / px[slot_ts-14400] - 1 | 0.0 | (-0.2, 0.2) | v8 |
| btc_pre_1h_4h_ratio | (px_now - px_1h_ago) / (px_now - px_4h_ago + 1e-9) | 0.0 | (-inf, inf) | v10 |
| btc_dist_1k | min(frac - floor(frac), ceil(frac) - frac) where frac=px/1000 | 0.25 | [0, 0.5] | v8 |

### 1.5 Lag Features

Previous market outcomes and flow metrics. Use the unified 22k market timeline for lookups.

| Feature | Formula | Neutral | Range | Since |
|---------|---------|---------|-------|-------|
| lag_1_outcome .. lag_5_outcome | target of N-th previous slot | 0.5 | {0, 0.5, 1} | v6 |
| lag_streak | consecutive same-direction outcomes before this slot | 0.0 | [0, inf) | v6 |
| prev_slot_up_ratio_1 .. _5 | up_ratio of N-th previous slot | 0.5 | [0, 1] | v6 |
| prev_slot_n_ticks_1 .. _5 | tick count of N-th previous slot | 0.0 | [0, inf) | v17 |
| prev_slot_vol_1 .. _5 | total volume of N-th previous slot | 0.0 | [0, inf) | v17 |

Time-gap guard: if the gap between current slot and lag slot exceeds lag_n * 300 * 3 seconds, fill with neutral values (0.5 for ratios, 0.0 for counts).

### 1.6 Temporal Features

Cyclical time encoding to capture intraday and day-of-week patterns.

| Feature | Formula | Neutral | Range | Since |
|---------|---------|---------|-------|-------|
| hour_sin | sin(2pi * hour / 24) | 0.0 | [-1, 1] | v8 |
| hour_cos | cos(2pi * hour / 24) | 0.0 | [-1, 1] | v8 |
| dow_sin | sin(2pi * weekday / 7) | 0.0 | [-1, 1] | v8 |
| dow_cos | cos(2pi * weekday / 7) | 0.0 | [-1, 1] | v8 |

### 1.7 Interaction Features

Compound features that combine two signals. Survived permutation importance pruning.

| Feature | Formula | Neutral | Range | Since |
|---------|---------|---------|-------|-------|
| btc_signal_conviction | up_ratio * (1 - stability) | 0.0 | [0, 1] | v10 |
| btc_momentum_vol_sync | momentum * vol_accel | 0.0 | (-inf, inf) | v10 |
| hour_x_up_ratio | up_ratio * (hour / 24) | 0.0 | [0, 1] | v17 |
| hour_x_tw_ur | tw_up_ratio * (hour / 24) | 0.0 | [0, 1] | v17 |

---

## 2. Feature Selection

After computing all 56+ features, LightGBM mean importance across 5 walk-forward folds is used to select the top 30.

Process:
1. Train a screening LightGBM (n_estimators=300, lr=0.05, max_depth=4) on each fold
2. Accumulate feature_importances_ (split-based) across folds
3. Average importance across folds
4. Sort descending, take top TOP_N_FEATS=30

This reduces the feature-to-sample ratio from ~56:22k (1:400) to 30:22k (1:740), keeping the model well-regularized. In the v8-v12 era with only 601 samples, pruning from 63 to ~30 features was critical — the ratio went from 9.5:1 (overfitting) to 22:1 (stable).

---

## 3. Feature Hall of Fame

Top 10 features by LightGBM importance in v18 (22,319 samples):

| Rank | Feature | Notes |
|------|---------|-------|
| 1 | btc_inslot_ret | In-slot BTC spot return. Was buried with 7k samples, 22k reveals it as #1. |
| 2 | btc_vwap_up | Up token VWAP — pricing signal for bullish sentiment. |
| 3 | btc_pre_5m_ret | 5-minute pre-slot BTC spot momentum. |
| 4 | btc_vwap_dn | Down token VWAP — bearish pricing. |
| 5 | btc_up_w1 | 2nd 30s window — early commitment signal. |
| 6 | btc_pre_30m_ret | 30-minute pre-slot trend. Only multi-horizon spot that consistently survives. |
| 7 | btc_vwap_spread | VWAP up - VWAP down. Directional pricing gap. |
| 8 | btc_momentum | Late windows minus early windows. |
| 9 | btc_pre_1h_ret | 1-hour pre-slot momentum. |
| 10 | prev_slot_up_ratio_1 | Previous slot's up ratio — autocorrelation in flow direction. |

Key observation: With 3x more data (v18 vs v17), btc_inslot_ret jumped from outside top-10 to #1. Spot price during the observation window is the single strongest predictor when you have enough data to overcome the noise.

---

## 4. Feature Graveyard

Features tested and consistently dropped by importance pruning:

| Feature | Versions Tried | Why It Fails |
|---------|---------------|--------------|
| ETH/SOL cross-asset features | v4 | Correlated but adds noise, not signal. BTC-only is cleaner. |
| OB Down token (bid/ask/depth) | v5-v7 | best_bid_size always 0 for resolved markets — pure noise. |
| btc_price_percentile | v11 | 4h price regime irrelevant at 5-min scale. |
| btc_final_burst | v11 | Redundant with btc_up_w5 and btc_up_w5_zscore. |
| btc_realized_vol_5s / _10s | v7-v8 | Vol clustering not predictive at slot level. |
| Most lag outcomes (lag_2..5) | v6+ | Only lag_1 and lag_streak survive; lag_2/3/4/5 mostly noise. |
| btc_vol_zscore | v8 | Volume anomaly dropped; up_ratio zscore much stronger. |
| btc_inslot_vol | v8 | Within-slot spot volatility is noise. |
| Inslot ETH/SOL spot | v4-v5 | Removed in v5, never came back. |

---

## 5. Feature Parity: Training vs Live Trader

CRITICAL SECTION. The model is trained in train_v18_modal.py and served in deploy/live_trader.py. Any mismatch between how features are computed in training vs live causes train-serve skew — the model receives inputs from a different distribution than it learned.

### 5.1 The 6 Parity Bugs Found 2026-06-04

These were identified during a systematic audit comparing train_v18_modal.py and live_trader.py line by line.

#### a) btc_size_disparity: division vs subtraction

| | Training (v18) | Live Trader (pre-fix) |
|--|----------------|----------------------|
| Formula | avg_up - avg_dn (subtraction) | avg_up / (avg_dn + 1e-8) (division) |
| Range | (-inf, inf), neutral=0.0 | (0, inf), neutral=1.0 |

Impact: The model learned subtraction semantics (positive = Up trades are larger). Division gives a ratio that is always positive and centered on 1.0, not 0.0. The model sees a completely different distribution at inference.

Fix: Live trader changed to subtraction to match training.

#### b) btc_buy_ratio: count-based vs dollar-weighted

| | Training (v18) | Live Trader (pre-fix) |
|--|----------------|----------------------|
| Formula | sum(size_usdc where side=BUY) / total_volume | count(side=BUY) / count(all) |
| Scale | Dollar-weighted | Count-weighted |

Impact: A single $10k BUY trade among 100 small SELLs gives ~0.9 in training but 0.01 live. Very different signal.

Fix: Live trader changed to dollar-weighted (sum size_usdc) to match training.

#### c) btc_dist_1k: floor vs nearest distance

| | Training (v18) | Live Trader (pre-fix) |
|--|----------------|----------------------|
| Formula | min(frac - floor(frac), ceil(frac) - frac) | frac - floor(frac) |
| Example | px=103,700 -> 0.3 (distance to nearest 1k) | px=103,700 -> 0.7 (distance from floor only) |

Impact: floor-only gives distance from below, so values near a round number from above (e.g., $104,000 -> 103,950 = 0.95) look far away. Nearest-distance correctly captures proximity from either side.

Fix: Live trader changed to min(frac - floor, ceil - frac).

#### d) btc_pre_*_ret: slot-start vs obs-end price reference

| | Training (v18) | Live Trader (pre-fix) |
|--|----------------|----------------------|
| Numerator price | px at obs_end (slot_ts + 180s) | px at slot_ts (slot start) |
| Formula | px[slot_ts + 180] / px[slot_ts - window] - 1 | px[slot_ts] / px[slot_ts - window] - 1 |

Impact: Training sees 3 extra minutes of price movement in the numerator. With BTC moving ~0.05% per 3 minutes, this shifts the feature by a small but systematic amount. For btc_pre_5m_ret (the #3 feature), the 3-min difference is 60% of the lookback window — huge skew.

Fix: Live trader changed to use px at obs_end (slot_ts + OBSERVE_SECS).

#### e) btc_pre_1h_4h_ratio: wrong price reference

| | Training (v18) | Live Trader (pre-fix) |
|--|----------------|----------------------|
| px_now | px at obs_end (slot_ts + 180s) | px at slot_ts |
| px_1h_ago | spot_at(slot_ts - 3600) | spot_at(slot_ts - 3600) |
| px_4h_ago | spot_at(slot_ts - 14400) | spot_at(slot_ts - 14400) |

Impact: Same issue as (d) but for a ratio feature. The numerator and denominator both use px_now, so the error doesn't cancel — it shifts the ratio's centering.

Fix: Live trader changed to use px at obs_end.

#### f) btc_up_w5_zscore: per-window vs overall up_ratio stats

| | Training (v18) | Live Trader (pre-fix) |
|--|----------------|----------------------|
| Mean/std source | mu20, sd20 from up_ratio history (overall) | mean/std of w5 values from last 20 slots (per-window) |
| Semantics | "How unusual is w5 relative to overall flow?" | "How unusual is w5 relative to recent w5 values?" |

Impact: up_ratio has a tighter distribution than individual sub-window values. Using overall stats makes the zscore more sensitive — a w5 of 0.7 might be z=1.5 with overall stats but z=0.8 with per-window stats. btc_up_w5_zscore was historically the #2 most important feature.

Fix: Live trader changed to use overall up_ratio mean/std from 20-slot lookback (matching training).

### 5.2 Parity Verification Checklist

When adding a new feature or version:
1. Write the formula ONCE, share between train and live (or copy exactly)
2. Print feature values for 3 test slots from both train and live code
3. Check neutral fill values match (see section 6)
4. Check mathematical operations (division vs subtraction, floor vs min)
5. Check price reference points (slot_ts vs slot_ts + OBS_SECS)
6. Check lookback source (overall vs per-window, dollar vs count)

---

## 6. Sanity Check Probes

The training script includes a 3-probe sanity check after final model training. Each probe constructs a synthetic feature vector with known properties and checks that predict_proba returns sensible values.

### Neutral Fill Values by Feature Type

| Type | Features | Neutral Value | Rationale |
|------|----------|---------------|-----------|
| Return features | *_ret, momentum, vwap_spread, disparity, zscore, sin, cos, streak | 0.0 | No directional signal |
| Ratio features | up_ratio, vwap_up, vwap_dn, buy_ratio, tw_up_ratio, up_w0..w5 | 0.5 | Equal weight both sides |
| Distance features | dist_1k | 0.25 | Midpoint of [0, 0.5] |
| Volume features | dollar_vol | 5000.0 | Typical slot volume |
| Count features | n_ticks | 100.0 | Typical tick count |

### Probe Definitions

UP probe (mildly bullish):
  btc_up_ratio=0.75, btc_tw_up_ratio=0.75, btc_vwap_up=0.55, btc_vwap_dn=0.45,
  btc_vwap_spread=0.10, btc_momentum=0.05, btc_inslot_ret=0.001,
  btc_pre_5m_ret=0.0005, btc_signal_conviction=0.7, btc_buy_ratio=0.6,
  all btc_up_w* = 0.65

DOWN probe (mirror of UP):
  btc_up_ratio=0.25, btc_tw_up_ratio=0.25, btc_vwap_up=0.45, btc_vwap_dn=0.55,
  btc_vwap_spread=-0.10, btc_momentum=-0.05, btc_inslot_ret=-0.001,
  btc_pre_5m_ret=-0.0005, btc_signal_conviction=0.7, btc_buy_ratio=0.4,
  all btc_up_w* = 0.35

NEUTRAL probe: all features at neutral values.

Expected: P(UP | UP_probe) > P(UP | NEUTRAL_probe) > P(UP | DOWN_probe)

If this assertion fails, investigate:
- Feature sign inversions
- Calibration artifacts (isotonic can be non-monotonic with few training points)
- Feature parity bugs distorting the learned mapping
