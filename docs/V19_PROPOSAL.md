# V19 Feature Proposal — L2 Orderbook & Data-Driven Improvements

## Executive Summary

v18 achieves AUC=0.8966 with 30 features using only `last_trade_price` events (908/slot).
The poly_l2 data contains **2 massive untapped data sources**:
- `book` events: ~2,700 full orderbook snapshots per slot (45-55 ask levels, 48+ bid levels)
- `price_change` events: ~68,000 individual fills per slot with best_ask/best_bid

This is 70x more data per slot than we currently use. The orderbook structure
captures market maker positioning, liquidity dynamics, and institutional flow
that trade-level data cannot see.

---

## Data Available (from poly_l2 inspection)

| Event Type | Count/Slot | Fields | Currently Used |
|-----------|-----------|--------|---------------|
| book | ~2,700 | ask_prices[45], ask_sizes[45], bid_prices[48], bid_sizes[48] | NO |
| price_change | ~68,000 | best_ask, best_bid, pc_price, pc_size, pc_side | NO |
| last_trade_price | ~900 | trade_price, trade_size, trade_side | YES (only this) |

Book snapshots provide full depth ladder every ~4 seconds. Price_change events
are individual order fills showing real-time best bid/ask evolution.

---

## Proposed V19 Features

### Category 1: Orderbook Depth Features (from `book` events)
**Expected impact: +0.003-0.008 AUC** (new signal dimension, comparable to v17→v18 data expansion)

#### 1a. Depth Imbalance (3 features)
```
ob_tob_imbalance = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size)
ob_depth_imbalance_5c = (bid_depth_within_5c - ask_depth_within_5c) / total_depth
ob_depth_imbalance_full = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth)
```
**Rationale:** Depth imbalance at different levels reveals where liquidity is positioned.
Top-of-book imbalance was 0.25 → 0.36 → -0.50 → -0.36 in the sample slot, showing
massive directional information as the market evolved. v11 tried this but OB Down token
had best_bid_size=0 for resolved markets — the issue was data quality, not signal quality.
The **Up token** orderbook should have real depth.

#### 1b. Spread Dynamics (3 features)
```
ob_spread_open = ask_prices[0] - bid_prices[0]  (first snapshot)
ob_spread_close = ask_prices[0] - bid_prices[0]  (last snapshot in obs window)
ob_spread_volatility = std(spread across all snapshots)
```
**Rationale:** Spread widening signals uncertainty; narrowing signals consensus.
Sample data shows spread going from 0.01 to 0.04 — 4x widening during the slot.

#### 1c. Book Pressure (3 features)
```
ob_bid_wall = max(bid_sizes) / mean(bid_sizes)  # large resting orders
ob_ask_wall = max(ask_sizes) / mean(ask_sizes)
ob_wall_ratio = ob_bid_wall / ob_ask_wall  # which side has bigger walls
```
**Rationale:** Large resting orders ("walls") indicate institutional positioning.
In the sample, bid sizes range from 5 to 22,000+ at the 0.01 level — extreme
concentration at tail levels.

#### 1d. Depth Evolution (4 features)
```
ob_mid_drift = mid_close - mid_open  # midpoint movement during observation
ob_depth_trend = (total_depth_close - total_depth_open) / total_depth_open  # liquidity change
ob_imbalance_momentum = imb_close - imb_open  # imbalance trend
ob_depth_concentration = depth_top3_levels / total_depth  # how concentrated
```
**Rationale:** The sample shows dramatic evolution: total depth went from ~93k to
~58k (37% reduction), and the bid/ask ratio flipped from bid-heavy to ask-heavy.
This directional liquidity withdrawal is a strong signal.

### Category 2: Price Change (Fill) Features (from `price_change` events)
**Expected impact: +0.002-0.005 AUC**

#### 2a. Fill Flow Analysis (3 features)
```
pc_buy_volume = sum(pc_size where pc_side == 'BUY')
pc_sell_volume = sum(pc_size where pc_side == 'SELL')
pc_flow_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)
```
**Rationale:** 68k fills vs 900 trades — 75x more granular order flow data.
The price_change events show every individual fill, not just the aggregated
trade result. This captures partial fills and iceberg orders.

#### 2b. Fill Aggressiveness (3 features)
```
pc_avg_fill_size = mean(pc_size)
pc_large_fill_ratio = count(pc_size > 2*median) / total_fills  # large fills
pc_fill_rate = total_fills / observation_seconds  # fills per second
```
**Rationale:** Large fills indicate informed trading. Fill rate acceleration
signals urgency. These are microstructure signals invisible in trade-level data.

#### 2c. Best Bid/Ask Dynamics (3 features)
```
pc_bbo_spread_mean = mean(best_ask - best_bid)  # average spread from fills
pc_bbo_mid_trend = linear_slope(midpoint over time)  # directional BBO movement
pc_bbo_cross_count = count(best_bid[t] >= best_ask[t-1])  # book crossing events
```
**Rationale:** The best_ask and best_bid from 68k events give millisecond-resolution
BBO tracking. Midpoint trend is a proxy for informed order flow pressure.

### Category 3: Temporal Orderbook Features (windowed, like btc_up_w0..w5)
**Expected impact: +0.002-0.004 AUC** (temporal structure proven valuable for tick data)

#### 3a. Windowed OB Imbalance (6 features)
```
ob_imb_w0..w5 = depth_imbalance computed per 30s window
```
**Rationale:** btc_up_w0..w5 are Hall of Fame features. Applying the same
30s window structure to orderbook imbalance captures how the book evolves
through the slot. The final window (ob_imb_w5) should be especially predictive,
paralleling btc_up_w5 being #1 feature.

#### 3b. OB Momentum (1 feature)
```
ob_momentum = mean(ob_imb_w3, w4, w5) - mean(ob_imb_w0, w1, w2)
```
**Rationale:** Same pattern as btc_momentum (Hall of Fame, #8 feature).

### Category 4: Feature Interactions (Not Yet Tried)
**Expected impact: +0.001-0.003 AUC** (interactions worked well in v10: +0.002)

#### 4a. Cross-Domain Interactions
```
ob_imb_x_inslot_ret = ob_depth_imbalance × btc_inslot_ret
  # OB agrees with spot → strong conviction
ob_imb_x_up_ratio = ob_depth_imbalance × btc_up_ratio
  # OB agrees with CLOB flow → double confirmation
ob_spread_x_momentum = ob_spread_close × btc_momentum
  # Wide spread + strong momentum = informed trading
```
**Rationale:** btc_signal_conviction (v10, interaction: up_ratio × stability)
made top-15 features. Cross-domain interactions between orderbook structure
and trade flow / spot price create entirely new signal dimensions.

#### 4b. Spot × OB Interactions
```
inslot_ret_x_hour = btc_inslot_ret × hour_sin
  # BTC moves differently in Asian vs US hours
pre_5m_ret_x_ob_imb = btc_pre_5m_ret × ob_depth_imbalance
  # Spot trend confirmed by orderbook positioning
```

### Category 5: Additional Data-Driven Improvements

#### 5a. Market Age Feature (1 feature)
```
market_age_hours = (slot_ts - min_slot_ts) / 3600
```
**Rationale:** The fold AUC trend (0.88→0.91) shows the model performs better
on later markets, likely due to increased liquidity and market maturity.
A market age proxy could help the model learn regime differences.

#### 5b. Data Expansion (ongoing)
- pmdata.dev has data from ~Feb 15 2026 onward and grows daily
- v18 uses Mar-Jun 2026 (22k markets). Adding Feb-Mar could give 25-30k markets
- v17→v18 showed +0.004 AUC from 3x more data
- Expected: +0.001-0.002 from additional 10-20% data

#### 5c. Feature Count Management
- v18 uses 30 features. Adding ~25 new OB features would give 55+ candidates
- Apply existing permutation importance pruning (TOP_N_FEATS=30)
- The pruning should aggressively cut weak OB features, keeping only the ~8-10 strongest
- Keep feature/sample ratio above 15:1 (22k samples / 30 features = 733:1, plenty of headroom)

---

## Implementation Priority

| Priority | Feature Group | New Features | Expected AUC Impact | Implementation Complexity |
|----------|-------------|-------------|--------------------|----|
| 1 (HIGH) | Depth Imbalance (1a) | 3 | +0.002-0.004 | Low — straightforward array ops |
| 2 (HIGH) | Fill Flow (2a) | 3 | +0.002-0.003 | Low — sum/filter on price_change |
| 3 (MED) | Depth Evolution (1d) | 4 | +0.002-0.003 | Medium — need open/close snapshots |
| 4 (MED) | Windowed OB (3a-b) | 7 | +0.002-0.004 | Medium — 30s window structure |
| 5 (MED) | Cross-Domain Interactions (4a) | 3 | +0.001-0.003 | Low — multiply existing features |
| 6 (LOW) | Spread Dynamics (1b) | 3 | +0.001-0.002 | Low |
| 7 (LOW) | Book Pressure (1c) | 3 | +0.001-0.002 | Low |
| 8 (LOW) | Fill Aggressiveness (2b) | 3 | +0.001 | Low |
| 9 (LOW) | BBO Dynamics (2c) | 3 | +0.001 | Medium |

**Total candidate features: ~32 new**
**After pruning: expect ~8-12 to survive**
**Realistic total AUC improvement: +0.005-0.015**

---

## Implementation Notes

### Data Pipeline Changes
1. **Fetch both `book` and `price_change` events** from poly_l2 (currently only `last_trade_price`)
2. Pre-compute OB features per slot and save to a new parquet:
   - `/ob_features_full.parquet` — one row per market_id with all OB features
3. In training, merge OB features with existing tick features by market_id

### Key Risk: OB Down Token Issue (from Feature Graveyard)
v7 discovered `best_bid_size` is always 0 for **Down token** resolved markets.
The v11 code (which I inspected) already handles this by computing features from
whichever token has real depth (Up token). Ensure v19 does the same:
- Filter book events to the Up token only, OR
- Use the market_slug directly (btc-updown-5m-{slot_ts} returns Up token data)

### Live Trading Compatibility
The live_trader.py already has OB feature computation code (lines 656, 948) that
fetches from CLOB REST /book endpoint. The poly_l2 book events have identical
structure (ask_prices/sizes, bid_prices/sizes arrays), so training/inference
feature parity is straightforward.

### Computational Cost
- v18 training: ~45min on Modal (8 CPU, 32GB)
- v19 with OB features: estimate ~60-75min (additional parquet downloads + array ops)
- Optuna 150 trials: feature count unchanged (still 30 after pruning), so trial time similar

---

## Experiment History Pattern Analysis

| Version Jump | Change Type | AUC Delta | Lesson |
|-------------|------------|----------|--------|
| v5→v6 | Lag features | +0.001 | Small but real |
| v6→v10 | Interactions + calibration fix | +0.003 | Interactions work |
| v10→v17 | Extended lag context + temporal | +0.038 | More data = transformative |
| v17→v18 | 3x data expansion | +0.004 | More data (diminishing returns) |
| v18→v19 (est.) | L2 orderbook features | +0.005-0.015 | New signal dimension |

The biggest gains came from new data dimensions (v10→v17: +0.038, v17→v18: +0.004).
L2 orderbook data is the biggest untapped data dimension available — it's an entirely
new view of market microstructure that trade-level data cannot capture. This is
analogous to the v10→v17 jump where extended lag context added a new signal dimension.

**Conservative estimate: AUC 0.900-0.905**
**Optimistic estimate: AUC 0.905-0.912**
