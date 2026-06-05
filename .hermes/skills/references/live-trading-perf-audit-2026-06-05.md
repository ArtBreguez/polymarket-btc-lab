# Live Trading Performance Audit — 2026-06-05

## Latency Bottlenecks (pre-fix)

### Critical Path (worst-case per iteration):
| Step | Time (typical) | Time (worst) | Notes |
|------|---------------|--------------|-------|
| fetch_inslot_trades | 2-6s | 48s | Sequential yes+no, up to 16 HTTP calls |
| build_features + OB fetch | 50-100ms | 5s | OB fetch = 1 CLOB /book call (5s timeout) |
| get_ask_price | 0ms (WS cache) | 19.5s | 3 retries × (5s timeout + backoff) |
| get_market_mid | ~1s | 15s | 1 Gamma + 2 CLOB /book calls |
| Order + fill polling | 5-10s | 35s | 6 polls × 5s sleep = 30s + cancel |
| **Total worst-case** | **~10s** | **~122s** | **Exceeds 70s entry window!** |

### Fixes Applied:
1. **Parallel trade fetch**: ThreadPoolExecutor(2) for yes+no tokens → ~50% reduction (4-6s → 2-3s)
2. **Parallel price fetch**: get_ask_price + get_market_mid in ThreadPoolExecutor(2) → measured 78-113ms (was sequential ~2-10s)
3. **Progressive fill polling**: delays [1,2,3,4,5,5]=20s total (was [5,5,5,5,5,5]=30s). First fill detected 4s sooner on average.

## Data Quality (pre-fix)

### Features always zero (7 of 23 zero features):
| Feature | Root Cause | Fix |
|---------|-----------|-----|
| btc_pre_5m_vol | Hardcoded 0.0 ("not used in v18") | Compute via _ret_vol() on pre-window segment |
| btc_pre_15m_vol | Same | Same |
| btc_pre_30m_vol | Same | Same |
| btc_pre_1h_vol | Same | Same |
| btc_pre_4h_vol | Same | Same |
| ob_depth_change | Hardcoded 0.0 | Compute total_depth(close) - total_depth(open) using cached open snapshot |
| x_fill_imb_x_buy | ob_fill_imbalance never computed | Not fixed (requires fill data not available in live) |

### Result:
- Before: 81/104 features non-zero (23 zero)
- After: 88-89/104 features non-zero (15-16 zero) — **+7-8 features now real**

### Remaining zero features (~15):
- x_fill_imb_x_buy: depends on ob_fill_imbalance (not available live)
- btc_up_w{0-4}_zscore: zero when history < 5 entries per window (warmup)
- btc_realized_vol_5s/10s: zero when history < 3 entries
- prev_slot_n_ticks/vol: zero when no history
- Various conditional zeros depending on market state

### OB temporal features quality issue (not fixed):
- ob_imb_w1 is INTERPOLATED (average of w0 and w2), not measured
- Training used 6×30s OB snapshots; live uses 1-2 snapshots with interpolation
- These were top-5 features in training but are approximate in live
- Fix would require multiple OB polls during observation window (adds latency)

### Data-API lag impact:
- At t=170s (entry window start), only trades from t=0-50s available (~30% of observation)
- At t=240s (end), trades from t=0-120s available (~67% of observation)
- Bot NEVER sees full 180s of observation data due to ~120s data-api lag
- Later iterations in the entry window see more data → better predictions
