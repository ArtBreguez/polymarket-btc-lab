# Feature Parity Bugs — 2026-06-04 Post-Mortem

## Impact
- Live win rate: 42% (worse than coin flip) despite 81% backtest accuracy
- P&L: -$40.75 over 64 trades (27W/37L)
- Root cause: 6 features computed differently in training vs live trader

## Bug 1: btc_size_disparity — Subtraction vs Division
- **Training** (train_v18_modal.py L243): `avg_up - avg_dn` (difference, range ~[-10, +10])
- **Live** (live_trader.py L837): `avg_up / (avg_dn + 1e-8)` (ratio, range [0, inf])
- **Impact**: Completely different distribution. Model trained on differences receives ratios.

## Bug 2: btc_buy_ratio — Dollar vs Count Weighted
- **Training** (L224-225): `buy_dollar_vol / total_dollar_vol`
- **Live** (L848): `count_of_buy_ticks / total_ticks`
- **Impact**: A few large BUY orders dominate dollar ratio but not count ratio.

## Bug 3: btc_dist_1k — Nearest vs Floor Distance
- **Training** (L321): `min(frac, 1-frac)` — distance to NEAREST $1k (range [0, 0.5])
- **Live** (L497): `(price % 1000) / 1000` — fraction above FLOOR (range [0, 1.0])
- **Impact**: At $103,800: training=0.2, live=0.8. Completely wrong for half the range.

## Bug 4: btc_pre_*_ret — Different Price Reference
- **Training** (L293-303): `px_now = spot_at(slot_ts + 180)` — includes 3min inslot movement
- **Live** (L473-491): `segment [slot_ts-window, slot_ts)` — pure pre-slot return
- **Affects**: btc_pre_5m_ret, btc_pre_30m_ret, btc_pre_1h_ret, btc_pre_4h_ret (4 features)
- **Impact**: Training encodes current inslot BTC move (strong signal). Live misses it entirely.

## Bug 5: btc_pre_1h_4h_ratio — Wrong Base Price
- **Training**: Uses `px_now = spot_at(slot_ts + OBS_SECS)` (observation-end price)
- **Live**: Uses `px_now = spot_open` (slot-start price)

## Bug 6: btc_up_w5_zscore — Wrong Reference Distribution
- **Training** (L280): z-score of btc_up_w5 using OVERALL up_ratio mean/std from 20 prior slots
- **Live** (L900-908): z-score of btc_up_w5 using WINDOW 5's OWN historical mean/std
- **Impact**: Different reference distributions → different z-score values.

## Prevention
1. After ANY training feature change, run a feature parity diff between train script and live_trader
2. The DataQualityGate (deploy/data_quality_gate.py) now validates feature ranges at runtime
3. Feature Parity Checklist in SKILL.md must be checked before every deploy
4. Consider extracting shared feature computation into a single module used by both training and live
