# 10 — Roadmap

## v19: L2 Orderbook Features

**Status**: Proposed (see `docs/V19_PROPOSAL.md` for full details)

v18 uses only `last_trade_price` events (~900/slot). The poly_l2 data contains 2 massive untapped sources:
- `book` events: ~2,700 full orderbook snapshots per slot (45+ ask levels, 48+ bid levels)
- `price_change` events: ~68,000 individual fills per slot

This is **70x more data per slot** than currently used.

### Proposed Features

**Depth Imbalance** (3 features):
- `ob_tob_imbalance` — top-of-book bid/ask size ratio
- `ob_depth_imbalance_5c` — depth within 5 cents
- `ob_depth_imbalance_full` — total bid vs ask depth

**Spread Dynamics** (3 features):
- `ob_spread_open`, `ob_spread_close` — spread at start/end of observation
- `ob_spread_volatility` — spread stability during observation

**Book Pressure** (3 features):
- `ob_bid_wall`, `ob_ask_wall` — large resting order detection
- `ob_wall_ratio` — which side has bigger walls (institutional positioning)

**Depth Evolution** (4 features):
- `ob_mid_drift` — midpoint movement during observation
- `ob_depth_trend` — liquidity change (sample showed 37% depth reduction)
- `ob_imbalance_momentum` — imbalance trend direction
- `ob_depth_concentration` — how concentrated depth is at top levels

**Expected impact**: +0.003–0.008 AUC (new signal dimension)

**Key risk**: v11 tried OB features but used DOWN token data (which has `best_bid_size=0`). v19 must use UP token orderbook. The signal quality issue was data quality, not concept quality.

---

## Daily Data Expansion Cron

**Status**: Planned

Currently data is manually fetched and uploaded. Plan:
1. Cron job to fetch new resolved markets daily from pmdata.dev
2. Fetch corresponding Binance spot candles locally
3. Append to `ticks_btc_full_clean.parquet`
4. Upload to Modal Volume

Dataset grows ~280 markets/day. By end of June 2026: ~30k+ markets.

More data has been the single biggest driver of improvement (v17→v18: +0.004 AUC from 3x data alone).

---

## Automated Retraining

**Status**: Planned (depends on daily data expansion)

Pipeline:
1. Daily data expansion cron runs
2. If dataset has grown by >500 markets since last training:
   - Trigger `train_v18_modal.py` (or latest version)
   - Auto-promote if 2/3 metrics beat champion
   - Auto-deploy via `gh workflow run deploy.yml`
3. Alert on regression (new model worse than champion)

Safeguards:
- Gate requires 2/3 metrics improvement (existing)
- Sanity check must pass (existing)
- Human approval for first automated promotion (new)

---

## Regime Detection

**Status**: Research idea

BTC 5-minute markets behave differently in:
- High volatility (>2% daily moves) vs low volatility
- Trending vs ranging markets
- High liquidity (US hours) vs low liquidity (Asian hours)

Approach:
- Classify market regime from pre-slot spot data
- Use regime as a categorical feature or train regime-specific models
- Temporal features (hour_sin/cos) partially capture this already

Risk: With 22k samples, regime segmentation may fragment the data too much. Need 50k+ samples first.

---

## Position Sizing Optimization

**Status**: Research idea

Currently flat $1.50 per trade. Improvements:
- **Kelly criterion**: Size proportional to edge (model_prob - market_price)
- **Confidence scaling**: Larger positions when model confidence is very high (>80%)
- **Volatility adjustment**: Reduce size in high-volatility regimes
- **Bankroll management**: Cap total exposure per time window

Prerequisite: Positive live P&L over sustained period (>100 trades) to validate edge estimate accuracy.

---

## Priority Order

| Priority | Item | Expected Impact | Dependency |
|----------|------|-----------------|------------|
| 1 | Daily data expansion cron | +++ (more data = better model) | Local cron setup |
| 2 | v19 L2 orderbook features | +0.003–0.008 AUC | Feature engineering |
| 3 | Automated retraining | Operational efficiency | Daily data expansion |
| 4 | Position sizing | Better capital efficiency | Proven live edge |
| 5 | Regime detection | Unknown | 50k+ samples |
