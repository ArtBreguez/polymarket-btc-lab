# 10 — Roadmap

## Completed

### v19: L2 Orderbook Features — DONE
Real L2 orderbook features from pmdata poly_l2 data. Added 9 OB features (ob_mid, ob_mid_drift, ob_weighted_imb, ob_imb_w0/w1/w2, ob_imb_momentum, ob_ask_depth_5c, ob_total_depth) + 3 cross-domain interactions. AUC: 0.8979 → 0.9000.

### v21: Ablation Study — DONE (Current Champion)
Pruned from 40 → 30 features. Removed 10 low-importance features (zscore, early windows, noisy lags). AUC maintained at 0.9002, accuracy improved to 81.34%.

### Position Sizing — DONE
Auto-sizing shares based on wallet balance. Linear scaling: 5 shares at $20 → 40 shares at $700. Risk cap: never spend >10% of balance per trade. Configurable via env vars (AUTO_SHARES, FIXED_SHARES, etc.).

### WebSocket Resilience — DONE
ws_manager.py with exponential backoff, active zombie detection, Binance REST fallback. 154 unit tests covering all components.

---

## In Progress

### Daily Data Expansion

**Status**: Blocked (pmdata.dev API key expired)

Currently data is manually fetched and uploaded. Plan:
1. Renew pmdata API key
2. Cron job to fetch new resolved markets daily
3. Fetch corresponding Binance spot candles
4. Append to training dataset
5. Upload to Modal Volume

Dataset grows ~280 markets/day. Target: 30k+ markets for v22.

More data has been the single biggest driver of improvement (v17→v18: +0.004 AUC from 3x data alone).

---

## Planned

### Gate 4: Circuit Breaker

**Status**: Designed, not integrated in live loop

Stop trading when recent win rate drops below 40% over last 20 trades. Already in DataQualityGate code but not wired into the main trading loop.

### Automated Retraining

**Status**: Planned (depends on daily data expansion)

Pipeline:
1. Daily data expansion cron runs
2. If dataset has grown by >500 markets since last training:
   - Trigger `train_v21_modal.py` (or latest version)
   - Auto-promote if 2/3 metrics beat champion
   - Auto-deploy via Fly.io
3. Alert on regression (new model worse than champion)

---

## Research Ideas

### Regime Detection

BTC 5-minute markets behave differently in:
- High volatility (>2% daily moves) vs low volatility
- Trending vs ranging markets
- High liquidity (US hours) vs low liquidity (Asian hours)

Risk: With 22k samples, regime segmentation may fragment the data too much. Need 50k+ samples first.

### Kelly Criterion Sizing

Scale position size proportional to edge (model_prob - market_price) instead of linear balance scaling. Prerequisite: 200+ live trades to validate edge accuracy.

---

## Priority Order

| Priority | Item | Status | Expected Impact |
|----------|------|--------|-----------------|
| 1 | Renew pmdata API key | Blocked | Unlocks dataset expansion |
| 2 | Daily data expansion cron | Planned | +++ (more data = better model) |
| 3 | Gate 4 circuit breaker | Designed | Risk reduction |
| 4 | Automated retraining | Planned | Operational efficiency |
| 5 | Regime detection | Research | Unknown, needs 50k+ samples |
| 6 | Kelly criterion sizing | Research | Better capital efficiency |
