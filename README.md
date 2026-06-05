# Polymarket BTC 5-Min Prediction Lab

ML pipeline for predicting Bitcoin 5-minute Up/Down markets on Polymarket. Trains LightGBM models on CLOB order flow + L2 orderbook + Binance spot data, deploys to Fly.io for live trading.

## Current Champion: v21

| Metric | Value |
|--------|-------|
| Walk-Forward AUC | 0.9002 |
| Walk-Forward Brier | 0.1290 |
| Walk-Forward Accuracy | 81.34% |
| Features | 30 (pruned from 40 via ablation study) |
| Training samples | 22,319 markets |
| Dataset range | Mar 13 - Jun 3, 2026 (83 days) |
| Ticks | 68.3M clean trades |

## Architecture

```
Polymarket CLOB ──┐
                  ├── live_trader.py ──> Predictions ──> Orders
Binance Spot ─────┘        │
                     30 features
                     LightGBM + Isotonic calibration
```

**Data sources (live):**
- CLOB WebSocket: real-time Up/Down trades (order flow)
- CLOB REST API: L2 orderbook snapshots (mid, spread, imbalance, depth)
- Binance REST: BTC/USDT 1-minute candles (spot returns, volatility)

**Data sources (training):**
- Polymarket Data API: historical ticks for 22k+ markets
- pmdata.dev: L2 orderbook snapshots (pre-computed features)
- Binance API: historical 1-minute OHLCV candles

## Features (v21 — 30 features)

| Category | Count | Examples |
|----------|-------|---------|
| CLOB tick flow | 10 | btc_up_ratio, btc_vwap_up/dn, btc_momentum, btc_buy_ratio |
| L2 orderbook | 7 | ob_mid, ob_mid_drift, ob_weighted_imb, ob_imb_w0/w2 |
| Binance spot | 5 | btc_inslot_ret, btc_pre_5m/30m/1h/4h_ret |
| Lag history | 4 | prev_slot_up_ratio_1/2/3/5 |
| Cross-domain | 3 | x_ob_drift_x_inslot, x_depth_x_momentum, x_imb_x_ur |
| Temporal | 1 | hour_x_up_ratio |

See [docs/wiki/02-feature-engineering.md](docs/wiki/02-feature-engineering.md) for full catalog.

## Training Pipeline

1. Load dataset from Modal Volume (22k markets, 68M ticks, OB features, Binance spot)
2. Build 30 features per market (tick aggregation + spot returns + OB snapshots)
3. Optuna hyperparameter search (150 trials, LightGBM)
4. Walk-forward validation (5 time-series folds, gap=5)
5. Isotonic calibration (CalibratedClassifierCV, cv=3)
6. Promotion gate: must beat champion on 2/3 metrics (AUC, Brier, Accuracy)
7. Upload to HuggingFace on promotion

Runs on Modal (8 CPU, 32GB RAM). Training time: ~40 min.

```bash
# Train new version
modal run scripts/train_v21_modal.py
```

## Live Trading

Deployed on Fly.io (Amsterdam, `polymarket-maker-mm`).

**Flow:** Every 5-min slot, observe 180s of CLOB trades → build features → predict → place order if confidence > 60% and edge > 10%.

**Safety layers (DataQualityGate):**
1. Data completeness (min 50 ticks, 3/6 sub-windows)
2. Feature sanity (finite values, bounded ranges)
3. Prediction sanity (reject > 99% or < 1%)
4. Execution gate (ask price in [0.10, 0.95], circuit breaker WR < 40%)
5. Cold start protection (warmup 3 slots after restart)

**WebSocket resilience (ws_manager.py):**
- Exponential backoff with jitter (5s → 60s max)
- Active zombie detection (ping/pong probe before killing)
- Binance REST fallback (WS blocked in EU, HTTP 451)

```bash
# Deploy
cd deploy && fly deploy --app polymarket-maker-mm --remote-only
```

## Project Structure

```
polymarket-btc-lab/
├── deploy/
│   ├── live_trader.py          # Live trading bot (v21, 30 features)
│   ├── ws_manager.py           # Resilient WebSocket manager
│   ├── data_quality_gate.py    # 5-layer safety gate
│   ├── Dockerfile
│   └── fly.toml
├── scripts/
│   ├── train_v21_modal.py      # Current training script (champion)
│   ├── train_v19_modal.py      # Previous version
│   ├── fetch_ob_features_modal.py
│   └── fetch_ticks_modal.py
├── tests/
│   └── test_ws_manager.py      # 41 tests
├── docs/
│   ├── EXPERIMENTS.md           # Version history (v4 → v21)
│   ├── PIPELINE.md              # Pipeline deep-dive
│   └── wiki/                    # Technical wiki (8 pages)
└── README.md
```

## HuggingFace

- **Model + Data:** [artbreguez/polymarket-btc-model](https://huggingface.co/artbreguez/polymarket-btc-model)
  - `champion.pkl` — trained model bundle (LightGBM + calibration + feature list)
  - `champion_meta.json` — metrics, version, ablation results
  - `data/` — complete training dataset (all_markets.csv, ticks, OB features, Binance spot)

## Version History

| Version | AUC | Features | Key Change |
|---------|-----|----------|-----------|
| v4 | 0.720 | 15 | Baseline: tick flow only |
| v8 | 0.853 | 63 | Binance spot + multi-scale zscores |
| v16 | 0.887 | 56 | Lag features + slot history |
| v18 | 0.898 | 40 | Feature pruning + walk-forward |
| v19 | 0.900 | 40 | L2 orderbook features (real OB data) |
| v21 | 0.900 | 30 | Ablation study: pruned 10 low-value features |

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for full history.

## Live Performance

- Win rate: 74% (40W / 14L over 54 trades)
- P&L: +$30.61
- Wallet balance: $75.08 USDC
