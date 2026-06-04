# Polymarket BTC 5-Minute Model — Project Wiki

## Project Summary

This project builds and operates a machine-learning system that predicts the outcome of Polymarket BTC 5-minute binary markets ("Will BTC go up in the next 5 minutes?"). The pipeline ingests tick-level CLOB data from Polymarket (via pmdata.dev API and local collection), combines it with Binance BTC spot candles, engineers 30 features (sub-window up-ratios, z-scores, order flow, spot returns, lag outcomes), trains a LightGBM classifier with isotonic calibration on Modal.com cloud, publishes the champion model to HuggingFace, and deploys a live trader on Fly.io that places real orders when model confidence exceeds a configurable edge threshold.

---

## Table of Contents

| # | Page | Description |
|---|------|-------------|
| 00 | [Architecture](00-architecture.md) | System architecture, data flow, component responsibilities, costs |
| 01 | [Data Pipeline](01-data-pipeline.md) | Data sources, Modal Volume schema, pmdata.dev API, Binance spot |
| 02 | [Feature Engineering](02-feature-engineering.md) | 30-feature set, sub-windows, z-scores, spot, lags |
| 03 | [Training Pipeline](03-training-pipeline.md) | Modal training, Optuna HPO, walk-forward CV, promotion gate |
| 04 | [Model Evaluation](04-model-evaluation.md) | Metrics, calibration plots, fold stability, champion comparison |
| 05 | [Deployment](05-deployment.md) | HuggingFace model card, GitHub Actions CI, Fly.io deployment |
| 06 | [Live Trading](06-live-trading.md) | Live trader logic, edge thresholds, staking, Polymarket API |
| 07 | [Experiment Log](../EXPERIMENTS.md) | Canonical record of all model versions and results |
| 08 | [Troubleshooting](08-troubleshooting.md) | Common errors, debugging playbooks, Modal/Fly gotchas |
| 09 | [Anti-Patterns](09-anti-patterns.md) | Mistakes to avoid (leakage, overfitting, feature bloat) |
| 10 | [Roadmap](10-roadmap.md) | Planned improvements, v19 proposal, multi-asset expansion |

---

## Current State (v18 Champion)

| Metric | Value |
|--------|-------|
| Walk-Forward AUC | 0.8966 |
| Brier Score | 0.1318 |
| Accuracy (0.5 threshold) | 0.8104 |
| Training markets | 22,237 |
| Feature count | 30 |
| Model | LightGBM + isotonic calibration |
| Data coverage | Feb 15, 2026 – present |
| Data sources | local CLOB (Mar–Apr) + pmdata.dev (Apr 12 – Jun 3) |

---

## Quick-Start Commands

### Train a new model version
```bash
# Run v18 training on Modal (requires MODAL_TOKEN_ID + MODAL_TOKEN_SECRET)
modal run scripts/train_v18_modal.py

# Or use the convenience wrapper
bash train.sh
```

### Deploy live trader to Fly.io
```bash
cd deploy/
fly deploy --app polymarket-maker-mm
fly logs --app polymarket-maker-mm
```

### Monitor live trader
```bash
# Tail logs
fly logs --app polymarket-maker-mm

# Check status
fly status --app polymarket-maker-mm

# Check model version on HuggingFace
python scripts/validate_model.py
```

### Expand dataset (fetch new markets)
```bash
# Fetch new market metadata
python scripts/fetch_new_markets.py

# Fetch ticks from pmdata.dev
python scripts/fetch_pmdata_ticks.py --workers 12

# Fetch Binance spot candles (run from non-geo-blocked host)
python scripts/fetch_binance_spot.py
```

---

## Architecture Diagram

```
                         TRAINING PIPELINE
                         =================

  +--------------+     +-----------------+     +------------------+
  | pmdata.dev   |---->| Modal Volume    |---->| Modal Training   |
  | (poly_l2 API)|     | "btc-local-data"|     | (CPU 8-core,32G) |
  +--------------+     |                 |     |  LightGBM+Optuna |
                       | ticks_btc_full_ |     |  walk-forward CV |
  +--------------+     |  clean.parquet  |     +--------+---------+
  | Binance Spot |     | all_markets.csv |              |
  | (pre-fetched |---->| spot_1m_*.csv   |              | model.pkl
  |  locally due |     +-----------------+              | + metrics
  |  to geo-block)|                                     v
  +--------------+                            +------------------+
                                              | HuggingFace Hub  |
                                              | artbreguez/      |
                                              | polymarket-btc-  |
                                              | model             |
                                              +--------+---------+
                                                       |
                         DEPLOYMENT                    |
                         ==========                    v
                                              +------------------+
                                              | GitHub Actions   |
                                              | (CI: lint, test, |
                                              |  promote)        |
                                              +--------+---------+
                                                       |
                                                       v
                                              +------------------+
                                              | Fly.io           |
                                              | "polymarket-     |
                                              |  maker-mm"       |
                                              | (AMS region,     |
                                              |  perf-1x, 2G)    |
                                              |                  |
                                              | live_trader.py   |
                                              | + Binance WS     |
                                              | + Polymarket CLOB|
                                              +------------------+
```

---

## Key File Paths

```
polymarket-btc-lab/
├── src/btc_lab/
│   ├── features.py          # Shared feature engineering functions
│   ├── config.py             # Configuration constants
│   └── plugin.py             # Plugin interface
├── scripts/
│   ├── train_v18_modal.py    # Current champion training script
│   ├── fetch_pmdata_ticks.py # pmdata.dev tick fetcher
│   ├── fetch_binance_spot.py # Binance spot candle fetcher
│   ├── fetch_new_markets.py  # Market metadata fetcher
│   ├── build_dataset_modal.py# Dataset builder on Modal
│   ├── validate_model.py     # Model validation / gate check
│   ├── promote_champion.py   # Promote model to champion on HF
│   └── coverage_check.py     # Data coverage report
├── deploy/
│   ├── live_trader.py        # Live trading bot (runs on Fly.io)
│   ├── fly.toml              # Fly.io deployment config
│   └── requirements.txt      # Runtime dependencies
├── tests/
│   └── test_features.py      # Feature parity tests
├── docs/
│   ├── EXPERIMENTS.md         # Experiment log (v4 → v18)
│   ├── PIPELINE.md            # Legacy pipeline docs
│   ├── V19_PROPOSAL.md        # v19 design proposal
│   └── wiki/                  # This wiki
├── train.sh                   # One-command train wrapper
└── pyproject.toml             # Project metadata, dependencies
```

---

## Secrets Required

| Secret | Where Used | How to Set |
|--------|-----------|------------|
| `HF_TOKEN` | Training (upload model to HuggingFace) | Modal secret `hf-token`, also `huggingface-cli login` |
| `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` | Modal CLI for training runs | `modal token set` |
| `FLY_API_TOKEN` | Fly.io deployment | `fly auth login` or `FLY_API_TOKEN` env var |
| pmdata.dev API key (`sk-5uX...Ijko`) | `fetch_pmdata_ticks.py` | Hardcoded in script (rotate if compromised) |
| `POLY_PRIVATE_KEY` | Live trader (Polymarket orders) | Fly.io secret |
| `POLY_SAFE_ADDRESS` | Live trader (proxy wallet) | Fly.io secret |
| `MM_BUILDER_KEY` / `SECRET` / `PASSPHRASE` | Live trader (Builder API) | Fly.io secrets |

---

## Getting Started (New Agent Checklist)

1. Read this README top to bottom
2. Read [Architecture](00-architecture.md) to understand the system
3. Read [Data Pipeline](01-data-pipeline.md) to understand data sources
4. Read [Experiment Log](../EXPERIMENTS.md) for full version history
5. Check current champion metrics on HuggingFace: `artbreguez/polymarket-btc-model`
6. If training a new version: copy `train_v18_modal.py` → `train_v19_modal.py`, change what you need
7. Gate rule: new model MUST beat champion AUC on walk-forward CV or it does not ship
