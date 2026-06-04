# 00 — System Architecture

## Overview

The Polymarket BTC prediction system is a cloud-native ML pipeline with four major stages: data ingestion, training, model publishing, and live trading. Each stage runs on a different platform chosen for cost, latency, and reliability.

---

## Component Map

| Component | Platform | Purpose | Cost |
|-----------|----------|---------|------|
| Data ingestion (pmdata.dev) | Local / any server | Fetch historical CLOB ticks via REST API | ~$0 (API key required) |
| Data ingestion (Binance) | Local (non-geo-blocked) | Fetch BTC/USDT 1-minute spot candles | Free |
| Data storage | Modal Volume `btc-local-data` | Persistent storage for parquets + CSVs | ~$0.50/GB/mo |
| Training | Modal.com (CPU 8-core, 32 GB RAM) | LightGBM + Optuna HPO, walk-forward CV | ~$0.10–0.30 per run |
| Model registry | HuggingFace Hub (`artbreguez/polymarket-btc-model`) | Versioned model storage + model card | Free |
| CI/CD | GitHub Actions | Lint, test, promotion gate | Free tier |
| Live trader | Fly.io `polymarket-maker-mm` (AMS, perf-1x, 2 GB) | Real-time prediction + order execution | ~$10/mo |

**Total estimated cost: ~$12/month** (dominated by Fly.io VM).

---

## Data Flow

```
1. DATA COLLECTION
   pmdata.dev API ──(poly_l2 parquets)──> local disk ──> Modal Volume
   Binance REST    ──(1m candles CSV)───> local disk ──> Modal Volume
   
   Note: Binance API is geo-blocked from Modal's US servers.
   Spot data must be pre-fetched locally and uploaded to the volume.

2. TRAINING (Modal)
   Modal Volume ──> train_v18_modal.py
   ├── Load ticks_btc_full_clean.parquet (22k markets, 68M ticks)
   ├── Load all_markets.csv (market metadata + resolution)
   ├── Load spot_1m candles (Binance)
   ├── Engineer 30 features per market
   ├── Purged walk-forward CV (5 folds, gap=5 slots)
   ├── Optuna HPO (50-100 trials, maximize AUC)
   ├── Isotonic calibration on held-out fold
   ├── Gate check: AUC > champion AUC
   └── Upload model.pkl + metrics to HuggingFace

3. DEPLOYMENT
   HuggingFace ──> GitHub Actions (optional promote step)
                ──> Fly.io (live_trader.py downloads model on startup)

4. LIVE TRADING (Fly.io)
   Every 5-minute slot:
   ├── Observe CLOB ticks for 3 minutes (t=0 to t=180s)
   ├── Build features from ticks + Binance WS spot stream
   ├── Predict probability with model
   ├── If confidence > 60% AND edge >= 10%: place order
   └── Settle position after slot ends
```

---

## Network Constraints

| Constraint | Impact | Mitigation |
|-----------|--------|------------|
| Binance geo-blocked on Modal (US) | Cannot fetch spot data during training | Pre-fetch locally, upload to Modal Volume |
| Polymarket CLOB WebSocket | Live trader needs persistent connection | Fly.io AMS region (close to EU infra) |
| pmdata.dev rate limits | Throttled at ~10 req/s | ThreadPoolExecutor with backoff, progress checkpointing |
| HuggingFace upload | Large model files (~50-100 MB) | Single pickle upload, not a blocker |

---

## Component Responsibilities

### `scripts/train_v18_modal.py`
- Runs on Modal as a serverless function
- Reads data from Modal Volume `/btc_local/`
- Engineers features, trains LightGBM, tunes with Optuna
- Performs walk-forward cross-validation (5 folds, purged)
- Applies isotonic calibration
- Gates against current champion (must beat AUC)
- Uploads winner to HuggingFace

### `scripts/fetch_pmdata_ticks.py`
- Fetches CLOB tick data from pmdata.dev poly_l2 API
- Converts poly_l2 format to internal tick schema
- Handles checkpointing for resumable downloads

### `scripts/fetch_binance_spot.py`
- Downloads BTC/USDT 1-minute candles from Binance REST API
- Must run from a non-geo-blocked location
- Output: CSV files uploaded to Modal Volume

### `deploy/live_trader.py`
- Downloads champion model from HuggingFace on startup
- Connects to Binance WebSocket for real-time spot data
- Observes Polymarket CLOB ticks each 5-minute slot
- Builds features matching training pipeline exactly
- Places orders via Polymarket Builder API
- Runs 24/7 on Fly.io

### `src/btc_lab/features.py`
- Shared feature engineering code (imported by both training and live)
- Ensures feature parity between training and inference

---

## Failure Modes

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Modal training timeout (>2h) | Modal logs / email alert | Reduce Optuna trials or data subset |
| HuggingFace upload fails | Training script exit code | Re-run training; model is idempotent |
| Fly.io crash / OOM | `fly status`, `fly logs` | Auto-restart (Fly handles this); check memory usage |
| Binance WS disconnect | Live trader log warns | Auto-reconnect logic in spot_daemon thread |
| Polymarket API error | Live trader log + skip slot | Exponential backoff; skip slot if unrecoverable |
| Stale model (>7 days) | Manual check | Retrain with latest data |

---

## See Also

- [Data Pipeline](01-data-pipeline.md) — detailed data source documentation
- [Experiment Log](../EXPERIMENTS.md) — full version history with metrics
- [Deployment](05-deployment.md) — Fly.io and CI/CD details
