# Polymarket BTC — ML Pipeline Technical Documentation

> **Audience**: ML engineers and contributors to the `polymarket-btc-lab` repository.
> **Last updated**: 2025 (auto-maintained; see Changelog section of model card).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Dataset](#2-dataset)
3. [Feature Engineering (v8)](#3-feature-engineering-v8)
4. [Training Protocol](#4-training-protocol)
5. [Promotion Workflow](#5-promotion-workflow)
6. [CI/CD Pipeline](#6-cicd-pipeline)
7. [Live Trader](#7-live-trader)
8. [Anti-Leakage Measures](#8-anti-leakage-measures)
9. [Adding a New Version](#9-adding-a-new-version)
10. [Known Issues & Limitations](#10-known-issues--limitations)

---

## 1. Architecture Overview

End-to-end flow from raw data to live orders on Polymarket:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        POLYMARKET BTC ML PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐
  │   BrockMisner/       │   HuggingFace Datasets
  │   polymarket-btc-    │   616 resolved BTC 5-min
  │   updown (HF)        │   binary markets
  └──────────┬───────────┘
             │  hf_hub_download()
             ▼
  ┌──────────────────────┐
  │  Feature Engineering │   scripts/train_vN_modal.py
  │  (Modal.com cloud)   │   Sub-windows, zscores, spot,
  │                      │   OB, lags, time features
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  LightGBM + Optuna   │   Bayesian HPO, 50–100 trials
  │  Training            │   Isotonic calibration wrapper
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Purged Walk-Forward │   N folds, gap=5 slots,
  │  Validation          │   mean AUC / Acc / Brier
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐     ✗ Does NOT beat champion
  │  Champion Gate       │────────────────────────────► Discard challenger
  │  (fetch AUC from HF) │
  └──────────┬───────────┘
             │ ✓ Beats champion
             ▼
  ┌──────────────────────┐
  │  promote_champion.py │   Upload champion.pkl +
  │  → HuggingFace       │   champion_meta.json to HF
  └──────────┬───────────┘
             │  HF model repo update triggers
             ▼
  ┌──────────────────────┐
  │  GitHub Actions      │   deploy.yml detects new
  │  (CI/CD)             │   champion → triggers deploy
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Fly.io Deploy       │   Docker container with
  │  (live_trader.py)    │   live_trader + champion.pkl
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Live Trader         │   Subscribes to Polymarket
  │  live_trader.py      │   WebSocket feed, predicts
  │                      │   at t = 170–240 s
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Polymarket CLOB     │   Places limit orders via
  │  Order Placement     │   BuilderApiKey REST API
  └──────────────────────┘
```

---

## 2. Dataset

**Source**: [`BrockMisner/polymarket-btc-updown`](https://huggingface.co/datasets/BrockMisner/polymarket-btc-updown)

### Overview

| Field            | Value |
|------------------|-------|
| Markets          | 616 resolved BTC 5-min binary markets |
| Resolution type  | YES = BTC closed UP, NO = BTC closed DOWN/FLAT |
| Time range       | 2024–2025 (exact range depends on dataset version) |
| Format           | Parquet files partitioned by market |

### File Structure

Each market is represented by one or more Parquet files containing:

| Column | Description |
|--------|-------------|
| `market_id` | Polymarket market condition ID (hex string) |
| `slot_start_ts` | Unix timestamp (ms) of the 5-min slot start |
| `trade_ts_ms` | Unix timestamp (ms) of each CLOB trade tick |
| `side` | `YES` or `NO` (token side of the trade) |
| `size` | Trade size in USD |
| `price` | Trade price (0–1 probability) |
| `outcome` | Resolved outcome: `1` = YES (UP), `0` = NO (DOWN) |
| `btc_open` | BTC/USD spot price at slot open |
| `btc_close` | BTC/USD spot price at slot close |

Spot OHLCV context (pre-slot 5m/15m/30m/1h/4h) is joined in during feature engineering.

### Loading the Dataset

```python
from datasets import load_dataset

ds = load_dataset("BrockMisner/polymarket-btc-updown", split="train")
df = ds.to_pandas()
```

---

## 3. Feature Engineering (v8)

All features are computed in `scripts/train_v8_modal.py` inside the `build_features()` function.
Features are computed **per slot** from tick-level CLOB data collected during the slot window (t = 0–300 s).

### 3.1 Sub-Window Features (6 × 30 s windows)

The 5-min slot is divided into 6 consecutive 30-second windows (`w0`–`w5`):

| Feature | Description |
|---------|-------------|
| `up_w0` … `up_w5` | YES-side volume fraction in each 30-second window |

`up_wN = sum(size where side=YES in window N) / sum(size in window N)`

This captures intra-slot order-flow dynamics and momentum shifts.

### 3.2 Sub-Window Z-Scores

Each sub-window fraction is standardized against its historical mean and std over a lookback of recent slots:

| Feature | Description |
|---------|-------------|
| `up_w0_z` … `up_w5_z` | Z-score of each sub-window fraction vs. lookback |

Formula: `z = (up_wN - mean(up_wN[t-L:t])) / (std(up_wN[t-L:t]) + 1e-8)`

### 3.3 Multi-Scale Z-Scores

Up-ratio (total YES fraction) is z-scored over three lookback scales to capture mean-reversion
at different timescales:

| Feature | Description |
|---------|-------------|
| `up_ratio_z5`  | Z-score over last 5 slots |
| `up_ratio_z10` | Z-score over last 10 slots |
| `up_ratio_z20` | Z-score over last 20 slots |

### 3.4 Time-Weighted Order Flow

Trades within the slot are weighted by their position in time — more recent trades receive higher weight.
This captures late-stage informed flow (smart money typically arrives near the prediction window).

| Feature | Description |
|---------|-------------|
| `tw_up_ratio` | Time-weighted YES fraction (linear decay, most recent = weight 1.0) |
| `tw_up_ratio_z` | Z-score of tw_up_ratio over lookback |

### 3.5 Classic Order-Book Features

| Feature | Description |
|---------|-------------|
| `up_ratio` | Raw YES-side volume fraction over full slot |
| `momentum` | `up_w5 - up_w0` (late vs. early window) |
| `vwap_spread` | Difference between YES VWAP and NO VWAP prices |
| `buy_ratio` | Fraction of trades classified as aggressive buys |

### 3.6 Realized Volatility

Tick-level return variance computed over short windows within the slot:

| Feature | Description |
|---------|-------------|
| `realized_vol_5s`  | Variance of log-returns over 5-second micro-windows |
| `realized_vol_10s` | Variance of log-returns over 10-second micro-windows |

High realized vol indicates uncertain, fast-moving price action.

### 3.7 Spot Context Features

BTC spot OHLCV data from before the slot open, joined via timestamp:

| Feature | Description |
|---------|-------------|
| `spot_ret_5m`   | BTC log-return over 5 min before slot |
| `spot_ret_15m`  | BTC log-return over 15 min before slot |
| `spot_ret_30m`  | BTC log-return over 30 min before slot |
| `spot_ret_1h`   | BTC log-return over 1 h before slot |
| `spot_ret_4h`   | BTC log-return over 4 h before slot |
| `spot_vol_5m`   | BTC realized vol over 5 min before slot |
| `spot_vol_15m`  | BTC realized vol over 15 min before slot |

### 3.8 Orderbook Features (UP Token Only)

Snapshot of the Polymarket CLOB orderbook for the YES token, taken at t ≈ 170 s.
**The DOWN-token orderbook was dropped in v6** — it was found to be all-zeros in
the majority of markets (no active market-making on the NO side).

| Feature | Description |
|---------|-------------|
| `ob_bid`        | Best bid price on YES token |
| `ob_ask`        | Best ask price on YES token |
| `ob_spread`     | `ob_ask - ob_bid` |
| `ob_depth_bid`  | Total USD depth within 3 cents of best bid |
| `ob_depth_ask`  | Total USD depth within 3 cents of best ask |
| `ob_imbalance`  | `(depth_bid - depth_ask) / (depth_bid + depth_ask + 1e-8)` |

### 3.9 Lag Outcome Features

Previous market outcomes, treating the markets as a time-series:

| Feature | Description |
|---------|-------------|
| `lag_outcome_1` | Outcome of the previous slot (1=UP, 0=DOWN) |
| `lag_outcome_2` | Outcome 2 slots ago |
| `lag_outcome_3` | Outcome 3 slots ago |
| `streak_up`     | Count of consecutive UP outcomes leading into this slot |

These features are carefully constructed to use only data from **before** the current slot opens.

### 3.10 Round-Number Proximity

BTC has known support/resistance at round dollar amounts:

| Feature | Description |
|---------|-------------|
| `dist_1k`  | Fractional distance to nearest $1,000 boundary |
| `dist_5k`  | Fractional distance to nearest $5,000 boundary |
| `dist_10k` | Fractional distance to nearest $10,000 boundary |

### 3.11 Time Features

Cyclical encoding to capture intraday and day-of-week patterns:

| Feature | Description |
|---------|-------------|
| `hour_sin` | `sin(2π × hour / 24)` |
| `hour_cos` | `cos(2π × hour / 24)` |
| `dow_sin`  | `sin(2π × day_of_week / 7)` |
| `dow_cos`  | `cos(2π × day_of_week / 7)` |

---

## 4. Training Protocol

### 4.1 Modal.com Setup

Training runs in a Modal cloud function defined in `scripts/train_v8_modal.py`:

```python
import modal

app = modal.App("polymarket-btc-v8")

@app.function(cpu=2, memory=4096, timeout=3600)
def train():
    # downloads dataset, builds features, runs Optuna+LightGBM, uploads to HF
    ...
```

Run with:
```bash
modal run scripts/train_v8_modal.py
```

### 4.2 LightGBM Configuration

Base parameters (before Optuna tuning):
```python
BASE_PARAMS = {
    "objective":     "binary",
    "metric":        "auc",
    "verbosity":     -1,
    "boosting_type": "gbdt",
}
```

Optuna search space typically covers: `num_leaves`, `learning_rate`, `n_estimators`,
`min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.

### 4.3 Purged Walk-Forward Validation

The dataset is split chronologically into `N` folds (typically 5). For each fold:

```
Train window:  [fold_0 ... fold_k-1]  (minus gap)
Gap (purge):   5 slots (~25 minutes) — discarded
Test window:   [fold_k]
```

The **gap of 5 slots** prevents leakage from lag features (lag_1, lag_2, lag_3)
and time-series autocorrelation in order-flow signals.

Objective passed to Optuna:
```python
def objective(trial):
    params = {**BASE_PARAMS, **trial.suggest_...}
    fold_aucs = []
    for train_idx, test_idx in purged_wf_splits(X, gap=5):
        model = lgb.train(params, ...)
        fold_aucs.append(roc_auc_score(y_test, preds))
    return np.mean(fold_aucs)
```

### 4.4 Champion Gate

Before promotion, the challenger's WF AUC is compared against the **live champion**:

```python
champion_meta = json.loads(
    hf_hub_download(HF_REPO_ID, "champion_meta.json", token=hf_token)
)
champion_auc = champion_meta["wf_auc"]

if challenger_auc > champion_auc:
    promote_champion(challenger_bundle, challenger_meta, hf_token)
else:
    print(f"Gate failed: {challenger_auc:.4f} <= {champion_auc:.4f}. Not promoting.")
```

---

## 5. Promotion Workflow

`scripts/promote_champion.py` handles the upload step after a challenger passes the gate.

### 5.1 What Gets Uploaded

| File | Description |
|------|-------------|
| `champion.pkl` | Python pickle of the champion bundle (see schema below) |
| `champion_meta.json` | Metadata JSON (see schema below) |
| `README.md` | Auto-generated model card (via `update_model_card.py`) |

### 5.2 champion.pkl Bundle Schema

```python
bundle = {
    "model":    calibrated_lgb_classifier,  # sklearn CalibratedClassifierCV wrapping LGBMClassifier
    "features": ["up_w0", "up_w1", ..., "hour_cos"],  # ordered feature name list
    "version":  "v8",
    "wf_auc":   0.8529,
}
```

### 5.3 champion_meta.json Schema

```json
{
  "version":        "v8",
  "wf_auc":         0.8529,
  "wf_acc":         0.7802,
  "wf_brier":       0.1707,
  "fold_aucs":      [0.851, 0.849, 0.856, 0.854, 0.852],
  "n_features":     42,
  "n_train_samples": 580,
  "features":       ["up_w0", "up_w1", "..."],
  "best_params":    {"num_leaves": 31, "learning_rate": 0.05, "...": "..."},
  "algorithm":      "LightGBM + isotonic calibration",
  "promoted_at":    "2025-06-01T12:34:56Z",
  "dataset":        "BrockMisner/polymarket-btc-updown",
  "hf_repo":        "artbreguez/polymarket-btc-model"
}
```

### 5.4 Updating the Model Card After Promotion

```bash
python scripts/update_model_card.py --hf-token $HF_TOKEN
```

Or from Python:
```python
from scripts.update_model_card import update_model_card
update_model_card(meta=meta_dict, hf_token=hf_token)
```

---

## 6. CI/CD Pipeline

The repository uses two GitHub Actions workflows:

### 6.1 `ci.yml` — Continuous Integration

Triggers: every push and pull request.

Steps:
1. Checkout repository
2. Set up Python 3.11
3. `pip install -r requirements.txt`
4. `pytest tests/` — unit tests for feature engineering and model loading
5. `python scripts/update_model_card.py --hf-token ${{ secrets.HF_TOKEN }} --dry-run`
   (validates model card generation without uploading)

### 6.2 `deploy.yml` — Continuous Deployment

Triggers: push to `main` **or** a webhook from HuggingFace when `champion.pkl` changes.

Steps:
1. Checkout repository
2. Download `champion.pkl` and `champion_meta.json` from HF
3. Build Docker image: `docker build -t polymarket-btc-trader .`
4. Push image to Fly.io registry
5. `flyctl deploy --image ... --app polymarket-btc-trader`
6. `flyctl status` — verify deployment health

Secrets required:
- `HF_TOKEN` — HuggingFace read token
- `FLY_API_TOKEN` — Fly.io deploy token
- `POLYMARKET_API_KEY` — Polymarket builder API key
- `POLYMARKET_API_SECRET` — corresponding secret

---

## 7. Live Trader

`live_trader.py` is the production inference and order-placement service.

### 7.1 Startup

On container start:
1. Downloads `champion.pkl` from HuggingFace
2. Deserializes the bundle (model + feature list)
3. Connects to Polymarket WebSocket feed (`wss://ws-subscriptions-clob.polymarket.com`)
4. Subscribes to all active BTC 5-min binary markets

### 7.2 Per-Slot Lifecycle

```
t =   0 s  Slot opens. Market detected on WebSocket.
t = 0–170 s  CLOB trades stream in. Order-flow accumulators updated in real-time.
t = 170 s  Prediction window opens.
t = 170–240 s  Features computed from accumulated data. predict_proba() called.
t = 240 s  Prediction window closes. Order submitted if confidence > threshold.
t = 300 s  Slot closes. Oracle resolves. P&L updated.
```

### 7.3 Feature Computation at Inference Time

At t ≈ 170–240 s, `live_trader.py` replicates the exact same feature computation
as `train_v8_modal.py::build_features()`. Both share a common module:

```
scripts/
  features_v8.py      # shared feature logic (imported by both train and live_trader)
  train_v8_modal.py   # imports features_v8.build_features()
live_trader.py        # imports features_v8.build_features()
```

This ensures **no train-serve skew**.

### 7.4 Order Placement

Orders are placed via the Polymarket CLOB REST API using a `BuilderApiKey`:

```python
from py_clob_client import ClobClient

client = ClobClient(
    host="https://clob.polymarket.com",
    key=POLYMARKET_API_KEY,
    chain_id=137,  # Polygon
)

if prob_up > BUY_THRESHOLD:
    client.create_limit_order(
        token_id=market["yes_token_id"],
        side="BUY",
        price=round(prob_up, 2),
        size=POSITION_SIZE_USD,
    )
```

Orders are limit orders placed at the model's predicted probability, providing
liquidity to the market while expressing a directional view.

### 7.5 Risk Controls

- Maximum position size per slot: configurable via `MAX_POSITION_USD` env var
- Confidence threshold: only trade when `|prob_up - 0.5| > MIN_EDGE` (default 0.05)
- Maximum open positions: at most `MAX_CONCURRENT_POSITIONS` active bets
- Circuit breaker: if 5 consecutive losses occur, pause trading for 30 minutes

---

## 8. Anti-Leakage Measures

Data leakage is the primary risk in time-series ML. The following measures are enforced:

### 8.1 Purged Walk-Forward Gap

A gap of **5 slots (25 minutes)** is dropped between the training tail and the test head
in every fold. This eliminates:
- Direct lag feature leakage (`lag_outcome_1/2/3` uses outcomes from 1–3 slots before)
- CLOB order-flow autocorrelation (order-flow is serially correlated at sub-minute lags)
- Streak feature leakage (`streak_up` can look back 10+ slots)

### 8.2 No Future Data in Features

All features are computed strictly from data at `t < slot_open`:
- Spot returns use OHLCV candles that **closed before** the slot start timestamp
- Lag outcomes use outcomes of **previous** markets only
- Orderbook snapshot at t ≈ 170 s is still within the slot (before close)

### 8.3 Lag Feature Construction

Lag features are built with an explicit sort-by-timestamp followed by `.shift(N)`:
```python
df = df.sort_values("slot_start_ts")
df["lag_outcome_1"] = df["outcome"].shift(1)  # strictly prior slot
df["lag_outcome_2"] = df["outcome"].shift(2)
df["lag_outcome_3"] = df["outcome"].shift(3)
```

The first 3 rows (insufficient lag history) are dropped from training.

### 8.4 Fair Champion Gate

When comparing a challenger to the current champion:
- The champion's AUC is read from `champion_meta.json` (stored at promotion time)
- It was computed on a **different** (older) test set; direct AUC comparison has slight
  bias but is conservative (challenger must beat a potentially optimistic champion AUC)
- A future improvement would re-evaluate both models on the same held-out test set

---

## 9. Adding a New Version

Follow these steps to iterate on the model:

### Step 1 — Copy the Training Script

```bash
cp scripts/train_v8_modal.py scripts/train_v9_modal.py
```

Update the version string inside the new file:
```python
VERSION = "v9"
```

### Step 2 — Add / Modify Features

Edit `scripts/features_v8.py` (or create `scripts/features_v9.py`) to add new features.
Ensure all new features:
- Use only data available before `slot_open`
- Are included in the feature name list returned by `build_features()`
- Handle `NaN` values (fill with 0 or median, document the choice)

### Step 3 — Run Training on Modal

```bash
modal run scripts/train_v9_modal.py
```

This will:
- Download the dataset from HF
- Build features
- Run Optuna HPO (50–100 trials)
- Evaluate walk-forward AUC
- Compare against current champion
- If gate passes → upload new champion to HF automatically

### Step 4 — Verify the Gate Result

Check the Modal logs for:
```
Challenger AUC: 0.XXXX  Champion AUC: 0.8529
Gate PASSED — promoting v9 to champion.
```

or:
```
Gate FAILED — challenger (0.XXXX) did not beat champion (0.8529). Skipping.
```

### Step 5 — Update the Model Card

If promoted:
```bash
python scripts/update_model_card.py --hf-token $HF_TOKEN
```

### Step 6 — Update CHANGELOG in Model Card

Add a row to the `CHANGELOG` list in `scripts/update_model_card.py`:
```python
("v9", "0.XXXX", "0.XXXX", "0.XXXX", "Description of key changes"),
```

### Step 7 — Open a Pull Request

- Commit `scripts/train_v9_modal.py` and `scripts/features_v9.py`
- CI will run automatically; verify all tests pass
- Merge to `main` triggers auto-deploy to Fly.io

---

## 10. Known Issues & Limitations

### 10.1 Small Dataset

The dataset currently contains only **616 resolved markets**. After feature engineering
(dropping early rows for lag initialization, dropping rows with insufficient lookback for
z-scores), the effective training set is approximately 580–600 samples.

With such a small dataset:
- AUC confidence intervals are approximately ±0.02 (bootstrap 95% CI)
- Optuna may overfit the walk-forward objective with too many trials (use early stopping)
- Adding features aggressively increases the risk of false discoveries

### 10.2 No Historical Tick Data for Expansion

The Polymarket CLOB does not provide a free historical tick API. The dataset is
limited to what BrockMisner collected. Expanding the dataset would require:
- Running a CLOB WebSocket collector continuously (ongoing data collection)
- Purchasing historical data (not currently available from Polymarket)

### 10.3 Oracle Agreement and Label Noise

Polymarket markets resolve based on an independent oracle (UMA protocol). Resolution statistics:
- ~98% of markets resolve within 30 s of slot close (low latency)
- ~1–2% of markets enter dispute (extended resolution period)
- Disputed markets are excluded from the training dataset

The oracle's BTC price source may occasionally differ from the spot price used in spot context
features (different exchange, different timestamp). This introduces a small amount of label noise.

### 10.4 Market Microstructure Drift

The statistical properties of Polymarket CLOB order flow can change over time due to:
- Changes in the set of active market makers
- Changes in Polymarket's fee structure
- Changes in BTC spot market volatility regime

Model performance should be monitored in production. A significant drop in live accuracy
(sustained below 52% over 50+ slots) should trigger retraining.

### 10.5 DOWN-Token Orderbook

The YES (UP) token orderbook is the primary liquidity venue. The NO (DOWN) token orderbook
was found to be effectively empty (all-zeros spread, zero depth) in the majority of markets
in the dataset. It was dropped in v6 and has not been revisited. If market structure changes
and the NO token develops active two-sided liquidity, it should be re-included.

### 10.6 Single-Asset Scope

This model is trained exclusively on BTC 5-min binary markets. It should not be applied
to ETH or other crypto assets without full retraining. The feature distributions (especially
spot returns, volatility levels, and round-number proximity) are asset-specific.

---

*End of Pipeline Documentation*
