# 03 — Training Pipeline

> Last updated: 2026-06-04 (v18 champion)
> Owner: ML team

---

## Table of Contents

1. [10-Step Training Process](#1-10-step-training-process)
2. [Modal Configuration](#2-modal-configuration)
3. [Optuna Hyperparameter Search](#3-optuna-hyperparameter-search)
4. [Walk-Forward Validation](#4-walk-forward-validation)
5. [Calibration](#5-calibration)
6. [Champion Gate](#6-champion-gate)
7. [Running a Training Job](#7-running-a-training-job)
8. [Creating a New Version](#8-creating-a-new-version)
9. [Binance Spot Data](#9-binance-spot-data)
10. [Artifacts](#10-artifacts)

---

## 1. 10-Step Training Process

Every training run (scripts/train_v18_modal.py) follows this exact sequence:

### Step 1: Load Champion Metrics

Download champion_meta.json from HuggingFace (artbreguez/polymarket-btc-model). Extract wf_auc, wf_brier, wf_acc as the gate thresholds. Falls back to hardcoded v17 defaults if download fails.

```
champion = {version, wf_auc, wf_brier, wf_acc}
```

### Step 2: Load Markets

Read all_markets.csv from Modal Volume. Contains 22,319 markets with columns: market_id, slot_ts, target. Sort chronologically. Build rank index for lag lookups.

### Step 3: Load Binance Spot

Read binance_spot_full.parquet from Modal Volume. Contains 119k 1-minute candles (Mar 13 - Jun 4, 2026). Build sorted arrays (spot_ts_arr, spot_px_arr) for binary-search price lookups via np.searchsorted.

IMPORTANT: Binance API blocks Modal's US-region IPs (HTTP 451). Spot data MUST be pre-fetched locally and uploaded to the Modal Volume. See section 9.

### Step 4: Load Ticks

Read ticks_btc_full_clean.parquet from Modal Volume via row-group streaming (to stay within 32GB RAM). 22,237 markets, 68.3M ticks. Filter to [0, 180s) observation window. Build per-slot aggregates (slot_vol_up, slot_vol_dn, slot_up_ratio, slot_nticks).

### Step 5: Feature Engineering

For each of the 22k markets, compute all 56+ features:
- CLOB flow (up_ratio, buy_ratio, vwap, momentum, etc.)
- 6x30s sub-windows (btc_up_w0..w5)
- Z-scores (5/10/20 slot lookback)
- Spot features (inslot_ret, pre_5m/30m/1h/4h_ret, dist_1k, 1h_4h_ratio)
- Lag features (lag_1..5_outcome, prev_slot_up_ratio_1..5, lag_streak)
- Temporal (hour_sin/cos, dow_sin/cos, hour_x_up_ratio, hour_x_tw_ur)
- Interaction (signal_conviction, momentum_vol_sync)

Output: DataFrame of shape (22,319 x 57).

### Step 6: Feature Selection

Train a screening LightGBM (n_estimators=300, lr=0.05, max_depth=4) on each of 5 walk-forward folds. Average feature_importances_ across folds. Select top 30 features (TOP_N_FEATS=30).

### Step 7: Optuna Tuning

Run 150 Bayesian optimization trials with 4 parallel workers. Each trial trains on all 5 WF folds and returns mean AUC. See section 3 for search space details.

### Step 8: Walk-Forward Evaluation

Using the best Optuna params, run final 5-fold walk-forward evaluation with isotonic calibration. Record per-fold AUC, Brier, accuracy. Report means.

### Step 9: Champion Gate

Compare challenger metrics against champion_meta.json. Require 2/3 metrics to beat the champion:
- AUC: challenger > champion (higher is better)
- Brier: challenger < champion (lower is better)
- Accuracy: challenger > champion (higher is better)

If score < 2, stop. Model is NOT promoted.

### Step 10: Save & Promote

If gate passes:
1. Train final model on ALL data with best params + isotonic calibration
2. Run sanity check probes (UP > NEUTRAL > DOWN)
3. Pickle the champion bundle (model + features + metadata)
4. Upload champion.pkl and champion_meta.json to HuggingFace

---

## 2. Modal Configuration

```python
app = modal.App("btc-v18-run", image=image)

@app.function(
    cpu=8,              # 8 vCPUs for Optuna parallel trials
    memory=32768,       # 32 GB RAM — needed for 68M-tick parquet
    timeout=7200,       # 2 hours — typical run is ~45min
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/btc_local": modal.Volume.from_name("btc-local-data")},
)
```

| Parameter | Value | Why |
|-----------|-------|-----|
| cpu | 8 | Optuna runs 4 workers; each trial trains 5 folds. 8 cores keeps CPU busy. |
| memory | 32768 MB (32 GB) | 68M-tick parquet + feature DataFrame + multiple LightGBM models in memory. |
| timeout | 7200 s (2 hours) | 150 Optuna trials take ~30-45 min. Buffer for data loading + promotion. |
| secrets | [hf-token] | HuggingFace token for downloading champion_meta.json and uploading new champion. |
| volumes | {"/btc_local": btc-local-data} | Pre-uploaded data: ticks, markets CSV, Binance spot. |

Image dependencies:
```python
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pyarrow>=18.0", "pandas>=2.2", "lightgbm==4.6.0",
    "scikit-learn==1.8.0", "numpy>=1.26", "optuna>=3.6",
    "huggingface_hub>=0.26",
)
```

---

## 3. Optuna Hyperparameter Search

### Configuration

- Trials: 150 (diminishing returns past 150; 300 was tested, gained <0.001 AUC)
- Workers: 4 (n_jobs=4)
- Direction: maximize (AUC)
- Sampler: TPE (default)

### Search Space

| Parameter | Type | Range | Scale | v18 Best |
|-----------|------|-------|-------|----------|
| n_estimators | int | [200, 800] | linear | 635 |
| learning_rate | float | [0.01, 0.15] | log | 0.01171 |
| max_depth | int | [3, 7] | linear | 4 |
| num_leaves | int | [8, 63] | linear | 47 |
| min_child_samples | int | [20, 100] | linear | 84 |
| subsample | float | [0.6, 1.0] | linear | 0.7647 |
| colsample_bytree | float | [0.6, 1.0] | linear | 0.6114 |
| reg_alpha | float | [1e-4, 1.0] | log | 0.2289 |
| reg_lambda | float | [1e-4, 1.0] | log | 0.0001395 |

Fixed parameters: random_state=42, verbose=-1, objective="binary", metric="auc", boosting_type="gbdt".

### Objective Function

Each trial evaluates mean AUC across 5 walk-forward folds with gap=5:

```python
def objective(trial):
    params = {suggest params from search space}
    aucs = []
    for tr_idx, val_idx in TimeSeriesSplit(n_splits=5, gap=5).split(X_sel):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_sel[tr_idx], y[tr_idx])
        p = m.predict_proba(X_sel[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], p))
    return np.mean(aucs)
```

Note: Optuna tuning does NOT use isotonic calibration — it optimizes raw LightGBM AUC. Calibration is applied only in the final WF evaluation (Step 8) and final model training (Step 10).

---

## 4. Walk-Forward Validation

### Configuration

- Folds: 5 (N_SPLITS=5)
- Gap: 5 slots (WF_GAP=5)
- Splitter: sklearn.model_selection.TimeSeriesSplit

### Why 5 Folds?

With 22,319 samples and 5 folds, each test fold gets ~4,400 samples. This is enough for stable AUC estimates (std < 0.01 across folds). The v18 fold AUCs were [0.8831, 0.8877, 0.9006, 0.9020, 0.9099] — tight range.

In the v8-v12 era with only 601 samples, 7 folds was tested (v12) and was worse — each fold had only ~85 test samples, causing noisy AUC estimates.

### Why Gap=5?

The gap discards 5 slots (~25 minutes) between the training and test windows. This prevents leakage from:

1. Lag features: lag_1..3_outcome directly uses previous slot targets. Without a gap, the last training slot's target leaks into the first test slot's lag_1.

2. Z-score lookback: up_ratio_zscore_20s uses the last 20 slots' up_ratio. Without a gap, training-set slots contaminate the z-score computation for early test slots.

3. BTC spot autocorrelation: BTC price has ~5-10 minute autocorrelation. A gap of 5 slots (25 min) exceeds this, ensuring spot returns in test are independent.

Gap=3 was tested — it gave slightly higher AUC in some seeds but riskier. Gap=5 is conservative and correct.

### Fold Diagram

```
Fold 0:  [========= TRAIN =========][GAP][ TEST ]
Fold 1:  [============= TRAIN =============][GAP][ TEST ]
Fold 2:  [================= TRAIN =================][GAP][ TEST ]
Fold 3:  [===================== TRAIN =====================][GAP][ TEST ]
Fold 4:  [========================= TRAIN =========================][GAP][ TEST ]
```

TimeSeriesSplit is expanding-window: each fold's training set includes all data before the gap+test block, growing monotonically.

---

## 5. Calibration

### ALWAYS Use Isotonic

```python
from sklearn.calibration import CalibratedClassifierCV

cal = CalibratedClassifierCV(base_lgb, cv=3, method="isotonic")
cal.fit(X_train, y_train)
```

| Method | Best Brier | Version | Notes |
|--------|-----------|---------|-------|
| Isotonic | 0.1318 | v18 | Non-parametric, fits any calibration curve shape |
| Sigmoid (Platt) | 0.1809 | v9 | Only 2 parameters, underfits this distribution |

Sigmoid was tested in v9 and worsened Brier from 0.1562 to 0.1809. LightGBM's raw probability distribution is not well-approximated by a logistic function — isotonic's non-parametric approach fits it much better, even with limited data.

The calibration uses cv=3 (3-fold cross-validation within CalibratedClassifierCV) to avoid overfitting the isotonic mapping.

---

## 6. Champion Gate

The gate prevents regressions by requiring a new model to beat the current champion on at least 2 of 3 metrics.

### Metrics

| Metric | Direction | Champion v17 | v18 Result | Pass? |
|--------|-----------|-------------|------------|-------|
| WF AUC | higher better | 0.8925 | 0.8966 | YES |
| WF Brier | lower better | 0.1342 | 0.1318 | YES |
| WF Accuracy | higher better | 0.8032 | 0.8104 | YES |

Gate score: 3/3 (needed 2/3).

### Implementation

```python
champion = json.load(hf_hub_download("champion_meta.json"))

beats_auc   = wf_auc   > champion["wf_auc"]
beats_brier = wf_brier < champion["wf_brier"]
beats_acc   = wf_acc   > champion["wf_acc"]
score = sum([beats_auc, beats_brier, beats_acc])

if score >= 2:
    promote()
```

### Gate Bug History

In v1-v12, the gate re-evaluated the champion with default LightGBM params, deflating its AUC from ~0.85 to ~0.82. This let weak models (v11, v12) incorrectly promote. Fixed by using champion_meta.json values directly — these were computed with the champion's own Optuna params during its training run.

---

## 7. Running a Training Job

### Prerequisites

1. Modal CLI installed and authenticated (`modal setup`)
2. Modal secret "hf-token" configured with HF_TOKEN
3. Modal volume "btc-local-data" populated with:
   - ticks_btc_full_clean.parquet (22k markets, 68M ticks)
   - all_markets.csv (22k market timeline)
   - binance_spot_full.parquet (119k candles)

### Run

```bash
modal run scripts/train_v18_modal.py
```

This will:
1. Spin up an 8-CPU, 32GB Modal container
2. Load data from the btc-local-data volume
3. Run feature engineering (~5 min)
4. Run 150 Optuna trials (~25-35 min)
5. Run walk-forward evaluation (~5 min)
6. Evaluate gate against champion
7. Promote to HF if gate passes

Total time: ~45-60 minutes. Cost: ~$0.50-$1.00.

### Monitoring

Modal provides live logs in the terminal. Key log lines to watch:

```
[HH:MM:SS] Step 2: Loading all_markets.csv...
[HH:MM:SS] Markets: 22319 (50.2% UP)
[HH:MM:SS] Ticks loaded: 68300000 rows for 22237 markets
[HH:MM:SS] Feature matrix: 22319 rows x 57 cols
[HH:MM:SS] Top features: [btc_inslot_ret, btc_vwap_up, ...]
[HH:MM:SS] Best trial AUC=0.8970 params={...}
[HH:MM:SS] WF results: AUC=0.8966 | Brier=0.1318 | Acc=0.8104
[HH:MM:SS] Gate vs v17: AUC Y | Brier Y | Acc Y -> 3/3
[HH:MM:SS] PROMOTING v18!
```

---

## 8. Creating a New Version

To create v19 from the current v18:

### 1. Copy the Training Script

```bash
cp scripts/train_v18_modal.py scripts/train_v19_modal.py
```

### 2. Update Version Strings

In train_v19_modal.py, find and replace:
- "btc-v18-run" -> "btc-v19-run" (Modal app name)
- "v18" -> "v19" (version in model_data and meta dicts)
- Update the docstring with what changed

### 3. Modify Features/Params

Options:
- Add new features in Step 5 (they'll be auto-evaluated by Step 6 selection)
- Change the feature selection threshold (TOP_N_FEATS)
- Modify the Optuna search space (wider/narrower ranges)
- Change WF config (but stick with 5 folds, gap=5 unless you have a good reason)

### 4. Update Data (Optional)

If new markets are available:
- Regenerate ticks_btc_full_clean.parquet with new data
- Update all_markets.csv
- Re-fetch Binance spot (section 9)
- Upload to Modal volume

### 5. Run

```bash
modal run scripts/train_v19_modal.py
```

The gate will automatically compare against the current champion (v18). If v19 is worse on 2+ metrics, it won't promote — safe to experiment.

### 6. Update Documentation

After promotion:
- Add v19 entry to docs/EXPERIMENTS.md
- Update docs/wiki/02-feature-engineering.md if features changed
- Update the Feature Hall of Fame / Graveyard

---

## 9. Binance Spot Data

### The Problem

Binance API returns HTTP 451 (Unavailable For Legal Reasons) from Modal's US-region IPs. This means training scripts cannot fetch spot data inline during Modal execution.

### The Solution

Pre-fetch locally and upload to the Modal Volume.

#### Step 1: Fetch Locally

```bash
python scripts/fetch_spot_full.py
```

This script:
- Fetches BTC/USDT 1-minute klines from Binance REST API
- Covers the full date range needed (Mar 2026 - present)
- Saves to binance_spot_full.parquet locally
- Output: ~119k candles, columns: [timestamp_ms, open, high, low, close, volume]

If scripts/fetch_spot_full.py doesn't exist, use scripts/fetch_binance_spot.py — it does the same thing.

#### Step 2: Upload to Modal Volume

```bash
modal volume put btc-local-data binance_spot_full.parquet /binance_spot_full.parquet
```

#### Step 3: Verify

```bash
modal volume ls btc-local-data
# Should show:
#   ticks_btc_full_clean.parquet
#   all_markets.csv
#   binance_spot_full.parquet
```

The training script loads from /btc_local/binance_spot_full.parquet with a fallback to /btc_local/binance_spot_local.parquet (older filename).

### Updating Spot Data

Re-fetch periodically to cover new dates:
```bash
python scripts/fetch_spot_full.py          # fetches full range
modal volume put btc-local-data binance_spot_full.parquet /binance_spot_full.parquet
```

---

## 10. Artifacts

### champion.pkl (Pickle Bundle)

```python
{
    "version":  "v18",
    "features": ["btc_inslot_ret", "btc_vwap_up", ...],  # ordered list of 30 features
    "model":    CalibratedClassifierCV,                    # sklearn wrapper around LGBMClassifier
    "wf_auc":   0.8966,
    "wf_brier": 0.1318,
    "wf_acc":   0.8104,
}
```

### champion_meta.json

```json
{
    "version": "v18",
    "wf_auc": 0.8966,
    "wf_brier": 0.1318,
    "wf_acc": 0.8104,
    "features": ["btc_inslot_ret", "btc_vwap_up", "..."],
    "n_samples": 22319,
    "n_features": 30,
    "timestamp": "2026-06-04T...",
    "changes": "Full 22k-market dataset, 3x more training data vs v17"
}
```

Both are uploaded to HuggingFace (artbreguez/polymarket-btc-model) and downloaded by the live trader on startup.
