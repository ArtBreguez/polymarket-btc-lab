"""
update_model_card.py
====================
Generates and uploads a HuggingFace model card (README.md) for
artbreguez/polymarket-btc-model.

Standalone usage:
    python scripts/update_model_card.py --hf-token $HF_TOKEN
    python scripts/update_model_card.py --hf-token $HF_TOKEN --dry-run
    python scripts/update_model_card.py --hf-token $HF_TOKEN --meta champion_meta.json

Importable usage:
    from scripts.update_model_card import update_model_card
    update_model_card(meta=meta_dict, hf_token="hf_...")
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_ID        = "artbreguez/polymarket-btc-model"
HF_DATASET_ID     = "BrockMisner/polymarket-btc-updown"
HF_META_FILENAME  = "champion_meta.json"
HF_README_FILENAME = "README.md"

CHANGELOG = [
    # (version, AUC, Acc, Brier, n_samples, key_changes)
    ("v4",  "0.8430", "—",      "—",      "601",  "Baseline multi-crypto — ETH/SOL noise → deprecated"),
    ("v5",  "0.8553", "—",      "—",      "601",  "BTC-only, OB ts_ms fix, Optuna WF objective"),
    ("v6",  "0.8559", "0.7738", "0.1562", "601",  "Lag outcomes, purged WF gap=5 — best Brier ever"),
    ("v7",  "0.8536", "0.7598", "0.1593", "601",  "Realized vol, tw_up_ratio, VWAP trend; ensemble removed"),
    ("v8",  "0.8529", "0.7802", "0.1707", "601",  "6×30s sub-windows, multi-scale zscore — 63 features, pruned"),
    ("v9",  "0.8519", "0.7842", "0.1809", "601",  "Sigmoid calib (worse), 63→27 features, 3 new tick feats"),
    ("v10", "0.8547", "0.7902", "0.1554", "601",  "**CHAMPION** — isotonic back, interaction feats, min_child 5–40"),
    ("v11", "0.8533", "0.7839", "0.1585", "601",  "price_percentile + final_burst — no gain over v10, rejected"),
    ("v12", "0.8542", "0.7762", "0.1634", "601",  "7-fold WF — test sets too small, rejected"),
    ("v13", "—",      "—",      "—",      "7062", "Local-only 2026 data — spot_price_usdt 99.6% null, rejected"),
    ("v14", "—",      "—",      "—",      "7062", "Local 2026 + Binance 1m spot klines — full spot coverage (training)"),
    ("v17", "0.8873", "0.8059", "0.1426", "15257", "Extended lag context + temporal features + expanded dataset"),
    ("v18", "0.8979", "0.8126", "0.1293", "22319", "22k markets, feature pruning (40 features), walk-forward validation"),
    ("v19", "0.9000", "0.8127", "0.1291", "22319", "Real L2 orderbook features from pmdata, OB-CLOB interactions"),
    ("v21", "0.9002", "0.8134", "0.1290", "22319", "Ablation study: pruned to 30 features, same AUC, better accuracy"),
]

USAGE_SNIPPET = '''\
```python
import pickle
from huggingface_hub import hf_hub_download
import pandas as pd

HF_TOKEN = "hf_..."  # your HuggingFace read token

# Download and load the champion bundle
path = hf_hub_download(
    repo_id="artbreguez/polymarket-btc-model",
    filename="champion.pkl",
    token=HF_TOKEN,
)
with open(path, "rb") as f:
    bundle = pickle.load(f)

model    = bundle["model"]     # CalibratedClassifierCV wrapping LightGBMClassifier
features = bundle["features"]  # ordered list of feature names (must match your DataFrame)
version  = bundle["version"]   # e.g. "v10"

# Build a one-row feature DataFrame (fill unknowns with 0.0) and predict
X = pd.DataFrame([your_feature_dict])[features].fillna(0.0)
prob_up = model.predict_proba(X)[0, 1]   # P(BTC closes UP in this 5-min slot)
print(f"P(UP) = {prob_up:.3f}")
```'''

# ---------------------------------------------------------------------------
# Full feature registry with descriptions and categories
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: dict[str, tuple[str, str]] = {
    # (category, description)

    # ── CLOB volume / flow ──────────────────────────────────────────────────
    "btc_up_ratio":          ("CLOB Flow",    "Fraction of total USDC volume traded on the UP token during the observation window"),
    "btc_vol_up":            ("CLOB Flow",    "Total USDC volume on the UP token"),
    "btc_vol_dn":            ("CLOB Flow",    "Total USDC volume on the DOWN token"),
    "btc_n_ticks":           ("CLOB Flow",    "Number of trades executed in the observation window"),
    "btc_avg_size":          ("CLOB Flow",    "Average trade size in USDC across all ticks"),
    "btc_buy_ratio":         ("CLOB Flow",    "Fraction of trades that are BUY-side (taker aggressor)"),
    "btc_vwap_spread":       ("CLOB Flow",    "Spread between volume-weighted average price of UP and DOWN tokens"),
    "btc_vwap_up":           ("CLOB Flow",    "Volume-weighted average price of the UP token"),
    "btc_vwap_dn":           ("CLOB Flow",    "Volume-weighted average price of the DOWN token"),
    "btc_size_disparity":    ("CLOB Flow",    "Ratio of mean UP trade size to mean DOWN trade size"),

    # ── CLOB temporal / momentum ────────────────────────────────────────────
    "btc_momentum":          ("CLOB Momentum", "Difference in mean up_ratio between the last 3 vs first 3 sub-windows (trend in last 90s)"),
    "btc_tw_up_ratio":       ("CLOB Momentum", "Time-weighted up_ratio — exponentially weights recent trades more heavily"),
    "btc_vwap_trend":        ("CLOB Momentum", "Change in VWAP(UP) from first half to second half of the observation window"),
    "btc_vwmom":             ("CLOB Momentum", "Volume-weighted momentum: dot product of per-window volume weight and (up_ratio − 0.5)"),
    "btc_tick_accel":        ("CLOB Momentum", "Ratio of trade count in last 30s vs first 30s — measures activity acceleration"),
    "btc_vol_accel":         ("CLOB Momentum", "Ratio of USDC volume in last 90s vs first 90s — volume acceleration"),
    "btc_up_ratio_stability": ("CLOB Momentum","Standard deviation of per-window up_ratio — measures directional consistency"),

    # ── CLOB sub-windows (6 × 30s) ─────────────────────────────────────────
    "btc_up_w0":             ("Sub-window",   "up_ratio in seconds 0–30 of the slot"),
    "btc_up_w1":             ("Sub-window",   "up_ratio in seconds 30–60 of the slot"),
    "btc_up_w2":             ("Sub-window",   "up_ratio in seconds 60–90 of the slot"),
    "btc_up_w3":             ("Sub-window",   "up_ratio in seconds 90–120 of the slot"),
    "btc_up_w4":             ("Sub-window",   "up_ratio in seconds 120–150 of the slot"),
    "btc_up_w5":             ("Sub-window",   "up_ratio in seconds 150–180 of the slot — strongest single feature"),

    # ── Cross-slot z-scores (historical context) ────────────────────────────
    "btc_up_ratio_zscore_5s":     ("Z-score",  "Z-score of current up_ratio vs last 5 slots"),
    "btc_up_ratio_zscore_10s":    ("Z-score",  "Z-score of current up_ratio vs last 10 slots"),
    "btc_up_ratio_zscore_20s":    ("Z-score",  "Z-score of current up_ratio vs last 20 slots"),
    "btc_up_ratio_hist_mean_5s":  ("Z-score",  "Historical mean up_ratio over the last 5 slots"),
    "btc_up_ratio_hist_mean_10s": ("Z-score",  "Historical mean up_ratio over the last 10 slots"),
    "btc_up_ratio_hist_mean_20s": ("Z-score",  "Historical mean up_ratio over the last 20 slots"),
    "btc_up_w0_zscore":           ("Z-score",  "Z-score of btc_up_w0 vs last 20 slots — window 0 deviation"),
    "btc_up_w1_zscore":           ("Z-score",  "Z-score of btc_up_w1 vs last 20 slots"),
    "btc_up_w2_zscore":           ("Z-score",  "Z-score of btc_up_w2 vs last 20 slots"),
    "btc_up_w3_zscore":           ("Z-score",  "Z-score of btc_up_w3 vs last 20 slots"),
    "btc_up_w4_zscore":           ("Z-score",  "Z-score of btc_up_w4 vs last 20 slots"),
    "btc_up_w5_zscore":           ("Z-score",  "Z-score of btc_up_w5 vs last 20 slots — most informative z-score"),
    "btc_vol_zscore":             ("Z-score",  "Z-score of current slot USDC volume vs last 20 slots"),
    "btc_vol_ratio":              ("Z-score",  "Ratio of current slot volume to 20-slot rolling mean"),
    "btc_realized_vol_5s":        ("Z-score",  "Std of BTC 5min returns over last 5 slots — short-term realized volatility"),
    "btc_realized_vol_10s":       ("Z-score",  "Std of BTC 5min returns over last 10 slots — medium-term realized volatility"),

    # ── BTC spot price features (Binance 1m klines) ─────────────────────────
    "btc_inslot_ret":        ("Spot Price",   "BTC spot return during the observation window (close/open − 1)"),
    "btc_inslot_vol":        ("Spot Price",   "BTC spot price volatility within the observation window (std/mean of 1m closes)"),
    "btc_pre_5m_ret":        ("Spot Price",   "BTC spot return in the 5 minutes before slot open"),
    "btc_pre_5m_vol":        ("Spot Price",   "BTC spot volatility in the 5 minutes before slot open"),
    "btc_pre_15m_ret":       ("Spot Price",   "BTC spot return in the 15 minutes before slot open"),
    "btc_pre_15m_vol":       ("Spot Price",   "BTC spot volatility in the 15 minutes before slot open"),
    "btc_pre_30m_ret":       ("Spot Price",   "BTC spot return in the 30 minutes before slot open"),
    "btc_pre_30m_vol":       ("Spot Price",   "BTC spot volatility in the 30 minutes before slot open"),
    "btc_pre_1h_ret":        ("Spot Price",   "BTC spot return in the 1 hour before slot open"),
    "btc_pre_1h_vol":        ("Spot Price",   "BTC spot volatility in the 1 hour before slot open"),
    "btc_pre_4h_ret":        ("Spot Price",   "BTC spot return in the 4 hours before slot open — macro directional context"),
    "btc_pre_4h_vol":        ("Spot Price",   "BTC spot volatility in the 4 hours before slot open"),
    "btc_dist_1k":           ("Spot Price",   "Distance of BTC spot price to the nearest $1,000 round level (0–1)"),
    "btc_dist_5k":           ("Spot Price",   "Distance of BTC spot price to the nearest $5,000 round level (0–1)"),
    "btc_dist_10k":          ("Spot Price",   "Distance of BTC spot price to the nearest $10,000 round level (0–1)"),

    # ── Lag / autocorrelation ───────────────────────────────────────────────
    "lag_1_outcome":         ("Lag",          "Resolution of the immediately preceding slot (1=UP, 0=DOWN)"),
    "lag_2_outcome":         ("Lag",          "Resolution of 2 slots ago"),
    "lag_3_outcome":         ("Lag",          "Resolution of 3 slots ago"),
    "lag_streak":            ("Lag",          "Number of consecutive slots with the same outcome as the most recent — streak length"),

    # ── Time-of-day / calendar ──────────────────────────────────────────────
    "hour_sin":              ("Time",         "Sine encoding of hour-of-day (24h cycle) — captures intraday seasonality"),
    "hour_cos":              ("Time",         "Cosine encoding of hour-of-day"),
    "dow_sin":               ("Time",         "Sine encoding of day-of-week (7-day cycle)"),
    "dow_cos":               ("Time",         "Cosine encoding of day-of-week"),

    # ── Order book (hardcoded 0 in current champion — placeholder for v11+) ─
    "ob_up_bid":             ("Order Book",   "Best bid price of the UP token at observation time (0.5 if unavailable)"),
    "ob_up_ask":             ("Order Book",   "Best ask price of the UP token at observation time (0.5 if unavailable)"),
    "ob_up_spread":          ("Order Book",   "Bid-ask spread of the UP token (0.0 if unavailable)"),
    "ob_implied_prob":       ("Order Book",   "Mid-price implied probability of UP = (bid+ask)/2 (0.5 if unavailable)"),
    "ob_up_bid_depth":       ("Order Book",   "Normalized bid depth within 5¢ of mid for UP token (0.0 if unavailable)"),
    "ob_up_ask_depth":       ("Order Book",   "Normalized ask depth within 5¢ of mid for UP token (0.0 if unavailable)"),
    "ob_up_imbalance":       ("Order Book",   "Order book imbalance = (bid_depth − ask_depth) / total at best level (0.0 if unavailable)"),
}

CATEGORY_ORDER = [
    "CLOB Flow",
    "CLOB Momentum",
    "Sub-window",
    "Z-score",
    "Spot Price",
    "Lag",
    "Time",
    "Order Book",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value: Any, precision: int = 4, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _safe(meta: dict, key: str, default: Any = None) -> Any:
    return meta.get(key, default)


def _get_feature_list(meta: dict) -> list[str]:
    raw = meta.get("feature_list", meta.get("features", []))
    return raw if isinstance(raw, list) else []


def _get_feature_count(meta: dict) -> int | str:
    raw = meta.get("features", meta.get("feature_list", []))
    if isinstance(raw, int):
        return raw
    if isinstance(raw, list):
        return len(raw)
    return "N/A"


# ---------------------------------------------------------------------------
# YAML front-matter
# ---------------------------------------------------------------------------

def _build_frontmatter(meta: dict) -> str:
    auc   = _fmt(_safe(meta, "wf_auc"),   4, "0.0")
    acc   = _fmt(_safe(meta, "wf_acc"),   4, "0.0")
    brier = _fmt(_safe(meta, "wf_brier"), 4, "0.0")
    return f"""\
---
language: en
license: mit
tags:
  - polymarket
  - prediction-markets
  - binary-classification
  - lightgbm
  - order-flow
  - btc
  - crypto
metrics:
  - type: roc_auc
    value: {auc}
  - type: accuracy
    value: {acc}
  - type: brier_score
    value: {brier}
---"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_overview() -> str:
    return """\
## Overview

This model predicts whether the **BTC/USD price will close higher (UP) or lower/flat (DOWN)**
at the end of each 5-minute Polymarket binary resolution window.

It is trained on **CLOB order-flow signals** extracted from Polymarket's BTC 5-min markets,
combined with **BTC spot price context** (returns, volatility, proximity to round levels)
sourced from Binance 1-minute klines.

The primary use-case is to drive a live trading bot that places CLOB limit orders on Polymarket
during the `t = 170–240s` window of each 5-min slot based on real-time signal strength."""


def _section_model_details(meta: dict) -> str:
    version    = _safe(meta, "version", "v10")
    n_feat     = _get_feature_count(meta)
    n_samples  = _safe(meta, "n_samples", _safe(meta, "n_train_samples", "N/A"))
    n_splits   = _safe(meta, "wf_n_splits", 5)
    gap        = _safe(meta, "wf_gap", 5)
    spot_src   = _safe(meta, "spot_source", "spot_price_usdt from CLOB ticks")

    return f"""\
## Model Details

| Field                | Value |
|----------------------|-------|
| Version              | `{version}` |
| Algorithm            | LightGBM binary classifier |
| Calibration          | Isotonic regression (`CalibratedClassifierCV`, method=`isotonic`) |
| Features             | {n_feat} |
| Training samples     | {n_samples} resolved markets |
| Walk-forward folds   | {n_splits} folds, purge gap = {gap} slots |
| Target               | `1` = BTC UP, `0` = BTC DOWN/FLAT in 5 min |
| Observation window   | t = 0–180s (prediction at t ≈ 170–240s) |
| Spot price source    | {spot_src} |"""


def _section_performance(meta: dict) -> str:
    auc         = _fmt(_safe(meta, "wf_auc"),   4)
    acc         = _fmt(_safe(meta, "wf_acc"),   4)
    brier       = _fmt(_safe(meta, "wf_brier"), 4)
    promoted_at = _safe(meta, "promoted_at", "N/A")
    fold_aucs   = _safe(meta, "fold_aucs",   [])
    champ_auc   = _fmt(_safe(meta, "champion_compared_auc"), 4, "N/A")

    fold_block = ""
    if fold_aucs:
        rows = "\n".join(f"| {i+1} | {_fmt(v, 4)} |" for i, v in enumerate(fold_aucs))
        fold_block = f"\n\n### Per-Fold AUCs\n\n| Fold | AUC |\n|------|-----|\n{rows}"

    return f"""\
## Performance

| Metric               | Value |
|----------------------|-------|
| Walk-Forward AUC     | {auc} |
| Walk-Forward Accuracy| {acc} |
| Walk-Forward Brier   | {brier} |
| Gate compared vs     | {champ_auc} (previous champion AUC) |
| Promoted at          | {promoted_at} |
{fold_block}"""


def _section_features(meta: dict) -> str:
    features = _get_feature_list(meta)

    if not features:
        return """\
## Features

Feature list not available in `champion_meta.json`. Run with a bundle that includes `feature_list`."""

    from collections import defaultdict
    groups: dict[str, list[str]] = defaultdict(list)
    unknown: list[str] = []

    for f in features:
        if f in FEATURE_REGISTRY:
            cat = FEATURE_REGISTRY[f][0]
            groups[cat].append(f)
        else:
            unknown.append(f)

    lines = [
        "## Features",
        "",
        f"Total active features: **{len(features)}**",
        "",
        "> Features are selected per training run via permutation importance pruning",
        "> (threshold `imp_mean > 0.0005`). Not all features below are present in every version.",
        "",
    ]

    for cat in CATEGORY_ORDER:
        feats = groups.get(cat, [])
        if not feats:
            continue
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Feature | Description |")
        lines.append("|---------|-------------|")
        for feat in feats:
            _, desc = FEATURE_REGISTRY[feat]
            lines.append(f"| `{feat}` | {desc} |")
        lines.append("")

    if unknown:
        lines.append("### Other")
        lines.append("")
        lines.append("| Feature | Description |")
        lines.append("|---------|-------------|")
        for feat in unknown:
            lines.append(f"| `{feat}` | — |")
        lines.append("")

    return "\n".join(lines)


def _section_feature_engineering() -> str:
    return """\
## Feature Engineering

### Data Sources

| Source | What | Used For |
|--------|------|----------|
| Polymarket CLOB `/trades` | Tick-level trades (price, size_usdc, outcome, side, timestamp) | All CLOB flow / sub-window / z-score features |
| Binance `GET /api/v3/klines` | 1-minute OHLCV for BTCUSDT | All spot price features (returns, volatility, distances) |

### CLOB Tick Processing
- Only trades within `t ∈ [0, 180s)` of each slot's open are used (observation window)
- Volume is measured in **USDC** (`size_usdc = price × size`) — not raw share counts
- `up_ratio` = `vol_up / (vol_up + vol_dn)` — the primary directional signal
- Sub-windows divide the 180s window into 6 × 30s buckets

### Spot Price Processing
- Binance 1m closes are used as a time-series for pre-slot and inslot features
- Pre-slot features use a 4-hour lookback buffer starting before the first slot
- Inslot features use 1m candles within `[slot_ts, slot_ts + 180s]`
- Round-level features (`btc_dist_*`) measure distance from psychological support/resistance

### Cross-Slot Context
- Up-ratio z-scores are computed over rolling windows of 5/10/20 **preceding slots**
- Lag features use the resolved outcome (0/1) of the N previous slots
- Realized volatility uses spot returns of the N preceding 5-min windows
- Purge gap of 5 slots prevents leakage when lag features cross fold boundaries"""


def _section_hyperparameters(meta: dict) -> str:
    params: dict = _safe(meta, "best_params", {})

    if not params:
        return """\
## Hyperparameters

Best hyperparameters not stored in `champion_meta.json` for this version."""

    rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in sorted(params.items()))
    return f"""\
## Hyperparameters

Tuned via **Optuna TPE** sampler — objective = mean walk-forward AUC across all folds.
150 trials per run.

| Parameter | Value |
|-----------|-------|
{rows}"""


def _section_training(meta: dict) -> str:
    version   = _safe(meta, "version", "v10")
    n_samples = _safe(meta, "n_samples", "N/A")
    notes     = _safe(meta, "notes", "")

    return f"""\
## Training

- **Platform**: [Modal.com](https://modal.com) — 8 vCPU, 32 GB RAM, 2h timeout
- **Script**: `scripts/train_{version}_modal.py`
- **Dataset**: Resolved BTC 5-min binary markets (`{n_samples}` samples after feature engineering)
- **Primary data**: Polymarket CLOB tick history (outcome, price, size_usdc per trade)
- **Spot data**: Binance BTCUSDT 1m klines (pre-fetched, stored in Modal Volume)
- **Framework**: LightGBM 4.6.0 + scikit-learn CalibratedClassifierCV (isotonic)
- **Tuning**: Optuna 150 trials, walk-forward AUC objective
- **Promotion gate**: New model must beat current champion on ≥ 2 of 3 metrics (AUC ↑, Brier ↓, Acc ↑)

{f"> **Notes:** {notes}" if notes else ""}"""


def _section_usage() -> str:
    return f"""\
## Usage

{USAGE_SNIPPET}

### Live Inference Notes

The live trader (`deploy/live_trader.py`) replicates the same feature computation:
- Fetches inslot trades from Polymarket data-api (`/trades?asset=<full_token_id>`)
- Fetches current BTC spot from Binance (`/api/v3/ticker/price?symbol=BTCUSDT`)
- Fetches 1m klines from Binance for pre-slot return/vol features
- Maintains a rolling slot history for z-scores, lag features, and realized vol

> **Critical**: feature parity between training and live is enforced manually.
> If a feature is added to training, it **must** be added to `live_trader.py` before deployment."""


def _section_anti_overfitting() -> str:
    return """\
## Anti-Overfitting Measures

| Measure | Detail |
|---------|--------|
| Purged walk-forward CV | Gap of **5 slots** between train tail and test head — prevents leakage through lag features |
| Live champion gate | New model fetches current champion AUC from HuggingFace at runtime — no hardcoded baselines |
| 2-of-3 gate | Must beat champion on AUC **and** at least one of Brier/Acc — single-metric gaming blocked |
| Permutation importance pruning | Features with `imp_mean ≤ 0.0005` are dropped before HPO — reduces noise dimensions |
| No ensembles | Single LightGBM model — ensembles tested in v7, removed due to overfit risk at ~600 samples |
| No lookahead | All CLOB features use `t < observation_end`; all spot features use `t < slot_open` |
| OB sanity check | AUC > 0.99 in any fold triggers abort — guards against silent data loading failures |"""


def _section_lessons() -> str:
    return """\
## Experiment Lessons

Full experiment log: [`docs/EXPERIMENTS.md`](https://github.com/ArtBreguez/polymarket-btc-lab/blob/main/docs/EXPERIMENTS.md)

### What Works

| Finding | Since |
|---------|-------|
| **Isotonic calibration** — superior to sigmoid for ~600 samples (fewer parameters, fits ECE better) | v6 |
| **Purged WF gap=5** — mandatory when using lag features; gap=3 consistently leaks | v6 |
| **btc_up_w5** (last 30s up_ratio) — most important single feature by permutation importance | v8 |
| **Multi-scale z-scores** (5/10/20 slots) — top 3 features are always z-scores | v8 |
| **~30 features** — optimal for 601 samples (ratio ≈ 20:1 samples/feature) | v9 |
| **Optuna on WF AUC** — optimizing fold-average AUC directly beats CV accuracy objective | v9 |
| **min_child_samples 5–40** in Optuna search space — allows model to find right tree depth | v10 |
| **5 folds** — stable estimate for 601 samples; 7 folds makes test sets too small (~85 samples) | v10 |
| **Single regime only** — mixing Apr 2025 + Mar-Apr 2026 data degraded all 3 metrics (v12 lesson) | v13 |

### What Doesn't Work

| Finding | Version |
|---------|---------|
| ❌ ETH/SOL cross-asset features — correlated noise, zero marginal information | v5 |
| ❌ Ensemble (LightGBM + Logistic Regression) — overfit at 601 samples | v7 |
| ❌ OB Down token — `best_bid_size` is always 0 for resolved markets in training data | v7 |
| ❌ Sigmoid/Platt calibration — only 2 degrees of freedom, underfits at this scale | v9 |
| ❌ 63+ features — variance explodes, fold AUC range widens to 0.81–0.91 | v8 |
| ❌ `price_percentile` (4h range position) — no signal at 5-min resolution | v11 |
| ❌ `final_burst` (last 30s volume) — fully redundant with `btc_up_w5_zscore` | v11 |
| ❌ 7-fold WF — test fold ≈ 85 samples, estimates too noisy | v12 |
| ❌ Mixing 2025 + 2026 data — regime shift (volatility, liquidity structure) hurts all metrics | v12 |
| ❌ `spot_price_usdt` from CLOB ticks — 99.6% null in local 2026 data (only websocket ticks had it) | v13 |"""


def _section_changelog(meta: dict) -> str:
    current_version = _safe(meta, "version", "")
    rows = []
    for v, auc, acc, brier, n_samp, changes in CHANGELOG:
        marker = " ◀ current" if v == current_version else ""
        rows.append(f"| {v}{marker} | {auc} | {acc} | {brier} | {n_samp} | {changes} |")
    table = "\n".join(rows)

    return f"""\
## Changelog

| Version | AUC    | Acc    | Brier  | Samples | Key Changes |
|---------|--------|--------|--------|---------|-------------|
{table}"""


def _section_notes() -> str:
    return """\
## Notes & Known Limitations

- **Dataset size**: The HuggingFace training set contains only ~601 resolved BTC 5-min markets (Apr 2025).
  The local 2026 dataset extends this to 7,062 markets but is a different market regime.
  AUC confidence intervals are wide (~±0.02) — report ranges, not point estimates.

- **Regime sensitivity**: Model performance is tied to the market microstructure of its training period.
  Significant changes in Polymarket liquidity, spread structure, or BTC volatility regime may degrade performance.
  Monitor rolling 30-day AUC in production.

- **Spot feature dependency**: Pre-slot and inslot BTC spot features require Binance API access at inference time.
  If Binance is unreachable, these features default to 0.0 — the model still runs but with degraded accuracy.

- **Order book features** (`ob_*`): Currently hardcoded to neutral values (`0.5` / `0.0`) in the current champion.
  Full L2 OB features were tested in v11 but caused live asymmetry (training used 2 snapshots; live had 1).
  Pending fix before re-enabling.

- **Lag features cold start**: The live bot needs ≥ 25 resolved slots of history before lag/z-score features
  stabilize. Fresh deployments may show reduced confidence for the first ~2 hours.

- **Binance geo-block**: Modal's Amsterdam region (used for training) is blocked by Binance (HTTP 451).
  Spot klines must be pre-fetched locally via `scripts/fetch_binance_spot.py` and stored in the Modal Volume."""


# ---------------------------------------------------------------------------
# Card assembler
# ---------------------------------------------------------------------------

def build_model_card(meta: dict) -> str:
    version = _safe(meta, "version", "v10")
    auc     = _fmt(_safe(meta, "wf_auc"), 4)

    sections = [
        _build_frontmatter(meta),
        f"# Polymarket BTC Champion — {version} (WF AUC {auc})",
        "",
        _section_overview(),
        "",
        _section_model_details(meta),
        "",
        _section_performance(meta),
        "",
        _section_features(meta),
        "",
        _section_feature_engineering(),
        "",
        _section_hyperparameters(meta),
        "",
        _section_anti_overfitting(),
        "",
        _section_training(meta),
        "",
        _section_usage(),
        "",
        _section_changelog(meta),
        "",
        _section_lessons(),
        "",
        _section_notes(),
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# HuggingFace I/O
# ---------------------------------------------------------------------------

_meta_for_commit: dict = {}


def _load_meta_from_hf(hf_token: str) -> dict:
    from huggingface_hub import hf_hub_download
    print(f"[update_model_card] Downloading {HF_META_FILENAME} from {HF_REPO_ID} ...")
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_META_FILENAME,
        repo_type="model",
        token=hf_token,
        force_download=True,
    )
    with open(path) as f:
        return json.load(f)


def _upload_readme(readme_content: str, hf_token: str, meta: dict) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    version = _safe(meta, "version", "vN")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp.write(readme_content)
        tmp_path = tmp.name

    print(f"[update_model_card] Uploading {HF_README_FILENAME} to {HF_REPO_ID} ...")
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=HF_README_FILENAME,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=f"Update model card ({version}) — full feature descriptions, feature engineering section",
    )
    Path(tmp_path).unlink(missing_ok=True)
    print("[update_model_card] Done.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_model_card(meta: dict, hf_token: str) -> str:
    readme = build_model_card(meta)
    _upload_readme(readme, hf_token, meta)
    return readme


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and upload a HuggingFace model card for artbreguez/polymarket-btc-model."
    )
    parser.add_argument("--hf-token", required=True, help="HuggingFace write token")
    parser.add_argument("--meta", default=None, help="Path to local champion_meta.json (downloads from HF if omitted)")
    parser.add_argument("--dry-run", action="store_true", help="Generate README but do NOT upload — prints to stdout")
    parser.add_argument("--output", default=None, help="Also write generated README to this local file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.meta:
        meta_path = Path(args.meta)
        if not meta_path.exists():
            print(f"[update_model_card] ERROR: {meta_path} not found.", file=sys.stderr)
            sys.exit(1)
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"[update_model_card] Loaded meta from {meta_path}")
    else:
        meta = _load_meta_from_hf(args.hf_token)

    readme = build_model_card(meta)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(readme)
        print(f"[update_model_card] Wrote README to {out_path}")

    if args.dry_run:
        print("\n" + "=" * 72)
        print(readme)
        print("=" * 72)
        print("[update_model_card] Dry-run — skipping upload.")
        return

    _upload_readme(readme, args.hf_token, meta)


if __name__ == "__main__":
    main()
