"""
update_model_card.py
====================
Generates and uploads a HuggingFace model card (README.md) for
artbreguez/polymarket-btc-model.

Standalone usage:
    python scripts/update_model_card.py --hf-token $HF_TOKEN [--meta champion_meta.json]

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

HF_REPO_ID = "artbreguez/polymarket-btc-model"
HF_META_FILENAME = "champion_meta.json"
HF_README_FILENAME = "README.md"

CHANGELOG = [
    ("v4", "0.8430", "—",      "—",      "Baseline multi-crypto (deprecated)"),
    ("v5", "0.8553", "—",      "—",      "BTC-only, OB ts_ms fix, Optuna WF"),
    ("v6", "0.8559", "0.7738", "0.1562", "Lagged outcomes, purged WF gap=5, perm importance"),
    ("v7", "0.8536", "0.7598", "0.1593", "Realized vol, tw_up_ratio, VWAP trend; ensemble removed"),
    ("v8", "0.8529", "0.7802", "0.1707", "6x30s sub-windows, multi-scale zscore (5/10/20), no ensemble"),
]

USAGE_SNIPPET = '''\
```python
import pickle
from huggingface_hub import hf_hub_download
import pandas as pd

HF_TOKEN = "hf_..."  # your read token

# Load champion bundle
path = hf_hub_download(
    "artbreguez/polymarket-btc-model",
    "champion.pkl",
    token=HF_TOKEN,
)
with open(path, "rb") as f:
    bundle = pickle.load(f)

model    = bundle["model"]     # LightGBM classifier (with isotonic calibration)
features = bundle["features"]  # ordered list of feature names

# Build a one-row feature DataFrame and predict
X = pd.DataFrame([your_feature_dict], columns=features)
prob_up = model.predict_proba(X)[0, 1]  # P(BTC closes UP in 5 min)
print(f"P(UP) = {prob_up:.3f}")
```'''

# Feature category keywords — used to auto-categorize features
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Sub-window",  ["_w0", "_w1", "_w2", "_w3", "_w4", "_w5", "up_w"]),
    ("Z-score",     ["zscore", "z5", "z10", "z20", "tw_up"]),
    ("Spot",        ["spot_", "ret_", "vol_", "btc_"]),
    ("Orderbook",   ["bid", "ask", "spread", "depth", "imbalance", "ob_"]),
    ("Lag",         ["lag_", "streak", "prev_", "outcome_"]),
    ("Time",        ["hour", "dow", "sin", "cos", "round_"]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _categorize_feature(name: str) -> str:
    nl = name.lower()
    for category, keywords in _CATEGORY_RULES:
        if any(kw in nl for kw in keywords):
            return category
    return "Classic OB / Other"


def _fmt(value: Any, precision: int = 4, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _safe(meta: dict, key: str, default: Any = None) -> Any:
    return meta.get(key, default)


# ---------------------------------------------------------------------------
# YAML front-matter builder
# ---------------------------------------------------------------------------

def _build_frontmatter(meta: dict) -> str:
    auc    = _fmt(_safe(meta, "wf_auc"),    4, "0.0")
    acc    = _fmt(_safe(meta, "wf_acc"),    4, "0.0")
    brier  = _fmt(_safe(meta, "wf_brier"),  4, "0.0")

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
---\
"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_overview() -> str:
    return """\
## Overview

This model predicts whether the BTC/USD price will be **higher (UP) or lower/flat
(DOWN)** at the close of each 5-minute Polymarket resolution window.

It is trained exclusively on **order-flow signals** extracted from the Polymarket
CLOB (Central Limit Order Book) for BTC 5-min binary markets, supplemented by BTC
spot price context (returns, volatility) sourced from a public dataset.

The primary use-case is to drive the **live_trader.py** bot, which places CLOB
limit orders on Polymarket during the t = 170–240 s window of each 5-min slot.\
"""


def _section_model_details(meta: dict) -> str:
    version   = _safe(meta, "version",        "v8")
    algorithm = _safe(meta, "algorithm",      "LightGBM + isotonic calibration")
    # "features" key can be an int (count) or list — handle both
    _feat_val = meta.get("features", meta.get("feature_list", []))
    _feat_count = _feat_val if isinstance(_feat_val, int) else len(_feat_val)
    n_feat    = _safe(meta, "n_features", _feat_count or "N/A")
    n_samples = _safe(meta, "n_samples", _safe(meta, "n_train_samples", "N/A"))

    return f"""\
## Model Details

| Field             | Value |
|-------------------|-------|
| Version           | {version} |
| Algorithm         | {algorithm} |
| Number of features| {n_feat} |
| Training samples  | {n_samples} |
| Calibration       | Isotonic regression (sklearn CalibratedClassifierCV) |
| Target            | Binary: 1 = BTC UP, 0 = BTC DOWN/FLAT in 5 min |
| Prediction window | t = 0–300 s (prediction made at t ≈ 170–240 s) |\
"""


def _section_performance(meta: dict) -> str:
    auc         = _fmt(_safe(meta, "wf_auc"),   4)
    acc         = _fmt(_safe(meta, "wf_acc"),   4)
    brier       = _fmt(_safe(meta, "wf_brier"), 4)
    promoted_at = _safe(meta, "promoted_at", "N/A")
    fold_aucs   = _safe(meta, "fold_aucs",   [])

    fold_rows = ""
    if fold_aucs:
        rows = [f"| {i+1} | {_fmt(v, 4)} |" for i, v in enumerate(fold_aucs)]
        fold_rows = "\n### Per-Fold AUCs\n\n| Fold | AUC |\n|------|-----|\n" + "\n".join(rows)

    return f"""\
## Performance

| Metric            | Value |
|-------------------|-------|
| Walk-Forward AUC  | {auc} |
| Walk-Forward Acc  | {acc} |
| Walk-Forward Brier| {brier} |
| Promoted at       | {promoted_at} |
{fold_rows}\
"""


def _section_features(meta: dict) -> str:
    # "features" can be int (count) or list — prefer "feature_list"
    raw = meta.get("feature_list", meta.get("features", []))
    features: list[str] = raw if isinstance(raw, list) else []

    if not features:
        return """\
## Features

Feature list not available in champion_meta.json.
"""

    # Group by category
    from collections import defaultdict
    groups: dict[str, list[str]] = defaultdict(list)
    for f in features:
        groups[_categorize_feature(f)].append(f)

    lines = ["## Features", "", f"Total features: **{len(features)}**", ""]
    for cat, feats in groups.items():
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Feature | Description |")
        lines.append("|---------|-------------|")
        for feat in feats:
            lines.append(f"| `{feat}` | — |")
        lines.append("")

    return "\n".join(lines)


def _section_hyperparameters(meta: dict) -> str:
    params: dict = _safe(meta, "best_params", {})

    if not params:
        return """\
## Hyperparameters

Best hyperparameters not available in champion_meta.json.
"""

    rows = [f"| `{k}` | {v} |" for k, v in sorted(params.items())]
    table = "\n".join(rows)

    return f"""\
## Hyperparameters

Tuned via Optuna (walk-forward cross-validation objective = mean AUC).

| Parameter | Value |
|-----------|-------|
{table}\
"""


def _section_anti_overfitting() -> str:
    return """\
## Anti-Overfitting Measures

| Measure | Detail |
|---------|--------|
| Purged walk-forward | Gap of 5 slots between train tail and test head to prevent leakage across adjacent markets |
| Champion gate | New model must exceed the **current champion's AUC** fetched live from HuggingFace before promotion |
| Fair gate | Champion AUC is re-evaluated on the same test folds as the challenger to ensure comparability |
| No ensemble | Single LightGBM model only — ensembles were removed in v7 to reduce complexity and overfit risk |
| Feature importance | Permutation importance used (v6+) to prune uninformative features before each run |
| No look-ahead | All features computed from data strictly before slot open; spot context uses pre-slot OHLCV only |\
"""


def _section_training(meta: dict) -> str:
    n_samples = _safe(meta, "n_train_samples", "N/A")
    version   = _safe(meta, "version", "v8")

    return f"""\
## Training

- **Cloud platform**: [Modal.com](https://modal.com) — training script `scripts/train_{version}_modal.py`
- **Dataset**: [`BrockMisner/polymarket-btc-updown`](https://huggingface.co/datasets/BrockMisner/polymarket-btc-updown) on HuggingFace
- **Dataset size**: 616 resolved BTC 5-min binary markets (~{n_samples} labeled samples after feature engineering)
- **Framework**: LightGBM 4.x + scikit-learn CalibratedClassifierCV (isotonic)
- **Tuning**: Optuna TPE sampler, 50–100 trials, walk-forward objective
- **Hardware**: Modal CPU container (2 vCPU, 4 GB RAM), typical runtime < 15 min\
"""


def _section_usage() -> str:
    return f"""\
## Usage

{USAGE_SNIPPET}\
"""


def _section_changelog() -> str:
    rows = [
        f"| {v} | {auc} | {acc} | {brier} | {changes} |"
        for v, auc, acc, brier, changes in CHANGELOG
    ]
    table = "\n".join(rows)

    return f"""\
## Changelog

| Version | AUC    | Acc    | Brier  | Key Changes |
|---------|--------|--------|--------|-------------|
{table}\
"""


def _section_notes() -> str:
    return """\
## Notes & Known Limitations

- **Small dataset**: Only 616 resolved BTC 5-min markets are currently available.
  This limits statistical power; AUC confidence intervals are wide (~±0.02).
- **No tick-level history**: The public dataset does not include historical tick data,
  preventing expansion via synthetic sampling or data augmentation.
- **Oracle agreement**: Polymarket oracle resolves ~98 % of markets within 30 s of
  close; ~2 % are disputed or delayed, introducing occasional label noise.
- **Market microstructure shift**: Polymarket CLOB spreads and liquidity conditions
  change over time. Model performance may degrade if market structure shifts
  significantly from the training period.
- **Single asset**: The model is BTC-specific. Applying it to ETH or other assets
  without retraining is not recommended.\
"""


# ---------------------------------------------------------------------------
# Card assembler
# ---------------------------------------------------------------------------

def build_model_card(meta: dict) -> str:
    version = _safe(meta, "version", "v8")
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
        _section_hyperparameters(meta),
        "",
        _section_anti_overfitting(),
        "",
        _section_training(meta),
        "",
        _section_usage(),
        "",
        _section_changelog(),
        "",
        _section_notes(),
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# HuggingFace I/O
# ---------------------------------------------------------------------------

def _load_meta_from_hf(hf_token: str) -> dict:
    from huggingface_hub import hf_hub_download

    print(f"[update_model_card] Downloading {HF_META_FILENAME} from {HF_REPO_ID} ...")
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_META_FILENAME,
        repo_type="model",
        token=hf_token,
    )
    with open(path, "r") as f:
        return json.load(f)


def _upload_readme(readme_content: str, hf_token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp.write(readme_content)
        tmp_path = tmp.name

    print(f"[update_model_card] Uploading {HF_README_FILENAME} to {HF_REPO_ID} ...")
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=HF_README_FILENAME,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=f"Auto-update model card ({_safe(meta_for_commit, 'version', 'vN')})",
    )
    Path(tmp_path).unlink(missing_ok=True)
    print("[update_model_card] Done.")


# Module-level variable used to pass version into commit message — set in update_model_card()
meta_for_commit: dict = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_model_card(meta: dict, hf_token: str) -> str:
    """
    Generate a model card from *meta* and upload it to HuggingFace.

    Parameters
    ----------
    meta:      champion_meta.json as a Python dict.
    hf_token:  HuggingFace write token.

    Returns
    -------
    The generated README.md content as a string.
    """
    global meta_for_commit
    meta_for_commit = meta

    readme = build_model_card(meta)
    _upload_readme(readme, hf_token)
    return readme


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and upload a HuggingFace model card for artbreguez/polymarket-btc-model."
    )
    parser.add_argument(
        "--hf-token",
        required=True,
        help="HuggingFace write token (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help=(
            "Path to a local champion_meta.json. "
            "If omitted, the file is downloaded from HuggingFace."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the README but do NOT upload it. Prints to stdout.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Also write the generated README to this local path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── load meta ──────────────────────────────────────────────────────────
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

    # ── build card ─────────────────────────────────────────────────────────
    readme = build_model_card(meta)

    # ── output ─────────────────────────────────────────────────────────────
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

    # ── upload ─────────────────────────────────────────────────────────────
    global meta_for_commit
    meta_for_commit = meta
    _upload_readme(readme, args.hf_token)


if __name__ == "__main__":
    main()
