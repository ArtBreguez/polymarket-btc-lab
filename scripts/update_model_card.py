"""
update_model_card.py — Gera e sobe o README.md do modelo no HF
===============================================================
Uso:
    python scripts/update_model_card.py
    python scripts/update_model_card.py --dry-run
"""
from __future__ import annotations
import argparse, os, sys, tempfile
from pathlib import Path

HF_REPO_ID = "artbreguez/polymarket-btc-model"

# ── Changelog ────────────────────────────────────────────────────────────────
CHANGELOG = [
    ("v4",  "0.843", "—",     "—",     "601",   "Baseline multi-crypto — ETH/SOL noise → deprecated"),
    ("v5",  "0.855", "—",     "—",     "601",   "BTC-only, OB ts_ms fix, Optuna WF objective"),
    ("v6",  "0.856", "0.774", "0.156", "601",   "Lag outcomes, purged WF gap=5 — best Brier ever"),
    ("v7",  "0.854", "0.760", "0.159", "601",   "Realized vol, tw_up_ratio — ensemble removed"),
    ("v8",  "0.853", "0.780", "0.171", "601",   "6×30s sub-windows, multi-scale zscore — 63 features"),
    ("v10", "0.855", "0.790", "0.155", "601",   "Isotonic calib, interaction feats — prev champion"),
    ("v17", "0.887", "0.806", "0.143", "15257", "Extended lags + temporal features + expanded dataset"),
    ("v18", "0.898", "0.813", "0.129", "22319", "22k markets, 40 features, walk-forward validation"),
    ("v19", "0.900", "0.813", "0.129", "22319", "Real L2 orderbook features from pmdata"),
    ("v21", "0.900", "0.813", "0.129", "22319", "Ablation: pruned to 30 features, same AUC"),
    ("v26", "0.856", "0.779", "0.183", "22319", "40 real-time features, OBS_SECS=60 — no OB temporal"),
    ("v28", "0.848", "0.773", "0.184", "39197", "73 features — inflated by OB leakage (t=108-168s)"),
    ("v29", "0.792", "0.710", "0.186", "39197", "**CHAMPION** — leakage removed, 20 clean features, OBS_SECS=60"),
]

# ── Feature registry v29 (20 features ativas) ────────────────────────────────
FEATURES_V29 = [
    # (name, group, description)
    ("btc_inslot_ret",         "Spot",     "BTC spot return during t=[0,60s] (close/open − 1)"),
    ("btc_inslot_range",       "Spot",     "(high − low) / close during t=[0,60s]"),
    ("btc_pre_5m_ret",         "Spot",     "BTC spot return in the 5min before slot open"),
    ("btc_dist_1k",            "Spot",     "Distance to nearest $1k round level (0–0.5)"),
    ("btc_spot_vol_ratio",     "Spot",     "Volume in this 5min slot / mean 5min volume over last 1h"),
    ("ob_imbalance",           "Orderbook","bid_vol / (bid_vol + ask_vol) — snapshot open t~5-10s"),
    ("ob_depth_ratio",         "Orderbook","bid_depth / ask_depth within ±5¢ of mid — snapshot open"),
    ("ob_total_depth",         "Orderbook","Total bid + ask depth — snapshot open"),
    ("clob_spread_mean",       "CLOB",     "Mean bid-ask spread over t=[0,60s] from WS stream"),
    ("clob_spread_trend",      "CLOB",     "Linear slope of spread over t=[0,60s]"),
    ("clob_mid_volatility",    "CLOB",     "Std of mid-price changes over t=[0,60s]"),
    ("clob_ask_pressure",      "CLOB",     "Fraction of ask-side price moves that are downward"),
    ("btc_up_w1",              "Flow",     "UP-ratio in the second half of the 60s window"),
    ("btc_size_disparity",     "Flow",     "abs(mean_buy_size − mean_sell_size) / mean_size"),
    ("btc_up_ratio_zscore_5s", "Flow",     "Z-score of current up_ratio vs last 5 slots"),
    ("prev_slot_up_ratio_3",   "Lag",      "UP-ratio of the slot 3 periods ago (t−15min)"),
    ("prev_slot_up_ratio_5",   "Lag",      "UP-ratio of the slot 5 periods ago (t−25min)"),
    ("lag_ur_zscore_20",       "Lag",      "Z-score of prev slot up_ratio vs 20-slot rolling window"),
    ("x_imb_x_ur",             "Cross",    "ob_imbalance × btc_up_ratio — order book × flow interaction"),
    ("x_depth_x_vol",          "Cross",    "ob_depth_ratio × btc_vol_1h — liquidity × volatility interaction"),
]

# ── Leakage removidas ────────────────────────────────────────────────────────
LEAKAGE_REMOVED = [
    ("ob_mid",           "0.60", "OB snapshot capturado em t=108-168s no training, t<60s no live"),
    ("ob_imbalance_end", "—",    "Snapshot 'end' usa dados do slot completo (t>60s)"),
    ("ob_spread_end",    "—",    "Idem"),
    ("ob_depth_change",  "—",    "end − open: depende do snapshot end"),
    ("ob_imb_momentum",  "—",    "Idem"),
    ("clob_mid_velocity","—",    "Slope calculado sobre slot completo no training"),
    ("ob_weighted_imb",  "—",    "Inclui dados pós-60s"),
    ("ob_bid_depth_5c",  "—",    "Snapshot contaminado"),
    ("ob_ask_depth_5c",  "—",    "Snapshot contaminado"),
]

CARD_TEMPLATE = """\
---
language: en
license: mit
tags:
  - prediction-market
  - lightgbm
  - binary-classification
  - btc
  - polymarket
  - tabular
datasets:
  - artbreguez/polymarket-btc-model
---

# Polymarket BTC 5-min Prediction Model — {version}

**LightGBM classifier** that predicts whether BTC will close UP or DOWN
in the next 5-minute Polymarket slot.

| Metric | Value |
|--------|-------|
| Version | `{version}` |
| WF AUC | **{wf_auc}** |
| Brier score | {brier} |
| Accuracy | {acc} |
| Features | {n_features} (auto-selected, zero lookahead) |
| Training markets | {n_markets:,} |
| OBS window | 60s (t=0..60s, zero leakage) |
| Holdout | 7-10 Jun 2026 (never seen by model) |

> **Backtest metrics (holdout)** will be added here after the holdout pipeline completes.

---

## Leakage audit — v29

Previous versions (v26-v28) achieved inflated AUC (~0.85) due to OB features
captured at t=108-168s in training but only at t<60s in live trading.
The `ob_mid` feature alone had correlation=0.60 with the target.

**9 features removed in v29:**

| Feature | Corr(target) | Reason |
|---------|-------------|--------|
{leakage_rows}

After removal, AUC dropped from 0.848 → **0.792** (honest baseline).
Win rate dropped from 96.8% → realistic levels.

---

## Features (v29 — 20 active)

All features computed strictly within t=[0, 60s] of slot start.
Auto-selected by permutation importance (threshold = 15% of median importance).

| # | Feature | Group | Description |
|---|---------|-------|-------------|
{feature_rows}

### Data sources

| Source | Data | Latency |
|--------|------|---------|
| Binance WebSocket | BTC/USDT 1m klines + trades | ~real-time |
| Polymarket CLOB WebSocket | Book snapshots, price_change events | ~real-time |
| Polymarket data-api | Slot ticks (historical) | ~120s lag |
| CLOB REST /book | OB snapshot on-demand | ~100-300ms |

---

## Dataset

- **Training:** `data/all_markets.csv` + `data/new_markets.csv` — 39,197 resolved markets (Mar–Jun 2026)
- **Holdout:** `data/holdout_markets.csv` — 7-10 Jun 2026, never seen by v29
- **Ticks:** `data/ticks_btc_full_clean.parquet` + `data/new_ticks_pmdata.parquet`
- **OB features:** `data/ob_features_full.parquet` (pmdata.dev poly_l2, t<60s)
- **Spot:** `data/binance_spot_full.parquet` (Binance 1m klines)

See [`docs/holdout_policy.md`](docs/holdout_policy.md) for the holdout protocol.

---

## Usage

```python
import pickle
from huggingface_hub import hf_hub_download
import pandas as pd

# Download champion bundle (requires HF token for private repo)
path = hf_hub_download(
    repo_id="artbreguez/polymarket-btc-model",
    filename="champion.pkl",
    token="hf_...",
)
with open(path, "rb") as f:
    bundle = pickle.load(f)

model    = bundle["model"]     # CalibratedClassifierCV(LightGBM)
features = bundle["features"]  # list of 20 feature names
version  = bundle["version"]   # "v29_20f_rt"
wf_auc   = bundle["wf_auc"]    # 0.7918

# Build feature dict (must match features list exactly)
feat_dict = {{...}}  # compute your features here
X = pd.DataFrame([feat_dict])[features].fillna(0.0)
prob_up = model.predict_proba(X)[0, 1]   # P(BTC closes UP)
print(f"P(UP) = {{prob_up:.3f}}")
```

**Live trading filters applied on top of model output:**
- `prob_up >= 0.55` (confidence)
- `edge = prob_up - ask >= 0.07`
- `ask ∈ [0.42, 0.65]`
- `n_ticks >= 50` in slot (data completeness gate)

---

## Version history

| Version | AUC | Acc | Brier | Markets | Notes |
|---------|-----|-----|-------|---------|-------|
{changelog_rows}

---

## Repository structure

```
champion.pkl          — Model bundle (model + features list + metadata)
champion_meta.json    — Training metadata (JSON)
data/
  all_markets.csv             — Training markets
  new_markets.csv             — Additional training markets
  holdout_markets.csv         — Holdout (7-10 Jun, never seen by v29)
  holdout_ticks.parquet       — Holdout ticks
  holdout_ob_features.parquet — Holdout OB features
  ticks_btc_full_clean.parquet
  new_ticks_pmdata.parquet
  binance_spot_full.parquet
  binance_spot_local.parquet
  ob_features_full.parquet
```

---

*Model trained and deployed by [@artbreguez](https://github.com/artbreguez).
Live bot running on Fly.io (AMS). Feature parity enforced: train == live.*
"""


def build_card(meta: dict) -> str:
    version    = meta.get("version", "v29_20f_rt")
    wf_auc     = f'{meta.get("wf_auc", 0.7918):.4f}'
    brier      = f'{meta.get("brier", 0.1857):.4f}'
    acc        = f'{meta.get("acc", 0.7097):.4f}'
    n_features = meta.get("n_features", 20)
    n_markets  = meta.get("n_markets", 39197)

    feature_rows = "\n".join(
        f"| {i+1} | `{name}` | {group} | {desc} |"
        for i, (name, group, desc) in enumerate(FEATURES_V29)
    )

    leakage_rows = "\n".join(
        f"| `{feat}` | {corr} | {reason} |"
        for feat, corr, reason in LEAKAGE_REMOVED
    )

    changelog_rows = "\n".join(
        f"| {v} | {auc} | {ac} | {br} | {mkts} | {notes} |"
        for v, auc, ac, br, mkts, notes in CHANGELOG
    )

    return CARD_TEMPLATE.format(
        version=version,
        wf_auc=wf_auc,
        brier=brier,
        acc=acc,
        n_features=n_features,
        n_markets=n_markets,
        feature_rows=feature_rows,
        leakage_rows=leakage_rows,
        changelog_rows=changelog_rows,
    )


def update_model_card(meta: dict | None = None, hf_token: str = "", dry_run: bool = False):
    if meta is None:
        meta = {
            "version": "v29_20f_rt",
            "wf_auc": 0.7918,
            "brier": 0.1857,
            "acc": 0.7097,
            "n_features": 20,
            "n_markets": 39197,
        }

    card = build_card(meta)

    if dry_run:
        print(card)
        return

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(card)
        tmp_path = f.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=f"docs: update model card — {meta.get('version', 'v29')}",
    )
    Path(tmp_path).unlink()
    print(f"Model card updated on HF: {HF_REPO_ID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hf-token", default="")
    args = parser.parse_args()

    token = args.hf_token
    if not token:
        # Tentar .env local
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "HF_TOKEN" in line:
                    token = line.split("=", 1)[1].strip()
                    break

    update_model_card(hf_token=token, dry_run=args.dry_run)
