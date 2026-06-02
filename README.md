# polymarket-btc-lab

ML pipeline for predicting UP/DOWN outcomes on Polymarket BTC 5-minute binary markets.

**Signal source:** Polymarket CLOB order flow (first 3 minutes of each 5-minute slot)  
**Model:** LightGBM, trained on Modal.com cloud GPU  
**Champion:** v8 — AUC 0.8529, Acc 0.7802 (63 features, purged walk-forward CV)  
**Live deployment:** Fly.io (`polymarket-maker-mm`), auto-deployed via GitHub Actions

---

## Overview

Each Polymarket BTC 5-minute market resolves UP or DOWN based on the Chainlink oracle price at slot end. This pipeline:

1. Ingests order flow ticks from the Polymarket CLOB during `[t=0, t=180s)` of each slot
2. Computes 63 features (sub-window flow, multi-scale zscore, VWAP, realized vol, lag outcomes, spot context)
3. Trains a LightGBM classifier with Optuna-tuned hyperparameters using purged walk-forward CV
4. Promotes the winning model to HuggingFace if it beats the current champion
5. Deploys the live trader to Fly.io, which places orders at `t=170–240s` of each slot

---

## Repository Structure

```
scripts/
  train_v5_modal.py        # v5: BTC-only, OB ts_ms fix, Optuna WF objective
  train_v6_modal.py        # v6: lagged outcomes (lag1/2/3), lag streak, purged WF gap=5
  train_v7_modal.py        # v7: realized vol, tw_up_ratio, vwap_trend
  train_v8_modal.py        # v8: CURRENT — 6x30s sub-windows, multi-scale zscore
  promote_champion.py      # sanity-check + upload local pkl to HF as champion
  validate_model.py        # CI validation (called by GitHub Actions)
  build_dataset_modal.py   # dataset exploration utilities

deploy/
  live_trader.py           # live prediction + order placement (Fly.io)
  Dockerfile
  fly.toml
  requirements.txt

src/btc_lab/
  features.py              # feature computation (shared between training and live)
  plugin.py                # BtcUpDownPlugin for pmlab framework
  config.py                # paths and constants

tests/
  test_features.py

.github/workflows/
  ci.yml                   # validate_model.py on every push
  deploy.yml               # deploy to Fly.io when champion.pkl changes on HF

artifacts/                 # local model pkls and datasets (gitignored)
```

---

## Quick Start

```bash
uv sync
```

**Train** (runs on Modal cloud, ~15 min, ~$0.20):
```bash
modal run scripts/train_v8_modal.py
```

**Promote** a local model to HuggingFace champion:
```bash
python scripts/promote_champion.py --model artifacts/btc_model_v8.pkl
```

**Live trader** is auto-deployed to Fly.io via GitHub Actions when `champion.pkl` changes on HuggingFace. No manual deploy step needed.

---

## Infrastructure

| Component | Service |
|-----------|---------|
| Training | [Modal.com](https://modal.com) (cloud GPU) |
| Model registry | HuggingFace — `artbreguez/polymarket-btc-model` (private) |
| Dataset | HuggingFace — `BrockMisner/polymarket-btc-updown` (private mirror) |
| Live deployment | Fly.io — `polymarket-maker-mm` app |
| CI/CD | GitHub Actions |

---

## Dataset

- **616** resolved BTC 5-minute Polymarket markets (all available in BrockMisner)
- Features computed from CLOB ticks in `[t=0, t=180s)` of each slot
- **Label:** 1 = BTC UP at resolution (Chainlink oracle), 0 = DOWN
- Oracle agreement: Binance 85.4%, Chainlink first_vs_last 97.4%
- Dataset grows organically — no historical tick data exists for other markets

---

## Model Champion Progression

All versions use purged walk-forward CV with `gap=5` slots.

| Version | AUC | Acc | Brier | Notes |
|---------|-----|-----|-------|-------|
| v4 | 0.843 | — | — | Baseline; multi-crypto (ETH/SOL features); deprecated |
| v5 | 0.8553 | — | — | BTC-only; OB `ts_ms` fix; Optuna WF objective |
| v6 | 0.8559 | 0.7738 | 0.1562 | Lagged outcomes (lag1/2/3), lag streak, permutation importance |
| v7 | 0.8536 | 0.7598 | 0.1593 | Up/Down OB split, realized vol, `tw_up_ratio`, `vwap_trend`; ensemble LightGBM+LR tested but hurt |
| **v8** | **0.8529** | **0.7802** | **0.1707** | **CURRENT CHAMPION** — 63 features, 6×30s sub-windows, multi-scale zscore |

> v8 fold AUCs: `[0.824, 0.908, 0.872, 0.814, 0.847]`

---

## v8 — Current Champion

### Features (63 total)

| Group | Features |
|-------|----------|
| Sub-window order flow (6×30s) | `btc_up_w0..w5`, `btc_up_w0_zscore..w5_zscore` |
| Multi-scale zscore | `btc_up_ratio_zscore_5s/10s/20s`, `btc_up_ratio_hist_mean_5s/20s` |
| Time-weighted | `btc_tw_up_ratio`, `btc_vwmom`, `btc_vwap_trend` |
| Classic OB | `btc_up_ratio`, `btc_momentum`, `btc_vwap_spread`, `btc_vwap_up/dn`, `btc_buy_ratio`, `btc_avg_size`, `btc_vol_up/dn`, `btc_vol_ratio`, `btc_n_ticks` |
| Realized vol | `btc_realized_vol_5s`, `btc_realized_vol_10s`, `btc_tick_accel` |
| Spot context | `btc_inslot_ret/vol`, `btc_pre_5m/15m/30m/1h/4h_ret/vol` |
| Orderbook (UP side only) | `ob_up_bid/ask/spread/bid_depth/ask_depth/imbalance`, `ob_implied_prob` |
| Lag outcomes | `lag_1/2/3_outcome`, `lag_streak` |
| Round numbers | `btc_dist_1k`, `btc_dist_5k`, `btc_dist_10k` |
| Time | `hour_sin/cos`, `dow_sin/cos` |

### Top Features (permutation importance)

1. `btc_up_ratio_zscore`
2. `btc_up_w2`
3. `btc_up_w1`
4. `btc_up_w5`
5. `btc_tw_up_ratio`
6. `btc_vwap_trend`
7. `btc_vwmom`
8. `btc_momentum`
9. `btc_vwap_spread`

### Best Hyperparameters

```python
n_estimators     = 198
num_leaves       = 27
min_child_samples = 41
subsample        = 0.534
colsample_bytree = 0.428
reg_alpha        = 0.978
reg_lambda       = 1.108
learning_rate    = 0.00617
```

---

## Anti-Overfitting Measures

- **Purged walk-forward CV** with `gap=5` slots — prevents leakage from lag features
- **Optuna optimizes on WF AUC** (not held-out CV); falls back to baseline if Optuna overfits
- **OB Down token dropped** — `best_bid_size` is always `0.0` for resolved markets (pure noise)
- **No ensemble** — LightGBM+LR combination tested at v7 but hurt performance (0.8479 vs 0.8536 solo)
- **Champion AUC read from HF metadata** — never hardcoded
- **Gated comparison** — re-evaluates current champion with the same purged WF before comparing to challenger

---

## Live Trading Strategy

- **Entry window:** `t = 170–240s` into each 5-minute slot
- **Signal:** model confidence exceeding a configurable threshold
- **Stake:** configurable USDC per trade
- **Order placement:** Polymarket SDK via `BuilderApiKey` (no direct CLOB API calls needed)

---

## Known Pitfalls

**Polymarket data-api**
- Trades are returned newest-first; time filter params are silently ignored — must paginate via `offset`

**Parquet type mismatch**
- `orderbook.parquet` `market_id` is `STRING`; `markets.parquet` `market_id` is `INT` — PyArrow will OOM if the filter is not cast before join

**OB Down token**
- `best_bid_size` for the DOWN token is always `0.0` in resolved markets — drop from features entirely

**Dollar values**
- `to_df()` already returns dollar values; never divide by `1e9`

**Databento**
- `stype_in='continuous'` is required for `.c.0` continuous symbols

**Dataset growth**
- The dataset can only expand organically as new markets resolve; no historical tick data exists for backfilling other markets

---

## CI/CD

**`ci.yml`** — runs on every push:
```
validate_model.py → loads champion from HF, runs sanity checks
```

**`deploy.yml`** — triggers when `champion.pkl` changes on HuggingFace:
```
docker build → fly deploy → polymarket-maker-mm (Fly.io)
```

---

## Secrets Required

| Secret | Used by |
|--------|---------|
| `HF_TOKEN` | Training, promote, CI, deploy |
| `POLYMARKET_API_KEY` / `BuilderApiKey` | Live trader |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | Training |
| `FLY_API_TOKEN` | GitHub Actions deploy |
