# 04 — Model Evaluation

## Metrics

| Metric | Direction | Purpose |
|--------|-----------|---------|
| WF AUC | Higher better | Primary gate metric — discrimination ability |
| Brier Score | Lower better | Calibration quality (predicted prob vs actual outcome) |
| Accuracy | Higher better | At 0.5 threshold — simple classification correctness |
| Fold AUC range | Tighter better | Stability across temporal splits |

**WF AUC** is the gating metric for champion promotion. A model must beat the current champion on at least 2/3 metrics (AUC, Brier, Accuracy) to promote.

---

## Walk-Forward (WF) Protocol

Purged walk-forward cross-validation prevents temporal leakage:

1. **5 folds** — data sorted chronologically, split into 5 sequential blocks
2. **Gap = 5 slots** — 5 markets (~25 min) purged between train/test to prevent lag feature leakage
3. Each fold trains on all prior data and tests on the next block
4. Metrics are averaged across all 5 test folds

Why not 7 folds? Tested in v12 — each fold gets only ~85 test samples (vs ~120 with 5 folds), producing noisier AUC estimates. Stick with 5.

Why gap=5? Lag features (lag_1/2/3_outcome, lag_streak) look back up to 3 slots. Gap=5 guarantees no information leaks from test into train through lag features.

---

## Fold Interpretation

Example fold AUCs from v18: [0.8831, 0.8877, 0.9006, 0.9020, 0.9099]

- **Improving trend** (fold 1 < fold 5) → model generalizes better on recent data
- **Wide range** (e.g., v8: 0.814–0.908) → too many features, overfitting risk
- **Tight range** (e.g., v18: 0.883–0.910) → stable, well-regularized model

If any single fold AUC drops below 0.75, investigate whether that time period has unusual market conditions (low liquidity, regime change).

---

## Sanity Check Protocol

Run by `scripts/validate_model.py` in CI before every deploy:

1. **Download** champion.pkl from HuggingFace (`artbreguez/polymarket-btc-model`)
2. **Feature count** > 0
3. **WF AUC** >= 0.65 (minimum deployment bar)
4. **Directional probes** — 4 synthetic scenarios:
   - **Neutral**: up_ratio=0.50, momentum=0.0 → baseline P(UP)
   - **UP scenario**: up_ratio=0.70, momentum=+0.15 → must produce P(UP) > neutral
   - **DOWN scenario**: up_ratio=0.30, momentum=-0.15 → must produce P(UP) < neutral
   - **Strong UP**: up_ratio=0.80, momentum=+0.25 → must produce P(UP) >= 0.40

If any check fails, CI blocks deployment.

---

## Champion Progression: v4 → v18

| Version | AUC | Acc | Brier | Features | Samples | Key Change |
|---------|-----|-----|-------|----------|---------|------------|
| v4 | 0.843 | — | — | ~40 | 601 | Multi-crypto baseline (ETH/SOL — later removed) |
| v5 | 0.8553 | — | — | — | 601 | BTC-only, OB ts_ms fix |
| v6 | 0.8559 | 0.7738 | **0.1562** | — | 601 | Lag outcomes, purged WF gap=5 |
| v7 | 0.8536 | 0.7598 | 0.1593 | — | 601 | Time-weighted flow, VWAP trend, ensemble tested & rejected |
| v8 | 0.8529 | 0.7802 | 0.1707 | 63 | 601 | 6x30s sub-windows, multi-scale zscore |
| v9 | 0.8519 | 0.7842 | 0.1809 | 27 | 601 | Sigmoid calibration (worse Brier — reverted) |
| v10 | **0.8547** | **0.7902** | **0.1554** | 31 | 601 | Isotonic back, interaction features, gate fix |
| v11 | 0.8533 | 0.7839 | 0.1585 | 22 | 601 | price_percentile + final_burst (both dropped) |
| v12 | 0.8542 | 0.7762 | 0.1634 | 38 | 601 | 7-fold WF (worse — reverted to 5) |
| v13-v16 | ~0.85 | — | — | ~30 | 601+ | Iterative improvements, expanded lag context |
| v17 | 0.8925 | 0.8032 | 0.1342 | ~30 | 7,062 | Extended lag context + temporal features |
| **v18** | **0.8966** | **0.8104** | **0.1318** | **30** | **22,319** | 3x data expansion, pre-fetched Binance spot |

Key inflection points:
- **v5**: BTC-only scope established
- **v10**: Gate bug fixed, interaction features, isotonic calibration locked in
- **v17**: Extended lag context from 15k tickless markets
- **v18**: 3x data expansion (22k markets, 68M ticks) — more data > better features

---

## Gate Bug (Historical)

v11 and v12 promoted incorrectly because the gate re-evaluated the champion with default LightGBM params (not Optuna-tuned), deflating champion AUC from ~0.85 to ~0.82. Fixed in commit 84b7bbf: gate now reads `CHAMPION_AUC` from `champion_meta.json` directly.
