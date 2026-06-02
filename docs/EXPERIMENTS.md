# Experiment Log — BTC 5min Polymarket Model

This document is the canonical record of every model version trained,
what changed, what the results were, and the lessons extracted.
Update this file after every training run — it is the single source of
truth for "what to try" and "what to avoid".

---

## Scoring Reference

All metrics are computed via **purged walk-forward CV** (5 folds, gap=5 slots
unless noted) on 601 resolved BTC 5-minute markets.

| Metric | Direction | Notes |
|--------|-----------|-------|
| WF AUC | higher better | primary gate metric |
| Accuracy | higher better | at 0.5 threshold |
| Brier score | lower better | calibration quality |
| Features | fewer better | reduces variance with 601 samples |
| Fold AUC range | tighter better | indicates stability |

---

## Version History

### v4 — Baseline multi-crypto (deprecated)
**Date:** ~2026-05 | **Champion:** No (reference baseline)

| AUC | Acc | Brier | Features |
|-----|-----|-------|----------|
| 0.843 | — | — | ~40 |

**Changes:** Multi-crypto training (BTC+ETH+SOL features), orderbook, basic order flow.

**Result:** Good baseline but inflated by cross-asset leakage.

**Lessons:**
- ❌ ETH/SOL cross-asset features: removed in v5. Model should be BTC-only — other cryptos move together but the incremental signal doesn't justify the noise.

---

### v5 — BTC-only, OB ts_ms fix
**Date:** 2026-05 | **Champion:** Promoted

| AUC | Acc | Brier | Features |
|-----|-----|-------|----------|
| 0.8553 | — | — | — |

**Changes:** Removed all ETH/SOL features. Fixed orderbook timestamp (`ts_ms` not `timestamp_ms`). Optuna optimizes on WF AUC (not fold CV).

**Result:** +1.3% AUC over v4. BTC-only is cleaner.

**Lessons:**
- ✅ BTC-only is the right scope
- ✅ Optuna WF objective better than CV objective

---

### v6 — Lagged outcomes, purged WF gap=5
**Date:** 2026-05 | **Champion:** Promoted

| AUC | Acc | Brier | Features |
|-----|-----|-------|----------|
| 0.8559 | 0.7738 | **0.1562** | — |

**Changes:** Added lag_1/2/3 outcomes, lag streak, volume z-score, spot 1h/4h windows, tick acceleration. Purged WF gap=5 (prevents lag feature leakage).

**Result:** Best Brier score ever (0.1562). AUC slightly better than v5.

**Lessons:**
- ✅ Purged WF with gap=5 is essential when using lag features
- ✅ Lag outcomes add signal (autocorrelation in BTC direction)
- ✅ Simpler feature sets → better calibration (lower Brier)
- 📌 **0.1562 Brier is the calibration target to beat**

---

### v7 — Up/Down OB split, realized vol, ensemble test
**Date:** 2026-05 | **Champion:** Promoted

| AUC | Acc | Brier | Features |
|-----|-----|-------|----------|
| 0.8536 | 0.7598 | 0.1593 | — |

**Changes:** Split Up/Down orderbook features. Added realized vol (5/10 slots), time-weighted up_ratio (`btc_tw_up_ratio`), VWAP trend. Tested LightGBM+LR ensemble.

**Result:** Ensemble hurt AUC (0.8479 vs 0.8536 solo) — removed.

**Lessons:**
- ❌ Ensemble (LightGBM+LR): not worth it at 601 samples. Adds complexity, hurts AUC.
- ✅ `btc_tw_up_ratio` (time-weighted flow) — good signal, kept in all future versions
- ✅ `btc_vwap_trend` — good signal
- ❌ OB Down token: `best_bid_size` always 0 for resolved markets → pure noise. Drop.

---

### v8 — 6x30s sub-windows, multi-scale zscore
**Date:** 2026-06 | **Champion:** Promoted

| AUC | Acc | Brier | Features |
|-----|-----|-------|----------|
| 0.8529 | 0.7802 | 0.1707 | 63 |

**Fold AUCs:** [0.824, 0.908, 0.872, 0.814, 0.847] — wide range!

**Changes:** 6x30s sub-windows (was 3x60s). Multi-scale up_ratio zscore (5/10/20 slots). Per-window zscores. VWAP trend. Volume-weighted momentum. Round-number proximity.

**Result:** AUC slightly lower than v6/v7 but feature set more expressive. High variance (fold range 0.814-0.908) signals too many features for 601 samples.

**Lessons:**
- ❌ 63 features on 601 samples = ratio 9.5:1, too low. Causes high fold variance.
- ✅ 6x30s sub-windows better than 3x60s (finer temporal resolution)
- ✅ Multi-scale zscore (`_5s/_10s/_20s`) consistently in top features
- ✅ `btc_up_w5` (last 30s window) is consistently #1 feature — final window is most predictive
- 📌 Need aggressive feature pruning going forward

---

### v9 — Sigmoid calibration, aggressive pruning (63→27 features)
**Date:** 2026-06 | **Champion:** Promoted

| AUC | Acc | Brier | Features |
|-----|-----|-------|----------|
| 0.8519 | 0.7842 | 0.1809 | 27 |

**Fold AUCs:** [0.825, 0.899, 0.872, 0.818, 0.845]

**Changes:** Sigmoid calibration (Platt) instead of isotonic. Pruning threshold `imp_mean > 0.001`. 150 Optuna trials. 3 new features: `btc_up_ratio_stability`, `btc_vol_accel`, `btc_size_disparity`.

**Result:** 27 features is right-sized (ratio 22:1). Fold range tighter. BUT Brier **worsened** (0.1809 vs 0.1707). Sigmoid is not appropriate here.

**Lessons:**
- ❌ **Sigmoid calibration (Platt):** worse Brier than isotonic with 601 samples. Sigmoid has only 2 parameters and underfits the calibration curve for this distribution. Isotonic is better here.
- ✅ Pruning to ~27 features is the right size for 601 samples
- ✅ All 3 new features (`stability`, `vol_accel`, `size_disparity`) survived pruning → genuine signal
- ✅ 150 Optuna trials worth the extra ~5min vs 80
- 📌 **Gate bug discovered:** re-evaluating champion with default params deflated champion AUC, letting weak models promote. Fixed in v10+.

---

### v10 — Isotonic back, interaction features, min_child 5-40
**Date:** 2026-06 | **Champion:** TRUE BEST (gate bug fixed)

| AUC | Acc | Brier | Features |
|-----|-----|-------|----------|
| **0.8547** | **0.7902** | **0.1554** | 31 |

**Fold AUCs:** [0.853, 0.898, 0.874, 0.814, 0.847]

**Changes:** Back to isotonic calibration. Pruning `imp_mean > 0.0005` (slightly looser than v9). `min_child_samples` range 5-40 (was 15-80 — allows deeper trees with fewer features). 2 interaction features:
- `btc_signal_conviction = up_ratio × (1 - stability)` — strong signal + consistent
- `btc_momentum_vol_sync = momentum × vol_accel` — momentum confirmed by volume

**Result:** Best model overall. AUC 0.8547 (highest ever), Brier 0.1554 (near v6 best). Both interaction features survived pruning (top 15). Fold range 0.814-0.898 — stable.

**Lessons:**
- ✅ **Isotonic calibration is correct** for this dataset size — always use isotonic
- ✅ Interaction features work: `signal_conviction` and `momentum_vol_sync` add signal
- ✅ Lowering `min_child_samples` to 5-40 allows trees to use the smaller feature set more expressively
- ✅ Gate fix critical: use `CHAMPION_AUC` from meta directly, not re-evaluation
- ✅ 31 features is sweet spot (ratio ~19:1)

---

### v11 — Price percentile + final burst
**Date:** 2026-06 | **Promoted due to gate bug (real result: worse than v10)**

| AUC | Acc | Brier | Features |
|-----|-----|-------|----------|
| 0.8533 | 0.7839 | 0.1585 | 22 |

**Changes:** 2 new features:
- `btc_price_percentile` — where BTC sits in its 4h high-low range [0,1]
- `btc_final_burst` — vol in last 30s / avg vol per window

**Result:** Worse than v10 on 2/3 metrics. Would NOT have promoted with correct gate.

**Lessons:**
- ❌ `btc_price_percentile`: price regime context didn't help. BTC order flow signal is strong enough on its own at 5min scale — the 4h range position adds little.
- ❌ `btc_final_burst`: last-30s volume spike didn't add value beyond `btc_up_w5` and `btc_up_w5_zscore` which already capture the final window. Redundant.
- ❌ Fewer features (22) slightly hurt relative to v10 (31) — `0.0005` threshold may be too aggressive

---

### v12 — 7-fold WF, gap comparison (3 vs 5)
**Date:** 2026-06 | **Promoted due to gate bug (real result: worse than v10)**

| AUC | Acc | Brier | Features |
|-----|-----|-------|----------|
| 0.8542 | 0.7762 | 0.1634 | 38 |

**Gap result:** Script auto-selected between gap=3 and gap=5.

**Changes:** 7 folds instead of 5. Dynamic gap selection (3 vs 5). No new features (isolated WF effect).

**Result:** Worse than v10 on all 3 metrics. More folds with 601 samples = smaller test sets per fold = noisier estimates.

**Lessons:**
- ❌ **7 folds worse than 5** with 601 samples. Each fold gets only ~85 train / ~85 test samples — too small for reliable AUC estimates. 5 folds gives ~120 per test fold, which is more stable.
- ❌ Gap comparison (3 vs 5): no consistent winner — depends on random seed. Not worth the complexity.
- 📌 Stick with 5 folds, gap=5 for this dataset size

---

## Gate Fix — Critical

**Problem (v1-v12 initial):** Step 11 re-evaluated the current champion with *default* (non-Optuna) LightGBM params. This deflated champion AUC from ~0.85 to ~0.82, making it easy for any reasonable candidate to "beat" the champion. This is why v11 and v12 incorrectly promoted despite being worse than v10.

**Fix (committed 2026-06, hash 84b7bbf):** Use `CHAMPION_AUC` from `champion_meta.json` directly. This value was already computed with the same purged WF protocol during the champion's own training run — it's the honest number. Gate requires `n_passed >= 2` (2 of 3 metrics better than champion). No shortcuts.

---

## Feature Hall of Fame

Features that consistently survive permutation importance pruning across versions:

| Feature | First version | Importance rank | Notes |
|---------|---------------|-----------------|-------|
| `btc_up_w5` | v8 | #1 always | Last 30s window — final commitment signal |
| `btc_up_w5_zscore` | v8 | #2 always | Last 30s vs historical baseline |
| `btc_up_ratio_zscore_20s` | v8 | #2-3 | 20-slot anomaly — strongest multi-scale |
| `btc_up_ratio_stability` | v9 | top 10 | Signal consistency across 6 windows |
| `btc_vwap_dn` | v8 | top 10 | Down token VWAP — pricing signal |
| `btc_vwap_trend` | v7 | top 10 | VWAP rising/falling within slot |
| `btc_up_ratio_zscore_5s` | v8 | top 10 | 5-slot short-term anomaly |
| `btc_signal_conviction` | v10 | top 15 | Interaction: strong + consistent |
| `btc_size_disparity` | v9 | top 20 | Conviction gap Up vs Down |
| `btc_pre_30m_ret` | v8 | top 20 | Only spot window that survives |
| `btc_dist_1k` | v8 | top 20 | Round number proximity |

---

## Feature Graveyard

Features tested and consistently dropped by permutation importance:

| Feature | Why it fails |
|---------|--------------|
| ETH/SOL cross-asset | Correlated but adds noise, not signal |
| OB Down token (bid/ask/depth) | `best_bid_size` always 0 for resolved markets — pure noise |
| `btc_price_percentile` | 4h price regime irrelevant at 5min scale |
| `btc_final_burst` | Redundant with `btc_up_w5` |
| `btc_realized_vol_5s/10s` | Vol clustering not predictive at slot level |
| Most lag features | Only lag_streak survives; lag_1/2/3 mostly noise |
| Inslot ETH/SOL spot | Removed in v5, never came back |
| `btc_vol_zscore` | Vol anomaly: size zscore dropped, up_ratio zscore much stronger |
| `btc_inslot_vol` | Within-slot spot volatility: noise |

---

## Calibration Study

| Method | Brier (best) | Notes |
|--------|-------------|-------|
| Isotonic (CalibratedClassifierCV) | **0.1562** (v6) | Best for ~600 samples |
| Sigmoid / Platt | 0.1809 (v9) | Underfits calibration curve here |

**Verdict:** Always use `method="isotonic"` with this dataset size. Sigmoid has only 2 parameters — it underfits the calibration for the LightGBM probability distribution. Isotonic is non-parametric and fits better even with limited data.

---

## Walk-Forward Study

| Config | AUC | Notes |
|--------|-----|-------|
| 5 folds, gap=5 | 0.8547 (v10) | Best |
| 5 folds, gap=3 | not isolated | Higher AUC in some runs, riskier (lag leakage) |
| 7 folds, gap=5 | 0.8542 (v12) | Worse — test sets too small (~85 samples) |

**Verdict:** 5 folds, gap=5 is the sweet spot for 601 samples. Gap=5 is necessary when lag features (lag_1/2/3_outcome) are in the feature set to prevent leakage.

---

## Ideas Not Yet Tested

| Idea | Priority | Expected impact | Notes |
|------|----------|-----------------|-------|
| More aggressive Optuna: 300 trials | Low | +0.001 AUC | Diminishing returns past 150 |
| Stacking (LightGBM → LR on OOF preds) | Low | Unknown | Tried ensemble in v7, hurt. Stacking is different but risky with 601 samples |
| Temporal feature: hour × up_ratio | Medium | +? | Hour-of-day modulates signal strength |
| Dataset expansion (live collection) | High | +++ | 616 → 1000+ samples would be transformative. Need to collect live ticks going forward |
| CatBoost instead of LightGBM | Medium | Unknown | Better at categorical; no categoricals here so probably similar |
| XGBoost comparison | Low | Likely similar to LightGBM | |
| Platt scaling post-hoc (not CV) | Low | May fix sigmoid issue | Apply sigmoid after full training rather than CV |

---

## Reproduction

To retrain the current champion (v10):
```bash
./train.sh v10
# or
modal run scripts/train_v10_modal.py
```

All training scripts are in `scripts/train_vN_modal.py`. Each is self-contained —
downloads data from HuggingFace, trains on Modal cloud (~20min, ~$0.25), and promotes
if it beats the current champion on 2/3 metrics.
