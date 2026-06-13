# Experiment Log — BTC 5min Polymarket Model

This document is the canonical record of every model version trained,
what changed, what the results were, and the lessons extracted.
Update this file after every training run — it is the single source of
truth for "what to try" and "what to avoid".

---

## Scoring Reference

All metrics are computed via **purged walk-forward CV** (5 folds, gap=5 slots
unless noted) on 22k+ resolved BTC 5-minute markets.

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

To retrain the current champion (v21):
```bash
modal run scripts/train_v21_modal.py
```

All training scripts are in `scripts/train_vN_modal.py`. Each is self-contained —
loads data from Modal Volume, trains on Modal cloud (~42min, ~$0.50), and promotes
if it beats the current champion on 2/3 metrics.

---

### v13–v16 — Iterative improvements (not documented individually)

Various improvements on top of v10: extended lag context using `new_markets.csv` (15,257 markets from pmdata.dev), temporal features, prev_slot_up_ratio from expanded timeline. Champion progressed from v10 AUC 0.8547 to v16 AUC ~0.8475 and eventually v17.

---

### v17 — Extended lag context + temporal features
**Date:** 2026-06-03 | **Champion:** Promoted

| AUC | Acc | Brier | Features |
|-----|-----|-------|----------|
| 0.8925 | 0.8032 | 0.1342 | ~30 |

**Changes:** Extended lag context using `new_markets.csv` (15,257 markets, Apr 12 - Jun 3 2026) for richer lag features. Added temporal features (hour_sin/cos, dow_sin/cos, hour × up_ratio). Used combined 22k timeline for prev_slot_up_ratio.

**Data:** 7,062 markets with ticks (local CLOB), 15,257 markets for lag context only.

**Result:** Significant jump from v10/v12 era. Best AUC (0.8925), best Acc (0.8032), best Brier (0.1342) ever.

**Lessons:**
- ✅ Extended lag context from tickless markets adds real signal
- ✅ Temporal features (hour × up_ratio) work — listed as untested in v12 era
- ✅ 22k timeline for prev_slot_up_ratio better than 7k-only

---

### v18 — 3x data expansion (22k markets with ticks)
**Date:** 2026-06-04 | **Champion:** Promoted

| AUC | Acc | Brier | Features | Samples |
|-----|-----|-------|----------|---------|
| **0.8966** | **0.8104** | **0.1318** | 30 | 22,319 |

**Fold AUCs:** [0.8831, 0.8877, 0.9006, 0.9020, 0.9099] — improving trend!

**Changes:**
- **3x MORE DATA:** `ticks_btc_full_clean.parquet` (22,237 markets, 68.3M ticks, Mar–Jun 2026) vs 7,062 markets in v17. Sources: local CLOB (Mar-Apr) + pmdata.dev retroactive ticks (Apr 12 - Jun 3).
- **UNIFIED TIMELINE:** all 22k markets now have real ticks → `prev_slot_up_ratio` uses actual data, not 0.5 fallback.
- **BINANCE SPOT:** pre-fetched `binance_spot_full.parquet` (119k 1m candles, Mar-Jun 2026) loaded from Modal Volume instead of inline API fetch (Binance blocks Modal's US region).

**Data:** Modal Volume `btc-local-data`:
- `/ticks_btc_full_clean.parquet` — 22,237 markets, 68.3M ticks
- `/all_markets.csv` — 22,319 markets
- `/binance_spot_full.parquet` — 119k candles (Mar 13 – Jun 4 2026)

**Best Optuna params (150 trials, best AUC=0.8970):**
```python
n_estimators     = 635
learning_rate    = 0.01171
max_depth        = 4
num_leaves       = 47
min_child_samples = 84
subsample        = 0.7647
colsample_bytree = 0.6114
reg_alpha        = 0.2289
reg_lambda       = 0.0001395
```

**Top 10 features:**
1. `btc_inslot_ret` ← NEW #1 (was not top-10 before — more data reveals spot signal)
2. `btc_vwap_up`
3. `btc_pre_5m_ret`
4. `btc_vwap_dn`
5. `btc_up_w1`
6. `btc_pre_30m_ret`
7. `btc_vwap_spread`
8. `btc_momentum`
9. `btc_pre_1h_ret`
10. `prev_slot_up_ratio_1`

**Gate results:** 3/3 metrics beat v17 champion:
- AUC: 0.8966 > 0.8925 ✓
- Brier: 0.1318 < 0.1342 ✓
- Acc: 0.8104 > 0.8032 ✓

**Sanity check:** ⚠️ UP scenario → 0.202 (wanted >0.55), Neutral → 0.976 (wanted ~0.50). Values are inverted from expected — investigate whether the model learned an inverted signal or if the sanity check probes are miscalibrated for the new feature set.

**Result:** Best model ever. AUC 0.8966 (+0.0041 vs v17), Brier 0.1318 (-0.0024), Acc 81%. Fold AUCs show improving trend (0.88→0.91) suggesting the model generalizes better on later data. Feature importance shifted: `btc_inslot_ret` (in-slot BTC spot return) became #1 — the 3x larger dataset revealed that spot price movement during the observation window is the strongest predictor.

**Lessons:**
- ✅ **3x more data = meaningful improvement** — AUC +0.004, Acc +0.7%, Brier -0.002. More data > better features at this stage.
- ✅ **Fold AUC trend improving** (0.88→0.91) — model generalizes better on recent data (more liquidity).
- ✅ **`btc_inslot_ret` is #1 feature** — in-slot spot return was buried in noise with 7k samples, 22k reveals it clearly.
- ✅ **Pre-fetched Binance spot** — Binance API blocks Modal US region (HTTP 451). Must fetch locally and upload to Modal Volume.
- ⚠️ **Sanity check needs review** — the neutral/UP probes may need recalibration for the expanded feature set.
- 📌 Dataset can be expanded further: pmdata.dev has data from ~Feb 15 2026 and grows daily.

---

### v19 — L2 Orderbook features (book + price_change from poly_l2)
**Date:** 2026-06-04 | **Champion:** Promoted (3/3)

**Changes:**
- Added ~20 L2 orderbook features from pmdata poly_l2 (book snapshots + price_change events)
- Pre-computed OB features for 22,189 markets (ob_features_full.parquet on Modal Volume)
- Two-stage pipeline: fetch_ob_features_modal.py → train_v19_modal.py
- 5 cross-domain interaction features (OB × CLOB: imb×ur, depth×momentum, etc.)
- Feature selection expanded to top 40 (vs 30 in v18)

**Results:**
| Metric | v18 (champion) | v19 | Δ |
|--------|----------------|-----|---|
| AUC    | 0.8966         | **0.9000** | +0.0034 |
| Brier  | 0.1318         | **0.1291** | -0.0027 |
| Acc    | 0.8104         | **0.8127** | +0.0023 |

**Gate:** 3/3 ✓ AUC ✓ Brier ✓ Acc
**Sanity:** UP=0.897 > Neutral=0.453 > DOWN=0.099 ✓

**OB Features added:**
- Book snapshots (~2,700/slot): ob_mid, ob_spread, ob_imbalance, ob_depth_ratio, ob_bid/ask_depth_5c, ob_total_depth, ob_weighted_imb, ob_mid_drift, ob_imbalance_end, ob_spread_end, ob_depth_change, ob_imb_momentum, ob_imb_w0/w1/w2
- Price changes (~68,000/slot): ob_pc_up_ratio, ob_pc_volatility, ob_pc_count, ob_fill_imbalance
- Cross-domain: x_imb_x_ur, x_depth_x_momentum, x_spread_x_vol, x_ob_drift_x_inslot, x_fill_imb_x_buy

**Takeaways:**
- 🎯 Crossed AUC 0.90 barrier — OB features add real signal
- L2 depth imbalance and BBO dynamics complement CLOB flow features
- Two-stage pre-compute pattern (fetch→Volume→train) works well for expensive per-market API data
- Markets missing OB data (~0.6% failure rate) filled with neutral defaults — no exclusion needed

---

## Feature Hall of Fame (Updated v21)

| Feature | First version | v21 rank | Notes |
|---------|---------------|----------|-------|
| `btc_inslot_ret` | v8 | **#1** | In-slot spot return — strongest signal |
| `ob_mid_drift` | v19 | **#2** | OB midpoint drift open→close — L2 feature |
| `btc_pre_5m_ret` | v8 | #3 | 5-min pre-slot spot return |
| `btc_vwap_up` | v8 | #4 | Up token VWAP |
| `x_ob_drift_x_inslot` | v19 | #5 | Interaction: OB drift × inslot return |
| `btc_up_w1` | v8 | #6 | 2nd 30s window |
| `btc_pre_30m_ret` | v8 | #7 | 30-min pre-slot spot return |
| `ob_weighted_imb` | v19 | #8 | Weighted OB imbalance |
| `btc_vwap_dn` | v8 | #9 | Down token VWAP |
| `ob_mid` | v19 | #10 | OB midpoint price |

### Feature Graveyard (Pruned in v21)

| Feature | Removed in | Reason |
|---------|-----------|--------|
| `ob_total_depth` | v21 | 0.7% importance, absolute value leak |
| `btc_up_ratio_zscore_5s` | v21 | Noisy short zscore, needs warm history |
| `btc_up_ratio_zscore_20s` | v21 | Noisy long zscore, needs warm history |
| `btc_pre_1h_4h_ratio` | v21 | Cold buffer issues in live |
| `btc_up_w0` | v21 | Earliest window = most noise |
| `prev_slot_up_ratio_4` | v21 | 4 slots back, mostly noise |
| `btc_dist_1k` | v21 | Weak round-number signal |
| `hour_x_tw_ur` | v21 | Temporal overfit risk |
| `ob_imb_w1` | v21 | Interpolated in live (not measured) |
| `hour_cos` | v21 | Calendar overfit risk |

---

## v20: Dataset Expansion Attempt (FAILED)

**Date:** 2026-06-04
**Result:** NOT PROMOTED (0/3 gate)

Attempted to expand dataset using pmdata.dev API for additional markets.
API key was expired ("API key is invalid or expired"), resulting in 0 new
markets fetched. The v20 model trained on the same data as v19 with
slightly different hyperparameters, failed to beat the champion on any metric.

**Lesson:** Always verify API credentials before running expansion pipelines.

---

## v21: Feature Ablation & Pruning

**Date:** 2026-06-05
**Result:** PROMOTED (3/3 gate) — 30 features, AUC=0.9002

### Motivation

v19 used 40 features but ~10 had <1.5% importance and some had live data
quality issues (interpolated OB, cold zscore buffers, temporal overfit).
Goal: remove features that don't contribute without losing performance.

### Methodology: Ablation Study

1. Ranked all 40 features by LightGBM importance (gain)
2. Trained 3 variants with Optuna (150 trials each):
   - **40 features** (baseline): AUC=0.9002, Brier=0.1289, Acc=81.33%
   - **35 features** (top 35): AUC=0.9001, Brier=0.1290, Acc=81.23%
   - **30 features** (top 30): AUC=0.9002, Brier=0.1290, Acc=81.34%
3. Walk-forward evaluation (5 folds) for all variants
4. Selected 30-feature variant: matched 40-feat AUC with better accuracy

### Results

| Metric | v19 (champion) | v21 (30 feat) | Delta |
|--------|---------------|---------------|-------|
| AUC | 0.9000 | 0.9002 | +0.0002 |
| Brier | 0.1291 | 0.1290 | -0.0001 |
| Accuracy | 81.27% | 81.34% | +0.07% |
| Features | 40 | 30 | -25% |

### Impact on Live System

- Removed 137 lines of dead feature computation from live_trader.py
- Faster build_features: fewer numpy operations per slot
- Cleaner codebase: no more interpolated/synthetic feature paths
- No loss in prediction quality

### Best Hyperparameters (Optuna, 150 trials)

```
n_estimators: 460
learning_rate: 0.0129
max_depth: 6
num_leaves: 58
min_child_samples: 95
subsample: 0.602
colsample_bytree: 0.603
reg_alpha: 0.0056
reg_lambda: 0.0234
```

---

## v22: OBS Window Alignment A/B Test

**Date:** 2026-06-06 | **Champion:** No

A/B test: variant A (OBS_SECS=60, match live) vs variant B (OBS_SECS=180, full window).
Neither variant beat v21 champion (AUC=0.9002). Root cause: removing the 180s tick data from the OBS window
hurt the CLOB flow features that depended on a full 3-minute observation window.

**Lesson:**
- Training OBS window must match what live actually receives — but shrinking the window naively degrades features built for 180s
- Confirmed: live CLOB data-api lags ~120s, so at t=60s only partial tick data available

---

## v23: Live-Aligned Feature Formulas

**Date:** 2026-06-07 | **Champion:** No

Fixed formula mismatches vs v22: tw_up_ratio (linear→exp decay), momentum (unified n_windows formula),
x_ob_drift_x_inslot (was always zero in live due to ordering bug).
Focus on OBS_SECS=60. Did not beat v21.

**Lesson:**
- Formula alignment is necessary but not sufficient — fixing bugs without a stronger signal source won't lift AUC

---

## v24–v25: CLOB WS Microstructure Features

**Date:** 2026-06-08 | **Champion:** No (v25)

v25 added 10 CLOB WS real-time features (clob_imb_mean, clob_spread_mean, clob_mid_velocity, etc.)
from BrockMisner/polymarket-btc-updown dataset (CLOB book + price_change events).
Total: 50 features. OBS_SECS=60.

**Result:** Did not beat v21. CLOB WS features had signal but couldn't compensate for smaller OBS window.

**Lesson:**
- CLOB WS features (clob_*) are real-time zero-lag — keep them, but need better data alignment

---

## v26–v27: Real-Time Only Philosophy

**Date:** 2026-06-09 | **Champion:** No

Philosophy shift: ONLY features computable with <5s lag (Binance kline WS + CLOB REST /book + CLOB WS).
Excluded tick-based features (data-api ~120s lag). OB snapshot from pmdata poly_l2.

**Result:** AUC degraded vs v21 — tick features (btc_up_ratio etc.) do carry signal even with lag.

**Lesson:**
- Pure real-time restriction too aggressive at this dataset size
- Tick features from data-api are available by t=60s decision time (lag ~30-40s, not 120s) — safe to keep

---

## v28: Full Feature Parity (train == live)

**Date:** 2026-06-10 | **Champion:** No

Exact same feature set as deploy/live_trader.py. Includes:
- Group A: Binance spot (kline WS): pre-slot returns, in-slot ret/range/vol, dist round numbers
- Group B: CLOB REST /book snapshot (t~60s): imbalance, spread, depth
- Group C: Tick-based order flow (data-api, available by t=60s): up_ratio, n_ticks, momentum, buy_ratio, etc.
- Group D: CLOB WS price_change: spread/mid dynamics, fill imbalance
- Group E: Ring buffer (lag outcomes, zscores)
- Group F: Cross-interaction features

Used ob_features_full.parquet (CLOB window [108,168s) — mismatch with live [0,60s)).

**Result:** Trained but champion gate details unclear — superceded by v29.

---

## v29: 20 Real-Time Features — CURRENT CHAMPION

**Date:** 2026-06-11 | **Champion:** PROMOTED — WF AUC=0.7918

| Metric | Value |
|--------|-------|
| WF AUC | **0.7918** |
| WF Brier | — |
| Features | **20** |
| OBS_SECS | 60 |

**Changes vs v28:**
- Pruned to 20 features via importance ablation
- ob_features_full.parquet (CLOB window [108,168s)) used for training
- Live: clob_features window_secs=168 (later fixed to 60 for paridade)
- Retornos spot (btc_inslot_ret, btc_inslot_range, btc_pre_5m_ret): brutos (sem normalização vol_1h)

**Live state (2026-06-13):**
- Champion ativo no HF: artbreguez/polymarket-btc-model
- Wallet W4/L1, P&L=+$5.78, saldo ~$23 USDC
- CLOB window live: [0,60s) — leve mismatch vs treino [108,168s), mas melhor que antes
- Retornos brutos live = paridade com treino ✅

**Lessons:**
- 20 features é suficiente para 22k mercados — menos overfitting
- RETURN_RANGE deve ser ±0.05 quando retornos são brutos (não normalizar por vol_1h no live)
- CLOB window [0,60s) vs [108,168s) é mismatch conhecido — resolver no v31

---

## v30: Vol_1h Normalization + CLOB [0,60s) Parity

**Date:** 2026-06-13 | **Champion:** No (AUC=0.7783 < champion 0.7918)

| Metric | Value |
|--------|-------|
| WF AUC | 0.7783 ± 0.0083 |
| AV AUC | 0.9508 (drift severo) |
| Features | 10 |
| Optuna trials | 89/200 (cortado por timeout) |

**Changes vs v29:**
- FIX: btc_inslot_ret/range/pre_*_ret normalizados por vol_1h — tentativa de match com live
- CLOB: usa ob_features_v31.parquet (janela [0,60s)) em vez de ob_features_full.parquet ([108,168s))

**Result:** Não promoveu. Optuna cortado no trial 89/200 pelo timeout Modal.
AV AUC 0.9508 indica drift severo entre folds (possível sinal de overfitting ou Optuna incompleto).

**Lessons:**
- Normalização por vol_1h piora AUC — os splits do LightGBM foram treinados com retornos brutos (v29)
  e a normalização criou uma distribuição diferente sem benefício mensurável
- REVERTER: manter retornos brutos no live para paridade com v29 ✅ (feito)
- Optuna precisa de 200 trials completos — não cortar por timeout no Modal
- Próximo passo (v31): fetch local pmdata [0,168s) + retreinar com janela unificada

