# 09 — Anti-Patterns (What Failed)

This page documents every approach that was tried and produced worse results. Do not re-attempt these without new data or a fundamentally different setup.

---

## Sigmoid Calibration (v9)

**What**: Replace isotonic calibration (CalibratedClassifierCV, method="isotonic") with Platt scaling (method="sigmoid").

**Result**: Brier worsened from 0.1707 → 0.1809. Sigmoid has only 2 parameters and underfits the calibration curve for LightGBM's probability distribution with ~600 samples.

**Verdict**: Always use isotonic. It's non-parametric and fits better even with limited data.

---

## Ensemble LightGBM + Logistic Regression (v7)

**What**: Blend LightGBM and LR predictions with weight 0.65/0.35.

**Result**: AUC dropped from 0.8536 (solo LGB) to 0.8479 (ensemble). LR can't capture the non-linear interactions that LightGBM exploits.

**Verdict**: Not worth it at this dataset size. Adds complexity with no benefit.

---

## 7 Walk-Forward Folds (v12)

**What**: Increase from 5 to 7 folds for more evaluation points.

**Result**: AUC 0.8542 (worse than v10's 0.8547). Each fold gets only ~85 test samples — too noisy for reliable AUC estimates. 5 folds gives ~120 per fold.

**Verdict**: Stick with 5 folds for datasets under ~5,000 samples.

---

## Gap < 5 in Walk-Forward (v12)

**What**: Use gap=3 instead of gap=5 between train/test folds.

**Result**: No consistent improvement over gap=5. Riskier because lag features (lag_1/2/3_outcome) look back 3 slots — gap=3 allows potential leakage.

**Verdict**: Gap=5 is the safe minimum when lag features are present.

---

## ETH/SOL Cross-Asset Features (v4)

**What**: Include ETH and SOL price/volume/orderflow features alongside BTC.

**Result**: Inflated baseline AUC (0.843) via cross-asset correlation, but added noise not signal. Removing them in v5 improved AUC to 0.8553.

**Verdict**: BTC-only is the correct scope. Cryptos move together, but the incremental signal from alts doesn't justify the noise at 5-minute scale.

---

## OB Down Token Features (v7)

**What**: Use orderbook data from the DOWN token (best_bid_size, best_ask_size, depth).

**Result**: `best_bid_size` is always 0 for resolved markets — the DOWN token orderbook is essentially empty. Pure noise.

**Verdict**: Only use UP token orderbook data. DOWN token OB data is structurally missing.

---

## price_percentile Feature (v11)

**What**: Where BTC sits in its 4-hour high-low range, scaled [0,1]. Intended to capture regime context.

**Result**: Did not survive permutation importance pruning. BTC order flow signal at 5-minute scale is strong enough — the 4h price position adds nothing.

**Verdict**: Regime features at multi-hour scale don't help 5-minute predictions.

---

## final_burst Feature (v11)

**What**: Volume in last 30s divided by average volume per window. Intended to capture late-slot conviction spikes.

**Result**: Redundant with `btc_up_w5` and `btc_up_w5_zscore`, which already capture last-window dynamics. Did not add incremental signal.

**Verdict**: The sub-window features already encode final-window behavior.

---

## Inline Binance Fetch from Modal (v18 initial)

**What**: Fetch Binance spot data via REST API directly from Modal cloud compute during training.

**Result**: HTTP 451 — Binance blocks Modal's US region for legal/regulatory reasons.

**Verdict**: Always pre-fetch Binance data locally and upload to Modal Volume. Never rely on Binance API access from cloud compute.

---

## Hardcoded Sanity Probes

**What**: Fixed synthetic feature vectors to test model directionality (e.g., up_ratio=0.70 → expect P(UP) > 0.55).

**Result**: Probes become stale as feature importance shifts. v18 made `btc_inslot_ret` the #1 feature, but probes didn't set it — causing "inverted" results even though the model was correct.

**Verdict**: Sanity probes must evolve with the feature set. Test the top-5 features by importance, not a fixed list. Better: use held-out real data for sanity checks instead of synthetic scenarios.

---

## Feature Parity Drift

**What**: Changing feature computation in training without updating `deploy/live_trader.py` (or vice versa).

**Result**: Silent failures — model produces garbage predictions because input features don't match what it was trained on. At least 6 distinct bugs traced to this (see 08-troubleshooting.md).

**Verdict**: Every training feature change MUST be mirrored in live_trader.py. Run `tests/test_features.py` after any change. Consider a shared feature module (currently features are duplicated between training scripts and live_trader.py).
