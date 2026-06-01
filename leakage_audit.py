"""
Leakage audit for polymarket-btc-lab tick features.
"""
import sys
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/ubuntu/polymarket-btc-lab/src')

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

from btc_lab.config import MARKETS_PATH, DATASET_PATH
from btc_lab.features import build_tick_features

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load markets metadata
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Load markets metadata")
print("=" * 60)

markets = pd.read_parquet(MARKETS_PATH)
btc_5m = markets[
    (markets['crypto'] == 'BTC') &
    (markets['timeframe'] == '5-minute') &
    (markets['resolution'] != -1)
].copy()
btc_5m['market_id'] = btc_5m['market_id'].astype(str)

print(btc_5m[['market_id', 'start_ts', 'end_ts', 'resolution']].head(5))
print(f"start_ts range: {btc_5m['start_ts'].min()} to {btc_5m['start_ts'].max()}")
print(f"end_ts range:   {btc_5m['end_ts'].min()} to {btc_5m['end_ts'].max()}")

now = int(time.time())
sample_start = btc_5m['start_ts'].iloc[0]
print(f"\nNow (unix seconds): {now}")
print(f"Sample start_ts: {sample_start}")
print(f"Difference from now: {abs(sample_start - now):.0f}")
is_seconds = abs(sample_start - now) < 1e9
print(f"start_ts in seconds? {is_seconds}")
print(f"Total resolved BTC 5-min markets: {len(btc_5m)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Sample 10 markets and check for leakage
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Leakage audit on 10-market sample")
print("=" * 60)

audit_markets = btc_5m.head(10).set_index('market_id')[['start_ts', 'end_ts', 'resolution']]
audit_ids = set(audit_markets.index.tolist())

TICKS_PATH = "/home/ubuntu/polymarket-btc-lab/data/ticks_btc_5min.parquet"
pf = pq.ParquetFile(TICKS_PATH)

rows = []
for batch in pf.iter_batches(batch_size=100000):
    df = batch.to_pandas()
    filtered = df[df['market_id'].astype(str).isin(audit_ids)]
    if len(filtered) > 0:
        rows.append(filtered[['market_id', 'timestamp_ms', 'side', 'outcome', 'size_usdc', 'spot_price_usdt']])
    if sum(len(r) for r in rows) > 50000:
        break

ticks_sample = pd.concat(rows, ignore_index=True)
ticks_sample['market_id'] = ticks_sample['market_id'].astype(str)

print(f"\nSampled {len(ticks_sample):,} ticks for 10 markets\n")
print("=== LEAKAGE AUDIT PER MARKET ===")

leakage_found = False
total_ticks_sample = 0
total_after_sample = 0

for mid, row in audit_markets.iterrows():
    mt = ticks_sample[ticks_sample['market_id'] == mid]
    if len(mt) == 0:
        print(f"\nMarket {mid} | No ticks found in sample")
        continue

    start_ms = row['start_ts'] * 1000  # seconds -> ms
    end_ms = row['end_ts'] * 1000

    too_early = mt[mt['timestamp_ms'] < start_ms]
    too_late = mt[mt['timestamp_ms'] > end_ms]
    within = mt[(mt['timestamp_ms'] >= start_ms) & (mt['timestamp_ms'] <= end_ms)]

    total_ticks_sample += len(mt)
    total_after_sample += len(too_late)

    print(f"\nMarket {mid} | resolution={row['resolution']}")
    print(f"  Window: {start_ms} to {end_ms} ({(end_ms - start_ms)/1000:.0f}s)")
    print(f"  Ticks: {len(mt)} total | {len(within)} within | {len(too_early)} before | {len(too_late)} AFTER (leak?)")
    if len(too_late) > 0:
        leakage_found = True
        print(f"  *** LEAKAGE DETECTED: {len(too_late)} ticks after end_ts! ***")
        print(f"  Latest tick: {mt['timestamp_ms'].max()} vs end_ms: {end_ms}")
        print(f"  Overshoot: {(mt['timestamp_ms'].max() - end_ms)/1000:.1f}s past end")
    print(f"  timestamp_ms range: {mt['timestamp_ms'].min()} to {mt['timestamp_ms'].max()}")

print(f"\nLeakage found in sample: {leakage_found}")
if total_ticks_sample > 0:
    print(f"Sample leak rate: {total_after_sample}/{total_ticks_sample} = {100*total_after_sample/total_ticks_sample:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Full streaming audit + clean tick extraction
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Full streaming audit + clean tick extraction")
print("=" * 60)

markets_dict = btc_5m.set_index('market_id')[['start_ts', 'end_ts']].to_dict('index')

pf = pq.ParquetFile(TICKS_PATH)
clean_rows = []
total_ticks_global = 0
total_outside_global = 0
total_before_global = 0
total_after_global = 0

t0 = time.time()
for i, batch in enumerate(pf.iter_batches(batch_size=200000)):
    df = batch.to_pandas()
    df['market_id'] = df['market_id'].astype(str)
    filtered = df[df['market_id'].isin(markets_dict.keys())].copy()

    if len(filtered) > 0:
        total_ticks_global += len(filtered)

        # Vectorized window filter
        start_ms_col = filtered['market_id'].map(lambda x: markets_dict[x]['start_ts'] * 1000)
        end_ms_col = filtered['market_id'].map(lambda x: markets_dict[x]['end_ts'] * 1000)

        mask_before = filtered['timestamp_ms'] < start_ms_col
        mask_after = filtered['timestamp_ms'] > end_ms_col
        mask_in = ~mask_before & ~mask_after

        total_before_global += mask_before.sum()
        total_after_global += mask_after.sum()
        total_outside_global += (mask_before | mask_after).sum()

        cols_to_keep = ['market_id', 'timestamp_ms', 'outcome', 'side', 'price', 'size_usdc', 'spot_price_usdt']
        available_cols = [c for c in cols_to_keep if c in filtered.columns]
        clean_rows.append(filtered[mask_in][available_cols])

    if i % 50 == 0:
        elapsed = time.time() - t0
        print(f"  batch {i}, ticks processed: {total_ticks_global:,}, "
              f"outside window: {total_outside_global:,}, elapsed: {elapsed:.1f}s")

clean_ticks = pd.concat(clean_rows, ignore_index=True)
total_clean = len(clean_ticks)

print(f"\n--- Full Audit Results ---")
print(f"Total ticks in resolved BTC 5-min markets: {total_ticks_global:,}")
print(f"Ticks BEFORE window: {total_before_global:,} ({100*total_before_global/max(1,total_ticks_global):.2f}%)")
print(f"Ticks AFTER window:  {total_after_global:,} ({100*total_after_global/max(1,total_ticks_global):.2f}%)")
print(f"Ticks OUTSIDE window (total): {total_outside_global:,} ({100*total_outside_global/max(1,total_ticks_global):.2f}%)")
print(f"Clean ticks (within window): {total_clean:,} ({100*total_clean/max(1,total_ticks_global):.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Build tick features from clean ticks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Build tick features + merge with price features")
print("=" * 60)

tick_features = build_tick_features(clean_ticks)
tick_features['market_id'] = tick_features['market_id'].astype(str)

print(f"Markets covered by clean ticks: {len(tick_features)}")
print(f"Clean tick features shape: {tick_features.shape}")

price_features = pd.read_parquet(DATASET_PATH)
price_features['market_id'] = price_features['market_id'].astype(str)
print(f"Price features shape (original dataset): {price_features.shape}")

combined = price_features.merge(tick_features, on='market_id', how='inner')
combined.to_parquet("/home/ubuntu/polymarket-btc-lab/artifacts/btc_5min_dataset_v3_clean.parquet", index=False)

print(f"\nCombined clean dataset: {combined.shape}")
print(f"Markets in price features: {len(price_features)}")
print(f"Markets with clean ticks: {len(tick_features)}")
print(f"Markets dropped (no clean ticks): {len(price_features) - len(combined)}")
print(f"Saved: artifacts/btc_5min_dataset_v3_clean.parquet")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Retrain on clean dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Retrain on clean dataset")
print("=" * 60)

from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
import lightgbm as lgb

# Sort by time for temporal split
if 'start_ts' in combined.columns:
    combined_sorted = combined.sort_values('start_ts')
else:
    combined_sorted = combined.copy()

exclude_cols = ['market_id', 'start_ts', 'target', 'resolution', 'end_ts',
                'crypto', 'timeframe', 'question', 'condition_id']
feature_cols = [c for c in combined_sorted.columns if c not in exclude_cols]
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

X = combined_sorted[feature_cols].values
y = combined_sorted['target'].values

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Test class balance: UP={y_test.sum()}/{len(y_test)} ({100*y_test.mean():.1f}%)")

# Train LightGBM
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)

acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, proba)
brier = brier_score_loss(y_test, proba)

print(f"\n--- Clean Model Metrics ---")
print(f"Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
print(f"AUC-ROC:   {auc:.4f}")
print(f"Brier:     {brier:.4f}")

# Compare to baseline (price-only model)
print(f"\n--- Comparison ---")
print(f"Previous (leaky) accuracy: 91.9%")
print(f"Clean model accuracy:      {acc*100:.1f}%")
delta = acc*100 - 91.9
print(f"Delta: {delta:+.1f}pp — {'LEAKAGE CONFIRMED' if delta < -5 else 'No significant drop'}")

# Feature importances
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\n--- Top 10 Feature Importances ---")
for feat, imp in importances.head(10).items():
    print(f"  {feat:35s}: {imp:6.0f}")

# Save model
model_path = "/home/ubuntu/polymarket-btc-lab/artifacts/btc_model_v3_clean.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved: {model_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL LEAKAGE AUDIT SUMMARY")
print("=" * 60)
print(f"1. Ticks outside [start_ts, end_ts]:")
print(f"   Before window: {total_before_global:,} ({100*total_before_global/max(1,total_ticks_global):.2f}%)")
print(f"   After  window: {total_after_global:,} ({100*total_after_global/max(1,total_ticks_global):.2f}%)")
print(f"   Total outside: {total_outside_global:,} ({100*total_outside_global/max(1,total_ticks_global):.2f}%)")
print(f"\n2. Clean ticks remaining: {total_clean:,}")
print(f"   Markets covered: {len(tick_features)}")
print(f"\n3. Model comparison:")
print(f"   Leaky accuracy:  91.9%")
print(f"   Clean accuracy:  {acc*100:.1f}%")
print(f"   Leaky AUC-ROC:   unknown")
print(f"   Clean AUC-ROC:   {auc:.4f}")
print(f"   Clean Brier:     {brier:.4f}")
print(f"\n4. Verdict: {'LEAKAGE CONFIRMED — accuracy drop of ' + f'{abs(delta):.1f}pp' if delta < -5 else 'No major leakage detected'}")
