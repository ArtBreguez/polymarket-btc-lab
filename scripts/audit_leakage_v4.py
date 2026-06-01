"""
Leakage Audit for v4 temporal features.

Check: what fraction of ticks in the v4 "late" window are after end_ts?
The buggy code uses bins=[-inf, 90, 210, inf] — the inf upper bound captures
ticks after end_ts (start_ts + 300s), leaking future info.
"""
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(REPO, "artifacts")
TICKS_PATH = "/dev/shm/hf_cache/datasets--BrockMisner--polymarket-btc-updown/blobs/e5becdbc73952d75816aece06baf35fc3c4a6892984712b8cf0a1792c2936ef2"
MARKETS_PATH = "/home/ubuntu/polymarket-btc-bot/data/raw/markets.parquet"

print("=" * 70)
print("LEAKAGE AUDIT — v4 temporal features")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load v3 dataset and markets
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 1: Loading v3 dataset and markets metadata...")

v3 = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "btc_5min_dataset_v3_clean.parquet"))
print(f"  v3 shape: {v3.shape}")

markets = pd.read_parquet(MARKETS_PATH)
print(f"  markets shape: {markets.shape}")
print(f"  markets cols: {markets.columns.tolist()}")
print(f"  markets start_ts dtype: {markets['start_ts'].dtype}")
print(f"  markets end_ts dtype: {markets['end_ts'].dtype}")

# Filter to BTC 5-min resolved markets
btc_5m = markets[
    (markets['crypto'] == 'BTC') &
    (markets['timeframe'] == '5-minute') &
    (markets['resolution'] != -1)
].copy()
btc_5m['market_id'] = btc_5m['market_id'].astype(str)
print(f"  BTC 5-min resolved markets: {len(btc_5m)}")

# Check if start_ts is in seconds or milliseconds
sample_ts = btc_5m['start_ts'].iloc[0]
import time
now_sec = int(time.time())
is_seconds = abs(sample_ts - now_sec) < 1e9
print(f"  Sample start_ts: {sample_ts}, is_seconds: {is_seconds}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Merge v3 with markets to get end_ts
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 2: Merging v3 with markets to get end_ts...")

v3['market_id'] = v3['market_id'].astype(str)
market_ids = v3['market_id'].tolist()

btc_meta = btc_5m[btc_5m['market_id'].isin(market_ids)][['market_id', 'start_ts', 'end_ts']].copy()
print(f"  Markets matched in v3: {len(btc_meta)}")

# Convert timestamps to milliseconds
if is_seconds:
    btc_meta['start_ts_ms'] = btc_meta['start_ts'] * 1000
    btc_meta['end_ts_ms'] = btc_meta['end_ts'] * 1000
else:
    btc_meta['start_ts_ms'] = btc_meta['start_ts']
    btc_meta['end_ts_ms'] = btc_meta['end_ts']

# Check window duration
btc_meta['window_sec'] = (btc_meta['end_ts_ms'] - btc_meta['start_ts_ms']) / 1000
print(f"  Window duration stats (seconds):")
print(f"    mean: {btc_meta['window_sec'].mean():.1f}")
print(f"    median: {btc_meta['window_sec'].median():.1f}")
print(f"    min: {btc_meta['window_sec'].min():.1f}")
print(f"    max: {btc_meta['window_sec'].max():.1f}")

start_ts_map = dict(zip(btc_meta['market_id'], btc_meta['start_ts_ms']))
end_ts_map = dict(zip(btc_meta['market_id'], btc_meta['end_ts_ms']))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load ticks for these 616 markets
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 3: Loading ticks for 616 markets (efficient pyarrow filter)...")

market_ids_set = set(market_ids)
ticks_table = pq.read_table(
    TICKS_PATH,
    columns=["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"],
    filters=[("market_id", "in", market_ids)]
)
ticks = ticks_table.to_pandas()
print(f"  Loaded {len(ticks):,} ticks for {ticks['market_id'].nunique()} markets")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute t_sec for each tick
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 4: Computing t_sec (relative time from market start)...")

ticks['start_ts_ms'] = ticks['market_id'].map(start_ts_map)
ticks['end_ts_ms'] = ticks['market_id'].map(end_ts_map)
ticks['t_sec'] = (ticks['timestamp_ms'] - ticks['start_ts_ms']) / 1000.0

total_ticks = len(ticks)
valid_range = ticks[(ticks['t_sec'] >= 0) & (ticks['t_sec'] <= 300)]
pre_market = ticks[ticks['t_sec'] < 0]
post_end = ticks[ticks['t_sec'] > 300]
exactly_in = ticks[(ticks['t_sec'] >= 0) & (ticks['t_sec'] <= 300)]

print(f"\n  t_sec range: [{ticks['t_sec'].min():.1f}, {ticks['t_sec'].max():.1f}]")
print(f"  Total ticks: {total_ticks:,}")
print(f"  Pre-market (t_sec < 0): {len(pre_market):,} ({len(pre_market)/total_ticks*100:.2f}%)")
print(f"  In-market (0 ≤ t_sec ≤ 300): {len(valid_range):,} ({len(valid_range)/total_ticks*100:.2f}%)")
print(f"  Post-end (t_sec > 300): {len(post_end):,} ({len(post_end)/total_ticks*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Analyze which window the post-end ticks fall in (using buggy bins)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 5: Window assignment with BUGGY bins [-inf, 90, 210, inf]...")

ticks['window_buggy'] = pd.cut(
    ticks['t_sec'],
    bins=[-np.inf, 90, 210, np.inf],
    labels=["early", "mid", "late"]
)

buggy_window_dist = ticks.groupby('window_buggy').size()
print(f"  Window distribution (buggy):")
print(buggy_window_dist.to_string())

print("\n  Post-end ticks by buggy window:")
post_end_by_window = ticks[ticks['t_sec'] > 300]['window_buggy'].value_counts()
print(post_end_by_window.to_string())

# Fraction of late window that is post-end
late_ticks = ticks[ticks['window_buggy'] == 'late']
late_post_end = late_ticks[late_ticks['t_sec'] > 300]
print(f"\n  Late window total ticks: {len(late_ticks):,}")
print(f"  Late window post-end (leaky) ticks: {len(late_post_end):,}")
if len(late_ticks) > 0:
    print(f"  Fraction of late window that is leaky: {len(late_post_end)/len(late_ticks)*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Analyze with CLEAN bins [-inf, 90, 210, 300]
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6: Window assignment with CLEAN bins [-inf, 90, 210, 300]...")

clean_ticks = ticks[(ticks['t_sec'] >= 0) & (ticks['t_sec'] <= 300)].copy()
clean_ticks['window_clean'] = pd.cut(
    clean_ticks['t_sec'],
    bins=[-np.inf, 90, 210, 300],
    labels=["early", "mid", "late"]
)

print(f"  Clean ticks (0 ≤ t_sec ≤ 300): {len(clean_ticks):,}")
clean_window_dist = clean_ticks['window_clean'].value_counts()
print(f"  Window distribution (clean):")
print(clean_window_dist.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 7. Per-market leakage stats
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 7: Per-market leakage stats (sample 10 markets)...")

sample_markets = v3['market_id'].iloc[:10].tolist()
for mid in sample_markets:
    mkt_ticks = ticks[ticks['market_id'] == mid]
    if len(mkt_ticks) == 0:
        continue
    post = mkt_ticks[mkt_ticks['t_sec'] > 300]
    print(f"  Market {mid}: {len(mkt_ticks)} ticks, {len(post)} post-end "
          f"({len(post)/len(mkt_ticks)*100:.0f}%), "
          f"t_sec range [{mkt_ticks['t_sec'].min():.0f}, {mkt_ticks['t_sec'].max():.0f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("LEAKAGE AUDIT SUMMARY")
print("=" * 70)
print(f"  Total ticks for 616 markets:      {total_ticks:>12,}")
print(f"  Ticks in valid window [0, 300s]:  {len(valid_range):>12,}  ({len(valid_range)/total_ticks*100:.1f}%)")
print(f"  Post-end leaky ticks (t > 300s):  {len(post_end):>12,}  ({len(post_end)/total_ticks*100:.1f}%)")
print(f"  All post-end go to 'late' window: YES (inf upper bound)")
print(f"  Fraction of 'late' that is leaky: {len(late_post_end)/len(late_ticks)*100:.1f}%")
print()
print("  CONCLUSION: The 'late' window in the buggy v4 dataset contains")
print(f"  {len(late_post_end):,} post-resolution ticks ({len(late_post_end)/total_ticks*100:.1f}% of all ticks),")
print("  which directly encode the market outcome — a clear data leakage.")
print("  This inflates accuracy from ~90% to ~96%.")
print("=" * 70)
