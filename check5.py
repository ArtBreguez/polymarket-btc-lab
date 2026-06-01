import pandas as pd
import numpy as np
import pyarrow.parquet as pq

TICKS_PATH = "/dev/shm/hf_cache/datasets--BrockMisner--polymarket-btc-updown/blobs/e5becdbc73952d75816aece06baf35fc3c4a6892984712b8cf0a1792c2936ef2"
MARKETS_PATH = "/home/ubuntu/polymarket-btc-bot/data/raw/markets.parquet"

v3 = pd.read_parquet("artifacts/btc_5min_dataset_v3_clean.parquet")
v3['market_id'] = v3['market_id'].astype(str)
# v3 start_ts is datetime64[ms,UTC] -> milliseconds
v3['start_ts_ms'] = v3['start_ts'].astype('int64')

markets = pd.read_parquet(MARKETS_PATH)
btc_5m = markets[(markets['crypto']=='BTC') & (markets['timeframe']=='5-minute') & (markets['resolution']!=-1)].copy()
btc_5m['market_id'] = btc_5m['market_id'].astype(str)
btc_meta = btc_5m[btc_5m['market_id'].isin(v3['market_id'].tolist())][['market_id','start_ts','end_ts']].copy()
btc_meta['end_ts_ms'] = btc_meta['end_ts'] * 1000  # seconds -> ms

# Build lookups using v3 start_ts (NOT markets.start_ts!)
start_ts_map = dict(zip(v3['market_id'], v3['start_ts_ms']))  # v3 prediction start
end_ts_ms_map = dict(zip(btc_meta['market_id'], btc_meta['end_ts_ms']))  # market end

# Compute prediction end = v3_start_ts + 300s
end_5min_map = {mid: ms + 300_000 for mid, ms in start_ts_map.items()}

market_ids = v3['market_id'].tolist()

print("Loading ticks...")
ticks_table = pq.read_table(TICKS_PATH, 
    columns=["market_id","timestamp_ms","outcome","side","price","size_usdc"],
    filters=[("market_id","in",market_ids)])
ticks = ticks_table.to_pandas()
print(f"Loaded {len(ticks):,} ticks for {ticks['market_id'].nunique()} markets")

# Use v3 start_ts as reference (SAME as original build_temporal_features.py)
ticks['start_ts_ms'] = ticks['market_id'].map(start_ts_map)
ticks['end_ts_ms'] = ticks['market_id'].map(end_ts_ms_map)
ticks['end_5min_ms'] = ticks['market_id'].map(end_5min_map)

ticks['t_sec'] = (ticks['timestamp_ms'] - ticks['start_ts_ms']) / 1000.0
ticks['t_sec_to_end'] = (ticks['end_ts_ms'] - ticks['timestamp_ms']) / 1000.0

print(f"\nt_sec range: [{ticks['t_sec'].min():.1f}, {ticks['t_sec'].max():.1f}]")

# How many ticks in each region?
in_5min = ticks[(ticks['t_sec'] >= 0) & (ticks['t_sec'] <= 300)]
post_5min = ticks[(ticks['t_sec'] > 300) & (ticks['t_sec_to_end'] >= 0)]
post_market = ticks[ticks['t_sec_to_end'] < 0]
before_start = ticks[ticks['t_sec'] < 0]

total = len(ticks)
print(f"\nTotal ticks:                {total:>8,}")
print(f"Before v3 start (t<0):     {len(before_start):>8,}  ({len(before_start)/total*100:.1f}%)")
print(f"In 5-min window [0,300s]:  {len(in_5min):>8,}  ({len(in_5min)/total*100:.1f}%)")
print(f"After 5min, before mkt end:{len(post_5min):>8,}  ({len(post_5min)/total*100:.1f}%)")
print(f"After market end:          {len(post_market):>8,}  ({len(post_market)/total*100:.1f}%)")

# With buggy bins [-inf, 90, 210, inf]:
ticks['window_buggy'] = pd.cut(ticks['t_sec'], bins=[-np.inf,90,210,np.inf], labels=['early','mid','late'])
print(f"\nBuggy window distribution:")
print(ticks['window_buggy'].value_counts().sort_index().to_string())

late_ticks = ticks[ticks['window_buggy'] == 'late']
late_post_300 = late_ticks[late_ticks['t_sec'] > 300]
print(f"\nLate window (t>210, buggy): {len(late_ticks):,}")
print(f"  Of which t>300 (LEAKY):   {len(late_post_300):,}  ({len(late_post_300)/len(late_ticks)*100:.1f}%)")

# Per-market summary
print("\nPer-market summary (first 5):")
for mid in v3['market_id'].iloc[:5]:
    mt = ticks[ticks['market_id']==mid]
    if len(mt)==0: continue
    t5 = mt[(mt['t_sec']>=0)&(mt['t_sec']<=300)]
    t_post = mt[mt['t_sec']>300]
    print(f"  {mid}: total={len(mt)}, in[0,300]={len(t5)}, post300={len(t_post)} ({len(t_post)/len(mt)*100:.0f}%), t_range=[{mt['t_sec'].min():.0f},{mt['t_sec'].max():.0f}]")
