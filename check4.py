import pandas as pd
import numpy as np
import pyarrow.parquet as pq

TICKS_PATH = "/dev/shm/hf_cache/datasets--BrockMisner--polymarket-btc-updown/blobs/e5becdbc73952d75816aece06baf35fc3c4a6892984712b8cf0a1792c2936ef2"

v3 = pd.read_parquet("artifacts/btc_5min_dataset_v3_clean.parquet")
v3['market_id'] = v3['market_id'].astype(str)
v3['start_ts_ms'] = v3['start_ts'].astype('int64')  # datetime64[ms] -> ms int

markets = pd.read_parquet('/home/ubuntu/polymarket-btc-bot/data/raw/markets.parquet')
btc_5m = markets[(markets['crypto']=='BTC') & (markets['timeframe']=='5-minute') & (markets['resolution']!=-1)].copy()
btc_5m['market_id'] = btc_5m['market_id'].astype(str)

# Merge to get end_ts
btc_meta = btc_5m[btc_5m['market_id'].isin(v3['market_id'].tolist())][['market_id','start_ts','end_ts']].copy()
btc_meta['end_ts_ms'] = btc_meta['end_ts'] * 1000  # seconds -> ms

# Build lookup: v3 start_ts_ms (prediction window start) and end_ts_ms
start_ts_v3_map = dict(zip(v3['market_id'], v3['start_ts_ms']))
end_ts_ms_map = dict(zip(btc_meta['market_id'], btc_meta['end_ts_ms']))

mid = v3['market_id'].iloc[0]
print(f"Market {mid}:")
print(f"  v3 start_ts_ms: {start_ts_v3_map[mid]}")
print(f"  end_ts_ms: {end_ts_ms_map[mid]}")
print(f"  v3_start to end_ts (sec): {(end_ts_ms_map[mid] - start_ts_v3_map[mid])/1000:.1f}")

# Load ticks for 1 market to check
market_ids = v3['market_id'].tolist()
ticks_t = pq.read_table(TICKS_PATH, columns=['market_id','timestamp_ms'], filters=[('market_id','=',mid)])
ticks = ticks_t.to_pandas()
print(f"\nTicks for market {mid}: {len(ticks)}")

# Use v3 start_ts as reference (as original script does)
t_sec_v3 = (ticks['timestamp_ms'] - start_ts_v3_map[mid]) / 1000
print(f"t_sec range (vs v3 start_ts): [{t_sec_v3.min():.1f}, {t_sec_v3.max():.1f}]")
in_5min = (t_sec_v3 >= 0) & (t_sec_v3 <= 300)
post_300 = t_sec_v3 > 300
print(f"Ticks in [0,300s]: {in_5min.sum()}")
print(f"Ticks in (300s,end]: {post_300.sum()}")
