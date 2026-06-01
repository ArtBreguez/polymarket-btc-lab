import pandas as pd
import numpy as np
import pyarrow.parquet as pq

v3 = pd.read_parquet("artifacts/btc_5min_dataset_v3_clean.parquet")
print("v3 start_ts dtype:", v3['start_ts'].dtype)
print("v3 start_ts sample:", v3['start_ts'].iloc[:3].tolist())
v3['start_ts_ms'] = v3['start_ts'].astype('int64')
print("v3 start_ts as int64 (ms):", v3['start_ts_ms'].iloc[:3].tolist())

markets = pd.read_parquet('/home/ubuntu/polymarket-btc-bot/data/raw/markets.parquet')
btc_5m = markets[(markets['crypto']=='BTC') & (markets['timeframe']=='5-minute') & (markets['resolution']!=-1)].copy()
btc_5m['market_id'] = btc_5m['market_id'].astype(str)
v3['market_id'] = v3['market_id'].astype(str)

mid = v3['market_id'].iloc[0]
print(f"\nMarket ID: {mid}")
mkt_row = btc_5m[btc_5m['market_id']==mid].iloc[0]
print(f"Market start_ts (sec): {mkt_row['start_ts']}")
print(f"Market end_ts (sec): {mkt_row['end_ts']}")
print(f"v3 start_ts (ms): {v3[v3['market_id']==mid]['start_ts_ms'].iloc[0]}")

mkt_start_ms = mkt_row['start_ts'] * 1000
mkt_end_ms = mkt_row['end_ts'] * 1000
v3_start_ms = v3[v3['market_id']==mid]['start_ts_ms'].iloc[0]

print(f"\nMarket start_ts (ms): {mkt_start_ms}")
print(f"Market end_ts (ms): {mkt_end_ms}")
print(f"v3 start_ts (ms): {v3_start_ms}")
print(f"Diff mkt_start to v3_start (sec): {(v3_start_ms - mkt_start_ms)/1000:.1f}")
print(f"Diff v3_start to mkt_end (sec): {(mkt_end_ms - v3_start_ms)/1000:.1f}")
