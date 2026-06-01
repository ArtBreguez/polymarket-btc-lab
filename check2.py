import pandas as pd

markets = pd.read_parquet('/home/ubuntu/.cache/huggingface/hub/datasets--BrockMisner--polymarket-btc-updown/snapshots/c0209a1f9930caf677648479b84b62ea3a6864b5/data/markets.parquet')
btc = markets[(markets['crypto'] == 'BTC') & (markets['timeframe'] == '5-minute') & (markets['resolution'] != -1)].copy()

row = btc[btc['market_id'] == 1884811].iloc[0]
print('Market 1884811:')
print(f"  start_ts: {row['start_ts']}")
print(f"  end_ts: {row['end_ts']}")
print(f"  end_ts - 300: {row['end_ts'] - 300}")
print(f"  Price tick min ts: 1775518083")
print(f"  Diff price_min from (end_ts-300): {1775518083 - (row['end_ts'] - 300)} seconds")
print(f"  Diff price_min from v3 start_ts (1775518083): 0 - same!")
print()
print("So V3 start_ts = first price tick timestamp, which is roughly end_ts-300")
print("The actual 5-min market window = [end_ts - 300, end_ts] in seconds")

# So the ticks window is: [end_ts - 300, end_ts] in seconds
# = [end_ts*1000 - 300000, end_ts*1000] in ms
v3 = pd.read_parquet('artifacts/btc_5min_dataset_v3_clean.parquet')
btc['market_id'] = btc['market_id'].astype(str)
v3['market_id'] = v3['market_id'].astype(str)
btc2 = btc[['market_id', 'start_ts', 'end_ts']].rename(columns={'start_ts': 'mkt_start_ts', 'end_ts': 'mkt_end_ts'})
merged = v3.merge(btc2, on='market_id')

# The 5-min tick window should be [mkt_end_ts - 300, mkt_end_ts]
merged['tick_start_ms'] = (merged['mkt_end_ts'] - 300) * 1000
merged['tick_end_ms'] = merged['mkt_end_ts'] * 1000
merged['v3_start_ms'] = merged['start_ts'].astype('int64') // 1_000_000

# Check alignment
diff = merged['v3_start_ms'] - merged['tick_start_ms']
print(f"\nDiff between v3_start_ts and (mkt_end_ts-300)*1000:")
print(diff.describe())
print("Distribution of diffs (in ms):")
print(diff.value_counts().head(10))
