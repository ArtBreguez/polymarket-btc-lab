import pyarrow.parquet as pq
import pandas as pd

markets = pd.read_parquet('/home/ubuntu/.cache/huggingface/hub/datasets--BrockMisner--polymarket-btc-updown/snapshots/c0209a1f9930caf677648479b84b62ea3a6864b5/data/markets.parquet')
btc = markets[(markets['crypto'] == 'BTC') & (markets['timeframe'] == '5-minute') & (markets['resolution'] != -1)].copy()
v3 = pd.read_parquet('artifacts/btc_5min_dataset_v3_clean.parquet')
btc['market_id'] = btc['market_id'].astype(str)
v3['market_id'] = v3['market_id'].astype(str)
# rename to avoid collision
btc2 = btc[['market_id', 'start_ts', 'end_ts']].rename(columns={'start_ts': 'mkt_start_ts', 'end_ts': 'mkt_end_ts'})
merged = v3.merge(btc2, on='market_id')
print("Merged:", len(merged))
print(merged[['market_id', 'mkt_start_ts', 'mkt_end_ts']].head(3))
print("Window sec:", (merged['mkt_end_ts'] - merged['mkt_start_ts']).value_counts().head())

sample = merged.iloc[0]
mid = sample['market_id']
start_ms = int(sample['mkt_start_ts']) * 1000
end_ms = int(sample['mkt_end_ts']) * 1000
print(f"\nMarket {mid}: start_ms={start_ms}, end_ms={end_ms}, window_sec={int(sample['mkt_end_ts'] - sample['mkt_start_ts'])}")

pf = pq.ParquetFile('data/ticks_btc_5min.parquet')
all_rows = []
for i in range(pf.metadata.num_row_groups):
    df = pf.read_row_group(i, columns=['market_id', 'timestamp_ms']).to_pandas()
    match = df[df['market_id'].astype(str) == str(mid)]
    if len(match) > 0:
        all_rows.append(match)

if all_rows:
    ticks = pd.concat(all_rows)
    print(f"Total ticks for market {mid}: {len(ticks)}")
    in_window = ticks[(ticks['timestamp_ms'] >= start_ms) & (ticks['timestamp_ms'] <= end_ms)]
    print(f"Ticks in window: {len(in_window)}")
    rel_sec = (ticks['timestamp_ms'] - start_ms) / 1000
    print(f"Relative sec range all ticks: {rel_sec.min():.1f} to {rel_sec.max():.1f}")
