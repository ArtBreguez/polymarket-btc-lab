import pandas as pd
from pathlib import Path

markets = pd.read_parquet('/home/ubuntu/polymarket-btc-bot/data/raw/markets.parquet')
btc5 = markets[(markets['crypto']=='BTC') & (markets['timeframe']=='5-minute') & (markets['resolution']!=-1)].copy()
btc5['slot_ts'] = btc5['slug'].str.split('-').str[-1].astype(float).astype(int)
slot_min = btc5['slot_ts'].min(); slot_max = btc5['slot_ts'].max()
print(f'BTC 5min slots: {len(btc5)} resolved | range {slot_min} to {slot_max}')

spot = pd.read_parquet('/home/ubuntu/polymarket-btc-lab/data/data/spot_prices/part-0.parquet')
spot_btc = spot[spot['symbol'].isin(['btcusdt','btc/usd'])].copy()
spot_btc['ts_s'] = spot_btc['ts_ms'] // 1000
spot_min = spot_btc['ts_s'].min(); spot_max = spot_btc['ts_s'].max()
print(f'Spot BTC: {len(spot_btc):,} rows | range {spot_min} to {spot_max}')

overlap = btc5[(btc5['slot_ts'] >= spot_min) & (btc5['slot_ts'] <= spot_max)]
print(f'Slots with spot coverage: {len(overlap)} / {len(btc5)}')
print(f'Spot resolution: {spot_btc.groupby("source")["ts_ms"].diff().median()} ms median gap')

# Multi-crypto: ETH/SOL ticks size
import os
for crypto in ['ETH','SOL']:
    p = Path(f'/home/ubuntu/polymarket-btc-lab/data/data/ticks/crypto={crypto}/timeframe=5-minute/part-0.parquet')
    if p.exists():
        size_mb = p.stat().st_size / 1024 / 1024
        print(f'{crypto} ticks parquet: {size_mb:.0f} MB (on disk)')
