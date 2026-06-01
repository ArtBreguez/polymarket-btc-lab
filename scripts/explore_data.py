import pandas as pd
df = pd.read_parquet('/home/ubuntu/polymarket-btc-bot/data/raw/markets.parquet')
print('Total markets:', len(df))
print('Columns:', list(df.columns))
btc5 = df[(df['crypto']=='BTC') & (df['timeframe']=='5-minute') & (df['resolution']!=-1)]
print('BTC 5min resolved:', len(btc5))
for c in ['ETH','SOL','XRP','BNB','DOGE','HYPE']:
    sub = df[(df['crypto']==c) & (df['timeframe']=='5-minute') & (df['resolution']!=-1)]
    if len(sub):
        print(c, '5min:', len(sub))
print()
for tf in sorted(df[df['crypto']=='BTC']['timeframe'].unique()):
    sub = df[(df['crypto']=='BTC') & (df['timeframe']==tf) & (df['resolution']!=-1)]
    print('BTC', tf, ':', len(sub))

# spot_prices
sp = pd.read_parquet('/home/ubuntu/polymarket-btc-lab/data/data/spot_prices/part-0.parquet')
print('\nspot_prices shape:', sp.shape)
print(sp['symbol'].unique())
btc_spot = sp[sp['symbol'].str.contains('btc', case=False)]
print('BTC spot rows:', len(btc_spot))
print('BTC spot range:', btc_spot['ts_ms'].min(), 'to', btc_spot['ts_ms'].max())
print(btc_spot.head(3))

# prices BTC 5min
p = pd.read_parquet('/home/ubuntu/polymarket-btc-lab/data/data/prices/crypto=BTC/timeframe=5-minute/part-0.parquet')
print('\nprices BTC 5min shape:', p.shape)
print(p.head(3))
print('unique markets:', p['market_id'].nunique())
