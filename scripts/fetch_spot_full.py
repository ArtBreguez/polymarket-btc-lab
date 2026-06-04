"""Fetch full Binance spot data for v18 training range and upload to Modal Volume."""
import json, time, urllib.request, numpy as np, pandas as pd

# Range needed: all_markets.csv min slot - 4h to max slot + 6min
spot_start_s = 1773398700 - 4*3600
spot_end_s   = 1780502400 + 360

fetch_start_ms = spot_start_s * 1000
fetch_end_ms   = spot_end_s * 1000

candles = []
cur_ms = fetch_start_ms
batch_n = 0
while cur_ms < fetch_end_ms:
    url = (
        f"https://api.binance.com/api/v3/klines?"
        f"symbol=BTCUSDT&interval=1m"
        f"&startTime={cur_ms}&limit=1000"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            batch = json.loads(resp.read())
    except Exception as e:
        print(f"Error at {cur_ms}: {e} — retrying")
        time.sleep(2)
        continue
    if not batch:
        break
    candles.extend(batch)
    cur_ms = int(batch[-1][0]) + 60000
    batch_n += 1
    if batch_n % 20 == 0:
        print(f"  batch {batch_n}, {len(candles)} candles so far")
    time.sleep(0.12)

print(f"Total candles fetched: {len(candles)}")

df = pd.DataFrame([{
    "timestamp_ms": int(c[0]),
    "open": float(c[1]),
    "high": float(c[2]),
    "low": float(c[3]),
    "close": float(c[4]),
    "volume": float(c[5]),
} for c in candles])
df = df.sort_values("timestamp_ms").drop_duplicates("timestamp_ms").reset_index(drop=True)
print(f"Unique candles: {len(df)}")

out_path = "/tmp/binance_spot_full.parquet"
df.to_parquet(out_path, index=False)
print(f"Saved to {out_path}")
