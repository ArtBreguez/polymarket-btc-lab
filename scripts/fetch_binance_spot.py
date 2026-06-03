"""
Fetch BTC/USDT 1-minute klines from Binance for the local dataset range.

Saves data/binance_spot_local.parquet with columns:
  - timestamp_ms (int64): open time of each candle in ms
  - open, high, low, close (float64): OHLCV prices
  - volume (float64): BTC volume

Range covers local_markets.csv slot range minus 4h (for pre-slot features)
through max slot + 5min (for inslot features).

Usage:
    python scripts/fetch_binance_spot.py
    python scripts/fetch_binance_spot.py --start-ms 1773384300000 --end-ms 1775925900000
"""

import argparse
import time
import pandas as pd
import requests
from pathlib import Path

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
BATCH_SIZE = 1000
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "binance_spot_local.parquet"
LOCAL_MARKETS_PATH = Path(__file__).parent.parent / "data" / "local_markets.csv"


def fetch_klines_batch(start_ms: int, end_ms: int) -> list:
    """Fetch up to BATCH_SIZE 1m klines from Binance."""
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": BATCH_SIZE,
    }
    for attempt in range(5):
        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/5] {exc} — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch klines after 5 retries (start={start_ms})")


def fetch_all_klines(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Page through Binance klines API and return a full DataFrame."""
    all_rows = []
    cursor = start_ms
    batch_num = 0

    while cursor < end_ms:
        batch = fetch_klines_batch(cursor, end_ms)
        if not batch:
            break

        all_rows.extend(batch)
        batch_num += 1
        last_open_ms = int(batch[-1][0])
        progress_pct = (last_open_ms - start_ms) / (end_ms - start_ms) * 100
        last_ts = pd.to_datetime(last_open_ms, unit="ms", utc=True)
        print(f"  Batch {batch_num:3d}: {len(batch):4d} candles | last={last_ts} | {progress_pct:.1f}%")

        # Next batch starts 1 minute after the last candle's open
        cursor = last_open_ms + 60_000

        # Respect Binance rate limits (1200 req/min weight, klines = 2 weight)
        if batch_num % 10 == 0:
            time.sleep(0.5)

    if not all_rows:
        raise RuntimeError("No klines returned from Binance")

    print(f"\n  Total candles fetched: {len(all_rows):,}")

    # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
    df = pd.DataFrame(all_rows, columns=[
        "timestamp_ms", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    df = df[["timestamp_ms", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float64")

    df = df.drop_duplicates("timestamp_ms").sort_values("timestamp_ms").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance BTC spot klines")
    parser.add_argument("--start-ms", type=int, default=None,
                        help="Start time in milliseconds (default: auto from local_markets.csv)")
    parser.add_argument("--end-ms", type=int, default=None,
                        help="End time in milliseconds (default: auto from local_markets.csv)")
    args = parser.parse_args()

    if args.start_ms and args.end_ms:
        start_ms = args.start_ms
        end_ms = args.end_ms
    else:
        print("Reading local_markets.csv to determine date range...")
        markets = pd.read_csv(LOCAL_MARKETS_PATH)
        slot_min = int(markets["slot_ts"].min())
        slot_max = int(markets["slot_ts"].max())
        # 4h pre-buffer + 5min post-buffer + 1 extra candle
        start_ms = (slot_min - 4 * 3600) * 1000
        end_ms = (slot_max + 5 * 60 + 60) * 1000

    start_dt = pd.to_datetime(start_ms, unit="ms", utc=True)
    end_dt = pd.to_datetime(end_ms, unit="ms", utc=True)
    total_mins = (end_ms - start_ms) / 60_000
    print(f"Fetching BTCUSDT 1m klines from Binance")
    print(f"  Range: {start_dt} → {end_dt}")
    print(f"  ~{total_mins:.0f} minutes ≈ {total_mins/BATCH_SIZE:.1f} batches")
    print()

    df = fetch_all_klines(start_ms, end_ms)

    # Verify coverage
    actual_start = pd.to_datetime(df["timestamp_ms"].min(), unit="ms", utc=True)
    actual_end = pd.to_datetime(df["timestamp_ms"].max(), unit="ms", utc=True)
    print(f"\nActual coverage: {actual_start} → {actual_end}")
    print(f"Rows: {len(df):,}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False, compression="snappy")
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nSaved to {OUTPUT_PATH} ({size_mb:.2f} MB)")

    # Quick sanity check
    print(f"\nSample rows:")
    print(df.head(3).to_string())
    print(f"\nBTC close range: ${df['close'].min():,.0f} — ${df['close'].max():,.0f}")


if __name__ == "__main__":
    main()
