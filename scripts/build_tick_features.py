"""Build tick-based features for all resolved BTC 5-min markets and merge with price features."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from btc_lab.features import build_tick_features

TICKS_PATH = Path("/home/ubuntu/polymarket-btc-lab/data/ticks_btc_5min.parquet")
DATASET_PATH = Path("/home/ubuntu/polymarket-btc-lab/artifacts/btc_5min_dataset.parquet")
OUTPUT_PATH = Path("/home/ubuntu/polymarket-btc-lab/artifacts/btc_5min_dataset_v2.parquet")

BATCH_SIZE = 200_000
TIME_LIMIT_SECONDS = 600  # 10 min safety limit


def main() -> None:
    # STEP 1: Load resolved market IDs
    print(f"Loading existing dataset from {DATASET_PATH} ...")
    dataset = pd.read_parquet(DATASET_PATH)
    resolved_ids = set(dataset['market_id'].astype(str).tolist())
    print(f"Markets to process: {len(resolved_ids)}")

    # STEP 2: Stream ticks file
    print(f"\nStreaming ticks from {TICKS_PATH} ...")
    pf = pq.ParquetFile(TICKS_PATH)
    total_batches = pf.metadata.num_row_groups
    print(f"Total row groups: {total_batches}, batch_size={BATCH_SIZE:,}")

    rows = []
    start_time = time.time()

    for i, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT_SECONDS:
            print(f"\nWARNING: Time limit ({TIME_LIMIT_SECONDS}s) reached at batch {i}. Stopping early.")
            break

        df = batch.to_pandas()
        filtered = df[df['market_id'].astype(str).isin(resolved_ids)]
        if len(filtered) > 0:
            rows.append(filtered[['market_id', 'timestamp_ms', 'outcome', 'side', 'price', 'size_usdc', 'spot_price_usdt']])

        if i % 50 == 0:
            acc = sum(len(r) for r in rows)
            print(f"  batch {i}/{total_batches}, accumulated {acc:,} rows, elapsed {elapsed:.1f}s")

    acc = sum(len(r) for r in rows)
    print(f"\nDone streaming. Total accumulated rows: {acc:,}")

    if not rows:
        print("ERROR: No ticks found for resolved markets!")
        sys.exit(1)

    ticks_df = pd.concat(rows, ignore_index=True)
    print(f"Total ticks for resolved markets: {len(ticks_df):,}")
    print(f"Unique markets with ticks: {ticks_df['market_id'].nunique()}")

    # STEP 3: Build tick features
    print("\nBuilding tick features ...")
    tick_features = build_tick_features(ticks_df)
    print(f"Tick features shape: {tick_features.shape}")

    # STEP 4: Merge with price features
    print("\nMerging with price features ...")
    dataset['market_id'] = dataset['market_id'].astype(str)
    tick_features['market_id'] = tick_features['market_id'].astype(str)

    combined = dataset.merge(tick_features, on='market_id', how='inner')
    print(f"Combined dataset shape: {combined.shape}")
    print(f"\nColumns:\n{list(combined.columns)}")

    # Missing value counts
    missing = combined.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\nMissing value counts:\n{missing.to_string()}")
    else:
        print("\nNo missing values!")

    # Target distribution
    print(f"\nTarget distribution:\n{combined['target'].value_counts().to_string()}")

    # STEP 5: Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved combined dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
