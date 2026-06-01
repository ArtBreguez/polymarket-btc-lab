"""Build the BTC 5-min training dataset from raw Polymarket parquet files."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from btc_lab.config import DATASET_PATH, MARKETS_PATH, PRICES_5MIN_PATH
from btc_lab.features import build_price_features


def load_resolved_markets() -> pd.DataFrame:
    """Load markets.parquet and filter to resolved BTC 5-min markets."""
    print(f"Loading markets from {MARKETS_PATH} ...")
    markets = pd.read_parquet(MARKETS_PATH)
    print(f"  Total rows: {len(markets):,}")
    print(f"  Columns: {list(markets.columns)}")

    # Inspect column names/values to find the right filters
    if "crypto" in markets.columns:
        markets = markets[markets["crypto"] == "BTC"]
    if "timeframe" in markets.columns:
        tf_vals = markets["timeframe"].unique()
        print(f"  Timeframe values: {tf_vals}")
        markets = markets[markets["timeframe"].isin(["5-minute", "5-min"])]

    # Filter to resolved markets (resolution != -1)
    if "resolution" in markets.columns:
        markets = markets[markets["resolution"] != -1]
    else:
        # Try to find a resolution-like column
        res_cols = [c for c in markets.columns if "resol" in c.lower()]
        print(f"  Resolution-like columns: {res_cols}")
        if res_cols:
            markets = markets[markets[res_cols[0]] != -1]

    print(f"  Resolved BTC 5-min markets: {len(markets):,}")
    if "resolution" in markets.columns:
        print(f"  Resolution distribution:\n{markets['resolution'].value_counts()}")
    return markets.reset_index(drop=True)


def load_prices_for_markets(resolved_markets: pd.DataFrame) -> pd.DataFrame:
    """Load prices for resolved markets using batch iteration (safe, no full load)."""
    target_ids = set(resolved_markets["market_id"].astype(str).tolist())
    print(f"\nLoading prices for {len(target_ids):,} markets from {PRICES_5MIN_PATH} ...")

    pf = pq.ParquetFile(PRICES_5MIN_PATH)
    rows = []
    total_scanned = 0
    for batch in pf.iter_batches(batch_size=50_000):
        df = batch.to_pandas()
        total_scanned += len(df)
        filtered = df[df["market_id"].astype(str).isin(target_ids)]
        if len(filtered) > 0:
            rows.append(filtered)

    prices_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(f"  Scanned {total_scanned:,} price rows, kept {len(prices_df):,}")
    if len(prices_df) > 0:
        print(f"  Price columns: {list(prices_df.columns)}")
    return prices_df


def main() -> None:
    resolved_markets = load_resolved_markets()

    # Ensure market_id column exists — inspect to find it
    id_cols = [c for c in resolved_markets.columns if "market_id" in c.lower() or c == "id"]
    print(f"  Market ID-like columns: {id_cols}")

    prices_df = load_prices_for_markets(resolved_markets)

    if len(prices_df) == 0:
        print("ERROR: No price data found for the resolved markets!")
        sys.exit(1)

    print("\nBuilding feature matrix ...")
    dataset = build_price_features(prices_df, resolved_markets)
    print(f"  Dataset shape: {dataset.shape}")
    print(f"  Feature columns: {list(dataset.columns)}")
    print(f"  Target distribution:\n{dataset['target'].value_counts()}")
    print(f"\nSample rows:\n{dataset.head(3).to_string()}")

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH, index=False)
    print(f"\nSaved dataset to {DATASET_PATH}")


if __name__ == "__main__":
    main()
