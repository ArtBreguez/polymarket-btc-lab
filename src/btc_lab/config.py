"""Path constants for the btc_lab project."""

from pathlib import Path

HF_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "datasets--BrockMisner--polymarket-btc-updown"
    / "snapshots"
    / "c0209a1f9930caf677648479b84b62ea3a6864b5"
    / "data"
)

MARKETS_PATH = HF_SNAPSHOT / "markets.parquet"
PRICES_5MIN_PATH = HF_SNAPSHOT / "prices/crypto=BTC/timeframe=5-minute/part-0.parquet"
SPOT_PRICES_PATH = HF_SNAPSHOT / "spot_prices/part-0.parquet"
TICKS_5MIN_PATH = Path.home() / "polymarket-btc-lab/data/ticks_btc_5min.parquet"

DATASET_PATH = Path.home() / "polymarket-btc-lab/artifacts/btc_5min_dataset.parquet"
