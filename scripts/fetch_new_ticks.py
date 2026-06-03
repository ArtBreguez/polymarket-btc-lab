"""
fetch_new_ticks.py
==================
For each market in data/new_markets.csv, fetch tick data from Polymarket
data-api and save to data/new_ticks.parquet.

The tick schema matches ticks_btc_5min.parquet (same columns used in training):
  market_id, timestamp_ms, outcome, side, price, size_usdc

KEY FIX: use clobTokenIds from gamma API slug endpoint — gives full 76-digit
token IDs required by data-api. CLOB /markets/{condition_id} returns short IDs
that cause the API to return a global feed (wrong markets, 0 inslot ticks).

Usage:
    python scripts/fetch_new_ticks.py
    python scripts/fetch_new_ticks.py --workers 10 --limit 1000
"""
import argparse, json, time, requests, re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR      = Path(__file__).parent.parent / "data"
MARKETS_FILE  = DATA_DIR / "new_markets.csv"
OUT_FILE      = DATA_DIR / "new_ticks.parquet"
PROGRESS_FILE = DATA_DIR / "new_ticks_progress.json"
GAMMA_API     = "https://gamma-api.polymarket.com/markets"
DATA_API      = "https://data-api.polymarket.com/trades"
MAX_PAGES     = 20
PAGE_SIZE     = 500

SCHEMA = pa.schema([
    pa.field("market_id",    pa.string()),
    pa.field("timestamp_ms", pa.int64()),
    pa.field("outcome",      pa.string()),
    pa.field("side",         pa.string()),
    pa.field("price",        pa.float64()),
    pa.field("size_usdc",    pa.float64()),
])


def get_full_token_ids(slug: str) -> tuple[str, str]:
    """
    Fetch full 76-digit clobTokenIds from gamma slug endpoint.
    Returns (yes_token_id, no_token_id) or ("", "").
    """
    try:
        r = requests.get(
            f"https://gamma-api.polymarket.com/markets/slug/{slug}",
            timeout=10,
        )
        if not r.ok:
            return "", ""
        d = r.json()
        m = d if isinstance(d, dict) else (d[0] if d else {})
        token_ids = m.get("clobTokenIds", "[]")
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if len(token_ids) >= 2:
            # Index 0 = Up/Yes, Index 1 = Down/No (confirmed from CLOB outcomes)
            return str(token_ids[0]), str(token_ids[1])
    except Exception:
        pass
    return "", ""


def fetch_ticks_for_market(
    market_id: str, slug: str, slot_ts: int
) -> list[dict]:
    """Fetch all inslot ticks for a market using full token IDs."""
    yes_token, no_token = get_full_token_ids(slug)
    if not yes_token:
        return []

    all_ticks = []
    slot_end_ts = slot_ts + 300  # 5min window (seconds)

    for outcome_label, token_id in [("Up", yes_token), ("Down", no_token)]:
        if not token_id:
            continue

        for page in range(MAX_PAGES):
            try:
                params = {
                    "asset":  token_id,
                    "limit":  PAGE_SIZE,
                    "offset": page * PAGE_SIZE,
                }
                r = requests.get(DATA_API, params=params, timeout=15)
                if not r.ok:
                    break
                batch = r.json()
                if not batch:
                    break

                # Filter to slot window
                inslot = []
                for t in batch:
                    ts_s = int(t.get("timestamp", 0))
                    if slot_ts <= ts_s < slot_end_ts:
                        price = float(t.get("price", 0))
                        size  = float(t.get("size", 0))
                        inslot.append({
                            "market_id":    market_id,
                            "timestamp_ms": ts_s * 1000,
                            "outcome":      outcome_label,
                            "side":         t.get("side", "").upper(),
                            "price":        price,
                            "size_usdc":    price * size,
                        })
                all_ticks.extend(inslot)

                # data-api returns trades in RANDOM order (not chronological).
                # Only stop if the whole batch has no hits AND all timestamps
                # are well outside the slot window on BOTH sides — meaning
                # we're in a dense region of different market activity.
                ts_values = [int(t.get("timestamp", 0)) for t in batch]
                if ts_values:
                    max_ts = max(ts_values)
                    min_ts = min(ts_values)
                    # If all trades are more than 10min before the slot,
                    # and none landed in the slot, stop paging.
                    if max_ts < slot_ts - 600 and not inslot:
                        break

            except Exception:
                break

    return all_ticks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only N markets (for testing)")
    args = parser.parse_args()

    markets = pd.read_csv(MARKETS_FILE)
    markets["market_id"] = markets["market_id"].astype(str)
    print(f"Markets to process: {len(markets):,}")

    # Resume from progress
    done_ids: set[str] = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            done_ids = set(json.load(f))
        print(f"Already done: {len(done_ids):,}, remaining: {len(markets) - len(done_ids):,}")

    todo = markets[~markets["market_id"].isin(done_ids)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"Processing {len(todo):,} markets with {args.workers} workers...")

    # Load existing ticks if resuming
    if OUT_FILE.exists():
        existing = pd.read_parquet(OUT_FILE)
        print(f"Existing ticks: {len(existing):,}")
    else:
        existing = None

    all_new_ticks: list[dict] = []
    processed = 0
    empty = 0

    def process_market(row):
        mid      = str(row["market_id"])
        slug     = row["slug"]
        slot_ts  = int(row["slot_ts"])
        ticks    = fetch_ticks_for_market(mid, slug, slot_ts)
        return mid, ticks

    batch_size = 50
    for batch_start in range(0, len(todo), batch_size):
        batch = todo.iloc[batch_start : batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_market, row): row["market_id"]
                       for _, row in batch.iterrows()}
            for fut in as_completed(futures):
                mid, ticks = fut.result()
                done_ids.add(mid)
                if ticks:
                    all_new_ticks.extend(ticks)
                    processed += 1
                else:
                    empty += 1

        # Save progress every batch
        with open(PROGRESS_FILE, "w") as f:
            json.dump(list(done_ids), f)

        total_done = batch_start + len(batch)
        print(
            f"  {total_done:,}/{len(todo):,} | "
            f"with_ticks={processed} empty={empty} "
            f"new_ticks={len(all_new_ticks):,}"
        )
        time.sleep(0.1)

    # Save ticks
    if all_new_ticks:
        new_df = pd.DataFrame(all_new_ticks)
        if existing is not None:
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined = combined.drop_duplicates(
            subset=["market_id", "timestamp_ms", "outcome", "side"]
        )
        combined.to_parquet(OUT_FILE, index=False, compression="snappy")
        print(f"\nSaved {len(combined):,} total ticks to {OUT_FILE}")
        print(f"  New ticks added: {len(new_df):,}")
        print(f"  Markets with ticks: {combined['market_id'].nunique():,}")
    else:
        print("\nNo new ticks collected.")

    print(f"Progress saved: {len(done_ids):,} markets done")


if __name__ == "__main__":
    main()
