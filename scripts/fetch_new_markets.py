"""
fetch_new_markets.py
====================
Fetches BTC 5min resolved markets from Apr 12 → Jun 3 2026 (the gap since
our local dataset ends at Apr 11). Saves to data/new_markets.csv.

Then fetch_new_ticks.py collects the tick data for those markets.

Usage:
    python scripts/fetch_new_markets.py
"""
import json, time, requests, csv, re
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR  = Path(__file__).parent.parent / "data"
OUT_FILE  = DATA_DIR / "new_markets.csv"
WORKERS   = 20
PAUSE_S   = 0.3

# Our dataset ends at Apr 11 16:35 UTC (slot_ts=1775925300)
# We want everything after that up to now minus 1h (ensure resolution)
DATASET_END_TS = 1775925300
NOW_TS         = int(time.time()) - 3600  # 1h buffer

# Generate all 5min slot timestamps in range
slots = list(range(DATASET_END_TS + 300, NOW_TS, 300))
print(f"Scanning {len(slots):,} slots from "
      f"{datetime.fromtimestamp(slots[0], tz=timezone.utc).strftime('%Y-%m-%d')} to "
      f"{datetime.fromtimestamp(slots[-1], tz=timezone.utc).strftime('%Y-%m-%d')}")

# Resume: load already-fetched slots
already_slots = set()
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        for row in csv.DictReader(f):
            already_slots.add(int(row["slot_ts"]))
    print(f"Already fetched: {len(already_slots):,} slots, remaining: {len(slots) - len(already_slots):,}")

todo = [s for s in slots if s not in already_slots]
print(f"Fetching {len(todo):,} slots...")

write_header = not OUT_FILE.exists() or OUT_FILE.stat().st_size == 0
out_f = open(OUT_FILE, "a", newline="")
writer = csv.DictWriter(out_f, fieldnames=["market_id", "slug", "slot_ts", "target"])
if write_header:
    writer.writeheader()

found = 0
not_found = 0
not_closed = 0

def fetch_slot(slot_ts):
    slug = f"btc-updown-5m-{slot_ts}"
    try:
        r = requests.get(
            f"https://gamma-api.polymarket.com/markets/slug/{slug}",
            timeout=10,
        )
        if not r.ok:
            return None, "not_found"
        data = r.json()
        if not data or isinstance(data, list) and len(data) == 0:
            return None, "not_found"
        m = data if isinstance(data, dict) else data[0]
        if not m.get("closed"):
            return None, "not_closed"
        op = m.get("outcomePrices", "")
        if not op:
            return None, "no_outcome"
        prices = json.loads(op) if isinstance(op, str) else op
        target = 1 if float(prices[0]) >= 0.5 else 0
        return {
            "market_id": str(m["id"]),
            "slug":      m["slug"],
            "slot_ts":   slot_ts,
            "target":    target,
        }, "ok"
    except Exception as e:
        return None, f"error:{e}"

batch_size = 200
for batch_start in range(0, len(todo), batch_size):
    batch = todo[batch_start:batch_start + batch_size]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_slot, s): s for s in batch}
        for fut in as_completed(futures):
            result, status = fut.result()
            if status == "ok" and result:
                writer.writerow(result)
                out_f.flush()
                found += 1
            elif status == "not_found":
                not_found += 1
            elif status == "not_closed":
                not_closed += 1

    progress = batch_start + len(batch)
    print(f"  {progress:,}/{len(todo):,} | found={found} not_found={not_found} not_closed={not_closed}")
    time.sleep(PAUSE_S)

out_f.close()
print(f"\nDone. Found {found:,} new resolved markets.")
print(f"Saved to {OUT_FILE}")
