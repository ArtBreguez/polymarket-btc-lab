"""
fetch_local_targets.py

Fetch resolution targets for all market_ids in data/local_market_ids.json
via Gamma API. Saves results incrementally to data/local_markets.csv.

Runs slowly on purpose — 20 workers, 0.5s pause between batches of 100.
Safe to interrupt and resume (skips already-fetched market_ids).
"""
import json, time, requests, csv, os, concurrent.futures
from pathlib import Path

DATA_DIR   = Path(__file__).parent.parent / "data"
MIDS_FILE  = DATA_DIR / "local_market_ids.json"
OUT_FILE   = DATA_DIR / "local_markets.csv"
BATCH_SIZE = 100
WORKERS    = 20
PAUSE_S    = 0.5

with open(MIDS_FILE) as f:
    all_mids = json.load(f)

# Load already-fetched
already = set()
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            already.add(row["market_id"])
    print(f"Already fetched: {len(already)}, remaining: {len(all_mids) - len(already)}")

# Open CSV for append
write_header = not OUT_FILE.exists()
out_f = open(OUT_FILE, "a", newline="")
writer = csv.DictWriter(out_f, fieldnames=["market_id", "slug", "slot_ts", "target"])
if write_header:
    writer.writeheader()

todo = [m for m in all_mids if m not in already]
print(f"Fetching {len(todo)} markets...")

def fetch(mid):
    try:
        r = requests.get(
            f"https://gamma-api.polymarket.com/markets/{mid}",
            timeout=8,
        )
        if not r.ok:
            return None
        d = r.json()
        slug = d.get("slug", "")
        if "btc-updown-5m" not in slug:
            return None
        if not d.get("closed"):
            return None
        op = d.get("outcomePrices", "")
        if not op:
            return None
        prices = json.loads(op) if isinstance(op, str) else op
        target = 1 if float(prices[0]) >= 0.5 else 0
        slot_ts = int(slug.split("-")[-1])
        return {"market_id": mid, "slug": slug, "slot_ts": slot_ts, "target": target}
    except Exception:
        return None

total_ok = 0
for i in range(0, len(todo), BATCH_SIZE):
    batch = todo[i: i + BATCH_SIZE]
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(fetch, batch))
    ok = [r for r in results if r]
    for row in ok:
        writer.writerow(row)
    out_f.flush()
    total_ok += len(ok)
    pct = 100 * (i + len(batch)) / len(todo)
    print(f"  {i + len(batch)}/{len(todo)} ({pct:.0f}%) — {len(ok)}/{len(batch)} BTC ok — total: {total_ok}")
    time.sleep(PAUSE_S)

out_f.close()
print(f"\nDone. {total_ok} BTC 5min markets saved to {OUT_FILE}")
