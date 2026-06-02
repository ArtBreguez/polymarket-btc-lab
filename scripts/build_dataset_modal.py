"""
build_dataset_modal.py — Resolve BTC 5min markets and build updated dataset

Strategy:
  1. Download existing BrockMisner dataset (markets + ticks) as base
  2. Fetch ALL BTC 5min markets from Polymarket API (via gamma-api)
  3. For each market with slot_ts in the past: resolve via Binance klines
     (verified: close > open = UP = resolution 1)
  4. Cross-validate: compare our resolutions vs BrockMisner's 616 known labels
     Must achieve >99.5% agreement before proceeding
  5. Build updated markets.parquet with all resolved markets
  6. Upload to artbreguez/polymarket-btc-updown (private) on HuggingFace

NOTE: Ticks parquet from BrockMisner (2.4GB) is copied as-is since it's
      the ground truth for Polymarket order flow data.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow>=18.0",
        "pandas>=2.2",
        "numpy>=1.26",
        "requests>=2.32",
        "huggingface_hub>=0.26",
        "tqdm>=4.66",
    )
)

app = modal.App("polymarket-btc-build-dataset", image=image)

@app.function(
    cpu=4,
    memory=16384,
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
)
def build_and_upload():
    import gc, json, logging, os, sys, time, tempfile, warnings
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests
    from tqdm import tqdm
    from huggingface_hub import hf_hub_download, HfApi

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    HF_TOKEN       = os.environ.get("HF_TOKEN", "")
    SRC_DATASET    = "BrockMisner/polymarket-btc-updown"
    DST_DATASET    = "artbreguez/polymarket-btc-updown"
    DATA_DIR       = Path("/tmp/btc_dataset")
    DATA_DIR.mkdir(exist_ok=True)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")

    api = HfApi(token=HF_TOKEN)

    # ── Step 1: Create destination dataset repo if needed ─────────────────────
    log.info("Step 1: Ensuring HF dataset repo %s exists...", DST_DATASET)
    try:
        api.repo_info(DST_DATASET, repo_type="dataset")
        log.info("  Repo already exists.")
    except Exception:
        api.create_repo(DST_DATASET, repo_type="dataset", private=True)
        log.info("  Created private dataset repo: %s", DST_DATASET)

    # ── Step 2: Download source data from BrockMisner ─────────────────────────
    log.info("Step 2: Downloading source data from %s...", SRC_DATASET)
    files_to_copy = [
        "data/markets.parquet",
        "data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet",
        "data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet",
    ]
    for f in files_to_copy:
        dest = DATA_DIR / f
        if dest.exists():
            log.info("  Cached: %s (%.0f MB)", f, dest.stat().st_size / 1e6)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("  Downloading %s ...", f)
        t0 = time.time()
        hf_hub_download(repo_id=SRC_DATASET, filename=f, repo_type="dataset",
                        token=HF_TOKEN, local_dir=str(DATA_DIR),
                        local_dir_use_symlinks=False)
        log.info("    → %.1fs, %.0f MB", time.time() - t0, dest.stat().st_size / 1e6)

    # ── Step 3: Load existing markets ─────────────────────────────────────────
    log.info("Step 3: Loading existing markets from BrockMisner...")
    markets_src = pd.read_parquet(DATA_DIR / "data/markets.parquet")
    btc5_src = markets_src[
        (markets_src["crypto"] == "BTC") &
        (markets_src["timeframe"] == "5-minute")
    ].copy()
    btc5_src["slot_ts"] = btc5_src["slug"].apply(
        lambda s: int(str(s).split("-")[-1]) if str(s).split("-")[-1].isdigit() else 0)
    btc5_src = btc5_src[btc5_src["slot_ts"] > 0]
    known_resolved = btc5_src[btc5_src["resolution"].isin([0, 1])]
    log.info("  Existing resolved markets: %d", len(known_resolved))
    log.info("  Last resolved: %s",
             datetime.fromtimestamp(known_resolved["slot_ts"].max(), tz=timezone.utc))

    # ── Step 4: Extract ALL slot_ts from BrockMisner (has all 11k slugs) ────────
    log.info("Step 4: Extracting all BTC 5min slot_ts from BrockMisner markets...")
    # BrockMisner has all 11k markets (resolution=-1 for unresolved)
    btc5_all = markets_src[
        (markets_src["crypto"] == "BTC") &
        (markets_src["timeframe"] == "5-minute")
    ].copy()
    btc5_all["slot_ts"] = btc5_all["slug"].apply(
        lambda s: int(str(s).split("-")[-1]) if str(s).split("-")[-1].isdigit() else 0)
    btc5_all = btc5_all[btc5_all["slot_ts"] > 0].sort_values("slot_ts")
    log.info("  Total BTC 5min markets in BrockMisner: %d", len(btc5_all))
    log.info("  Resolution breakdown: %s", dict(btc5_all["resolution"].value_counts()))

    now_ts    = datetime.now(timezone.utc).timestamp()
    past_mask = btc5_all["slot_ts"] < (now_ts - 600)
    past_all  = btc5_all[past_mask]
    log.info("  Slots in the past (resolvable): %d", len(past_all))

    # Build lookup: slot_ts → row metadata
    slot_meta = btc5_all.set_index("slot_ts").to_dict("index")

    # Merge with known resolved to get market_id → condition_id mapping
    known_map = known_resolved.set_index("slot_ts")[
        ["market_id", "condition_id", "up_token_id", "down_token_id",
         "start_ts", "end_ts", "closed_ts", "fee_rate_bps"]
    ].to_dict("index")

    # ── Step 5: Resolve via Binance klines + cross-validate ───────────────────
    log.info("Step 5: Resolving markets via Binance BTCUSDT 1m klines...")

    BINANCE_URL = "https://api.binance.com/api/v3/klines"
    session = requests.Session()

    # Use all past slots from BrockMisner
    past_slots = sorted(past_all["slot_ts"].tolist())
    log.info("  Slots to resolve: %d", len(past_slots))

    # Batch-fetch Binance klines (process in chunks of 500)
    SLOT_DUR = 300   # 5 minutes
    resolution_map = {}   # slot_ts → 0 or 1
    errors = 0
    verified_vs_known = {"match": 0, "mismatch": 0, "missing": 0}

    def resolve_batch(slots_batch):
        """Fetch Binance klines for a batch of slots and return resolutions."""
        results = {}
        for slot_ts in slots_batch:
            try:
                r = session.get(
                    BINANCE_URL,
                    params={
                        "symbol":    "BTCUSDT",
                        "interval":  "1m",
                        "startTime": slot_ts * 1000,
                        "endTime":   (slot_ts + SLOT_DUR) * 1000,
                        "limit":     6,
                    },
                    timeout=15,
                )
                if r.status_code == 429:   # rate limit
                    time.sleep(5)
                    r = session.get(BINANCE_URL, params={
                        "symbol": "BTCUSDT", "interval": "1m",
                        "startTime": slot_ts * 1000,
                        "endTime": (slot_ts + SLOT_DUR) * 1000,
                        "limit": 6,
                    }, timeout=15)
                if r.status_code != 200:
                    continue
                klines = r.json()
                if not klines:
                    continue
                open_price  = float(klines[0][1])
                close_price = float(klines[-1][4])
                results[slot_ts] = 1 if close_price > open_price else 0
            except Exception:
                pass
        return results

    CHUNK = 200
    log.info("  Processing %d slots in chunks of %d...", len(past_slots), CHUNK)
    for i in range(0, len(past_slots), CHUNK):
        chunk = past_slots[i:i + CHUNK]
        batch_res = resolve_batch(chunk)
        resolution_map.update(batch_res)
        errors += len(chunk) - len(batch_res)
        if (i // CHUNK) % 5 == 0:
            log.info("  Progress: %d/%d resolved, %d errors",
                     len(resolution_map), len(past_slots), errors)
        time.sleep(0.05)   # gentle rate limiting

    log.info("  Total resolved: %d | Errors/missing: %d",
             len(resolution_map), errors)

    # ── Step 6: Cross-validate against known BrockMisner labels ───────────────
    log.info("Step 6: Cross-validating against BrockMisner ground truth...")
    for _, row in known_resolved.iterrows():
        ts  = row["slot_ts"]
        our = resolution_map.get(ts)
        if our is None:
            verified_vs_known["missing"] += 1
        elif our == int(row["resolution"]):
            verified_vs_known["match"] += 1
        else:
            verified_vs_known["mismatch"] += 1
            log.warning("  MISMATCH slot_ts=%d | ours=%d | known=%d",
                        ts, our, int(row["resolution"]))

    total_checked = verified_vs_known["match"] + verified_vs_known["mismatch"]
    agreement = verified_vs_known["match"] / max(total_checked, 1)
    log.info("  Validation: match=%d mismatch=%d missing=%d | agreement=%.3f%%",
             verified_vs_known["match"], verified_vs_known["mismatch"],
             verified_vs_known["missing"], agreement * 100)

    if agreement < 0.995:
        raise RuntimeError(
            f"Cross-validation FAILED: agreement={agreement:.3f} < 0.995. "
            "Check Binance resolution logic before proceeding.")
    log.info("  Cross-validation PASSED (%.2f%% agreement)", agreement * 100)

    # ── Step 7: Build updated markets.parquet ─────────────────────────────────
    log.info("Step 7: Building updated markets.parquet...")

    # Build full set of resolved BTC 5min markets
    new_rows = []
    for slot_ts, res in sorted(resolution_map.items()):
        # Check if we have this in the known dataset (use those metadata)
        if slot_ts in known_map:
            meta = known_map[slot_ts]
            row = {
                "market_id":    meta.get("market_id", ""),
                "question":     f"Bitcoin Up or Down 5min slot {slot_ts}",
                "crypto":       "BTC",
                "timeframe":    "5-minute",
                "volume":       0.0,
                "resolution":   res,
                "start_ts":     meta.get("start_ts", slot_ts - 300),
                "end_ts":       meta.get("end_ts", slot_ts + 300),
                "closed_ts":    meta.get("closed_ts", slot_ts + 300),
                "condition_id": meta.get("condition_id", ""),
                "up_token_id":  meta.get("up_token_id", ""),
                "down_token_id": meta.get("down_token_id", ""),
                "slug":         f"btc-updown-5m-{slot_ts}",
                "fee_rate_bps": meta.get("fee_rate_bps", 0),
            }
        else:
            # Use metadata from BrockMisner slot_meta lookup
            meta_src = slot_meta.get(slot_ts, {})
            row = {
                "market_id":    str(meta_src.get("market_id", "")),
                "question":     meta_src.get("question", f"Bitcoin Up or Down 5min {slot_ts}"),
                "crypto":       "BTC",
                "timeframe":    "5-minute",
                "volume":       float(meta_src.get("volume", 0) or 0),
                "resolution":   res,
                "start_ts":     int(meta_src.get("start_ts", slot_ts - 300) or slot_ts - 300),
                "end_ts":       int(meta_src.get("end_ts", slot_ts + 300) or slot_ts + 300),
                "closed_ts":    int(meta_src.get("closed_ts", slot_ts + 310) or slot_ts + 310),
                "condition_id": str(meta_src.get("condition_id", "")),
                "up_token_id":  str(meta_src.get("up_token_id", "")),
                "down_token_id": str(meta_src.get("down_token_id", "")),
                "slug":         f"btc-updown-5m-{slot_ts}",
                "fee_rate_bps": int(meta_src.get("fee_rate_bps", 0) or 0),
            }
        new_rows.append(row)

    df_new = pd.DataFrame(new_rows).sort_values("slot_ts").reset_index(drop=True)
    log.info("  New resolved BTC 5min markets: %d (was %d)",
             len(df_new), len(known_resolved))
    log.info("  Target balance: %s", dict(df_new["resolution"].value_counts()))

    # Merge back with ALL other crypto markets from BrockMisner (keep them)
    other_markets = markets_src[
        ~((markets_src["crypto"] == "BTC") &
          (markets_src["timeframe"] == "5-minute"))
    ].copy()
    log.info("  Other crypto markets (preserved): %d", len(other_markets))

    # Build final markets table
    btc5_cols = ["market_id", "question", "crypto", "timeframe", "volume",
                 "resolution", "start_ts", "end_ts", "closed_ts", "condition_id",
                 "up_token_id", "down_token_id", "slug", "fee_rate_bps"]
    df_final = pd.concat([
        other_markets[btc5_cols],
        df_new[btc5_cols],
    ], ignore_index=True)
    log.info("  Final markets table: %d rows", len(df_final))

    # Save updated markets.parquet
    markets_out = DATA_DIR / "updated_markets.parquet"
    df_final.to_parquet(markets_out, index=False)
    log.info("  Saved: %s (%.1f MB)", markets_out, markets_out.stat().st_size / 1e6)

    # ── Step 8: Upload to artbreguez HF dataset ───────────────────────────────
    log.info("Step 8: Uploading to %s...", DST_DATASET)

    # Upload updated markets.parquet
    api.upload_file(
        path_or_fileobj=str(markets_out),
        path_in_repo="data/markets.parquet",
        repo_id=DST_DATASET, repo_type="dataset",
        commit_message=(f"Update markets: {len(df_new)} BTC 5min resolved "
                        f"(was 616, now {len(df_new)}) | Binance cross-validated"),
    )
    log.info("  Uploaded markets.parquet (%d rows)", len(df_final))

    # Copy ticks and orderbook from BrockMisner (unchanged — ground truth)
    for fname in [
        "data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet",
        "data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet",
    ]:
        local = DATA_DIR / fname
        if local.exists():
            log.info("  Uploading %s (%.0f MB)...", fname, local.stat().st_size / 1e6)
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=fname,
                repo_id=DST_DATASET, repo_type="dataset",
                commit_message=f"Copy from BrockMisner: {fname}",
            )

    # Upload a dataset card
    readme = f"""---
license: mit
private: true
---

# polymarket-btc-updown (artbreguez fork)

Updated version of BrockMisner/polymarket-btc-updown with resolved labels.

## What's different

- `data/markets.parquet`: BTC 5-minute markets resolved via Binance BTCUSDT klines
  - **{len(df_new)} BTC 5min markets resolved** (vs 616 in BrockMisner)
  - Cross-validated: {verified_vs_known['match']}/{total_checked} matches with BrockMisner ({agreement*100:.2f}%)
  - Resolution logic: `resolution = 1 if btc_close > btc_open else 0` (Binance 1m klines)
- `data/ticks/`: original Polymarket order flow ticks (copied from BrockMisner)
- `data/orderbook/`: original Polymarket orderbook snapshots (copied from BrockMisner)

## Resolution methodology

1. For each BTC 5min slot at timestamp `slot_ts`:
   - Fetch Binance klines for `[slot_ts, slot_ts+300]`
   - `open_price` = open of first 1m candle
   - `close_price` = close of last 1m candle
   - `resolution = 1 (UP)` if `close_price > open_price`, else `0 (DOWN)`
2. Cross-validated against BrockMisner ground truth: {agreement*100:.2f}% agreement

Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as fp:
        fp.write(readme)
    api.upload_file(
        path_or_fileobj=fp.name,
        path_in_repo="README.md",
        repo_id=DST_DATASET, repo_type="dataset",
        commit_message="Add dataset card",
    )

    log.info("=" * 60)
    log.info("DONE")
    log.info("  BTC 5min resolved: %d (was 616)", len(df_new))
    log.info("  Cross-validation: %.2f%% agreement with BrockMisner", agreement * 100)
    log.info("  Dataset: https://huggingface.co/datasets/%s", DST_DATASET)
    log.info("=" * 60)

    return {
        "n_resolved":    len(df_new),
        "n_errors":      errors,
        "agreement_pct": round(agreement * 100, 2),
        "dataset_url":   f"https://huggingface.co/datasets/{DST_DATASET}",
    }


@app.local_entrypoint()
def main():
    print("Building and uploading updated BTC dataset to HF...")
    r = build_and_upload.remote()
    print(f"\n{'='*55}")
    print(f"DATASET BUILD COMPLETE")
    print(f"  Resolved markets:  {r['n_resolved']}")
    print(f"  Errors:            {r['n_errors']}")
    print(f"  Cross-validation:  {r['agreement_pct']}% agreement with BrockMisner")
    print(f"  Dataset:           {r['dataset_url']}")
    print(f"{'='*55}")
