"""
upload_expansion_data.py — Upload expanded dataset to Modal Volume
===================================================================
Merges new ticks + new markets + updated Binance spot into Modal Volume.
"""
import modal

LOCAL_VOL = modal.Volume.from_name("btc-local-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow>=18.0", "pandas>=2.2")
)

app = modal.App("btc-upload-expansion", image=image)


@app.function(
    cpu=2,
    memory=8192,
    timeout=1800,
    volumes={"/btc_local": LOCAL_VOL},
)
def upload_data(
    new_ticks_bytes: bytes,
    new_markets_bytes: bytes,
    binance_spot_bytes: bytes,
):
    import io
    import pandas as pd
    from pathlib import Path

    LOCAL_DIR = Path("/btc_local")

    # 1. Merge ticks
    print("Loading existing ticks...")
    existing = pd.read_parquet(str(LOCAL_DIR / "ticks_btc_full_clean.parquet"))
    print(f"Existing: {len(existing)} rows, {existing['market_id'].nunique()} markets")

    new_ticks = pd.read_parquet(io.BytesIO(new_ticks_bytes))
    print(f"New ticks: {len(new_ticks)} rows, {new_ticks['market_id'].nunique()} markets")

    # Ensure consistent schema
    for col in ["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"]:
        if col not in new_ticks.columns:
            new_ticks[col] = 0
    new_ticks["market_id"] = new_ticks["market_id"].astype(str)

    # Remove duplicates: keep existing + add truly new
    existing["market_id"] = existing["market_id"].astype(str)
    existing_mids = set(existing["market_id"].unique())
    truly_new = new_ticks[~new_ticks["market_id"].isin(existing_mids)]
    print(f"Truly new ticks: {len(truly_new)} rows, {truly_new['market_id'].nunique()} markets")

    merged = pd.concat([existing, truly_new[existing.columns]], ignore_index=True)
    merged.to_parquet(str(LOCAL_DIR / "ticks_btc_full_clean.parquet"), index=False)
    print(f"Merged ticks: {len(merged)} rows, {merged['market_id'].nunique()} markets")

    # Also save new ticks separately for v22 training
    new_ticks.to_parquet(str(LOCAL_DIR / "new_ticks_pmdata.parquet"), index=False)

    # 2. Merge markets
    existing_mkts = pd.read_csv(str(LOCAL_DIR / "all_markets.csv"))
    existing_mkts["market_id"] = existing_mkts["market_id"].astype(str)
    
    new_mkts = pd.read_csv(io.BytesIO(new_markets_bytes))
    new_mkts["market_id"] = new_mkts["market_id"].astype(str)
    
    existing_mid_set = set(existing_mkts["market_id"])
    truly_new_mkts = new_mkts[~new_mkts["market_id"].isin(existing_mid_set)]
    
    if len(truly_new_mkts) > 0 and "target" in truly_new_mkts.columns:
        merged_mkts = pd.concat([existing_mkts, truly_new_mkts[existing_mkts.columns]], ignore_index=True)
        merged_mkts = merged_mkts.sort_values("slot_ts").reset_index(drop=True)
        merged_mkts.to_csv(str(LOCAL_DIR / "all_markets.csv"), index=False)
        print(f"Merged markets: {len(merged_mkts)}")
    else:
        # Save new_markets.csv separately for v22 to pick up
        new_mkts.to_csv(str(LOCAL_DIR / "new_markets.csv"), index=False)
        print(f"Saved new_markets.csv ({len(new_mkts)} rows) — no target column, will need resolution")

    # 3. Update Binance spot
    spot_local = pd.read_parquet(io.BytesIO(binance_spot_bytes))
    
    spot_full_path = LOCAL_DIR / "binance_spot_full.parquet"
    if spot_full_path.exists():
        spot_existing = pd.read_parquet(str(spot_full_path))
        spot_merged = pd.concat([spot_existing, spot_local], ignore_index=True)
        spot_merged = spot_merged.drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
        spot_merged.to_parquet(str(spot_full_path), index=False)
        print(f"Merged Binance spot: {len(spot_merged)} candles")
    else:
        spot_local.to_parquet(str(spot_full_path), index=False)
        print(f"Created Binance spot: {len(spot_local)} candles")

    # Also save as binance_spot_local for backward compat
    spot_local.to_parquet(str(LOCAL_DIR / "binance_spot_local.parquet"), index=False)

    LOCAL_VOL.commit()
    print("Volume committed!")


@app.local_entrypoint()
def main():
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "data"

    ticks_path = data_dir / "new_ticks_pmdata.parquet"
    mkts_path  = data_dir / "new_markets.csv"
    spot_path  = data_dir / "binance_spot_local.parquet"

    print(f"Uploading: ticks={ticks_path.stat().st_size/1e6:.1f}MB, "
          f"markets={mkts_path.stat().st_size/1e3:.0f}KB, "
          f"spot={spot_path.stat().st_size/1e3:.0f}KB")

    with open(ticks_path, "rb") as f:
        ticks_bytes = f.read()
    with open(mkts_path, "rb") as f:
        mkts_bytes = f.read()
    with open(spot_path, "rb") as f:
        spot_bytes = f.read()

    upload_data.remote(ticks_bytes, mkts_bytes, spot_bytes)
    print("Done!")
