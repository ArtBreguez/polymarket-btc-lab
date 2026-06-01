"""
STEP 1+2: Leakage audit + build clean v4 dataset.

Confirms leakage in v4_temporal, then builds v4_clean using proper end_ts filter.
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings("ignore")

ARTIFACTS = "artifacts"
DATA = "data"

print("=" * 60)
print("STEP 1: Leakage Audit")
print("=" * 60)

v3 = pd.read_parquet(f"{ARTIFACTS}/btc_5min_dataset_v3_clean.parquet")
market_ids = v3["market_id"].tolist()
v3["start_ts_ms"] = v3["start_ts"].astype("int64")
v3["end_ts_ms"] = v3["start_ts_ms"] + 300_000  # 5 min in ms

# Load ticks for our 616 markets
print(f"Loading ticks for {len(market_ids)} markets...")
ticks = pq.read_table(
    f"{DATA}/ticks_btc_5min.parquet",
    columns=["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"],
    filters=[("market_id", "in", market_ids)]
).to_pandas()
print(f"Total ticks loaded: {len(ticks):,}")

# Add timing info
start_map = dict(zip(v3["market_id"], v3["start_ts_ms"]))
end_map   = dict(zip(v3["market_id"], v3["end_ts_ms"]))
ticks["start_ts_ms"] = ticks["market_id"].map(start_map)
ticks["end_ts_ms"]   = ticks["market_id"].map(end_map)
ticks["t_sec"] = (ticks["timestamp_ms"] - ticks["start_ts_ms"]) / 1000.0

# Count leakage
pre_start   = (ticks["t_sec"] < 0).sum()
in_window   = ((ticks["t_sec"] >= 0) & (ticks["t_sec"] <= 300)).sum()
post_end    = (ticks["t_sec"] > 300).sum()
total       = len(ticks)

print(f"\nTick timing breakdown:")
print(f"  Pre-start  (t < 0):      {pre_start:>8,}  ({pre_start/total*100:.1f}%)")
print(f"  In-window  (0-300s):     {in_window:>8,}  ({in_window/total*100:.1f}%)")
print(f"  Post-end   (t > 300s):   {post_end:>8,}  ({post_end/total*100:.1f}%)  ← LEAKAGE")
print(f"  Total:                   {total:>8,}")

# Compare with v3 n_ticks
v4_old = pd.read_parquet(f"{ARTIFACTS}/btc_5min_dataset_v4_temporal.parquet")
print(f"\nv3 n_ticks (clean): mean={v3['n_ticks'].mean():.0f}, total={v3['n_ticks'].sum():.0f}")
print(f"v4 total_temporal:  mean={v4_old['total_temporal_n_ticks'].mean():.0f}, total={v4_old['total_temporal_n_ticks'].sum():.0f}")
print(f"Excess (leakage):   mean={v4_old['total_temporal_n_ticks'].mean() - v3['n_ticks'].mean():.0f} ticks/market")

print("\n" + "=" * 60)
print("STEP 2: Build v4_clean (with proper end_ts filter)")
print("=" * 60)

# Filter to clean window only
ticks_clean = ticks[(ticks["t_sec"] >= 0) & (ticks["t_sec"] <= 300)].copy()
print(f"Clean ticks: {len(ticks_clean):,} (removed {len(ticks) - len(ticks_clean):,} leakage ticks)")

# Window assignment with hard caps
ticks_clean["window"] = pd.cut(
    ticks_clean["t_sec"],
    bins=[-0.001, 90, 210, 300.001],
    labels=["early", "mid", "late"]
)

print(f"Window distribution:\n{ticks_clean['window'].value_counts()}")

# Outcome flags
ticks_clean["is_up"]   = ticks_clean["outcome"] == "Up"
ticks_clean["is_down"] = ticks_clean["outcome"] == "Down"
ticks_clean["vol_up"]   = ticks_clean["size_usdc"] * ticks_clean["is_up"]
ticks_clean["vol_down"] = ticks_clean["size_usdc"] * ticks_clean["is_down"]

def window_features(group, wlabel):
    w = group[group["window"] == wlabel]
    vol_up   = w["vol_up"].sum()
    vol_down = w["vol_down"].sum()
    total    = vol_up + vol_down
    up_ratio = vol_up / (total + 1e-8)
    vwap_up  = (w[w["is_up"]]["price"] * w[w["is_up"]]["size_usdc"]).sum() / (vol_up + 1e-8)
    vwap_dn  = (w[w["is_down"]]["price"] * w[w["is_down"]]["size_usdc"]).sum() / (vol_down + 1e-8)
    return {
        f"{wlabel}_vol_up":    vol_up,
        f"{wlabel}_vol_down":  vol_down,
        f"{wlabel}_up_ratio":  up_ratio,
        f"{wlabel}_n_ticks":   len(w),
        f"{wlabel}_n_up":      int(w["is_up"].sum()),
        f"{wlabel}_n_down":    int(w["is_down"].sum()),
        f"{wlabel}_total_vol": total,
        f"{wlabel}_vwap_up":   vwap_up,
        f"{wlabel}_vwap_down": vwap_dn,
    }

records = []
for mid, grp in ticks_clean.groupby("market_id"):
    row = {"market_id": mid}
    for w in ["early", "mid", "late"]:
        row.update(window_features(grp, w))
    total_n = row["early_n_ticks"] + row["mid_n_ticks"] + row["late_n_ticks"]
    row["total_temporal_n_ticks"] = total_n
    row["momentum_vol_ratio"]   = row["early_up_ratio"] / (row["late_up_ratio"] + 1e-8)
    row["late_surge"]           = row["late_n_ticks"] / (total_n + 1e-8)
    row["early_mid_vol_diff"]   = row["early_up_ratio"] - row["mid_up_ratio"]
    row["mid_late_vol_diff"]    = row["mid_up_ratio"]  - row["late_up_ratio"]
    row["vol_trend"]            = row["late_total_vol"] / (row["early_total_vol"] + 1e-8)
    records.append(row)

temporal_df = pd.DataFrame(records)
print(f"Temporal features for {len(temporal_df)} markets")

# Merge with v3
v4_clean = v3.drop(columns=["start_ts_ms", "end_ts_ms"]).merge(temporal_df, on="market_id", how="left")
temporal_cols = [c for c in v4_clean.columns if c not in v3.columns and c != "start_ts_ms"]
v4_clean[temporal_cols] = v4_clean[temporal_cols].fillna(0)

v4_clean.to_parquet(f"{ARTIFACTS}/btc_5min_dataset_v4_clean.parquet", index=False)
print(f"Saved: {ARTIFACTS}/btc_5min_dataset_v4_clean.parquet  shape={v4_clean.shape}")

# Sanity: n_ticks_clean should match v3
n_clean_temporal = v4_clean["total_temporal_n_ticks"].mean()
n_v3 = v3["n_ticks"].mean()
print(f"\nSanity check — mean ticks/market:")
print(f"  v3 n_ticks:             {n_v3:.0f}")
print(f"  v4_clean total_temporal:{n_clean_temporal:.0f}")
print(f"  Difference:             {n_clean_temporal - n_v3:.0f}  (should be ~0 or small)")
print("\nStep 2 done.")
