"""
Build temporal (early/mid/late window) tick features for BTC Polymarket model.
- Loads 616 resolved markets from v3 clean dataset
- Filters ticks to only those 616 markets using pyarrow filter (efficient)
- Computes per-segment features: early (0-90s), mid (90-210s), late (210-300s)
- Trains LGBMClassifier on v4 dataset and compares to v3
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"

# ─────────────────────────────────────────────────────────────
# 1. Load v3 dataset
# ─────────────────────────────────────────────────────────────
print("="*60)
print("Step 1: Loading v3 dataset...")
v3 = pd.read_parquet(f"{ARTIFACTS_DIR}/btc_5min_dataset_v3_clean.parquet")
print(f"  v3 shape: {v3.shape}")
print(f"  Columns: {list(v3.columns)}")

market_ids = v3["market_id"].tolist()
print(f"  Markets: {len(market_ids)}")

# start_ts is datetime[ms, UTC]; convert to milliseconds integer for comparison with timestamp_ms
v3["start_ts_ms"] = v3["start_ts"].astype("int64")  # already ms since this is datetime64[ms]

# ─────────────────────────────────────────────────────────────
# 2. Load ticks efficiently using pyarrow filter
# ─────────────────────────────────────────────────────────────
print("\nStep 2: Loading ticks with pyarrow filter (only 616 markets)...")
ticks_table = pq.read_table(
    f"{DATA_DIR}/ticks_btc_5min.parquet",
    columns=["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"],
    filters=[("market_id", "in", market_ids)]
)
ticks = ticks_table.to_pandas()
print(f"  Loaded {len(ticks):,} ticks for {ticks['market_id'].nunique()} markets")
print(f"  Sample:\n{ticks.head(3)}")
print(f"  outcome values: {ticks['outcome'].unique()}")
print(f"  side values: {ticks['side'].unique()}")

# ─────────────────────────────────────────────────────────────
# 3. Compute temporal features per market
# ─────────────────────────────────────────────────────────────
print("\nStep 3: Computing temporal features...")

# Build start_ts_ms lookup dict
start_ts_map = dict(zip(v3["market_id"], v3["start_ts_ms"]))

# Add relative time offset from market start (in seconds)
ticks["start_ts_ms"] = ticks["market_id"].map(start_ts_map)
ticks["t_sec"] = (ticks["timestamp_ms"] - ticks["start_ts_ms"]) / 1000.0

# Assign window
ticks["window"] = pd.cut(
    ticks["t_sec"],
    bins=[-np.inf, 90, 210, np.inf],
    labels=["early", "mid", "late"]
)

print(f"  t_sec range: [{ticks['t_sec'].min():.1f}, {ticks['t_sec'].max():.1f}]")
print(f"  Window distribution:\n{ticks['window'].value_counts()}")

# Flag up vs down ticks
ticks["is_up"] = ticks["outcome"] == "Up"
ticks["is_down"] = ticks["outcome"] == "Down"
ticks["vol_up"] = ticks["size_usdc"] * ticks["is_up"]
ticks["vol_down"] = ticks["size_usdc"] * ticks["is_down"]

def compute_window_features(group, window_label):
    """Compute features for one window of one market."""
    w = group[group["window"] == window_label]
    
    vol_up = w["vol_up"].sum()
    vol_down = w["vol_down"].sum()
    n_ticks = len(w)
    n_up = w["is_up"].sum()
    n_down = w["is_down"].sum()
    total_vol = vol_up + vol_down
    
    up_ratio = vol_up / (total_vol + 1e-8)
    vwap_up = (w[w["is_up"]]["price"] * w[w["is_up"]]["size_usdc"]).sum() / (vol_up + 1e-8)
    vwap_down = (w[w["is_down"]]["price"] * w[w["is_down"]]["size_usdc"]).sum() / (vol_down + 1e-8)
    
    return {
        f"{window_label}_vol_up": vol_up,
        f"{window_label}_vol_down": vol_down,
        f"{window_label}_up_ratio": up_ratio,
        f"{window_label}_n_ticks": n_ticks,
        f"{window_label}_n_up": n_up,
        f"{window_label}_n_down": n_down,
        f"{window_label}_total_vol": total_vol,
        f"{window_label}_vwap_up": vwap_up,
        f"{window_label}_vwap_down": vwap_down,
    }

# Group ticks by market_id and compute features
records = []
for mid, group in ticks.groupby("market_id"):
    row = {"market_id": mid}
    for wlabel in ["early", "mid", "late"]:
        row.update(compute_window_features(group, wlabel))
    
    total_n_ticks = row["early_n_ticks"] + row["mid_n_ticks"] + row["late_n_ticks"]
    row["total_temporal_n_ticks"] = total_n_ticks
    
    # Derived features
    row["momentum_vol_ratio"] = row["early_up_ratio"] / (row["late_up_ratio"] + 1e-8)
    row["late_surge"] = row["late_n_ticks"] / (total_n_ticks + 1e-8)
    row["early_mid_vol_diff"] = row["early_up_ratio"] - row["mid_up_ratio"]
    row["mid_late_vol_diff"] = row["mid_up_ratio"] - row["late_up_ratio"]
    row["vol_trend"] = row["late_total_vol"] / (row["early_total_vol"] + 1e-8)
    
    records.append(row)

temporal_df = pd.DataFrame(records)
print(f"  Temporal features computed for {len(temporal_df)} markets")
print(f"  New feature columns: {len(temporal_df.columns) - 1}")

# ─────────────────────────────────────────────────────────────
# 4. Merge with v3 dataset → v4
# ─────────────────────────────────────────────────────────────
print("\nStep 4: Merging with v3 dataset...")
v4 = v3.drop(columns=["start_ts_ms"]).merge(temporal_df, on="market_id", how="left")
print(f"  v4 shape: {v4.shape}")
print(f"  Missing temporal data: {v4['early_n_ticks'].isna().sum()} rows")

# Fill NAs with 0 for temporal features
temporal_cols = [c for c in v4.columns if any(c.startswith(p) for p in ["early_", "mid_", "late_", "momentum_", "vol_trend", "total_temporal"])]
v4[temporal_cols] = v4[temporal_cols].fillna(0)

# Save v4 dataset
v4.to_parquet(f"{ARTIFACTS_DIR}/btc_5min_dataset_v4_temporal.parquet", index=False)
print(f"  Saved: {ARTIFACTS_DIR}/btc_5min_dataset_v4_temporal.parquet")

# ─────────────────────────────────────────────────────────────
# 5. Train and compare LGBMClassifier
# ─────────────────────────────────────────────────────────────
print("\nStep 5: Training LGBM models (5-fold CV)...")

# Define feature sets
V3_FEATURES = [
    "first_price", "last_price", "price_mean", "price_std", "price_min", "price_max",
    "price_momentum", "n_ticks", "price_at_25pct", "price_at_50pct", "price_at_75pct",
    "hour_of_day_sin", "hour_of_day_cos", "day_of_week_sin", "day_of_week_cos",
    "total_volume_usdc", "buy_volume_usdc", "sell_volume_usdc", "buy_sell_imbalance",
    "up_volume_usdc", "down_volume_usdc", "up_down_volume_ratio",
    "n_trades", "n_buy_trades", "n_sell_trades", "avg_trade_size",
    "spot_price_start", "spot_price_end", "spot_return", "spot_volatility",
    "spot_price_mean", "vwap_up", "vwap_down"
]

NEW_FEATURES = temporal_cols + ["early_mid_vol_diff", "mid_late_vol_diff"]
# deduplicate
NEW_FEATURES = list(set(temporal_cols))

V4_FEATURES = [f for f in V3_FEATURES if f in v4.columns] + [f for f in NEW_FEATURES if f in v4.columns]

TARGET = "target"
y = v4[TARGET].values

def cv_evaluate(df, features, label):
    """5-fold CV with LGBMClassifier."""
    X = df[features].fillna(0).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, accs = [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
        )
        
        proba = model.predict_proba(X_val)[:, 1]
        pred = model.predict(X_val)
        aucs.append(roc_auc_score(y_val, proba))
        accs.append(accuracy_score(y_val, pred))
    
    print(f"\n  [{label}]")
    print(f"  AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Acc:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    return np.mean(aucs), np.mean(accs)

# Filter features to those that exist in v4
v3_feats = [f for f in V3_FEATURES if f in v4.columns]
v4_feats = V4_FEATURES

print(f"  v3 features: {len(v3_feats)}")
print(f"  v4 features: {len(v4_feats)}")

auc_v3, acc_v3 = cv_evaluate(v4, v3_feats, "v3 (baseline)")
auc_v4, acc_v4 = cv_evaluate(v4, v4_feats, "v4 (+ temporal)")

print("\n" + "="*60)
print("COMPARISON SUMMARY:")
print(f"  v3 AUC:  {auc_v3:.4f}  |  v4 AUC:  {auc_v4:.4f}  |  Δ = {auc_v4 - auc_v3:+.4f}")
print(f"  v3 Acc:  {acc_v3:.4f}  |  v4 Acc:  {acc_v4:.4f}  |  Δ = {acc_v4 - acc_v3:+.4f}")

# ─────────────────────────────────────────────────────────────
# 6. Final model: train on ALL v4 data, print feature importances
# ─────────────────────────────────────────────────────────────
print("\nStep 6: Training final model on full v4 data...")
X_all = v4[v4_feats].fillna(0).values

final_model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=10,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42,
    verbose=-1
)
final_model.fit(X_all, y)

# Feature importances
importances = pd.DataFrame({
    "feature": v4_feats,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 20 Feature Importances (v4 model):")
print(importances.head(20).to_string(index=False))

# Highlight which are temporal features
temporal_in_top20 = importances.head(20)[importances.head(20)["feature"].isin(NEW_FEATURES)]
print(f"\nTemporal features in top 20: {len(temporal_in_top20)}")
print(temporal_in_top20[["feature","importance"]].to_string(index=False))

# Save importances
importances.to_csv(f"{ARTIFACTS_DIR}/feature_importances_v4.csv", index=False)
print(f"\nSaved feature importances: {ARTIFACTS_DIR}/feature_importances_v4.csv")

print("\n" + "="*60)
print("DONE! Files created:")
print(f"  - {ARTIFACTS_DIR}/btc_5min_dataset_v4_temporal.parquet")
print(f"  - {ARTIFACTS_DIR}/feature_importances_v4.csv")
