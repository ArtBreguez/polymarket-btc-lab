#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pyarrow>=18.0",
#   "pandas>=2.2",
#   "lightgbm==4.6.0",
#   "scikit-learn==1.8.0",
#   "numpy>=1.26",
#   "optuna>=3.6",
#   "huggingface_hub>=0.26",
# ]
# ///
"""
train_v5.py — BTC 5min model v5 (HF Jobs)

New vs v4:
  - Orderbook features: bid/ask spread, depth imbalance, book skew at slot open
  - Cross-asset tick features: ETH/SOL order flow at same inslot window
  - BTC spot return from spot_price_usdt already in ticks (no separate lookup needed)
  - Pre-slot BTC spot from ticks of adjacent slots
  - 100 Optuna trials (vs 40)
  - class_weight=balanced + isotonic calibration

Data sources (downloaded from HF BrockMisner/polymarket-btc-updown):
  - data/markets.parquet
  - data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet   (~2.3GB)
  - data/ticks/crypto=ETH/timeframe=5-minute/part-0.parquet
  - data/ticks/crypto=SOL/timeframe=5-minute/part-0.parquet
  - data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet (~3.2GB)

Output:
  - Uploads champion.pkl to artbreguez/polymarket-btc-model if beats v4 AUC=0.843
"""

import gc
import json
import logging
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HF_DATASET    = "BrockMisner/polymarket-btc-updown"
HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
HF_TOKEN      = os.environ.get("HF_TOKEN")
CHAMPION_AUC  = 0.843   # v4 benchmark

OBS_SECS      = 180     # first 3 min of each 5-min slot
OPTUNA_TRIALS = 100
DATA_DIR      = Path("/tmp/hf_data")
DATA_DIR.mkdir(exist_ok=True)

# ── Step 1: Download data from HF ─────────────────────────────────────────────
def download_hf_files():
    from huggingface_hub import hf_hub_download
    files_to_download = [
        "data/markets.parquet",
        "data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet",
        "data/ticks/crypto=ETH/timeframe=5-minute/part-0.parquet",
        "data/ticks/crypto=SOL/timeframe=5-minute/part-0.parquet",
        "data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet",
        "data/spot_prices/part-0.parquet",
    ]
    for f in files_to_download:
        dest = DATA_DIR / f
        if dest.exists():
            log.info("  Already cached: %s", f)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("  Downloading %s ...", f)
        t0 = time.time()
        hf_hub_download(
            repo_id=HF_DATASET,
            filename=f,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=str(DATA_DIR),
            local_dir_use_symlinks=False,
        )
        log.info("    → %.1fs", time.time() - t0)

# ── Step 2: Load markets ──────────────────────────────────────────────────────
def load_markets() -> pd.DataFrame:
    m = pd.read_parquet(DATA_DIR / "data/markets.parquet")
    btc = m[
        (m["crypto"] == "BTC") &
        (m["timeframe"] == "5-minute") &
        (m["resolution"].notna()) &
        (m["resolution"] != -1)
    ].copy()
    btc["slot_ts"] = btc["slug"].apply(
        lambda s: int(str(s).split("-")[-1]) if str(s).split("-")[-1].isdigit() else 0
    )
    btc = btc[btc["slot_ts"] > 0].sort_values("slot_ts")
    log.info("Resolved BTC markets: %d", len(btc))
    return btc

# ── Step 3: Order flow features from ticks ───────────────────────────────────
def compute_tick_features(df: pd.DataFrame, label: str) -> dict:
    """Compute order flow features for a group of inslot ticks."""
    n = len(df)
    if n == 0:
        return {
            f"{label}_n_ticks": 0.0, f"{label}_up_ratio": 0.5,
            f"{label}_momentum": 0.0, f"{label}_vwap_spread": 0.0,
            f"{label}_vol_up": 0.0, f"{label}_vol_dn": 0.0,
            f"{label}_buy_ratio": 0.5, f"{label}_avg_size": 0.0,
            f"{label}_up_w0": 0.5, f"{label}_up_w1": 0.5, f"{label}_up_w2": 0.5,
        }

    is_up = df["outcome"] == "Up"
    vol_up = (df["size_usdc"] * is_up).sum()
    vol_dn = (df["size_usdc"] * ~is_up).sum()
    total = vol_up + vol_dn + 1e-8

    vwap_up = (df.loc[is_up, "price"] * df.loc[is_up, "size_usdc"]).sum() / (vol_up + 1e-8) if is_up.any() else 0.5
    vwap_dn = (df.loc[~is_up, "price"] * df.loc[~is_up, "size_usdc"]).sum() / (vol_dn + 1e-8) if (~is_up).any() else 0.5

    def ur(mask):
        vu = (df.loc[mask, "size_usdc"] * (df.loc[mask, "outcome"] == "Up")).sum()
        t = df.loc[mask, "size_usdc"].sum() + 1e-8
        return float(vu / t)

    w0 = df["t_sec"] < 60
    w1 = (df["t_sec"] >= 60) & (df["t_sec"] < 120)
    w2 = df["t_sec"] >= 120

    up_w0 = ur(w0) if w0.any() else 0.5
    up_w1 = ur(w1) if w1.any() else 0.5
    up_w2 = ur(w2) if w2.any() else 0.5

    return {
        f"{label}_n_ticks":     float(n),
        f"{label}_vol_up":      float(vol_up),
        f"{label}_vol_dn":      float(vol_dn),
        f"{label}_up_ratio":    float(vol_up / total),
        f"{label}_vwap_up":     float(vwap_up),
        f"{label}_vwap_dn":     float(vwap_dn),
        f"{label}_vwap_spread": float(vwap_up - vwap_dn),
        f"{label}_buy_ratio":   float((df["side"] == "BUY").sum() / (n + 1e-8)),
        f"{label}_avg_size":    float(total / n),
        f"{label}_momentum":    float(up_w2 - up_w0),
        f"{label}_up_w0":       float(up_w0),
        f"{label}_up_w1":       float(up_w1),
        f"{label}_up_w2":       float(up_w2),
    }

# ── Step 4: Orderbook features ────────────────────────────────────────────────
def compute_orderbook_features(ob_df: pd.DataFrame) -> dict:
    """
    Orderbook snapshot at slot open.
    Columns expected: market_id, timestamp_ms, bids (JSON), asks (JSON) or
    separate bid_price/bid_size/ask_price/ask_size cols.
    """
    if ob_df is None or len(ob_df) == 0:
        return {
            "ob_spread": 0.0, "ob_mid": 0.5,
            "ob_bid_depth_5": 0.0, "ob_ask_depth_5": 0.0,
            "ob_imbalance": 0.0, "ob_skew": 0.0,
        }

    # Use first snapshot (closest to slot open)
    row = ob_df.iloc[0]
    cols = set(ob_df.columns)

    # Try to extract bid/ask prices and sizes
    # Format varies: may have best_bid, best_ask, or levels
    if "best_bid" in cols and "best_ask" in cols:
        bid = float(row.get("best_bid", 0) or 0)
        ask = float(row.get("best_ask", 1) or 1)
        spread = ask - bid
        mid = (bid + ask) / 2
        bid_d = float(row.get("bid_size_5", row.get("bid_depth", 0)) or 0)
        ask_d = float(row.get("ask_size_5", row.get("ask_depth", 0)) or 0)
    elif "bids" in cols:
        try:
            bids = json.loads(row["bids"]) if isinstance(row["bids"], str) else row["bids"]
            asks = json.loads(row["asks"]) if isinstance(row["asks"], str) else row["asks"]
            if bids and asks:
                bid = float(bids[0][0]) if bids else 0
                ask = float(asks[0][0]) if asks else 1
                spread = ask - bid
                mid = (bid + ask) / 2
                bid_d = sum(float(b[1]) for b in bids[:5]) if bids else 0
                ask_d = sum(float(a[1]) for a in asks[:5]) if asks else 0
            else:
                return compute_orderbook_features(None)
        except Exception:
            return compute_orderbook_features(None)
    else:
        return compute_orderbook_features(None)

    total_depth = bid_d + ask_d + 1e-8
    imbalance = (bid_d - ask_d) / total_depth   # positive = more bids (bullish)
    skew = mid - 0.5   # how far mid is from 50/50

    return {
        "ob_spread":       float(spread),
        "ob_mid":          float(mid),
        "ob_bid_depth_5":  float(bid_d),
        "ob_ask_depth_5":  float(ask_d),
        "ob_imbalance":    float(imbalance),
        "ob_skew":         float(skew),
    }

# ── Step 5: Build full dataset ────────────────────────────────────────────────
def build_dataset(markets: pd.DataFrame) -> pd.DataFrame:
    from datetime import datetime, timezone

    market_ids = set(markets["market_id"])
    slot_map   = dict(zip(markets["market_id"], markets["slot_ts"]))
    target_map = dict(zip(markets["market_id"], markets["resolution"].astype(int)))

    # ── Load BTC ticks ────────────────────────────────────────────────────────
    log.info("Loading BTC ticks...")
    btc_ticks = pq.read_table(
        str(DATA_DIR / "data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet"),
        columns=["market_id", "timestamp_ms", "outcome", "side", "price",
                 "size_usdc", "spot_price_usdt"],
        filters=[("market_id", "in", list(market_ids))],
    ).to_pandas()
    btc_ticks["slot_ts"] = btc_ticks["market_id"].map(slot_map)
    btc_ticks["t_sec"] = btc_ticks["timestamp_ms"] / 1000 - btc_ticks["slot_ts"]
    btc_inslot = btc_ticks[(btc_ticks["t_sec"] >= 0) & (btc_ticks["t_sec"] < OBS_SECS)].copy()
    log.info("  BTC inslot ticks: %d (markets: %d)", len(btc_inslot), btc_inslot["market_id"].nunique())

    # Build spot return features from spot_price_usdt in ticks
    # Pre-slot: find BTC price OBS_SECS before slot_ts using ticks of PREVIOUS slots
    # We'll build a spot timeline from all ticks and lookup by timestamp
    log.info("Building BTC spot timeline from ticks...")
    spot_timeline = btc_ticks[["timestamp_ms", "spot_price_usdt"]].dropna()
    spot_timeline = spot_timeline.drop_duplicates("timestamp_ms").set_index("timestamp_ms").sort_index()
    spot_ts = spot_timeline.index.values
    spot_px = spot_timeline["spot_price_usdt"].values
    del btc_ticks
    gc.collect()

    # ── Load cross-asset ticks (ETH, SOL) ─────────────────────────────────────
    cross_ticks = {}
    for crypto in ["ETH", "SOL"]:
        p = DATA_DIR / f"data/ticks/crypto={crypto}/timeframe=5-minute/part-0.parquet"
        if not p.exists():
            continue
        log.info("Loading %s ticks...", crypto)
        df = pq.read_table(
            str(p),
            columns=["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"],
        ).to_pandas()
        df["crypto"] = crypto
        cross_ticks[crypto] = df
        log.info("  %s: %d rows", crypto, len(df))

    # ── Load orderbook ────────────────────────────────────────────────────────
    ob_path = DATA_DIR / "data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet"
    ob_by_market = {}
    if ob_path.exists():
        log.info("Loading BTC orderbook...")
        ob_schema = pq.read_schema(str(ob_path))
        ob_cols = [f.name for f in ob_schema]
        log.info("  Orderbook cols: %s", ob_cols[:10])
        ob_df = pq.read_table(
            str(ob_path),
            filters=[("market_id", "in", list(market_ids))],
        ).to_pandas()
        log.info("  Orderbook rows: %d", len(ob_df))
        # Group by market_id, sort by timestamp, keep first snapshot
        for mid, grp in ob_df.groupby("market_id"):
            ob_by_market[mid] = grp.sort_values("timestamp_ms").head(5)
        del ob_df
        gc.collect()

    # ── Build cross-asset slot index ──────────────────────────────────────────
    # For ETH/SOL: find ticks within [slot_ts, slot_ts+OBS_SECS) by timestamp
    # We create a slot_ts → list of (outcome, side, price, size_usdc, t_sec) lookup
    log.info("Indexing cross-asset ticks by slot time...")
    cross_by_slot = {}
    for crypto, df in cross_ticks.items():
        for _, row in markets[["market_id", "slot_ts"]].iterrows():
            slot_ts_val = row["slot_ts"]
            t0_ms = slot_ts_val * 1000
            t1_ms = (slot_ts_val + OBS_SECS) * 1000
            mask = (df["timestamp_ms"] >= t0_ms) & (df["timestamp_ms"] < t1_ms)
            subset = df[mask].copy()
            if len(subset) > 0:
                subset["t_sec"] = (subset["timestamp_ms"] / 1000) - slot_ts_val
                key = (crypto, slot_ts_val)
                cross_by_slot[key] = subset
    del cross_ticks
    gc.collect()
    log.info("  Cross-asset slots indexed: %d", len(cross_by_slot))

    # ── Build records ─────────────────────────────────────────────────────────
    records = []
    btc_grps = dict(list(btc_inslot.groupby("market_id")))

    for mid, target in target_map.items():
        btc_grp = btc_grps.get(mid)
        if btc_grp is None or len(btc_grp) < 5:
            continue

        slot_ts = slot_map[mid]
        dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        dow = dt.weekday()

        feat = {
            "market_id": mid,
            "slot_ts":   slot_ts,
            "target":    target,
            # Time
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin":  np.sin(2 * np.pi * dow / 7),
            "dow_cos":  np.cos(2 * np.pi * dow / 7),
        }

        # BTC tick features
        feat.update(compute_tick_features(btc_grp, "btc"))

        # BTC inslot spot return (from spot_price_usdt in ticks)
        spot_in = btc_grp["spot_price_usdt"].dropna()
        if len(spot_in) >= 2:
            p_open = float(spot_in.iloc[0])
            p_close = float(spot_in.iloc[-1])
            feat["btc_inslot_ret"] = (p_close - p_open) / (p_open + 1e-8)
            feat["btc_inslot_vol"] = float(spot_in.std() / (spot_in.mean() + 1e-8))
        else:
            feat["btc_inslot_ret"] = 0.0
            feat["btc_inslot_vol"] = 0.0

        # Pre-slot BTC spot returns from spot timeline
        for window_s, label in [(300, "5m"), (900, "15m"), (1800, "30m")]:
            t0_ms = (slot_ts - window_s) * 1000
            t1_ms = slot_ts * 1000
            idx = np.searchsorted(spot_ts, [t0_ms, t1_ms])
            seg = spot_px[idx[0]:idx[1]]
            if len(seg) >= 2:
                feat[f"btc_pre_{label}_ret"] = float((seg[-1] - seg[0]) / (seg[0] + 1e-8))
                feat[f"btc_pre_{label}_vol"] = float(np.std(seg) / (np.mean(seg) + 1e-8))
            else:
                feat[f"btc_pre_{label}_ret"] = 0.0
                feat[f"btc_pre_{label}_vol"] = 0.0

        # Cross-asset tick features (ETH, SOL)
        for crypto in ["ETH", "SOL"]:
            cross_grp = cross_by_slot.get((crypto, slot_ts))
            lbl = crypto.lower()
            feat.update(compute_tick_features(cross_grp if cross_grp is not None else pd.DataFrame(), lbl))

        # Orderbook features
        ob = ob_by_market.get(mid)
        feat.update(compute_orderbook_features(ob))

        records.append(feat)

    df_out = pd.DataFrame(records).sort_values("slot_ts").reset_index(drop=True)
    log.info("Dataset built: %d samples, %d features", len(df_out), len(df_out.columns) - 3)
    return df_out

# ── Step 6: Walk-forward eval ─────────────────────────────────────────────────
def walk_forward(df: pd.DataFrame, features: list, params: dict = None) -> dict:
    df = df.sort_values("slot_ts").reset_index(drop=True)
    n = len(df)
    n_splits = 5
    fold_size = n // (n_splits + 1)

    default_params = dict(
        objective="binary", class_weight="balanced",
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=-1,
    )
    p = {**default_params, **(params or {})}

    aucs, accs, briers = [], [], []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = min(train_end + fold_size, n)
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]
        if len(test) < 20:
            continue
        m = lgb.LGBMClassifier(**p)
        m.fit(train[features].fillna(0), train["target"])
        prob = m.predict_proba(test[features].fillna(0))[:, 1]
        y_te = test["target"]
        aucs.append(roc_auc_score(y_te, prob))
        accs.append(((prob >= 0.5) == y_te).mean())
        briers.append(brier_score_loss(y_te, prob))

    return {
        "wf_auc":   float(np.mean(aucs)),
        "wf_acc":   float(np.mean(accs)),
        "wf_brier": float(np.mean(briers)),
        "fold_aucs": aucs,
    }

# ── Step 7: Optuna HPO ────────────────────────────────────────────────────────
def optuna_search(df: pd.DataFrame, features: list) -> dict:
    split = int(len(df) * 0.75)
    train = df.iloc[:split]
    val = df.iloc[split:]
    X_tr = train[features].fillna(0)
    y_tr = train["target"]
    X_va = val[features].fillna(0)
    y_va = val["target"]

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            learning_rate=trial.suggest_float("lr", 0.005, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
            objective="binary", class_weight="balanced",
            verbose=-1, n_jobs=-1,
        )
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        return roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    log.info("Optuna best AUC: %.4f", study.best_value)
    return study.best_params

# ── Step 8: Train final + calibrate ──────────────────────────────────────────
def train_final(df: pd.DataFrame, features: list, best_params: dict):
    X = df[features].fillna(0)
    y = df["target"]
    params = {**best_params, "objective": "binary", "class_weight": "balanced",
              "verbose": -1, "n_jobs": -1}
    # Fix key name from optuna (lr → learning_rate)
    if "lr" in params:
        params["learning_rate"] = params.pop("lr")
    base = lgb.LGBMClassifier(**params)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=TimeSeriesSplit(n_splits=3))
    cal.fit(X, y)
    return cal

# ── Step 9: Promote to HF ─────────────────────────────────────────────────────
def promote(model_path: Path, bundle: dict):
    from huggingface_hub import HfApi
    import tempfile

    api = HfApi(token=HF_TOKEN)
    wf_auc = bundle["wf_auc"]

    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="champion.pkl",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message=f"Champion v5: AUC={wf_auc:.4f} | orderbook+cross-asset+balanced+hpo",
    )

    meta = {
        "version": "v5",
        "features": len(bundle["features"]),
        "wf_auc": wf_auc,
        "wf_acc": bundle["wf_acc"],
        "wf_brier": bundle["wf_brier"],
        "n_samples": bundle["n_samples"],
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": "v5: orderbook features + ETH/SOL cross-asset order flow + BTC inslot spot",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(meta, f, indent=2)
        tmp = f.name

    api.upload_file(
        path_or_fileobj=tmp,
        path_in_repo="champion_meta.json",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message=f"Champion v5 metadata AUC={wf_auc:.4f}",
    )
    import os; os.unlink(tmp)
    log.info("🏆 Champion v5 promoted to HF: https://huggingface.co/%s", HF_MODEL_REPO)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("BTC 5min Model v5 — HF Jobs Training")
    log.info("=" * 60)

    if not HF_TOKEN:
        log.error("HF_TOKEN not set!")
        sys.exit(1)

    # 1. Download data
    log.info("Step 1: Downloading data from HF...")
    download_hf_files()

    # 2. Markets
    log.info("Step 2: Loading markets...")
    markets = load_markets()

    # 3. Build dataset
    log.info("Step 3: Building dataset...")
    dataset_path = DATA_DIR / "btc_dataset_v5.parquet"
    if dataset_path.exists():
        log.info("  Loading cached dataset...")
        df = pd.read_parquet(dataset_path)
    else:
        df = build_dataset(markets)
        df.to_parquet(dataset_path, index=False)

    vc = df["target"].value_counts()
    log.info("Target balance: %s", dict(vc))

    NON_FEAT = {"market_id", "slot_ts", "target"}
    features = [c for c in df.columns if c not in NON_FEAT]
    log.info("Features: %d  Samples: %d", len(features), len(df))

    # Print feature groups
    ob_feats = [f for f in features if f.startswith("ob_")]
    eth_feats = [f for f in features if f.startswith("eth_")]
    sol_feats = [f for f in features if f.startswith("sol_")]
    btc_feats = [f for f in features if f.startswith("btc_")]
    log.info("  BTC: %d | ETH: %d | SOL: %d | Orderbook: %d | Other: %d",
             len(btc_feats), len(eth_feats), len(sol_feats), len(ob_feats),
             len(features) - len(btc_feats) - len(eth_feats) - len(sol_feats) - len(ob_feats))

    # 4. Walk-forward baseline
    log.info("Step 4: Walk-forward baseline...")
    wf_base = walk_forward(df, features)
    log.info("  Baseline WF AUC: %.4f  Acc: %.4f  Brier: %.4f",
             wf_base["wf_auc"], wf_base["wf_acc"], wf_base["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_base["fold_aucs"]])

    # 5. Optuna HPO
    log.info("Step 5: Optuna HPO (%d trials)...", OPTUNA_TRIALS)
    best_params = optuna_search(df, features)

    # 6. Walk-forward with best params
    log.info("Step 6: Walk-forward (optimized)...")
    wf_opt = walk_forward(df, features, best_params)
    log.info("  Optimized WF AUC: %.4f  Acc: %.4f  Brier: %.4f",
             wf_opt["wf_auc"], wf_opt["wf_acc"], wf_opt["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_opt["fold_aucs"]])

    # 7. Train final
    log.info("Step 7: Training final model with isotonic calibration...")
    final_model = train_final(df, features, best_params)

    # 8. Save
    model_path = DATA_DIR / "btc_model_v5.pkl"
    bundle = {
        "model": final_model,
        "features": features,
        "wf_auc": wf_opt["wf_auc"],
        "wf_acc": wf_opt["wf_acc"],
        "wf_brier": wf_opt["wf_brier"],
        "version": "v5",
        "n_samples": len(df),
        "best_params": best_params,
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Model saved: %s", model_path)

    # 9. Compare and promote
    log.info("=" * 60)
    log.info("RESULTS")
    log.info("  v4 champion WF AUC:    %.4f", CHAMPION_AUC)
    log.info("  v5 baseline WF AUC:    %.4f", wf_base["wf_auc"])
    log.info("  v5 optimized WF AUC:   %.4f", wf_opt["wf_auc"])

    if wf_opt["wf_auc"] > CHAMPION_AUC:
        log.info("✅ v5 beats champion! Promoting...")
        promote(model_path, bundle)
    else:
        log.warning("❌ v5 (%.4f) does NOT beat v4 champion (%.4f)", wf_opt["wf_auc"], CHAMPION_AUC)
        log.info("   Model saved at %s for analysis.", model_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
