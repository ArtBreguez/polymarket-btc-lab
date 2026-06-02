#!/usr/bin/env python3
"""
train_v5_local.py — BTC 5min model v5 (memory-efficient for 8GB RAM)

New vs v4:
  - Orderbook features (filtered to 616 markets — manageable)
  - Cross-asset tick features: ETH/SOL order flow (timestamp-based lookup)
  - BTC spot return from spot_price_usdt already in ticks
  - 60 Optuna trials
  - class_weight=balanced + isotonic calibration
"""

import gc
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone
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

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger(__name__)

ROOT      = Path("/home/ubuntu/polymarket-btc-lab")
DATA      = ROOT / "data" / "data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

MARKETS_PATH   = Path("/home/ubuntu/polymarket-btc-bot/data/raw/markets.parquet")
BTC_TICKS_PATH = ROOT / "data" / "ticks_btc_5min.parquet"
ETH_TICKS_PATH = DATA / "ticks/crypto=ETH/timeframe=5-minute/part-0.parquet"
SOL_TICKS_PATH = DATA / "ticks/crypto=SOL/timeframe=5-minute/part-0.parquet"
OB_PATH        = DATA / "orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet"
DATASET_OUT    = ARTIFACTS / "btc_dataset_v5.parquet"
MODEL_OUT      = ARTIFACTS / "btc_model_v5.pkl"

HF_TOKEN      = os.environ.get("HF_TOKEN", "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")
HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
CHAMPION_AUC  = 0.843
OBS_SECS      = 180
OPTUNA_TRIALS = 60


def load_markets() -> pd.DataFrame:
    m = pd.read_parquet(MARKETS_PATH)
    btc = m[(m["crypto"] == "BTC") & (m["timeframe"] == "5-minute") &
            (m["resolution"].notna()) & (m["resolution"] != -1)].copy()
    btc["slot_ts"] = btc["slug"].apply(
        lambda s: int(str(s).split("-")[-1]) if str(s).split("-")[-1].isdigit() else 0)
    return btc[btc["slot_ts"] > 0].sort_values("slot_ts")


def tick_features(grp: pd.DataFrame, label: str) -> dict:
    n = len(grp)
    if n == 0:
        return {f"{label}_{k}": v for k, v in {
            "n_ticks": 0.0, "up_ratio": 0.5, "momentum": 0.0,
            "vwap_spread": 0.0, "vol_up": 0.0, "vol_dn": 0.0,
            "buy_ratio": 0.5, "avg_size": 0.0,
            "up_w0": 0.5, "up_w1": 0.5, "up_w2": 0.5,
        }.items()}
    is_up = grp["outcome"] == "Up"
    vol_up = (grp["size_usdc"] * is_up).sum()
    vol_dn = (grp["size_usdc"] * ~is_up).sum()
    total = vol_up + vol_dn + 1e-8
    vwap_up = (grp.loc[is_up, "price"] * grp.loc[is_up, "size_usdc"]).sum() / (vol_up + 1e-8) if is_up.any() else 0.5
    vwap_dn = (grp.loc[~is_up, "price"] * grp.loc[~is_up, "size_usdc"]).sum() / (vol_dn + 1e-8) if (~is_up).any() else 0.5

    def ur(mask):
        if not mask.any(): return 0.5
        vu = (grp.loc[mask, "size_usdc"] * (grp.loc[mask, "outcome"] == "Up")).sum()
        return float(vu / (grp.loc[mask, "size_usdc"].sum() + 1e-8))

    w0 = grp["t_sec"] < 60
    w1 = (grp["t_sec"] >= 60) & (grp["t_sec"] < 120)
    w2 = grp["t_sec"] >= 120
    up_w0, up_w1, up_w2 = ur(w0), ur(w1), ur(w2)
    return {
        f"{label}_n_ticks":     float(n),
        f"{label}_vol_up":      float(vol_up),
        f"{label}_vol_dn":      float(vol_dn),
        f"{label}_up_ratio":    float(vol_up / total),
        f"{label}_vwap_up":     float(vwap_up),
        f"{label}_vwap_dn":     float(vwap_dn),
        f"{label}_vwap_spread": float(vwap_up - vwap_dn),
        f"{label}_buy_ratio":   float((grp["side"] == "BUY").sum() / (n + 1e-8)),
        f"{label}_avg_size":    float(total / n),
        f"{label}_momentum":    float(up_w2 - up_w0),
        f"{label}_up_w0":       float(up_w0),
        f"{label}_up_w1":       float(up_w1),
        f"{label}_up_w2":       float(up_w2),
    }


def orderbook_features(ob_grp) -> dict:
    empty = {"ob_spread": 0.0, "ob_mid": 0.5, "ob_imbalance": 0.0,
             "ob_bid_depth": 0.0, "ob_ask_depth": 0.0, "ob_skew": 0.0}
    if ob_grp is None or len(ob_grp) == 0:
        return empty
    cols = set(ob_grp.columns)
    row = ob_grp.iloc[0]
    try:
        if "best_bid" in cols:
            bid = float(row.get("best_bid") or 0)
            ask = float(row.get("best_ask") or 1)
            bid_d = float(row.get("bid_size_5") or row.get("bid_depth") or 0)
            ask_d = float(row.get("ask_size_5") or row.get("ask_depth") or 0)
        elif "bids" in cols:
            bids = json.loads(row["bids"]) if isinstance(row["bids"], str) else (row["bids"] or [])
            asks = json.loads(row["asks"]) if isinstance(row["asks"], str) else (row["asks"] or [])
            if not bids or not asks: return empty
            bid = float(bids[0][0]); ask = float(asks[0][0])
            bid_d = sum(float(b[1]) for b in bids[:5])
            ask_d = sum(float(a[1]) for a in asks[:5])
        else:
            return empty
        spread = ask - bid
        mid = (bid + ask) / 2
        total = bid_d + ask_d + 1e-8
        return {
            "ob_spread":     float(spread),
            "ob_mid":        float(mid),
            "ob_imbalance":  float((bid_d - ask_d) / total),
            "ob_bid_depth":  float(bid_d),
            "ob_ask_depth":  float(ask_d),
            "ob_skew":       float(mid - 0.5),
        }
    except Exception:
        return empty


def build_dataset(markets: pd.DataFrame) -> pd.DataFrame:
    market_ids = set(markets["market_id"])
    slot_map   = dict(zip(markets["market_id"], markets["slot_ts"]))
    target_map = dict(zip(markets["market_id"], markets["resolution"].astype(int)))

    # ── BTC ticks ─────────────────────────────────────────────────────────────
    log.info("Loading BTC ticks (filtered)...")
    btc = pq.read_table(BTC_TICKS_PATH,
        columns=["market_id","timestamp_ms","outcome","side","price","size_usdc","spot_price_usdt"],
        filters=[("market_id","in",list(market_ids))]).to_pandas()
    btc["slot_ts"] = btc["market_id"].map(slot_map)
    btc["t_sec"] = btc["timestamp_ms"] / 1000 - btc["slot_ts"]
    btc_inslot = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < OBS_SECS)].copy()
    log.info("  BTC inslot: %d ticks, %d markets", len(btc_inslot), btc_inslot["market_id"].nunique())

    # Build spot timeline from BTC ticks (all, not just inslot)
    spot_tl = btc[["timestamp_ms","spot_price_usdt"]].dropna().drop_duplicates("timestamp_ms")
    spot_tl = spot_tl.set_index("timestamp_ms").sort_index()
    spot_ts_arr = spot_tl.index.values
    spot_px_arr = spot_tl["spot_price_usdt"].values
    del btc; gc.collect()

    # ── ETH ticks (load all, process by slot timestamp) ─────────────────────
    log.info("Loading ETH ticks (timestamp-based)...")
    eth = None
    if ETH_TICKS_PATH.exists():
        eth = pq.read_table(ETH_TICKS_PATH,
            columns=["timestamp_ms","outcome","side","price","size_usdc"]).to_pandas()
        eth = eth.sort_values("timestamp_ms").reset_index(drop=True)
        log.info("  ETH: %d rows", len(eth))

    # ── SOL ticks ─────────────────────────────────────────────────────────────
    log.info("Loading SOL ticks (timestamp-based)...")
    sol = None
    if SOL_TICKS_PATH.exists():
        sol = pq.read_table(SOL_TICKS_PATH,
            columns=["timestamp_ms","outcome","side","price","size_usdc"]).to_pandas()
        sol = sol.sort_values("timestamp_ms").reset_index(drop=True)
        log.info("  SOL: %d rows", len(sol))

    # ── Orderbook (filtered to our markets) ───────────────────────────────────
    ob_by_market = {}
    if OB_PATH.exists():
        log.info("Loading BTC orderbook (filtered)...")
        ob_schema = pq.read_schema(str(OB_PATH))
        log.info("  OB schema: %s", [f.name for f in ob_schema][:12])
        try:
            ob = pq.read_table(OB_PATH,
                filters=[("market_id","in",list(market_ids))]).to_pandas()
            log.info("  OB rows: %d", len(ob))
            for mid, grp in ob.groupby("market_id"):
                ob_by_market[mid] = grp.sort_values("timestamp_ms").head(3)
            del ob; gc.collect()
        except Exception as e:
            log.warning("  OB load failed: %s — skipping orderbook", e)

    # ── Build records ─────────────────────────────────────────────────────────
    btc_grps = dict(list(btc_inslot.groupby("market_id")))
    records = []

    for mid, target in target_map.items():
        grp = btc_grps.get(mid)
        if grp is None or len(grp) < 5:
            continue

        slot_ts = slot_map[mid]
        dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        dow = dt.weekday()

        feat = {
            "market_id": mid, "slot_ts": slot_ts, "target": target,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin":  np.sin(2 * np.pi * dow / 7),
            "dow_cos":  np.cos(2 * np.pi * dow / 7),
        }

        # BTC tick features
        feat.update(tick_features(grp, "btc"))

        # BTC inslot spot return
        spot_in = grp["spot_price_usdt"].dropna()
        if len(spot_in) >= 2:
            p0, p1 = float(spot_in.iloc[0]), float(spot_in.iloc[-1])
            feat["btc_inslot_ret"] = (p1 - p0) / (p0 + 1e-8)
            feat["btc_inslot_vol"] = float(spot_in.std() / (spot_in.mean() + 1e-8))
        else:
            feat["btc_inslot_ret"] = 0.0
            feat["btc_inslot_vol"] = 0.0

        # Pre-slot BTC spot returns
        for w_s, lbl in [(300, "5m"), (900, "15m"), (1800, "30m")]:
            t0_ms = (slot_ts - w_s) * 1000
            t1_ms = slot_ts * 1000
            idx0, idx1 = np.searchsorted(spot_ts_arr, [t0_ms, t1_ms])
            seg = spot_px_arr[idx0:idx1]
            if len(seg) >= 2:
                feat[f"btc_pre_{lbl}_ret"] = float((seg[-1] - seg[0]) / (seg[0] + 1e-8))
                feat[f"btc_pre_{lbl}_vol"] = float(np.std(seg) / (np.mean(seg) + 1e-8))
            else:
                feat[f"btc_pre_{lbl}_ret"] = 0.0
                feat[f"btc_pre_{lbl}_vol"] = 0.0

        # ETH cross-asset tick features (timestamp window)
        for crypto_df, lbl in [(eth, "eth"), (sol, "sol")]:
            if crypto_df is not None:
                t0_ms = slot_ts * 1000
                t1_ms = (slot_ts + OBS_SECS) * 1000
                idx0 = np.searchsorted(crypto_df["timestamp_ms"].values, t0_ms)
                idx1 = np.searchsorted(crypto_df["timestamp_ms"].values, t1_ms)
                sub = crypto_df.iloc[idx0:idx1].copy()
                if len(sub) > 0:
                    sub["t_sec"] = sub["timestamp_ms"] / 1000 - slot_ts
                    feat.update(tick_features(sub, lbl))
                else:
                    feat.update(tick_features(pd.DataFrame(), lbl))
            else:
                feat.update(tick_features(pd.DataFrame(), lbl))

        # Orderbook features
        feat.update(orderbook_features(ob_by_market.get(mid)))

        records.append(feat)

    return pd.DataFrame(records).sort_values("slot_ts").reset_index(drop=True)


def walk_forward(df, features, params=None):
    df = df.sort_values("slot_ts").reset_index(drop=True)
    n, n_splits = len(df), 5
    fold_size = n // (n_splits + 1)
    base = dict(objective="binary", class_weight="balanced", n_estimators=300,
                learning_rate=0.05, num_leaves=31, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
    p = {**base, **(params or {})}
    aucs, accs, briers = [], [], []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = min(train_end + fold_size, n)
        tr, te = df.iloc[:train_end], df.iloc[train_end:test_end]
        if len(te) < 20: continue
        m = lgb.LGBMClassifier(**p)
        m.fit(tr[features].fillna(0), tr["target"])
        prob = m.predict_proba(te[features].fillna(0))[:, 1]
        y = te["target"]
        aucs.append(roc_auc_score(y, prob))
        accs.append(((prob >= 0.5) == y).mean())
        briers.append(brier_score_loss(y, prob))
    return {"wf_auc": float(np.mean(aucs)), "wf_acc": float(np.mean(accs)),
            "wf_brier": float(np.mean(briers)), "fold_aucs": aucs}


def optuna_search(df, features):
    split = int(len(df) * 0.75)
    tr, va = df.iloc[:split], df.iloc[split:]
    Xtr, ytr = tr[features].fillna(0), tr["target"]
    Xva, yva = va[features].fillna(0), va["target"]
    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            learning_rate=trial.suggest_float("lr", 0.005, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            objective="binary", class_weight="balanced", verbose=-1, n_jobs=-1,
        )
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr)
        return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    log.info("Optuna best AUC: %.4f", study.best_value)
    return study.best_params


def main():
    log.info("=" * 60)
    log.info("BTC v5 Training (local, memory-efficient)")
    log.info("=" * 60)

    markets = load_markets()
    log.info("Markets: %d", len(markets))

    if DATASET_OUT.exists():
        log.info("Loading cached dataset...")
        df = pd.read_parquet(DATASET_OUT)
    else:
        log.info("Building dataset...")
        df = build_dataset(markets)
        df.to_parquet(DATASET_OUT, index=False)
        log.info("Saved: %s", DATASET_OUT)

    vc = df["target"].value_counts()
    log.info("Balance: %s", dict(vc))
    NON_FEAT = {"market_id", "slot_ts", "target"}
    features = [c for c in df.columns if c not in NON_FEAT]
    log.info("Features: %d  Samples: %d", len(features), len(df))

    ob_feats  = [f for f in features if f.startswith("ob_")]
    eth_feats = [f for f in features if f.startswith("eth_")]
    sol_feats = [f for f in features if f.startswith("sol_")]
    log.info("  BTC: %d | ETH: %d | SOL: %d | Orderbook: %d",
             len([f for f in features if f.startswith("btc_")]),
             len(eth_feats), len(sol_feats), len(ob_feats))

    # Baseline WF
    log.info("Baseline walk-forward...")
    wf_base = walk_forward(df, features)
    log.info("  WF AUC: %.4f  Acc: %.4f  Brier: %.4f",
             wf_base["wf_auc"], wf_base["wf_acc"], wf_base["wf_brier"])
    log.info("  Folds: %s", [f"{x:.3f}" for x in wf_base["fold_aucs"]])

    # Optuna
    log.info("Optuna HPO (%d trials)...", OPTUNA_TRIALS)
    best = optuna_search(df, features)

    # Fix key name
    if "lr" in best:
        best["learning_rate"] = best.pop("lr")

    # Optimized WF
    log.info("Optimized walk-forward...")
    wf_opt = walk_forward(df, features, best)
    log.info("  WF AUC: %.4f  Acc: %.4f  Brier: %.4f",
             wf_opt["wf_auc"], wf_opt["wf_acc"], wf_opt["wf_brier"])
    log.info("  Folds: %s", [f"{x:.3f}" for x in wf_opt["fold_aucs"]])

    # Train final with calibration
    log.info("Training final model...")
    X = df[features].fillna(0)
    y = df["target"]
    params = {**best, "objective": "binary", "class_weight": "balanced",
              "verbose": -1, "n_jobs": -1}
    base_model = lgb.LGBMClassifier(**params)
    cal = CalibratedClassifierCV(base_model, method="isotonic", cv=TimeSeriesSplit(n_splits=3))
    cal.fit(X, y)

    bundle = {
        "model": cal, "features": features,
        "wf_auc": wf_opt["wf_auc"], "wf_acc": wf_opt["wf_acc"],
        "wf_brier": wf_opt["wf_brier"], "version": "v5",
        "n_samples": len(df), "best_params": best,
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Model saved: %s", MODEL_OUT)

    log.info("=" * 60)
    log.info("RESULTS")
    log.info("  v4 champion:   %.4f", CHAMPION_AUC)
    log.info("  v5 baseline:   %.4f", wf_base["wf_auc"])
    log.info("  v5 optimized:  %.4f", wf_opt["wf_auc"])

    if wf_opt["wf_auc"] > CHAMPION_AUC:
        log.info("✅ v5 beats champion! Running promote_champion.py...")
        import subprocess
        r = subprocess.run([
            sys.executable,
            str(ROOT / "scripts/promote_champion.py"),
            "--model", str(MODEL_OUT),
            "--notes", f"v5 orderbook+cross-asset+spot AUC={wf_opt['wf_auc']:.4f}",
            "--hf-token", HF_TOKEN,
        ])
        if r.returncode == 0:
            log.info("🏆 Champion promoted! Push to main to auto-deploy.")
        else:
            log.error("Promote failed")
    else:
        log.warning("❌ v5 (%.4f) does NOT beat v4 (%.4f) — saved locally",
                    wf_opt["wf_auc"], CHAMPION_AUC)
    log.info("Done.")


if __name__ == "__main__":
    main()
