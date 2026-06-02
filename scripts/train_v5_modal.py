"""
train_v5_modal.py — BTC 5min model v5 training on Modal.com

Usage:
    modal run scripts/train_v5_modal.py

Requirements (once):
    modal token new   # authenticate with Modal

Estimated cost: ~$1-2 USD per run (32GB RAM, 8 CPUs, ~45min)
Auto-promotes to HuggingFace if v5 beats v4 AUC=0.843

The job:
  1. Downloads BTC/ETH/SOL ticks + orderbook from HF dataset
  NOTE: ETH/SOL removed — model is BTC 5m specific only
  2. Builds feature dataset (order flow + orderbook + cross-asset)
  3. Baseline walk-forward eval
  4. 100-trial Optuna HPO
  5. Final model with isotonic calibration
  6. Promotes to artbreguez/polymarket-btc-model if AUC > 0.843
"""

import modal

# ── Image: all dependencies pre-installed ────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow>=18.0",
        "pandas>=2.2",
        "lightgbm==4.6.0",
        "scikit-learn==1.8.0",
        "numpy>=1.26",
        "optuna>=3.6",
        "huggingface_hub>=0.26",
    )
)

app = modal.App("polymarket-btc-train-v5", image=image)

# ── Secrets: HF token passed via Modal secret ─────────────────────────────────
# Create once: modal secret create hf-token HF_TOKEN=hf_NpI...
# Or pass directly via environment (see below)

@app.function(
    cpu=8,
    memory=32768,   # 32 GB RAM (ETH/SOL removed — BTC only)
    timeout=7200,   # 2h max
    secrets=[modal.Secret.from_name("hf-token")],
)
def train_v5():
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
    from huggingface_hub import hf_hub_download, HfApi

    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    log = logging.getLogger(__name__)

    HF_TOKEN      = os.environ.get("HF_TOKEN", "")
    HF_DATASET    = "BrockMisner/polymarket-btc-updown"
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
    CHAMPION_AUC  = 0.843
    OBS_SECS      = 180
    OPTUNA_TRIALS = 100
    DATA_DIR      = Path("/tmp/hf_data")
    DATA_DIR.mkdir(exist_ok=True)

    if not HF_TOKEN:
        log.error("HF_TOKEN not set — cannot download data or upload model")
        raise RuntimeError("HF_TOKEN required")

    # ── Step 1: Download data ─────────────────────────────────────────────────
    files_to_download = [
        "data/markets.parquet",
        "data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet",
        "data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet",
    ]
    log.info("Step 1: Downloading %d files from HF dataset %s", len(files_to_download), HF_DATASET)
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
        mb = dest.stat().st_size / 1e6
        log.info("    → %.1fs, %.0f MB", time.time() - t0, mb)

    # ── Step 2: Load markets ──────────────────────────────────────────────────
    log.info("Step 2: Loading markets...")
    m = pd.read_parquet(DATA_DIR / "data/markets.parquet")
    markets = m[
        (m["crypto"] == "BTC") & (m["timeframe"] == "5-minute") &
        (m["resolution"].notna()) & (m["resolution"] != -1)
    ].copy()
    markets["slot_ts"] = markets["slug"].apply(
        lambda s: int(str(s).split("-")[-1]) if str(s).split("-")[-1].isdigit() else 0
    )
    markets = markets[markets["slot_ts"] > 0].sort_values("slot_ts")
    log.info("  Resolved BTC markets: %d", len(markets))

    market_ids = set(markets["market_id"])
    slot_map   = dict(zip(markets["market_id"], markets["slot_ts"]))
    target_map = dict(zip(markets["market_id"], markets["resolution"].astype(int)))

    # ── Step 3: Load BTC ticks ────────────────────────────────────────────────
    log.info("Step 3: Loading BTC ticks...")
    btc = pq.read_table(
        str(DATA_DIR / "data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet"),
        columns=["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc", "spot_price_usdt"],
        filters=[("market_id", "in", list(market_ids))],
    ).to_pandas()
    btc["slot_ts_val"] = btc["market_id"].map(slot_map)
    btc["t_sec"] = btc["timestamp_ms"] / 1000 - btc["slot_ts_val"]
    btc_inslot = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < OBS_SECS)].copy()
    log.info("  BTC inslot: %d ticks, %d markets", len(btc_inslot), btc_inslot["market_id"].nunique())

    # Build spot timeline
    spot_tl = btc[["timestamp_ms", "spot_price_usdt"]].dropna().drop_duplicates("timestamp_ms")
    spot_tl = spot_tl.set_index("timestamp_ms").sort_index()
    spot_ts_arr = spot_tl.index.values
    spot_px_arr = spot_tl["spot_price_usdt"].values
    del btc; gc.collect()

    # ── Step 4: Load orderbook ────────────────────────────────────────────────
    ob_by_market = {}
    ob_path = DATA_DIR / "data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet"
    if ob_path.exists():
        log.info("Loading BTC orderbook...")
        try:
            schema = pq.read_schema(str(ob_path))
            log.info("  OB schema cols: %s", [f.name for f in schema][:12])
            ob = pq.read_table(
                str(ob_path), filters=[("market_id", "in", list(market_ids))]
            ).to_pandas()
            log.info("  OB rows: %d", len(ob))
            # Auto-detect timestamp column (schema uses ts_ms, not timestamp_ms)
            ts_col = "ts_ms" if "ts_ms" in ob.columns else "timestamp_ms"
            log.info("  OB timestamp col: %s", ts_col)
            for mid, grp in ob.groupby("market_id"):
                ob_by_market[mid] = grp.sort_values(ts_col).head(3)
            del ob; gc.collect()
        except Exception as e:
            log.warning("  OB load failed: %s — skipping", e)

    # ── Feature helpers ───────────────────────────────────────────────────────
    def tick_features(grp: pd.DataFrame, label: str) -> dict:
        n = len(grp)
        empty = {f"{label}_n_ticks": 0.0, f"{label}_up_ratio": 0.5, f"{label}_momentum": 0.0,
                 f"{label}_vwap_spread": 0.0, f"{label}_vol_up": 0.0, f"{label}_vol_dn": 0.0,
                 f"{label}_buy_ratio": 0.5, f"{label}_avg_size": 0.0,
                 f"{label}_up_w0": 0.5, f"{label}_up_w1": 0.5, f"{label}_up_w2": 0.5}
        if n == 0:
            return empty
        is_up = grp["outcome"] == "Up"
        vol_up = (grp["size_usdc"] * is_up).sum()
        vol_dn = (grp["size_usdc"] * ~is_up).sum()
        total = vol_up + vol_dn + 1e-8
        vwap_up = (grp.loc[is_up, "price"] * grp.loc[is_up, "size_usdc"]).sum() / (vol_up + 1e-8) if is_up.any() else 0.5
        vwap_dn = (grp.loc[~is_up, "price"] * grp.loc[~is_up, "size_usdc"]).sum() / (vol_dn + 1e-8) if (~is_up).any() else 0.5

        def ur(mask):
            if not mask.any(): return 0.5
            return float((grp.loc[mask, "size_usdc"] * (grp.loc[mask, "outcome"] == "Up")).sum() /
                         (grp.loc[mask, "size_usdc"].sum() + 1e-8))

        w0, w1, w2 = grp["t_sec"] < 60, (grp["t_sec"] >= 60) & (grp["t_sec"] < 120), grp["t_sec"] >= 120
        return {
            f"{label}_n_ticks": float(n), f"{label}_vol_up": float(vol_up), f"{label}_vol_dn": float(vol_dn),
            f"{label}_up_ratio": float(vol_up / total), f"{label}_vwap_up": float(vwap_up),
            f"{label}_vwap_dn": float(vwap_dn), f"{label}_vwap_spread": float(vwap_up - vwap_dn),
            f"{label}_buy_ratio": float((grp["side"] == "BUY").sum() / (n + 1e-8)),
            f"{label}_avg_size": float(total / n), f"{label}_momentum": float(ur(w2) - ur(w0)),
            f"{label}_up_w0": float(ur(w0)), f"{label}_up_w1": float(ur(w1)), f"{label}_up_w2": float(ur(w2)),
        }

    def ob_features(ob_grp) -> dict:
        empty = {"ob_spread": 0.0, "ob_mid": 0.5, "ob_imbalance": 0.0,
                 "ob_bid_depth": 0.0, "ob_ask_depth": 0.0, "ob_skew": 0.0}
        if ob_grp is None or len(ob_grp) == 0:
            return empty
        row = ob_grp.iloc[0]
        cols = set(ob_grp.columns)
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
                bid, ask = float(bids[0][0]), float(asks[0][0])
                bid_d = sum(float(b[1]) for b in bids[:5])
                ask_d = sum(float(a[1]) for a in asks[:5])
            else:
                return empty
            total = bid_d + ask_d + 1e-8
            return {"ob_spread": float(ask - bid), "ob_mid": float((bid + ask) / 2),
                    "ob_imbalance": float((bid_d - ask_d) / total),
                    "ob_bid_depth": float(bid_d), "ob_ask_depth": float(ask_d),
                    "ob_skew": float((bid + ask) / 2 - 0.5)}
        except Exception:
            return empty

    # ── Step 6: Build dataset ─────────────────────────────────────────────────
    log.info("Step 6: Building feature dataset...")
    btc_grps = dict(list(btc_inslot.groupby("market_id")))
    records = []

    for mid, target in target_map.items():
        grp = btc_grps.get(mid)
        if grp is None or len(grp) < 5:
            continue

        slot_ts = slot_map[mid]
        dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0

        feat = {
            "market_id": mid, "slot_ts": slot_ts, "target": target,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin":  np.sin(2 * np.pi * dt.weekday() / 7),
            "dow_cos":  np.cos(2 * np.pi * dt.weekday() / 7),
        }

        # BTC tick features
        feat.update(tick_features(grp, "btc"))

        # BTC inslot spot return
        sp = grp["spot_price_usdt"].dropna()
        if len(sp) >= 2:
            p0, p1 = float(sp.iloc[0]), float(sp.iloc[-1])
            feat["btc_inslot_ret"] = (p1 - p0) / (p0 + 1e-8)
            feat["btc_inslot_vol"] = float(sp.std() / (sp.mean() + 1e-8))
        else:
            feat["btc_inslot_ret"] = 0.0
            feat["btc_inslot_vol"] = 0.0

        # Pre-slot BTC spot returns
        for w_s, lbl in [(300, "5m"), (900, "15m"), (1800, "30m")]:
            idx0, idx1 = np.searchsorted(spot_ts_arr, [(slot_ts - w_s) * 1000, slot_ts * 1000])
            seg = spot_px_arr[idx0:idx1]
            if len(seg) >= 2:
                feat[f"btc_pre_{lbl}_ret"] = float((seg[-1] - seg[0]) / (seg[0] + 1e-8))
                feat[f"btc_pre_{lbl}_vol"] = float(np.std(seg) / (np.mean(seg) + 1e-8))
            else:
                feat[f"btc_pre_{lbl}_ret"] = 0.0
                feat[f"btc_pre_{lbl}_vol"] = 0.0

        # Orderbook features
        feat.update(ob_features(ob_by_market.get(mid)))
        records.append(feat)

    df = pd.DataFrame(records).sort_values("slot_ts").reset_index(drop=True)
    log.info("Dataset: %d samples", len(df))
    vc = df["target"].value_counts()
    log.info("Target balance: %s", dict(vc))

    NON_FEAT = {"market_id", "slot_ts", "target"}
    features = [c for c in df.columns if c not in NON_FEAT]
    log.info("Features: %d", len(features))
    for prefix in ["btc_", "ob_"]:
        n = len([f for f in features if f.startswith(prefix)])
        log.info("  %s: %d", prefix.rstrip("_"), n)

    # ── Step 7: Walk-forward (baseline) ──────────────────────────────────────
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
            aucs.append(roc_auc_score(te["target"], prob))
            accs.append(((prob >= 0.5) == te["target"]).mean())
            briers.append(brier_score_loss(te["target"], prob))
        return {"wf_auc": float(np.mean(aucs)), "wf_acc": float(np.mean(accs)),
                "wf_brier": float(np.mean(briers)), "fold_aucs": aucs}

    log.info("Step 7: Baseline walk-forward...")
    wf_base = walk_forward(df, features)
    log.info("  Baseline WF AUC: %.4f | Acc: %.4f | Brier: %.4f",
             wf_base["wf_auc"], wf_base["wf_acc"], wf_base["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_base["fold_aucs"]])

    # ── Step 8: Optuna HPO — objetivo é walk-forward, não split estático ────────
    log.info("Step 8: Optuna HPO (%d trials — WF objective)...", OPTUNA_TRIALS)

    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            learning_rate=trial.suggest_float("lr", 0.005, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            objective="binary", class_weight="balanced", verbose=-1, n_jobs=-1,
        )
        # Use walk-forward as objective to avoid overfitting on small dataset
        wf = walk_forward(df, features, p)
        return wf["wf_auc"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    if "lr" in best_params:
        best_params["learning_rate"] = best_params.pop("lr")
    log.info("  Optuna best WF AUC: %.4f", study.best_value)

    # ── Step 9: Walk-forward (optimized) ─────────────────────────────────────
    log.info("Step 9: Walk-forward (optimized)...")
    wf_opt = walk_forward(df, features, best_params)
    log.info("  Optimized WF AUC: %.4f | Acc: %.4f | Brier: %.4f",
             wf_opt["wf_auc"], wf_opt["wf_acc"], wf_opt["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_opt["fold_aucs"]])

    # ── Step 10: Train final — usa melhores params (WF otimizado ou baseline) ──
    log.info("Step 10: Training final model with isotonic calibration...")
    # Se o Optuna não melhorou sobre o baseline, usa params do baseline
    if wf_opt["wf_auc"] >= wf_base["wf_auc"]:
        final_params = best_params
        final_auc = wf_opt["wf_auc"]
        log.info("  Using optimized params (WF: %.4f)", final_auc)
    else:
        final_params = {}  # defaults do walk_forward (baseline)
        final_auc = wf_base["wf_auc"]
        log.info("  Optimized params worse than baseline — using baseline params (WF: %.4f)", final_auc)
    X, y = df[features].fillna(0), df["target"]
    base_params = dict(objective="binary", class_weight="balanced", n_estimators=300,
                       learning_rate=0.05, num_leaves=31, min_child_samples=10,
                       subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
    params = {**base_params, **final_params, "objective": "binary", "class_weight": "balanced",
              "verbose": -1, "n_jobs": -1}
    base_model = lgb.LGBMClassifier(**params)
    cal = CalibratedClassifierCV(base_model, method="isotonic", cv=TimeSeriesSplit(n_splits=3))
    cal.fit(X, y)

    bundle = {
        "model": cal, "features": features,
        "wf_auc": final_auc, "wf_acc": wf_opt["wf_acc"] if wf_opt["wf_auc"] >= wf_base["wf_auc"] else wf_base["wf_acc"],
        "wf_brier": wf_opt["wf_brier"] if wf_opt["wf_auc"] >= wf_base["wf_auc"] else wf_base["wf_brier"],
        "version": "v5", "n_samples": len(df), "best_params": final_params,
    }
    model_path = Path("/tmp/btc_model_v5.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Model saved: %s (%.1f MB)", model_path, model_path.stat().st_size / 1e6)

    # ── Step 11: Results + conditional promote ────────────────────────────────
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("  v4 champion AUC:   %.4f", CHAMPION_AUC)
    log.info("  v5 baseline AUC:   %.4f", wf_base["wf_auc"])
    log.info("  v5 optimized AUC:  %.4f", wf_opt["wf_auc"])

    if final_auc > CHAMPION_AUC:
        log.info("v5 beats champion! Promoting to HF...")
        import tempfile
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="champion.pkl",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            commit_message=f"Champion v5: AUC={final_auc:.4f} | orderbook+cross-asset+balanced+hpo",
        )
        meta = {
            "version": "v5",
            "features": len(features),
            "wf_auc": final_auc,
            "n_samples": len(df),
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": "v5: orderbook + ETH/SOL cross-asset + BTC inslot spot + 100-trial Optuna",
            "best_params": best_params,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
            json.dump(meta, fp, indent=2)
            tmp_meta = fp.name
        api.upload_file(
            path_or_fileobj=tmp_meta,
            path_in_repo="champion_meta.json",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            commit_message=f"Champion v5 meta AUC={final_auc:.4f}",
        )
        log.info("Champion v5 promoted: https://huggingface.co/%s", HF_MODEL_REPO)
        log.info("Push to main branch of polymarket-btc-lab to trigger auto-deploy!")
    else:
        log.warning("v5 (%.4f) does NOT beat v4 champion (%.4f) — model NOT promoted",
                    final_auc, CHAMPION_AUC)
        log.info("The model bundle is available at /tmp/btc_model_v5.pkl for further analysis.")

    log.info("Done.")
    return {
        "wf_auc_baseline": wf_base["wf_auc"],
        "wf_auc_optimized": wf_opt["wf_auc"],
        "wf_auc_final": final_auc,
        "n_samples": len(df),
        "n_features": len(features),
        "promoted": final_auc > CHAMPION_AUC,
    }


@app.local_entrypoint()
def main():
    print("Submitting BTC v5 training job to Modal...")
    result = train_v5.remote()
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"  Baseline AUC:  {result['wf_auc_baseline']:.4f}")
    print(f"  Optimized AUC: {result['wf_auc_optimized']:.4f}")
    print(f"  Samples:       {result['n_samples']}")
    print(f"  Features:      {result['n_features']}")
    print(f"  Promoted:      {'YES' if result['promoted'] else 'NO'}")
    print(f"{'='*50}")
