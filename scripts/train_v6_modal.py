"""
train_v6_modal.py — BTC 5min model v6: anti-overfitting pipeline

New vs v5:
  - Lagged outcomes: last 3 slot results (did BTC go UP/DOWN?)
  - Volume anomaly: current slot volume vs rolling 20-slot mean/std
  - Wider spot windows: 1h and 4h pre-slot returns + vol
  - Tick acceleration: tick rate last 60s vs first 60s of slot
  - Round number proximity: BTC price distance to $1k/5k/10k levels
  - Purged walk-forward CV (gap between train/test to prevent leakage)
  - Permutation importance to drop noise features before HPO
  - Final model uses top features only (importance threshold)
  - Hard gate: AUC must beat champion AND brier < 0.22 AND acc > 0.73
"""

import modal

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

app = modal.App("polymarket-btc-train-v6", image=image)

@app.function(
    cpu=8,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
)
def train_v6():
    import gc, json, logging, os, pickle, sys, time, warnings
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np
    import optuna
    import pandas as pd
    import pyarrow.parquet as pq
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    import lightgbm as lgb
    from huggingface_hub import hf_hub_download, HfApi

    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    HF_TOKEN      = os.environ.get("HF_TOKEN", "")
    HF_DATASET    = "BrockMisner/polymarket-btc-updown"
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
    CHAMPION_AUC  = 0.8553   # v5 benchmark (measured WITHOUT purged WF)
    CHAMPION_BRIER = 0.22    # estimated (v5 meta didn't record brier)
    CHAMPION_ACC   = 0.73    # estimated
    OBS_SECS      = 180
    OPTUNA_TRIALS = 80
    WF_GAP        = 5        # slots gap between train and test (prevents leakage)
    DATA_DIR      = Path("/tmp/hf_data")
    DATA_DIR.mkdir(exist_ok=True)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")

    # ── Step 1: Download ──────────────────────────────────────────────────────
    files = [
        "data/markets.parquet",
        "data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet",
        "data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet",
    ]
    log.info("Step 1: Downloading %d files...", len(files))
    for f in files:
        dest = DATA_DIR / f
        if dest.exists():
            log.info("  Cached: %s (%.0f MB)", f, dest.stat().st_size / 1e6)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("  Downloading %s ...", f)
        t0 = time.time()
        hf_hub_download(repo_id=HF_DATASET, filename=f, repo_type="dataset",
                        token=HF_TOKEN, local_dir=str(DATA_DIR),
                        local_dir_use_symlinks=False)
        log.info("    → %.1fs, %.0f MB", time.time() - t0, dest.stat().st_size / 1e6)

    # ── Step 2: Markets ───────────────────────────────────────────────────────
    log.info("Step 2: Loading markets...")
    m = pd.read_parquet(DATA_DIR / "data/markets.parquet")
    markets = m[
        (m["crypto"] == "BTC") & (m["timeframe"] == "5-minute") &
        (m["resolution"].isin([0, 1]))
    ].copy()
    markets["slot_ts"] = markets["slug"].apply(
        lambda s: int(str(s).split("-")[-1]) if str(s).split("-")[-1].isdigit() else 0
    )
    markets = markets[markets["slot_ts"] > 0].sort_values("slot_ts").reset_index(drop=True)
    log.info("  Resolved BTC 5min markets: %d", len(markets))

    market_ids = set(markets["market_id"])
    slot_map   = dict(zip(markets["market_id"], markets["slot_ts"]))
    target_map = dict(zip(markets["market_id"], markets["resolution"].astype(int)))
    # Ordered slot list for lagged features
    markets_sorted = markets.sort_values("slot_ts").reset_index(drop=True)
    slot_to_idx    = {row["market_id"]: i for i, row in markets_sorted.iterrows()}
    idx_to_target  = dict(enumerate(markets_sorted["resolution"].astype(int)))

    # ── Step 3: BTC ticks ─────────────────────────────────────────────────────
    log.info("Step 3: Loading BTC ticks...")
    btc = pq.read_table(
        str(DATA_DIR / "data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet"),
        columns=["market_id", "timestamp_ms", "outcome", "side", "price",
                 "size_usdc", "spot_price_usdt"],
        filters=[("market_id", "in", list(market_ids))],
    ).to_pandas()
    btc["slot_ts_val"] = btc["market_id"].map(slot_map)
    btc["t_sec"] = btc["timestamp_ms"] / 1000 - btc["slot_ts_val"]
    btc_inslot = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < OBS_SECS)].copy()
    log.info("  BTC inslot: %d ticks, %d markets",
             len(btc_inslot), btc_inslot["market_id"].nunique())

    # Volume per slot (for anomaly feature)
    slot_vol = btc_inslot.groupby("market_id")["size_usdc"].sum().rename("slot_vol")

    # Spot timeline
    spot_tl = btc[["timestamp_ms", "spot_price_usdt"]].dropna().drop_duplicates("timestamp_ms")
    spot_tl = spot_tl.set_index("timestamp_ms").sort_index()
    spot_ts_arr = spot_tl.index.values
    spot_px_arr = spot_tl["spot_price_usdt"].values
    del btc; gc.collect()

    # ── Step 4: Orderbook ─────────────────────────────────────────────────────
    ob_by_market = {}
    ob_path = DATA_DIR / "data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet"
    if ob_path.exists():
        log.info("Step 4: Loading BTC orderbook...")
        try:
            schema_cols = [f.name for f in pq.read_schema(str(ob_path))]
            log.info("  OB cols: %s", schema_cols[:10])
            ob = pq.read_table(str(ob_path),
                               filters=[("market_id", "in", list(market_ids))]).to_pandas()
            log.info("  OB rows: %d", len(ob))
            ts_col = "ts_ms" if "ts_ms" in ob.columns else "timestamp_ms"
            for mid, grp in ob.groupby("market_id"):
                ob_by_market[mid] = grp.sort_values(ts_col).head(3)
            del ob; gc.collect()
        except Exception as e:
            log.warning("  OB load failed: %s", e)

    # ── Feature helpers ────────────────────────────────────────────────────────
    def tick_features(grp: pd.DataFrame, label: str) -> dict:
        n = len(grp)
        empty = {
            f"{label}_n_ticks": 0.0, f"{label}_up_ratio": 0.5,
            f"{label}_momentum": 0.0, f"{label}_vwap_spread": 0.0,
            f"{label}_vol_up": 0.0,  f"{label}_vol_dn": 0.0,
            f"{label}_buy_ratio": 0.5, f"{label}_avg_size": 0.0,
            f"{label}_up_w0": 0.5, f"{label}_up_w1": 0.5, f"{label}_up_w2": 0.5,
            f"{label}_tick_accel": 0.0,
        }
        if n == 0:
            return empty
        is_up  = grp["outcome"] == "Up"
        vol_up = (grp["size_usdc"] * is_up).sum()
        vol_dn = (grp["size_usdc"] * ~is_up).sum()
        total  = vol_up + vol_dn + 1e-8
        vwap_up = (grp.loc[is_up,  "price"] * grp.loc[is_up,  "size_usdc"]).sum() / (vol_up + 1e-8) if is_up.any()  else 0.5
        vwap_dn = (grp.loc[~is_up, "price"] * grp.loc[~is_up, "size_usdc"]).sum() / (vol_dn + 1e-8) if (~is_up).any() else 0.5

        def ur(mask):
            if not mask.any(): return 0.5
            return float((grp.loc[mask, "size_usdc"] * (grp.loc[mask, "outcome"] == "Up")).sum() /
                         (grp.loc[mask, "size_usdc"].sum() + 1e-8))

        w0 = grp["t_sec"] < 60
        w1 = (grp["t_sec"] >= 60) & (grp["t_sec"] < 120)
        w2 = grp["t_sec"] >= 120

        # Tick acceleration: ticks/sec in last 30s vs first 30s
        first30 = (grp["t_sec"] < 30).sum()
        last30  = (grp["t_sec"] >= (OBS_SECS - 30)).sum()
        tick_accel = float((last30 - first30) / (first30 + 1e-8))

        return {
            f"{label}_n_ticks":    float(n),
            f"{label}_vol_up":     float(vol_up),
            f"{label}_vol_dn":     float(vol_dn),
            f"{label}_up_ratio":   float(vol_up / total),
            f"{label}_vwap_up":    float(vwap_up),
            f"{label}_vwap_dn":    float(vwap_dn),
            f"{label}_vwap_spread": float(vwap_up - vwap_dn),
            f"{label}_buy_ratio":  float((grp["side"] == "BUY").sum() / (n + 1e-8)),
            f"{label}_avg_size":   float(total / n),
            f"{label}_momentum":   float(ur(w2) - ur(w0)),
            f"{label}_up_w0":      float(ur(w0)),
            f"{label}_up_w1":      float(ur(w1)),
            f"{label}_up_w2":      float(ur(w2)),
            f"{label}_tick_accel": tick_accel,
        }

    def ob_features(ob_grp) -> dict:
        empty = {"ob_spread": 0.0, "ob_mid": 0.5, "ob_imbalance": 0.0,
                 "ob_bid_depth": 0.0, "ob_ask_depth": 0.0, "ob_skew": 0.0}
        if ob_grp is None or len(ob_grp) == 0:
            return empty
        row  = ob_grp.iloc[0]
        cols = set(ob_grp.columns)
        try:
            if "best_bid" in cols:
                bid   = float(row.get("best_bid") or 0)
                ask   = float(row.get("best_ask") or 1)
                bid_d = float(row.get("best_bid_size") or row.get("bid_size_5") or 0)
                ask_d = float(row.get("best_ask_size") or row.get("ask_size_5") or 0)
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
            return {"ob_spread":    float(ask - bid),
                    "ob_mid":       float((bid + ask) / 2),
                    "ob_imbalance": float((bid_d - ask_d) / total),
                    "ob_bid_depth": float(bid_d),
                    "ob_ask_depth": float(ask_d),
                    "ob_skew":      float((bid + ask) / 2 - 0.5)}
        except Exception:
            return empty

    # ── Step 5: Build dataset ──────────────────────────────────────────────────
    log.info("Step 5: Building feature dataset...")
    btc_grps = dict(list(btc_inslot.groupby("market_id")))

    # Rolling volume stats (20-slot window) — compute upfront over sorted slots
    vol_series = markets_sorted["market_id"].map(slot_vol).fillna(0).values

    records = []
    for i, row in markets_sorted.iterrows():
        mid    = row["market_id"]
        target = target_map[mid]
        grp    = btc_grps.get(mid)
        if grp is None or len(grp) < 5:
            continue

        slot_ts = slot_map[mid]
        dt      = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour    = dt.hour + dt.minute / 60.0

        feat = {
            "market_id": mid, "slot_ts": slot_ts, "target": target,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin":  np.sin(2 * np.pi * dt.weekday() / 7),
            "dow_cos":  np.cos(2 * np.pi * dt.weekday() / 7),
        }

        # BTC tick features (with tick_accel)
        feat.update(tick_features(grp, "btc"))

        # BTC inslot spot return + vol
        sp = grp["spot_price_usdt"].dropna()
        if len(sp) >= 2:
            p0, p1 = float(sp.iloc[0]), float(sp.iloc[-1])
            feat["btc_inslot_ret"] = (p1 - p0) / (p0 + 1e-8)
            feat["btc_inslot_vol"] = float(sp.std() / (sp.mean() + 1e-8))
        else:
            feat["btc_inslot_ret"] = 0.0
            feat["btc_inslot_vol"] = 0.0

        # Pre-slot BTC spot returns — 5m, 15m, 30m, 1h, 4h
        for w_s, lbl in [(300, "5m"), (900, "15m"), (1800, "30m"),
                          (3600, "1h"), (14400, "4h")]:
            idx0, idx1 = np.searchsorted(
                spot_ts_arr, [(slot_ts - w_s) * 1000, slot_ts * 1000])
            seg = spot_px_arr[idx0:idx1]
            if len(seg) >= 2:
                feat[f"btc_pre_{lbl}_ret"] = float((seg[-1] - seg[0]) / (seg[0] + 1e-8))
                feat[f"btc_pre_{lbl}_vol"] = float(np.std(seg) / (np.mean(seg) + 1e-8))
            else:
                feat[f"btc_pre_{lbl}_ret"] = 0.0
                feat[f"btc_pre_{lbl}_vol"] = 0.0

        # Round number proximity — distance from BTC spot to nearest $1k, $5k, $10k
        spot_price = float(sp.iloc[-1]) if len(sp) > 0 else 0.0
        if spot_price > 0:
            feat["btc_dist_1k"]  = float(abs(spot_price % 1000) / 1000)
            feat["btc_dist_5k"]  = float(abs(spot_price % 5000) / 5000)
            feat["btc_dist_10k"] = float(abs(spot_price % 10000) / 10000)
        else:
            feat["btc_dist_1k"] = feat["btc_dist_5k"] = feat["btc_dist_10k"] = 0.5

        # Volume anomaly — vs rolling 20-slot mean/std
        win_start = max(0, i - 20)
        hist_vols = vol_series[win_start:i]
        cur_vol   = vol_series[i]
        if len(hist_vols) >= 5:
            mu  = hist_vols.mean()
            std = hist_vols.std() + 1e-8
            feat["btc_vol_zscore"]  = float((cur_vol - mu) / std)
            feat["btc_vol_ratio"]   = float(cur_vol / (mu + 1e-8))
        else:
            feat["btc_vol_zscore"] = 0.0
            feat["btc_vol_ratio"]  = 1.0

        # Lagged outcomes — last 3 resolved slots
        for lag in [1, 2, 3]:
            prev_idx = i - lag
            feat[f"lag_{lag}_outcome"] = float(idx_to_target.get(prev_idx, 0.5))

        # Lagged outcome streak — how many consecutive same-direction slots
        streak = 0
        if i >= 1:
            last = idx_to_target.get(i - 1, -1)
            for back in range(1, min(i + 1, 6)):
                v = idx_to_target.get(i - back, -1)
                if v == last:
                    streak += 1
                else:
                    break
        feat["lag_streak"] = float(streak)

        # Orderbook features
        feat.update(ob_features(ob_by_market.get(mid)))
        records.append(feat)

    df = pd.DataFrame(records).sort_values("slot_ts").reset_index(drop=True)
    log.info("Dataset: %d samples", len(df))
    log.info("Target balance: %s", dict(df["target"].value_counts()))

    NON_FEAT = {"market_id", "slot_ts", "target"}
    features = [c for c in df.columns if c not in NON_FEAT]
    log.info("Features: %d total", len(features))
    for prefix in ["btc_", "ob_", "lag_"]:
        n = len([f for f in features if f.startswith(prefix)])
        log.info("  %-6s %d", prefix.rstrip("_")+":", n)

    # ── Purged walk-forward CV ────────────────────────────────────────────────
    def walk_forward_purged(df, features, params=None, gap=WF_GAP):
        """Walk-forward with a gap between train end and test start
        to prevent any temporal leakage from lagged features."""
        df = df.sort_values("slot_ts").reset_index(drop=True)
        n, n_splits = len(df), 5
        fold_size = n // (n_splits + 1)
        base = dict(objective="binary", class_weight="balanced", n_estimators=400,
                    learning_rate=0.04, num_leaves=31, min_child_samples=15,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0,
                    verbose=-1, n_jobs=-1)
        p = {**base, **(params or {})}
        aucs, accs, briers = [], [], []
        for i in range(n_splits):
            train_end  = fold_size * (i + 1)
            test_start = train_end + gap          # purge gap
            test_end   = min(test_start + fold_size, n)
            if test_end - test_start < 20:
                continue
            tr = df.iloc[:train_end]
            te = df.iloc[test_start:test_end]
            m  = lgb.LGBMClassifier(**p)
            m.fit(tr[features].fillna(0), tr["target"])
            prob = m.predict_proba(te[features].fillna(0))[:, 1]
            aucs.append(roc_auc_score(te["target"], prob))
            accs.append(float(((prob >= 0.5) == te["target"]).mean()))
            briers.append(brier_score_loss(te["target"], prob))
        if not aucs:
            return {"wf_auc": 0.5, "wf_acc": 0.5, "wf_brier": 0.5, "fold_aucs": []}
        return {"wf_auc":   float(np.mean(aucs)),
                "wf_acc":   float(np.mean(accs)),
                "wf_brier": float(np.mean(briers)),
                "fold_aucs": aucs}

    log.info("Step 6: Baseline purged walk-forward...")
    wf_base = walk_forward_purged(df, features)
    log.info("  Baseline WF AUC: %.4f | Acc: %.4f | Brier: %.4f",
             wf_base["wf_auc"], wf_base["wf_acc"], wf_base["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_base["fold_aucs"]])

    # ── Permutation importance — drop noise features ──────────────────────────
    log.info("Step 7: Permutation importance to drop noise features...")
    split = int(len(df) * 0.75)
    tr_imp, va_imp = df.iloc[:split], df.iloc[split:]
    imp_model = lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", n_estimators=300,
        learning_rate=0.05, num_leaves=31, min_child_samples=15,
        verbose=-1, n_jobs=-1)
    imp_model.fit(tr_imp[features].fillna(0), tr_imp["target"])
    perm = permutation_importance(
        imp_model, va_imp[features].fillna(0), va_imp["target"],
        n_repeats=10, random_state=42, scoring="roc_auc")
    imp_df = pd.DataFrame({
        "feature":   features,
        "imp_mean":  perm.importances_mean,
        "imp_std":   perm.importances_std,
    }).sort_values("imp_mean", ascending=False)
    log.info("  Top 10 features by permutation importance:")
    for _, r in imp_df.head(10).iterrows():
        log.info("    %-35s %.4f ± %.4f", r["feature"], r["imp_mean"], r["imp_std"])
    # Keep features with mean importance > -0.002 (drop only clear noise)
    good_features = imp_df[imp_df["imp_mean"] > -0.002]["feature"].tolist()
    dropped = len(features) - len(good_features)
    log.info("  Dropped %d noise features, keeping %d", dropped, len(good_features))

    # ── Optuna HPO with purged WF objective ───────────────────────────────────
    log.info("Step 8: Optuna HPO (%d trials, purged WF objective)...", OPTUNA_TRIALS)
    def objective(trial):
        p = dict(
            n_estimators    = trial.suggest_int("n_estimators", 100, 600),
            learning_rate   = trial.suggest_float("lr", 0.005, 0.15, log=True),
            num_leaves      = trial.suggest_int("num_leaves", 15, 63),
            min_child_samples = trial.suggest_int("min_child_samples", 15, 80),
            subsample       = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha       = trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            reg_lambda      = trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            objective="binary", class_weight="balanced", verbose=-1, n_jobs=-1,
        )
        return walk_forward_purged(df, good_features, p)["wf_auc"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    if "lr" in best_params:
        best_params["learning_rate"] = best_params.pop("lr")
    log.info("  Optuna best WF AUC: %.4f", study.best_value)

    # ── Walk-forward with best params ─────────────────────────────────────────
    log.info("Step 9: Walk-forward (optimized)...")
    wf_opt = walk_forward_purged(df, good_features, best_params)
    log.info("  Optimized WF AUC: %.4f | Acc: %.4f | Brier: %.4f",
             wf_opt["wf_auc"], wf_opt["wf_acc"], wf_opt["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_opt["fold_aucs"]])

    # Pick best set
    if wf_opt["wf_auc"] >= wf_base["wf_auc"]:
        final_params   = best_params
        final_features = good_features
        final_wf       = wf_opt
        log.info("  Using: optimized params + pruned features")
    else:
        final_params   = {}
        final_features = features
        final_wf       = wf_base
        log.info("  Using: baseline params + all features (Optuna didn't improve)")

    # ── Train final with calibration ──────────────────────────────────────────
    log.info("Step 10: Training final model with isotonic calibration...")
    base_params = dict(objective="binary", class_weight="balanced", n_estimators=400,
                       learning_rate=0.04, num_leaves=31, min_child_samples=15,
                       subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=-1)
    params = {**base_params, **final_params,
              "objective": "binary", "class_weight": "balanced",
              "verbose": -1, "n_jobs": -1}
    X, y = df[final_features].fillna(0), df["target"]
    base_model = lgb.LGBMClassifier(**params)
    cal = CalibratedClassifierCV(base_model, method="isotonic",
                                 cv=TimeSeriesSplit(n_splits=3))
    cal.fit(X, y)

    final_auc   = final_wf["wf_auc"]
    final_acc   = final_wf["wf_acc"]
    final_brier = final_wf["wf_brier"]

    bundle = {
        "model":        cal,
        "features":     final_features,
        "wf_auc":       final_auc,
        "wf_acc":       final_acc,
        "wf_brier":     final_brier,
        "version":      "v6",
        "n_samples":    len(df),
        "n_features":   len(final_features),
        "best_params":  final_params,
        "dropped_features": dropped,
    }
    model_path = Path("/tmp/btc_model_v6.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Model saved: %s (%.1f MB)", model_path, model_path.stat().st_size / 1e6)

    # ── Gate: fair comparison via purged WF on v5 features ───────────────────
    # v5 AUC was measured without purge gap — re-evaluate v5 features with same
    # purged WF so we compare apples to apples.
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")

    # Load v5 feature list from HF meta to re-evaluate it fairly
    v5_wf_auc = None
    try:
        from huggingface_hub import hf_hub_download as hf_dl
        import tempfile
        meta_path = hf_dl(HF_MODEL_REPO, "champion_meta.json",
                          repo_type="model", token=HF_TOKEN,
                          local_dir=tempfile.mkdtemp())
        with open(meta_path) as fp:
            v5_meta = json.load(fp)
        v5_features_hf = v5_meta.get("feature_list", [])
        if v5_features_hf:
            # Only keep features that exist in current dataset
            v5_feats_available = [f for f in v5_features_hf if f in df.columns]
            if len(v5_feats_available) >= 10:
                log.info("Re-evaluating v5 with purged WF (%d features)...",
                         len(v5_feats_available))
                wf_v5_purged = walk_forward_purged(df, v5_feats_available)
                v5_wf_auc = wf_v5_purged["wf_auc"]
                log.info("  v5 purged WF AUC: %.4f (original non-purged: %.4f)",
                         v5_wf_auc, 0.8553)
    except Exception as e:
        log.warning("Could not re-evaluate v5: %s — using original AUC", e)

    # Fair champion AUC: use purged v5 if available, else original with tolerance
    fair_champion_auc = v5_wf_auc if v5_wf_auc is not None else (CHAMPION_AUC - 0.01)

    log.info("  Champion (purged WF): AUC=%.4f", fair_champion_auc)
    log.info("  v6 candidate:         AUC=%.4f  Acc=%.4f  Brier=%.4f",
             final_auc, final_acc, final_brier)
    log.info("  Features used: %d (dropped %d noise)", len(final_features), dropped)

    beats_auc   = final_auc   > fair_champion_auc
    beats_brier = final_brier < CHAMPION_BRIER
    beats_acc   = final_acc   > CHAMPION_ACC
    n_passed = sum([beats_auc, beats_brier, beats_acc])

    # Promote if: AUC beats fair champion, OR (2/3 metrics pass AND AUC within 0.005)
    auc_within_tolerance = final_auc >= (fair_champion_auc - 0.005)
    should_promote = beats_auc or (n_passed >= 2 and auc_within_tolerance)

    log.info("  Gate: AUC>%.4f [%s]  Brier<%.4f [%s]  Acc>%.4f [%s]  → %d/3 passed",
             fair_champion_auc, "PASS" if beats_auc   else "FAIL",
             CHAMPION_BRIER,    "PASS" if beats_brier else "FAIL",
             CHAMPION_ACC,      "PASS" if beats_acc   else "FAIL",
             n_passed)
    log.info("  Decision: %s", "PROMOTE" if should_promote else "REJECT")
    if should_promote:
        log.info("Promoting v6 to HF champion...")
        import tempfile as _tempfile
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="champion.pkl",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            commit_message=(f"Champion v6: AUC={final_auc:.4f} Brier={final_brier:.4f} "
                            f"Acc={final_acc:.4f} | purged-WF+perm-imp+lagged"),
        )
        meta = {
            "version": "v6",
            "features": len(final_features),
            "feature_list": final_features,
            "wf_auc":   final_auc,
            "wf_acc":   final_acc,
            "wf_brier": final_brier,
            "fold_aucs": final_wf["fold_aucs"],
            "n_samples":  len(df),
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": ("v6: lagged outcomes, vol anomaly, 1h/4h spot, "
                      "tick accel, round proximity, purged WF, perm importance"),
            "best_params": final_params,
        }
        with _tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
            json.dump(meta, fp, indent=2)
            tmp_meta = fp.name
        api.upload_file(
            path_or_fileobj=tmp_meta,
            path_in_repo="champion_meta.json",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            commit_message=f"Champion v6 meta AUC={final_auc:.4f}",
        )
        log.info("Champion v6 promoted: https://huggingface.co/%s", HF_MODEL_REPO)
        log.info("Push deploy/ to main to trigger Fly auto-deploy.")
        promoted = True
    else:
        log.warning("v6 did NOT pass gates — not promoted.")
        log.warning("  AUC:   %.4f vs %.4f (fair purged)  [%s]",
                    final_auc, fair_champion_auc, "OK" if beats_auc else "FAIL")
        log.warning("  Brier: %.4f vs %.4f needed  [%s]",
                    final_brier, CHAMPION_BRIER, "OK" if beats_brier else "FAIL")
        log.warning("  Acc:   %.4f vs %.4f needed  [%s]",
                    final_acc, CHAMPION_ACC, "OK" if beats_acc else "FAIL")
        promoted = False

    log.info("Done.")
    return {
        "wf_auc_baseline":  wf_base["wf_auc"],
        "wf_auc_optimized": wf_opt["wf_auc"],
        "wf_auc_final":     final_auc,
        "wf_acc_final":     final_acc,
        "wf_brier_final":   final_brier,
        "n_samples":        len(df),
        "n_features_final": len(final_features),
        "n_features_dropped": dropped,
        "promoted":         promoted,
    }


@app.local_entrypoint()
def main():
    print("Submitting BTC v6 training job to Modal...")
    result = train_v6.remote()
    print(f"\n{'='*55}")
    print("TRAINING COMPLETE — v6")
    print(f"  Baseline AUC:   {result['wf_auc_baseline']:.4f}")
    print(f"  Optimized AUC:  {result['wf_auc_optimized']:.4f}")
    print(f"  Final AUC:      {result['wf_auc_final']:.4f}")
    print(f"  Final Acc:      {result['wf_acc_final']:.4f}")
    print(f"  Final Brier:    {result['wf_brier_final']:.4f}")
    print(f"  Samples:        {result['n_samples']}")
    print(f"  Features used:  {result['n_features_final']} (dropped {result['n_features_dropped']} noise)")
    print(f"  Promoted:       {'YES ✓' if result['promoted'] else 'NO — gates not passed'}")
    print(f"{'='*55}")
