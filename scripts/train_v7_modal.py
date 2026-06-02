"""
train_v7_modal.py — BTC 5min model v7

New vs v6:
  - Fix: champion AUC loaded from HF meta (no more hardcoding)
  - Fix: lag index uses slot_ts rank, not raw dataset index (no gaps)
  - Fix: btc_inslot_ret uses spot_ts_arr open (slot_ts anchor), not first tick
  - Feature: Up/Down OB split — implied probability from Up token bid/ask
              + ob_up_bid_depth, ob_dn_bid_depth, ob_depth_ratio
              (the strongest market-implied signal)
  - Feature: realized vol of last 5/10 slot returns (vol clustering)
  - Feature: historical up_ratio mean/std over last 20 slots (anomaly)
  - Model: ensemble LightGBM + LogisticRegression (avg probabilities)
           better generalization on small datasets
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

app = modal.App("polymarket-btc-train-v7", image=image)

@app.function(
    cpu=8,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
)
def train_v7():
    import gc, json, logging, os, pickle, sys, time, warnings, tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np
    import optuna
    import pandas as pd
    import pyarrow.parquet as pq
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
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
    OBS_SECS      = 180
    OPTUNA_TRIALS = 80
    WF_GAP        = 5
    DATA_DIR      = Path("/tmp/hf_data")
    DATA_DIR.mkdir(exist_ok=True)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")

    # ── Step 1: Load champion metrics from HF (no hardcoding) ─────────────────
    log.info("Step 1: Loading champion metrics from HF...")
    champion = {"version": "none", "wf_auc": 0.0, "wf_brier": 1.0, "wf_acc": 0.0,
                "feature_list": []}
    try:
        meta_path = hf_hub_download(HF_MODEL_REPO, "champion_meta.json",
                                    repo_type="model", token=HF_TOKEN,
                                    local_dir=tempfile.mkdtemp())
        with open(meta_path) as fp:
            champion = json.load(fp)
        log.info("  Champion: %s | AUC=%.4f | Brier=%s | Acc=%s",
                 champion.get("version"), champion.get("wf_auc"),
                 champion.get("wf_brier", "N/A"), champion.get("wf_acc", "N/A"))
    except Exception as e:
        log.warning("  Could not load champion meta: %s — will compare against AUC=0", e)

    CHAMPION_AUC   = float(champion.get("wf_auc",   0.0))
    CHAMPION_BRIER = float(champion.get("wf_brier", 1.0)) if champion.get("wf_brier") else 0.22
    CHAMPION_ACC   = float(champion.get("wf_acc",   0.0)) if champion.get("wf_acc")   else 0.73
    CHAMPION_FEATS = champion.get("feature_list", [])

    # ── Step 2: Download data ──────────────────────────────────────────────────
    files = [
        "data/markets.parquet",
        "data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet",
        "data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet",
    ]
    log.info("Step 2: Downloading %d files...", len(files))
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

    # ── Step 3: Markets ────────────────────────────────────────────────────────
    log.info("Step 3: Loading markets...")
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

    # Sorted slot list — use slot_ts rank for lag, not dataset index (fix #2)
    markets_sorted  = markets.sort_values("slot_ts").reset_index(drop=True)
    # Map slot_ts → (rank_idx, target) so gaps in dataset don't corrupt lags
    slotts_to_rank   = {row["slot_ts"]: i for i, row in markets_sorted.iterrows()}
    rank_to_target   = dict(enumerate(markets_sorted["resolution"].astype(int)))
    rank_to_slotts   = dict(enumerate(markets_sorted["slot_ts"]))

    # ── Step 4: BTC ticks ─────────────────────────────────────────────────────
    log.info("Step 4: Loading BTC ticks...")
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

    slot_vol = btc_inslot.groupby("market_id")["size_usdc"].sum().rename("slot_vol")

    # Spot timeline
    spot_tl = btc[["timestamp_ms", "spot_price_usdt"]].dropna().drop_duplicates("timestamp_ms")
    spot_tl = spot_tl.set_index("timestamp_ms").sort_index()
    spot_ts_arr = spot_tl.index.values
    spot_px_arr = spot_tl["spot_price_usdt"].values
    del btc; gc.collect()

    # ── Step 5: Orderbook — split Up vs Down token ────────────────────────────
    log.info("Step 5: Loading BTC orderbook (Up/Down split)...")
    ob_up_by_market = {}   # Up token OB
    ob_dn_by_market = {}   # Down token OB
    ob_path = DATA_DIR / "data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet"
    if ob_path.exists():
        try:
            schema_cols = [f.name for f in pq.read_schema(str(ob_path))]
            log.info("  OB cols: %s", schema_cols)
            ob = pq.read_table(str(ob_path),
                               filters=[("market_id", "in", list(market_ids))]).to_pandas()
            log.info("  OB rows: %d | outcomes: %s",
                     len(ob), ob["outcome"].unique().tolist() if "outcome" in ob.columns else "N/A")
            ts_col = "ts_ms" if "ts_ms" in ob.columns else "timestamp_ms"
            for mid, grp in ob.groupby("market_id"):
                grp_sorted = grp.sort_values(ts_col)
                if "outcome" in grp.columns:
                    up_rows = grp_sorted[grp_sorted["outcome"] == "Up"].head(3)
                    dn_rows = grp_sorted[grp_sorted["outcome"] == "Down"].head(3)
                    if len(up_rows): ob_up_by_market[mid] = up_rows
                    if len(dn_rows): ob_dn_by_market[mid] = dn_rows
                else:
                    ob_up_by_market[mid] = grp_sorted.head(3)
            del ob; gc.collect()
            log.info("  OB indexed: %d up, %d down markets",
                     len(ob_up_by_market), len(ob_dn_by_market))
        except Exception as e:
            log.warning("  OB load failed: %s", e)

    # ── Feature helpers ────────────────────────────────────────────────────────
    def tick_features(grp: pd.DataFrame, label: str) -> dict:
        n = len(grp)
        empty = {
            f"{label}_n_ticks": 0.0, f"{label}_up_ratio": 0.5,
            f"{label}_momentum": 0.0, f"{label}_vwap_spread": 0.0,
            f"{label}_vol_up": 0.0,   f"{label}_vol_dn": 0.0,
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
        first30 = (grp["t_sec"] < 30).sum()
        last30  = (grp["t_sec"] >= (OBS_SECS - 30)).sum()

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
            f"{label}_momentum":    float(ur(w2) - ur(w0)),
            f"{label}_up_w0":       float(ur(w0)),
            f"{label}_up_w1":       float(ur(w1)),
            f"{label}_up_w2":       float(ur(w2)),
            f"{label}_tick_accel":  float((last30 - first30) / (first30 + 1e-8)),
        }

    def ob_token_features(ob_grp, label: str) -> dict:
        """Features from a single token's (Up or Down) orderbook."""
        empty = {f"ob_{label}_bid": 0.5, f"ob_{label}_ask": 0.5,
                 f"ob_{label}_spread": 0.0, f"ob_{label}_bid_depth": 0.0,
                 f"ob_{label}_ask_depth": 0.0, f"ob_{label}_imbalance": 0.0}
        if ob_grp is None or len(ob_grp) == 0:
            return empty
        row  = ob_grp.iloc[0]
        try:
            bid   = float(row.get("best_bid") or 0)
            ask   = float(row.get("best_ask") or 1)
            bid_d = float(row.get("best_bid_size") or 0)
            ask_d = float(row.get("best_ask_size") or 0)
            total = bid_d + ask_d + 1e-8
            return {
                f"ob_{label}_bid":       bid,
                f"ob_{label}_ask":       ask,
                f"ob_{label}_spread":    float(ask - bid),
                f"ob_{label}_bid_depth": float(bid_d),
                f"ob_{label}_ask_depth": float(ask_d),
                f"ob_{label}_imbalance": float((bid_d - ask_d) / total),
            }
        except Exception:
            return empty

    def spot_open_at_slot(slot_ts: int) -> float:
        """BTC spot price at exactly slot_ts (or nearest prior tick)."""
        idx = np.searchsorted(spot_ts_arr, slot_ts * 1000, side="right") - 1
        if idx >= 0:
            return float(spot_px_arr[idx])
        return 0.0

    # ── Step 6: Build dataset ──────────────────────────────────────────────────
    log.info("Step 6: Building feature dataset...")
    btc_grps  = dict(list(btc_inslot.groupby("market_id")))
    vol_series = markets_sorted["market_id"].map(slot_vol).fillna(0).values

    # Pre-compute up_ratio per slot for historical anomaly feature
    up_ratio_series = np.array([
        float(
            btc_grps[row["market_id"]]["size_usdc"][btc_grps[row["market_id"]]["outcome"] == "Up"].sum() /
            (btc_grps[row["market_id"]]["size_usdc"].sum() + 1e-8)
        ) if row["market_id"] in btc_grps else 0.5
        for _, row in markets_sorted.iterrows()
    ])

    records = []
    for i, row in markets_sorted.iterrows():
        mid    = row["market_id"]
        target = target_map[mid]
        grp    = btc_grps.get(mid)
        if grp is None or len(grp) < 5:
            continue

        slot_ts = slot_map[mid]
        rank    = slotts_to_rank[slot_ts]   # use slot_ts-based rank (fix #2)
        dt      = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour    = dt.hour + dt.minute / 60.0

        feat = {
            "market_id": mid, "slot_ts": slot_ts, "target": target,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin":  np.sin(2 * np.pi * dt.weekday() / 7),
            "dow_cos":  np.cos(2 * np.pi * dt.weekday() / 7),
        }

        # BTC tick features
        feat.update(tick_features(grp, "btc"))

        # Fix #3: BTC inslot spot return anchored at slot_ts (not first tick)
        spot_open = spot_open_at_slot(slot_ts)
        sp = grp["spot_price_usdt"].dropna()
        if spot_open > 0 and len(sp) >= 1:
            p_close = float(sp.iloc[-1])
            feat["btc_inslot_ret"] = (p_close - spot_open) / (spot_open + 1e-8)
            feat["btc_inslot_vol"] = float(sp.std() / (sp.mean() + 1e-8)) if len(sp) >= 2 else 0.0
        else:
            feat["btc_inslot_ret"] = 0.0
            feat["btc_inslot_vol"] = 0.0

        # Pre-slot BTC spot returns: 5m, 15m, 30m, 1h, 4h
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

        # Round number proximity
        if spot_open > 0:
            feat["btc_dist_1k"]  = float(abs(spot_open % 1000) / 1000)
            feat["btc_dist_5k"]  = float(abs(spot_open % 5000) / 5000)
            feat["btc_dist_10k"] = float(abs(spot_open % 10000) / 10000)
        else:
            feat["btc_dist_1k"] = feat["btc_dist_5k"] = feat["btc_dist_10k"] = 0.5

        # Volume anomaly
        win_start = max(0, rank - 20)
        hist_vols = vol_series[win_start:rank]
        cur_vol   = vol_series[rank]
        if len(hist_vols) >= 5:
            mu  = hist_vols.mean()
            std = hist_vols.std() + 1e-8
            feat["btc_vol_zscore"] = float((cur_vol - mu) / std)
            feat["btc_vol_ratio"]  = float(cur_vol / (mu + 1e-8))
        else:
            feat["btc_vol_zscore"] = 0.0
            feat["btc_vol_ratio"]  = 1.0

        # Realized vol of last 5 / 10 slot returns (vol clustering)
        for win, lbl in [(5, "5s"), (10, "10s")]:
            win_start_r = max(0, rank - win)
            past_rets = []
            for back_rank in range(win_start_r, rank):
                back_slotts = rank_to_slotts.get(back_rank)
                if back_slotts is None:
                    continue
                idx0, idx1 = np.searchsorted(
                    spot_ts_arr, [(back_slotts - 300) * 1000, back_slotts * 1000])
                seg = spot_px_arr[idx0:idx1]
                if len(seg) >= 2:
                    past_rets.append((seg[-1] - seg[0]) / (seg[0] + 1e-8))
            feat[f"btc_realized_vol_{lbl}"] = float(np.std(past_rets)) if len(past_rets) >= 3 else 0.0

        # Historical up_ratio anomaly (last 20 slots)
        win_start_ur = max(0, rank - 20)
        hist_ur = up_ratio_series[win_start_ur:rank]
        cur_ur  = up_ratio_series[rank]
        if len(hist_ur) >= 5:
            feat["btc_up_ratio_zscore"] = float((cur_ur - hist_ur.mean()) / (hist_ur.std() + 1e-8))
            feat["btc_up_ratio_hist_mean"] = float(hist_ur.mean())
        else:
            feat["btc_up_ratio_zscore"]    = 0.0
            feat["btc_up_ratio_hist_mean"] = 0.5

        # Lagged outcomes — using slot_ts rank (fix #2)
        for lag in [1, 2, 3]:
            feat[f"lag_{lag}_outcome"] = float(rank_to_target.get(rank - lag, 0.5))

        # Lag streak
        streak = 0
        if rank >= 1:
            last_val = rank_to_target.get(rank - 1, -1)
            for back in range(1, min(rank + 1, 6)):
                v = rank_to_target.get(rank - back, -1)
                if v == last_val and v != -1:
                    streak += 1
                else:
                    break
        feat["lag_streak"] = float(streak)

        # OB Up/Down token features (new in v7)
        feat.update(ob_token_features(ob_up_by_market.get(mid), "up"))
        feat.update(ob_token_features(ob_dn_by_market.get(mid), "dn"))

        # Derived OB features: implied probability + depth ratio
        ob_up_bid = feat.get("ob_up_bid", 0.5)
        ob_dn_bid = feat.get("ob_dn_bid", 0.5)
        ob_up_dep = feat.get("ob_up_bid_depth", 0.0)
        ob_dn_dep = feat.get("ob_dn_bid_depth", 0.0)
        # Up token bid ≈ market-implied P(UP) — most important feature
        feat["ob_implied_prob_up"]   = float(ob_up_bid)
        # Depth imbalance: more depth on Up = market leans bullish
        feat["ob_depth_ratio"]       = float(ob_up_dep / (ob_dn_dep + 1e-8))
        feat["ob_depth_sum"]         = float(ob_up_dep + ob_dn_dep)

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

    # ── Purged walk-forward CV ─────────────────────────────────────────────────
    def walk_forward_purged(df, feats, params=None, gap=WF_GAP, use_ensemble=False):
        df = df.sort_values("slot_ts").reset_index(drop=True)
        n, n_splits = len(df), 5
        fold_size = n // (n_splits + 1)
        base = dict(objective="binary", class_weight="balanced", n_estimators=400,
                    learning_rate=0.04, num_leaves=31, min_child_samples=15,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=-1)
        p = {**base, **(params or {})}
        aucs, accs, briers = [], [], []
        for i in range(n_splits):
            train_end  = fold_size * (i + 1)
            test_start = train_end + gap
            test_end   = min(test_start + fold_size, n)
            if test_end - test_start < 20:
                continue
            tr = df.iloc[:train_end]
            te = df.iloc[test_start:test_end]
            Xtr, ytr = tr[feats].fillna(0), tr["target"]
            Xte, yte = te[feats].fillna(0), te["target"]

            lgb_model = lgb.LGBMClassifier(**p)
            lgb_model.fit(Xtr, ytr)
            lgb_prob = lgb_model.predict_proba(Xte)[:, 1]

            if use_ensemble:
                lr_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=0.1, class_weight="balanced",
                                              max_iter=500, solver="lbfgs")),
                ])
                lr_pipe.fit(Xtr, ytr)
                lr_prob = lr_pipe.predict_proba(Xte)[:, 1]
                prob = (lgb_prob * 0.65 + lr_prob * 0.35)
            else:
                prob = lgb_prob

            aucs.append(roc_auc_score(yte, prob))
            accs.append(float(((prob >= 0.5) == yte).mean()))
            briers.append(brier_score_loss(yte, prob))

        if not aucs:
            return {"wf_auc": 0.5, "wf_acc": 0.5, "wf_brier": 0.5, "fold_aucs": []}
        return {"wf_auc":   float(np.mean(aucs)),
                "wf_acc":   float(np.mean(accs)),
                "wf_brier": float(np.mean(briers)),
                "fold_aucs": aucs}

    # ── Step 7: Baseline purged WF ─────────────────────────────────────────────
    log.info("Step 7: Baseline purged walk-forward...")
    wf_base = walk_forward_purged(df, features)
    log.info("  Baseline WF AUC: %.4f | Acc: %.4f | Brier: %.4f",
             wf_base["wf_auc"], wf_base["wf_acc"], wf_base["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_base["fold_aucs"]])

    # ── Step 8: Permutation importance ────────────────────────────────────────
    log.info("Step 8: Permutation importance...")
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
        "feature":  features,
        "imp_mean": perm.importances_mean,
        "imp_std":  perm.importances_std,
    }).sort_values("imp_mean", ascending=False)
    log.info("  Top 15 features:")
    for _, r in imp_df.head(15).iterrows():
        log.info("    %-40s %.4f ± %.4f", r["feature"], r["imp_mean"], r["imp_std"])
    good_features = imp_df[imp_df["imp_mean"] > -0.002]["feature"].tolist()
    dropped = len(features) - len(good_features)
    log.info("  Dropped %d noise features, keeping %d", dropped, len(good_features))

    # ── Step 9: Optuna HPO ────────────────────────────────────────────────────
    log.info("Step 9: Optuna HPO (%d trials)...", OPTUNA_TRIALS)
    def objective(trial):
        p = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 600),
            learning_rate     = trial.suggest_float("lr", 0.005, 0.15, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 15, 63),
            min_child_samples = trial.suggest_int("min_child_samples", 15, 80),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            objective="binary", class_weight="balanced", verbose=-1, n_jobs=-1,
        )
        return walk_forward_purged(df, good_features, p, use_ensemble=False)["wf_auc"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    if "lr" in best_params:
        best_params["learning_rate"] = best_params.pop("lr")
    log.info("  Optuna best WF AUC: %.4f", study.best_value)

    # ── Step 10: Walk-forward — optimized LightGBM ───────────────────────────
    log.info("Step 10: WF with optimized LightGBM...")
    wf_lgb = walk_forward_purged(df, good_features, best_params, use_ensemble=False)
    log.info("  LightGBM WF AUC: %.4f | Acc: %.4f | Brier: %.4f",
             wf_lgb["wf_auc"], wf_lgb["wf_acc"], wf_lgb["wf_brier"])

    # ── Step 11: Walk-forward — ensemble LightGBM + LR ───────────────────────
    log.info("Step 11: WF with ensemble (LightGBM + LogReg)...")
    wf_ens = walk_forward_purged(df, good_features, best_params, use_ensemble=True)
    log.info("  Ensemble WF AUC: %.4f | Acc: %.4f | Brier: %.4f",
             wf_ens["wf_auc"], wf_ens["wf_acc"], wf_ens["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_ens["fold_aucs"]])

    # Pick best: ensemble or lgb-only
    if wf_ens["wf_auc"] >= wf_lgb["wf_auc"]:
        final_wf      = wf_ens
        use_ens_final = True
        log.info("  Winner: ensemble")
    else:
        final_wf      = wf_lgb
        use_ens_final = False
        log.info("  Winner: LightGBM only (ensemble didn't help)")

    if final_wf["wf_auc"] < wf_base["wf_auc"]:
        log.info("  Both worse than baseline — falling back to baseline features+params")
        final_wf      = wf_base
        best_params   = {}
        good_features = features
        use_ens_final = False

    final_auc, final_acc, final_brier = (
        final_wf["wf_auc"], final_wf["wf_acc"], final_wf["wf_brier"])

    # ── Step 12: Re-evaluate champion on same purged WF ───────────────────────
    log.info("Step 12: Re-evaluating champion (%s) with purged WF...",
             champion.get("version", "?"))
    champ_feats_available = [f for f in CHAMPION_FEATS if f in df.columns]
    if len(champ_feats_available) >= 10:
        wf_champ = walk_forward_purged(df, champ_feats_available)
        fair_champ_auc = wf_champ["wf_auc"]
        log.info("  Champion purged WF AUC: %.4f (original: %.4f)",
                 fair_champ_auc, CHAMPION_AUC)
    else:
        fair_champ_auc = max(CHAMPION_AUC - 0.01, 0.0)
        log.info("  Champion features not in dataset — using AUC=%.4f with tolerance",
                 fair_champ_auc)

    # ── Step 13: Train final model ────────────────────────────────────────────
    log.info("Step 13: Training final model...")
    base_p = dict(objective="binary", class_weight="balanced", n_estimators=400,
                  learning_rate=0.04, num_leaves=31, min_child_samples=15,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=-1)
    params = {**base_p, **best_params,
              "objective": "binary", "class_weight": "balanced",
              "verbose": -1, "n_jobs": -1}
    X, y = df[good_features].fillna(0), df["target"]

    lgb_final = lgb.LGBMClassifier(**params)
    cal_lgb   = CalibratedClassifierCV(lgb_final, method="isotonic",
                                        cv=TimeSeriesSplit(n_splits=3))
    cal_lgb.fit(X, y)

    if use_ens_final:
        lr_final = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, class_weight="balanced",
                                      max_iter=500, solver="lbfgs")),
        ])
        lr_final.fit(X, y)
        final_model_obj = {"lgb": cal_lgb, "lr": lr_final,
                           "ensemble": True, "lgb_weight": 0.65}
    else:
        final_model_obj = {"lgb": cal_lgb, "ensemble": False}

    bundle = {
        "model":            final_model_obj,
        "features":         good_features,
        "wf_auc":           final_auc,
        "wf_acc":           final_acc,
        "wf_brier":         final_brier,
        "fold_aucs":        final_wf["fold_aucs"],
        "version":          "v7",
        "n_samples":        len(df),
        "n_features":       len(good_features),
        "best_params":      best_params,
        "ensemble":         use_ens_final,
        "dropped_features": dropped,
        "champion_compared_auc": fair_champ_auc,
    }
    model_path = Path("/tmp/btc_model_v7.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Model saved: %.1f MB", model_path.stat().st_size / 1e6)

    # ── Gate ──────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("  Champion (%s) purged WF AUC: %.4f",
             champion.get("version","?"), fair_champ_auc)
    log.info("  v7 candidate:  AUC=%.4f  Acc=%.4f  Brier=%.4f  Ensemble=%s",
             final_auc, final_acc, final_brier, use_ens_final)

    beats_auc   = final_auc   > fair_champ_auc
    beats_brier = final_brier < CHAMPION_BRIER
    beats_acc   = final_acc   > CHAMPION_ACC
    n_passed    = sum([beats_auc, beats_brier, beats_acc])
    auc_close   = final_auc >= (fair_champ_auc - 0.005)
    should_promote = beats_auc or (n_passed >= 2 and auc_close)

    log.info("  Gate: AUC>%.4f[%s] Brier<%.4f[%s] Acc>%.4f[%s] → %d/3 | %s",
             fair_champ_auc, "✓" if beats_auc   else "✗",
             CHAMPION_BRIER,  "✓" if beats_brier else "✗",
             CHAMPION_ACC,    "✓" if beats_acc   else "✗",
             n_passed, "PROMOTE" if should_promote else "REJECT")

    promoted = False
    if should_promote:
        log.info("Promoting v7 to HF champion...")
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="champion.pkl",
            repo_id=HF_MODEL_REPO, repo_type="model",
            commit_message=(f"Champion v7: AUC={final_auc:.4f} Brier={final_brier:.4f} "
                            f"Acc={final_acc:.4f} ensemble={use_ens_final}"),
        )
        meta_out = {
            "version": "v7",
            "feature_list": good_features,
            "features": len(good_features),
            "wf_auc": final_auc, "wf_acc": final_acc, "wf_brier": final_brier,
            "fold_aucs": final_wf["fold_aucs"],
            "n_samples": len(df),
            "ensemble": use_ens_final,
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "champion_compared_auc": fair_champ_auc,
            "notes": ("v7: Up/Down OB split, realized vol clustering, up_ratio anomaly, "
                      "lag index fix, inslot_ret fix, champion AUC from HF, ensemble"),
            "best_params": best_params,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
            json.dump(meta_out, fp, indent=2)
        api.upload_file(
            path_or_fileobj=fp.name,
            path_in_repo="champion_meta.json",
            repo_id=HF_MODEL_REPO, repo_type="model",
            commit_message=f"Champion v7 meta AUC={final_auc:.4f}",
        )
        log.info("Champion v7 promoted: https://huggingface.co/%s", HF_MODEL_REPO)
        promoted = True
    else:
        log.warning("v7 not promoted — gates not passed.")

    log.info("Done.")
    return {
        "wf_auc_baseline":  wf_base["wf_auc"],
        "wf_auc_lgb":       wf_lgb["wf_auc"],
        "wf_auc_ensemble":  wf_ens["wf_auc"],
        "wf_auc_final":     final_auc,
        "wf_acc_final":     final_acc,
        "wf_brier_final":   final_brier,
        "champion_fair_auc": fair_champ_auc,
        "n_samples":        len(df),
        "n_features_final": len(good_features),
        "n_features_dropped": dropped,
        "ensemble":         use_ens_final,
        "promoted":         promoted,
    }


@app.local_entrypoint()
def main():
    print("Submitting BTC v7 training job to Modal...")
    r = train_v7.remote()
    print(f"\n{'='*55}")
    print("TRAINING COMPLETE — v7")
    print(f"  Baseline AUC:     {r['wf_auc_baseline']:.4f}")
    print(f"  LightGBM AUC:     {r['wf_auc_lgb']:.4f}")
    print(f"  Ensemble AUC:     {r['wf_auc_ensemble']:.4f}")
    print(f"  Final AUC:        {r['wf_auc_final']:.4f}")
    print(f"  Final Acc:        {r['wf_acc_final']:.4f}")
    print(f"  Final Brier:      {r['wf_brier_final']:.4f}")
    print(f"  Champion AUC:     {r['champion_fair_auc']:.4f} (purged WF)")
    print(f"  Samples:          {r['n_samples']}")
    print(f"  Features:         {r['n_features_final']} (dropped {r['n_features_dropped']})")
    print(f"  Ensemble:         {r['ensemble']}")
    print(f"  Promoted:         {'YES ✓' if r['promoted'] else 'NO'}")
    print(f"{'='*55}")
