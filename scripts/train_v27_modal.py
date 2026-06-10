"""
train_v27_modal.py — BTC 5min model v27 (REAL-TIME ONLY + REALISTIC BACKTEST)
================================================================================
Philosophy: Only use features computable in real-time with <5s lag.
  - Binance spot (WebSocket, <2s lag): price returns, volatility, distance to round numbers
  - L2 Orderbook (CLOB REST /book, <5s lag): imbalance, spread, depth, drift
  - Temporal: hour/dow cyclical encoding
  - Lag history (from ring buffer, previous resolved slots)
  - Cross-interaction features between OB and spot
  
EXCLUDED (by design):
  - Tick-based features (data-api ~120s lag) — these are the btc_up_ratio, btc_n_ticks etc.
    They rely on stale data and create train/live mismatch when OBSERVE_SECS=60.
  - CLOB WS features — no historical training data (0% coverage in v25 training).
    Will be re-added when sufficient live-logged data exists.

Pipeline:
  1. Download data from HF (markets, ticks for lag history, spot, OB features)
  2. Build features using ONLY real-time sources
  3. Feature importance screening → select top N
  4. Optuna hyperparameter tuning (150 trials)
  5. Walk-forward evaluation (5-fold TimeSeriesSplit)
  6. Realistic P&L backtest with Polymarket fees
  7. Promote if beats champion
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

vol = modal.Volume.from_name("btc-training-cache", create_if_missing=True)
app = modal.App("btc-v27-run", image=image)


@app.function(
    cpu=8,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/cache": vol},
)
def train_v27():
    import gc, json, logging, math, os, pickle, sys, time, warnings
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
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    HF_TOKEN      = os.environ.get("HF_TOKEN", "")
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
    SLOT_DURATION = 300
    OPTUNA_TRIALS = 150
    WF_GAP        = 5
    N_SPLITS      = 5
    DATA_DIR      = Path("/tmp/btc_data")
    DATA_DIR.mkdir(exist_ok=True)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 0: DOWNLOAD DATA
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 0: Downloading training data from HuggingFace...")
    import shutil

    DATA_FILES = {
        "all_markets.csv": "data/all_markets.csv",
        "new_markets.csv": "data/new_markets.csv",
        "ticks_btc_full_clean.parquet": "data/ticks_btc_full_clean.parquet",
        "new_ticks_pmdata.parquet": "data/new_ticks_pmdata.parquet",
        "binance_spot_full.parquet": "data/binance_spot_full.parquet",
        "binance_spot_local.parquet": "data/binance_spot_local.parquet",
        "ob_features_full.parquet": "data/ob_features_full.parquet",
    }
    for local_name, hf_path in DATA_FILES.items():
        local_path = DATA_DIR / local_name
        if local_path.exists():
            log.info("  %s already cached", local_name)
            continue
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO, filename=hf_path,
                token=HF_TOKEN, repo_type="model",
                local_dir=str(DATA_DIR), local_dir_use_symlinks=False,
            )
            src = DATA_DIR / hf_path
            if src.exists() and not local_path.exists():
                shutil.move(str(src), str(local_path))
            log.info("  Downloaded %s (%.1fMB)", local_name, local_path.stat().st_size / 1e6)
        except Exception as e:
            log.warning("  Could not download %s: %s (may be optional)", local_name, e)

    # Champion metrics
    champion = {"version": "v25-v25_60s_30f", "wf_auc": 0.8575, "wf_brier": 0.1539, "wf_acc": 0.7739}
    try:
        meta_path = hf_hub_download(
            repo_id=HF_MODEL_REPO, filename="champion_meta.json",
            token=HF_TOKEN, repo_type="model"
        )
        with open(meta_path) as f:
            champion = json.load(f)
        log.info("Champion: %s AUC=%.4f Brier=%.4f Acc=%.4f",
                 champion["version"], champion["wf_auc"], champion["wf_brier"], champion["wf_acc"])
    except Exception as e:
        log.warning("Could not load champion meta: %s — using defaults", e)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD MARKETS
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 1: Loading markets...")

    markets = pd.read_csv(DATA_DIR / "all_markets.csv")
    markets["market_id"] = markets["market_id"].astype(str)
    markets["slot_ts"]   = markets["slot_ts"].astype(int)

    # Merge expansion markets
    new_mkt_path = DATA_DIR / "new_markets.csv"
    if new_mkt_path.exists():
        new_mkts = pd.read_csv(new_mkt_path)
        new_mkts["market_id"] = new_mkts["market_id"].astype(str)
        new_mkts["slot_ts"]   = new_mkts["slot_ts"].astype(int)
        if "target" in new_mkts.columns:
            existing_ids = set(markets["market_id"])
            truly_new = new_mkts[~new_mkts["market_id"].isin(existing_ids)]
            if len(truly_new) > 0:
                markets = pd.concat([markets, truly_new[markets.columns]], ignore_index=True)
                log.info("  Added %d expansion markets", len(truly_new))

    markets = markets.sort_values("slot_ts").reset_index(drop=True)
    log.info("Total markets: %d (%.1f%% UP)", len(markets), 100 * markets["target"].mean())

    markets["rank"] = range(len(markets))
    slot_to_rank  = dict(zip(markets["slot_ts"], markets["rank"]))
    all_targets   = markets["target"].values
    all_slot_ts   = markets["slot_ts"].values
    all_mids      = markets["market_id"].values

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: LOAD OB FEATURES
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 2: Loading L2 orderbook features...")
    ob_path = DATA_DIR / "ob_features_full.parquet"
    if not ob_path.exists():
        raise RuntimeError("ob_features_full.parquet not found!")

    ob_df = pd.read_parquet(str(ob_path))
    ob_df["market_id"] = ob_df["market_id"].astype(str)
    ob_by_market = ob_df.set_index("market_id").to_dict("index")
    ob_cols = [c for c in ob_df.columns if c != "market_id"]
    log.info("OB features: %d markets, %d features: %s", len(ob_df), len(ob_cols), ob_cols[:5])

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: LOAD BINANCE SPOT
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 3: Loading Binance spot...")
    spot_dfs = []
    for sp in ["binance_spot_full.parquet", "binance_spot_local.parquet"]:
        sp_path = DATA_DIR / sp
        if sp_path.exists():
            spot_dfs.append(pd.read_parquet(str(sp_path)))
    spot_df = pd.concat(spot_dfs, ignore_index=True) if spot_dfs else pd.DataFrame()
    spot_df = spot_df.sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
    spot_ts_arr = (spot_df["timestamp_ms"].values // 1000).astype(np.int64)
    spot_px_arr = spot_df["close"].values.astype(np.float64)
    spot_vol_arr = spot_df["volume"].values.astype(np.float64) if "volume" in spot_df.columns else np.zeros(len(spot_df))
    spot_hi_arr = spot_df["high"].values.astype(np.float64) if "high" in spot_df.columns else spot_px_arr.copy()
    spot_lo_arr = spot_df["low"].values.astype(np.float64) if "low" in spot_df.columns else spot_px_arr.copy()
    log.info("Binance spot: %d candles (%.0f days)", len(spot_ts_arr), (spot_ts_arr[-1] - spot_ts_arr[0]) / 86400)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: LOAD TICKS (for lag history only)
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 4: Loading ticks (for lag features only)...")
    all_mids_set = set(markets["market_id"].tolist())
    tick_cols = ["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"]

    pf = pq.ParquetFile(str(DATA_DIR / "ticks_btc_full_clean.parquet"))
    chunks = []
    for rg_i in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg_i, columns=tick_cols).to_pandas()
        chunk["market_id"] = chunk["market_id"].astype(str)
        chunk = chunk[chunk["market_id"].isin(all_mids_set)]
        if len(chunk):
            chunks.append(chunk)

    new_tick_path = DATA_DIR / "new_ticks_pmdata.parquet"
    if new_tick_path.exists():
        new_ticks = pd.read_parquet(str(new_tick_path))
        new_ticks["market_id"] = new_ticks["market_id"].astype(str)
        new_ticks = new_ticks[new_ticks["market_id"].isin(all_mids_set)]
        if len(new_ticks) > 0:
            for col in tick_cols:
                if col not in new_ticks.columns:
                    new_ticks[col] = 0
            chunks.append(new_ticks[tick_cols])

    gc.collect()
    btc = pd.concat(chunks, ignore_index=True)
    btc = btc[btc["timestamp_ms"] > 0]

    slot_ts_map = dict(zip(markets["market_id"], markets["slot_ts"]))
    btc["slot_ts_val"] = btc["market_id"].map(slot_ts_map).astype(float)
    btc["t_sec"] = btc["timestamp_ms"].astype(float) / 1000 - btc["slot_ts_val"]
    btc = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < 300)]
    log.info("Total ticks: %d for %d markets", len(btc), btc["market_id"].nunique())

    # Compute per-slot aggregates for lag features
    OBS_SECS = 60
    filtered = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < OBS_SECS)]
    up_f = filtered[filtered["outcome"] == "Up"]
    dn_f = filtered[filtered["outcome"] == "Down"]
    slot_vol_up  = up_f.groupby("market_id")["size_usdc"].sum()
    slot_vol_dn  = dn_f.groupby("market_id")["size_usdc"].sum()
    slot_vol_tot = slot_vol_up.add(slot_vol_dn, fill_value=0)
    slot_up_ratio = slot_vol_up / slot_vol_tot.clip(lower=1e-9)
    slot_nticks   = filtered.groupby("market_id").size()

    del btc, chunks, filtered, up_f, dn_f
    gc.collect()
    log.info("Lag data computed. Memory freed.")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: BUILD FEATURES (REAL-TIME ONLY)
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 5: Building REAL-TIME features for %d markets...", len(markets))

    def spot_at(ts_s):
        idx = np.searchsorted(spot_ts_arr, ts_s, side="right") - 1
        idx = max(0, min(idx, len(spot_px_arr) - 1))
        return float(spot_px_arr[idx])

    def spot_vol_at(ts_start, ts_end):
        """Sum of Binance candle volumes in [ts_start, ts_end)."""
        i0 = int(np.searchsorted(spot_ts_arr, ts_start, side="left"))
        i1 = int(np.searchsorted(spot_ts_arr, ts_end, side="right"))
        if i1 > i0:
            return float(spot_vol_arr[i0:i1].sum())
        return 0.0

    def spot_volatility(ts_start, ts_end):
        """Std of 1-min returns in [ts_start, ts_end)."""
        i0 = int(np.searchsorted(spot_ts_arr, ts_start, side="left"))
        i1 = int(np.searchsorted(spot_ts_arr, ts_end, side="right"))
        if i1 - i0 >= 3:
            px = spot_px_arr[i0:i1]
            rets = np.diff(px) / px[:-1]
            return float(np.std(rets))
        return 0.0

    rows = []
    skipped_no_ob = 0

    for rank_i, row in markets.iterrows():
        mid     = row["market_id"]
        slot_ts = int(row["slot_ts"])
        target  = int(row["target"])
        ext_rank = slot_to_rank.get(slot_ts, rank_i)

        feat = {}

        # ── A. SPOT FEATURES (Binance, <2s lag) ──────────────────────
        obs_end_ts = slot_ts + OBS_SECS  # t=60s
        px_now = spot_at(obs_end_ts)

        def pre_ret(h_sec):
            px_h = spot_at(slot_ts - h_sec)
            return (px_now / px_h - 1) if px_h > 0 else 0.0

        feat["btc_pre_5m_ret"]  = pre_ret(300)
        feat["btc_pre_15m_ret"] = pre_ret(900)
        feat["btc_pre_30m_ret"] = pre_ret(1800)
        feat["btc_pre_1h_ret"]  = pre_ret(3600)
        feat["btc_pre_4h_ret"]  = pre_ret(14400)

        # In-slot return during [slot_ts, slot_ts+60s]
        t0_idx = int(np.searchsorted(spot_ts_arr, slot_ts, side="left"))
        t1_idx = int(np.searchsorted(spot_ts_arr, obs_end_ts, side="right"))
        if t1_idx > t0_idx:
            inslot_px = spot_px_arr[t0_idx:t1_idx]
            feat["btc_inslot_ret"] = float(inslot_px[-1] / inslot_px[0] - 1) if inslot_px[0] > 0 else 0.0
            inslot_hi = spot_hi_arr[t0_idx:t1_idx]
            inslot_lo = spot_lo_arr[t0_idx:t1_idx]
            feat["btc_inslot_range"] = float((inslot_hi.max() - inslot_lo.min()) / px_now) if px_now > 0 else 0.0
        else:
            feat["btc_inslot_ret"] = 0.0
            feat["btc_inslot_range"] = 0.0

        # Volatility features
        feat["btc_vol_1h"]  = spot_volatility(slot_ts - 3600, slot_ts)
        feat["btc_vol_4h"]  = spot_volatility(slot_ts - 14400, slot_ts)
        feat["btc_vol_inslot"] = spot_volatility(slot_ts, obs_end_ts)

        # Momentum consistency
        px_1h  = spot_at(slot_ts - 3600)
        px_4h  = spot_at(slot_ts - 14400)
        if px_now > 0 and px_1h > 0 and px_4h > 0 and abs(px_now - px_4h) > 1:
            feat["btc_pre_1h_4h_ratio"] = (px_now - px_1h) / (px_now - px_4h + 1e-9)
        else:
            feat["btc_pre_1h_4h_ratio"] = 0.0

        # Distance to round numbers
        px_k = px_now / 1000
        feat["btc_dist_1k"] = min(px_k - math.floor(px_k), math.ceil(px_k) - px_k)

        # Volume features
        feat["btc_spot_vol_ratio"] = (
            spot_vol_at(slot_ts - 300, slot_ts) / (spot_vol_at(slot_ts - 3600, slot_ts - 300) / 11 + 1e-9)
        )

        # ── B. L2 ORDERBOOK FEATURES (<5s lag) ───────────────────────
        ob = ob_by_market.get(mid)
        if ob is not None:
            for col in ob_cols:
                # Preserve prefix for ob_* and clob_* columns; add ob_ to others
                key = col if (col.startswith("ob_") or col.startswith("clob_")) else f"ob_{col}"
                feat[key] = float(ob.get(col, 0.0))

            # Cross-features (OB x Spot)
            feat["x_imb_x_inslot"] = float(ob.get("ob_imbalance", 0)) * feat["btc_inslot_ret"]
            feat["x_drift_x_ret5m"] = float(ob.get("ob_mid_drift", 0)) * feat["btc_pre_5m_ret"]
            feat["x_spread_x_vol"] = float(ob.get("ob_spread", 0)) * feat["btc_vol_1h"]
            feat["x_imb_end_x_ret"] = float(ob.get("ob_imbalance_end", 0)) * feat["btc_inslot_ret"]
            feat["x_depth_x_vol"] = float(ob.get("ob_depth_ratio", 1)) * feat["btc_vol_1h"]
        else:
            skipped_no_ob += 1
            for col in ob_cols:
                key = col if (col.startswith("ob_") or col.startswith("clob_")) else f"ob_{col}"
                if "ratio" in col or "imbalance" in col or "imb" in col:
                    feat[key] = 0.0
                elif "spread" in col:
                    feat[key] = 0.02
                elif "depth" in col and "5c" in col:
                    feat[key] = 0.5
                elif "mid" in col and "drift" not in col:
                    feat[key] = 0.5
                else:
                    feat[key] = 0.0
            feat["x_imb_x_inslot"] = 0.0
            feat["x_drift_x_ret5m"] = 0.0
            feat["x_spread_x_vol"] = 0.0
            feat["x_imb_end_x_ret"] = 0.0
            feat["x_depth_x_vol"] = 0.0

        # ── C. LAG FEATURES (from ring buffer) ───────────────────────
        lag_streak = 0
        streak_dir = None

        for lag_n in range(1, 6):
            prev_rank = ext_rank - lag_n
            if prev_rank >= 0:
                prev_target = int(all_targets[prev_rank])
                prev_slot   = int(all_slot_ts[prev_rank])
                prev_mid    = all_mids[prev_rank]

                time_gap = slot_ts - prev_slot
                if time_gap > lag_n * SLOT_DURATION * 3:
                    feat[f"lag_{lag_n}_outcome"]       = 0.5
                    feat[f"prev_slot_up_ratio_{lag_n}"] = 0.5
                    feat[f"prev_slot_n_ticks_{lag_n}"]  = 0.0
                    feat[f"prev_slot_vol_{lag_n}"]      = 0.0
                    continue

                feat[f"lag_{lag_n}_outcome"]        = float(prev_target)
                feat[f"prev_slot_up_ratio_{lag_n}"] = float(slot_up_ratio.get(prev_mid, 0.5))
                feat[f"prev_slot_n_ticks_{lag_n}"]  = float(slot_nticks.get(prev_mid, 0.0))
                feat[f"prev_slot_vol_{lag_n}"]       = float(slot_vol_tot.get(prev_mid, 0.0))

                if lag_n == 1:
                    streak_dir = prev_target; lag_streak = 1
                elif prev_target == streak_dir:
                    lag_streak += 1
            else:
                for k in [f"lag_{lag_n}_outcome", f"prev_slot_up_ratio_{lag_n}",
                          f"prev_slot_n_ticks_{lag_n}", f"prev_slot_vol_{lag_n}"]:
                    feat[k] = 0.0

        feat["lag_streak"] = float(lag_streak)

        # Z-scores from lag history
        def _hist_ur(lookback=20):
            vals = []
            for d in range(1, lookback + 1):
                pr = ext_rank - d
                if pr < 0:
                    break
                pm = all_mids[pr]
                v = slot_up_ratio.get(pm, None)
                if v is not None:
                    vals.append(v)
            return vals

        hist_vals = _hist_ur(20)
        if len(hist_vals) >= 3:
            mu20 = np.mean(hist_vals); sd20 = max(np.std(hist_vals), 0.01)
            # Use prev_slot_up_ratio_1 as "current" for z-score
            cur_ur = feat.get("prev_slot_up_ratio_1", 0.5)
            feat["lag_ur_zscore_20"] = np.clip((cur_ur - mu20) / sd20, -5, 5)
        else:
            feat["lag_ur_zscore_20"] = 0.0

        hist5 = hist_vals[:5]
        if len(hist5) >= 2:
            mu5 = np.mean(hist5); sd5 = max(np.std(hist5), 0.01)
            cur_ur = feat.get("prev_slot_up_ratio_1", 0.5)
            feat["lag_ur_zscore_5"] = np.clip((cur_ur - mu5) / sd5, -5, 5)
        else:
            feat["lag_ur_zscore_5"] = 0.0

        # ── D. TEMPORAL FEATURES ─────────────────────────────────────
        dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        dow  = dt.weekday()

        feat["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        feat["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        feat["dow_sin"]  = math.sin(2 * math.pi * dow / 7)
        feat["dow_cos"]  = math.cos(2 * math.pi * dow / 7)

        feat["target"] = target
        rows.append(feat)

    df = pd.DataFrame(rows)
    FEATURE_COLS = [c for c in df.columns if c != "target"]
    log.info("Feature matrix: %d rows x %d features (no_ob=%d)", len(df), len(FEATURE_COLS), skipped_no_ob)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["target"].values.astype(int)
    log.info("Class balance: %d UP, %d DOWN (%.1f%% UP)", y.sum(), (y == 0).sum(), 100 * y.mean())

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: FEATURE SELECTION
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 6: Feature importance screening...")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP)
    screen = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=30, random_state=42, verbose=-1
    )
    feat_importances = np.zeros(len(FEATURE_COLS))
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        screen.fit(X[tr_idx], y[tr_idx])
        feat_importances += screen.feature_importances_
    feat_importances /= N_SPLITS

    feat_rank = np.argsort(feat_importances)[::-1]

    # Log ALL features by importance
    log.info("Feature importance ranking (ALL %d features):", len(FEATURE_COLS))
    for i, idx in enumerate(feat_rank):
        log.info("  %3d. %-35s importance=%.1f", i+1, FEATURE_COLS[idx], feat_importances[idx])

    # Test multiple feature counts
    FEATURE_COUNTS = [40, 30, 25, 20, 15]
    best_combo = None
    best_auc = -1

    for n_feats in FEATURE_COUNTS:
        top_n = [FEATURE_COLS[i] for i in feat_rank[:n_feats]]
        X_n = df[top_n].values.astype(np.float32)

        aucs = []
        for tr_idx, val_idx in TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP).split(X_n):
            m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                    num_leaves=20, min_child_samples=30, random_state=42, verbose=-1)
            m.fit(X_n[tr_idx], y[tr_idx])
            p = m.predict_proba(X_n[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], p))
        mean_auc = np.mean(aucs)
        log.info("  %2d features: AUC=%.4f (folds: %s)", n_feats, mean_auc, [f"{a:.4f}" for a in aucs])

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_combo = (n_feats, top_n)

    TOP_N_FEATS = best_combo[0]
    top_features = best_combo[1]
    log.info("Selected: %d features (AUC=%.4f)", TOP_N_FEATS, best_auc)
    log.info("Top features: %s", top_features[:10])

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: OPTUNA TUNING
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 7: Optuna tuning (%d trials)...", OPTUNA_TRIALS)

    X_top = df[top_features].values.astype(np.float32)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "num_leaves":        trial.suggest_int("num_leaves", 8, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "random_state": 42, "verbose": -1,
        }
        aucs = []
        for tr_idx, val_idx in TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP).split(X_top):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_top[tr_idx], y[tr_idx])
            p = m.predict_proba(X_top[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], p))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=4, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"random_state": 42, "verbose": -1})
    log.info("Best Optuna AUC=%.4f, params=%s", study.best_value, best_params)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 8: WALK-FORWARD EVALUATION + CALIBRATION
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 8: Walk-forward evaluation with calibration...")

    wf_aucs, wf_briers, wf_accs = [], [], []
    fold_preds = []  # Store for backtest

    for fold, (tr_idx, val_idx) in enumerate(
        TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP).split(X_top)
    ):
        base = lgb.LGBMClassifier(**best_params)
        cal  = CalibratedClassifierCV(base, cv=3, method="isotonic")
        cal.fit(X_top[tr_idx], y[tr_idx])
        p = cal.predict_proba(X_top[val_idx])[:, 1]

        auc = roc_auc_score(y[val_idx], p)
        brier = brier_score_loss(y[val_idx], p)
        acc = (p.round() == y[val_idx]).mean()
        wf_aucs.append(auc)
        wf_briers.append(brier)
        wf_accs.append(acc)

        for vi, pi in zip(val_idx, p):
            fold_preds.append({"idx": vi, "prob_up": float(pi), "target": int(y[vi]), "fold": fold})

        log.info("  Fold %d: AUC=%.4f Brier=%.4f Acc=%.4f (n=%d)", fold, auc, brier, acc, len(val_idx))

    mean_auc   = float(np.mean(wf_aucs))
    mean_brier = float(np.mean(wf_briers))
    mean_acc   = float(np.mean(wf_accs))
    log.info("WALK-FORWARD: AUC=%.4f Brier=%.4f Acc=%.4f", mean_auc, mean_brier, mean_acc)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 9: REALISTIC P&L BACKTEST
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 9: Realistic P&L backtest...")

    # Polymarket fee structure
    FEE_RATE = 0.02  # 2% on net profit (only charged when you win)
    MIN_CONF = 0.55   # minimum model confidence to trade
    MIN_EDGE = 0.03   # minimum edge (model_prob - market_price)
    # Market ask is approximated from ob_mid (the real polymarket price)
    # ob_mid is the midpoint of the order book, close to the market price

    preds_df = pd.DataFrame(fold_preds)
    preds_df = preds_df.sort_values("idx")
    # Attach ob features from the feature matrix
    # ob_mid is the OPEN snapshot (~t=0-30s), ob_mid + ob_mid_drift = close snapshot (~t=150-300s)
    # In live, bot buys at t=170-240s, so the close_mid is the realistic ask price
    preds_df["ob_mid"] = [df.iloc[int(i)].get("ob_mid", 0.5) for i in preds_df["idx"]]
    preds_df["ob_mid_drift"] = [df.iloc[int(i)].get("ob_mid_drift", 0.0) for i in preds_df["idx"]]
    preds_df["close_mid"] = preds_df["ob_mid"] + preds_df["ob_mid_drift"]  # realistic entry price

    # Simulate trading
    total_trades = 0
    wins = 0
    losses = 0
    total_pnl = 0.0
    total_fees = 0.0
    total_staked = 0.0
    pnl_by_conf = {}  # Track P&L by confidence bucket
    pnl_by_edge = {}  # Track P&L by edge bucket

    for _, pred in preds_df.iterrows():
        prob = pred["prob_up"]
        target = pred["target"]
        ob_mid_val = pred["ob_mid"]
        close_mid_val = pred["close_mid"]
        # Clamp close_mid to reasonable range
        close_mid_val = max(0.05, min(0.95, close_mid_val))

        # Use CLOSE_MID as realistic entry price (bot buys at t=170-240s)
        # ob_mid is the price of the UP token on Polymarket
        if prob > 0.5:
            direction = "UP"
            confidence = prob
            market_ask = close_mid_val  # price to buy UP token at entry time
        else:
            direction = "DOWN"
            confidence = 1 - prob
            market_ask = 1 - close_mid_val  # price to buy DOWN token at entry time

        # Edge = model confidence - market price
        edge = confidence - market_ask

        # Apply trading filters
        if confidence < MIN_CONF:
            continue
        if edge < MIN_EDGE:
            continue
        if market_ask < 0.30 or market_ask > 0.70:
            continue

        # Simulate trade with $10 per trade
        stake = 10.0
        shares = stake / market_ask

        # Did we win?
        won = (direction == "UP" and target == 1) or (direction == "DOWN" and target == 0)

        if won:
            gross_profit = shares * 1.0 - stake  # shares * $1 payout - cost
            fee = gross_profit * FEE_RATE
            net_pnl = gross_profit - fee
            total_fees += fee
            wins += 1
        else:
            net_pnl = -stake
            losses += 1

        total_pnl += net_pnl
        total_staked += stake
        total_trades += 1

        # Track by confidence bucket
        bucket = f"{int(confidence * 100)}%"
        if bucket not in pnl_by_conf:
            pnl_by_conf[bucket] = {"trades": 0, "wins": 0, "pnl": 0.0}
        pnl_by_conf[bucket]["trades"] += 1
        pnl_by_conf[bucket]["wins"] += int(won)
        pnl_by_conf[bucket]["pnl"] += net_pnl

        # Track by edge bucket
        edge_bucket = f"{int(edge * 100)}%"
        if edge_bucket not in pnl_by_edge:
            pnl_by_edge[edge_bucket] = {"trades": 0, "wins": 0, "pnl": 0.0}
        pnl_by_edge[edge_bucket]["trades"] += 1
        pnl_by_edge[edge_bucket]["wins"] += int(won)
        pnl_by_edge[edge_bucket]["pnl"] += net_pnl

    wr = wins / total_trades * 100 if total_trades > 0 else 0
    log.info("BACKTEST RESULTS:")
    log.info("  Trades: %d (W:%d / L:%d = %.1f%% WR)", total_trades, wins, losses, wr)
    log.info("  P&L: $%.2f (fees: $%.2f, staked: $%.2f)", total_pnl, total_fees, total_staked)
    log.info("  Avg P&L per trade: $%.3f", total_pnl / total_trades if total_trades > 0 else 0)
    log.info("  ROI: %.1f%%", (total_pnl / total_staked) * 100 if total_staked > 0 else 0)

    log.info("  BY CONFIDENCE BUCKET:")
    for bucket in sorted(pnl_by_conf.keys()):
        b = pnl_by_conf[bucket]
        bwr = b["wins"] / b["trades"] * 100 if b["trades"] > 0 else 0
        log.info("    %s: %d trades, %.1f%% WR, P&L=$%.2f", bucket, b["trades"], bwr, b["pnl"])

    log.info("  BY EDGE BUCKET:")
    for bucket in sorted(pnl_by_edge.keys()):
        b = pnl_by_edge[bucket]
        bwr = b["wins"] / b["trades"] * 100 if b["trades"] > 0 else 0
        log.info("    %s edge: %d trades, %.1f%% WR, P&L=$%.2f", bucket, b["trades"], bwr, b["pnl"])

    # ══════════════════════════════════════════════════════════════════════
    # STEP 10: SANITY CHECK + PROMOTE
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 10: Sanity check + promotion decision...")

    # Train final model on ALL data
    final_base  = lgb.LGBMClassifier(**best_params)
    final_model = CalibratedClassifierCV(final_base, cv=3, method="isotonic")
    final_model.fit(X_top, y)

    # Sanity: UP signal → high prob, DOWN signal → low prob
    def _neutral_value(fname):
        if "dist_1k" in fname: return 0.25
        if "ticks" in fname or "count" in fname: return 100.0
        if "ratio" in fname and "depth" not in fname and "1h_4h" not in fname: return 0.5
        if "vwap" in fname: return 0.5
        if "ob_mid" in fname and "drift" not in fname: return 0.5
        if "ob_spread" in fname: return 0.02
        if "ob_depth_5c" in fname: return 0.5
        if "ob_total_depth" in fname: return 1000.0
        if "depth_ratio" in fname: return 1.0
        return 0.0

    baseline_feats = {f: _neutral_value(f) for f in top_features}
    up_feats = dict(baseline_feats)
    up_feats.update({k: v for k, v in {
        "btc_inslot_ret": 0.002, "btc_pre_5m_ret": 0.001, "btc_pre_1h_ret": 0.003,
        "ob_imbalance": 0.3, "ob_imbalance_end": 0.3, "ob_mid_drift": 0.02,
        "ob_depth_ratio": 1.3, "ob_fill_imbalance": 0.2, "ob_imb_momentum": 0.1,
    }.items() if k in up_feats})

    down_feats = dict(baseline_feats)
    down_feats.update({k: v for k, v in {
        "btc_inslot_ret": -0.002, "btc_pre_5m_ret": -0.001, "btc_pre_1h_ret": -0.003,
        "ob_imbalance": -0.3, "ob_imbalance_end": -0.3, "ob_mid_drift": -0.02,
        "ob_depth_ratio": 0.7, "ob_fill_imbalance": -0.2, "ob_imb_momentum": -0.1,
    }.items() if k in down_feats})

    up_arr    = pd.DataFrame([up_feats])[top_features].values.astype(np.float32)
    neut_arr  = pd.DataFrame([baseline_feats])[top_features].values.astype(np.float32)
    down_arr  = pd.DataFrame([down_feats])[top_features].values.astype(np.float32)
    prob_up   = final_model.predict_proba(up_arr)[0, 1]
    prob_neut = final_model.predict_proba(neut_arr)[0, 1]
    prob_down = final_model.predict_proba(down_arr)[0, 1]
    log.info("Sanity: UP -> %.3f | Neutral -> %.3f | DOWN -> %.3f", prob_up, prob_neut, prob_down)
    sanity_ok = prob_up > prob_neut > prob_down
    if not sanity_ok:
        log.error("SANITY CHECK FAILED! UP=%.3f Neutral=%.3f DOWN=%.3f", prob_up, prob_neut, prob_down)

    # Promotion decision
    beats_auc   = mean_auc   > champion["wf_auc"]
    beats_brier = mean_brier < champion["wf_brier"]
    beats_acc   = mean_acc   > champion["wf_acc"]
    score = sum([beats_auc, beats_brier, beats_acc])

    log.info("vs Champion (%s): AUC %s (%.4f vs %.4f) | Brier %s (%.4f vs %.4f) | Acc %s (%.4f vs %.4f) -> %d/3",
             champion["version"],
             "Y" if beats_auc else "N", mean_auc, champion["wf_auc"],
             "Y" if beats_brier else "N", mean_brier, champion["wf_brier"],
             "Y" if beats_acc else "N", mean_acc, champion["wf_acc"],
             score)

    version_tag = f"v27_{TOP_N_FEATS}f_rt"

    if (score >= 1 or mean_auc > 0.845) and sanity_ok:
        log.info("PROMOTING %s! (%d/3 metrics, backtest P&L=$%.2f)", version_tag, score, total_pnl)

        model_data = {
            "version":  version_tag,
            "features": top_features,
            "model":    final_model,
            "wf_auc":   mean_auc,
            "wf_brier": mean_brier,
            "wf_acc":   mean_acc,
            "obs_secs": OBS_SECS,
        }
        meta = {
            "version":   version_tag,
            "wf_auc":    mean_auc,
            "wf_brier":  mean_brier,
            "wf_acc":    mean_acc,
            "features":  top_features,
            "n_samples": len(y),
            "n_features": len(top_features),
            "obs_secs":  OBS_SECS,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backtest": {
                "trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": wr,
                "pnl": total_pnl,
                "fees": total_fees,
                "roi_pct": (total_pnl / total_staked) * 100 if total_staked > 0 else 0,
                "by_confidence": pnl_by_conf,
                "by_edge": pnl_by_edge,
            },
            "changes": (
                f"v27: REAL-TIME features only (no tick-based features with 120s lag). "
                f"{len(top_features)} features from Binance spot + L2 OB + lag history. "
                f"Rigorous feature pruning: tested {FEATURE_COUNTS}. "
                f"Realistic backtest: {total_trades} trades, {wr:.1f}% WR, P&L=${total_pnl:.2f}. "
                f"OBS_SECS={OBS_SECS}."
            ),
        }

        import tempfile
        api = HfApi(token=HF_TOKEN)
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path  = Path(tmpdir) / "champion.pkl"
            meta_path = Path(tmpdir) / "champion_meta.json"
            with open(pkl_path, "wb") as f:
                pickle.dump(model_data, f)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            api.upload_file(path_or_fileobj=str(pkl_path), path_in_repo="champion.pkl",
                            repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN)
            api.upload_file(path_or_fileobj=str(meta_path), path_in_repo="champion_meta.json",
                            repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN)

        log.info("%s promoted to HF! AUC=%.4f Brier=%.4f Acc=%.4f (%d features, %ds obs)",
                 version_tag, mean_auc, mean_brier, mean_acc, len(top_features), OBS_SECS)
    else:
        reasons = []
        if score < 1 and mean_auc <= 0.845: reasons.append(f"AUC too low ({mean_auc:.4f})")
        if not sanity_ok: reasons.append("sanity check failed")
        log.info("NOT PROMOTED: %s. Reasons: %s", version_tag, "; ".join(reasons))

    log.info("=" * 70)
    log.info("v27 training complete.")


@app.local_entrypoint()
def main():
    train_v27.remote()
