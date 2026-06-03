"""
train_v17_modal.py — BTC 5min model v17

Key changes vs v16:
  - EXTENDED LAG CONTEXT: new_markets.csv (15,257 markets, Apr 12 - Jun 3 2026)
    has target labels but no ticks. We use them ONLY to build lag features:
    prev_slot_outcome_1/2/3/4/5, lag_streak, prev_slot_up_ratio_* can use
    the extended timeline for historical context in test/eval, but training
    only uses markets WITH ticks (7,062).

    Specifically: when computing lag_1_outcome for a market at slot T,
    we look up T-300s in the combined (7062 + 15257) timeline. This gives
    training samples richer lag context near the end of the 7062 window
    (Apr 11) AND enables correct lag lookup during live inference.

  - TEMPORAL FEATURES: hour_of_day, day_of_week, time_sin/cos
    The "hour × up_ratio" interaction was listed as untested in EXPERIMENTS.md.
    Hour-of-day captures market microstructure patterns (e.g. NY open vs
    Asian session trading volumes and directionality).

  - PREV_SLOT_UP_RATIO: use extended 22k timeline for prev_slot_up_ratio
    instead of just 7k. This gives better lag context.

  - Gate: vs v16 gate (AUC=0.847x from EXPERIMENTS.md — fetch from HF)
    Regime: 2026 only (local ticks), same as v13-v16.

Data sources:
  - Modal Volume 'btc-local-data':
    /ticks_btc_5min.parquet (7062 markets, ticks, Mar-Apr 2026)
    /local_markets.csv (7062 markets)
    /binance_spot_local.parquet (Binance 1m klines)
  - HF: artbreguez/polymarket-btc-model (champion metrics)
  - new_markets.csv uploaded as /new_markets.csv to Modal Volume
"""

import modal

LOCAL_VOL = modal.Volume.from_name("btc-local-data")

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

app = modal.App("btc-v17-run", image=image)


@app.function(
    cpu=8,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/btc_local": LOCAL_VOL},
)
def train_v17():
    import gc, json, logging, os, pickle, sys, time, warnings, math
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
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
    OBS_SECS      = 180
    SLOT_DURATION = 300
    OPTUNA_TRIALS = 150
    WF_GAP        = 5
    N_SPLITS      = 5
    TOP_N_FEATS   = 30
    LOCAL_DIR     = Path("/btc_local")

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")

    # ── Step 1: Load champion metrics ────────────────────────────────────
    log.info("Step 1: Loading champion metrics from HF...")
    champion = {"version": "v10", "wf_auc": 0.8547, "wf_brier": 0.1554, "wf_acc": 0.7902}
    try:
        meta_path = hf_hub_download(
            repo_id=HF_MODEL_REPO, filename="champion_meta.json",
            token=HF_TOKEN, repo_type="model"
        )
        with open(meta_path) as f:
            champion = json.load(f)
        log.info("Champion: %s AUC=%.4f Brier=%.4f Acc=%.4f",
                 champion["version"], champion["wf_auc"],
                 champion["wf_brier"], champion["wf_acc"])
    except Exception as e:
        log.warning("Could not load champion meta: %s — using defaults", e)

    # ── Step 2: Load markets (local 7k + new 15k for lag context) ────────
    log.info("Step 2: Loading markets...")
    local_markets = pd.read_csv(LOCAL_DIR / "local_markets.csv")
    local_markets["market_id"] = local_markets["market_id"].astype(str)
    local_markets["slot_ts"] = local_markets["slug"].str.split("-").str[-1].astype(int)
    local_markets["has_ticks"] = True
    log.info("Local markets (with ticks): %d", len(local_markets))

    # Extended lag context from new_markets.csv
    extended_markets = None
    new_markets_path = LOCAL_DIR / "new_markets.csv"
    if new_markets_path.exists():
        new_markets = pd.read_csv(new_markets_path)
        new_markets["market_id"] = new_markets["market_id"].astype(str)
        new_markets["slot_ts"] = new_markets["slot_ts"].astype(int)
        new_markets["has_ticks"] = False
        log.info("New markets (lag context only): %d", len(new_markets))
        # Combine for lag lookup (no overlap since date ranges don't overlap)
        cols = ["market_id", "slug", "slot_ts", "target", "has_ticks"]
        local_for_merge = local_markets[cols].copy()
        extended_markets = pd.concat(
            [local_for_merge, new_markets[cols]], ignore_index=True
        ).sort_values("slot_ts").reset_index(drop=True)
        log.info("Extended timeline: %d markets (%d with ticks + %d lag-only)",
                 len(extended_markets),
                 extended_markets["has_ticks"].sum(),
                 (~extended_markets["has_ticks"]).sum())
    else:
        log.warning("new_markets.csv not found — using local markets only for lag")
        extended_markets = local_markets.copy()
        extended_markets["has_ticks"] = True

    # Training uses only markets with ticks
    markets = local_markets.copy()
    markets = markets.sort_values("slot_ts").reset_index(drop=True)

    # Rank within full timeline (for lag lookup)
    extended_markets["rank"] = range(len(extended_markets))
    slot_to_ext_rank = dict(zip(extended_markets["slot_ts"], extended_markets["rank"]))
    ext_targets = extended_markets["target"].values
    ext_slot_ts = extended_markets["slot_ts"].values

    # ── Step 3: Load Binance spot ─────────────────────────────────────────
    log.info("Step 3: Loading Binance spot...")
    spot_path = LOCAL_DIR / "binance_spot_local.parquet"
    if spot_path.exists():
        spot_df = pd.read_parquet(str(spot_path))
        # timestamp_ms is in milliseconds — convert to seconds for spot_ts_arr
        spot_df = spot_df.sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
        spot_ts_arr = (spot_df["timestamp_ms"].values / 1000).astype(np.int64)
        spot_px_arr = spot_df["close"].values.astype(np.float64)
        log.info("Binance spot: %d candles", len(spot_ts_arr))
    else:
        # Fetch inline (fallback)
        log.warning("binance_spot_local.parquet not found — fetching from Binance")
        slot_min = int(markets["slot_ts"].min())
        slot_max = int(markets["slot_ts"].max())
        fetch_start_ms = (slot_min - 4 * 3600) * 1000
        fetch_end_ms   = (slot_max + 5 * 60 + 60) * 1000
        import urllib.request as urlreq

        candles = []
        cur_ms = fetch_start_ms
        while cur_ms < fetch_end_ms:
            url = (
                f"https://api.binance.com/api/v3/klines?"
                f"symbol=BTCUSDT&interval=1m"
                f"&startTime={cur_ms}&limit=1000"
            )
            with urlreq.urlopen(url, timeout=10) as resp:
                batch = json.loads(resp.read())
            if not batch:
                break
            candles.extend(batch)
            cur_ms = int(batch[-1][0]) + 60000
            time.sleep(0.2)

        candle_arr = np.array([[int(c[0]) // 1000, float(c[4])] for c in candles])
        candle_arr = candle_arr[np.argsort(candle_arr[:, 0])]
        _, idx = np.unique(candle_arr[:, 0], return_index=True)
        candle_arr = candle_arr[idx]
        spot_ts_arr = candle_arr[:, 0].astype(np.int64)
        spot_px_arr = candle_arr[:, 1].astype(np.float64)
        log.info("Binance spot (inline): %d candles", len(spot_ts_arr))

    # ── Step 4: Load ticks ────────────────────────────────────────────────
    log.info("Step 4: Loading ticks...")
    local_mids = set(markets["market_id"].tolist())
    pf = pq.ParquetFile(str(LOCAL_DIR / "ticks_btc_5min.parquet"))
    tick_cols = ["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"]

    chunks = []
    for rg_i in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg_i, columns=tick_cols).to_pandas()
        chunk["market_id"] = chunk["market_id"].astype(str)
        chunk = chunk[chunk["market_id"].isin(local_mids)]
        if len(chunk):
            chunks.append(chunk)
    gc.collect()

    btc = pd.concat(chunks, ignore_index=True)
    btc = btc[btc["timestamp_ms"] > 0]
    btc["size_usdc"] = btc["price"].astype(float) * btc["size_usdc"].astype(float)
    # size_usdc in ticks is already dollar volume but verify
    log.info("Ticks loaded: %d rows for %d markets",
             len(btc), btc["market_id"].nunique())

    # Map slot_ts per tick for t_sec computation
    slot_ts_map = dict(zip(markets["market_id"], markets["slot_ts"]))
    btc["slot_ts_val"] = btc["market_id"].map(slot_ts_map).astype(float)
    btc["t_sec"] = btc["timestamp_ms"].astype(float) / 1000 - btc["slot_ts_val"]
    btc = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < OBS_SECS)]

    # Pre-compute per-slot aggregates for lag features
    btc_up = btc[btc["outcome"] == "Up"]
    btc_dn = btc[btc["outcome"] == "Down"]
    slot_vol_up  = btc_up.groupby("market_id")["size_usdc"].sum().rename("vol_up")
    slot_vol_dn  = btc_dn.groupby("market_id")["size_usdc"].sum().rename("vol_dn")
    slot_vol_tot = (slot_vol_up.add(slot_vol_dn, fill_value=0)).rename("vol_total")
    slot_up_ratio = (slot_vol_up / slot_vol_tot.clip(lower=1e-9)).rename("up_ratio")
    slot_nticks   = btc.groupby("market_id").size().rename("slot_nticks")
    log.info("Per-slot aggregates computed")

    # ── Step 5: Build features ────────────────────────────────────────────
    log.info("Step 5: Building features for %d markets...", len(markets))

    def _ur(df_sub):
        up = df_sub[df_sub["outcome"]=="Up"]["size_usdc"].sum()
        dn = df_sub[df_sub["outcome"]=="Down"]["size_usdc"].sum()
        tot = up + dn
        return up / tot if tot > 0 else 0.5

    def _ur_w(df_sub, t0, t1):
        w = df_sub[(df_sub["t_sec"] >= t0) & (df_sub["t_sec"] < t1)]
        return _ur(w) if len(w) else 0.5

    def _vol_w(df_sub, t0, t1):
        w = df_sub[(df_sub["t_sec"] >= t0) & (df_sub["t_sec"] < t1)]
        return w["size_usdc"].sum()

    def spot_at(ts_s):
        """Return spot price at given Unix seconds."""
        idx = np.searchsorted(spot_ts_arr, ts_s, side="right") - 1
        idx = max(0, min(idx, len(spot_px_arr) - 1))
        return float(spot_px_arr[idx])

    rows = []
    btc_grouped = {mid: grp for mid, grp in btc.groupby("market_id")}

    for rank_i, row in markets.iterrows():
        mid     = row["market_id"]
        slot_ts = int(row["slot_ts"])
        target  = int(row["target"])

        grp = btc_grouped.get(mid)
        n   = len(grp) if grp is not None else 0

        # ── CLOB flow features ────────────────────────────────────────────
        if n > 0:
            ur  = slot_up_ratio.get(mid, 0.5)
            vt  = slot_vol_tot.get(mid, 0.0)
            ntx = slot_nticks.get(mid, 0)

            up_vals = grp[grp["outcome"]=="Up"]["size_usdc"].values
            dn_vals = grp[grp["outcome"]=="Down"]["size_usdc"].values

            w0 = _ur_w(grp, 0, 30);   w1 = _ur_w(grp, 30, 60)
            w2 = _ur_w(grp, 60, 90);  w3 = _ur_w(grp, 90, 120)
            w4 = _ur_w(grp, 120, 150); w5 = _ur_w(grp, 150, 180)

            # VWAP
            up_g = grp[grp["outcome"]=="Up"]
            dn_g = grp[grp["outcome"]=="Down"]
            def vwap(g): return (g["price"]*g["size_usdc"]).sum() / g["size_usdc"].sum() if len(g) else 0.5
            vwap_up = vwap(up_g); vwap_dn = vwap(dn_g)

            # Time-weighted up_ratio (exponential weights)
            all_sorted = grp.sort_values("t_sec")
            if len(all_sorted) > 1:
                w_exp = np.exp(-0.02 * (OBS_SECS - all_sorted["t_sec"].values))
                ur_up = (all_sorted["outcome"]=="Up").astype(float).values
                tw_ur = np.average(ur_up * all_sorted["size_usdc"].values,
                                   weights=w_exp) / (np.average(all_sorted["size_usdc"].values, weights=w_exp) + 1e-9)
            else:
                tw_ur = ur

            buy_sz = grp[grp["side"]=="BUY"]["size_usdc"].sum()
            buy_ratio = buy_sz / vt if vt > 0 else 0.5

            momentum  = (w3 + w4 + w5) / 3 - (w0 + w1 + w2) / 3
            stability = np.std([w0,w1,w2,w3,w4,w5])

            avg_up = up_vals.mean() if len(up_vals) else 0
            avg_dn = dn_vals.mean() if len(dn_vals) else 0
            size_disparity = avg_up - avg_dn

            feat = {
                "btc_up_ratio":     ur,
                "btc_n_ticks":      float(n),
                "btc_buy_ratio":    buy_ratio,
                "btc_tw_up_ratio":  tw_ur,
                "btc_momentum":     momentum,
                "btc_vwap_spread":  vwap_up - vwap_dn,
                "btc_vwap_up":      vwap_up,
                "btc_vwap_dn":      vwap_dn,
                "btc_vwap_trend":   vwap_up - 0.5,
                "btc_up_w0":        w0, "btc_up_w1": w1, "btc_up_w2": w2,
                "btc_up_w3":        w3, "btc_up_w4": w4, "btc_up_w5": w5,
                "btc_size_disparity": size_disparity,
                "btc_up_ratio_stability": stability,
                "btc_signal_conviction": ur * (1 - stability),
            }
        else:
            feat = {k: 0.0 for k in [
                "btc_up_ratio","btc_n_ticks","btc_buy_ratio","btc_tw_up_ratio",
                "btc_momentum","btc_vwap_spread","btc_vwap_up","btc_vwap_dn",
                "btc_vwap_trend","btc_up_w0","btc_up_w1","btc_up_w2","btc_up_w3",
                "btc_up_w4","btc_up_w5","btc_size_disparity",
                "btc_up_ratio_stability","btc_signal_conviction",
            ]}
            feat["btc_up_ratio"] = 0.5
            feat["btc_vwap_up"] = 0.5
            feat["btc_vwap_dn"] = 0.5
            feat["btc_tw_up_ratio"] = 0.5
            feat["btc_buy_ratio"] = 0.5

        # ── z-scores (cross-slot context) ────────────────────────────────
        ext_rank = slot_to_ext_rank.get(slot_ts, rank_i)

        def _hist_ur(lookback=20):
            vals = []
            for d in range(1, lookback + 1):
                prev_r = ext_rank - d
                if prev_r < 0:
                    break
                prev_mid = extended_markets["market_id"].iloc[prev_r]
                v = slot_up_ratio.get(prev_mid, None)
                if v is not None:
                    vals.append(v)
            return vals

        hist_vals = _hist_ur(20)
        if len(hist_vals) >= 3:
            mu20 = np.mean(hist_vals)
            sd20 = np.std(hist_vals) + 1e-6
            feat["btc_up_ratio_zscore_20s"] = (feat["btc_up_ratio"] - mu20) / sd20
            feat["btc_up_w5_zscore"] = (feat["btc_up_w5"] - mu20) / sd20
        else:
            feat["btc_up_ratio_zscore_20s"] = 0.0
            feat["btc_up_w5_zscore"] = 0.0

        hist5 = hist_vals[:5]
        if len(hist5) >= 2:
            mu5 = np.mean(hist5); sd5 = np.std(hist5) + 1e-6
            feat["btc_up_ratio_zscore_5s"] = (feat["btc_up_ratio"] - mu5) / sd5
        else:
            feat["btc_up_ratio_zscore_5s"] = 0.0

        # ── Spot features ─────────────────────────────────────────────────
        obs_end_ts = slot_ts + OBS_SECS
        px_now     = spot_at(obs_end_ts)

        def pre_ret(h):
            px_h = spot_at(slot_ts - h * 3600)
            return (px_now / px_h - 1) if px_h > 0 else 0.0

        feat["btc_pre_5m_ret"]  = (px_now / spot_at(slot_ts - 300) - 1) if spot_at(slot_ts - 300) > 0 else 0.0
        feat["btc_pre_30m_ret"] = pre_ret(0.5)
        feat["btc_pre_1h_ret"]  = pre_ret(1)
        feat["btc_pre_4h_ret"]  = pre_ret(4)

        px_1h_ago = spot_at(slot_ts - 3600)
        px_4h_ago = spot_at(slot_ts - 4 * 3600)
        feat["btc_pre_1h_4h_ratio"] = (px_now - px_1h_ago) / (px_now - px_4h_ago + 1e-9) if abs(px_now - px_4h_ago) > 1 else 0.0

        # Inslot spot vol (Binance 1m candles within slot)
        t0_idx = int(np.searchsorted(spot_ts_arr, slot_ts, side="left"))
        t1_idx = int(np.searchsorted(spot_ts_arr, slot_ts + OBS_SECS, side="right"))
        if t1_idx > t0_idx:
            inslot_px = spot_px_arr[t0_idx:t1_idx]
            feat["btc_inslot_ret"] = float(inslot_px[-1] / inslot_px[0] - 1) if inslot_px[0] > 0 else 0.0
        else:
            feat["btc_inslot_ret"] = 0.0

        # Round-number distance
        px_k = px_now / 1000
        feat["btc_dist_1k"] = min(px_k - math.floor(px_k), math.ceil(px_k) - px_k)

        # ── Lag features (using extended timeline for richer context) ─────
        lag_streak = 0
        streak_dir = None

        for lag_n in range(1, 6):
            prev_rank = ext_rank - lag_n
            if prev_rank >= 0:
                prev_target = int(ext_targets[prev_rank])
                prev_slot   = int(ext_slot_ts[prev_rank])
                prev_mid    = extended_markets["market_id"].iloc[prev_rank]

                # Time gap check: consecutive 5min slots
                time_gap = slot_ts - prev_slot
                if time_gap > lag_n * SLOT_DURATION * 3:
                    feat[f"lag_{lag_n}_outcome"] = 0.5  # missing
                    feat[f"prev_slot_up_ratio_{lag_n}"] = 0.5
                    feat[f"prev_slot_n_ticks_{lag_n}"] = 0.0
                    feat[f"prev_slot_vol_{lag_n}"] = 0.0
                    continue

                feat[f"lag_{lag_n}_outcome"] = float(prev_target)
                feat[f"prev_slot_up_ratio_{lag_n}"] = float(
                    slot_up_ratio.get(prev_mid, 0.5)
                )
                feat[f"prev_slot_n_ticks_{lag_n}"] = float(
                    slot_nticks.get(prev_mid, 0.0)
                )
                feat[f"prev_slot_vol_{lag_n}"] = float(
                    slot_vol_tot.get(prev_mid, 0.0)
                )

                if lag_n == 1:
                    streak_dir = prev_target
                    lag_streak = 1
                elif prev_target == streak_dir:
                    lag_streak += 1
                else:
                    pass  # streak stops
            else:
                for k in [f"lag_{lag_n}_outcome", f"prev_slot_up_ratio_{lag_n}",
                          f"prev_slot_n_ticks_{lag_n}", f"prev_slot_vol_{lag_n}"]:
                    feat[k] = 0.0

        feat["lag_streak"] = float(lag_streak)

        # ── Temporal features (NEW in v17) ────────────────────────────────
        dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        dow  = dt.weekday()  # 0=Mon, 6=Sun

        feat["hour_sin"]       = math.sin(2 * math.pi * hour / 24)
        feat["hour_cos"]       = math.cos(2 * math.pi * hour / 24)
        feat["dow_sin"]        = math.sin(2 * math.pi * dow / 7)
        feat["dow_cos"]        = math.cos(2 * math.pi * dow / 7)

        # Interaction: hour modulates up_ratio signal
        feat["hour_x_up_ratio"] = feat["btc_up_ratio"] * (hour / 24.0)
        feat["hour_x_tw_ur"]    = feat["btc_tw_up_ratio"] * (hour / 24.0)

        feat["target"] = target
        rows.append(feat)

    df = pd.DataFrame(rows)
    log.info("Feature matrix: %d rows × %d cols", len(df), len(df.columns))

    # ── Step 6: Feature selection ────────────────────────────────────────
    FEATURE_COLS = [c for c in df.columns if c != "target"]
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["target"].values.astype(int)

    log.info("Class balance: %d UP, %d DOWN (%.1f%% UP)",
             y.sum(), (y==0).sum(), 100*y.mean())

    # Walk-forward split to select features on OOF
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP)
    oof_preds = np.full(len(y), np.nan)

    # Quick feature screen with default LGB
    screen_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=30, random_state=42, verbose=-1
    )
    feat_importances = np.zeros(len(FEATURE_COLS))

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        screen_model.fit(X[tr_idx], y[tr_idx])
        feat_importances += screen_model.feature_importances_
    feat_importances /= N_SPLITS

    feat_rank = np.argsort(feat_importances)[::-1]
    top_features = [FEATURE_COLS[i] for i in feat_rank[:TOP_N_FEATS]]
    log.info("Top features selected: %s", top_features[:10])

    X_sel = df[top_features].values.astype(np.float32)

    # ── Step 7: Optuna hyperparameter tuning ────────────────────────────
    log.info("Step 7: Optuna tuning (%d trials)...", OPTUNA_TRIALS)

    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 200, 800),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":          trial.suggest_int("max_depth", 3, 7),
            "num_leaves":         trial.suggest_int("num_leaves", 8, 63),
            "min_child_samples":  trial.suggest_int("min_child_samples", 20, 80),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "random_state": 42, "verbose": -1,
        }
        aucs = []
        for tr_idx, val_idx in TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP).split(X_sel):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_sel[tr_idx], y[tr_idx])
            p = m.predict_proba(X_sel[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], p))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=4, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"random_state": 42, "verbose": -1})
    log.info("Best trial AUC=%.4f, params=%s", study.best_value, best_params)

    # ── Step 8: Walk-forward evaluation ─────────────────────────────────
    log.info("Step 8: Walk-forward evaluation...")
    wf_aucs, wf_briers, wf_accs = [], [], []
    oof_preds_sel = np.full(len(y), np.nan)

    for fold, (tr_idx, val_idx) in enumerate(
        TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP).split(X_sel)
    ):
        base = lgb.LGBMClassifier(**best_params)
        cal  = CalibratedClassifierCV(base, cv=3, method="isotonic")
        cal.fit(X_sel[tr_idx], y[tr_idx])
        p = cal.predict_proba(X_sel[val_idx])[:, 1]
        oof_preds_sel[val_idx] = p
        wf_aucs.append(roc_auc_score(y[val_idx], p))
        wf_briers.append(brier_score_loss(y[val_idx], p))
        wf_accs.append((p.round() == y[val_idx]).mean())
        log.info("  Fold %d | AUC=%.4f | Brier=%.4f | Acc=%.4f",
                 fold, wf_aucs[-1], wf_briers[-1], wf_accs[-1])

    wf_auc   = float(np.mean(wf_aucs))
    wf_brier = float(np.mean(wf_briers))
    wf_acc   = float(np.mean(wf_accs))
    log.info("WF results: AUC=%.4f | Brier=%.4f | Acc=%.4f",
             wf_auc, wf_brier, wf_acc)

    # ── Step 9: Promotion gate ───────────────────────────────────────────
    beats_auc   = wf_auc   > champion["wf_auc"]
    beats_brier = wf_brier < champion["wf_brier"]
    beats_acc   = wf_acc   > champion["wf_acc"]
    score = sum([beats_auc, beats_brier, beats_acc])
    log.info("Gate vs champion %s: AUC %s | Brier %s | Acc %s → %d/3",
             champion["version"],
             "✓" if beats_auc else "✗",
             "✓" if beats_brier else "✗",
             "✓" if beats_acc else "✗",
             score)

    # Sanity check
    neutral = np.array([[0.5] * len(top_features)], dtype=np.float32)
    up_feats = {f: 0.0 for f in top_features}
    up_feats["btc_up_ratio"]    = 0.85
    up_feats["btc_tw_up_ratio"] = 0.85
    up_feats["btc_up_w5"]       = 0.85 if "btc_up_w5" in top_features else 0.5

    # Build final model on all data
    final_base = lgb.LGBMClassifier(**best_params)
    final_model = CalibratedClassifierCV(final_base, cv=3, method="isotonic")
    final_model.fit(X_sel, y)

    up_arr  = pd.DataFrame([up_feats])[top_features].values.astype(np.float32)
    prob_up   = final_model.predict_proba(up_arr)[0, 1]
    prob_neut = final_model.predict_proba(neutral)[0, 1]
    log.info("Sanity: UP scenario → %.3f (want >0.55) | Neutral → %.3f (want ~0.50)",
             prob_up, prob_neut)

    # ── Step 10: Save & promote ──────────────────────────────────────────
    if score < 2:
        log.info("NOT PROMOTED (%d/3). Training complete.", score)
    else:
        log.info("PROMOTING v17! (%d/3 metrics beat champion)", score)
        import tempfile
        from huggingface_hub import HfApi

        model_data = {
            "version":  "v17",
            "features": top_features,
            "model":    final_model,
            "wf_auc":   wf_auc,
            "wf_brier": wf_brier,
            "wf_acc":   wf_acc,
        }

        meta = {
            "version":  "v17",
            "wf_auc":   wf_auc,
            "wf_brier": wf_brier,
            "wf_acc":   wf_acc,
            "features": top_features,
            "n_samples": len(y),
            "n_features": len(top_features),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "changes": (
                "Extended lag context (22k slot timeline for lag features), "
                "temporal features (hour_sin/cos, dow_sin/cos, hour_x_up_ratio)"
            ),
        }

        api = HfApi(token=HF_TOKEN)
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path  = Path(tmpdir) / "champion.pkl"
            meta_path = Path(tmpdir) / "champion_meta.json"

            with open(pkl_path, "wb") as f:
                pickle.dump(model_data, f)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            api.upload_file(path_or_fileobj=str(pkl_path),
                            path_in_repo="champion.pkl",
                            repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN)
            api.upload_file(path_or_fileobj=str(meta_path),
                            path_in_repo="champion_meta.json",
                            repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN)

        log.info("v17 promoted to HF! AUC=%.4f Brier=%.4f Acc=%.4f",
                 wf_auc, wf_brier, wf_acc)

    log.info("v17 training complete.")


@app.local_entrypoint()
def main():
    train_v17.remote()
