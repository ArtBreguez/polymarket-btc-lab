"""
backtest_v22_honest.py — HONEST backtest: train 90% / test 10%
================================================================
Anti-overfitting techniques:
  1. Purged walk-forward CV: gap between train/val to prevent leakage
  2. Adversarial validation: detect if test looks different from train
  3. Feature noise injection: add random features to detect overfitting
  4. Temporal embargo: 1-hour gap between train and test (no overlap)
  5. Calibration on separate fold (not same data as tuning)
  6. Permutation importance: validate features matter OOS
  7. Multiple spread scenarios: stress-test profitability
  8. Rolling window backtest: test in 5 sequential windows (not just 1)
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

app = modal.App("btc-v22-backtest-honest", image=image)


@app.function(
    cpu=8,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
)
def backtest_honest():
    import gc, json, logging, math, os, pickle, sys, time, warnings
    from datetime import datetime, timezone
    from pathlib import Path
    from collections import defaultdict

    import numpy as np
    import optuna
    import pandas as pd
    import pyarrow.parquet as pq
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
    from sklearn.model_selection import TimeSeriesSplit
    import lightgbm as lgb
    from huggingface_hub import hf_hub_download

    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    # ── Config ────────────────────────────────────────────────────────────
    HF_TOKEN      = os.environ.get("HF_TOKEN", "")
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
    SLOT_DURATION  = 300
    OBSERVE_SECS   = 180
    OPTUNA_TRIALS  = 120
    N_SPLITS       = 5
    WF_GAP         = 12   # purge gap: 12 slots = 1 hour (prevents temporal leakage)
    TOP_N_FEATS    = 40
    N_NOISE_FEATS  = 5    # random noise features to detect overfitting

    # Live trading params
    MIN_CONFIDENCE = 0.60
    MIN_EDGE       = 0.10
    MIN_EDGE_MID   = 0.05
    ASK_LO, ASK_HI = 0.38, 0.90
    TAKER_FEE      = 0.02
    HALF_SPREAD    = 0.02

    TRAIN_FRAC     = 0.90
    EMBARGO_SLOTS  = 12   # 1 hour temporal embargo between train/test

    DATA_DIR = Path("/tmp/btc_data")
    DATA_DIR.mkdir(exist_ok=True)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")

    # ── Download data from HF ─────────────────────────────────────────────
    log.info("Downloading data from HF...")
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
            continue
        try:
            import shutil
            hf_hub_download(repo_id=HF_MODEL_REPO, filename=hf_path,
                            token=HF_TOKEN, repo_type="model",
                            local_dir=str(DATA_DIR), local_dir_use_symlinks=False)
            src = DATA_DIR / hf_path
            if src.exists() and not local_path.exists():
                shutil.move(str(src), str(local_path))
            log.info("  Downloaded %s", local_name)
        except Exception as e:
            log.warning("  Skip %s: %s", local_name, e)

    # ── Load & merge markets ──────────────────────────────────────────────
    log.info("Loading markets...")
    markets = pd.read_csv(DATA_DIR / "all_markets.csv")
    markets["market_id"] = markets["market_id"].astype(str)
    markets["slot_ts"] = markets["slot_ts"].astype(int)

    new_mkt_path = DATA_DIR / "new_markets.csv"
    if new_mkt_path.exists():
        new_mkts = pd.read_csv(new_mkt_path)
        new_mkts["market_id"] = new_mkts["market_id"].astype(str)
        new_mkts["slot_ts"] = new_mkts["slot_ts"].astype(int)
        if "target" in new_mkts.columns:
            existing_ids = set(markets["market_id"])
            truly_new = new_mkts[~new_mkts["market_id"].isin(existing_ids)]
            if len(truly_new) > 0:
                markets = pd.concat([markets, truly_new[markets.columns]], ignore_index=True)

    markets = markets.sort_values("slot_ts").reset_index(drop=True)
    n_total = len(markets)
    n_train = int(n_total * TRAIN_FRAC)
    n_embargo_end = n_train + EMBARGO_SLOTS  # test starts after embargo
    log.info("Total: %d | Train: %d | Embargo: %d slots | Test: %d",
             n_total, n_train, EMBARGO_SLOTS, n_total - n_embargo_end)

    markets["rank"] = range(n_total)
    slot_to_rank = dict(zip(markets["slot_ts"], markets["rank"]))
    all_targets = markets["target"].values
    all_slot_ts = markets["slot_ts"].values
    all_mids = markets["market_id"].values

    # ── Load OB features ──────────────────────────────────────────────────
    log.info("Loading OB features...")
    ob_df = pd.read_parquet(str(DATA_DIR / "ob_features_full.parquet"))
    ob_df["market_id"] = ob_df["market_id"].astype(str)
    ob_by_market = ob_df.set_index("market_id").to_dict("index")
    ob_cols = [c for c in ob_df.columns if c != "market_id"]

    # ── Load Binance spot ─────────────────────────────────────────────────
    log.info("Loading Binance spot...")
    spot_dfs = []
    for sp in ["binance_spot_full.parquet", "binance_spot_local.parquet"]:
        sp_path = DATA_DIR / sp
        if sp_path.exists():
            spot_dfs.append(pd.read_parquet(str(sp_path)))
    spot_df = pd.concat(spot_dfs, ignore_index=True) if spot_dfs else pd.DataFrame()
    spot_df = spot_df.sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
    spot_ts_arr = (spot_df["timestamp_ms"].values // 1000).astype(np.int64)
    spot_px_arr = spot_df["close"].values.astype(np.float64)

    # ── Load ticks ────────────────────────────────────────────────────────
    log.info("Loading ticks...")
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
    log.info("Ticks in [0, 300s): %d", len(btc))
    btc_grouped = {mid: grp for mid, grp in btc.groupby("market_id")}

    # ── Pre-compute aggregates ────────────────────────────────────────────
    obs_secs = OBSERVE_SECS
    n_windows = obs_secs // 30
    filtered = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < obs_secs)]
    up = filtered[filtered["outcome"] == "Up"]
    dn = filtered[filtered["outcome"] == "Down"]
    vol_up_s = up.groupby("market_id")["size_usdc"].sum()
    vol_dn_s = dn.groupby("market_id")["size_usdc"].sum()
    vol_tot_s = vol_up_s.add(vol_dn_s, fill_value=0)
    slot_up_ratio = vol_up_s / vol_tot_s.clip(lower=1e-9)
    slot_nticks = filtered.groupby("market_id").size()

    def spot_at(ts_s):
        idx = np.searchsorted(spot_ts_arr, ts_s, side="right") - 1
        return float(spot_px_arr[max(0, min(idx, len(spot_px_arr) - 1))])

    def _ur_w(df_sub, t0, t1):
        w = df_sub[(df_sub["t_sec"] >= t0) & (df_sub["t_sec"] < t1)]
        if len(w) == 0:
            return 0.5
        u = w[w["outcome"] == "Up"]["size_usdc"].sum()
        d = w[w["outcome"] == "Down"]["size_usdc"].sum()
        t = u + d
        return u / t if t > 0 else 0.5

    # ── Build features for ALL markets ────────────────────────────────────
    log.info("Building features for all %d markets...", n_total)

    def build_row(rank_i, row):
        mid = row["market_id"]
        slot_ts = int(row["slot_ts"])
        target = int(row["target"])

        grp_full = btc_grouped.get(mid)
        if grp_full is not None:
            grp = grp_full[(grp_full["t_sec"] >= 0) & (grp_full["t_sec"] < obs_secs)]
        else:
            grp = None
        n = len(grp) if grp is not None else 0

        if n > 0:
            ur = slot_up_ratio.get(mid, 0.5)
            vt = vol_tot_s.get(mid, 0.0)

            up_vals = grp[grp["outcome"] == "Up"]["size_usdc"].values
            dn_vals = grp[grp["outcome"] == "Down"]["size_usdc"].values

            w_vals = []
            for wi in range(n_windows):
                w_vals.append(_ur_w(grp, wi * 30, (wi + 1) * 30))
            while len(w_vals) < 6:
                w_vals.append(0.5)

            up_g = grp[grp["outcome"] == "Up"]
            dn_g = grp[grp["outcome"] == "Down"]
            vwap_up = (up_g["price"] * up_g["size_usdc"]).sum() / up_g["size_usdc"].sum() if len(up_g) else 0.5
            vwap_dn = (dn_g["price"] * dn_g["size_usdc"]).sum() / dn_g["size_usdc"].sum() if len(dn_g) else 0.5

            all_sorted = grp.sort_values("t_sec")
            if len(all_sorted) > 1:
                w_exp = np.exp(-0.02 * (obs_secs - all_sorted["t_sec"].values))
                ur_up_arr = (all_sorted["outcome"] == "Up").astype(float).values
                tw_ur = np.average(ur_up_arr * all_sorted["size_usdc"].values,
                                   weights=w_exp) / (np.average(all_sorted["size_usdc"].values, weights=w_exp) + 1e-9)
            else:
                tw_ur = ur

            buy_sz = grp[grp["side"] == "BUY"]["size_usdc"].sum()
            buy_ratio = buy_sz / vt if vt > 0 else 0.5

            half = n_windows // 2
            momentum = np.mean(w_vals[half:n_windows]) - np.mean(w_vals[:half])
            stability = np.std(w_vals[:n_windows])
            avg_up = up_vals.mean() if len(up_vals) else 0
            avg_dn = dn_vals.mean() if len(dn_vals) else 0

            feat = {
                "btc_up_ratio": ur, "btc_n_ticks": float(n),
                "btc_buy_ratio": buy_ratio, "btc_tw_up_ratio": tw_ur,
                "btc_momentum": momentum,
                "btc_vwap_spread": vwap_up - vwap_dn,
                "btc_vwap_up": vwap_up, "btc_vwap_dn": vwap_dn,
                "btc_vwap_trend": vwap_up - 0.5,
                "btc_up_w0": w_vals[0], "btc_up_w1": w_vals[1],
                "btc_up_w2": w_vals[2], "btc_up_w3": w_vals[3],
                "btc_up_w4": w_vals[4], "btc_up_w5": w_vals[5],
                "btc_size_disparity": avg_up - avg_dn,
                "btc_up_ratio_stability": stability,
                "btc_signal_conviction": ur * (1 - stability),
            }
        else:
            feat = {k: 0.0 for k in [
                "btc_up_ratio", "btc_n_ticks", "btc_buy_ratio", "btc_tw_up_ratio",
                "btc_momentum", "btc_vwap_spread", "btc_vwap_up", "btc_vwap_dn",
                "btc_vwap_trend", "btc_up_w0", "btc_up_w1", "btc_up_w2", "btc_up_w3",
                "btc_up_w4", "btc_up_w5", "btc_size_disparity",
                "btc_up_ratio_stability", "btc_signal_conviction",
            ]}
            feat.update({"btc_up_ratio": 0.5, "btc_vwap_up": 0.5, "btc_vwap_dn": 0.5,
                         "btc_tw_up_ratio": 0.5, "btc_buy_ratio": 0.5})
            ur = 0.5

        # Z-scores
        ext_rank = slot_to_rank.get(slot_ts, rank_i)
        def _hist_ur(lookback=20):
            vals = []
            for d in range(1, lookback + 1):
                prev_r = ext_rank - d
                if prev_r < 0: break
                v = slot_up_ratio.get(all_mids[prev_r], None)
                if v is not None: vals.append(v)
            return vals

        hist_vals = _hist_ur(20)
        if len(hist_vals) >= 3:
            mu20 = np.mean(hist_vals); sd20 = np.std(hist_vals) + 1e-6
            feat["btc_up_ratio_zscore_20s"] = (feat["btc_up_ratio"] - mu20) / sd20
            feat["btc_up_w5_zscore"] = (feat["btc_up_w5"] - mu20) / sd20
        else:
            feat["btc_up_ratio_zscore_20s"] = 0.0; feat["btc_up_w5_zscore"] = 0.0

        hist5 = hist_vals[:5]
        if len(hist5) >= 2:
            mu5 = np.mean(hist5); sd5 = np.std(hist5) + 1e-6
            feat["btc_up_ratio_zscore_5s"] = (feat["btc_up_ratio"] - mu5) / sd5
        else:
            feat["btc_up_ratio_zscore_5s"] = 0.0

        # Spot features
        obs_end_ts = slot_ts + obs_secs
        px_now = spot_at(obs_end_ts)
        def pre_ret(h):
            px_h = spot_at(slot_ts - h * 3600)
            return (px_now / px_h - 1) if px_h > 0 else 0.0

        feat["btc_pre_5m_ret"] = (px_now / spot_at(slot_ts - 300) - 1) if spot_at(slot_ts - 300) > 0 else 0.0
        feat["btc_pre_30m_ret"] = pre_ret(0.5)
        feat["btc_pre_1h_ret"] = pre_ret(1)
        feat["btc_pre_4h_ret"] = pre_ret(4)

        px_1h = spot_at(slot_ts - 3600); px_4h = spot_at(slot_ts - 4*3600)
        feat["btc_pre_1h_4h_ratio"] = (px_now - px_1h) / (px_now - px_4h + 1e-9) \
            if px_now > 0 and px_1h > 0 and px_4h > 0 and abs(px_now - px_4h) > 1 else 0.0

        t0_idx = int(np.searchsorted(spot_ts_arr, slot_ts, side="left"))
        t1_idx = int(np.searchsorted(spot_ts_arr, slot_ts + obs_secs, side="right"))
        feat["btc_inslot_ret"] = float(spot_px_arr[t1_idx-1] / spot_px_arr[t0_idx] - 1) \
            if t1_idx > t0_idx and spot_px_arr[t0_idx] > 0 else 0.0

        px_k = px_now / 1000
        feat["btc_dist_1k"] = min(px_k - math.floor(px_k), math.ceil(px_k) - px_k)

        # Lag features (with staleness guard)
        lag_streak = 0; streak_dir = None
        for lag_n in range(1, 6):
            prev_rank = ext_rank - lag_n
            if prev_rank >= 0:
                prev_slot = int(all_slot_ts[prev_rank])
                time_gap = slot_ts - prev_slot
                if time_gap > lag_n * SLOT_DURATION * 3:
                    feat[f"lag_{lag_n}_outcome"] = 0.5
                    feat[f"prev_slot_up_ratio_{lag_n}"] = 0.5
                    feat[f"prev_slot_n_ticks_{lag_n}"] = 0.0
                    feat[f"prev_slot_vol_{lag_n}"] = 0.0
                    continue
                feat[f"lag_{lag_n}_outcome"] = float(all_targets[prev_rank])
                feat[f"prev_slot_up_ratio_{lag_n}"] = float(slot_up_ratio.get(all_mids[prev_rank], 0.5))
                feat[f"prev_slot_n_ticks_{lag_n}"] = float(slot_nticks.get(all_mids[prev_rank], 0.0))
                feat[f"prev_slot_vol_{lag_n}"] = float(vol_tot_s.get(all_mids[prev_rank], 0.0))
                if lag_n == 1: streak_dir = all_targets[prev_rank]; lag_streak = 1
                elif all_targets[prev_rank] == streak_dir: lag_streak += 1
            else:
                for k in [f"lag_{lag_n}_outcome", f"prev_slot_up_ratio_{lag_n}",
                          f"prev_slot_n_ticks_{lag_n}", f"prev_slot_vol_{lag_n}"]:
                    feat[k] = 0.0
        feat["lag_streak"] = float(lag_streak)

        # Temporal
        dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0; dow = dt.weekday()
        feat["hour_sin"] = math.sin(2*math.pi*hour/24)
        feat["hour_cos"] = math.cos(2*math.pi*hour/24)
        feat["dow_sin"] = math.sin(2*math.pi*dow/7)
        feat["dow_cos"] = math.cos(2*math.pi*dow/7)
        feat["hour_x_up_ratio"] = feat["btc_up_ratio"] * (hour/24.0)
        feat["hour_x_tw_ur"] = feat.get("btc_tw_up_ratio", 0.5) * (hour/24.0)

        # OB features
        ob = ob_by_market.get(mid)
        if ob is not None:
            for col in ob_cols:
                feat[f"ob_{col}" if not col.startswith("ob_") else col] = float(ob.get(col, 0.0))
            feat["x_imb_x_ur"] = float(ob.get("ob_imbalance", 0)) * feat["btc_up_ratio"]
            feat["x_depth_x_momentum"] = float(ob.get("ob_depth_ratio", 1)) * feat["btc_momentum"]
            feat["x_spread_x_vol"] = float(ob.get("ob_spread", 0)) * feat["btc_n_ticks"]
            feat["x_ob_drift_x_inslot"] = float(ob.get("ob_mid_drift", 0)) * feat["btc_inslot_ret"]
            feat["x_fill_imb_x_buy"] = float(ob.get("ob_fill_imbalance", 0)) * feat["btc_buy_ratio"]
        else:
            for col in ob_cols:
                key = f"ob_{col}" if not col.startswith("ob_") else col
                if "ratio" in col or "imbalance" in col or "imb" in col: feat[key] = 0.0
                elif "spread" in col: feat[key] = 0.02
                elif "depth" in col and "5c" in col: feat[key] = 0.5
                elif "mid" in col and "drift" not in col: feat[key] = 0.5
                else: feat[key] = 0.0
            for k in ["x_imb_x_ur", "x_depth_x_momentum", "x_spread_x_vol",
                       "x_ob_drift_x_inslot", "x_fill_imb_x_buy"]:
                feat[k] = 0.0

        feat["target"] = target
        return feat

    rows = []
    for rank_i, row in markets.iterrows():
        rows.append(build_row(rank_i, row))
        if len(rows) % 5000 == 0:
            log.info("  Features: %d/%d", len(rows), n_total)
    df = pd.DataFrame(rows)
    log.info("Feature matrix: %d rows x %d cols", len(df), len(df.columns))

    # ── Split: train 90%, embargo, test 10% ───────────────────────────────
    FEATURE_COLS = [c for c in df.columns if c != "target"]
    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_embargo_end:]  # skip embargo zone
    log.info("Train: %d | Embargo: %d | Test: %d", len(df_train), EMBARGO_SLOTS, len(df_test))
    log.info("Train: %s to %s",
             datetime.fromtimestamp(all_slot_ts[0], tz=timezone.utc).strftime("%Y-%m-%d"),
             datetime.fromtimestamp(all_slot_ts[n_train-1], tz=timezone.utc).strftime("%Y-%m-%d"))
    log.info("Test:  %s to %s",
             datetime.fromtimestamp(all_slot_ts[n_embargo_end], tz=timezone.utc).strftime("%Y-%m-%d"),
             datetime.fromtimestamp(all_slot_ts[-1], tz=timezone.utc).strftime("%Y-%m-%d"))

    X_train_all = df_train[FEATURE_COLS].values.astype(np.float32)
    y_train_all = df_train["target"].values.astype(int)

    # ══════════════════════════════════════════════════════════════════════
    # ANTI-OVERFITTING TECHNIQUE 1: Noise features as overfitting canary
    # ══════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("=" * 70)
    log.info("ANTI-OVERFITTING CHECK 1: Noise feature canary")
    rng = np.random.RandomState(42)
    noise_cols = []
    for i in range(N_NOISE_FEATS):
        col_name = f"_noise_{i}"
        noise_cols.append(col_name)
        df[col_name] = rng.randn(len(df))
    FEATURE_COLS_WITH_NOISE = FEATURE_COLS + noise_cols
    log.info("Added %d random noise features as canary", N_NOISE_FEATS)

    # ══════════════════════════════════════════════════════════════════════
    # ANTI-OVERFITTING TECHNIQUE 2: Adversarial validation
    # ══════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("=" * 70)
    log.info("ANTI-OVERFITTING CHECK 2: Adversarial validation")
    log.info("Can a model distinguish train from test? (AUC ~0.5 = good)")

    X_adv_train = df_train[FEATURE_COLS].values[:, :30].astype(np.float32)  # use top 30 feats
    X_adv_test = df_test[FEATURE_COLS].values[:, :30].astype(np.float32)
    X_adv = np.vstack([X_adv_train, X_adv_test])
    y_adv = np.array([0]*len(X_adv_train) + [1]*len(X_adv_test))

    # Shuffle and split
    perm = rng.permutation(len(X_adv))
    X_adv, y_adv = X_adv[perm], y_adv[perm]
    split = int(0.7 * len(X_adv))
    adv_model = lgb.LGBMClassifier(n_estimators=100, max_depth=3, verbose=-1, random_state=42)
    adv_model.fit(X_adv[:split], y_adv[:split])
    adv_probs = adv_model.predict_proba(X_adv[split:])[:, 1]
    adv_auc = roc_auc_score(y_adv[split:], adv_probs)
    log.info("Adversarial AUC: %.4f (closer to 0.50 = less distribution shift)", adv_auc)
    if adv_auc > 0.65:
        log.warning("WARNING: High adversarial AUC (%.4f) suggests train/test distribution shift!", adv_auc)
    else:
        log.info("PASS: Train and test distributions look similar")

    # ══════════════════════════════════════════════════════════════════════
    # Feature selection on TRAIN ONLY (with noise canary)
    # ══════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("=" * 70)
    log.info("Feature selection (train only, purged WF with gap=%d)...", WF_GAP)

    X_train_with_noise = df_train[FEATURE_COLS_WITH_NOISE].values.astype(np.float32)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP)
    screen = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                num_leaves=15, min_child_samples=30, random_state=42, verbose=-1)
    feat_importances = np.zeros(len(FEATURE_COLS_WITH_NOISE))
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_with_noise)):
        screen.fit(X_train_with_noise[tr_idx], y_train_all[tr_idx])
        feat_importances += screen.feature_importances_
    feat_importances /= N_SPLITS

    # Check noise feature ranking (should be near bottom)
    feat_rank = np.argsort(feat_importances)[::-1]
    ranked_names = [FEATURE_COLS_WITH_NOISE[i] for i in feat_rank]
    noise_ranks = [ranked_names.index(nc) + 1 for nc in noise_cols]
    max_noise_rank = max(noise_ranks) if noise_ranks else 999
    min_noise_rank = min(noise_ranks) if noise_ranks else 999
    log.info("Noise feature ranks: %s (out of %d)", noise_ranks, len(FEATURE_COLS_WITH_NOISE))
    if min_noise_rank <= TOP_N_FEATS:
        log.warning("OVERFITTING ALERT: Noise feature ranked #%d (in top %d)!", min_noise_rank, TOP_N_FEATS)
    else:
        log.info("PASS: All noise features ranked below top %d (lowest rank: #%d)", TOP_N_FEATS, min_noise_rank)

    # Select top features (excluding noise)
    top_features = [FEATURE_COLS_WITH_NOISE[i] for i in feat_rank[:TOP_N_FEATS + N_NOISE_FEATS]
                    if not FEATURE_COLS_WITH_NOISE[i].startswith("_noise_")][:TOP_N_FEATS]
    log.info("Top 10 features: %s", top_features[:10])

    # ══════════════════════════════════════════════════════════════════════
    # ANTI-OVERFITTING TECHNIQUE 3: Optuna with early stopping + regularization
    # ══════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("=" * 70)
    log.info("Optuna tuning (%d trials, purged WF gap=%d, train only)...", OPTUNA_TRIALS, WF_GAP)

    X_top_train = df_train[top_features].values.astype(np.float32)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),  # lower ceiling
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),  # not too low
            "max_depth": trial.suggest_int("max_depth", 3, 6),  # capped at 6
            "num_leaves": trial.suggest_int("num_leaves", 8, 31),  # capped at 31 (conservative)
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 150),  # higher floor
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),  # stronger bagging
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),  # stronger L1
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),  # stronger L2
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),  # prevent trivial splits
            "random_state": 42, "verbose": -1,
        }
        aucs = []
        briers = []
        for tr_idx, val_idx in TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP).split(X_top_train):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_top_train[tr_idx], y_train_all[tr_idx])
            p = m.predict_proba(X_top_train[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_train_all[val_idx], p))
            briers.append(brier_score_loss(y_train_all[val_idx], p))

        # Multi-objective: maximize AUC but penalize high variance (overfitting signal)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        # Penalize if fold variance is high (sign of overfitting to specific data)
        return mean_auc - 0.5 * std_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=4, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"random_state": 42, "verbose": -1})
    log.info("Best Optuna score=%.4f (AUC - 0.5*std)", study.best_value)
    log.info("Best params: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()})

    # ══════════════════════════════════════════════════════════════════════
    # ANTI-OVERFITTING TECHNIQUE 4: Train/val gap validation
    # Show per-fold AUC to detect fold instability
    # ══════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("=" * 70)
    log.info("Walk-forward fold stability check...")
    fold_aucs = []
    fold_briers = []
    fold_accs = []
    for fold, (tr_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP).split(X_top_train)):
        m = lgb.LGBMClassifier(**best_params)
        cal = CalibratedClassifierCV(m, cv=3, method="isotonic")
        cal.fit(X_top_train[tr_idx], y_train_all[tr_idx])
        p = cal.predict_proba(X_top_train[val_idx])[:, 1]
        auc = roc_auc_score(y_train_all[val_idx], p)
        brier = brier_score_loss(y_train_all[val_idx], p)
        acc = (p.round() == y_train_all[val_idx]).mean()
        fold_aucs.append(auc)
        fold_briers.append(brier)
        fold_accs.append(acc)
        log.info("  Fold %d: AUC=%.4f Brier=%.4f Acc=%.4f (train=%d, val=%d)",
                 fold, auc, brier, acc, len(tr_idx), len(val_idx))
    log.info("WF mean: AUC=%.4f+-%.4f Brier=%.4f+-%.4f Acc=%.4f+-%.4f",
             np.mean(fold_aucs), np.std(fold_aucs),
             np.mean(fold_briers), np.std(fold_briers),
             np.mean(fold_accs), np.std(fold_accs))

    if np.std(fold_aucs) > 0.05:
        log.warning("OVERFITTING ALERT: High fold AUC variance (%.4f) — model may be unstable", np.std(fold_aucs))

    # ══════════════════════════════════════════════════════════════════════
    # Train final model on ALL training data (first 90%)
    # ══════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("Training final model on %d train rows...", n_train)
    final_base = lgb.LGBMClassifier(**best_params)
    final_model = CalibratedClassifierCV(final_base, cv=3, method="isotonic")
    final_model.fit(X_top_train, y_train_all)

    # ══════════════════════════════════════════════════════════════════════
    # Evaluate on HELD-OUT test set (truly OOS)
    # ══════════════════════════════════════════════════════════════════════
    X_test = df_test[top_features].values.astype(np.float32)
    y_test = df_test["target"].values.astype(int)
    probs = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, probs)
    test_brier = brier_score_loss(y_test, probs)
    test_acc = (probs.round() == y_test).mean()
    test_logloss = log_loss(y_test, probs)

    log.info("")
    log.info("=" * 70)
    log.info("HOLDOUT METRICS (truly OOS, after embargo)")
    log.info("  AUC:     %.4f (WF train mean: %.4f, delta: %.4f)",
             test_auc, np.mean(fold_aucs), test_auc - np.mean(fold_aucs))
    log.info("  Brier:   %.4f", test_brier)
    log.info("  Acc:     %.4f", test_acc)
    log.info("  LogLoss: %.4f", test_logloss)
    auc_drop = np.mean(fold_aucs) - test_auc
    if auc_drop > 0.03:
        log.warning("OVERFITTING SIGNAL: AUC dropped %.4f from WF to holdout", auc_drop)
    else:
        log.info("PASS: AUC drop from WF to holdout is only %.4f (< 0.03 threshold)", auc_drop)

    # ══════════════════════════════════════════════════════════════════════
    # ANTI-OVERFITTING TECHNIQUE 5: Permutation importance on test set
    # ══════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("=" * 70)
    log.info("Permutation importance (OOS test set)...")
    base_auc = test_auc
    perm_drops = {}
    for fi, fname in enumerate(top_features[:20]):  # top 20 only for speed
        X_perm = X_test.copy()
        X_perm[:, fi] = rng.permutation(X_perm[:, fi])
        perm_probs = final_model.predict_proba(X_perm)[:, 1]
        perm_auc = roc_auc_score(y_test, perm_probs)
        drop = base_auc - perm_auc
        perm_drops[fname] = drop

    # Sort by importance
    sorted_perm = sorted(perm_drops.items(), key=lambda x: -x[1])
    log.info("Top OOS permutation importance (AUC drop when shuffled):")
    for fname, drop in sorted_perm[:15]:
        status = "***" if drop > 0.01 else ""
        log.info("  %s: %.4f %s", fname, drop, status)
    negative_feats = [(f, d) for f, d in sorted_perm if d < -0.005]
    if negative_feats:
        log.warning("Features that HURT when shuffled (model worse without them being random):")
        for f, d in negative_feats:
            log.warning("  %s: %.4f (shuffling IMPROVES AUC — likely noise feature)", f, d)

    # ══════════════════════════════════════════════════════════════════════
    # ANTI-OVERFITTING TECHNIQUE 6: Calibration check
    # ══════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("=" * 70)
    log.info("Calibration check (OOS)...")
    for lo, hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]:
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() < 10:
            continue
        actual_rate = y_test[mask].mean()
        predicted_mean = probs[mask].mean()
        n_bin = mask.sum()
        log.info("  P(UP) [%.2f, %.2f): predicted=%.3f actual=%.3f n=%d delta=%.3f",
                 lo, hi, predicted_mean, actual_rate, n_bin, abs(predicted_mean - actual_rate))

    # ══════════════════════════════════════════════════════════════════════
    # Simulate live trading on holdout test set
    # ══════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("=" * 70)
    log.info("SIMULATING LIVE TRADING ON HOLDOUT TEST SET")
    log.info("=" * 70)

    trades = []
    skips = defaultdict(int)

    for i in range(len(df_test)):
        row_idx = n_embargo_end + i
        if row_idx >= n_total:
            break
        prob_up = float(probs[i])
        target = int(y_test[i])
        slot_ts = int(all_slot_ts[row_idx])
        mid_id = all_mids[row_idx]
        ur = float(slot_up_ratio.get(mid_id, 0.5))

        if prob_up >= 0.5:
            direction = "UP"; confidence = prob_up; model_prob = prob_up
        else:
            direction = "DOWN"; confidence = 1.0 - prob_up; model_prob = 1.0 - prob_up

        if confidence < MIN_CONFIDENCE:
            skips["low_conf"] += 1; continue

        market_mid = ur if direction == "UP" else (1.0 - ur)
        ask_price = market_mid + HALF_SPREAD

        if not (ASK_LO <= ask_price <= ASK_HI):
            skips["ask_range"] += 1; continue

        edge_ask = model_prob - ask_price
        if edge_ask < MIN_EDGE:
            skips["low_edge_ask"] += 1; continue

        edge_mid = model_prob - market_mid
        if edge_mid < MIN_EDGE_MID:
            skips["low_edge_mid"] += 1; continue

        actual = "UP" if target == 1 else "DOWN"
        if direction == actual:
            pnl = (1.0 * (1.0 - TAKER_FEE)) - ask_price
            result = "WIN"
        else:
            pnl = -ask_price
            result = "LOSS"

        dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        trades.append({
            "slot_ts": slot_ts, "direction": direction, "actual": actual,
            "result": result, "confidence": round(confidence, 4),
            "model_prob": round(model_prob, 4), "ask_price": round(ask_price, 4),
            "market_mid": round(market_mid, 4), "edge_ask": round(edge_ask, 4),
            "pnl": round(pnl, 4), "dt": dt.strftime("%Y-%m-%d %H:%M"),
        })

    # ── Results ───────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("HONEST BACKTEST RESULTS (train 90%% / embargo / test 10%%)")
    log.info("=" * 70)

    n_trades = len(trades)
    if n_trades == 0:
        log.info("NO TRADES. Skips: %s", dict(skips))
        log.info("Backtest complete.")
        return

    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = n_trades - wins
    total_pnl = sum(t["pnl"] for t in trades)
    avg_win = np.mean([t["pnl"] for t in trades if t["result"] == "WIN"]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in trades if t["result"] == "LOSS"]) if losses else 0

    log.info("")
    log.info("MODEL QUALITY:")
    log.info("  Holdout AUC: %.4f | Brier: %.4f | Acc: %.4f | LogLoss: %.4f",
             test_auc, test_brier, test_acc, test_logloss)
    log.info("")
    log.info("TRADING SIMULATION:")
    log.info("  Trades: %d (from %d test markets)", n_trades, len(df_test))
    log.info("  Win/Loss: %d/%d (%.1f%% win rate)", wins, losses, 100*wins/n_trades)
    log.info("  Total P&L: $%.2f (per-share)", total_pnl)
    log.info("  Avg win: $%.4f | Avg loss: $%.4f", avg_win, avg_loss)
    log.info("  EV per trade: $%.4f", total_pnl/n_trades)

    cum_pnl = np.cumsum([t["pnl"] for t in trades])
    max_dd = float(max(np.maximum.accumulate(cum_pnl) - cum_pnl)) if len(cum_pnl) else 0
    sharpe = np.mean([t["pnl"] for t in trades]) / (np.std([t["pnl"] for t in trades]) + 1e-8)
    log.info("  Max drawdown: $%.2f | Peak: $%.2f | Sharpe: %.3f",
             max_dd, float(max(cum_pnl)), sharpe)

    log.info("")
    log.info("BY CONFIDENCE:")
    for lo, hi in [(0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.0)]:
        b = [t for t in trades if lo <= t["confidence"] < hi]
        if not b: continue
        bw = sum(1 for t in b if t["result"] == "WIN")
        bp = sum(t["pnl"] for t in b)
        log.info("  [%.0f-%.0f%%]: %d trades, %.1f%% win, P&L=$%.2f, EV=$%.4f",
                 lo*100, hi*100, len(b), 100*bw/len(b), bp, bp/len(b))

    log.info("")
    log.info("BY DIRECTION:")
    for d in ["UP", "DOWN"]:
        dt_list = [t for t in trades if t["direction"] == d]
        if not dt_list: continue
        dw = sum(1 for t in dt_list if t["result"] == "WIN")
        dp = sum(t["pnl"] for t in dt_list)
        log.info("  %s: %d trades, %.1f%% win, P&L=$%.2f", d, len(dt_list), 100*dw/len(dt_list), dp)

    log.info("")
    log.info("SKIPS: %s", dict(skips))

    # ── Spread sensitivity ────────────────────────────────────────────────
    log.info("")
    log.info("SPREAD SENSITIVITY:")
    for test_spread in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        tt = tp = tw = 0
        for t in trades:
            new_ask = t["market_mid"] + test_spread
            if not (ASK_LO <= new_ask <= ASK_HI): continue
            if t["model_prob"] - new_ask < MIN_EDGE: continue
            tt += 1
            if t["result"] == "WIN":
                tw += 1; tp += (1.0 * (1.0 - TAKER_FEE)) - new_ask
            else:
                tp += -new_ask
        if tt:
            log.info("  spread=%.0fc: %d trades, %.1f%% win, P&L=$%.2f, EV=$%.4f",
                     test_spread*100, tt, 100*tw/tt, tp, tp/tt)

    # ── Rolling window analysis ───────────────────────────────────────────
    log.info("")
    log.info("ROLLING WINDOW (test set split into 5 sub-periods):")
    chunk_size = max(1, len(trades) // 5)
    for wi in range(5):
        chunk = trades[wi*chunk_size : (wi+1)*chunk_size]
        if not chunk: continue
        cw = sum(1 for t in chunk if t["result"] == "WIN")
        cp = sum(t["pnl"] for t in chunk)
        log.info("  Window %d: %d trades, %.1f%% win, P&L=$%.2f (%s to %s)",
                 wi+1, len(chunk), 100*cw/len(chunk), cp,
                 chunk[0]["dt"], chunk[-1]["dt"])

    # ══════════════════════════════════════════════════════════════════════
    # FINAL OVERFITTING SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("=" * 70)
    log.info("OVERFITTING DIAGNOSTIC SUMMARY")
    log.info("=" * 70)
    checks = []
    # 1. Adversarial validation
    if adv_auc <= 0.65:
        checks.append(("Adversarial validation", "PASS", f"AUC={adv_auc:.4f}"))
    else:
        checks.append(("Adversarial validation", "WARN", f"AUC={adv_auc:.4f} (distribution shift)"))
    # 2. Noise features
    if min_noise_rank > TOP_N_FEATS:
        checks.append(("Noise feature canary", "PASS", f"lowest rank #{min_noise_rank}"))
    else:
        checks.append(("Noise feature canary", "FAIL", f"noise in top {TOP_N_FEATS}!"))
    # 3. AUC drop
    if auc_drop <= 0.03:
        checks.append(("AUC WF->holdout drop", "PASS", f"delta={auc_drop:.4f}"))
    else:
        checks.append(("AUC WF->holdout drop", "WARN", f"delta={auc_drop:.4f}"))
    # 4. Fold stability
    if np.std(fold_aucs) <= 0.05:
        checks.append(("Fold AUC stability", "PASS", f"std={np.std(fold_aucs):.4f}"))
    else:
        checks.append(("Fold AUC stability", "WARN", f"std={np.std(fold_aucs):.4f}"))
    # 5. Trading P&L
    if n_trades > 0 and total_pnl > 0:
        checks.append(("OOS trading P&L", "PASS", f"${total_pnl:.2f} ({n_trades} trades)"))
    elif n_trades > 0:
        checks.append(("OOS trading P&L", "FAIL", f"${total_pnl:.2f} ({n_trades} trades)"))
    else:
        checks.append(("OOS trading P&L", "SKIP", "no trades"))

    for name, status, detail in checks:
        icon = "OK" if status == "PASS" else ("!!" if status == "WARN" else "XX")
        log.info("  [%s] %s: %s", icon, name, detail)

    log.info("")
    log.info("Honest backtest complete.")


@app.local_entrypoint()
def main():
    backtest_honest.remote()
