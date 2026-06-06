"""
backtest_v22.py — Faithful backtest of v22 champion against live trading rules
===============================================================================
Replicates the EXACT decision logic from deploy/live_trader.py:
  1. Builds features identically to training (same code path)
  2. Runs model prediction
  3. Applies ALL filters: min_conf 60%, ask range [0.38, 0.90], 
     edge_ask >= 10%, edge_mid >= 5%
  4. Computes P&L with 2% taker fee on wins

Ask price simulation:
  - We don't have historical order book asks, so we simulate using
    the market's up_ratio (volume-weighted) as a proxy for market mid
  - Ask = mid + half_spread (spread = 2-5 cents typical on Polymarket)
  - This is conservative: real asks can be better or worse

Walk-forward: only predicts on data the model hasn't seen (last 20% by time)
to avoid in-sample bias.
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
        "huggingface_hub>=0.26",
    )
)

app = modal.App("btc-v22-backtest", image=image)


@app.function(
    cpu=4,
    memory=16384,
    timeout=3600,
    secrets=[modal.Secret.from_name("hf-token")],
)
def backtest_v22():
    import gc, json, logging, math, os, pickle, sys, time, warnings
    from datetime import datetime, timezone
    from pathlib import Path
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    # ── Config: match live_trader.py exactly ──────────────────────────────
    HF_TOKEN      = os.environ.get("HF_TOKEN", "")
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
    MIN_CONFIDENCE = 0.60
    MIN_EDGE       = 0.10
    MIN_EDGE_MID   = 0.05
    ASK_LO, ASK_HI = 0.38, 0.90
    TAKER_FEE      = 0.02
    SLOT_DURATION  = 300
    OBSERVE_SECS   = 180
    HALF_SPREAD    = 0.02  # simulated half-spread (conservative)
    WARMUP_SLOTS   = 3     # match DataQualityGate

    DATA_DIR = Path("/tmp/btc_data")
    DATA_DIR.mkdir(exist_ok=True)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")

    # ── Step 1: Download data from HF ─────────────────────────────────────
    log.info("Step 1: Downloading data from HF...")
    DATA_FILES = {
        "all_markets.csv": "data/all_markets.csv",
        "new_markets.csv": "data/new_markets.csv",
        "ticks_btc_full_clean.parquet": "data/ticks_btc_full_clean.parquet",
        "new_ticks_pmdata.parquet": "data/new_ticks_pmdata.parquet",
        "binance_spot_full.parquet": "data/binance_spot_full.parquet",
        "binance_spot_local.parquet": "data/binance_spot_local.parquet",
        "ob_features_full.parquet": "data/ob_features_full.parquet",
        "champion.pkl": "champion.pkl",
        "champion_meta.json": "champion_meta.json",
    }
    for local_name, hf_path in DATA_FILES.items():
        local_path = DATA_DIR / local_name
        if local_path.exists():
            log.info("  %s cached", local_name)
            continue
        try:
            import shutil
            downloaded = hf_hub_download(
                repo_id=HF_MODEL_REPO, filename=hf_path,
                token=HF_TOKEN, repo_type="model",
                local_dir=str(DATA_DIR), local_dir_use_symlinks=False,
            )
            src = DATA_DIR / hf_path
            if src.exists() and not local_path.exists():
                shutil.move(str(src), str(local_path))
            log.info("  Downloaded %s", local_name)
        except Exception as e:
            log.warning("  Could not download %s: %s", local_name, e)

    # ── Step 2: Load champion model ───────────────────────────────────────
    log.info("Step 2: Loading champion model...")
    with open(DATA_DIR / "champion.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    features = bundle["features"]
    obs_secs = bundle.get("obs_secs", 180)
    log.info("Champion: %s, %d features, obs_secs=%d, AUC=%.4f",
             bundle["version"], len(features), obs_secs, bundle.get("wf_auc", 0))

    # ── Step 3: Load markets ──────────────────────────────────────────────
    log.info("Step 3: Loading markets...")
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
    log.info("Total markets: %d (%.1f%% UP)", len(markets), 100 * markets["target"].mean())

    markets["rank"] = range(len(markets))
    slot_to_rank = dict(zip(markets["slot_ts"], markets["rank"]))
    all_targets = markets["target"].values
    all_slot_ts = markets["slot_ts"].values
    all_mids = markets["market_id"].values

    # ── Step 4: Load OB features ──────────────────────────────────────────
    log.info("Step 4: Loading OB features...")
    ob_df = pd.read_parquet(str(DATA_DIR / "ob_features_full.parquet"))
    ob_df["market_id"] = ob_df["market_id"].astype(str)
    ob_by_market = ob_df.set_index("market_id").to_dict("index")
    ob_cols = [c for c in ob_df.columns if c != "market_id"]
    log.info("OB features: %d markets, %d features", len(ob_df), len(ob_cols))

    # ── Step 5: Load Binance spot ─────────────────────────────────────────
    log.info("Step 5: Loading Binance spot...")
    spot_dfs = []
    for sp in ["binance_spot_full.parquet", "binance_spot_local.parquet"]:
        sp_path = DATA_DIR / sp
        if sp_path.exists():
            spot_dfs.append(pd.read_parquet(str(sp_path)))
    spot_df = pd.concat(spot_dfs, ignore_index=True) if spot_dfs else pd.DataFrame()
    spot_df = spot_df.sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
    spot_ts_arr = (spot_df["timestamp_ms"].values // 1000).astype(np.int64)
    spot_px_arr = spot_df["close"].values.astype(np.float64)
    log.info("Binance spot: %d candles", len(spot_ts_arr))

    # ── Step 6: Load ticks ────────────────────────────────────────────────
    log.info("Step 6: Loading ticks...")
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
    log.info("Total ticks: %d", len(btc))

    slot_ts_map = dict(zip(markets["market_id"], markets["slot_ts"]))
    btc["slot_ts_val"] = btc["market_id"].map(slot_ts_map).astype(float)
    btc["t_sec"] = btc["timestamp_ms"].astype(float) / 1000 - btc["slot_ts_val"]
    btc = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < 300)]
    log.info("Ticks in [0, 300s): %d", len(btc))

    btc_grouped = {mid: grp for mid, grp in btc.groupby("market_id")}

    # ── Pre-compute aggregates ────────────────────────────────────────────
    n_windows = obs_secs // 30
    filtered = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < obs_secs)]
    up = filtered[filtered["outcome"] == "Up"]
    dn = filtered[filtered["outcome"] == "Down"]
    vol_up = up.groupby("market_id")["size_usdc"].sum()
    vol_dn = dn.groupby("market_id")["size_usdc"].sum()
    vol_tot = vol_up.add(vol_dn, fill_value=0)
    slot_up_ratio = vol_up / vol_tot.clip(lower=1e-9)
    slot_nticks = filtered.groupby("market_id").size()

    def spot_at(ts_s):
        idx = np.searchsorted(spot_ts_arr, ts_s, side="right") - 1
        idx = max(0, min(idx, len(spot_px_arr) - 1))
        return float(spot_px_arr[idx])

    def _ur(df_sub):
        u = df_sub[df_sub["outcome"] == "Up"]["size_usdc"].sum()
        d = df_sub[df_sub["outcome"] == "Down"]["size_usdc"].sum()
        t = u + d
        return u / t if t > 0 else 0.5

    def _ur_w(df_sub, t0, t1):
        w = df_sub[(df_sub["t_sec"] >= t0) & (df_sub["t_sec"] < t1)]
        return _ur(w) if len(w) else 0.5

    # ── Step 7: Build features & run backtest ─────────────────────────────
    # Walk-forward: use last 20% of data for out-of-sample backtest
    n_total = len(markets)
    n_test_start = int(n_total * 0.8)
    test_markets = markets.iloc[n_test_start:].copy()
    log.info("Backtest on last 20%%: %d markets (from rank %d)", len(test_markets), n_test_start)
    log.info("Test period: %s to %s",
             datetime.fromtimestamp(test_markets["slot_ts"].iloc[0], tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
             datetime.fromtimestamp(test_markets["slot_ts"].iloc[-1], tz=timezone.utc).strftime("%Y-%m-%d %H:%M"))

    # Slot history ring buffer (like live trader)
    slot_history = []  # list of {"slot_ts", "up_ratio", "target"}
    HIST_MAX = 50

    # Seed history from training data (like live trader does from chain)
    seed_start = max(0, n_test_start - 25)
    for i in range(seed_start, n_test_start):
        row = markets.iloc[i]
        mid = row["market_id"]
        slot_history.append({
            "slot_ts": int(row["slot_ts"]),
            "up_ratio": float(slot_up_ratio.get(mid, 0.5)),
            "target": int(row["target"]),
        })
    log.info("Seeded %d history slots", len(slot_history))

    # Results tracking
    trades = []
    skips = defaultdict(int)
    warmup_count = 0

    for idx, (_, row) in enumerate(test_markets.iterrows()):
        mid = row["market_id"]
        slot_ts = int(row["slot_ts"])
        target = int(row["target"])
        ext_rank = slot_to_rank.get(slot_ts, n_test_start + idx)

        # ── Warmup (match DataQualityGate) ────────────────────────────────
        if warmup_count < WARMUP_SLOTS:
            warmup_count += 1
            # Still update history
            slot_history.append({
                "slot_ts": slot_ts,
                "up_ratio": float(slot_up_ratio.get(mid, 0.5)),
                "target": target,
            })
            if len(slot_history) > HIST_MAX:
                slot_history = slot_history[-HIST_MAX:]
            skips["warmup"] += 1
            continue

        # ── Build features (identical to training) ────────────────────────
        grp_full = btc_grouped.get(mid)
        if grp_full is not None:
            grp = grp_full[(grp_full["t_sec"] >= 0) & (grp_full["t_sec"] < obs_secs)]
        else:
            grp = None
        n = len(grp) if grp is not None else 0

        if n > 0:
            ur = slot_up_ratio.get(mid, 0.5)
            vt = vol_tot.get(mid, 0.0)
            ntx = slot_nticks.get(mid, 0)

            up_vals = grp[grp["outcome"] == "Up"]["size_usdc"].values
            dn_vals = grp[grp["outcome"] == "Down"]["size_usdc"].values

            w_vals = []
            for wi in range(n_windows):
                t0_w, t1_w = wi * 30, (wi + 1) * 30
                w_vals.append(_ur_w(grp, t0_w, t1_w))
            while len(w_vals) < 6:
                w_vals.append(0.5)

            up_g = grp[grp["outcome"] == "Up"]
            dn_g = grp[grp["outcome"] == "Down"]
            def vwap(g):
                return (g["price"] * g["size_usdc"]).sum() / g["size_usdc"].sum() if len(g) else 0.5
            vwap_up = vwap(up_g)
            vwap_dn = vwap(dn_g)

            all_sorted = grp.sort_values("t_sec")
            if len(all_sorted) > 1:
                w_exp = np.exp(-0.02 * (obs_secs - all_sorted["t_sec"].values))
                ur_up = (all_sorted["outcome"] == "Up").astype(float).values
                tw_ur = np.average(ur_up * all_sorted["size_usdc"].values,
                                   weights=w_exp) / (np.average(all_sorted["size_usdc"].values, weights=w_exp) + 1e-9)
            else:
                tw_ur = ur

            buy_sz = grp[grp["side"] == "BUY"]["size_usdc"].sum()
            buy_ratio = buy_sz / vt if vt > 0 else 0.5

            if n_windows >= 4:
                half = n_windows // 2
                momentum = np.mean(w_vals[half:n_windows]) - np.mean(w_vals[:half])
            else:
                momentum = w_vals[-1] - w_vals[0] if n_windows >= 2 else 0.0

            stability = np.std(w_vals[:n_windows])
            avg_up = up_vals.mean() if len(up_vals) else 0
            avg_dn = dn_vals.mean() if len(dn_vals) else 0

            feat = {
                "btc_up_ratio": ur,
                "btc_n_ticks": float(n),
                "btc_buy_ratio": buy_ratio,
                "btc_tw_up_ratio": tw_ur,
                "btc_momentum": momentum,
                "btc_vwap_spread": vwap_up - vwap_dn,
                "btc_vwap_up": vwap_up,
                "btc_vwap_dn": vwap_dn,
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
            feat["btc_up_ratio"] = 0.5
            feat["btc_vwap_up"] = 0.5
            feat["btc_vwap_dn"] = 0.5
            feat["btc_tw_up_ratio"] = 0.5
            feat["btc_buy_ratio"] = 0.5
            ur = 0.5

        # ── Z-scores ─────────────────────────────────────────────────────
        def _hist_ur(lookback=20):
            vals = []
            for d in range(1, lookback + 1):
                prev_r = ext_rank - d
                if prev_r < 0:
                    break
                prev_mid = all_mids[prev_r]
                v = slot_up_ratio.get(prev_mid, None)
                if v is not None:
                    vals.append(v)
            return vals

        hist_vals = _hist_ur(20)
        if len(hist_vals) >= 3:
            mu20 = np.mean(hist_vals)
            sd20 = max(np.std(hist_vals) + 1e-6, 0.01)
            feat["btc_up_ratio_zscore_20s"] = float(np.clip((feat["btc_up_ratio"] - mu20) / sd20, -5, 5))
            feat["btc_up_w5_zscore"] = float(np.clip((feat["btc_up_w5"] - mu20) / sd20, -5, 5))
        else:
            feat["btc_up_ratio_zscore_20s"] = 0.0
            feat["btc_up_w5_zscore"] = 0.0

        hist5 = hist_vals[:5]
        if len(hist5) >= 2:
            mu5 = np.mean(hist5)
            sd5 = max(np.std(hist5) + 1e-6, 0.01)
            feat["btc_up_ratio_zscore_5s"] = float(np.clip((feat["btc_up_ratio"] - mu5) / sd5, -5, 5))
        else:
            feat["btc_up_ratio_zscore_5s"] = 0.0

        # ── Spot features ────────────────────────────────────────────────
        obs_end_ts = slot_ts + obs_secs
        px_now = spot_at(obs_end_ts)

        def pre_ret(h):
            px_h = spot_at(slot_ts - h * 3600)
            return (px_now / px_h - 1) if px_h > 0 else 0.0

        feat["btc_pre_5m_ret"] = (px_now / spot_at(slot_ts - 300) - 1) if spot_at(slot_ts - 300) > 0 else 0.0
        feat["btc_pre_30m_ret"] = pre_ret(0.5)
        feat["btc_pre_1h_ret"] = pre_ret(1)
        feat["btc_pre_4h_ret"] = pre_ret(4)

        px_1h_ago = spot_at(slot_ts - 3600)
        px_4h_ago = spot_at(slot_ts - 4 * 3600)
        if px_now > 0 and px_1h_ago > 0 and px_4h_ago > 0 and abs(px_now - px_4h_ago) > 1:
            feat["btc_pre_1h_4h_ratio"] = (px_now - px_1h_ago) / (px_now - px_4h_ago + 1e-9)
        else:
            feat["btc_pre_1h_4h_ratio"] = 0.0

        t0_idx = int(np.searchsorted(spot_ts_arr, slot_ts, side="left"))
        t1_idx = int(np.searchsorted(spot_ts_arr, slot_ts + obs_secs, side="right"))
        if t1_idx > t0_idx:
            inslot_px = spot_px_arr[t0_idx:t1_idx]
            feat["btc_inslot_ret"] = float(inslot_px[-1] / inslot_px[0] - 1) if inslot_px[0] > 0 else 0.0
        else:
            feat["btc_inslot_ret"] = 0.0

        px_k = px_now / 1000
        feat["btc_dist_1k"] = min(px_k - math.floor(px_k), math.ceil(px_k) - px_k)

        # ── Lag features (with staleness guard like live) ─────────────────
        lag_streak = 0
        streak_dir = None

        for lag_n in range(1, 6):
            prev_rank = ext_rank - lag_n
            if prev_rank >= 0:
                prev_target = int(all_targets[prev_rank])
                prev_slot = int(all_slot_ts[prev_rank])
                prev_mid_id = all_mids[prev_rank]

                time_gap = slot_ts - prev_slot
                if time_gap > lag_n * SLOT_DURATION * 3:
                    feat[f"lag_{lag_n}_outcome"] = 0.5
                    feat[f"prev_slot_up_ratio_{lag_n}"] = 0.5
                    feat[f"prev_slot_n_ticks_{lag_n}"] = 0.0
                    feat[f"prev_slot_vol_{lag_n}"] = 0.0
                    continue

                feat[f"lag_{lag_n}_outcome"] = float(prev_target)
                feat[f"prev_slot_up_ratio_{lag_n}"] = float(slot_up_ratio.get(prev_mid_id, 0.5))
                feat[f"prev_slot_n_ticks_{lag_n}"] = float(slot_nticks.get(prev_mid_id, 0.0))
                feat[f"prev_slot_vol_{lag_n}"] = float(vol_tot.get(prev_mid_id, 0.0))

                if lag_n == 1:
                    streak_dir = prev_target
                    lag_streak = 1
                elif prev_target == streak_dir:
                    lag_streak += 1
            else:
                for k in [f"lag_{lag_n}_outcome", f"prev_slot_up_ratio_{lag_n}",
                          f"prev_slot_n_ticks_{lag_n}", f"prev_slot_vol_{lag_n}"]:
                    feat[k] = 0.0

        feat["lag_streak"] = float(lag_streak)

        # ── Temporal features ────────────────────────────────────────────
        dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        dow = dt.weekday()

        feat["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        feat["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        feat["dow_sin"] = math.sin(2 * math.pi * dow / 7)
        feat["dow_cos"] = math.cos(2 * math.pi * dow / 7)
        feat["hour_x_up_ratio"] = feat["btc_up_ratio"] * (hour / 24.0)
        feat["hour_x_tw_ur"] = feat.get("btc_tw_up_ratio", 0.5) * (hour / 24.0)

        # ── L2 Orderbook features ────────────────────────────────────────
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
            feat["x_imb_x_ur"] = 0.0
            feat["x_depth_x_momentum"] = 0.0
            feat["x_spread_x_vol"] = 0.0
            feat["x_ob_drift_x_inslot"] = 0.0
            feat["x_fill_imb_x_buy"] = 0.0

        # ── Predict ──────────────────────────────────────────────────────
        X = pd.DataFrame([[feat.get(f, 0.0) for f in features]], columns=features)
        prob_up = float(model.predict_proba(X)[0, 1])

        # ── Simulate live trading decisions ──────────────────────────────
        # Direction = side with higher confidence
        if prob_up >= 0.5:
            direction = "UP"
            confidence = prob_up
            model_prob = prob_up
        else:
            direction = "DOWN"
            confidence = 1.0 - prob_up
            model_prob = 1.0 - prob_up

        # Filter 1: min confidence
        if confidence < MIN_CONFIDENCE:
            skips["low_conf"] += 1
            slot_history.append({"slot_ts": slot_ts, "up_ratio": float(ur), "target": target})
            if len(slot_history) > HIST_MAX:
                slot_history = slot_history[-HIST_MAX:]
            continue

        # Simulate ask price from market data
        # Market mid = up_ratio from ticks (volume-weighted probability)
        # This is the best proxy we have for what the CLOB mid was
        market_mid_up = float(ur) if n > 0 else 0.5
        market_mid_down = 1.0 - market_mid_up

        if direction == "UP":
            market_mid = market_mid_up
        else:
            market_mid = market_mid_down

        # Simulate ask = mid + half_spread (you have to pay above mid to buy)
        ask_price = market_mid + HALF_SPREAD

        # Filter 2: ask range
        if not (ASK_LO <= ask_price <= ASK_HI):
            skips["ask_range"] += 1
            slot_history.append({"slot_ts": slot_ts, "up_ratio": float(ur), "target": target})
            if len(slot_history) > HIST_MAX:
                slot_history = slot_history[-HIST_MAX:]
            continue

        # Filter 3: edge vs ask
        edge_vs_ask = model_prob - ask_price
        if edge_vs_ask < MIN_EDGE:
            skips["low_edge_ask"] += 1
            slot_history.append({"slot_ts": slot_ts, "up_ratio": float(ur), "target": target})
            if len(slot_history) > HIST_MAX:
                slot_history = slot_history[-HIST_MAX:]
            continue

        # Filter 4: edge vs mid
        edge_vs_mid = model_prob - market_mid
        if edge_vs_mid < MIN_EDGE_MID:
            skips["low_edge_mid"] += 1
            slot_history.append({"slot_ts": slot_ts, "up_ratio": float(ur), "target": target})
            if len(slot_history) > HIST_MAX:
                slot_history = slot_history[-HIST_MAX:]
            continue

        # Filter 5: ask-mid divergence (> 0.20)
        if abs(ask_price - market_mid) > 0.20:
            skips["ask_mid_diverge"] += 1
            slot_history.append({"slot_ts": slot_ts, "up_ratio": float(ur), "target": target})
            if len(slot_history) > HIST_MAX:
                slot_history = slot_history[-HIST_MAX:]
            continue

        # ── Execute trade ────────────────────────────────────────────────
        actual = "UP" if target == 1 else "DOWN"
        cost = ask_price  # per-share cost (1 share for simplicity)
        if direction == actual:
            # WIN: payout = 1.00 per share, minus taker fee
            pnl = (1.0 * (1.0 - TAKER_FEE)) - cost
            result = "WIN"
        else:
            # LOSS: lose the cost
            pnl = -cost
            result = "LOSS"

        trades.append({
            "slot_ts": slot_ts,
            "direction": direction,
            "actual": actual,
            "result": result,
            "confidence": round(confidence, 4),
            "model_prob": round(model_prob, 4),
            "ask_price": round(ask_price, 4),
            "market_mid": round(market_mid, 4),
            "edge_ask": round(edge_vs_ask, 4),
            "edge_mid": round(edge_vs_mid, 4),
            "pnl": round(pnl, 4),
            "target": target,
            "dt": dt.strftime("%Y-%m-%d %H:%M"),
        })

        # Update history
        slot_history.append({"slot_ts": slot_ts, "up_ratio": float(ur), "target": target})
        if len(slot_history) > HIST_MAX:
            slot_history = slot_history[-HIST_MAX:]

        if idx % 500 == 0 and idx > 0:
            wins = sum(1 for t in trades if t["result"] == "WIN")
            total_pnl = sum(t["pnl"] for t in trades)
            log.info("Progress %d/%d: %d trades, %d wins, P&L=%.2f",
                     idx, len(test_markets), len(trades), wins, total_pnl)

    # ── Step 8: Results ───────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("BACKTEST RESULTS — v22 champion")
    log.info("=" * 70)

    n_trades = len(trades)
    if n_trades == 0:
        log.info("NO TRADES — all filtered out")
        log.info("Skip reasons: %s", dict(skips))
        return

    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = n_trades - wins
    win_rate = wins / n_trades

    total_pnl = sum(t["pnl"] for t in trades)
    avg_win = np.mean([t["pnl"] for t in trades if t["result"] == "WIN"]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in trades if t["result"] == "LOSS"]) if losses else 0

    # By confidence bucket
    log.info("")
    log.info("OVERALL:")
    log.info("  Trades: %d (from %d test markets)", n_trades, len(test_markets))
    log.info("  Win/Loss: %d/%d (%.1f%% win rate)", wins, losses, win_rate * 100)
    log.info("  Total P&L: $%.2f (per-share basis)", total_pnl)
    log.info("  Avg win: $%.4f | Avg loss: $%.4f", avg_win, avg_loss)
    log.info("  Expected value per trade: $%.4f", total_pnl / n_trades)
    log.info("")

    # By confidence bucket
    conf_buckets = [(0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.0)]
    log.info("BY CONFIDENCE BUCKET:")
    for lo, hi in conf_buckets:
        bucket = [t for t in trades if lo <= t["confidence"] < hi]
        if not bucket:
            continue
        bw = sum(1 for t in bucket if t["result"] == "WIN")
        bpnl = sum(t["pnl"] for t in bucket)
        log.info("  [%.0f%%–%.0f%%]: %d trades, %d wins (%.1f%%), P&L=$%.2f, EV=$%.4f",
                 lo*100, hi*100, len(bucket), bw, 100*bw/len(bucket),
                 bpnl, bpnl/len(bucket))

    # By direction
    log.info("")
    log.info("BY DIRECTION:")
    for d in ["UP", "DOWN"]:
        dtrades = [t for t in trades if t["direction"] == d]
        if not dtrades:
            continue
        dw = sum(1 for t in dtrades if t["result"] == "WIN")
        dpnl = sum(t["pnl"] for t in dtrades)
        log.info("  %s: %d trades, %d wins (%.1f%%), P&L=$%.2f",
                 d, len(dtrades), dw, 100*dw/len(dtrades), dpnl)

    # Skip reasons
    log.info("")
    log.info("SKIP REASONS:")
    total_skips = sum(skips.values())
    for reason, count in sorted(skips.items(), key=lambda x: -x[1]):
        log.info("  %s: %d (%.1f%%)", reason, count, 100*count/len(test_markets))

    # Cumulative P&L curve stats
    cum_pnl = np.cumsum([t["pnl"] for t in trades])
    max_dd = 0
    peak = 0
    for v in cum_pnl:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    log.info("")
    log.info("RISK METRICS:")
    log.info("  Max drawdown: $%.2f", max_dd)
    log.info("  Peak P&L: $%.2f", max(cum_pnl))
    log.info("  Final P&L: $%.2f", cum_pnl[-1])
    sharpe = (np.mean([t["pnl"] for t in trades]) / (np.std([t["pnl"] for t in trades]) + 1e-8))
    log.info("  Per-trade Sharpe: %.3f", sharpe)

    # Time-based analysis
    log.info("")
    log.info("BY TIME OF DAY (UTC):")
    hour_buckets = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        h = int(t["dt"].split(" ")[1].split(":")[0])
        hour_buckets[h]["trades"] += 1
        if t["result"] == "WIN":
            hour_buckets[h]["wins"] += 1
        hour_buckets[h]["pnl"] += t["pnl"]
    for h in sorted(hour_buckets):
        b = hour_buckets[h]
        log.info("  %02d:00: %d trades, %.1f%% win, P&L=$%.2f",
                 h, b["trades"], 100*b["wins"]/b["trades"], b["pnl"])

    # Spread sensitivity analysis
    log.info("")
    log.info("SPREAD SENSITIVITY (what if spread was different?):")
    for test_spread in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
        test_trades = 0
        test_wins = 0
        test_pnl = 0.0
        for t in trades:
            # Recalculate with different spread
            new_ask = t["market_mid"] + test_spread
            if not (ASK_LO <= new_ask <= ASK_HI):
                continue
            new_edge = t["model_prob"] - new_ask
            if new_edge < MIN_EDGE:
                continue
            test_trades += 1
            cost = new_ask
            if t["result"] == "WIN":
                pnl_t = (1.0 * (1.0 - TAKER_FEE)) - cost
                test_wins += 1
            else:
                pnl_t = -cost
            test_pnl += pnl_t
        if test_trades > 0:
            log.info("  spread=%.0fc: %d trades, %.1f%% win, P&L=$%.2f, EV=$%.4f",
                     test_spread*100, test_trades, 100*test_wins/test_trades,
                     test_pnl, test_pnl/test_trades)

    log.info("")
    log.info("Backtest complete.")


@app.local_entrypoint()
def main():
    backtest_v22.remote()
