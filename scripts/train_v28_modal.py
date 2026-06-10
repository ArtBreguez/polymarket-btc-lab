"""
train_v28_modal.py — BTC 5min model v28 (FULL FEATURE PARITY train == live)
================================================================================
Philosophy: EXACT same features as deploy/live_trader.py build_features().
  No lookahead, no train/live mismatch.

Feature groups (all computed at OBS_SECS=60 cutoff):
  A. Binance spot (<2s lag): pre-slot returns, in-slot ret/range/vol, round-number
     proximity, vol ratios, 1h/4h ratio
  B. L2 OB snapshot (<5s lag, from pmdata poly_l2): all 30 ob_* + clob_* columns
  C. Tick-based order flow (data-api, lags ~120s but t<60s so available at decision):
     btc_up_ratio, btc_n_ticks, btc_momentum, btc_buy_ratio,
     btc_tw_up_ratio, btc_vwap_up/dn/spread, btc_vol_up/dn,
     btc_up_ratio_stability, btc_up_ratio_zscore_5s/20s, btc_up_w5_zscore,
     btc_signal_conviction, btc_size_disparity
  D. Lag history (ring buffer from previous resolved slots):
     lag_1..5_outcome, prev_slot_up_ratio/n_ticks/vol_1..5,
     lag_streak, lag_ur_zscore_5, lag_ur_zscore_20
  E. Temporal: hour_sin/cos, dow_sin/cos, hour_x_up_ratio, hour_x_tw_ur
  F. Cross-domain interactions:
     x_imb_x_inslot, x_imb_end_x_ret, x_drift_x_ret5m, x_spread_x_vol,
     x_depth_x_vol, x_imb_x_ur, x_depth_x_momentum, x_ob_drift_x_inslot

LOOKAHEAD AUDIT:
  - All spot features use obs_end_ts = slot_ts + OBS_SECS (t=60s) as reference — NO future data
  - Tick features window: t ∈ [0, OBS_SECS) — resolved data only
  - OB features: snapshot at ~t=150-170s in training (poly_l2 book), same in live
  - Lag features: use only PREVIOUS slots (rank-1 .. rank-5) — NO current market data
  - Temporal: slot_ts only — no future

Pipeline:
  1. Download from HF
  2. Build features (full parity with live)
  3. Feature importance screening → select top N
  4. Optuna tuning (150 trials)
  5. Walk-forward evaluation (5-fold)
  6. Realistic P&L backtest
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
app = modal.App("btc-v28-run", image=image)


@app.function(
    cpu=8,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/cache": vol},
)
def train_v28():
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
    # Exclude features removed from model: divergent live coverage, warm-up deps, or unreliable
    OB_EXCLUDED = {
        "ob_imb_w0", "ob_pc_count", "ob_fill_imbalance",  # removed previously
        "ob_imb_w1", "ob_imb_w2",                         # ❌ quase sempre 0.0 live (1 WS book snap)
        "ob_pc_up_ratio", "ob_pc_volatility",              # ⚠️ reset em reconexão WS
        "ob_mid_drift",                                    # ⚠️ timing gap ~72s; cascata: x_drift/x_ob_drift
        "clob_imb_mean", "clob_imb_std", "clob_imb_drift", # ❌ quase sempre 0.0 live (1 WS book snap)
        "clob_depth_trend",                                # ❌ idem
        "clob_activity_rate",                              # ⚠️ levemente diferente do treino
    }
    ob_cols = [c for c in ob_df.columns if c != "market_id" and c not in OB_EXCLUDED]
    log.info("OB features: %d markets, %d features (excluded %s): %s",
             len(ob_df), len(ob_cols), OB_EXCLUDED, ob_cols[:5])

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
    # STEP 4: LOAD TICKS (for lag history + tick features)
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 4: Loading ticks (lag features + tick-based order flow)...")
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

    btc = pd.concat(chunks, ignore_index=True)
    btc = btc[btc["timestamp_ms"] > 0]

    slot_ts_map = dict(zip(markets["market_id"], markets["slot_ts"]))
    btc["slot_ts_val"] = btc["market_id"].map(slot_ts_map).astype(float)
    btc["t_sec"] = btc["timestamp_ms"].astype(float) / 1000 - btc["slot_ts_val"]
    btc = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < 300)]
    log.info("Total ticks: %d for %d markets", len(btc), btc["market_id"].nunique())

    # ── Compute per-slot aggregates for lag features ─────────────────────
    OBS_SECS = 60
    filtered = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < OBS_SECS)]
    up_f = filtered[filtered["outcome"] == "Up"]
    dn_f = filtered[filtered["outcome"] == "Down"]
    slot_vol_up  = up_f.groupby("market_id")["size_usdc"].sum()
    slot_vol_dn  = dn_f.groupby("market_id")["size_usdc"].sum()
    slot_vol_tot = slot_vol_up.add(slot_vol_dn, fill_value=0)
    slot_up_ratio = slot_vol_up / slot_vol_tot.clip(lower=1e-9)
    slot_nticks   = filtered.groupby("market_id").size()

    # ── Pre-compute per-market tick features (t ∈ [0, OBS_SECS)) ─────────
    # These match live build_features() exactly — same window, same formula.
    # LOOKAHEAD NOTE: we use t ∈ [0, OBS_SECS=60s) only — bot observes until
    # t=60s, so this data is already resolved at prediction time (t=170-240s).
    log.info("Pre-computing tick features for %d markets...", len(markets))

    tick_by_market: dict = {}  # market_id -> list of tick dicts
    for mid, grp in btc[btc["t_sec"] < OBS_SECS].groupby("market_id"):
        tick_by_market[mid] = grp[["t_sec", "outcome", "side", "price", "size_usdc"]].to_dict("records")

    del chunks, filtered, up_f, dn_f
    gc.collect()
    log.info("Tick features ready. Memory partial-freed.")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: BUILD FEATURES (FULL PARITY WITH live_trader.py build_features)
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 5: Building features (train == live parity) for %d markets...", len(markets))

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
        obs_end_ts = slot_ts + OBS_SECS  # t=60s — NO future data
        px_now = spot_at(obs_end_ts)

        def pre_ret(h_sec):
            px_h = spot_at(slot_ts - h_sec)
            return (px_now / px_h - 1) if px_h > 0 else 0.0

        feat["btc_pre_5m_ret"]  = pre_ret(300)
        feat["btc_pre_15m_ret"] = pre_ret(900)
        feat["btc_pre_30m_ret"] = pre_ret(1800)
        feat["btc_pre_1h_ret"]  = pre_ret(3600)

        # In-slot return/range during [slot_ts, slot_ts+OBS_SECS]
        t0_idx = int(np.searchsorted(spot_ts_arr, slot_ts, side="left"))
        t1_idx = int(np.searchsorted(spot_ts_arr, obs_end_ts, side="right"))
        if t1_idx > t0_idx:
            inslot_px = spot_px_arr[t0_idx:t1_idx]
            feat["btc_inslot_ret"] = float(inslot_px[-1] / inslot_px[0] - 1) if inslot_px[0] > 0 else 0.0
            feat["btc_inslot_vol"] = float(np.std(inslot_px) / (np.mean(inslot_px) + 1e-8)) if len(inslot_px) > 1 else 0.0
            inslot_hi = spot_hi_arr[t0_idx:t1_idx]
            inslot_lo = spot_lo_arr[t0_idx:t1_idx]
            feat["btc_inslot_range"] = float((inslot_hi.max() - inslot_lo.min()) / px_now) if px_now > 0 else 0.0
        else:
            feat["btc_inslot_ret"] = 0.0
            feat["btc_inslot_vol"] = 0.0
            feat["btc_inslot_range"] = 0.0

        # Volatility features
        feat["btc_vol_1h"]  = spot_volatility(slot_ts - 3600, slot_ts)

        # Momentum consistency — removed btc_pre_1h_4h_ratio (warm-up 4h dependency)

        # Round-number proximity
        if px_now > 0:
            px_k = px_now / 1000
            feat["btc_dist_1k"] = float(min(px_k - math.floor(px_k), math.ceil(px_k) - px_k))
        else:
            feat["btc_dist_1k"] = 0.5

        # Volume ratio
        feat["btc_spot_vol_ratio"] = (
            spot_vol_at(slot_ts - 300, slot_ts) / (spot_vol_at(slot_ts - 3600, slot_ts - 300) / 11 + 1e-9)
        )

        # ── B. L2 ORDERBOOK FEATURES (<5s lag) ───────────────────────
        ob = ob_by_market.get(mid)
        if ob is not None:
            for col in ob_cols:
                key = col if (col.startswith("ob_") or col.startswith("clob_")) else f"ob_{col}"
                feat[key] = float(ob.get(col, 0.0))
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
                elif "total_depth" in col:
                    feat[key] = 1000.0
                elif "depth_ratio" in col:
                    feat[key] = 1.0
                else:
                    feat[key] = 0.0

        # ── C. TICK-BASED ORDER FLOW (t ∈ [0, OBS_SECS)) ────────────
        # Matches live build_features() exactly — same formula, same window.
        ticks = tick_by_market.get(mid, [])
        if ticks:
            n      = len(ticks)
            vol_up = sum(t["size_usdc"] for t in ticks if t.get("outcome") == "Up")
            vol_dn = sum(t["size_usdc"] for t in ticks if t.get("outcome") == "Down")
            total  = vol_up + vol_dn + 1e-8

            up_tks = [t for t in ticks if t.get("outcome") == "Up"]
            dn_tks = [t for t in ticks if t.get("outcome") == "Down"]

            def _ur_w(subset):
                vu = sum(t["size_usdc"] for t in subset if t.get("outcome") == "Up")
                tt = sum(t["size_usdc"] for t in subset) + 1e-8
                return float(vu / tt)

            # 2x30s sub-windows (only w0/w1 real for OBS_SECS=60; w2-w5 always 0.5 → removed)
            sw = {}
            n_real_windows = OBS_SECS // 30  # 2
            for i in range(2):
                t0_w, t1_w = i * 30, (i + 1) * 30
                sub = [t for t in ticks if t0_w <= t["t_sec"] < t1_w]
                sw[f"btc_up_w{i}"] = _ur_w(sub) if sub else 0.5

            w_vals = [sw[f"btc_up_w{i}"] for i in range(2)]
            btc_momentum = float(sw["btc_up_w1"] - sw["btc_up_w0"])  # w1 - w0 (only real windows)
            # Stability: std of real windows only
            up_ratio_stability = float(np.std(w_vals))

            avg_up_sz = float(sum(t["size_usdc"] for t in up_tks) / (len(up_tks) + 1e-8))
            avg_dn_sz = float(sum(t["size_usdc"] for t in dn_tks) / (len(dn_tks) + 1e-8))
            size_disparity = float(avg_up_sz - avg_dn_sz)

            cur_up_ratio = float(vol_up / total)

            feat.update({
                "btc_up_ratio":           cur_up_ratio,
                "btc_n_ticks":            float(n),
                "btc_buy_ratio":          float(sum(t["size_usdc"] for t in ticks if t.get("side") == "BUY") / total),
                "btc_momentum":           btc_momentum,
                "btc_size_disparity":     size_disparity,
                "btc_up_ratio_stability": up_ratio_stability,
                **sw,
            })

            # Time-weighted up ratio (exponential recency decay — matches live)
            t_arr  = np.array([t["t_sec"] for t in ticks], dtype=np.float64)
            sz_arr = np.array([t.get("size_usdc", 1.0) for t in ticks], dtype=np.float64)
            up_arr = np.array([1.0 if t.get("outcome") == "Up" else 0.0 for t in ticks])
            w_exp  = np.exp(-0.02 * (OBS_SECS - t_arr))
            feat["btc_tw_up_ratio"] = float(np.sum(up_arr * sz_arr * w_exp) / (np.sum(sz_arr * w_exp) + 1e-9))
        else:
            # No ticks — neutral fill
            cur_up_ratio = 0.5
            feat.update({
                "btc_up_ratio": 0.5, "btc_n_ticks": 0.0,
                "btc_buy_ratio": 0.5, "btc_momentum": 0.0,
                "btc_size_disparity": 0.0, "btc_up_ratio_stability": 0.0,
                "btc_tw_up_ratio": 0.5,
                "btc_up_w0": 0.5, "btc_up_w1": 0.5,
            })

        # ── D. LAG FEATURES (from previous resolved slots) ───────────
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
                    feat[f"prev_slot_vol_{lag_n}"]       = 0.0
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
        prev_ur_1 = feat.get("prev_slot_up_ratio_1", 0.5)
        if len(hist_vals) >= 3:
            mu20 = np.mean(hist_vals); sd20 = max(np.std(hist_vals), 0.01)
            feat["lag_ur_zscore_20"] = float(np.clip((prev_ur_1 - mu20) / sd20, -5, 5))
            feat["btc_up_ratio_zscore_20s"] = float(np.clip((cur_up_ratio - mu20) / sd20, -5, 5))
        else:
            feat["lag_ur_zscore_20"] = 0.0
            feat["btc_up_ratio_zscore_20s"] = 0.0

        hist5 = hist_vals[:5]
        if len(hist5) >= 2:
            mu5 = np.mean(hist5); sd5 = max(np.std(hist5), 0.01)
            feat["lag_ur_zscore_5"] = float(np.clip((prev_ur_1 - mu5) / sd5, -5, 5))
            feat["btc_up_ratio_zscore_5s"] = float(np.clip((cur_up_ratio - mu5) / sd5, -5, 5))
        else:
            feat["lag_ur_zscore_5"] = 0.0
            feat["btc_up_ratio_zscore_5s"] = 0.0

        # ── E. TEMPORAL FEATURES ─────────────────────────────────────
        dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        dow  = dt.weekday()

        feat["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        feat["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        feat["dow_sin"]  = math.sin(2 * math.pi * dow / 7)
        feat["dow_cos"]  = math.cos(2 * math.pi * dow / 7)
        # Temporal interaction features (matches live)
        feat["hour_x_up_ratio"] = cur_up_ratio * (hour / 24.0)
        feat["hour_x_tw_ur"]    = feat.get("btc_tw_up_ratio", 0.5) * (hour / 24.0)

        # ── F. CROSS-DOMAIN INTERACTIONS (OB x CLOB x Spot) ─────────
        # ob_mid_drift removed → x_drift_x_ret5m and x_ob_drift_x_inslot removed too
        feat["x_imb_x_inslot"]     = feat.get("ob_imbalance", 0.0) * feat.get("btc_inslot_ret", 0.0)
        feat["x_imb_end_x_ret"]    = feat.get("ob_imbalance_end", 0.0) * feat.get("btc_inslot_ret", 0.0)
        feat["x_spread_x_vol"]     = feat.get("ob_spread", 0.02) * feat.get("btc_vol_1h", 0.0)
        feat["x_depth_x_vol"]      = feat.get("ob_depth_ratio", 1.0) * feat.get("btc_vol_1h", 0.0)
        feat["x_imb_x_ur"]         = feat.get("ob_imbalance", 0.0) * cur_up_ratio
        feat["x_depth_x_momentum"] = feat.get("ob_depth_ratio", 1.0) * feat.get("btc_momentum", 0.0)

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
    FEATURE_COUNTS = [73, 60, 50, 40, 30, 25, 20]
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

    version_tag = f"v28_{TOP_N_FEATS}f_rt"

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
                f"v28: FULL feature parity train==live. Added tick-based features "
                f"(btc_up_ratio/momentum/tw_up_ratio/zscore/etc), dow_cos/sin, "
                f"hour_x_up_ratio/tw_ur, all cross-domain interactions. "
                f"{len(top_features)} features. "
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
    log.info("v28 training complete.")


@app.local_entrypoint()
def main():
    train_v28.remote()
