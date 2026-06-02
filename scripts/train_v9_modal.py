"""
train_v9_modal.py — BTC 5min model v9

Lessons from v8 permutation importance:
  #1: btc_up_ratio_zscore (multi-scale anomaly) — deepen
  #2-3: btc_up_w2/w5 (sub-window flow) — keep 6x30s
  Calibration: isotonic with 616 samples → overfits (~100pts/fold). Switch to sigmoid.
  Features: 63 → prune harder (imp_mean > 0.001 vs -0.002). Reduce variance.
  Optuna: 80 → 150 trials for better HPO coverage.

New in v9:
  - Calibration: sigmoid (Platt) instead of isotonic
  - Feature pruning threshold: imp_mean > 0.001 (stricter, reduce overfitting)
  - Optuna trials: 150 (vs 80)
  - 3 new tick features:
      btc_up_ratio_stability — std of 6 sub-window up_ratios (consistent vs erratic signal)
      btc_vol_accel          — volume in last 90s / first 90s (acceleration proxy)
      btc_size_disparity     — avg trade size Up / avg trade size Down (conviction gap)
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

app = modal.App("polymarket-btc-train-v9", image=image)

@app.function(
    cpu=8,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
)
def train_v9():
    import gc, json, logging, os, pickle, sys, time, warnings, tempfile
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
    OBS_SECS      = 180
    OPTUNA_TRIALS = 150
    WF_GAP        = 5
    DATA_DIR      = Path("/tmp/hf_data")
    DATA_DIR.mkdir(exist_ok=True)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")

    # ── Step 1: Load champion metrics from HF ─────────────────────────────────
    log.info("Step 1: Loading champion metrics from HF...")
    champion = {"version": "none", "wf_auc": 0.0, "wf_brier": 1.0,
                "wf_acc": 0.0, "feature_list": []}
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
        log.warning("  Could not load champion meta: %s", e)

    CHAMPION_AUC   = float(champion.get("wf_auc", 0.0))
    CHAMPION_BRIER = float(champion.get("wf_brier", 1.0)) if champion.get("wf_brier") else 0.22
    CHAMPION_ACC   = float(champion.get("wf_acc", 0.0))   if champion.get("wf_acc")   else 0.73
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

    market_ids     = set(markets["market_id"])
    slot_map       = dict(zip(markets["market_id"], markets["slot_ts"]))
    target_map     = dict(zip(markets["market_id"], markets["resolution"].astype(int)))
    markets_sorted = markets.sort_values("slot_ts").reset_index(drop=True)
    slotts_to_rank = {row["slot_ts"]: i for i, row in markets_sorted.iterrows()}
    rank_to_target = dict(enumerate(markets_sorted["resolution"].astype(int)))
    rank_to_slotts = dict(enumerate(markets_sorted["slot_ts"]))

    # ── Step 4: BTC ticks ─────────────────────────────────────────────────────
    log.info("Step 4: Loading BTC ticks...")
    btc = pq.read_table(
        str(DATA_DIR / "data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet"),
        columns=["market_id", "timestamp_ms", "outcome", "side",
                 "price", "size_usdc", "spot_price_usdt"],
        filters=[("market_id", "in", list(market_ids))],
    ).to_pandas()
    btc["slot_ts_val"] = btc["market_id"].map(slot_map)
    btc["t_sec"] = btc["timestamp_ms"] / 1000 - btc["slot_ts_val"]
    btc_inslot = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < OBS_SECS)].copy()
    log.info("  BTC inslot: %d ticks, %d markets",
             len(btc_inslot), btc_inslot["market_id"].nunique())

    slot_vol = btc_inslot.groupby("market_id")["size_usdc"].sum().rename("slot_vol")

    spot_tl     = btc[["timestamp_ms", "spot_price_usdt"]].dropna().drop_duplicates("timestamp_ms")
    spot_tl     = spot_tl.set_index("timestamp_ms").sort_index()
    spot_ts_arr = spot_tl.index.values
    spot_px_arr = spot_tl["spot_price_usdt"].values
    del btc; gc.collect()

    # ── Step 5: OB — Up token only (Down token is sparse/zero) ────────────────
    log.info("Step 5: Loading BTC orderbook (Up token only)...")
    ob_up_by_market = {}
    ob_path = DATA_DIR / "data/orderbook/crypto=BTC/timeframe=5-minute/part-0.parquet"
    if ob_path.exists():
        try:
            ob = pq.read_table(str(ob_path),
                               filters=[("market_id", "in", list(market_ids))]).to_pandas()
            ts_col = "ts_ms" if "ts_ms" in ob.columns else "timestamp_ms"
            if "outcome" in ob.columns:
                ob_up = ob[ob["outcome"] == "Up"]
                for mid, grp in ob_up.groupby("market_id"):
                    ob_up_by_market[mid] = grp.sort_values(ts_col).iloc[0]
            log.info("  OB Up token: %d markets indexed", len(ob_up_by_market))
            del ob; gc.collect()
        except Exception as e:
            log.warning("  OB load failed: %s", e)

    # ── Feature helpers ────────────────────────────────────────────────────────
    def tick_features_v9(grp: pd.DataFrame) -> dict:
        """
        v9 tick features — extends v8 with:
          btc_up_ratio_stability — consistency of directional signal across 6 windows
          btc_vol_accel          — volume acceleration (last 90s vs first 90s)
          btc_size_disparity     — avg trade size Up / Down (conviction gap)
        """
        n = len(grp)
        if n == 0:
            return {
                "btc_n_ticks": 0.0, "btc_up_ratio": 0.5,
                "btc_vol_up": 0.0, "btc_vol_dn": 0.0,
                "btc_momentum": 0.0, "btc_vwap_spread": 0.0,
                "btc_vwap_up": 0.5, "btc_vwap_dn": 0.5,
                "btc_buy_ratio": 0.5, "btc_avg_size": 0.0,
                "btc_tick_accel": 0.0,
                # 6x30s sub-windows
                **{f"btc_up_w{i}": 0.5 for i in range(6)},
                # time-weighted
                "btc_tw_up_ratio": 0.5,
                # VWAP trend
                "btc_vwap_trend": 0.0,
                # vol-weighted momentum
                "btc_vwmom": 0.0,
                # v9 new
                "btc_up_ratio_stability": 0.0,
                "btc_vol_accel":          1.0,
                "btc_size_disparity":     1.0,
            }

        is_up  = grp["outcome"] == "Up"
        vol_up = (grp["size_usdc"] * is_up).sum()
        vol_dn = (grp["size_usdc"] * ~is_up).sum()
        total  = vol_up + vol_dn + 1e-8

        vwap_up = (grp.loc[is_up,  "price"] * grp.loc[is_up,  "size_usdc"]).sum() / (vol_up + 1e-8) if is_up.any()  else 0.5
        vwap_dn = (grp.loc[~is_up, "price"] * grp.loc[~is_up, "size_usdc"]).sum() / (vol_dn + 1e-8) if (~is_up).any() else 0.5

        # 6 x 30s sub-windows
        def ur_window(mask):
            if not mask.any(): return 0.5
            sub = grp[mask]
            vu = (sub["size_usdc"] * (sub["outcome"] == "Up")).sum()
            return float(vu / (sub["size_usdc"].sum() + 1e-8))

        sw = {}
        for i in range(6):
            t0_w, t1_w = i * 30, (i + 1) * 30
            mask = (grp["t_sec"] >= t0_w) & (grp["t_sec"] < t1_w)
            sw[f"btc_up_w{i}"] = ur_window(mask)

        # Momentum from 6-window series
        w_vals = [sw[f"btc_up_w{i}"] for i in range(6)]
        btc_momentum = float(np.mean(w_vals[3:]) - np.mean(w_vals[:3]))

        # Time-weighted order flow: ticks near end of slot get higher weight
        # weight = exp(t_sec / OBS_SECS * 2) normalized
        weights = np.exp(grp["t_sec"].values / OBS_SECS * 2.0)
        weights /= weights.sum() + 1e-8
        tw_up = float((weights * (grp["outcome"] == "Up").values).sum())

        # VWAP trend: split slot in half, compare vwap of each half
        half = OBS_SECS / 2
        early = grp[grp["t_sec"] < half]
        late  = grp[grp["t_sec"] >= half]
        def vwap_up_half(g):
            up = g[g["outcome"] == "Up"]
            if len(up) == 0: return 0.5
            return float((up["price"] * up["size_usdc"]).sum() / (up["size_usdc"].sum() + 1e-8))
        vwap_trend = float(vwap_up_half(late) - vwap_up_half(early))

        # Volume-weighted momentum: weight each window by its volume
        vol_by_w = []
        ur_by_w  = []
        for i in range(6):
            t0_w, t1_w = i * 30, (i + 1) * 30
            sub = grp[(grp["t_sec"] >= t0_w) & (grp["t_sec"] < t1_w)]
            vol_by_w.append(sub["size_usdc"].sum())
            ur_by_w.append(ur_window((grp["t_sec"] >= t0_w) & (grp["t_sec"] < t1_w)))
        vol_by_w = np.array(vol_by_w)
        ur_by_w  = np.array(ur_by_w)
        total_w_vol = vol_by_w.sum() + 1e-8
        # weighted corr: do high-volume windows lean UP more?
        vwmom = float(np.dot(vol_by_w / total_w_vol, ur_by_w - 0.5))

        # Tick acceleration
        first30 = (grp["t_sec"] < 30).sum()
        last30  = (grp["t_sec"] >= (OBS_SECS - 30)).sum()

        # ── v9 new features ───────────────────────────────────────────────────
        # Stability of directional signal across the 6 windows
        w_vals = [sw[f"btc_up_w{i}"] for i in range(6)]
        up_ratio_stability = float(np.std(w_vals))  # low = consistent, high = erratic

        # Volume acceleration: last 90s vs first 90s
        vol_first90 = grp[grp["t_sec"] < 90]["size_usdc"].sum()
        vol_last90  = grp[grp["t_sec"] >= 90]["size_usdc"].sum()
        vol_accel   = float(vol_last90 / (vol_first90 + 1e-8))

        # Size disparity: avg Up trade size / avg Down trade size
        up_grp = grp[is_up]
        dn_grp = grp[~is_up]
        avg_size_up = float(up_grp["size_usdc"].mean()) if len(up_grp) > 0 else 1.0
        avg_size_dn = float(dn_grp["size_usdc"].mean()) if len(dn_grp) > 0 else 1.0
        size_disparity = float(avg_size_up / (avg_size_dn + 1e-8))

        return {
            "btc_n_ticks":    float(n),
            "btc_vol_up":     float(vol_up),
            "btc_vol_dn":     float(vol_dn),
            "btc_up_ratio":   float(vol_up / total),
            "btc_vwap_up":    float(vwap_up),
            "btc_vwap_dn":    float(vwap_dn),
            "btc_vwap_spread": float(vwap_up - vwap_dn),
            "btc_buy_ratio":  float((grp["side"] == "BUY").sum() / (n + 1e-8)),
            "btc_avg_size":   float(total / n),
            "btc_momentum":   btc_momentum,
            "btc_tick_accel": float((last30 - first30) / (first30 + 1e-8)),
            "btc_tw_up_ratio": tw_up,
            "btc_vwap_trend":  vwap_trend,
            "btc_vwmom":       vwmom,
            **sw,
            # v9 new
            "btc_up_ratio_stability": up_ratio_stability,
            "btc_vol_accel":          vol_accel,
            "btc_size_disparity":     size_disparity,
        }

    def spot_open_at(slot_ts: int) -> float:
        idx = np.searchsorted(spot_ts_arr, slot_ts * 1000, side="right") - 1
        return float(spot_px_arr[idx]) if idx >= 0 else 0.0

    # ── Step 6: Build dataset ──────────────────────────────────────────────────
    log.info("Step 6: Building feature dataset...")
    btc_grps     = dict(list(btc_inslot.groupby("market_id")))
    vol_series   = markets_sorted["market_id"].map(slot_vol).fillna(0).values

    # Pre-compute per-slot up_ratio for historical features
    up_ratio_series = np.array([
        float(
            btc_grps[r["market_id"]]["size_usdc"][
                btc_grps[r["market_id"]]["outcome"] == "Up"].sum() /
            (btc_grps[r["market_id"]]["size_usdc"].sum() + 1e-8)
        ) if r["market_id"] in btc_grps else 0.5
        for _, r in markets_sorted.iterrows()
    ])

    # Pre-compute per-slot sub-window up_ratios for window-level zscores
    # Shape: (n_markets, 6) — one value per 30s window per slot
    sw_series = np.full((len(markets_sorted), 6), 0.5)
    for i, row in markets_sorted.iterrows():
        grp = btc_grps.get(row["market_id"])
        if grp is None: continue
        for w in range(6):
            t0_w, t1_w = w * 30, (w + 1) * 30
            sub = grp[(grp["t_sec"] >= t0_w) & (grp["t_sec"] < t1_w)]
            if len(sub) > 0:
                vu = (sub["size_usdc"] * (sub["outcome"] == "Up")).sum()
                sw_series[i, w] = float(vu / (sub["size_usdc"].sum() + 1e-8))

    records = []
    for i, row in markets_sorted.iterrows():
        mid    = row["market_id"]
        target = target_map[mid]
        grp    = btc_grps.get(mid)
        if grp is None or len(grp) < 5:
            continue

        slot_ts = slot_map[mid]
        rank    = slotts_to_rank[slot_ts]
        dt      = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour    = dt.hour + dt.minute / 60.0

        feat = {
            "market_id": mid, "slot_ts": slot_ts, "target": target,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin":  np.sin(2 * np.pi * dt.weekday() / 7),
            "dow_cos":  np.cos(2 * np.pi * dt.weekday() / 7),
        }

        # Rich tick features (v8)
        feat.update(tick_features_v9(grp))

        # Inslot spot return (anchored at slot_ts)
        spot_open = spot_open_at(slot_ts)
        sp = grp["spot_price_usdt"].dropna()
        if spot_open > 0 and len(sp) >= 1:
            feat["btc_inslot_ret"] = (float(sp.iloc[-1]) - spot_open) / (spot_open + 1e-8)
            feat["btc_inslot_vol"] = float(sp.std() / (sp.mean() + 1e-8)) if len(sp) >= 2 else 0.0
        else:
            feat["btc_inslot_ret"] = 0.0
            feat["btc_inslot_vol"] = 0.0

        # Pre-slot spot returns: 5m, 15m, 30m, 1h, 4h
        for w_s, lbl in [(300,"5m"), (900,"15m"), (1800,"30m"), (3600,"1h"), (14400,"4h")]:
            idx0, idx1 = np.searchsorted(spot_ts_arr,
                                          [(slot_ts - w_s)*1000, slot_ts*1000])
            seg = spot_px_arr[idx0:idx1]
            if len(seg) >= 2:
                feat[f"btc_pre_{lbl}_ret"] = float((seg[-1]-seg[0]) / (seg[0]+1e-8))
                feat[f"btc_pre_{lbl}_vol"] = float(np.std(seg) / (np.mean(seg)+1e-8))
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
            feat["btc_vol_zscore"] = float((cur_vol - hist_vols.mean()) / (hist_vols.std() + 1e-8))
            feat["btc_vol_ratio"]  = float(cur_vol / (hist_vols.mean() + 1e-8))
        else:
            feat["btc_vol_zscore"] = 0.0
            feat["btc_vol_ratio"]  = 1.0

        # up_ratio anomaly — 3 windows: 5, 10, 20 slots
        cur_ur = up_ratio_series[rank]
        for win_ur, lbl_ur in [(5, "5s"), (10, "10s"), (20, "20s")]:
            ws = max(0, rank - win_ur)
            hist_ur = up_ratio_series[ws:rank]
            if len(hist_ur) >= 3:
                feat[f"btc_up_ratio_zscore_{lbl_ur}"]    = float((cur_ur - hist_ur.mean()) / (hist_ur.std() + 1e-8))
                feat[f"btc_up_ratio_hist_mean_{lbl_ur}"] = float(hist_ur.mean())
            else:
                feat[f"btc_up_ratio_zscore_{lbl_ur}"]    = 0.0
                feat[f"btc_up_ratio_hist_mean_{lbl_ur}"] = 0.5

        # Per-sub-window zscore vs last 20 slots (w1/w2 are strongest in v7)
        for w in range(6):
            cur_sw_ur = sw_series[rank, w]
            ws = max(0, rank - 20)
            hist_sw = sw_series[ws:rank, w]
            if len(hist_sw) >= 5:
                feat[f"btc_up_w{w}_zscore"] = float(
                    (cur_sw_ur - hist_sw.mean()) / (hist_sw.std() + 1e-8))
            else:
                feat[f"btc_up_w{w}_zscore"] = 0.0

        # Realized vol of last 5/10 slots (vol clustering)
        for win_rv, lbl_rv in [(5, "5s"), (10, "10s")]:
            ws = max(0, rank - win_rv)
            past_rets = []
            for back_rank in range(ws, rank):
                bslot = rank_to_slotts.get(back_rank)
                if bslot is None: continue
                idx0, idx1 = np.searchsorted(spot_ts_arr,
                                              [(bslot - 300)*1000, bslot*1000])
                seg = spot_px_arr[idx0:idx1]
                if len(seg) >= 2:
                    past_rets.append((seg[-1]-seg[0]) / (seg[0]+1e-8))
            feat[f"btc_realized_vol_{lbl_rv}"] = float(np.std(past_rets)) if len(past_rets) >= 3 else 0.0

        # Lag outcomes (only lag_2 was useful in v7, keep 1-3 for perm importance to decide)
        for lag in [1, 2, 3]:
            feat[f"lag_{lag}_outcome"] = float(rank_to_target.get(rank - lag, 0.5))

        # Lag streak
        streak = 0
        if rank >= 1:
            last_val = rank_to_target.get(rank - 1, -1)
            for back in range(1, min(rank + 1, 6)):
                v = rank_to_target.get(rank - back, -1)
                if v == last_val and v != -1: streak += 1
                else: break
        feat["lag_streak"] = float(streak)

        # OB: Up token only (Down is sparse/zero → dropped in v7)
        ob_up_row = ob_up_by_market.get(mid)
        if ob_up_row is not None:
            try:
                feat["ob_up_bid"]    = float(ob_up_row.get("best_bid") or 0.5)
                feat["ob_up_ask"]    = float(ob_up_row.get("best_ask") or 0.5)
                feat["ob_up_spread"] = float((ob_up_row.get("best_ask") or 0.5) -
                                              (ob_up_row.get("best_bid") or 0.5))
                # Implied probability ≈ Up token mid price
                feat["ob_implied_prob"] = float(
                    ((ob_up_row.get("best_bid") or 0) +
                     (ob_up_row.get("best_ask") or 1)) / 2)
                bid_d = float(ob_up_row.get("best_bid_size") or 0)
                ask_d = float(ob_up_row.get("best_ask_size") or 0)
                feat["ob_up_bid_depth"]  = bid_d
                feat["ob_up_ask_depth"]  = ask_d
                feat["ob_up_imbalance"]  = float((bid_d - ask_d) / (bid_d + ask_d + 1e-8))
            except Exception:
                feat.update({"ob_up_bid": 0.5, "ob_up_ask": 0.5, "ob_up_spread": 0.0,
                              "ob_implied_prob": 0.5, "ob_up_bid_depth": 0.0,
                              "ob_up_ask_depth": 0.0, "ob_up_imbalance": 0.0})
        else:
            feat.update({"ob_up_bid": 0.5, "ob_up_ask": 0.5, "ob_up_spread": 0.0,
                          "ob_implied_prob": 0.5, "ob_up_bid_depth": 0.0,
                          "ob_up_ask_depth": 0.0, "ob_up_imbalance": 0.0})

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
    def walk_forward_purged(df, feats, params=None, gap=WF_GAP):
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
            if test_end - test_start < 20: continue
            tr = df.iloc[:train_end]
            te = df.iloc[test_start:test_end]
            mdl = lgb.LGBMClassifier(**p)
            mdl.fit(tr[feats].fillna(0), tr["target"])
            prob = mdl.predict_proba(te[feats].fillna(0))[:, 1]
            aucs.append(roc_auc_score(te["target"], prob))
            accs.append(float(((prob >= 0.5) == te["target"]).mean()))
            briers.append(brier_score_loss(te["target"], prob))
        if not aucs:
            return {"wf_auc": 0.5, "wf_acc": 0.5, "wf_brier": 0.5, "fold_aucs": []}
        return {"wf_auc": float(np.mean(aucs)), "wf_acc": float(np.mean(accs)),
                "wf_brier": float(np.mean(briers)), "fold_aucs": aucs}

    # ── Step 7: Baseline ───────────────────────────────────────────────────────
    log.info("Step 7: Baseline purged walk-forward...")
    wf_base = walk_forward_purged(df, features)
    log.info("  Baseline WF AUC: %.4f | Acc: %.4f | Brier: %.4f",
             wf_base["wf_auc"], wf_base["wf_acc"], wf_base["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_base["fold_aucs"]])

    # ── Step 8: Permutation importance ────────────────────────────────────────
    log.info("Step 8: Permutation importance...")
    split = int(len(df) * 0.75)
    tr_imp, va_imp = df.iloc[:split], df.iloc[split:]
    imp_mdl = lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", n_estimators=300,
        learning_rate=0.05, num_leaves=31, min_child_samples=15,
        verbose=-1, n_jobs=-1)
    imp_mdl.fit(tr_imp[features].fillna(0), tr_imp["target"])
    perm = permutation_importance(
        imp_mdl, va_imp[features].fillna(0), va_imp["target"],
        n_repeats=15, random_state=42, scoring="roc_auc")
    imp_df = pd.DataFrame({
        "feature":  features,
        "imp_mean": perm.importances_mean,
        "imp_std":  perm.importances_std,
    }).sort_values("imp_mean", ascending=False)
    log.info("  Top 20 features:")
    for _, r in imp_df.head(20).iterrows():
        log.info("    %-45s %.4f ± %.4f", r["feature"], r["imp_mean"], r["imp_std"])
    good_features = imp_df[imp_df["imp_mean"] > 0.001]["feature"].tolist()
    dropped = len(features) - len(good_features)
    log.info("  Dropped %d noise features, keeping %d", dropped, len(good_features))

    # ── Step 9: Optuna HPO (purged WF objective) ──────────────────────────────
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
        return walk_forward_purged(df, good_features, p)["wf_auc"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    if "lr" in best_params:
        best_params["learning_rate"] = best_params.pop("lr")
    log.info("  Optuna best WF AUC: %.4f", study.best_value)

    # ── Step 10: Optimized WF ─────────────────────────────────────────────────
    log.info("Step 10: Walk-forward (optimized)...")
    wf_opt = walk_forward_purged(df, good_features, best_params)
    log.info("  Optimized WF AUC: %.4f | Acc: %.4f | Brier: %.4f",
             wf_opt["wf_auc"], wf_opt["wf_acc"], wf_opt["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_opt["fold_aucs"]])

    # Pick best
    if wf_opt["wf_auc"] >= wf_base["wf_auc"]:
        final_wf, final_params, final_feats = wf_opt, best_params, good_features
        log.info("  Using: optimized params")
    else:
        final_wf, final_params, final_feats = wf_base, {}, features
        log.info("  Using: baseline params (Optuna didn't improve)")

    final_auc, final_acc, final_brier = (
        final_wf["wf_auc"], final_wf["wf_acc"], final_wf["wf_brier"])

    # ── Step 11: Re-evaluate champion with same purged WF ─────────────────────
    log.info("Step 11: Champion comparison...")
    # Use the AUC from champion_meta.json directly — it was already computed
    # with the same purged WF protocol. Re-evaluating with default params
    # artificially deflates it and lets weaker models pass the gate.
    fair_champ_auc = CHAMPION_AUC
    log.info("  Champion (%s): AUC=%.4f  Brier=%.4f  Acc=%.4f",
             champion.get("version", "?"), CHAMPION_AUC, CHAMPION_BRIER, CHAMPION_ACC)

    # ── Step 12: Train final with calibration ─────────────────────────────────
    log.info("Step 12: Training final model...")
    base_p = dict(objective="binary", class_weight="balanced", n_estimators=400,
                  learning_rate=0.04, num_leaves=31, min_child_samples=15,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=-1)
    params = {**base_p, **final_params,
              "objective": "binary", "class_weight": "balanced",
              "verbose": -1, "n_jobs": -1}
    X, y = df[final_feats].fillna(0), df["target"]
    final_model = CalibratedClassifierCV(
        lgb.LGBMClassifier(**params), method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3))
    final_model.fit(X, y)

    bundle = {
        "model":            final_model,
        "features":         final_feats,
        "wf_auc":           final_auc,
        "wf_acc":           final_acc,
        "wf_brier":         final_brier,
        "fold_aucs":        final_wf["fold_aucs"],
        "version":          "v9",
        "n_samples":        len(df),
        "n_features":       len(final_feats),
        "best_params":      final_params,
        "ensemble":         False,
        "dropped_features": dropped,
        "champion_compared_auc": fair_champ_auc,
    }
    model_path = Path("/tmp/btc_model_v9.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Model saved: %.1f MB", model_path.stat().st_size / 1e6)

    # ── Gate ──────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("  Champion (%s) purged WF: AUC=%.4f  Brier=%.4f  Acc=%.4f",
             champion.get("version", "?"), fair_champ_auc,
             CHAMPION_BRIER, CHAMPION_ACC)
    log.info("  v8 candidate:            AUC=%.4f  Brier=%.4f  Acc=%.4f",
             final_auc, final_brier, final_acc)
    log.info("  Features: %d (dropped %d noise)", len(final_feats), dropped)

    beats_auc   = final_auc   > fair_champ_auc
    beats_brier = final_brier < CHAMPION_BRIER
    beats_acc   = final_acc   > CHAMPION_ACC
    n_passed    = sum([beats_auc, beats_brier, beats_acc])
    # Strict gate: must beat champion on at least 2 of 3 metrics
    should_promote = n_passed >= 2

    log.info("  Gate: AUC>%.4f[%s] Brier<%.4f[%s] Acc>%.4f[%s] → %d/3 | %s",
             fair_champ_auc, "✓" if beats_auc   else "✗",
             CHAMPION_BRIER,  "✓" if beats_brier else "✗",
             CHAMPION_ACC,    "✓" if beats_acc   else "✗",
             n_passed, "PROMOTE" if should_promote else "REJECT")

    promoted = False
    if should_promote:
        log.info("Promoting v8 to HF champion...")
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="champion.pkl",
            repo_id=HF_MODEL_REPO, repo_type="model",
            commit_message=(f"Champion v9: AUC={final_auc:.4f} Brier={final_brier:.4f} "
                            f"Acc={final_acc:.4f} | sigmoid calib + stricter pruning"),
        )
        meta_out = {
            "version":       "v9",
            "feature_list":  final_feats,
            "features":      len(final_feats),
            "wf_auc":        final_auc,
            "wf_acc":        final_acc,
            "wf_brier":      final_brier,
            "fold_aucs":     final_wf["fold_aucs"],
            "n_samples":     len(df),
            "ensemble":      False,
            "promoted_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "champion_compared_auc": fair_champ_auc,
            "notes": ("v9: sigmoid calibration (vs isotonic), stricter feature pruning "
                      "(imp>0.001), 150 Optuna trials, + btc_up_ratio_stability, "
                      "btc_vol_accel, btc_size_disparity"),
            "best_params":   final_params,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
            json.dump(meta_out, fp, indent=2)
        api.upload_file(
            path_or_fileobj=fp.name,
            path_in_repo="champion_meta.json",
            repo_id=HF_MODEL_REPO, repo_type="model",
            commit_message=f"Champion v8 meta AUC={final_auc:.4f}",
        )
        log.info("Champion v8 promoted: https://huggingface.co/%s", HF_MODEL_REPO)
        promoted = True
    else:
        log.warning("v8 not promoted.")

    log.info("Done.")
    return {
        "wf_auc_baseline":    wf_base["wf_auc"],
        "wf_auc_optimized":   wf_opt["wf_auc"],
        "wf_auc_final":       final_auc,
        "wf_acc_final":       final_acc,
        "wf_brier_final":     final_brier,
        "champion_fair_auc":  fair_champ_auc,
        "n_samples":          len(df),
        "n_features_final":   len(final_feats),
        "n_features_dropped": dropped,
        "promoted":           promoted,
    }


@app.local_entrypoint()
def main():
    print("Submitting BTC v9 training job to Modal...")
    r = train_v9.remote()
    print(f"\n{'='*55}")
    print("TRAINING COMPLETE — v9")
    print(f"  Baseline AUC:     {r['wf_auc_baseline']:.4f}")
    print(f"  Optimized AUC:    {r['wf_auc_optimized']:.4f}")
    print(f"  Final AUC:        {r['wf_auc_final']:.4f}")
    print(f"  Final Acc:        {r['wf_acc_final']:.4f}")
    print(f"  Final Brier:      {r['wf_brier_final']:.4f}")
    print(f"  Champion AUC:     {r['champion_fair_auc']:.4f} (purged WF)")
    print(f"  Samples:          {r['n_samples']}")
    print(f"  Features:         {r['n_features_final']} (dropped {r['n_features_dropped']})")
    print(f"  Promoted:         {'YES ✓' if r['promoted'] else 'NO'}")
    print(f"{'='*55}")
