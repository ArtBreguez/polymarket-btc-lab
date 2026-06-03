"""
train_v16_modal.py — BTC 5min model v16

Root cause analysis of v14/v15 plateau (AUC ~0.847):
  - AUC is NOT limited by feature count — adding 20→50 features gave zero AUC gain
  - Brier DEGRADED with more features (0.1636 → 0.1685): isotonic calibration
    overfits with too many inputs at 6707 samples
  - The ceiling for this feature set is ~0.847 AUC; to break it, we need NEW signals

Changes vs v15:
  - REMOVED 3 known-bad features computed at train time:
      btc_up_ratio:  fully redundant (= vol-weighted avg of sub-windows), imp~0 both runs
      btc_momentum:  linear combo of sub-windows (mean[w3-w5] - mean[w0-w2]), imp negative
      btc_spot_open: raw price level ($65k-$76k) — non-generalizable, causes overfit
  - FEATURE SELECTION: top-30 fixed instead of threshold
      Threshold (0.0005/0.0001) is unstable across runs; top-30 is deterministic,
      empirically best for Brier at ~6700 samples (same ratio as v10's ~30/601)
  - NEW SIGNAL: prev_slot_up_ratio
      lag_1_outcome = binary (0 or 1), discards conviction level
      prev_slot_up_ratio = continuous (e.g. 0.30, 0.62, 0.91)
      A slot with up_ratio=0.92 (strong consensus) is far more predictive than
      one with up_ratio=0.53 (barely above chance). This is a strictly richer signal.
  - NEW SIGNAL: prev_slot_n_ticks, prev_slot_vol_total
      Activity level of the previous slot captures market engagement and regime
  - KEPT: btc_pre_1h_4h_ratio (ranked #7 in v15, imp=0.0016 — confirmed signal)
  - GATE: vs v14 (AUC=0.8470, Brier=0.1636, Acc=0.7660) — same 2026 regime

Data source (unchanged):
  - Modal Volume 'btc-local-data': ticks + markets + binance spot
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

app = modal.App("btc-v16-run", image=image)

@app.function(
    cpu=8,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/btc_local": LOCAL_VOL},
)
def train_v16():
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
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
    OBS_SECS      = 180
    OPTUNA_TRIALS = 150
    WF_GAP        = 5
    N_SPLITS      = 5
    TOP_N_FEATS   = 30   # fixed top-N selection (not threshold)
    LOCAL_DIR     = Path("/btc_local")

    # Gate baseline: v14 on 2026 regime (fair comparison)
    V14_AUC   = 0.8470
    V14_BRIER = 0.1636
    V14_ACC   = 0.7660

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")

    # ── Step 1: HF champion (reference only) ──────────────────────────────
    log.info("Step 1: Loading HF champion metrics (reference)...")
    champion = {"version": "v10", "wf_auc": 0.8547, "wf_brier": 0.1554, "wf_acc": 0.7902}
    try:
        meta_path = hf_hub_download(HF_MODEL_REPO, "champion_meta.json",
                                    repo_type="model", token=HF_TOKEN,
                                    local_dir=tempfile.mkdtemp())
        with open(meta_path) as fp:
            champion = json.load(fp)
        log.info("  HF Champion: %s | AUC=%.4f", champion.get("version"), champion.get("wf_auc"))
    except Exception as e:
        log.warning("  Could not load champion meta: %s", e)
    log.info("  Gate (v14, 2026 regime): AUC=%.4f | Brier=%.4f | Acc=%.4f",
             V14_AUC, V14_BRIER, V14_ACC)

    # ── Step 2: Markets ────────────────────────────────────────────────────
    log.info("Step 2: Loading local markets...")
    markets = pd.read_csv(LOCAL_DIR / "local_markets.csv")
    markets["market_id"] = markets["market_id"].astype(str)
    if "slot_ts" not in markets.columns:
        markets["slot_ts"] = markets["slug"].apply(
            lambda s: int(str(s).split("-")[-1]) if str(s).split("-")[-1].isdigit() else 0
        )
    markets = markets[markets["slot_ts"] > 0].copy()
    markets["target"] = markets["target"].astype(int)
    markets = markets.sort_values("slot_ts").reset_index(drop=True)
    log.info("  Markets: %d | UP=%d DOWN=%d | range: %s -> %s",
             len(markets),
             (markets["target"] == 1).sum(),
             (markets["target"] == 0).sum(),
             datetime.utcfromtimestamp(markets["slot_ts"].min()).strftime("%Y-%m-%d"),
             datetime.utcfromtimestamp(markets["slot_ts"].max()).strftime("%Y-%m-%d"))

    market_ids     = set(markets["market_id"])
    slot_map       = dict(zip(markets["market_id"], markets["slot_ts"]))
    target_map     = dict(zip(markets["market_id"], markets["target"]))
    slotts_to_rank = {row["slot_ts"]: i for i, row in markets.iterrows()}
    rank_to_target = dict(enumerate(markets["target"]))
    rank_to_slotts = dict(enumerate(markets["slot_ts"]))

    # ── Step 3: Binance spot ───────────────────────────────────────────────
    log.info("Step 3: Loading Binance spot...")
    binance_df = pd.read_parquet(str(LOCAL_DIR / "binance_spot_local.parquet"))
    binance_df = binance_df.drop_duplicates("timestamp_ms").sort_values("timestamp_ms").reset_index(drop=True)
    spot_ts_arr = binance_df["timestamp_ms"].values.astype(np.float64)
    spot_px_arr = binance_df["close"].values.astype(np.float64)
    log.info("  Binance spot: %d candles | $%.0f - $%.0f",
             len(spot_ts_arr), spot_px_arr.min(), spot_px_arr.max())
    del binance_df; gc.collect()

    # ── Step 4: Ticks ─────────────────────────────────────────────────────
    log.info("Step 4: Loading ticks...")
    needed_cols = ["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"]
    pf = pq.ParquetFile(str(LOCAL_DIR / "ticks_btc_5min.parquet"))
    n_rg = pf.metadata.num_row_groups
    log.info("  Row groups: %d", n_rg)
    chunks, rows_loaded = [], 0
    for rg_i in range(n_rg):
        chunk = pf.read_row_group(rg_i, columns=needed_cols).to_pandas()
        chunk["market_id"] = chunk["market_id"].astype(str)
        chunk = chunk[chunk["market_id"].isin(market_ids)]
        if len(chunk) > 0:
            chunks.append(chunk)
            rows_loaded += len(chunk)
        if rg_i % 100 == 0:
            log.info("  rg %d/%d — kept %d rows", rg_i, n_rg, rows_loaded)
    del pf; gc.collect()

    btc = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    btc = btc[btc["timestamp_ms"] > 0]
    btc["slot_ts_val"] = btc["market_id"].map(slot_map)
    btc = btc.dropna(subset=["slot_ts_val"])
    btc["t_sec"] = btc["timestamp_ms"].astype(float) / 1000 - btc["slot_ts_val"].astype(float)
    btc_inslot = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < OBS_SECS)].copy()
    log.info("  Ticks: %d rows | Inslot: %d rows, %d markets",
             len(btc), len(btc_inslot), btc_inslot["market_id"].nunique())

    slot_vol = btc_inslot.groupby("market_id")["size_usdc"].sum().rename("slot_vol")
    slot_nticks = btc_inslot.groupby("market_id").size().rename("slot_nticks")
    del btc; gc.collect()

    # ── Spot helpers ───────────────────────────────────────────────────────
    def spot_at(ts_ms: float) -> float:
        idx = np.searchsorted(spot_ts_arr, ts_ms, side="right") - 1
        return float(spot_px_arr[idx]) if idx >= 0 else 0.0

    def inslot_spot_features(slot_ts: int) -> dict:
        s_ms = float(slot_ts) * 1000
        e_ms = s_ms + OBS_SECS * 1000
        idx0, idx1 = np.searchsorted(spot_ts_arr, [s_ms, e_ms])
        seg = spot_px_arr[idx0:idx1]
        if len(seg) >= 2:
            ret = float((seg[-1] - seg[0]) / (seg[0] + 1e-8))
            vol = float(np.std(seg) / (np.mean(seg) + 1e-8))
        else:
            ret, vol = 0.0, 0.0
        direction = 1.0 if ret > 1e-5 else (-1.0 if ret < -1e-5 else 0.0)
        return {"btc_inslot_ret": ret, "btc_inslot_vol": vol, "btc_inslot_direction": direction}

    # ── Tick features (no btc_up_ratio, no btc_momentum — redundant) ──────
    def tick_features(grp: pd.DataFrame) -> dict:
        n = len(grp)
        if n == 0:
            return {
                "btc_n_ticks": 0.0,
                "btc_vol_up": 0.0, "btc_vol_dn": 0.0,
                "btc_vwap_spread": 0.0, "btc_vwap_up": 0.5, "btc_vwap_dn": 0.5,
                "btc_buy_ratio": 0.5, "btc_avg_size": 0.0, "btc_tick_accel": 0.0,
                **{f"btc_up_w{i}": 0.5 for i in range(6)},
                "btc_tw_up_ratio": 0.5, "btc_vwap_trend": 0.0, "btc_vwmom": 0.0,
                "btc_up_ratio_stability": 0.0, "btc_vol_accel": 1.0, "btc_size_disparity": 1.0,
            }

        is_up  = grp["outcome"] == "Up"
        vol_up = (grp["size_usdc"] * is_up).sum()
        vol_dn = (grp["size_usdc"] * ~is_up).sum()
        total  = vol_up + vol_dn + 1e-8

        vwap_up = (grp.loc[is_up,  "price"] * grp.loc[is_up,  "size_usdc"]).sum() / (vol_up + 1e-8) if is_up.any()  else 0.5
        vwap_dn = (grp.loc[~is_up, "price"] * grp.loc[~is_up, "size_usdc"]).sum() / (vol_dn + 1e-8) if (~is_up).any() else 0.5

        def ur_window(mask):
            if not mask.any(): return 0.5
            sub = grp[mask]
            vu = (sub["size_usdc"] * (sub["outcome"] == "Up")).sum()
            return float(vu / (sub["size_usdc"].sum() + 1e-8))

        sw = {f"btc_up_w{i}": ur_window((grp["t_sec"] >= i*30) & (grp["t_sec"] < (i+1)*30))
              for i in range(6)}

        weights  = np.exp(grp["t_sec"].values / OBS_SECS * 2.0)
        weights /= weights.sum() + 1e-8
        tw_up    = float((weights * (grp["outcome"] == "Up").values).sum())

        half = OBS_SECS / 2
        def vwap_up_half(g):
            up = g[g["outcome"] == "Up"]
            if len(up) == 0: return 0.5
            return float((up["price"] * up["size_usdc"]).sum() / (up["size_usdc"].sum() + 1e-8))
        vwap_trend = float(vwap_up_half(grp[grp["t_sec"] >= half]) -
                           vwap_up_half(grp[grp["t_sec"] < half]))

        vol_by_w = np.array([grp[(grp["t_sec"] >= i*30) & (grp["t_sec"] < (i+1)*30)]["size_usdc"].sum()
                             for i in range(6)])
        ur_by_w  = np.array([sw[f"btc_up_w{i}"] for i in range(6)])
        vwmom    = float(np.dot(vol_by_w / (vol_by_w.sum() + 1e-8), ur_by_w - 0.5))

        first30 = (grp["t_sec"] < 30).sum()
        last30  = (grp["t_sec"] >= (OBS_SECS - 30)).sum()
        vol_first90 = grp[grp["t_sec"] < 90]["size_usdc"].sum()
        vol_last90  = grp[grp["t_sec"] >= 90]["size_usdc"].sum()
        up_grp = grp[is_up]
        dn_grp = grp[~is_up]

        return {
            "btc_n_ticks":    float(n),
            "btc_vol_up":     float(vol_up),
            "btc_vol_dn":     float(vol_dn),
            # btc_up_ratio REMOVED — redundant with sub-windows (imp~0 in v14/v15)
            "btc_vwap_up":    float(vwap_up),
            "btc_vwap_dn":    float(vwap_dn),
            "btc_vwap_spread": float(vwap_up - vwap_dn),
            "btc_buy_ratio":  float((grp["side"] == "BUY").sum() / (n + 1e-8)),
            "btc_avg_size":   float(total / n),
            # btc_momentum REMOVED — linear combo of sub-windows (imp negative in v15)
            "btc_tick_accel": float((last30 - first30) / (first30 + 1e-8)),
            "btc_tw_up_ratio": tw_up,
            "btc_vwap_trend":  vwap_trend,
            "btc_vwmom":       vwmom,
            **sw,
            "btc_up_ratio_stability": float(np.std(list(sw.values()))),
            "btc_vol_accel":          float(vol_last90 / (vol_first90 + 1e-8)),
            "btc_size_disparity":     float(
                (float(up_grp["size_usdc"].mean()) if len(up_grp) > 0 else 1.0) /
                (float(dn_grp["size_usdc"].mean()) if len(dn_grp) > 0 else 1.0 + 1e-8)
            ),
        }

    # ── Step 5: Build dataset ──────────────────────────────────────────────
    log.info("Step 5: Building feature dataset...")
    vol_series      = markets["market_id"].map(slot_vol).fillna(0).values
    nticks_series   = markets["market_id"].map(slot_nticks).fillna(0).values
    btc_grps        = dict(list(btc_inslot.groupby("market_id")))

    # Pre-compute per-slot up_ratio for the NEW prev_slot_up_ratio feature
    up_ratio_per_rank = np.array([
        float(
            btc_grps[markets.iloc[r]["market_id"]]["size_usdc"][
                btc_grps[markets.iloc[r]["market_id"]]["outcome"] == "Up"].sum() /
            (btc_grps[markets.iloc[r]["market_id"]]["size_usdc"].sum() + 1e-8)
        ) if markets.iloc[r]["market_id"] in btc_grps else 0.5
        for r in range(len(markets))
    ])

    sw_series = np.full((len(markets), 6), 0.5)
    for i, row in markets.iterrows():
        grp = btc_grps.get(row["market_id"])
        if grp is None: continue
        for w in range(6):
            sub = grp[(grp["t_sec"] >= w*30) & (grp["t_sec"] < (w+1)*30)]
            if len(sub) > 0:
                vu = (sub["size_usdc"] * (sub["outcome"] == "Up")).sum()
                sw_series[i, w] = float(vu / (sub["size_usdc"].sum() + 1e-8))

    records = []
    skipped = 0
    for i, row in markets.iterrows():
        mid    = row["market_id"]
        target = target_map[mid]
        grp    = btc_grps.get(mid)
        if grp is None or len(grp) < 5:
            skipped += 1
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

        feat.update(tick_features(grp))
        feat.update(inslot_spot_features(slot_ts))

        # btc_spot_open REMOVED — raw price level not generalizable

        # Pre-slot returns
        pre_rets = {}
        for w_s, lbl in [(300, "5m"), (900, "15m"), (1800, "30m"), (3600, "1h"), (14400, "4h")]:
            idx0, idx1 = np.searchsorted(spot_ts_arr, [(slot_ts - w_s) * 1000.0, slot_ts * 1000.0])
            seg = spot_px_arr[idx0:idx1]
            if len(seg) >= 2:
                feat[f"btc_pre_{lbl}_ret"] = float((seg[-1] - seg[0]) / (seg[0] + 1e-8))
                feat[f"btc_pre_{lbl}_vol"] = float(np.std(seg) / (np.mean(seg) + 1e-8))
                pre_rets[lbl] = feat[f"btc_pre_{lbl}_ret"]
            else:
                feat[f"btc_pre_{lbl}_ret"] = 0.0
                feat[f"btc_pre_{lbl}_vol"] = 0.0
                pre_rets[lbl] = 0.0

        # Short vs long momentum ratio (kept from v15 — ranked #7)
        r1h = pre_rets.get("1h", 0.0)
        r4h = pre_rets.get("4h", 0.0)
        feat["btc_pre_1h_4h_ratio"] = float(r1h / (abs(r4h) + 1e-6)) if r4h != 0.0 else 0.0

        # Psychological level distances
        spot_open = spot_at(float(slot_ts) * 1000)
        if spot_open > 0:
            feat["btc_dist_1k"]  = float(abs(spot_open % 1000) / 1000)
            feat["btc_dist_5k"]  = float(abs(spot_open % 5000) / 5000)
            feat["btc_dist_10k"] = float(abs(spot_open % 10000) / 10000)
        else:
            feat["btc_dist_1k"] = feat["btc_dist_5k"] = feat["btc_dist_10k"] = 0.5

        # Volume z-score
        win_start = max(0, rank - 20)
        hist_vols = vol_series[win_start:rank]
        cur_vol   = vol_series[rank]
        if len(hist_vols) >= 5:
            feat["btc_vol_zscore"] = float((cur_vol - hist_vols.mean()) / (hist_vols.std() + 1e-8))
            feat["btc_vol_ratio"]  = float(cur_vol / (hist_vols.mean() + 1e-8))
        else:
            feat["btc_vol_zscore"] = 0.0
            feat["btc_vol_ratio"]  = 1.0

        # Up-ratio z-scores
        cur_ur = up_ratio_per_rank[rank]
        for win_ur, lbl_ur in [(5, "5s"), (10, "10s"), (20, "20s")]:
            ws = max(0, rank - win_ur)
            hist_ur = up_ratio_per_rank[ws:rank]
            if len(hist_ur) >= 3:
                feat[f"btc_up_ratio_zscore_{lbl_ur}"]    = float((cur_ur - hist_ur.mean()) / (hist_ur.std() + 1e-8))
                feat[f"btc_up_ratio_hist_mean_{lbl_ur}"] = float(hist_ur.mean())
            else:
                feat[f"btc_up_ratio_zscore_{lbl_ur}"]    = 0.0
                feat[f"btc_up_ratio_hist_mean_{lbl_ur}"] = 0.5

        # Sub-window z-scores
        for w in range(6):
            cur_sw = sw_series[rank, w]
            ws = max(0, rank - 20)
            hist_sw = sw_series[ws:rank, w]
            feat[f"btc_up_w{w}_zscore"] = float(
                (cur_sw - hist_sw.mean()) / (hist_sw.std() + 1e-8)
            ) if len(hist_sw) >= 5 else 0.0

        # Realized vol
        for win_rv, lbl_rv in [(5, "5s"), (10, "10s")]:
            ws = max(0, rank - win_rv)
            past_rets = []
            for back_rank in range(ws, rank):
                bslot = rank_to_slotts.get(back_rank)
                if bslot is None: continue
                idx0, idx1 = np.searchsorted(spot_ts_arr, [(bslot - 300) * 1000.0, bslot * 1000.0])
                seg = spot_px_arr[idx0:idx1]
                if len(seg) >= 2:
                    past_rets.append((seg[-1] - seg[0]) / (seg[0] + 1e-8))
            feat[f"btc_realized_vol_{lbl_rv}"] = float(np.std(past_rets)) if len(past_rets) >= 3 else 0.0

        # Standard lag features
        for lag in [1, 2, 3]:
            feat[f"lag_{lag}_outcome"] = float(rank_to_target.get(rank - lag, 0.5)) if rank >= lag else 0.5
        streak = 0
        if rank >= 1:
            last_val = rank_to_target.get(rank - 1, -1)
            for back in range(1, min(rank + 1, 6)):
                v = rank_to_target.get(rank - back, -1)
                if v == last_val and v != -1: streak += 1
                else: break
        feat["lag_streak"] = float(min(streak, 5))

        # ── NEW: prev_slot continuous features ────────────────────────────
        # Richer than binary lag: captures conviction level of previous slot
        for lag_n in [1, 2]:
            prev_rank = rank - lag_n
            if prev_rank >= 0:
                feat[f"prev_slot_up_ratio_{lag_n}"] = float(up_ratio_per_rank[prev_rank])
                feat[f"prev_slot_n_ticks_{lag_n}"]  = float(nticks_series[prev_rank])
                feat[f"prev_slot_vol_{lag_n}"]      = float(vol_series[prev_rank])
            else:
                feat[f"prev_slot_up_ratio_{lag_n}"] = 0.5
                feat[f"prev_slot_n_ticks_{lag_n}"]  = 0.0
                feat[f"prev_slot_vol_{lag_n}"]      = 0.0

        records.append(feat)

    df = pd.DataFrame(records).sort_values("slot_ts").reset_index(drop=True)
    log.info("Dataset: %d samples (skipped %d)", len(df), skipped)
    log.info("Spot coverage: 100%% (Binance)")
    log.info("Target balance: %s", dict(df["target"].value_counts()))

    NON_FEAT = {"market_id", "slot_ts", "target"}
    features = [c for c in df.columns if c not in NON_FEAT]
    log.info("Features: %d total (no redundant/hardcoded)", len(features))

    # ── Walk-forward CV ────────────────────────────────────────────────────
    def walk_forward_purged(df, feats, params=None, gap=WF_GAP, n_splits=N_SPLITS):
        df = df.sort_values("slot_ts").reset_index(drop=True)
        n  = len(df)
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

    # ── Step 6: Baseline WF ────────────────────────────────────────────────
    log.info("Step 6: Baseline walk-forward...")
    wf_base = walk_forward_purged(df, features)
    log.info("  Baseline: AUC=%.4f | Acc=%.4f | Brier=%.4f",
             wf_base["wf_auc"], wf_base["wf_acc"], wf_base["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_base["fold_aucs"]])
    if wf_base["wf_auc"] > 0.99:
        raise RuntimeError("AUC > 0.99 — data loading bug. Aborting.")

    # ── Step 7: Permutation importance → top-30 ───────────────────────────
    log.info("Step 7: Permutation importance → top-%d selection...", TOP_N_FEATS)
    split = int(len(df) * 0.75)
    tr_imp, va_imp = df.iloc[:split], df.iloc[split:]
    imp_mdl = lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", n_estimators=300,
        learning_rate=0.05, num_leaves=31, min_child_samples=15, verbose=-1, n_jobs=-1)
    imp_mdl.fit(tr_imp[features].fillna(0), tr_imp["target"])
    perm = permutation_importance(
        imp_mdl, va_imp[features].fillna(0), va_imp["target"],
        n_repeats=15, random_state=42, scoring="roc_auc")
    imp_df = pd.DataFrame({
        "feature": features,
        "imp_mean": perm.importances_mean,
        "imp_std":  perm.importances_std,
    }).sort_values("imp_mean", ascending=False)

    log.info("  Top %d features:", TOP_N_FEATS)
    for _, r in imp_df.head(TOP_N_FEATS).iterrows():
        log.info("    %-45s %.4f +/- %.4f", r["feature"], r["imp_mean"], r["imp_std"])

    # Fixed top-N selection — deterministic, controls Brier
    good_features = imp_df.head(TOP_N_FEATS)["feature"].tolist()
    dropped = len(features) - len(good_features)
    log.info("  Selected top %d / %d features (dropped %d)", TOP_N_FEATS, len(features), dropped)

    # Check new signals
    for f in ["prev_slot_up_ratio_1", "prev_slot_up_ratio_2",
              "prev_slot_n_ticks_1", "prev_slot_vol_1", "btc_pre_1h_4h_ratio"]:
        rank_row = imp_df[imp_df["feature"] == f]
        if len(rank_row) > 0:
            rank_n = imp_df.index.get_loc(rank_row.index[0]) + 1
            in_top = f in good_features
            log.info("  %s: rank=%d imp=%.4f %s",
                     f, rank_n, rank_row.iloc[0]["imp_mean"], "✓ IN TOP-30" if in_top else "✗ outside top-30")

    # ── Step 8: Optuna HPO ─────────────────────────────────────────────────
    log.info("Step 8: Optuna HPO (%d trials)...", OPTUNA_TRIALS)
    def objective(trial):
        p = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 700),
            learning_rate     = trial.suggest_float("lr", 0.005, 0.15, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 15, 63),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
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

    # ── Step 9: Optimized WF ──────────────────────────────────────────────
    log.info("Step 9: Walk-forward (optimized)...")
    wf_opt = walk_forward_purged(df, good_features, best_params)
    log.info("  Optimized: AUC=%.4f | Acc=%.4f | Brier=%.4f",
             wf_opt["wf_auc"], wf_opt["wf_acc"], wf_opt["wf_brier"])
    log.info("  Fold AUCs: %s", [f"{x:.3f}" for x in wf_opt["fold_aucs"]])

    if wf_opt["wf_auc"] >= wf_base["wf_auc"]:
        final_wf, final_params, final_feats = wf_opt, best_params, good_features
        log.info("  Using: optimized params")
    else:
        final_wf, final_params, final_feats = wf_base, {}, features
        log.info("  Using: baseline params")

    final_auc, final_acc, final_brier = final_wf["wf_auc"], final_wf["wf_acc"], final_wf["wf_brier"]

    # ── Step 10: Final model ───────────────────────────────────────────────
    log.info("Step 10: Training final calibrated model...")
    base_p = dict(objective="binary", class_weight="balanced", n_estimators=400,
                  learning_rate=0.04, num_leaves=31, min_child_samples=15,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=-1)
    params = {**base_p, **final_params, "objective": "binary",
              "class_weight": "balanced", "verbose": -1, "n_jobs": -1}
    X, y = df[final_feats].fillna(0), df["target"]
    final_model = CalibratedClassifierCV(
        lgb.LGBMClassifier(**params), method="isotonic", cv=TimeSeriesSplit(n_splits=3))
    final_model.fit(X, y)

    bundle = {
        "model": final_model, "features": final_feats,
        "wf_auc": final_auc, "wf_acc": final_acc, "wf_brier": final_brier,
        "fold_aucs": final_wf["fold_aucs"], "version": "v16",
        "n_samples": len(df), "n_features": len(final_feats),
        "best_params": final_params, "ensemble": False,
        "spot_source": "binance_1m_klines", "hardcoded_features": "none",
        "feature_selection": f"top-{TOP_N_FEATS} by permutation importance",
    }
    model_path = Path("/tmp/btc_model_v16.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Model saved: %.1f MB", model_path.stat().st_size / 1e6)

    # ── Gate ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("  Gate (v14, 2026): AUC=%.4f  Brier=%.4f  Acc=%.4f", V14_AUC, V14_BRIER, V14_ACC)
    log.info("  v16 candidate:    AUC=%.4f  Brier=%.4f  Acc=%.4f", final_auc, final_brier, final_acc)
    log.info("  Samples: %d | Features: %d (top-%d)", len(df), len(final_feats), TOP_N_FEATS)

    beats_auc   = final_auc   > V14_AUC
    beats_brier = final_brier < V14_BRIER
    beats_acc   = final_acc   > V14_ACC
    n_passed    = sum([beats_auc, beats_brier, beats_acc])
    should_promote = n_passed >= 2

    log.info("  Gate: AUC>%.4f[%s] Brier<%.4f[%s] Acc>%.4f[%s] -> %d/3 | %s",
             V14_AUC, "OK" if beats_auc else "FAIL",
             V14_BRIER, "OK" if beats_brier else "FAIL",
             V14_ACC, "OK" if beats_acc else "FAIL",
             n_passed, "PROMOTE" if should_promote else "REJECT")

    promoted = False
    if should_promote:
        log.info("Promoting v16 to HF champion...")
        api = HfApi(token=HF_TOKEN)
        api.upload_file(path_or_fileobj=str(model_path), path_in_repo="champion.pkl",
                        repo_id=HF_MODEL_REPO, repo_type="model",
                        commit_message=(f"Champion v16: AUC={final_auc:.4f} Brier={final_brier:.4f} "
                                        f"Acc={final_acc:.4f} | {len(df)} samples top-{TOP_N_FEATS}"))
        meta_out = {
            "version": "v16", "feature_list": final_feats, "features": len(final_feats),
            "wf_auc": final_auc, "wf_acc": final_acc, "wf_brier": final_brier,
            "wf_gap": WF_GAP, "wf_n_splits": N_SPLITS, "fold_aucs": final_wf["fold_aucs"],
            "n_samples": len(df), "ensemble": False,
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "spot_source": "binance_1m_klines", "hardcoded_features": "none",
            "feature_selection": f"top-{TOP_N_FEATS}",
            "notes": (f"v16: removed btc_up_ratio/btc_momentum/btc_spot_open (redundant). "
                      f"Added prev_slot_up_ratio/n_ticks/vol (continuous lag signals). "
                      f"Top-{TOP_N_FEATS} fixed selection. Gate vs v14 (2026 regime)."),
            "best_params": final_params,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
            json.dump(meta_out, fp, indent=2)
        api.upload_file(path_or_fileobj=fp.name, path_in_repo="champion_meta.json",
                        repo_id=HF_MODEL_REPO, repo_type="model",
                        commit_message=f"Champion v16 meta AUC={final_auc:.4f}")
        log.info("Champion v16 promoted!")
        promoted = True
    else:
        log.warning("v16 not promoted.")

    log.info("Done.")
    return {
        "wf_auc_baseline": wf_base["wf_auc"], "wf_auc_optimized": wf_opt["wf_auc"],
        "wf_auc_final": final_auc, "wf_acc_final": final_acc, "wf_brier_final": final_brier,
        "v14_auc": V14_AUC, "n_samples": len(df),
        "n_features_final": len(final_feats), "promoted": promoted,
    }


@app.local_entrypoint()
def main():
    print("Submitting BTC v16 — top-30, no redundant features, prev_slot continuous lags")
    r = train_v16.remote()
    print(f"\n{'='*57}")
    print("TRAINING COMPLETE — v16")
    print(f"  Baseline AUC:     {r['wf_auc_baseline']:.4f}")
    print(f"  Optimized AUC:    {r['wf_auc_optimized']:.4f}")
    print(f"  Final AUC:        {r['wf_auc_final']:.4f}")
    print(f"  Final Acc:        {r['wf_acc_final']:.4f}")
    print(f"  Final Brier:      {r['wf_brier_final']:.4f}")
    print(f"  Gate (v14 AUC):   {r['v14_auc']:.4f}")
    print(f"  Samples:          {r['n_samples']}")
    print(f"  Features:         {r['n_features_final']} (top-30)")
    print(f"  Promoted:         {'YES ✓' if r['promoted'] else 'NO'}")
    print(f"{'='*57}")
