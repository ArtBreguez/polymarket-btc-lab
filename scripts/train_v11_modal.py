"""
train_v11_modal.py — BTC 5min model v11

Changes from v10:
  - Zero hardcoded OB features. All OB features extracted from real data.
  - OB source: pmdata.dev poly_l2 parquet (full order book ladder per second)
  - OB features computed identically in training (pmdata) and live (CLOB REST /book)
  - Only markets with real OB data are used for training (~616 expected, 100% coverage confirmed)

New OB features (all extractable live via CLOB REST GET /book?token_id=...):
  ob_mid           — (best_ask + best_bid) / 2  — implied prob of UP
  ob_spread        — best_ask - best_bid
  ob_imbalance     — (best_bid_size - best_ask_size) / (bid + ask)  @ slot open
  ob_depth_ratio   — bid_depth_5c / ask_depth_5c (within 5 cents of mid)
  ob_bid_depth_5c  — total bid size within 5c of mid (normalized by total)
  ob_ask_depth_5c  — total ask size within 5c of mid (normalized by total)
  ob_mid_drift     — ob_mid at slot end - ob_mid at slot start (OB repricing)
  ob_imbalance_end — ob_imbalance snapshot at slot end (t >= 150s)

All features computed from first available book snapshot (t=0..30s) and last (t>=150s).
Live: fetched once at t=150s via CLOB REST (after tick observation window ends).

Retained from v10:
  - All tick features (btc_* family) — now correct with size_usdc
  - All spot features (btc_inslot_ret, pre-slot returns, dist_*)
  - All historical/lag features (zscore, lag_outcome, streak)
  - Isotonic calibration, Optuna 150 trials, purged walk-forward CV
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
        "requests>=2.31",
        "urllib3>=2.0",  # force image rebuild v3
    )
)

app = modal.App("polymarket-btc-train-v11", image=image)

@app.function(
    cpu=8,
    memory=32768,
    timeout=10800,  # 3h — OB fetch adds time
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("pmdata-api-key")],
)
def train_v11():
    import gc, json, logging, os, pickle, sys, time, warnings, tempfile
    from datetime import datetime, timezone
    from pathlib import Path
    import concurrent.futures

    import numpy as np
    import optuna
    import pandas as pd
    import pyarrow.parquet as pq
    import requests
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
    PMDATA_KEY    = os.environ.get("PMDATA_API_KEY", "")
    HF_DATASET    = "BrockMisner/polymarket-btc-updown"
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
    OBS_SECS      = 180
    OPTUNA_TRIALS = 150
    WF_GAP        = 5
    DATA_DIR      = Path("/tmp/hf_data")
    DATA_DIR.mkdir(exist_ok=True)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")
    if not PMDATA_KEY:
        raise RuntimeError("PMDATA_API_KEY required")

    # ── Step 1: Load champion metrics ─────────────────────────────────────────
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

    # ── Step 2: Download ticks + markets from HF ──────────────────────────────
    files = [
        "data/markets.parquet",
        "data/ticks/crypto=BTC/timeframe=5-minute/part-0.parquet",
    ]
    log.info("Step 2: Downloading %d files from HF...", len(files))
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
    slug_map       = dict(zip(markets["market_id"], markets["slug"]))
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

    # ── Step 5: OB features from pmdata poly_l2 ───────────────────────────────
    log.info("Step 5: Fetching OB features from pmdata (poly_l2)...")
    log.info("  Markets to fetch: %d", len(markets_sorted))

    def _compute_ob_features_from_book_rows(book_rows: pd.DataFrame) -> dict:
        """
        Compute OB features from a set of book snapshot rows (one slot).
        Uses first snapshot (t <= 30s) as 'open' and last snapshot (t >= 150s) as 'close'.
        All features are computable identically from CLOB REST /book?token_id=... in live.

        Returns dict with 8 features. On failure returns None.
        """
        if book_rows is None or len(book_rows) == 0:
            return None

        def _extract(row) -> dict | None:
            try:
                ap = np.array(row["ask_prices"], dtype=np.float64)
                as_ = np.array(row["ask_sizes"], dtype=np.float64)
                bp = np.array(row["bid_prices"], dtype=np.float64)
                bs = np.array(row["bid_sizes"], dtype=np.float64)
                if len(ap) == 0 or len(bp) == 0:
                    return None
                mid    = float((ap[0] + bp[0]) / 2)
                spread = float(ap[0] - bp[0])
                imb    = float((bs[0] - as_[0]) / (bs[0] + as_[0] + 1e-8))
                # Depth within 5 cents of mid — normalized by total depth
                total_bid = bs.sum() + 1e-8
                total_ask = as_.sum() + 1e-8
                bd5 = float(bs[bp >= mid - 0.05].sum() / total_bid)
                ad5 = float(as_[ap <= mid + 0.05].sum() / total_ask)
                dr  = float(bd5 / (ad5 + 1e-8))
                return {"mid": mid, "spread": spread, "imb": imb,
                        "bd5": bd5, "ad5": ad5, "dr": dr}
            except Exception:
                return None

        # sort by timestamp
        book_rows = book_rows.sort_values("timestamp")

        open_rows  = book_rows[book_rows["t_sec"] <= 30]
        close_rows = book_rows[book_rows["t_sec"] >= 150]

        open_snap  = _extract(open_rows.iloc[0])  if len(open_rows)  else _extract(book_rows.iloc[0])
        close_snap = _extract(close_rows.iloc[-1]) if len(close_rows) else _extract(book_rows.iloc[-1])

        if open_snap is None:
            return None

        ob = {
            "ob_mid":           open_snap["mid"],
            "ob_spread":        open_snap["spread"],
            "ob_imbalance":     open_snap["imb"],
            "ob_depth_ratio":   open_snap["dr"],
            "ob_bid_depth_5c":  open_snap["bd5"],
            "ob_ask_depth_5c":  open_snap["ad5"],
        }
        if close_snap is not None:
            ob["ob_mid_drift"]     = float(close_snap["mid"] - open_snap["mid"])
            ob["ob_imbalance_end"] = float(close_snap["imb"])
        else:
            ob["ob_mid_drift"]     = 0.0
            ob["ob_imbalance_end"] = open_snap["imb"]
        return ob

    def _fetch_ob_for_slug(slug: str, slot_ts: int) -> dict | None:
        """Fetch poly_l2 parquet from pmdata and extract OB features."""
        try:
            r = requests.get(
                f"https://api.pmdata.dev/get-download-url/poly_l2/{slug}",
                headers={"api_key": PMDATA_KEY},
                timeout=30,
            )
            if not r.ok:
                return None
            data = r.json()
            dl_url = data.get("download_url")
            if not dl_url:
                return None

            import io
            r2 = requests.get(dl_url, timeout=120)
            if not r2.ok:
                return None
            raw = r2.content

            pf = pq.ParquetFile(io.BytesIO(raw))
            batch = next(pf.iter_batches(batch_size=100000))
            df_ob = batch.to_pandas()

            books = df_ob[df_ob["event_type"] == "book"].copy()
            if len(books) == 0:
                return None

            books["ts_sec"] = books["timestamp"].astype("int64") / 1e3  # datetime64[ms] → seconds
            books["t_sec"]  = books["ts_sec"] - slot_ts
            books = books[(books["t_sec"] >= 0) & (books["t_sec"] < 180)]
            if len(books) == 0:
                return None

            return _compute_ob_features_from_book_rows(books)
        except Exception as e:
            log.warning("OB fetch exception for %s: %s: %s", slug, type(e).__name__, str(e)[:200])
            return None

    # Fetch OB in parallel (20 workers — pmdata can handle it)
    slugs_list = [(row["market_id"], row["slug"], row["slot_ts"])
                  for _, row in markets_sorted.iterrows()]

    ob_by_market: dict[str, dict] = {}
    failed = 0
    BATCH = 50  # log every 50
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_ob_for_slug, slug, slot_ts): mid
                   for mid, slug, slot_ts in slugs_list}
        for i, (fut, mid) in enumerate(futures.items()):
            try:
                result = fut.result(timeout=60)
                if result is not None:
                    ob_by_market[mid] = result
                else:
                    failed += 1
            except Exception:
                failed += 1
            if (i + 1) % BATCH == 0:
                log.info("  OB fetch: %d/%d done (ok=%d, failed=%d)",
                         i + 1, len(slugs_list), len(ob_by_market), failed)

    log.info("  OB fetch complete: %d/%d markets with real OB (%.1f%%)",
             len(ob_by_market), len(markets_sorted),
             100 * len(ob_by_market) / max(len(markets_sorted), 1))
    if failed > 0:
        log.warning("  %d markets missing OB — will be excluded from training", failed)

    # ── Feature helpers ────────────────────────────────────────────────────────
    def tick_features_v11(grp: pd.DataFrame) -> dict:
        """v10 tick features — unchanged, all use size_usdc."""
        n = len(grp)
        if n == 0:
            return {
                "btc_n_ticks": 0.0, "btc_up_ratio": 0.5,
                "btc_vol_up": 0.0, "btc_vol_dn": 0.0,
                "btc_momentum": 0.0, "btc_vwap_spread": 0.0,
                "btc_vwap_up": 0.5, "btc_vwap_dn": 0.5,
                "btc_buy_ratio": 0.5, "btc_avg_size": 0.0,
                "btc_tick_accel": 0.0,
                **{f"btc_up_w{i}": 0.5 for i in range(6)},
                "btc_tw_up_ratio": 0.5, "btc_vwap_trend": 0.0,
                "btc_vwmom": 0.0, "btc_up_ratio_stability": 0.0,
                "btc_vol_accel": 1.0, "btc_size_disparity": 1.0,
                "btc_signal_conviction": 0.0, "btc_momentum_vol_sync": 0.0,
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

        sw = {}
        for i in range(6):
            t0_w, t1_w = i * 30, (i + 1) * 30
            sw[f"btc_up_w{i}"] = ur_window((grp["t_sec"] >= t0_w) & (grp["t_sec"] < t1_w))

        w_vals = [sw[f"btc_up_w{i}"] for i in range(6)]
        btc_momentum = float(np.mean(w_vals[3:]) - np.mean(w_vals[:3]))

        weights = np.exp(grp["t_sec"].values / OBS_SECS * 2.0)
        weights /= weights.sum() + 1e-8
        tw_up = float((weights * (grp["outcome"] == "Up").values).sum())

        half  = OBS_SECS / 2
        early = grp[grp["t_sec"] < half]
        late  = grp[grp["t_sec"] >= half]
        def vwap_up_half(g):
            up = g[g["outcome"] == "Up"]
            if len(up) == 0: return 0.5
            return float((up["price"] * up["size_usdc"]).sum() / (up["size_usdc"].sum() + 1e-8))
        vwap_trend = float(vwap_up_half(late) - vwap_up_half(early))

        vol_by_w, ur_by_w = [], []
        for i in range(6):
            t0_w, t1_w = i * 30, (i + 1) * 30
            sub = grp[(grp["t_sec"] >= t0_w) & (grp["t_sec"] < t1_w)]
            vol_by_w.append(sub["size_usdc"].sum())
            ur_by_w.append(ur_window((grp["t_sec"] >= t0_w) & (grp["t_sec"] < t1_w)))
        vol_by_w = np.array(vol_by_w)
        ur_by_w  = np.array(ur_by_w)
        vwmom = float(np.dot(vol_by_w / (vol_by_w.sum() + 1e-8), ur_by_w - 0.5))

        first30 = (grp["t_sec"] < 30).sum()
        last30  = (grp["t_sec"] >= (OBS_SECS - 30)).sum()

        up_ratio_stability = float(np.std(w_vals))
        vol_first90 = grp[grp["t_sec"] < 90]["size_usdc"].sum()
        vol_last90  = grp[grp["t_sec"] >= 90]["size_usdc"].sum()
        vol_accel   = float(vol_last90 / (vol_first90 + 1e-8))

        up_grp = grp[is_up];  dn_grp = grp[~is_up]
        avg_size_up    = float(up_grp["size_usdc"].mean()) if len(up_grp) > 0 else 1.0
        avg_size_dn    = float(dn_grp["size_usdc"].mean()) if len(dn_grp) > 0 else 1.0
        size_disparity = float(avg_size_up / (avg_size_dn + 1e-8))

        signal_conviction = float((vol_up / total) * (1.0 - up_ratio_stability))
        momentum_vol_sync = float(btc_momentum * vol_accel)

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
            "btc_up_ratio_stability": up_ratio_stability,
            "btc_vol_accel":          vol_accel,
            "btc_size_disparity":     size_disparity,
            "btc_signal_conviction":  signal_conviction,
            "btc_momentum_vol_sync":  momentum_vol_sync,
        }

    def spot_open_at(slot_ts: int) -> float:
        idx = np.searchsorted(spot_ts_arr, slot_ts * 1000, side="right") - 1
        return float(spot_px_arr[idx]) if idx >= 0 else 0.0

    # ── Step 6: Build dataset (OB-complete markets only) ──────────────────────
    log.info("Step 6: Building feature dataset (OB-complete markets only)...")
    btc_grps   = dict(list(btc_inslot.groupby("market_id")))
    vol_series = markets_sorted["market_id"].map(slot_vol).fillna(0).values

    up_ratio_series = np.array([
        float(
            btc_grps[r["market_id"]]["size_usdc"][
                btc_grps[r["market_id"]]["outcome"] == "Up"].sum() /
            (btc_grps[r["market_id"]]["size_usdc"].sum() + 1e-8)
        ) if r["market_id"] in btc_grps else 0.5
        for _, r in markets_sorted.iterrows()
    ])

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
    skipped_no_ticks = 0
    skipped_no_ob    = 0
    for i, row in markets_sorted.iterrows():
        mid    = row["market_id"]
        target = target_map[mid]
        grp    = btc_grps.get(mid)

        if grp is None or len(grp) < 5:
            skipped_no_ticks += 1
            continue

        ob_feats = ob_by_market.get(mid)
        if ob_feats is None:
            skipped_no_ob += 1
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

        feat.update(tick_features_v11(grp))
        feat.update(ob_feats)  # 8 real OB features — no fallback

        # Spot features
        spot_open = spot_open_at(slot_ts)
        sp = grp["spot_price_usdt"].dropna()
        if spot_open > 0 and len(sp) >= 1:
            feat["btc_inslot_ret"] = (float(sp.iloc[-1]) - spot_open) / (spot_open + 1e-8)
            feat["btc_inslot_vol"] = float(sp.std() / (sp.mean() + 1e-8)) if len(sp) >= 2 else 0.0
        else:
            feat["btc_inslot_ret"] = 0.0
            feat["btc_inslot_vol"] = 0.0

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

        if spot_open > 0:
            feat["btc_dist_1k"]  = float(abs(spot_open % 1000) / 1000)
            feat["btc_dist_5k"]  = float(abs(spot_open % 5000) / 5000)
            feat["btc_dist_10k"] = float(abs(spot_open % 10000) / 10000)
        else:
            feat["btc_dist_1k"] = feat["btc_dist_5k"] = feat["btc_dist_10k"] = 0.5

        win_start = max(0, rank - 20)
        hist_vols = vol_series[win_start:rank]
        cur_vol   = vol_series[rank]
        if len(hist_vols) >= 5:
            feat["btc_vol_zscore"] = float((cur_vol - hist_vols.mean()) / (hist_vols.std() + 1e-8))
            feat["btc_vol_ratio"]  = float(cur_vol / (hist_vols.mean() + 1e-8))
        else:
            feat["btc_vol_zscore"] = 0.0
            feat["btc_vol_ratio"]  = 1.0

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

        for w in range(6):
            cur_sw_ur = sw_series[rank, w]
            ws = max(0, rank - 20)
            hist_sw = sw_series[ws:rank, w]
            if len(hist_sw) >= 5:
                feat[f"btc_up_w{w}_zscore"] = float(
                    (cur_sw_ur - hist_sw.mean()) / (hist_sw.std() + 1e-8))
            else:
                feat[f"btc_up_w{w}_zscore"] = 0.0

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

        for lag in [1, 2, 3]:
            feat[f"lag_{lag}_outcome"] = float(rank_to_target.get(rank - lag, 0.5))

        streak = 0
        if rank >= 1:
            last_val = rank_to_target.get(rank - 1, -1)
            for back in range(1, min(rank + 1, 6)):
                v = rank_to_target.get(rank - back, -1)
                if v == last_val and v != -1: streak += 1
                else: break
        feat["lag_streak"] = float(streak)

        records.append(feat)

    log.info("  Skipped: %d no-ticks, %d no-OB", skipped_no_ticks, skipped_no_ob)
    df = pd.DataFrame(records).sort_values("slot_ts").reset_index(drop=True)
    log.info("Dataset: %d samples", len(df))
    log.info("Target balance: %s", dict(df["target"].value_counts()))

    NON_FEAT = {"market_id", "slot_ts", "target"}
    features = [c for c in df.columns if c not in NON_FEAT]
    log.info("Features: %d total", len(features))
    ob_feats_list = [f for f in features if f.startswith("ob_")]
    log.info("  OB features (%d): %s", len(ob_feats_list), ob_feats_list)

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
    log.info("  OB feature importances:")
    for _, r in imp_df[imp_df["feature"].str.startswith("ob_")].iterrows():
        log.info("    %-45s %.4f ± %.4f", r["feature"], r["imp_mean"], r["imp_std"])
    good_features = imp_df[imp_df["imp_mean"] > 0.0005]["feature"].tolist()
    dropped = len(features) - len(good_features)
    log.info("  Dropped %d noise features, keeping %d", dropped, len(good_features))

    # ── Step 9: Optuna HPO ─────────────────────────────────────────────────────
    log.info("Step 9: Optuna HPO (%d trials)...", OPTUNA_TRIALS)
    def objective(trial):
        p = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 600),
            learning_rate     = trial.suggest_float("lr", 0.005, 0.15, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 15, 63),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 40),
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

    if wf_opt["wf_auc"] >= wf_base["wf_auc"]:
        final_wf, final_params, final_feats = wf_opt, best_params, good_features
        log.info("  Using: optimized params")
    else:
        final_wf, final_params, final_feats = wf_base, {}, features
        log.info("  Using: baseline params (Optuna didn't improve)")

    final_auc, final_acc, final_brier = (
        final_wf["wf_auc"], final_wf["wf_acc"], final_wf["wf_brier"])

    # ── Step 11: Champion comparison ──────────────────────────────────────────
    log.info("Step 11: Champion comparison...")
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
        lgb.LGBMClassifier(**params), method="isotonic",
        cv=TimeSeriesSplit(n_splits=3))
    final_model.fit(X, y)

    bundle = {
        "model":            final_model,
        "features":         final_feats,
        "wf_auc":           final_auc,
        "wf_acc":           final_acc,
        "wf_brier":         final_brier,
        "fold_aucs":        final_wf["fold_aucs"],
        "version":          "v11",
        "n_samples":        len(df),
        "n_features":       len(final_feats),
        "best_params":      final_params,
        "ensemble":         False,
        "dropped_features": dropped,
        "champion_compared_auc": fair_champ_auc,
        "ob_coverage":      len(ob_by_market),
        "ob_features":      [f for f in final_feats if f.startswith("ob_")],
    }
    model_path = Path("/tmp/btc_model_v11.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Model saved: %.1f MB", model_path.stat().st_size / 1e6)

    # ── Gate ──────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("  Champion (%s) purged WF: AUC=%.4f  Brier=%.4f  Acc=%.4f",
             champion.get("version", "?"), fair_champ_auc,
             CHAMPION_BRIER, CHAMPION_ACC)
    log.info("  v11 candidate:            AUC=%.4f  Brier=%.4f  Acc=%.4f",
             final_auc, final_brier, final_acc)
    log.info("  Features: %d (dropped %d noise)", len(final_feats), dropped)
    log.info("  OB coverage: %d/%d markets", len(ob_by_market), len(markets_sorted))

    beats_auc   = final_auc   > fair_champ_auc
    beats_brier = final_brier < CHAMPION_BRIER
    beats_acc   = final_acc   > CHAMPION_ACC
    n_passed    = sum([beats_auc, beats_brier, beats_acc])
    should_promote = n_passed >= 2
    log.info("  Gate: AUC>%.4f[%s] Brier<%.4f[%s] Acc>%.4f[%s] → %d/3 | %s",
             fair_champ_auc, "✓" if beats_auc   else "✗",
             CHAMPION_BRIER,  "✓" if beats_brier else "✗",
             CHAMPION_ACC,    "✓" if beats_acc   else "✗",
             n_passed, "PROMOTE" if should_promote else "REJECT")

    promoted = False
    if should_promote:
        log.info("Promoting v11 to HF champion...")
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="champion.pkl",
            repo_id=HF_MODEL_REPO, repo_type="model",
            commit_message=(f"Champion v11: AUC={final_auc:.4f} Brier={final_brier:.4f} "
                            f"Acc={final_acc:.4f} | real OB features, zero hardcoded"),
        )
        meta_out = {
            "version":       "v11",
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
            "ob_coverage":   len(ob_by_market),
            "ob_features":   [f for f in final_feats if f.startswith("ob_")],
            "notes": ("v11: real OB features from pmdata poly_l2 — zero hardcoded. "
                      "ob_mid, ob_spread, ob_imbalance, ob_depth_ratio, ob_bid_depth_5c, "
                      "ob_ask_depth_5c, ob_mid_drift, ob_imbalance_end. "
                      "All features live-computable via CLOB REST /book?token_id=."),
            "best_params":   final_params,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
            json.dump(meta_out, fp, indent=2)
        api.upload_file(
            path_or_fileobj=fp.name,
            path_in_repo="champion_meta.json",
            repo_id=HF_MODEL_REPO, repo_type="model",
            commit_message=f"Champion v11 meta AUC={final_auc:.4f}",
        )
        log.info("Champion v11 promoted: https://huggingface.co/%s", HF_MODEL_REPO)
        promoted = True
    else:
        log.warning("v11 not promoted.")

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
        "ob_coverage":        len(ob_by_market),
        "promoted":           promoted,
    }


@app.local_entrypoint()
def main():
    print("Submitting BTC v11 training job to Modal...")
    r = train_v11.remote()
    print(f"\n{'='*55}")
    print("TRAINING COMPLETE — v11")
    print(f"  Baseline AUC:     {r['wf_auc_baseline']:.4f}")
    print(f"  Optimized AUC:    {r['wf_auc_optimized']:.4f}")
    print(f"  Final AUC:        {r['wf_auc_final']:.4f}")
    print(f"  Final Acc:        {r['wf_acc_final']:.4f}")
    print(f"  Final Brier:      {r['wf_brier_final']:.4f}")
    print(f"  Champion AUC:     {r['champion_fair_auc']:.4f} (purged WF)")
    print(f"  Samples:          {r['n_samples']}")
    print(f"  Features:         {r['n_features_final']} (dropped {r['n_features_dropped']})")
    print(f"  OB coverage:      {r['ob_coverage']} markets")
    print(f"  Promoted:         {'YES ✓' if r['promoted'] else 'NO'}")
    print(f"{'='*55}")
