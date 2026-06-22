"""
fetch_ob_features_modal.py — Pre-compute OB features from pmdata poly_l2
========================================================================
Fetches 'book' and 'price_change' events from pmdata.dev API for all 22k
markets. Computes ~15 orderbook features + 10 clob_* features per market
and saves as ob_features_full.parquet on Modal Volume.

Runs on Modal with 4 CPUs, 16GB RAM, parallel HTTP fetches.

CUTOFF DESIGN (no lookahead bias):
  Entry window in live: t=170-240s into the 300s slot.
  All features must use data strictly available at t=170s.
  CUTOFF_SEC = 168 (2s buffer for data latency) is applied everywhere.

  - OB static features (ob_mid, ob_spread, etc.): close snapshot t=150-168s
    Matches live: _build_ob_features uses current book at prediction time.
  - OB temporal features (ob_mid_drift, etc.): close(t=150-168s) - open(t=0-30s)
  - ob_imb_w2: t=[120, 168)
  - ob_pc_* aggregate: t=[0, 168)
  - clob_* features: t=[108, 168) — 60s window ending at t=168s,
    matching the live CLOB WS accumulator state at t=170s.

Usage:
    modal run scripts/fetch_ob_features_modal.py
"""
import modal

LOCAL_VOL = modal.Volume.from_name("btc-local-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow>=18.0",
        "pandas>=2.2",
        "numpy>=1.26",
        "requests>=2.31",
    )
)

app = modal.App("btc-fetch-ob", image=image)


@app.function(
    cpu=8,
    memory=32768,
    timeout=86400,  # 24 hours — resumable, skips already-done markets
    secrets=[modal.Secret.from_name("pmdata-api-key")],
    volumes={"/btc_local": LOCAL_VOL},
)
def fetch_ob_features(markets_file: str = "all_markets.csv",
                      out_file: str = "ob_features_full.parquet",
                      progress_file: str = "ob_progress.json",
                      force: bool = False):
    import gc, io, json, logging, os, sys, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import requests

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    PMDATA_KEY  = os.environ.get("PMDATA_API_KEY", "")
    HEADERS     = {"api_key": PMDATA_KEY}
    BASE_URL    = "https://api.pmdata.dev/download/poly_l2"  # new API endpoint
    OBS_SECS    = 180   # load full window, apply CUTOFF inside functions
    CUTOFF_SEC  = 168   # 2s before earliest entry at t=170s — zero lookahead
    LOCAL_DIR   = Path("/btc_local")
    # Parametrizável via argumentos — permite reusar a MESMA lógica para o holdout
    # (close[150,168] + clob[108,168]) sem duplicar o código de cálculo.
    # Defaults preservam o comportamento de treino.
    MARKETS_FILE  = markets_file
    OUT_FILE      = LOCAL_DIR / out_file
    PROGRESS_FILE = LOCAL_DIR / progress_file

    if not PMDATA_KEY:
        raise RuntimeError("PMDATA_API_KEY required — set as Modal Secret 'pmdata-api-key'")

    # ── Resume mode — skip already-processed markets ──────────────────────
    # Progress file tracks done market_ids so we can resume after timeout.
    # Set FORCE_RECOMPUTE=1 env var to wipe and start fresh.
    FORCE = force or os.environ.get("FORCE_RECOMPUTE", "0") == "1"
    if FORCE:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        if OUT_FILE.exists():
            OUT_FILE.unlink()
        log.info("FORCE_RECOMPUTE=1 — starting fresh")

    # ── Load markets ───────────────────────────────────────────────────────
    markets = pd.read_csv(LOCAL_DIR / MARKETS_FILE)
    markets["market_id"] = markets["market_id"].astype(str)
    markets["slot_ts"]   = markets["slot_ts"].astype(int)
    log.info("Markets file: %s | OUT: %s | total markets: %d",
             MARKETS_FILE, OUT_FILE.name, len(markets))

    if "slug" not in markets.columns:
        log.error("all_markets.csv missing 'slug' column! Columns: %s", list(markets.columns))
        raise RuntimeError("Need 'slug' column in all_markets.csv for pmdata API")

    done_ids: set = set()
    existing_rows = []

    # Load existing progress + data if resuming
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                done_ids = set(json.load(f))
            log.info("Resuming: %d markets already done", len(done_ids))
        except Exception as e:
            log.warning("Could not load progress file: %s — starting fresh", e)
            done_ids = set()

    if OUT_FILE.exists() and done_ids:
        try:
            prev_df = pd.read_parquet(OUT_FILE)
            existing_rows = prev_df.to_dict("records")
            log.info("Loaded %d existing rows from parquet", len(existing_rows))
        except Exception as e:
            log.warning("Could not load existing parquet: %s", e)
            existing_rows = []

    todo = markets[~markets["market_id"].isin(list(done_ids))]
    log.info("Markets remaining to process: %d / %d", len(todo), len(markets))

    # ── OB feature extraction ──────────────────────────────────────────────
    def _compute_ob_features(book_rows: pd.DataFrame, pc_rows: pd.DataFrame,
                              slot_ts: int) -> dict | None:
        """
        Compute orderbook features from book snapshots and price_change events.

        CUTOFF design (no lookahead):
          open snapshot: t=0-30s   (slot start state)
          close snapshot: t=150 to CUTOFF_SEC (last state strictly before entry)
          ob_imb_w2: t=120 to CUTOFF_SEC
          pc aggregates: t=0 to CUTOFF_SEC

        Static OB features (ob_mid, ob_spread, ob_imbalance, depth) come from
        close snapshot — matches live where _build_ob_features uses the current
        book at prediction time (~t=170s), not the slot-start book.
        """
        features = {}

        if book_rows is not None and len(book_rows) > 0:
            book_rows = book_rows.sort_values("t_sec")

            def _extract_snap(row) -> dict | None:
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
                    total_bid = bs.sum() + 1e-8
                    total_ask = as_.sum() + 1e-8
                    bd5 = float(bs[bp >= mid - 0.05].sum() / total_bid)
                    ad5 = float(as_[ap <= mid + 0.05].sum() / total_ask)
                    dr  = float(bd5 / (ad5 + 1e-8))
                    total_depth = float(bs.sum() + as_.sum())
                    bid_wt = float(np.sum(bs * np.exp(-10 * np.abs(bp - mid))))
                    ask_wt = float(np.sum(as_ * np.exp(-10 * np.abs(ap - mid))))
                    w_imb  = float((bid_wt - ask_wt) / (bid_wt + ask_wt + 1e-8))
                    return {"mid": mid, "spread": spread, "imb": imb,
                            "bd5": bd5, "ad5": ad5, "dr": dr,
                            "total_depth": total_depth, "w_imb": w_imb}
                except Exception:
                    return None

            # open = t=0-30s (slot start)
            # close = t=150 to CUTOFF_SEC (last available before entry at t=170s)
            open_rows  = book_rows[book_rows["t_sec"] <= 30]
            close_rows = book_rows[(book_rows["t_sec"] >= 150) & (book_rows["t_sec"] < CUTOFF_SEC)]

            open_snap  = _extract_snap(open_rows.iloc[0])   if len(open_rows)  else _extract_snap(book_rows.iloc[0])
            close_snap = _extract_snap(close_rows.iloc[-1]) if len(close_rows) else None

            if open_snap is None:
                return None

            # Use close_snap for static features — matches live (_build_ob_features
            # uses current book at prediction time, not slot-start book).
            # Fall back to open_snap if no close snapshot available.
            static = close_snap if close_snap is not None else open_snap
            features["ob_mid"]          = static["mid"]
            features["ob_spread"]       = static["spread"]
            features["ob_imbalance"]    = static["imb"]
            features["ob_depth_ratio"]  = static["dr"]
            features["ob_bid_depth_5c"] = static["bd5"]
            features["ob_ask_depth_5c"] = static["ad5"]
            features["ob_total_depth"]  = static["total_depth"]
            features["ob_weighted_imb"] = static["w_imb"]

            # Temporal features: close(t≈168s) vs open(t≈0-30s)
            if close_snap is not None:
                features["ob_mid_drift"]     = float(close_snap["mid"] - open_snap["mid"])
                features["ob_imbalance_end"] = float(close_snap["imb"])
                features["ob_spread_end"]    = float(close_snap["spread"])
                features["ob_depth_change"]  = float(close_snap["total_depth"] - open_snap["total_depth"])
                features["ob_imb_momentum"]  = float(close_snap["imb"] - open_snap["imb"])
            else:
                features["ob_mid_drift"]     = 0.0
                features["ob_imbalance_end"] = open_snap["imb"]
                features["ob_spread_end"]    = open_snap["spread"]
                features["ob_depth_change"]  = 0.0
                features["ob_imb_momentum"]  = 0.0

            # Windowed OB imbalance — w2 capped at CUTOFF_SEC (no lookahead)
            for w_i, (t0, t1) in enumerate([(0, 60), (60, 120), (120, CUTOFF_SEC)]):
                w_rows = book_rows[(book_rows["t_sec"] >= t0) & (book_rows["t_sec"] < t1)]
                if len(w_rows) > 0:
                    imbs = []
                    for _, r in w_rows.iterrows():
                        s = _extract_snap(r)
                        if s is not None:
                            imbs.append(s["imb"])
                    features[f"ob_imb_w{w_i}"] = float(np.mean(imbs)) if imbs else 0.0
                else:
                    features[f"ob_imb_w{w_i}"] = 0.0
        else:
            return None  # No book data = skip market

        # ── Price change features — filter to t < CUTOFF_SEC (no lookahead) ──
        if pc_rows is not None and len(pc_rows) > 0:
            pc_cut = pc_rows[pc_rows["t_sec"] < CUTOFF_SEC].copy()
            if len(pc_cut) > 0:
                pc_cut = pc_cut.sort_values("t_sec")
            pc_rows = pc_cut

        if pc_rows is not None and len(pc_rows) > 0:
            if "pc_price" in pc_rows.columns:
                prices = pc_rows["pc_price"].dropna().values.astype(float)
                diffs  = np.diff(prices)
                n_up   = int((diffs > 0).sum())
                n_dn   = int((diffs < 0).sum())
                total_changes = n_up + n_dn
                features["ob_pc_up_ratio"]   = float(n_up / (total_changes + 1e-8))
                features["ob_pc_volatility"] = float(np.std(diffs)) if len(diffs) > 1 else 0.0
                features["ob_pc_count"]      = float(total_changes)
            else:
                features["ob_pc_up_ratio"]   = 0.5
                features["ob_pc_volatility"] = 0.0
                features["ob_pc_count"]      = 0.0

            if "pc_side" in pc_rows.columns:
                buys     = pc_rows[pc_rows["pc_side"].str.upper() == "BUY"]
                sells    = pc_rows[pc_rows["pc_side"].str.upper() == "SELL"]
                buy_vol  = buys["pc_size"].sum()  if "pc_size"  in buys.columns  else len(buys)
                sell_vol = sells["pc_size"].sum() if "pc_size"  in sells.columns else len(sells)
                features["ob_fill_imbalance"] = float((buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8))
            else:
                features["ob_fill_imbalance"] = 0.0
        else:
            features["ob_pc_up_ratio"]    = 0.5
            features["ob_pc_volatility"]  = 0.0
            features["ob_pc_count"]       = 0.0
            features["ob_fill_imbalance"] = 0.0

        return features

    def _compute_clob_features(book_rows: pd.DataFrame, pc_rows: pd.DataFrame,
                                window_start: float = 108.0, window_end: float = 168.0) -> dict:
        """
        Compute clob_* features — 60s window ending at CUTOFF_SEC (t=168s).

        Window t=[108, 168) mirrors what the live CLOB WS accumulator holds
        at t=170s: accumulator has window_secs=60, so at t=170s it contains
        t=[110, 170]. We use t=[108, 168) for a 2s safety buffer.

        Mirrors live clob_features.py exactly:
          - real book snapshots for imbalance/depth
          - all events (book + best_ask/best_bid from price_change) for mid/spread/velocity
          - real_books + price_changes (no double-count) for activity_rate
        """
        zeros = {
            "clob_imb_mean": 0.0, "clob_imb_std": 0.0, "clob_imb_drift": 0.0,
            "clob_spread_mean": 0.0, "clob_spread_trend": 0.0,
            "clob_mid_velocity": 0.0, "clob_mid_volatility": 0.0,
            "clob_activity_rate": 0.0, "clob_depth_trend": 0.0,
            "clob_ask_pressure": 0.0,
        }

        def _linslope(t, y):
            if len(t) < 2 or t[-1] == t[0]:
                return 0.0
            try:
                return float(np.polyfit(t, y, 1)[0])
            except Exception:
                return 0.0

        w_books = book_rows[
            (book_rows["t_sec"] >= window_start) & (book_rows["t_sec"] < window_end)
        ] if len(book_rows) > 0 else pd.DataFrame()

        w_pcs = pc_rows[
            (pc_rows["t_sec"] >= window_start) & (pc_rows["t_sec"] < window_end)
        ] if len(pc_rows) > 0 else pd.DataFrame()

        n_real_books = len(w_books)
        n_pcs = len(w_pcs)

        if n_real_books + n_pcs < 5:
            return zeros

        time_span = window_end - window_start  # 60s

        # Real book snapshots: imbalance, depth, mid, spread
        real_imb, real_depth, real_mid, real_spread, real_ts = [], [], [], [], []
        for _, row in w_books.iterrows():
            try:
                ap = np.array(row["ask_prices"], dtype=np.float64)
                bp = np.array(row["bid_prices"], dtype=np.float64)
                as_ = np.array(row["ask_sizes"], dtype=np.float64)
                bs  = np.array(row["bid_sizes"], dtype=np.float64)
                if not len(ap) or not len(bp):
                    continue
                mid    = float((ap[0] + bp[0]) / 2)
                spread = float(ap[0] - bp[0])
                tsz    = float(as_[0] + bs[0])
                imb    = float((bs[0] - as_[0]) / tsz) if tsz > 0 else 0.0
                depth  = float(as_.sum() + bs.sum())
                real_imb.append(imb); real_depth.append(depth)
                real_mid.append(mid); real_spread.append(spread)
                real_ts.append(float(row["t_sec"]))
            except Exception:
                continue

        # price_change events: best_bid/best_ask for mid/spread time-series
        all_mid, all_spread, all_ts = list(real_mid), list(real_spread), list(real_ts)
        ask_prices_seq = []
        for _, row in w_pcs.iterrows():
            try:
                ba = float(row["best_ask"]) if pd.notna(row["best_ask"]) else None
                bb = float(row["best_bid"]) if pd.notna(row["best_bid"]) else None
                if ba is not None and bb is not None and ba > bb > 0:
                    all_mid.append((ba + bb) / 2)
                    all_spread.append(ba - bb)
                    all_ts.append(float(row["t_sec"]))
                side = str(row["pc_side"]).upper() if pd.notna(row["pc_side"]) else ""
                if side == "SELL" and pd.notna(row["pc_price"]):
                    ask_prices_seq.append(float(row["pc_price"]))
            except Exception:
                continue

        if len(all_mid) < 2:
            return zeros

        order = np.argsort(all_ts)
        all_ts_arr     = np.array(all_ts)[order]
        all_mid_arr    = np.array(all_mid)[order]
        all_spread_arr = np.array(all_spread)[order]
        t_rel          = all_ts_arr - all_ts_arr[0]

        feats = {}
        feats["clob_imb_mean"]       = float(np.mean(real_imb))             if real_imb           else 0.0
        feats["clob_imb_std"]        = float(np.std(real_imb))              if len(real_imb) > 1  else 0.0
        feats["clob_imb_drift"]      = float(real_imb[-1] - real_imb[0])   if len(real_imb) > 1  else 0.0
        feats["clob_spread_mean"]    = float(np.mean(all_spread_arr))
        feats["clob_spread_trend"]   = _linslope(t_rel, all_spread_arr)
        feats["clob_mid_velocity"]   = _linslope(t_rel, all_mid_arr)
        mid_diffs = np.diff(all_mid_arr)
        feats["clob_mid_volatility"] = float(np.std(mid_diffs))             if len(mid_diffs) > 0 else 0.0
        feats["clob_activity_rate"]  = float((n_real_books + n_pcs) / time_span)
        if len(real_depth) > 1:
            rta = np.array(real_ts)
            feats["clob_depth_trend"] = _linslope(rta - rta[0], np.array(real_depth))
        else:
            feats["clob_depth_trend"] = 0.0
        if len(ask_prices_seq) >= 2:
            diffs = np.diff(ask_prices_seq)
            moves = diffs[diffs != 0]
            feats["clob_ask_pressure"] = float((moves < 0).sum() / len(moves)) if len(moves) > 0 else 0.0
        else:
            feats["clob_ask_pressure"] = 0.0

        return feats

    def _fetch_one(market_id: str, slug: str, slot_ts: int) -> tuple[str, dict | None]:
        """Fetch poly_l2 parquet and extract OB + clob_* features."""
        try:
            # New API endpoint: /download/{data_type}/{slug}.parquet
            url = f"{BASE_URL}/{slug}.parquet"
            r = requests.get(url, headers=HEADERS, timeout=120)
            if not r.ok:
                return (market_id, None)

            df = pd.read_parquet(io.BytesIO(r.content))

            # Convert timestamp — poly_l2 uses datetime64[ms], convert to Unix seconds
            if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["ts_sec"] = df["timestamp"].astype("int64") / 1000.0  # ms → seconds
            else:
                df["ts_sec"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)
            df["t_sec"] = df["ts_sec"] - slot_ts

            # Load full [0, 180) window — CUTOFF_SEC applied inside functions
            df = df[(df["t_sec"] >= 0) & (df["t_sec"] < OBS_SECS)]

            books = df[df["event_type"] == "book"].copy()
            pcs   = df[df["event_type"] == "price_change"].copy()

            feats = _compute_ob_features(
                books if len(books) > 0 else None,
                pcs   if len(pcs)   > 0 else None,
                slot_ts,
            )
            if feats is not None:
                clob_feats = _compute_clob_features(
                    books if len(books) > 0 else pd.DataFrame(),
                    pcs   if len(pcs)   > 0 else pd.DataFrame(),
                    window_start=108.0, window_end=168.0,
                )
                feats.update(clob_feats)
                feats["market_id"] = market_id
            return (market_id, feats)
        except Exception:
            return (market_id, None)

    # ── Parallel fetch ─────────────────────────────────────────────────────
    new_rows = []
    success = 0
    failed  = 0
    BATCH_SIZE = 500
    WORKERS    = 40

    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo.iloc[batch_start:batch_start + BATCH_SIZE]

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {
                ex.submit(_fetch_one, str(r["market_id"]), r["slug"], int(r["slot_ts"])): str(r["market_id"])
                for _, r in batch.iterrows()
            }
            for fut in as_completed(futures):
                mid, feats = fut.result()
                done_ids.add(mid)
                if feats is not None:
                    new_rows.append(feats)
                    success += 1
                else:
                    failed += 1

        # Save progress every batch
        with open(PROGRESS_FILE, "w") as f:
            json.dump(list(done_ids), f)

        # Flush to parquet every batch
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            all_rows_df = pd.DataFrame(existing_rows + new_rows) if existing_rows else new_df
            all_rows_df = all_rows_df.drop_duplicates(subset=["market_id"])
            all_rows_df.to_parquet(OUT_FILE, index=False, compression="snappy")
            existing_rows = all_rows_df.to_dict("records")
            new_rows = []

        LOCAL_VOL.commit()  # persist to volume

        done_so_far = batch_start + len(batch)
        log.info("%d/%d | success=%d failed=%d total_saved=%d",
                 done_so_far, len(todo), success, failed, len(existing_rows))

        time.sleep(0.05)

    LOCAL_VOL.commit()
    log.info("DONE. OB features: %d markets (failed: %d)", len(existing_rows), failed)


@app.local_entrypoint()
def main(markets_file: str = "all_markets.csv",
         out_file: str = "ob_features_full.parquet",
         progress_file: str = "ob_progress.json",
         force: bool = False):
    fetch_ob_features.remote(markets_file=markets_file, out_file=out_file,
                             progress_file=progress_file, force=force)
