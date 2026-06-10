"""
fetch_ob_features_modal.py — Pre-compute OB features from pmdata poly_l2
========================================================================
Fetches 'book' and 'price_change' events from pmdata.dev API for all 22k
markets. Computes ~15 orderbook features per market and saves as
ob_features_full.parquet on Modal Volume.

Runs on Modal with 4 CPUs, 16GB RAM, parallel HTTP fetches.
Resume-safe via progress tracking.

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
    cpu=4,
    memory=16384,
    timeout=10800,  # 3 hours — 22k markets takes a while
    secrets=[modal.Secret.from_name("pmdata-api-key")],
    volumes={"/btc_local": LOCAL_VOL},
)
def fetch_ob_features():
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
    BASE_URL    = "https://api.pmdata.dev/get-download-url/poly_l2"
    OBS_SECS    = 180
    LOCAL_DIR   = Path("/btc_local")
    OUT_FILE    = LOCAL_DIR / "ob_features_full.parquet"
    PROGRESS_FILE = LOCAL_DIR / "ob_progress.json"

    if not PMDATA_KEY:
        raise RuntimeError("PMDATA_API_KEY required — set as Modal Secret 'pmdata-api-key'")

    # ── Load markets ───────────────────────────────────────────────────────
    markets = pd.read_csv(LOCAL_DIR / "all_markets.csv")
    markets["market_id"] = markets["market_id"].astype(str)
    markets["slot_ts"]   = markets["slot_ts"].astype(int)
    log.info("Total markets: %d", len(markets))

    # Check for slug column
    if "slug" not in markets.columns:
        log.error("all_markets.csv missing 'slug' column! Columns: %s", list(markets.columns))
        raise RuntimeError("Need 'slug' column in all_markets.csv for pmdata API")

    # ── Resume progress ────────────────────────────────────────────────────
    done_ids: set = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            done_ids = set(json.load(f))
        log.info("Already done: %d, remaining: %d", len(done_ids), len(markets) - len(done_ids))

    # Load existing features
    existing_rows = []
    if OUT_FILE.exists():
        existing_df = pd.read_parquet(OUT_FILE)
        existing_rows = existing_df.to_dict("records")
        log.info("Existing OB features: %d rows", len(existing_rows))

    todo = markets[~markets["market_id"].isin(done_ids)]
    log.info("Markets to process: %d", len(todo))

    # ── OB feature extraction ──────────────────────────────────────────────
    def _compute_ob_features(book_rows: pd.DataFrame, pc_rows: pd.DataFrame,
                              slot_ts: int) -> dict | None:
        """
        Compute orderbook features from book snapshots and price_change events.

        Book snapshots → depth, spread, imbalance, drift
        Price changes  → BBO dynamics, fill flow, aggressiveness
        """
        features = {}

        # ── Book snapshot features (open vs close of observation window) ────
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
                    # Depth within 5 cents of mid — normalized by total
                    total_bid = bs.sum() + 1e-8
                    total_ask = as_.sum() + 1e-8
                    bd5 = float(bs[bp >= mid - 0.05].sum() / total_bid)
                    ad5 = float(as_[ap <= mid + 0.05].sum() / total_ask)
                    dr  = float(bd5 / (ad5 + 1e-8))
                    # Total depth
                    total_depth = float(bs.sum() + as_.sum())
                    # Weighted depth (size * price proximity to mid)
                    bid_wt = float(np.sum(bs * np.exp(-10 * np.abs(bp - mid))))
                    ask_wt = float(np.sum(as_ * np.exp(-10 * np.abs(ap - mid))))
                    w_imb  = float((bid_wt - ask_wt) / (bid_wt + ask_wt + 1e-8))
                    return {"mid": mid, "spread": spread, "imb": imb,
                            "bd5": bd5, "ad5": ad5, "dr": dr,
                            "total_depth": total_depth, "w_imb": w_imb}
                except Exception:
                    return None

            # Open = first 30s, Close = last 30s
            open_rows  = book_rows[book_rows["t_sec"] <= 30]
            close_rows = book_rows[book_rows["t_sec"] >= 150]

            open_snap  = _extract_snap(open_rows.iloc[0])  if len(open_rows)  else _extract_snap(book_rows.iloc[0])
            close_snap = _extract_snap(close_rows.iloc[-1]) if len(close_rows) else _extract_snap(book_rows.iloc[-1])

            if open_snap is None:
                return None

            features["ob_mid"]          = open_snap["mid"]
            features["ob_spread"]       = open_snap["spread"]
            features["ob_imbalance"]    = open_snap["imb"]
            features["ob_depth_ratio"]  = open_snap["dr"]
            features["ob_bid_depth_5c"] = open_snap["bd5"]
            features["ob_ask_depth_5c"] = open_snap["ad5"]
            features["ob_total_depth"]  = open_snap["total_depth"]
            features["ob_weighted_imb"] = open_snap["w_imb"]

            if close_snap is not None:
                features["ob_mid_drift"]     = float(close_snap["mid"] - open_snap["mid"])
                features["ob_imbalance_end"] = float(close_snap["imb"])
                features["ob_spread_end"]    = float(close_snap["spread"])
                features["ob_depth_change"]  = float(close_snap["total_depth"] - open_snap["total_depth"])
                # Imbalance momentum: end - start
                features["ob_imb_momentum"]  = float(close_snap["imb"] - open_snap["imb"])
            else:
                features["ob_mid_drift"]     = 0.0
                features["ob_imbalance_end"] = open_snap["imb"]
                features["ob_spread_end"]    = open_snap["spread"]
                features["ob_depth_change"]  = 0.0
                features["ob_imb_momentum"]  = 0.0

            # ── Windowed OB imbalance (3 windows: 0-60, 60-120, 120-180) ────
            for w_i, (t0, t1) in enumerate([(0, 60), (60, 120), (120, 180)]):
                w_rows = book_rows[(book_rows["t_sec"] >= t0) & (book_rows["t_sec"] < t1)]
                if len(w_rows) > 0:
                    imbs = []
                    for _, r in w_rows.iterrows():
                        s = _extract_snap(r)
                        if s:
                            imbs.append(s["imb"])
                    features[f"ob_imb_w{w_i}"] = float(np.mean(imbs)) if imbs else 0.0
                else:
                    features[f"ob_imb_w{w_i}"] = 0.0
        else:
            return None  # No book data = skip market

        # ── Price change features (BBO dynamics, fill flow) ─────────────────
        if pc_rows is not None and len(pc_rows) > 0:
            pc_rows = pc_rows.sort_values("t_sec")

            # Count price changes in each direction
            if "price" in pc_rows.columns:
                prices = pc_rows["price"].values.astype(float)
                diffs  = np.diff(prices)
                n_up   = int((diffs > 0).sum())
                n_dn   = int((diffs < 0).sum())
                total_changes = n_up + n_dn
                features["ob_pc_up_ratio"] = float(n_up / (total_changes + 1e-8))
                features["ob_pc_volatility"] = float(np.std(diffs)) if len(diffs) > 1 else 0.0
                features["ob_pc_count"] = float(total_changes)
            else:
                features["ob_pc_up_ratio"]   = 0.5
                features["ob_pc_volatility"] = 0.0
                features["ob_pc_count"]      = 0.0

            # BBO dynamics from price_change events — if side info available
            if "side" in pc_rows.columns:
                buys  = pc_rows[pc_rows["side"].str.upper() == "BUY"]
                sells = pc_rows[pc_rows["side"].str.upper() == "SELL"]
                buy_vol  = buys["size"].sum() if "size" in buys.columns else len(buys)
                sell_vol = sells["size"].sum() if "size" in sells.columns else len(sells)
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
                                window_start: float = 120.0, window_end: float = 180.0) -> dict:
        """
        Compute clob_* features matching exactly what the live trader computes
        in clob_features.py — using the last 60s of the observation window
        (t=120-180s), which corresponds to the entry window in live trading.

        Uses the same poly_l2 data (book + price_change events) but from
        historical data instead of the live WS feed.

        Mirrors live: real book snapshots for imbalance/depth,
        all snapshots (including synthetic-equivalent from price_change
        best_bid/best_ask) for mid/spread/velocity.
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

        # ── Filter to window ───────────────────────────────────────────────
        w_books = book_rows[
            (book_rows["t_sec"] >= window_start) & (book_rows["t_sec"] < window_end)
        ] if book_rows is not None and len(book_rows) > 0 else pd.DataFrame()

        w_pcs = pc_rows[
            (pc_rows["t_sec"] >= window_start) & (pc_rows["t_sec"] < window_end)
        ] if pc_rows is not None and len(pc_rows) > 0 else pd.DataFrame()

        n_real_books = len(w_books)
        n_pcs = len(w_pcs)

        # Minimum data gate — same as live (5 real events, 5s span)
        if n_real_books + n_pcs < 5:
            return zeros

        time_span = window_end - window_start  # always 60s for historical

        # ── Real book snapshots: imbalance, depth, mid, spread ─────────────
        real_imb, real_depth, real_mid, real_spread = [], [], [], []
        real_ts = []
        for _, row in w_books.iterrows():
            try:
                ap = np.array(row["ask_prices"], dtype=np.float64)
                bp = np.array(row["bid_prices"], dtype=np.float64)
                as_ = np.array(row["ask_sizes"], dtype=np.float64)
                bs = np.array(row["bid_sizes"], dtype=np.float64)
                if len(ap) == 0 or len(bp) == 0:
                    continue
                mid = float((ap[0] + bp[0]) / 2)
                spread = float(ap[0] - bp[0])
                total_sz = float(as_[0] + bs[0])
                imb = float((bs[0] - as_[0]) / total_sz) if total_sz > 0 else 0.0
                depth = float(ap.sum() + bp.sum())  # approximation — consistent with live
                real_imb.append(imb)
                real_depth.append(depth)
                real_mid.append(mid)
                real_spread.append(spread)
                real_ts.append(float(row["t_sec"]))
            except Exception:
                continue

        # ── price_change events: best_bid/best_ask for mid/spread series ───
        # Mirrors live: use best_ask/best_bid from each price_change as
        # synthetic book snapshot for mid/spread/velocity computation.
        all_mid, all_spread, all_ts = list(real_mid), list(real_spread), list(real_ts)
        ask_prices_seq = []  # for ask_pressure

        for _, row in w_pcs.iterrows():
            try:
                ba = float(row["best_ask"]) if pd.notna(row.get("best_ask")) else None
                bb = float(row["best_bid"]) if pd.notna(row.get("best_bid")) else None
                if ba is not None and bb is not None and ba > bb > 0:
                    all_mid.append((ba + bb) / 2)
                    all_spread.append(ba - bb)
                    all_ts.append(float(row["t_sec"]))
                # ask_pressure: track SELL-side (ask) price moves
                side = str(row.get("pc_side", "")).upper()
                pc_price = row.get("pc_price")
                if side == "SELL" and pd.notna(pc_price):
                    ask_prices_seq.append(float(pc_price))
            except Exception:
                continue

        if len(all_mid) < 2:
            return zeros

        # Sort by time
        order = np.argsort(all_ts)
        all_ts_arr = np.array(all_ts)[order]
        all_mid_arr = np.array(all_mid)[order]
        all_spread_arr = np.array(all_spread)[order]
        t_rel = all_ts_arr - all_ts_arr[0]

        # ── Compute features ───────────────────────────────────────────────
        feats = {}

        # Imbalance — real books only
        feats["clob_imb_mean"]  = float(np.mean(real_imb))   if real_imb else 0.0
        feats["clob_imb_std"]   = float(np.std(real_imb))    if len(real_imb) > 1 else 0.0
        feats["clob_imb_drift"] = float(real_imb[-1] - real_imb[0]) if len(real_imb) > 1 else 0.0

        # Spread — all (synthetic has real best_ask/best_bid)
        feats["clob_spread_mean"]  = float(np.mean(all_spread_arr))
        feats["clob_spread_trend"] = _linslope(t_rel, all_spread_arr)

        # Mid — all
        feats["clob_mid_velocity"]   = _linslope(t_rel, all_mid_arr)
        mid_diffs = np.diff(all_mid_arr)
        feats["clob_mid_volatility"] = float(np.std(mid_diffs)) if len(mid_diffs) > 0 else 0.0

        # Activity rate — real books + price_changes (no double-count)
        feats["clob_activity_rate"] = float((n_real_books + n_pcs) / time_span)

        # Depth trend — real books only
        if len(real_depth) > 1:
            real_ts_arr = np.array(real_ts)
            t_real_rel = real_ts_arr - real_ts_arr[0]
            feats["clob_depth_trend"] = _linslope(t_real_rel, np.array(real_depth))
        else:
            feats["clob_depth_trend"] = 0.0

        # Ask pressure — fraction of consecutive ASK moves that went DOWN
        if len(ask_prices_seq) >= 2:
            diffs = np.diff(ask_prices_seq)
            moves = diffs[diffs != 0]
            feats["clob_ask_pressure"] = float((moves < 0).sum() / len(moves)) if len(moves) > 0 else 0.0
        else:
            feats["clob_ask_pressure"] = 0.0

        return feats

    def _fetch_one(market_id: str, slug: str, slot_ts: int) -> tuple[str, dict | None]:
        """Fetch poly_l2 parquet and extract OB + price_change features."""
        try:
            r = requests.get(f"{BASE_URL}/{slug}", headers=HEADERS, timeout=15)
            if not r.ok:
                return (market_id, None)
            data = r.json()
            dl_url = data.get("download_url")
            if not dl_url:
                return (market_id, None)

            r2 = requests.get(dl_url, timeout=120)
            if not r2.ok:
                return (market_id, None)

            df = pd.read_parquet(io.BytesIO(r2.content))

            # Convert timestamp — poly_l2 uses datetime64[ms], convert to Unix seconds
            if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["ts_sec"] = df["timestamp"].astype("int64") / 1000.0  # ms → seconds
            else:
                df["ts_sec"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)
            df["t_sec"] = df["ts_sec"] - slot_ts

            # Filter to observation window [0, 180)
            df = df[(df["t_sec"] >= 0) & (df["t_sec"] < OBS_SECS)]

            books = df[df["event_type"] == "book"].copy()
            pcs   = df[df["event_type"] == "price_change"].copy()

            feats = _compute_ob_features(
                books if len(books) > 0 else None,
                pcs if len(pcs) > 0 else None,
                slot_ts
            )
            if feats is not None:
                # Compute clob_* features from the same data (last 60s window)
                # to match exactly what the live trader sees at entry time (t=120-180s)
                clob_feats = _compute_clob_features(
                    books if len(books) > 0 else pd.DataFrame(),
                    pcs   if len(pcs)   > 0 else pd.DataFrame(),
                    window_start=120.0, window_end=180.0,
                )
                feats.update(clob_feats)
                feats["market_id"] = market_id
            return (market_id, feats)
        except Exception as e:
            return (market_id, None)

    # ── Parallel fetch ─────────────────────────────────────────────────────
    new_rows = []
    success = 0
    failed  = 0
    BATCH_SIZE = 200
    WORKERS    = 20

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

        time.sleep(0.05)  # tiny pause between batches

    LOCAL_VOL.commit()
    log.info("DONE. OB features: %d markets (failed: %d)", len(existing_rows), failed)


@app.local_entrypoint()
def main():
    fetch_ob_features.remote()
