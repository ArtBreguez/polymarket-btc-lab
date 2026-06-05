"""
train_v20_modal.py — BTC 5min model v20
========================================
Key changes vs v19:
  - DATASET EXPANSION: fetch new ticks from pmdata.dev for markets after Jun 3 2026
    and gap-fill Feb 15 – Mar 13 2026 via pmdata poly_l2 API
  - Inline OB feature extraction for new markets (same as fetch_ob_features_modal.py)
  - NEW FEATURES: btc_vol_regime, btc_vol_accel, ob_depth_change, btc_funding_proxy
  - TOP_N_FEATS = 45 (was 40)
  - Gate: vs current champion (v19, AUC=0.9000)

Data sources (Modal Volume 'btc-local-data'):
  /ticks_btc_full_clean.parquet  — 22,237+ markets, 68.3M+ clean ticks
  /all_markets.csv               — 22,319+ markets unified timeline
  /binance_spot_full.parquet     — 119k+ 1m BTCUSDT candles
  /ob_features_full.parquet      — L2 OB features (pre-computed + new)
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
        "requests>=2.31",
    )
)

app = modal.App("btc-v20-run", image=image)


@app.function(
    cpu=8,
    memory=32768,
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("pmdata-api-key")],
    volumes={"/btc_local": LOCAL_VOL},
)
def train_v20():
    import gc, json, logging, math, os, pickle, sys, time, warnings
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np
    import optuna
    import pandas as pd
    import pyarrow.parquet as pq
    import requests
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
    PMDATA_API_KEY = os.environ.get("PMDATA_API_KEY", "")
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
    OBS_SECS      = 180
    SLOT_DURATION = 300
    OPTUNA_TRIALS = 150
    WF_GAP        = 5
    N_SPLITS      = 5
    TOP_N_FEATS   = 45  # expanded from 40
    LOCAL_DIR     = Path("/btc_local")

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")
    if not PMDATA_API_KEY:
        raise RuntimeError("PMDATA_API_KEY required")

    # ── Step 0: Dataset expansion — fetch new markets from pmdata ─────────
    log.info("Step 0: Dataset expansion from pmdata.dev...")

    progress_path = LOCAL_DIR / "v20_progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
    else:
        progress = {"fetched_slots": [], "phase": "start"}

    fetched_set = set(progress.get("fetched_slots", []))

    # Load existing markets to find what slots we already have
    existing_markets = pd.read_csv(LOCAL_DIR / "all_markets.csv")
    existing_markets["market_id"] = existing_markets["market_id"].astype(str)
    existing_slots = set(existing_markets["slot_ts"].astype(int).tolist())
    log.info("Existing markets: %d, existing slots: %d", len(existing_markets), len(existing_slots))

    # Define slot ranges to fetch:
    # 1) Gap fill: Feb 15 - Mar 13 2026 (slot_ts 1771113600 to 1773398700)
    # 2) New markets: after Jun 3 2026 (slot_ts > 1780502400)
    GAP_START = 1771113600   # Feb 15, 2026
    GAP_END   = 1773398700   # Mar 13, 2026
    NEW_START = 1780502400   # Jun 3, 2026

    # Generate candidate slot timestamps (every 300s)
    gap_slots = list(range(GAP_START, GAP_END + 1, SLOT_DURATION))
    # For new markets, scan forward from NEW_START up to current time
    now_ts = int(time.time())
    new_slots = list(range(NEW_START, now_ts, SLOT_DURATION))

    # Filter out already-existing and already-fetched slots
    candidate_slots = [s for s in gap_slots + new_slots
                       if s not in existing_slots and s not in fetched_set]
    log.info("Candidate new slots to fetch: %d (gap: %d, new: %d)",
             len(candidate_slots),
             len([s for s in gap_slots if s not in existing_slots and s not in fetched_set]),
             len([s for s in new_slots if s not in existing_slots and s not in fetched_set]))

    def fetch_one_slot(slot_ts):
        """Fetch pmdata poly_l2 data for a single slot. Returns (ticks_df, ob_feats_dict, market_row) or None."""
        try:
            # Get download URL
            resp = requests.get(
                f"https://api.pmdata.dev/get-download-url/poly_l2/btc-updown-5m-{slot_ts}",
                headers={"api_key": PMDATA_API_KEY},
                timeout=30
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            download_url = data.get("download_url")
            if not download_url:
                return None

            # Download parquet
            pq_resp = requests.get(download_url, timeout=60)
            if pq_resp.status_code != 200:
                return None

            import io
            raw_df = pd.read_parquet(io.BytesIO(pq_resp.content))
            if raw_df.empty:
                return None

            # --- Extract target from market_resolved events ---
            resolved = raw_df[raw_df["event_type"] == "market_resolved"] if "event_type" in raw_df.columns else pd.DataFrame()
            target = None
            if len(resolved) > 0:
                winning = resolved.iloc[0].get("winning_outcome", None)
                if winning is not None:
                    winning_str = str(winning).lower().strip()
                    if winning_str == "yes":
                        target = 1  # UP won
                    elif winning_str == "no":
                        target = 0  # DOWN won
            if target is None:
                return None  # Skip unresolved markets

            # --- Extract ticks from last_trade_price events ---
            trades = raw_df[raw_df["event_type"] == "last_trade_price"].copy() if "event_type" in raw_df.columns else pd.DataFrame()
            ticks_list = []
            if len(trades) > 0:
                for _, tr in trades.iterrows():
                    ts_ms = int(tr.get("timestamp", 0))
                    if isinstance(tr.get("timestamp"), str):
                        try:
                            ts_ms = int(float(tr["timestamp"]) * 1000)
                        except:
                            continue
                    # Filter to observation window [slot_ts*1000, (slot_ts+180)*1000)
                    if ts_ms < slot_ts * 1000 or ts_ms >= (slot_ts + 180) * 1000:
                        continue
                    trade_price = float(tr.get("price", tr.get("trade_price", 0.5)))
                    trade_size = float(tr.get("size", tr.get("trade_size", 0)))
                    trade_side = str(tr.get("side", tr.get("trade_side", "BUY"))).upper()
                    ticks_list.append({
                        "market_id": str(slot_ts),
                        "timestamp_ms": ts_ms,
                        "outcome": "Up" if trade_price >= 0.5 else "Down",
                        "side": trade_side,
                        "price": trade_price,
                        "size_usdc": trade_price * trade_size,
                    })

            ticks_df = pd.DataFrame(ticks_list) if ticks_list else pd.DataFrame(
                columns=["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"])

            # --- Extract OB features from book events ---
            ob_feats = _extract_ob_features(raw_df, slot_ts)

            market_row = {
                "market_id": str(slot_ts),
                "slot_ts": slot_ts,
                "target": target,
            }
            return ticks_df, ob_feats, market_row

        except Exception as e:
            return None

    def _extract_ob_features(raw_df, slot_ts):
        """Extract OB features from pmdata parquet (same logic as fetch_ob_features_modal.py)."""
        feats = {"market_id": str(slot_ts)}

        # Book events
        books = raw_df[raw_df["event_type"] == "book"].copy() if "event_type" in raw_df.columns else pd.DataFrame()

        if len(books) > 0:
            # Parse timestamps
            if "timestamp" in books.columns:
                books["ts_ms"] = pd.to_numeric(books["timestamp"], errors="coerce")
            books = books.sort_values("ts_ms") if "ts_ms" in books.columns else books

            first_book = books.iloc[0]
            last_book = books.iloc[-1]

            # Basic OB features from first snapshot
            bid_depth = float(first_book.get("bid_depth", first_book.get("total_bid_depth", 0)))
            ask_depth = float(first_book.get("ask_depth", first_book.get("total_ask_depth", 0)))
            best_bid = float(first_book.get("best_bid", first_book.get("best_bid_price", 0)))
            best_ask = float(first_book.get("best_ask", first_book.get("best_ask_price", 0)))

            total_depth = bid_depth + ask_depth
            feats["ob_imbalance"] = (bid_depth - ask_depth) / (total_depth + 1e-9) if total_depth > 0 else 0.0
            feats["ob_spread"] = best_ask - best_bid
            feats["ob_mid"] = (best_ask + best_bid) / 2 if (best_ask + best_bid) > 0 else 0.5
            feats["ob_depth_ratio"] = bid_depth / (ask_depth + 1e-9) if ask_depth > 0 else 1.0

            # Depth within 5 cents of mid
            mid = feats["ob_mid"]
            feats["ob_bid_depth_5c"] = float(first_book.get("bid_depth_5c", bid_depth * 0.3))
            feats["ob_ask_depth_5c"] = float(first_book.get("ask_depth_5c", ask_depth * 0.3))
            feats["ob_total_depth"] = total_depth

            # Weighted imbalance
            feats["ob_weighted_imb"] = float(first_book.get("weighted_imbalance",
                feats["ob_imbalance"]))

            # Mid drift (first to last)
            last_best_bid = float(last_book.get("best_bid", last_book.get("best_bid_price", best_bid)))
            last_best_ask = float(last_book.get("best_ask", last_book.get("best_ask_price", best_ask)))
            last_mid = (last_best_ask + last_best_bid) / 2 if (last_best_ask + last_best_bid) > 0 else mid
            feats["ob_mid_drift"] = last_mid - mid

            # Imbalance momentum (first to last)
            last_bid_depth = float(last_book.get("bid_depth", last_book.get("total_bid_depth", bid_depth)))
            last_ask_depth = float(last_book.get("ask_depth", last_book.get("total_ask_depth", ask_depth)))
            last_total = last_bid_depth + last_ask_depth
            last_imb = (last_bid_depth - last_ask_depth) / (last_total + 1e-9) if last_total > 0 else 0.0
            feats["ob_imb_momentum"] = last_imb - feats["ob_imbalance"]

            # End-of-window values for depth change
            feats["ob_total_depth_end"] = last_total
            feats["ob_imbalance_end"] = last_imb

            # Temporal OB imbalance windows (0-60s, 60-120s, 120-180s)
            if "ts_ms" in books.columns:
                t0 = slot_ts * 1000
                for wi, (ws, we) in enumerate([(0, 60000), (60000, 120000), (120000, 180000)]):
                    window_books = books[(books["ts_ms"] >= t0 + ws) & (books["ts_ms"] < t0 + we)]
                    if len(window_books) > 0:
                        wb = window_books.iloc[len(window_books)//2]  # middle snapshot
                        wbd = float(wb.get("bid_depth", wb.get("total_bid_depth", 0)))
                        wad = float(wb.get("ask_depth", wb.get("total_ask_depth", 0)))
                        wt = wbd + wad
                        feats[f"ob_imb_w{wi}"] = (wbd - wad) / (wt + 1e-9) if wt > 0 else 0.0
                    else:
                        feats[f"ob_imb_w{wi}"] = 0.0
            else:
                feats["ob_imb_w0"] = feats["ob_imbalance"]
                feats["ob_imb_w1"] = feats["ob_imbalance"]
                feats["ob_imb_w2"] = last_imb
        else:
            # No book data — neutral defaults
            feats.update({
                "ob_imbalance": 0.0, "ob_spread": 0.02, "ob_mid": 0.5,
                "ob_depth_ratio": 1.0, "ob_bid_depth_5c": 0.5, "ob_ask_depth_5c": 0.5,
                "ob_total_depth": 1000.0, "ob_weighted_imb": 0.0, "ob_mid_drift": 0.0,
                "ob_imb_momentum": 0.0, "ob_total_depth_end": 1000.0,
                "ob_imbalance_end": 0.0,
                "ob_imb_w0": 0.0, "ob_imb_w1": 0.0, "ob_imb_w2": 0.0,
            })

        # Trade events → fill imbalance
        trades = raw_df[raw_df["event_type"] == "last_trade_price"] if "event_type" in raw_df.columns else pd.DataFrame()
        if len(trades) > 0:
            buy_fills = len(trades[trades.get("side", trades.get("trade_side", pd.Series())).astype(str).str.upper() == "BUY"])
            total_fills = len(trades)
            feats["ob_fill_imbalance"] = buy_fills / total_fills if total_fills > 0 else 0.5
        else:
            feats["ob_fill_imbalance"] = 0.5

        # Price change events → pc_up_ratio
        pc_events = raw_df[raw_df["event_type"] == "price_change"] if "event_type" in raw_df.columns else pd.DataFrame()
        if len(pc_events) > 0:
            if "side" in pc_events.columns:
                buy_pc = len(pc_events[pc_events["side"].astype(str).str.upper() == "BUY"])
            elif "trade_side" in pc_events.columns:
                buy_pc = len(pc_events[pc_events["trade_side"].astype(str).str.upper() == "BUY"])
            else:
                buy_pc = len(pc_events) // 2
            feats["ob_pc_up_ratio"] = buy_pc / len(pc_events) if len(pc_events) > 0 else 0.5
        else:
            feats["ob_pc_up_ratio"] = 0.5

        return feats

    # Parallel fetch with ThreadPoolExecutor
    new_ticks_all = []
    new_ob_feats_all = []
    new_markets_all = []
    batch_count = 0
    BATCH_SIZE = 200

    if candidate_slots:
        log.info("Fetching %d candidate slots from pmdata (parallel, batch=%d)...",
                 len(candidate_slots), BATCH_SIZE)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_one_slot, s): s for s in candidate_slots}
            done_count = 0

            for future in as_completed(futures):
                slot = futures[future]
                done_count += 1
                result = future.result()

                if result is not None:
                    ticks_df, ob_feats, market_row = result
                    if len(ticks_df) > 0:
                        new_ticks_all.append(ticks_df)
                    new_ob_feats_all.append(ob_feats)
                    new_markets_all.append(market_row)
                    fetched_set.add(slot)

                # Batch flush every BATCH_SIZE markets
                if len(new_markets_all) > 0 and len(new_markets_all) % BATCH_SIZE == 0:
                    batch_count += 1
                    log.info("Batch %d: flushing %d new markets (%d/%d done)...",
                             batch_count, BATCH_SIZE, done_count, len(candidate_slots))

                    # Save progress
                    progress["fetched_slots"] = list(fetched_set)
                    progress["phase"] = f"batch_{batch_count}"
                    with open(progress_path, "w") as f:
                        json.dump(progress, f)

                if done_count % 500 == 0:
                    log.info("  Progress: %d/%d slots checked", done_count, len(candidate_slots))

        log.info("Fetch complete: %d new markets found from %d candidates",
                 len(new_markets_all), len(candidate_slots))

    # Append new data to existing files on Volume
    if new_markets_all:
        log.info("Appending %d new markets to Volume files...", len(new_markets_all))

        # Append markets
        new_markets_df = pd.DataFrame(new_markets_all)
        combined_markets = pd.concat([existing_markets, new_markets_df], ignore_index=True)
        combined_markets = combined_markets.drop_duplicates(subset="market_id", keep="first")
        combined_markets.to_csv(LOCAL_DIR / "all_markets.csv", index=False)

        # Append ticks
        if new_ticks_all:
            new_ticks_df = pd.concat(new_ticks_all, ignore_index=True)
            log.info("New ticks: %d rows for %d markets",
                     len(new_ticks_df), new_ticks_df["market_id"].nunique())
            # Append to existing parquet
            existing_ticks = pd.read_parquet(str(LOCAL_DIR / "ticks_btc_full_clean.parquet"))
            combined_ticks = pd.concat([existing_ticks, new_ticks_df], ignore_index=True)
            combined_ticks = combined_ticks.drop_duplicates(subset=["market_id", "timestamp_ms"], keep="first")
            combined_ticks.to_parquet(str(LOCAL_DIR / "ticks_btc_full_clean.parquet"), index=False)
            del existing_ticks, combined_ticks
            gc.collect()

        # Append OB features
        if new_ob_feats_all:
            new_ob_df = pd.DataFrame(new_ob_feats_all)
            ob_path = LOCAL_DIR / "ob_features_full.parquet"
            if ob_path.exists():
                existing_ob = pd.read_parquet(str(ob_path))
                combined_ob = pd.concat([existing_ob, new_ob_df], ignore_index=True)
                combined_ob = combined_ob.drop_duplicates(subset="market_id", keep="first")
                combined_ob.to_parquet(str(ob_path), index=False)
                del existing_ob
            else:
                new_ob_df.to_parquet(str(ob_path), index=False)

        # Save final progress
        progress["fetched_slots"] = list(fetched_set)
        progress["phase"] = "complete"
        with open(progress_path, "w") as f:
            json.dump(progress, f)

        # Commit volume changes
        LOCAL_VOL.commit()
        log.info("Volume committed with new data")
    else:
        log.info("No new markets found, continuing with existing data")

    # ── Step 1: Champion metrics ──────────────────────────────────────────
    log.info("Step 1: Loading champion metrics from HF...")
    champion = {"version": "v19", "wf_auc": 0.9000, "wf_brier": 0.1291, "wf_acc": 0.8127}
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
        log.warning("Could not load champion meta: %s — using v19 defaults", e)

    # ── Step 2: Load unified markets ──────────────────────────────────────
    log.info("Step 2: Loading all_markets.csv...")
    markets = pd.read_csv(LOCAL_DIR / "all_markets.csv")
    markets["market_id"] = markets["market_id"].astype(str)
    markets["slot_ts"]   = markets["slot_ts"].astype(int)
    markets = markets.sort_values("slot_ts").reset_index(drop=True)
    log.info("Markets: %d (%.1f%% UP)", len(markets), 100 * markets["target"].mean())

    markets["rank"] = range(len(markets))
    slot_to_rank    = dict(zip(markets["slot_ts"], markets["rank"]))
    all_targets     = markets["target"].values
    all_slot_ts     = markets["slot_ts"].values
    all_mids        = markets["market_id"].values

    # ── Step 3: Load OB features ──────────────────────────────────────────
    log.info("Step 3: Loading OB features from ob_features_full.parquet...")
    ob_path = LOCAL_DIR / "ob_features_full.parquet"
    if not ob_path.exists():
        raise RuntimeError("ob_features_full.parquet not found! Run fetch_ob_features_modal.py first")

    ob_df = pd.read_parquet(str(ob_path))
    ob_df["market_id"] = ob_df["market_id"].astype(str)
    ob_by_market = ob_df.set_index("market_id").to_dict("index")
    ob_cols = [c for c in ob_df.columns if c != "market_id"]
    log.info("OB features loaded: %d markets, %d features (%s)",
             len(ob_df), len(ob_cols), ob_cols[:5])

    # Coverage check
    ob_market_ids = set(ob_df["market_id"])
    all_market_ids = set(markets["market_id"])
    coverage = len(ob_market_ids & all_market_ids) / len(all_market_ids) * 100
    log.info("OB coverage: %.1f%% (%d/%d markets)",
             coverage, len(ob_market_ids & all_market_ids), len(all_market_ids))

    # ── Step 4: Binance spot ──────────────────────────────────────────────
    log.info("Step 4: Loading Binance spot from Volume...")
    spot_path = LOCAL_DIR / "binance_spot_full.parquet"
    if not spot_path.exists():
        spot_path = LOCAL_DIR / "binance_spot_local.parquet"
    spot_df = pd.read_parquet(str(spot_path))
    spot_df = spot_df.sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
    spot_ts_arr = (spot_df["timestamp_ms"].values // 1000).astype(np.int64)
    spot_px_arr = spot_df["close"].values.astype(np.float64)
    log.info("Binance spot: %d candles (%.0f days)",
             len(spot_ts_arr), (spot_ts_arr[-1] - spot_ts_arr[0]) / 86400)

    # ── Step 5: Load ticks ────────────────────────────────────────────────
    log.info("Step 5: Loading ticks from ticks_btc_full_clean.parquet...")
    all_mids_set = set(markets["market_id"].tolist())
    tick_cols    = ["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"]
    pf           = pq.ParquetFile(str(LOCAL_DIR / "ticks_btc_full_clean.parquet"))

    chunks = []
    for rg_i in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg_i, columns=tick_cols).to_pandas()
        chunk["market_id"] = chunk["market_id"].astype(str)
        chunk = chunk[chunk["market_id"].isin(all_mids_set)]
        if len(chunk):
            chunks.append(chunk)
    gc.collect()

    btc = pd.concat(chunks, ignore_index=True)
    btc = btc[btc["timestamp_ms"] > 0]
    log.info("Ticks loaded: %d rows for %d markets",
             len(btc), btc["market_id"].nunique())

    slot_ts_map        = dict(zip(markets["market_id"], markets["slot_ts"]))
    btc["slot_ts_val"] = btc["market_id"].map(slot_ts_map).astype(float)
    btc["t_sec"]       = btc["timestamp_ms"].astype(float) / 1000 - btc["slot_ts_val"]
    btc                = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < OBS_SECS)]
    log.info("Ticks in [0, 180s): %d", len(btc))

    btc_up       = btc[btc["outcome"] == "Up"]
    btc_dn       = btc[btc["outcome"] == "Down"]
    slot_vol_up  = btc_up.groupby("market_id")["size_usdc"].sum()
    slot_vol_dn  = btc_dn.groupby("market_id")["size_usdc"].sum()
    slot_vol_tot = slot_vol_up.add(slot_vol_dn, fill_value=0)
    slot_up_ratio = (slot_vol_up / slot_vol_tot.clip(lower=1e-9))
    slot_nticks   = btc.groupby("market_id").size()
    log.info("Per-slot aggregates computed")

    # ── Step 6: Feature engineering ───────────────────────────────────────
    log.info("Step 6: Building features for %d markets...", len(markets))

    def _ur(df_sub):
        up  = df_sub[df_sub["outcome"] == "Up"]["size_usdc"].sum()
        dn  = df_sub[df_sub["outcome"] == "Down"]["size_usdc"].sum()
        tot = up + dn
        return up / tot if tot > 0 else 0.5

    def _ur_w(df_sub, t0, t1):
        w = df_sub[(df_sub["t_sec"] >= t0) & (df_sub["t_sec"] < t1)]
        return _ur(w) if len(w) else 0.5

    def spot_at(ts_s):
        idx = np.searchsorted(spot_ts_arr, ts_s, side="right") - 1
        idx = max(0, min(idx, len(spot_px_arr) - 1))
        return float(spot_px_arr[idx])

    def _realized_vol(ts_end, minutes=60):
        """Compute realized volatility = std(1m returns) * sqrt(minutes) over last N minutes."""
        idx_end = int(np.searchsorted(spot_ts_arr, ts_end, side="right"))
        idx_start = max(0, idx_end - minutes)
        if idx_end - idx_start < 5:
            return 0.0
        px_slice = spot_px_arr[idx_start:idx_end]
        if len(px_slice) < 2:
            return 0.0
        returns = np.diff(px_slice) / (px_slice[:-1] + 1e-12)
        return float(np.std(returns) * np.sqrt(len(returns)))

    btc_grouped = {mid: grp for mid, grp in btc.groupby("market_id")}
    rows = []
    skipped_no_ob = 0

    for rank_i, row in markets.iterrows():
        mid     = row["market_id"]
        slot_ts = int(row["slot_ts"])
        target  = int(row["target"])

        grp = btc_grouped.get(mid)
        n   = len(grp) if grp is not None else 0

        # ── CLOB flow features (same as v19) ───────────────────────────────
        if n > 0:
            ur  = slot_up_ratio.get(mid, 0.5)
            vt  = slot_vol_tot.get(mid, 0.0)
            ntx = slot_nticks.get(mid, 0)

            up_vals = grp[grp["outcome"] == "Up"]["size_usdc"].values
            dn_vals = grp[grp["outcome"] == "Down"]["size_usdc"].values

            w0 = _ur_w(grp, 0, 30);    w1 = _ur_w(grp, 30, 60)
            w2 = _ur_w(grp, 60, 90);   w3 = _ur_w(grp, 90, 120)
            w4 = _ur_w(grp, 120, 150); w5 = _ur_w(grp, 150, 180)

            up_g   = grp[grp["outcome"] == "Up"]
            dn_g   = grp[grp["outcome"] == "Down"]
            def vwap(g):
                return (g["price"] * g["size_usdc"]).sum() / g["size_usdc"].sum() if len(g) else 0.5
            vwap_up = vwap(up_g); vwap_dn = vwap(dn_g)

            all_sorted = grp.sort_values("t_sec")
            if len(all_sorted) > 1:
                w_exp = np.exp(-0.02 * (OBS_SECS - all_sorted["t_sec"].values))
                ur_up = (all_sorted["outcome"] == "Up").astype(float).values
                tw_ur = np.average(ur_up * all_sorted["size_usdc"].values,
                                   weights=w_exp) / (np.average(all_sorted["size_usdc"].values, weights=w_exp) + 1e-9)
            else:
                tw_ur = ur

            buy_sz    = grp[grp["side"] == "BUY"]["size_usdc"].sum()
            buy_ratio = buy_sz / vt if vt > 0 else 0.5
            momentum  = (w3 + w4 + w5) / 3 - (w0 + w1 + w2) / 3
            stability = np.std([w0, w1, w2, w3, w4, w5])
            avg_up    = up_vals.mean() if len(up_vals) else 0
            avg_dn    = dn_vals.mean() if len(dn_vals) else 0

            feat = {
                "btc_up_ratio":          ur,
                "btc_n_ticks":           float(n),
                "btc_buy_ratio":         buy_ratio,
                "btc_tw_up_ratio":       tw_ur,
                "btc_momentum":          momentum,
                "btc_vwap_spread":       vwap_up - vwap_dn,
                "btc_vwap_up":           vwap_up,
                "btc_vwap_dn":           vwap_dn,
                "btc_vwap_trend":        vwap_up - 0.5,
                "btc_up_w0": w0, "btc_up_w1": w1, "btc_up_w2": w2,
                "btc_up_w3": w3, "btc_up_w4": w4, "btc_up_w5": w5,
                "btc_size_disparity":    avg_up - avg_dn,
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
            feat["btc_up_ratio"]    = 0.5
            feat["btc_vwap_up"]     = 0.5
            feat["btc_vwap_dn"]     = 0.5
            feat["btc_tw_up_ratio"] = 0.5
            feat["btc_buy_ratio"]   = 0.5

        # ── Z-scores (cross-slot context) ─────────────────────────────────
        ext_rank = slot_to_rank.get(slot_ts, rank_i)

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
            mu20 = np.mean(hist_vals); sd20 = np.std(hist_vals) + 1e-6
            feat["btc_up_ratio_zscore_20s"] = (feat["btc_up_ratio"] - mu20) / sd20
            feat["btc_up_w5_zscore"]        = (feat["btc_up_w5"]    - mu20) / sd20
        else:
            feat["btc_up_ratio_zscore_20s"] = 0.0
            feat["btc_up_w5_zscore"]        = 0.0

        hist5 = hist_vals[:5]
        if len(hist5) >= 2:
            mu5 = np.mean(hist5); sd5 = np.std(hist5) + 1e-6
            feat["btc_up_ratio_zscore_5s"] = (feat["btc_up_ratio"] - mu5) / sd5
        else:
            feat["btc_up_ratio_zscore_5s"] = 0.0

        # ── Spot features (same as v19) ───────────────────────────────────
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
        if px_now > 0 and px_1h_ago > 0 and px_4h_ago > 0 and abs(px_now - px_4h_ago) > 1:
            feat["btc_pre_1h_4h_ratio"] = (px_now - px_1h_ago) / (px_now - px_4h_ago + 1e-9)
        else:
            feat["btc_pre_1h_4h_ratio"] = 0.0

        t0_idx = int(np.searchsorted(spot_ts_arr, slot_ts, side="left"))
        t1_idx = int(np.searchsorted(spot_ts_arr, slot_ts + OBS_SECS, side="right"))
        if t1_idx > t0_idx:
            inslot_px = spot_px_arr[t0_idx:t1_idx]
            feat["btc_inslot_ret"] = float(inslot_px[-1] / inslot_px[0] - 1) if inslot_px[0] > 0 else 0.0
        else:
            feat["btc_inslot_ret"] = 0.0

        px_k = px_now / 1000
        feat["btc_dist_1k"] = min(px_k - math.floor(px_k), math.ceil(px_k) - px_k)

        # ── NEW v20: Volatility & funding features ─────────────────────────
        vol_1h = _realized_vol(slot_ts, minutes=60)
        vol_30m = _realized_vol(slot_ts, minutes=30)

        # btc_vol_regime: bucket realized 1h vol
        if vol_1h < 0.005:
            feat["btc_vol_regime"] = 0.0  # low
        elif vol_1h < 0.015:
            feat["btc_vol_regime"] = 1.0  # med
        else:
            feat["btc_vol_regime"] = 2.0  # high

        # btc_vol_accel: acceleration of vol (30m vs 1h)
        feat["btc_vol_accel"] = (vol_30m / (vol_1h + 1e-9)) - 1.0

        # btc_funding_proxy: annualized 4h momentum as funding proxy
        feat["btc_funding_proxy"] = feat["btc_pre_4h_ret"] * 6

        # ── Lag features (same as v19) ────────────────────────────────────
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

        # ── Temporal features (same as v19) ───────────────────────────────
        dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        dow  = dt.weekday()

        feat["hour_sin"]       = math.sin(2 * math.pi * hour / 24)
        feat["hour_cos"]       = math.cos(2 * math.pi * hour / 24)
        feat["dow_sin"]        = math.sin(2 * math.pi * dow / 7)
        feat["dow_cos"]        = math.cos(2 * math.pi * dow / 7)
        feat["hour_x_up_ratio"] = feat["btc_up_ratio"] * (hour / 24.0)
        feat["hour_x_tw_ur"]    = feat["btc_tw_up_ratio"] * (hour / 24.0)

        # ── L2 Orderbook features ────────────────────────────────────────
        ob = ob_by_market.get(mid)
        if ob is not None:
            for col in ob_cols:
                feat[f"ob_{col}" if not col.startswith("ob_") else col] = float(ob.get(col, 0.0))

            # ── Cross-domain interactions: OB × CLOB ──────────────────────
            feat["x_imb_x_ur"] = float(ob.get("ob_imbalance", 0)) * feat["btc_up_ratio"]
            feat["x_depth_x_momentum"] = float(ob.get("ob_depth_ratio", 1)) * feat["btc_momentum"]
            feat["x_spread_x_vol"] = float(ob.get("ob_spread", 0)) * feat["btc_n_ticks"]
            feat["x_ob_drift_x_inslot"] = float(ob.get("ob_mid_drift", 0)) * feat["btc_inslot_ret"]
            feat["x_fill_imb_x_buy"] = float(ob.get("ob_fill_imbalance", 0)) * feat["btc_buy_ratio"]

            # ── NEW v20: ob_depth_change ──────────────────────────────────
            ob_start_depth = float(ob.get("ob_total_depth", 0))
            ob_end_depth = float(ob.get("ob_total_depth_end", 0))
            if ob_start_depth > 0 and ob_end_depth > 0:
                feat["ob_depth_change"] = (ob_end_depth - ob_start_depth) / (ob_start_depth + 1e-9)
            else:
                feat["ob_depth_change"] = 0.0
        else:
            # No OB data — fill with neutral defaults
            skipped_no_ob += 1
            for col in ob_cols:
                key = f"ob_{col}" if not col.startswith("ob_") else col
                if "ratio" in col or "imbalance" in col or "imb" in col:
                    feat[key] = 0.0
                elif "spread" in col:
                    feat[key] = 0.02  # typical spread
                elif "depth" in col and "5c" in col:
                    feat[key] = 0.5
                elif "mid" in col and "drift" not in col:
                    feat[key] = 0.5  # mid price
                else:
                    feat[key] = 0.0

            feat["x_imb_x_ur"]          = 0.0
            feat["x_depth_x_momentum"]  = 0.0
            feat["x_spread_x_vol"]      = 0.0
            feat["x_ob_drift_x_inslot"] = 0.0
            feat["x_fill_imb_x_buy"]    = 0.0
            feat["ob_depth_change"]     = 0.0

        feat["target"] = target
        rows.append(feat)

    df = pd.DataFrame(rows)
    log.info("Feature matrix: %d rows × %d cols (skipped_no_ob=%d)",
             len(df), len(df.columns), skipped_no_ob)

    # ── Step 7: Feature selection (expanded to top 45) ─────────────────────
    FEATURE_COLS = [c for c in df.columns if c != "target"]
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["target"].values.astype(int)
    log.info("Class balance: %d UP, %d DOWN (%.1f%% UP)",
             y.sum(), (y == 0).sum(), 100 * y.mean())

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

    feat_rank    = np.argsort(feat_importances)[::-1]
    top_features = [FEATURE_COLS[i] for i in feat_rank[:TOP_N_FEATS]]
    log.info("Top %d features: %s", TOP_N_FEATS, top_features[:15])

    # Log how many OB features made the cut
    ob_in_top = [f for f in top_features if f.startswith("ob_") or f.startswith("x_")]
    log.info("OB/interaction features in top %d: %d → %s", TOP_N_FEATS, len(ob_in_top), ob_in_top)

    # Log new v20 features status
    v20_new = [f for f in top_features if f in ("btc_vol_regime", "btc_vol_accel", "ob_depth_change", "btc_funding_proxy")]
    log.info("v20 new features in top %d: %s", TOP_N_FEATS, v20_new)

    X_sel = df[top_features].values.astype(np.float32)

    # ── Step 8: Optuna tuning ─────────────────────────────────────────────
    log.info("Step 8: Optuna tuning (%d trials)...", OPTUNA_TRIALS)

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
    log.info("Best trial AUC=%.4f params=%s", study.best_value, best_params)

    # ── Step 9: Walk-forward evaluation ───────────────────────────────────
    log.info("Step 9: Walk-forward evaluation...")
    wf_aucs, wf_briers, wf_accs = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(
        TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP).split(X_sel)
    ):
        base = lgb.LGBMClassifier(**best_params)
        cal  = CalibratedClassifierCV(base, cv=3, method="isotonic")
        cal.fit(X_sel[tr_idx], y[tr_idx])
        p = cal.predict_proba(X_sel[val_idx])[:, 1]
        wf_aucs.append(roc_auc_score(y[val_idx], p))
        wf_briers.append(brier_score_loss(y[val_idx], p))
        wf_accs.append((p.round() == y[val_idx]).mean())
        log.info("  Fold %d | AUC=%.4f | Brier=%.4f | Acc=%.4f",
                 fold, wf_aucs[-1], wf_briers[-1], wf_accs[-1])

    wf_auc   = float(np.mean(wf_aucs))
    wf_brier = float(np.mean(wf_briers))
    wf_acc   = float(np.mean(wf_accs))
    log.info("WF results: AUC=%.4f | Brier=%.4f | Acc=%.4f", wf_auc, wf_brier, wf_acc)

    # ── Step 10: Promotion gate ───────────────────────────────────────────
    beats_auc   = wf_auc   > champion["wf_auc"]
    beats_brier = wf_brier < champion["wf_brier"]
    beats_acc   = wf_acc   > champion["wf_acc"]
    score = sum([beats_auc, beats_brier, beats_acc])
    log.info("Gate vs %s: AUC %s | Brier %s | Acc %s → %d/3",
             champion["version"],
             "✓" if beats_auc else "✗",
             "✓" if beats_brier else "✗",
             "✓" if beats_acc else "✗",
             score)

    # Sanity check
    def _neutral_value(fname):
        if "dist_1k" in fname:
            return 0.25
        if "dollar_vol" in fname:
            return 5000.0
        if "ticks" in fname or "count" in fname:
            return 100.0
        if "up_ratio" in fname or "vwap_up" in fname or "vwap_dn" in fname or "buy_ratio" in fname:
            return 0.5
        if "vwap_spread" in fname:
            return 0.0
        if "ob_mid" in fname and "drift" not in fname:
            return 0.5
        if "ob_spread" in fname:
            return 0.02
        if "ob_depth_5c" in fname:
            return 0.5
        if "ob_total_depth" in fname:
            return 1000.0
        if "vol_regime" in fname:
            return 1.0  # medium vol regime
        if "vol_accel" in fname:
            return 0.0
        if "funding_proxy" in fname:
            return 0.0
        if "depth_change" in fname:
            return 0.0
        if any(k in fname for k in ("_ret", "zscore", "z_", "sin_", "cos_",
                                     "streak", "momentum", "stability",
                                     "disparity", "conviction", "signal",
                                     "imbalance", "imb", "drift", "change",
                                     "volatility", "fill", "x_")):
            return 0.0
        if "depth_ratio" in fname:
            return 1.0
        return 0.0

    baseline = {f: _neutral_value(f) for f in top_features}

    up_overrides = {
        "btc_up_ratio": 0.75, "btc_tw_up_ratio": 0.75,
        "btc_vwap_up": 0.55, "btc_vwap_dn": 0.45, "btc_vwap_spread": 0.10,
        "btc_momentum": 0.05, "btc_inslot_ret": 0.001,
        "btc_pre_5m_ret": 0.0005, "btc_signal_conviction": 0.7,
        "btc_buy_ratio": 0.6,
        # OB bullish signals
        "ob_imbalance": 0.3, "ob_imbalance_end": 0.3,
        "ob_mid_drift": 0.02, "ob_depth_ratio": 1.3,
        "ob_fill_imbalance": 0.2, "ob_imb_momentum": 0.1,
        "ob_pc_up_ratio": 0.6,
        # NEW v20 features — bullish
        "btc_vol_regime": 1.0,       # medium vol (trending)
        "btc_vol_accel": 0.3,        # accelerating vol (breakout)
        "ob_depth_change": 0.1,      # increasing depth (confidence)
        "btc_funding_proxy": 0.003,  # positive funding (bullish)
    }
    up_feats = dict(baseline)
    for k, v in up_overrides.items():
        if k in up_feats:
            up_feats[k] = v
    for f in top_features:
        if f.startswith("btc_up_w"):
            up_feats[f] = 0.65
        if f == "ob_imb_w0":
            up_feats[f] = 0.2
        if f == "ob_imb_w1":
            up_feats[f] = 0.25
        if f == "ob_imb_w2":
            up_feats[f] = 0.3

    down_overrides = {
        "btc_up_ratio": 0.25, "btc_tw_up_ratio": 0.25,
        "btc_vwap_up": 0.45, "btc_vwap_dn": 0.55, "btc_vwap_spread": -0.10,
        "btc_momentum": -0.05, "btc_inslot_ret": -0.001,
        "btc_pre_5m_ret": -0.0005, "btc_signal_conviction": 0.7,
        "btc_buy_ratio": 0.4,
        # OB bearish signals
        "ob_imbalance": -0.3, "ob_imbalance_end": -0.3,
        "ob_mid_drift": -0.02, "ob_depth_ratio": 0.7,
        "ob_fill_imbalance": -0.2, "ob_imb_momentum": -0.1,
        "ob_pc_up_ratio": 0.4,
        # NEW v20 features — bearish
        "btc_vol_regime": 2.0,        # high vol (panic)
        "btc_vol_accel": 0.5,         # vol spike
        "ob_depth_change": -0.1,      # decreasing depth (fleeing)
        "btc_funding_proxy": -0.003,  # negative funding (bearish)
    }
    down_feats = dict(baseline)
    for k, v in down_overrides.items():
        if k in down_feats:
            down_feats[k] = v
    for f in top_features:
        if f.startswith("btc_up_w"):
            down_feats[f] = 0.35
        if f == "ob_imb_w0":
            down_feats[f] = -0.2
        if f == "ob_imb_w1":
            down_feats[f] = -0.25
        if f == "ob_imb_w2":
            down_feats[f] = -0.3

    final_base  = lgb.LGBMClassifier(**best_params)
    final_model = CalibratedClassifierCV(final_base, cv=3, method="isotonic")
    final_model.fit(X_sel, y)

    up_arr    = pd.DataFrame([up_feats])[top_features].values.astype(np.float32)
    neut_arr  = pd.DataFrame([baseline])[top_features].values.astype(np.float32)
    down_arr  = pd.DataFrame([down_feats])[top_features].values.astype(np.float32)
    prob_up   = final_model.predict_proba(up_arr)[0, 1]
    prob_neut = final_model.predict_proba(neut_arr)[0, 1]
    prob_down = final_model.predict_proba(down_arr)[0, 1]
    log.info("Sanity: UP → %.3f | Neutral → %.3f | DOWN → %.3f", prob_up, prob_neut, prob_down)
    assert prob_up > prob_neut > prob_down, (
        f"Sanity gate FAILED: UP={prob_up:.3f} Neutral={prob_neut:.3f} DOWN={prob_down:.3f}"
    )

    # ── Step 11: Save & promote ───────────────────────────────────────────
    if score < 2:
        log.info("NOT PROMOTED (%d/3). Training complete.", score)
    else:
        log.info("PROMOTING v20! (%d/3 metrics beat champion)", score)
        import tempfile
        from huggingface_hub import HfApi

        model_data = {
            "version":  "v20",
            "features": top_features,
            "model":    final_model,
            "wf_auc":   wf_auc,
            "wf_brier": wf_brier,
            "wf_acc":   wf_acc,
        }
        meta = {
            "version":   "v20",
            "wf_auc":    wf_auc,
            "wf_brier":  wf_brier,
            "wf_acc":    wf_acc,
            "features":  top_features,
            "n_samples": len(y),
            "n_features": len(top_features),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "changes": (
                "Dataset expansion via pmdata (gap fill + new markets), "
                "new features: btc_vol_regime, btc_vol_accel, ob_depth_change, "
                "btc_funding_proxy. Inline OB extraction for new markets. "
                "Top 45 feature selection."
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

        log.info("v20 promoted to HF! AUC=%.4f Brier=%.4f Acc=%.4f",
                 wf_auc, wf_brier, wf_acc)

    log.info("v20 training complete.")


@app.local_entrypoint()
def main():
    train_v20.remote()
