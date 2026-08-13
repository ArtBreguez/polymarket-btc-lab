"""
fetch_ob_features_local.py — Pre-compute OB + CLOB features locally
=====================================================================
Fetches poly_l2 data from pmdata.dev for all 23k markets.
Computes features for TWO observation windows in one fetch per market:

  ob60_*  — t < 60s  (early signal, same as live OBS_SECS=60)
  ob180_* — t < 180s (late signal, right at entry time ~t=170-240s)

This lets the training script compare which window gives better AUC.

Extra features only available in ob180_*:
  ob180_imb_w1 — mean imbalance books t=[60,120s)
  ob180_imb_w2 — mean imbalance books t=[120,180s)
  ob180_mid_drift — ob180_mid - ob60_mid (temporal drift over slot)
  ob180_imb_momentum — ob180_imbalance - ob60_imbalance

Output: data/ob_features_v31.parquet (one row per market_id)
Resume: data/ob_progress_v31.json
"""

import io, json, logging, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

CUTOFF_60    = 60
CUTOFF_180   = 180
BASE_URL     = "https://api.pmdata.dev/download/poly_l2"
WORKERS      = 8
BATCH_SIZE   = 100
HTTP_TIMEOUT = 60

DATA_DIR    = Path("data")
OUT_FILE    = DATA_DIR / "ob_features_v31.parquet"
PROGRESS    = DATA_DIR / "ob_progress_v31.json"
MARKETS_CSV = DATA_DIR / "all_markets.csv"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout, force=True)
log = logging.getLogger(__name__)


def get_api_key() -> str:
    key = os.environ.get("PMDATA_API_KEY", "")
    if key:
        return key
    for raw in (Path.home() / ".env").read_text().splitlines():
        parts = raw.strip().split("=", 1)
        if len(parts) == 2 and parts[0].strip() == "PMDATA_API_KEY":
            return parts[1].strip().strip('"').strip("'")
    raise RuntimeError("PMDATA_API_KEY not set")


def _book_features(df_books: pd.DataFrame, t_sec_arr: np.ndarray, cutoff: float) -> dict | None:
    """
    Extract OB features from book snapshots with t < cutoff.
    Uses the LAST snap in window (most recent state before cutoff).
    Returns None if no book snaps found.
    """
    mask = t_sec_arr < cutoff
    if not mask.any():
        return None

    rows = df_books[mask]
    # Last snap in window
    row = rows.iloc[-1]
    try:
        ap  = np.asarray(row["ask_prices"], dtype=np.float64)
        asz = np.asarray(row["ask_sizes"],  dtype=np.float64)
        bp  = np.asarray(row["bid_prices"], dtype=np.float64)
        bsz = np.asarray(row["bid_sizes"],  dtype=np.float64)
        if not len(ap) or not len(bp):
            return None
        mid    = (ap[0] + bp[0]) / 2.0
        spread = ap[0] - bp[0]
        top    = bsz[0] + asz[0]
        imb    = (bsz[0] - asz[0]) / (top + 1e-8)
        tb     = bsz.sum() + 1e-8
        ta     = asz.sum() + 1e-8
        bd5    = bsz[bp >= mid - 0.05].sum() / tb
        ad5    = asz[ap <= mid + 0.05].sum() / ta
        dr     = bd5 / (ad5 + 1e-8)
        depth  = float(bsz.sum() + asz.sum())
        bw     = float(np.sum(bsz * np.exp(-10 * np.abs(bp - mid))))
        aw     = float(np.sum(asz * np.exp(-10 * np.abs(ap - mid))))
        wimb   = (bw - aw) / (bw + aw + 1e-8)
    except Exception:
        return None

    # ob_imb_w0: mean imbalance all snaps in [0, cutoff)
    imb_vals = []
    for _, r2 in rows.iterrows():
        try:
            b2  = np.asarray(r2["bid_sizes"], dtype=np.float64)
            a2  = np.asarray(r2["ask_sizes"], dtype=np.float64)
            t2  = b2[0] + a2[0]
            imb_vals.append((b2[0] - a2[0]) / (t2 + 1e-8))
        except Exception:
            pass

    return {
        "mid": float(mid), "spread": float(spread),
        "imbalance": float(imb), "depth_ratio": float(dr),
        "bid_depth_5c": float(bd5), "ask_depth_5c": float(ad5),
        "total_depth": float(depth), "weighted_imb": float(wimb),
        "imb_w0": float(np.mean(imb_vals)) if imb_vals else 0.0,
    }


def _windowed_imb(df_books: pd.DataFrame, t_sec_arr: np.ndarray,
                  windows: list[tuple]) -> dict:
    """Mean imbalance from book snaps in each (t0, t1) window."""
    result = {}
    for i, (t0, t1) in enumerate(windows):
        mask = (t_sec_arr >= t0) & (t_sec_arr < t1)
        imb_vals = []
        for _, row in df_books[mask].iterrows():
            try:
                b2 = np.asarray(row["bid_sizes"], dtype=np.float64)
                a2 = np.asarray(row["ask_sizes"], dtype=np.float64)
                t2 = b2[0] + a2[0]
                imb_vals.append((b2[0] - a2[0]) / (t2 + 1e-8))
            except Exception:
                pass
        result[f"imb_w{i}"] = float(np.mean(imb_vals)) if imb_vals else 0.0
    return result


def _pc_features(df_pc: pd.DataFrame, t_sec_arr: np.ndarray,
                 df_books: pd.DataFrame, b_t_sec: np.ndarray,
                 cutoff: float) -> dict:
    """
    Compute price_change and clob_* features for t < cutoff.
    Includes mid/spread time-series from books + pc best_bid/best_ask.
    """
    zeros = {
        "pc_up_ratio": 0.5, "pc_volatility": 0.0, "pc_count": 0.0,
        "fill_imbalance": 0.0,
        "clob_spread_mean": 0.0, "clob_spread_trend": 0.0,
        "clob_mid_velocity": 0.0, "clob_mid_volatility": 0.0,
        "clob_ask_pressure": 0.0,
    }

    p_mask = t_sec_arr < cutoff
    b_mask = b_t_sec < cutoff

    if not p_mask.any():
        return zeros

    pc = df_pc[p_mask]

    # ob_pc_*
    if "pc_price" in pc.columns:
        prices = pd.to_numeric(pc["pc_price"], errors="coerce").dropna().values.astype(np.float64)
        if len(prices) >= 2:
            diffs   = np.diff(prices)
            changes = diffs[diffs != 0]
            if len(changes) > 0:
                zeros["pc_up_ratio"]   = float((changes > 0).sum() / len(changes))
                zeros["pc_volatility"] = float(np.std(changes))
                zeros["pc_count"]      = float(len(changes))

    if "pc_side" in pc.columns and "pc_size" in pc.columns:
        sides = pc["pc_side"].str.upper().values
        szs   = pd.to_numeric(pc["pc_size"], errors="coerce").fillna(0).values.astype(np.float64)
        zeros["fill_imbalance"] = float(
            (szs[sides == "BUY"].sum() - szs[sides == "SELL"].sum()) /
            (szs.sum() + 1e-8)
        )

    # clob_*: mid/spread time series — books + pc best_bid/best_ask
    bk_mid, bk_sp, bk_ts = [], [], []
    for _, row in df_books[b_mask].iterrows():
        try:
            a0 = float(np.asarray(row["ask_prices"])[0])
            b0 = float(np.asarray(row["bid_prices"])[0])
            bk_mid.append((a0 + b0) / 2.0)
            bk_sp.append(a0 - b0)
            bk_ts.append(float(row["t_sec"]))
        except Exception:
            pass

    pc_mid_v, pc_sp_v, pc_ts_v = [], [], []
    if "best_ask" in pc.columns and "best_bid" in pc.columns:
        ba = pd.to_numeric(pc["best_ask"], errors="coerce").values.astype(np.float64)
        bb = pd.to_numeric(pc["best_bid"], errors="coerce").values.astype(np.float64)
        valid = (ba > bb) & (bb > 0) & ~np.isnan(ba) & ~np.isnan(bb)
        pc_mid_v  = ((ba[valid] + bb[valid]) / 2.0).tolist()
        pc_sp_v   = (ba[valid] - bb[valid]).tolist()
        pc_ts_v   = pc["t_sec"].values[valid].tolist()

    all_ts  = np.array(bk_ts + pc_ts_v)
    all_mid = np.array(bk_mid + pc_mid_v)
    all_sp  = np.array(bk_sp + pc_sp_v)

    if len(all_mid) >= 2:
        order = np.argsort(all_ts)
        ts_a, mid_a, sp_a = all_ts[order], all_mid[order], all_sp[order]
        t_rel = ts_a - ts_a[0]

        def _slope(t, y):
            if len(t) < 2 or t[-1] == t[0]:
                return 0.0
            try:
                return float(np.polyfit(t, y, 1)[0])
            except Exception:
                return 0.0

        zeros["clob_spread_mean"]    = float(np.mean(sp_a))
        zeros["clob_spread_trend"]   = _slope(t_rel, sp_a)
        zeros["clob_mid_velocity"]   = _slope(t_rel, mid_a)
        d = np.diff(mid_a)
        zeros["clob_mid_volatility"] = float(np.std(d)) if len(d) > 0 else 0.0

    if "pc_side" in pc.columns and "pc_price" in pc.columns:
        sides_v = pc["pc_side"].str.upper().values
        pp = pd.to_numeric(pc["pc_price"], errors="coerce").values.astype(np.float64)
        ask_seq = pp[(sides_v == "SELL") & ~np.isnan(pp)]
        if len(ask_seq) >= 2:
            d = np.diff(ask_seq)
            m = d[d != 0]
            zeros["clob_ask_pressure"] = float((m < 0).sum() / len(m)) if len(m) > 0 else 0.0

    return zeros


def compute_features(df: pd.DataFrame, slot_ts: int) -> dict | None:
    """
    Compute OB + CLOB features for BOTH t<60s and t<180s windows.
    Returns dict with ob60_* and ob180_* prefixed keys.
    Returns None if no book data at all.
    """
    # t_sec relative to slot_ts
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        t_sec = df["timestamp"].astype("int64").values / 1000.0 - slot_ts
    else:
        t_sec = pd.to_numeric(df["timestamp"], errors="coerce").values / 1000.0 - slot_ts

    # Filter to t=[0, 180s) — covers both windows
    mask = (t_sec >= 0) & (t_sec < CUTOFF_180)
    if not mask.any():
        return None

    df = df[mask].copy()
    df["t_sec"] = t_sec[mask]

    et = df["event_type"].values
    books = df[et == "book"].copy()
    pcs   = df[et == "price_change"].copy()

    if len(books) == 0:
        return None

    b_t  = books["t_sec"].values
    p_t  = pcs["t_sec"].values if len(pcs) > 0 else np.array([])

    # ── ob60_* features ──────────────────────────────────────────────────────
    ob60 = _book_features(books, b_t, CUTOFF_60)
    if ob60 is None:
        # No books in t<60s — try using very first book snap as fallback
        ob60 = _book_features(books, b_t, CUTOFF_180)
        if ob60 is None:
            return None
        ob60_fallback = True
    else:
        ob60_fallback = False

    # ── ob180_* features ─────────────────────────────────────────────────────
    ob180 = _book_features(books, b_t, CUTOFF_180)
    if ob180 is None:
        ob180 = dict(ob60)  # copy — avoid mutating ob60 when adding extra keys

    # Extra windows for ob180: w1=[60,120s), w2=[120,180s)
    extra = _windowed_imb(books, b_t, [(60, 120), (120, 180)])
    ob180["imb_w1"] = extra["imb_w0"]
    ob180["imb_w2"] = extra["imb_w1"]

    # Temporal drift: 180s snap vs 60s snap
    ob180["mid_drift"]     = ob180["mid"]     - ob60["mid"]
    ob180["imb_momentum"]  = ob180["imbalance"] - ob60["imbalance"]

    # ── pc features for each window ──────────────────────────────────────────
    if len(pcs) > 0:
        pc60  = _pc_features(pcs, p_t, books, b_t, CUTOFF_60)
        pc180 = _pc_features(pcs, p_t, books, b_t, CUTOFF_180)
    else:
        neutral = {
            "pc_up_ratio": 0.5, "pc_volatility": 0.0, "pc_count": 0.0,
            "fill_imbalance": 0.0, "clob_spread_mean": 0.0,
            "clob_spread_trend": 0.0, "clob_mid_velocity": 0.0,
            "clob_mid_volatility": 0.0, "clob_ask_pressure": 0.0,
        }
        pc60 = pc180 = neutral.copy()

    # ── Assemble with prefixes ────────────────────────────────────────────────
    feats: dict = {}

    for k, v in ob60.items():
        feats[f"ob60_{k}"] = v
    for k, v in pc60.items():
        feats[f"ob60_{k}"] = v   # pc60 keys: pc_up_ratio, clob_*, etc.

    for k, v in ob180.items():
        feats[f"ob180_{k}"] = v
    for k, v in pc180.items():
        feats[f"ob180_{k}"] = v

    return feats


def fetch_one(slug: str, slot_ts: int, mid: str, api_key: str) -> tuple:
    try:
        r = requests.get(f"{BASE_URL}/{slug}.parquet",
                         headers={"api_key": api_key}, timeout=HTTP_TIMEOUT)
        if not r.ok:
            return (slug, None)
        df = pd.read_parquet(io.BytesIO(r.content))
        feats = compute_features(df, slot_ts)
        if feats is not None:
            feats["market_id"] = mid
        return (slug, feats)
    except Exception:
        return (slug, None)


def main():
    api_key = get_api_key()
    log.info("PMDATA_API_KEY OK (len=%d)", len(api_key))

    markets = pd.read_csv(MARKETS_CSV)
    markets["market_id"] = markets["market_id"].astype(str)
    log.info("Total markets: %d", len(markets))

    done_slugs: set = set()
    n_existing = 0

    if PROGRESS.exists():
        try:
            done_slugs = set(json.loads(PROGRESS.read_text()))
            log.info("Resuming: %d done", len(done_slugs))
        except Exception:
            pass

    if OUT_FILE.exists() and done_slugs:
        try:
            n_existing = pq.read_metadata(OUT_FILE).num_rows
            log.info("Existing parquet: %d rows", n_existing)
        except Exception:
            pass

    # Reset progress if OBS windows changed (detect by checking col names)
    if n_existing > 0:
        try:
            schema = pq.read_schema(OUT_FILE)
            cols = [f.name for f in schema]
            if "ob60_mid" not in cols:
                log.info("Old schema detected (no ob60_* cols) — resetting progress")
                done_slugs = set()
                n_existing = 0
                OUT_FILE.unlink()
                PROGRESS.write_text("[]")
        except Exception:
            pass

    todo = markets[~markets["slug"].isin(done_slugs)].reset_index(drop=True)
    log.info("Remaining: %d markets", len(todo))

    tmp_file = Path(str(OUT_FILE) + ".new")

    success = failed = 0
    t0 = time.time()

    try:
        for batch_start in range(0, len(todo), BATCH_SIZE):
            batch = todo.iloc[batch_start : batch_start + BATCH_SIZE]

            batch_rows = []
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = {
                    ex.submit(fetch_one, r["slug"], int(r["slot_ts"]), r["market_id"], api_key): r["slug"]
                    for _, r in batch.iterrows()
                }
                for fut in as_completed(futs):
                    slug, feats = fut.result()
                    done_slugs.add(slug)
                    if feats is not None:
                        batch_rows.append(feats)
                        success += 1
                    else:
                        failed += 1

            PROGRESS.write_text(json.dumps(list(done_slugs)))

            if batch_rows:
                batch_df = pd.DataFrame(batch_rows)
                table = pa.Table.from_pandas(batch_df, preserve_index=False)
                out_path = tmp_file if n_existing > 0 else OUT_FILE
                # Write + immediately close so footer is flushed to disk.
                # pyarrow doesn't support append mode, so we accumulate via
                # read-merge-write on resume (handled below).
                if out_path.exists():
                    existing = pq.read_table(out_path)
                    merged = pa.concat_tables([existing, table])
                    pq.write_table(merged, out_path, compression="snappy")
                else:
                    pq.write_table(table, out_path, compression="snappy")

            done_n = batch_start + len(batch)
            elapsed = time.time() - t0
            rate = done_n / elapsed if elapsed > 0 else 1
            eta  = (len(todo) - done_n) / rate
            log.info("%d/%d | ok=%d fail=%d | %.1f/s | ETA=%.0fm",
                     done_n, len(todo), success, failed, rate, eta / 60)
    finally:
        pass  # writer is closed after each batch — no cleanup needed

    # Merge with existing rows if resuming
    if n_existing > 0 and tmp_file.exists():
        log.info("Merging %d new rows with %d existing...", success, n_existing)
        old = pq.read_table(OUT_FILE)
        new = pq.read_table(tmp_file)
        merged = pa.concat_tables([old, new])
        df_m = merged.to_pandas().drop_duplicates(subset=["market_id"])
        df_m.to_parquet(OUT_FILE, index=False, compression="snappy")
        tmp_file.unlink()
        log.info("Merged: %d total rows", len(df_m))

    final = pq.read_metadata(OUT_FILE).num_rows if OUT_FILE.exists() else 0
    log.info("DONE. %d rows saved (failed: %d) -> %s", final, failed, OUT_FILE)


if __name__ == "__main__":
    main()
