"""
fetch_ob_features_local.py — Pre-compute OB features locally (no Modal)
=======================================================================
Mesma lógica do fetch_ob_features_modal.py, mas roda diretamente na máquina.
Lê all_markets.csv, faz fetch paralelo da pmdata API, e salva em data/.

Economiza Modal — só treino vai pra lá.

Usage:
    python3 scripts/fetch_ob_features_local.py
    python3 scripts/fetch_ob_features_local.py --workers 20
    python3 scripts/fetch_ob_features_local.py --force    # recompute do zero
"""
import gc
import io
import json
import logging
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

PMDATA_KEY = os.environ.get("PMDATA_API_KEY", "")
HEADERS    = {"api_key": PMDATA_KEY}
BASE_URL   = "https://api.pmdata.dev/download/poly_l2"
OBS_SECS   = 180
CUTOFF_SEC = 168

DATA_DIR  = Path(__file__).parent.parent / "data"
OUT_FILE  = DATA_DIR / "ob_features_full.parquet"
PROG_FILE = DATA_DIR / "ob_progress.json"


# ── Feature computation (idêntico ao script Modal) ───────────────────────────

def _extract_snap(row) -> dict | None:
    try:
        ap  = np.array(row["ask_prices"], dtype=np.float64)
        as_ = np.array(row["ask_sizes"],  dtype=np.float64)
        bp  = np.array(row["bid_prices"], dtype=np.float64)
        bs  = np.array(row["bid_sizes"],  dtype=np.float64)
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


def _compute_ob_features(book_rows, pc_rows, slot_ts: int) -> dict | None:
    features = {}

    if book_rows is not None and len(book_rows) > 0:
        book_rows = book_rows.sort_values("t_sec")

        open_rows  = book_rows[book_rows["t_sec"] <= 30]
        close_rows = book_rows[(book_rows["t_sec"] >= 150) & (book_rows["t_sec"] < CUTOFF_SEC)]

        open_snap  = _extract_snap(open_rows.iloc[0])   if len(open_rows)  else _extract_snap(book_rows.iloc[0])
        close_snap = _extract_snap(close_rows.iloc[-1]) if len(close_rows) else None

        if open_snap is None:
            return None

        static = close_snap if close_snap is not None else open_snap
        features["ob_mid"]          = static["mid"]
        features["ob_spread"]       = static["spread"]
        features["ob_imbalance"]    = static["imb"]
        features["ob_depth_ratio"]  = static["dr"]
        features["ob_bid_depth_5c"] = static["bd5"]
        features["ob_ask_depth_5c"] = static["ad5"]
        features["ob_total_depth"]  = static["total_depth"]
        features["ob_weighted_imb"] = static["w_imb"]

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

        for w_i, (t0, t1) in enumerate([(0, 60), (60, 120), (120, CUTOFF_SEC)]):
            w_rows = book_rows[(book_rows["t_sec"] >= t0) & (book_rows["t_sec"] < t1)]
            if len(w_rows) > 0:
                imbs = [s["imb"] for r in w_rows.itertuples() if (s := _extract_snap(r)) is not None]
                features[f"ob_imb_w{w_i}"] = float(np.mean(imbs)) if imbs else 0.0
            else:
                features[f"ob_imb_w{w_i}"] = 0.0
    else:
        return None

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
            buy_vol  = buys["pc_size"].sum()  if "pc_size" in buys.columns  else len(buys)
            sell_vol = sells["pc_size"].sum() if "pc_size" in sells.columns else len(sells)
            features["ob_fill_imbalance"] = float((buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8))
        else:
            features["ob_fill_imbalance"] = 0.0
    else:
        features["ob_pc_up_ratio"]    = 0.5
        features["ob_pc_volatility"]  = 0.0
        features["ob_pc_count"]       = 0.0
        features["ob_fill_imbalance"] = 0.0

    return features


def _linslope(t, y) -> float:
    if len(t) < 2 or t[-1] == t[0]:
        return 0.0
    try:
        return float(np.polyfit(t, y, 1)[0])
    except Exception:
        return 0.0


def _compute_clob_features(book_rows, pc_rows,
                            window_start: float = 108.0,
                            window_end: float = 168.0) -> dict:
    zeros = {
        "clob_imb_mean": 0.0, "clob_imb_std": 0.0, "clob_imb_drift": 0.0,
        "clob_spread_mean": 0.0, "clob_spread_trend": 0.0,
        "clob_mid_velocity": 0.0, "clob_mid_volatility": 0.0,
        "clob_activity_rate": 0.0, "clob_depth_trend": 0.0,
        "clob_ask_pressure": 0.0,
    }

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

    time_span = window_end - window_start

    real_imb, real_depth, real_mid, real_spread, real_ts = [], [], [], [], []
    for _, row in w_books.iterrows():
        try:
            ap  = np.array(row["ask_prices"], dtype=np.float64)
            bp  = np.array(row["bid_prices"], dtype=np.float64)
            as_ = np.array(row["ask_sizes"],  dtype=np.float64)
            bs  = np.array(row["bid_sizes"],  dtype=np.float64)
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

    all_mid, all_spread, all_ts = list(real_mid), list(real_spread), list(real_ts)
    ask_prices_seq = []
    for _, row in w_pcs.iterrows():
        try:
            ba = float(row["best_ask"]) if pd.notna(row.get("best_ask")) else None
            bb = float(row["best_bid"]) if pd.notna(row.get("best_bid")) else None
            if ba is not None and bb is not None and ba > bb > 0:
                all_mid.append((ba + bb) / 2)
                all_spread.append(ba - bb)
                all_ts.append(float(row["t_sec"]))
            side = str(row.get("pc_side", "")).upper()
            if side == "SELL" and pd.notna(row.get("pc_price")):
                ask_prices_seq.append(float(row["pc_price"]))
        except Exception:
            continue

    if len(all_mid) < 2:
        return zeros

    order          = np.argsort(all_ts)
    all_ts_arr     = np.array(all_ts)[order]
    all_mid_arr    = np.array(all_mid)[order]
    all_spread_arr = np.array(all_spread)[order]
    t_rel          = all_ts_arr - all_ts_arr[0]

    feats = {}
    feats["clob_imb_mean"]       = float(np.mean(real_imb))           if real_imb          else 0.0
    feats["clob_imb_std"]        = float(np.std(real_imb))            if len(real_imb) > 1 else 0.0
    feats["clob_imb_drift"]      = float(real_imb[-1] - real_imb[0]) if len(real_imb) > 1 else 0.0
    feats["clob_spread_mean"]    = float(np.mean(all_spread_arr))
    feats["clob_spread_trend"]   = _linslope(t_rel, all_spread_arr)
    feats["clob_mid_velocity"]   = _linslope(t_rel, all_mid_arr)
    mid_diffs = np.diff(all_mid_arr)
    feats["clob_mid_volatility"] = float(np.std(mid_diffs)) if len(mid_diffs) > 0 else 0.0
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
    try:
        url = f"{BASE_URL}/{slug}.parquet"
        r = requests.get(url, headers=HEADERS, timeout=120)
        if not r.ok:
            return (market_id, None)

        df = pd.read_parquet(io.BytesIO(r.content))

        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["ts_sec"] = df["timestamp"].astype("int64") / 1000.0
        else:
            df["ts_sec"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)
        df["t_sec"] = df["ts_sec"] - slot_ts
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
            )
            feats.update(clob_feats)
            feats["market_id"] = market_id
        return (market_id, feats)
    except Exception:
        return (market_id, None)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20, help="Parallel HTTP threads (default 20)")
    parser.add_argument("--force",   action="store_true",  help="Recompute do zero (ignora progress)")
    parser.add_argument("--batch",   type=int, default=200, help="Batch size (default 200)")
    args = parser.parse_args()

    if not PMDATA_KEY:
        log.error("PMDATA_API_KEY não encontrada. Defina no .env ou ambiente.")
        sys.exit(1)

    if args.force:
        OUT_FILE.unlink(missing_ok=True)
        PROG_FILE.unlink(missing_ok=True)
        log.info("--force: arquivos removidos, começando do zero")

    # Carregar mercados
    markets = pd.read_csv(DATA_DIR / "all_markets.csv")
    markets["market_id"] = markets["market_id"].astype(str)
    markets["slot_ts"]   = markets["slot_ts"].astype(int)
    log.info("Total mercados: %d", len(markets))

    if "slug" not in markets.columns:
        log.error("all_markets.csv não tem coluna 'slug'!")
        sys.exit(1)

    # Resume
    done_ids: set = set()
    existing_rows = []

    if PROG_FILE.exists():
        try:
            done_ids = set(json.loads(PROG_FILE.read_text()))
            log.info("Resumindo: %d mercados já prontos", len(done_ids))
        except Exception as e:
            log.warning("Erro ao ler progress: %s", e)

    if OUT_FILE.exists() and done_ids:
        try:
            prev_df = pd.read_parquet(OUT_FILE)
            existing_rows = prev_df.to_dict("records")
            log.info("Carregado parquet existente: %d rows", len(existing_rows))
        except Exception as e:
            log.warning("Erro ao ler parquet: %s", e)

    todo = markets[~markets["market_id"].isin(list(done_ids))]
    log.info("Mercados pendentes: %d / %d", len(todo), len(markets))

    if len(todo) == 0:
        log.info("Tudo já processado!")
        return

    new_rows = []
    success = 0
    failed  = 0
    t0_total = time.time()

    for batch_start in range(0, len(todo), args.batch):
        batch = todo.iloc[batch_start:batch_start + args.batch]
        t0_batch = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
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

        # Salvar progress
        PROG_FILE.write_text(json.dumps(list(done_ids)))

        # Flush parquet
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            all_df = pd.DataFrame(existing_rows + new_rows) if existing_rows else new_df
            all_df = all_df.drop_duplicates(subset=["market_id"])
            all_df.to_parquet(OUT_FILE, index=False, compression="snappy")
            existing_rows = all_df.to_dict("records")
            new_rows = []

        done_so_far = batch_start + len(batch)
        elapsed = time.time() - t0_total
        rate = done_so_far / elapsed if elapsed > 0 else 1
        remaining = (len(todo) - done_so_far) / rate if rate > 0 else 0
        batch_time = time.time() - t0_batch

        log.info(
            "%d/%d | ok=%d fail=%d saved=%d | batch=%.1fs | eta=%.0fmin",
            done_so_far, len(todo), success, failed, len(existing_rows),
            batch_time, remaining / 60,
        )

    log.info("DONE. Total salvo: %d mercados (falhas: %d)", len(existing_rows), failed)
    log.info("Output: %s", OUT_FILE)


if __name__ == "__main__":
    main()
