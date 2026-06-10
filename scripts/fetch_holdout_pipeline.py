"""
fetch_holdout_pipeline.py — Busca mercados de 7-10 jun (genuinamente OOS pro v29)
===================================================================================
Pipeline completo em sequência:
  1. Fetch markets do Gamma (slots 7-10 jun resolvidos)
  2. Fetch ticks do data-api
  3. Fetch OB features do pmdata (janela t<60s, v29)
  4. Upload tudo pro HF como holdout set permanente

Esses dados NUNCA entram no treino — são o gate de validação real.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow>=18.0", "pandas>=2.2", "numpy>=1.26",
        "requests>=2.31", "huggingface_hub>=0.26",
    )
)

LOCAL_VOL = modal.Volume.from_name("btc-local-data")
app = modal.App("btc-fetch-holdout", image=image)

# Cutoff: último slot no dataset de treino (6 jun 19:10 UTC)
TRAIN_CUTOFF_TS = 1780773000
# Holdout começa no slot seguinte
HOLDOUT_START_TS = TRAIN_CUTOFF_TS + 300


@app.function(
    cpu=8,
    memory=16384,
    timeout=7200,
    secrets=[
        modal.Secret.from_name("pmdata-api-key"),
        modal.Secret.from_name("hf-token"),
    ],
    volumes={"/btc_local": LOCAL_VOL},
)
def fetch_holdout():
    import gc, io, json, logging, math, os, sys, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests
    from huggingface_hub import HfApi

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    PMDATA_KEY    = os.environ.get("PMDATA_API_KEY", "")
    HF_TOKEN      = os.environ.get("HF_TOKEN", "")
    HF_REPO       = "artbreguez/polymarket-btc-model"
    LOCAL_DIR     = Path("/btc_local")
    PMDATA_BASE   = "https://api.pmdata.dev/download/poly_l2"
    GAMMA_BASE    = "https://gamma-api.polymarket.com"
    DATA_API_BASE = "https://data-api.polymarket.com"
    OBS_SECS      = 60

    # Até agora - 1h de buffer para resolução
    now_ts = int(time.time()) - 3600

    log.info("=" * 60)
    log.info("HOLDOUT PIPELINE")
    log.info("Janela: %s → %s",
             __import__("datetime").datetime.utcfromtimestamp(HOLDOUT_START_TS),
             __import__("datetime").datetime.utcfromtimestamp(now_ts))
    log.info("=" * 60)

    # ── STEP 1: Fetch markets do Gamma ────────────────────────────────────
    log.info("STEP 1: Fetching markets from Gamma...")

    slots = list(range(HOLDOUT_START_TS, now_ts, 300))
    log.info("Slots to scan: %d", len(slots))

    holdout_markets = []

    def _fetch_market(slot_ts):
        slug = f"btc-updown-5m-{slot_ts}"
        try:
            r = requests.get(f"{GAMMA_BASE}/markets/slug/{slug}", timeout=10)
            if not r.ok:
                return None
            m = r.json()
            if isinstance(m, list):
                m = m[0] if m else None
            if not m or not m.get("closed"):
                return None
            op = m.get("outcomePrices", "")
            prices = json.loads(op) if isinstance(op, str) else op
            if not prices:
                return None
            target = 1 if float(prices[0]) >= 0.5 else 0
            return {
                "market_id": str(m["id"]),
                "slug":      m["slug"],
                "slot_ts":   slot_ts,
                "target":    target,
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=40) as ex:
        futures = {ex.submit(_fetch_market, s): s for s in slots}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            if result:
                holdout_markets.append(result)
            if i % 200 == 0:
                log.info("  %d/%d scanned, %d found", i, len(slots), len(holdout_markets))

    holdout_df = pd.DataFrame(holdout_markets).sort_values("slot_ts").reset_index(drop=True)
    log.info("Markets found: %d (%.1f%% UP)", len(holdout_df), 100 * holdout_df["target"].mean())

    if len(holdout_df) == 0:
        log.error("Nenhum mercado encontrado — abortando")
        return {"error": "no markets found"}

    # Salvar
    holdout_csv = LOCAL_DIR / "holdout_markets.csv"
    holdout_df.to_csv(holdout_csv, index=False)
    LOCAL_VOL.commit()
    log.info("Saved holdout_markets.csv")

    all_ids = set(holdout_df["market_id"].tolist())

    # ── STEP 2: Fetch ticks do data-api ──────────────────────────────────
    log.info("STEP 2: Fetching ticks for %d markets...", len(holdout_df))

    tick_rows = []
    TICK_COLS = ["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"]

    def _fetch_ticks(market_id, slug, slot_ts):
        """Busca ticks do data-api para um mercado."""
        try:
            # data-api usa condition_id ou market slug — tentar via trades endpoint
            url = f"{DATA_API_BASE}/trades"
            params = {"market": slug, "limit": 1000}
            r = requests.get(url, params=params, timeout=15)
            if not r.ok:
                return []
            trades = r.json()
            if not isinstance(trades, list):
                return []
            rows = []
            obs_end = slot_ts + 60  # só ticks dentro da janela OBS_SECS
            for t in trades:
                try:
                    ts_ms = int(t.get("timestamp", 0))
                    ts_s  = ts_ms / 1000
                    if ts_s < slot_ts or ts_s >= slot_ts + 300:
                        continue
                    outcome = "Up" if str(t.get("outcome", "")).lower() in ("yes", "up", "1") else "Down"
                    rows.append({
                        "market_id":   market_id,
                        "timestamp_ms": ts_ms,
                        "outcome":     outcome,
                        "side":        t.get("side", "BUY"),
                        "price":       float(t.get("price", 0.5)),
                        "size_usdc":   float(t.get("amount", t.get("size", 1.0))),
                    })
                except Exception:
                    continue
            return rows
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {
            ex.submit(_fetch_ticks, r["market_id"], r["slug"], r["slot_ts"]): r["market_id"]
            for _, r in holdout_df.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            tick_rows.extend(fut.result())
            if i % 100 == 0:
                log.info("  ticks: %d/%d markets done, %d ticks total",
                         i, len(holdout_df), len(tick_rows))

    if tick_rows:
        ticks_df = pd.DataFrame(tick_rows)
        holdout_ticks_path = LOCAL_DIR / "holdout_ticks.parquet"
        ticks_df.to_parquet(holdout_ticks_path, index=False)
        LOCAL_VOL.commit()
        log.info("Saved %d ticks for %d markets", len(ticks_df),
                 ticks_df["market_id"].nunique())
    else:
        log.warning("No ticks fetched")
        ticks_df = pd.DataFrame(columns=TICK_COLS)

    # ── STEP 3: Fetch OB features do pmdata ──────────────────────────────
    log.info("STEP 3: Fetching OB features from pmdata...")

    def _extract_snap(row):
        try:
            ap  = np.array(row["ask_prices"], dtype=np.float64)
            as_ = np.array(row["ask_sizes"],  dtype=np.float64)
            bp  = np.array(row["bid_prices"], dtype=np.float64)
            bs  = np.array(row["bid_sizes"],  dtype=np.float64)
            if not len(ap) or not len(bp):
                return None
            mid    = float((ap[0] + bp[0]) / 2)
            spread = float(ap[0] - bp[0])
            tsz    = float(as_[0] + bs[0])
            imb    = float((bs[0] - as_[0]) / tsz) if tsz > 0 else 0.0
            total_bid   = bs.sum() + 1e-8
            total_ask   = as_.sum() + 1e-8
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

    def _linslope(t, y):
        if len(t) < 2:
            return 0.0
        t = np.array(t, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        tm, ym = t.mean(), y.mean()
        denom = np.sum((t - tm) ** 2)
        return float(np.sum((t - tm) * (y - ym)) / denom) if denom > 1e-12 else 0.0

    def _compute_ob_features(book_rows, pc_rows):
        if book_rows is None or len(book_rows) == 0:
            return None
        book_rows = book_rows.sort_values("t_sec")
        open_rows = book_rows[book_rows["t_sec"] <= 30]
        open_snap = _extract_snap(open_rows.iloc[0]) if len(open_rows) else _extract_snap(book_rows.iloc[0])
        if open_snap is None:
            return None

        feats = {
            "ob_imbalance":    open_snap["imb"],
            "ob_depth_ratio":  open_snap["dr"],
            "ob_total_depth":  open_snap["total_depth"],
            "ob_spread":       open_snap["spread"],
            "ob_mid":          open_snap["mid"],
            "ob_bid_depth_5c": open_snap["bd5"],
            "ob_ask_depth_5c": open_snap["ad5"],
            "ob_weighted_imb": open_snap["w_imb"],
            "ob_mid_drift": 0.0, "ob_imbalance_end": open_snap["imb"],
            "ob_spread_end": open_snap["spread"], "ob_depth_change": 0.0,
            "ob_imb_momentum": 0.0,
            "ob_pc_up_ratio": 0.5, "ob_pc_volatility": 0.0,
            "ob_pc_count": 0.0, "ob_fill_imbalance": 0.0,
        }

        for w_i, (t0, t1) in enumerate([(0, 20), (20, 40), (40, 60)]):
            w = book_rows[(book_rows["t_sec"] >= t0) & (book_rows["t_sec"] < t1)]
            imbs = [s["imb"] for _, r in w.iterrows() if (s := _extract_snap(r)) is not None]
            feats[f"ob_imb_w{w_i}"] = float(np.mean(imbs)) if imbs else 0.0

        # CLOB features (janela 0-60s)
        w_books = book_rows[(book_rows["t_sec"] >= 0) & (book_rows["t_sec"] < OBS_SECS)]
        w_pcs   = (pc_rows[(pc_rows["t_sec"] >= 0) & (pc_rows["t_sec"] < OBS_SECS)]
                   if pc_rows is not None and len(pc_rows) > 0 else pd.DataFrame())

        zeros_clob = {
            "clob_imb_mean": 0.0, "clob_imb_std": 0.0, "clob_imb_drift": 0.0,
            "clob_spread_mean": 0.0, "clob_spread_trend": 0.0,
            "clob_mid_velocity": 0.0, "clob_mid_volatility": 0.0,
            "clob_activity_rate": 0.0, "clob_depth_trend": 0.0,
            "clob_ask_pressure": 0.0,
        }

        if len(w_books) + len(w_pcs) < 3:
            feats.update(zeros_clob)
            return feats

        real_imb, real_depth, real_ts = [], [], []
        all_mid, all_spread, all_ts   = [], [], []
        ask_prices_seq = []

        for _, r in w_books.iterrows():
            s = _extract_snap(r)
            if s:
                real_imb.append(s["imb"]); real_depth.append(s["total_depth"])
                real_ts.append(float(r["t_sec"])); all_mid.append(s["mid"])
                all_spread.append(s["spread"]); all_ts.append(float(r["t_sec"]))

        for _, r in w_pcs.iterrows():
            try:
                ba = float(r["best_ask"]) if pd.notna(r.get("best_ask")) else None
                bb = float(r["best_bid"]) if pd.notna(r.get("best_bid")) else None
                if ba and bb and ba > bb > 0:
                    all_mid.append((ba + bb) / 2); all_spread.append(ba - bb)
                    all_ts.append(float(r["t_sec"]))
                if str(r.get("pc_side", "")).upper() == "SELL" and pd.notna(r.get("pc_price")):
                    ask_prices_seq.append(float(r["pc_price"]))
            except Exception:
                continue

        if len(all_mid) < 2:
            feats.update(zeros_clob)
            return feats

        order   = np.argsort(all_ts)
        ts_arr  = np.array(all_ts)[order]
        mid_arr = np.array(all_mid)[order]
        sp_arr  = np.array(all_spread)[order]
        t_rel   = ts_arr - ts_arr[0]

        clob = {
            "clob_imb_mean":     float(np.mean(real_imb))           if real_imb          else 0.0,
            "clob_imb_std":      float(np.std(real_imb))            if len(real_imb) > 1 else 0.0,
            "clob_imb_drift":    float(real_imb[-1] - real_imb[0]) if len(real_imb) > 1 else 0.0,
            "clob_spread_mean":  float(np.mean(sp_arr)),
            "clob_spread_trend": _linslope(t_rel, sp_arr),
            "clob_mid_velocity": _linslope(t_rel, mid_arr),
            "clob_mid_volatility": float(np.std(np.diff(mid_arr))) if len(mid_arr) > 1 else 0.0,
            "clob_activity_rate": float((len(w_books) + len(w_pcs)) / OBS_SECS),
            "clob_depth_trend":  _linslope(
                np.array(real_ts) - (real_ts[0] if real_ts else 0),
                np.array(real_depth)
            ) if len(real_depth) > 1 else 0.0,
            "clob_ask_pressure": (
                float((np.diff(ask_prices_seq) < 0).sum() / len(np.diff(ask_prices_seq)[np.diff(ask_prices_seq) != 0]))
                if len(ask_prices_seq) >= 2 and len(np.diff(ask_prices_seq)[np.diff(ask_prices_seq) != 0]) > 0
                else 0.0
            ),
        }
        feats.update(clob)
        return feats

    def _fetch_ob_one(market_id, slug, slot_ts):
        try:
            url = f"{PMDATA_BASE}/{slug}.parquet"
            r   = requests.get(url, headers={"api_key": PMDATA_KEY}, timeout=60)
            if not r.ok:
                return market_id, None
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
            )
            if feats:
                feats["market_id"] = market_id
            return market_id, feats
        except Exception:
            return market_id, None

    ob_rows = []
    ob_success = 0
    ob_failed  = 0

    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {
            ex.submit(_fetch_ob_one, r["market_id"], r["slug"], r["slot_ts"]): r["market_id"]
            for _, r in holdout_df.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            mid, feats = fut.result()
            if feats:
                ob_rows.append(feats)
                ob_success += 1
            else:
                ob_failed += 1
            if i % 100 == 0:
                log.info("  ob: %d/%d done, ok=%d fail=%d",
                         i, len(holdout_df), ob_success, ob_failed)

    log.info("OB fetch: %d ok, %d failed", ob_success, ob_failed)

    if ob_rows:
        ob_df = pd.DataFrame(ob_rows)
        holdout_ob_path = LOCAL_DIR / "holdout_ob_features.parquet"
        ob_df.to_parquet(holdout_ob_path, index=False)
        LOCAL_VOL.commit()
        log.info("Saved holdout_ob_features.parquet (%d rows)", len(ob_df))
    else:
        log.warning("No OB features — backtest will use zeros for OB")
        ob_df = pd.DataFrame()

    # ── STEP 4: Upload para HF ────────────────────────────────────────────
    log.info("STEP 4: Uploading holdout files to HF...")
    api = HfApi(token=HF_TOKEN)

    uploads = [
        (LOCAL_DIR / "holdout_markets.csv",        "data/holdout_markets.csv"),
        (LOCAL_DIR / "holdout_ticks.parquet",       "data/holdout_ticks.parquet"),
        (LOCAL_DIR / "holdout_ob_features.parquet", "data/holdout_ob_features.parquet"),
    ]

    for local_path, repo_path in uploads:
        if not local_path.exists():
            log.warning("  SKIP (not found): %s", local_path.name)
            continue
        size_mb = local_path.stat().st_size / 1024 / 1024
        log.info("  Uploading %s (%.1f MB)...", local_path.name, size_mb)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=HF_REPO,
            repo_type="model",
            commit_message=f"holdout: {local_path.name} (7-10 jun 2026, never seen by v29)",
        )

    log.info("=" * 60)
    log.info("HOLDOUT PIPELINE COMPLETO")
    log.info("Markets: %d | Ticks: %d | OB: %d/%d",
             len(holdout_df), len(ticks_df), ob_success, len(holdout_df))
    log.info("=" * 60)

    return {
        "n_markets":   len(holdout_df),
        "n_ticks":     len(ticks_df),
        "ob_success":  ob_success,
        "ob_failed":   ob_failed,
        "pct_up":      round(float(holdout_df["target"].mean()) * 100, 1),
    }


@app.local_entrypoint()
def main():
    result = fetch_holdout.remote()
    print("\nRESULT:", result)
