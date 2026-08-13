"""
fetch_holdout_pipeline.py — Busca mercados 7-10 jun (holdout OOS pro v29)
==========================================================================
Roda LOCAL — sem Modal.

Pipeline:
  1. Fetch markets do Gamma (slots resolvidos pós TRAIN_CUTOFF)
  2. Fetch ticks do data-api
  3. Fetch OB features do pmdata (t<60s)
  4. Salva local + upload HF

Uso:
    cd ~/polymarket-btc-lab
    python scripts/fetch_holdout_pipeline.py
    python scripts/fetch_holdout_pipeline.py --skip-upload   # só local
"""
import argparse, io, json, logging, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).parent.parent / "data"
GAMMA_BASE    = "https://gamma-api.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"
PMDATA_BASE   = "https://api.pmdata.dev/download/poly_l2"
HF_REPO       = "artbreguez/polymarket-btc-model"
OBS_SECS      = 60

# Cutoff: último slot do dataset de treino v29 (6 jun 19:10 UTC)
TRAIN_CUTOFF_TS = 1780773000
HOLDOUT_START   = TRAIN_CUTOFF_TS + 300


def load_env():
    env = {}
    for path in [Path.home() / ".env", Path(__file__).parent.parent / ".env"]:
        if path.exists():
            for line in path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env


# ── STEP 1: Markets ───────────────────────────────────────────────────────────
def fetch_markets(now_ts: int) -> pd.DataFrame:
    slots = list(range(HOLDOUT_START, now_ts, 300))
    log.info("STEP 1: Scanning %d slots (%s → %s)...",
             len(slots),
             pd.Timestamp(HOLDOUT_START, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M"),
             pd.Timestamp(now_ts, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M"))

    # Resume: carregar já fetchados
    out_path = DATA_DIR / "holdout_markets.csv"
    done_slots = set()
    rows = []
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done_slots = set(existing["slot_ts"].tolist())
        rows = existing.to_dict("records")
        log.info("  Resume: %d já fetchados", len(done_slots))

    todo = [s for s in slots if s not in done_slots]
    log.info("  Restantes: %d", len(todo))

    def _fetch(slot_ts):
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
            return {
                "market_id": str(m["id"]),
                "slug":      m["slug"],
                "slot_ts":   slot_ts,
                "target":    1 if float(prices[0]) >= 0.5 else 0,
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=40) as ex:
        futures = {ex.submit(_fetch, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 200 == 0 or i == len(todo):
                log.info("  %d/%d scanned, %d found", i, len(todo), len(rows))

    df = pd.DataFrame(rows).sort_values("slot_ts").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    log.info("Markets: %d (%.1f%% UP) → %s", len(df), 100 * df["target"].mean(), out_path)
    return df


# ── STEP 2: Ticks ─────────────────────────────────────────────────────────────
def fetch_ticks(markets_df: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 2: Fetching ticks for %d markets...", len(markets_df))

    out_path = DATA_DIR / "holdout_ticks.parquet"
    done_ids = set()
    existing_rows = []
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        done_ids = set(existing["market_id"].astype(str).tolist())
        existing_rows = existing.to_dict("records")
        log.info("  Resume: %d já fetchados", len(done_ids))

    todo = markets_df[~markets_df["market_id"].isin(done_ids)]
    log.info("  Restantes: %d", len(todo))

    TICK_COLS = ["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"]

    def _fetch(market_id, slug, slot_ts):
        try:
            r = requests.get(
                f"{DATA_API_BASE}/trades",
                params={"market": slug, "limit": 500},
                timeout=15,
            )
            if not r.ok:
                return []
            trades = r.json()
            if not isinstance(trades, list):
                return []
            rows = []
            for t in trades:
                ts_ms = int(t.get("timestamp", 0))
                ts_s  = ts_ms / 1000
                if ts_s < slot_ts or ts_s >= slot_ts + 300:
                    continue
                outcome_raw = str(t.get("outcome", "")).lower()
                outcome = "Up" if outcome_raw in ("yes", "up", "1") else "Down"
                rows.append({
                    "market_id":    market_id,
                    "timestamp_ms": ts_ms,
                    "outcome":      outcome,
                    "side":         t.get("side", "BUY"),
                    "price":        float(t.get("price", 0.5)),
                    "size_usdc":    float(t.get("amount", t.get("size", 1.0))),
                })
            return rows
        except Exception:
            return []

    new_rows = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {
            ex.submit(_fetch, r["market_id"], r["slug"], r["slot_ts"]): r["market_id"]
            for _, r in todo.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            new_rows.extend(fut.result())
            if i % 100 == 0 or i == len(todo):
                log.info("  %d/%d done, %d ticks", i, len(todo), len(new_rows))

    all_rows = existing_rows + new_rows
    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(columns=TICK_COLS)
    if len(df):
        df.to_parquet(out_path, index=False)
    log.info("Ticks: %d total, %d markets → %s", len(df),
             df["market_id"].nunique() if len(df) else 0, out_path)
    return df


# ── STEP 3: OB features (pmdata) ─────────────────────────────────────────────
def fetch_ob(markets_df: pd.DataFrame, pmdata_key: str) -> pd.DataFrame:
    log.info("STEP 3: Fetching OB features (%d markets)...", len(markets_df))

    out_path = DATA_DIR / "holdout_ob_features.parquet"
    done_ids = set()
    existing_rows = []
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        done_ids = set(existing["market_id"].astype(str).tolist())
        existing_rows = existing.to_dict("records")
        log.info("  Resume: %d já fetchados", len(done_ids))

    todo = markets_df[~markets_df["market_id"].isin(done_ids)]
    log.info("  Restantes: %d", len(todo))

    headers = {"api_key": pmdata_key}

    def _extract_snap(row):
        try:
            ap  = np.array(row["ask_prices"], dtype=np.float64)
            as_ = np.array(row["ask_sizes"],  dtype=np.float64)
            bp  = np.array(row["bid_prices"], dtype=np.float64)
            bs  = np.array(row["bid_sizes"],  dtype=np.float64)
            if not len(ap) or not len(bp):
                return None
            mid  = float((ap[0] + bp[0]) / 2)
            spread = float(ap[0] - bp[0])
            tsz  = float(as_[0] + bs[0])
            imb  = float((bs[0] - as_[0]) / tsz) if tsz > 0 else 0.0
            bd5  = float(bs[bp >= mid - 0.05].sum() / (bs.sum() + 1e-8))
            ad5  = float(as_[ap <= mid + 0.05].sum() / (as_.sum() + 1e-8))
            dr   = float(bd5 / (ad5 + 1e-8))
            td   = float(bs.sum() + as_.sum())
            bwt  = float(np.sum(bs * np.exp(-10 * np.abs(bp - mid))))
            awt  = float(np.sum(as_ * np.exp(-10 * np.abs(ap - mid))))
            wimb = float((bwt - awt) / (bwt + awt + 1e-8))
            return {"mid": mid, "spread": spread, "imb": imb,
                    "bd5": bd5, "ad5": ad5, "dr": dr, "td": td, "wimb": wimb}
        except Exception:
            return None

    def _linslope(t, y):
        if len(t) < 2:
            return 0.0
        t, y = np.array(t, dtype=np.float64), np.array(y, dtype=np.float64)
        tm, ym = t.mean(), y.mean()
        d = np.sum((t - tm) ** 2)
        return float(np.sum((t - tm) * (y - ym)) / d) if d > 1e-12 else 0.0

    def _compute(book_rows, pc_rows):
        if book_rows is None or len(book_rows) == 0:
            return None
        book_rows = book_rows.sort_values("t_sec")
        open_rows = book_rows[book_rows["t_sec"] <= 30]
        snap = _extract_snap(open_rows.iloc[0]) if len(open_rows) else _extract_snap(book_rows.iloc[0])
        if not snap:
            return None

        feats = {
            "ob_imbalance": snap["imb"], "ob_depth_ratio": snap["dr"],
            "ob_total_depth": snap["td"], "ob_spread": snap["spread"],
            "ob_mid": snap["mid"], "ob_bid_depth_5c": snap["bd5"],
            "ob_ask_depth_5c": snap["ad5"], "ob_weighted_imb": snap["wimb"],
            "ob_mid_drift": 0.0, "ob_imbalance_end": snap["imb"],
            "ob_spread_end": snap["spread"], "ob_depth_change": 0.0,
            "ob_imb_momentum": 0.0, "ob_pc_up_ratio": 0.5,
            "ob_pc_volatility": 0.0, "ob_pc_count": 0.0, "ob_fill_imbalance": 0.0,
        }
        for wi, (t0, t1) in enumerate([(0, 20), (20, 40), (40, 60)]):
            w = book_rows[(book_rows["t_sec"] >= t0) & (book_rows["t_sec"] < t1)]
            imbs = [s["imb"] for _, r in w.iterrows() if (s := _extract_snap(r))]
            feats[f"ob_imb_w{wi}"] = float(np.mean(imbs)) if imbs else 0.0

        # CLOB features t=[0,60s]
        wb = book_rows[(book_rows["t_sec"] >= 0) & (book_rows["t_sec"] < OBS_SECS)]
        wp = (pc_rows[(pc_rows["t_sec"] >= 0) & (pc_rows["t_sec"] < OBS_SECS)]
              if pc_rows is not None and len(pc_rows) else pd.DataFrame())

        zeros = {k: 0.0 for k in ["clob_imb_mean","clob_imb_std","clob_imb_drift",
                                    "clob_spread_mean","clob_spread_trend","clob_mid_velocity",
                                    "clob_mid_volatility","clob_activity_rate",
                                    "clob_depth_trend","clob_ask_pressure"]}
        if len(wb) + len(wp) < 3:
            feats.update(zeros)
            return feats

        ri, rd, rt, am, asp, at_, asks = [], [], [], [], [], [], []
        for _, r in wb.iterrows():
            s = _extract_snap(r)
            if s:
                ri.append(s["imb"]); rd.append(s["td"]); rt.append(float(r["t_sec"]))
                am.append(s["mid"]); asp.append(s["spread"]); at_.append(float(r["t_sec"]))
        for _, r in wp.iterrows():
            try:
                ba = float(r["best_ask"]) if pd.notna(r.get("best_ask")) else None
                bb = float(r["best_bid"]) if pd.notna(r.get("best_bid")) else None
                if ba and bb and ba > bb > 0:
                    am.append((ba+bb)/2); asp.append(ba-bb); at_.append(float(r["t_sec"]))
                if str(r.get("pc_side","")).upper() == "SELL" and pd.notna(r.get("pc_price")):
                    asks.append(float(r["pc_price"]))
            except Exception:
                pass

        if len(am) < 2:
            feats.update(zeros); return feats

        order = np.argsort(at_)
        ts  = np.array(at_)[order]; ma = np.array(am)[order]; sa = np.array(asp)[order]
        tr  = ts - ts[0]
        md  = np.diff(ma)
        ask_diffs = np.diff(asks) if len(asks) >= 2 else np.array([])
        nonzero   = ask_diffs[ask_diffs != 0]

        feats.update({
            "clob_imb_mean":     float(np.mean(ri))           if ri          else 0.0,
            "clob_imb_std":      float(np.std(ri))            if len(ri)>1   else 0.0,
            "clob_imb_drift":    float(ri[-1]-ri[0])          if len(ri)>1   else 0.0,
            "clob_spread_mean":  float(np.mean(sa)),
            "clob_spread_trend": _linslope(tr, sa),
            "clob_mid_velocity": _linslope(tr, ma),
            "clob_mid_volatility": float(np.std(md))          if len(md)     else 0.0,
            "clob_activity_rate": float((len(wb)+len(wp))/OBS_SECS),
            "clob_depth_trend":  _linslope(np.array(rt)-rt[0] if rt else [0], np.array(rd)) if len(rd)>1 else 0.0,
            "clob_ask_pressure": float((nonzero<0).sum()/len(nonzero)) if len(nonzero) else 0.0,
        })
        return feats

    def _fetch_ob(market_id, slug, slot_ts):
        try:
            r = requests.get(f"{PMDATA_BASE}/{slug}.parquet",
                             headers=headers, timeout=30)
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
            feats = _compute(books if len(books) else None,
                             pcs   if len(pcs)   else None)
            if feats:
                feats["market_id"] = market_id
            return market_id, feats
        except Exception:
            return market_id, None

    ok, fail = 0, 0
    new_rows = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {
            ex.submit(_fetch_ob, r["market_id"], r["slug"], r["slot_ts"]): r["market_id"]
            for _, r in todo.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            mid, feats = fut.result()
            if feats:
                new_rows.append(feats); ok += 1
            else:
                fail += 1
            if i % 100 == 0 or i == len(todo):
                log.info("  %d/%d done, ok=%d fail=%d", i, len(todo), ok, fail)
                # Checkpoint: salvar progresso a cada 100
                if new_rows:
                    _df = pd.DataFrame(existing_rows + new_rows)
                    _df.to_parquet(out_path, index=False)

    all_rows = existing_rows + new_rows
    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    if len(df):
        df.to_parquet(out_path, index=False)
    log.info("OB: %d ok, %d failed → %s", ok, fail, out_path)
    return df


# ── STEP 4: Upload HF ─────────────────────────────────────────────────────────
def upload_hf(hf_token: str):
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    files = [
        (DATA_DIR / "holdout_markets.csv",        "data/holdout_markets.csv"),
        (DATA_DIR / "holdout_ticks.parquet",       "data/holdout_ticks.parquet"),
        (DATA_DIR / "holdout_ob_features.parquet", "data/holdout_ob_features.parquet"),
    ]
    log.info("STEP 4: Uploading holdout files to HF...")
    for local, remote in files:
        if not local.exists():
            log.warning("  SKIP (not found): %s", local.name)
            continue
        log.info("  %s (%.1f MB)...", local.name, local.stat().st_size / 1e6)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=HF_REPO,
            repo_type="model",
            commit_message=f"holdout: {local.name} (7-10 jun 2026)",
        )
    log.info("Upload completo.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-ticks",  action="store_true")
    parser.add_argument("--skip-ob",     action="store_true")
    args = parser.parse_args()

    env = load_env()
    pmdata_key = env.get("PMDATA_API_KEY", "")
    hf_token   = env.get("HF_TOKEN", "")

    if not pmdata_key:
        log.error("PMDATA_API_KEY não encontrado no .env")
        sys.exit(1)

    now_ts = int(time.time()) - 3600  # 1h buffer para resolução

    markets_df = fetch_markets(now_ts)
    if len(markets_df) == 0:
        log.error("Nenhum mercado encontrado"); sys.exit(1)

    if not args.skip_ticks:
        fetch_ticks(markets_df)

    if not args.skip_ob:
        fetch_ob(markets_df, pmdata_key)

    if not args.skip_upload:
        if not hf_token:
            log.warning("HF_TOKEN não encontrado — pulando upload")
        else:
            upload_hf(hf_token)

    log.info("Pipeline completo.")
