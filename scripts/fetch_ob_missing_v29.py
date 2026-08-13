"""
fetch_ob_missing_v29.py — Fetch OB features para os ~183 mercados sem OB
=========================================================================
- Lê missing_ob_markets.csv (gerado localmente com os IDs faltantes)
- Busca poly_l2 do pmdata.dev para cada mercado
- Computa features com janela t=0..60s (v29 — sem leakage)
- Mergeia no ob_features_full.parquet existente
- Faz upload do parquet atualizado para HF (single source of truth)
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow>=18.0",
        "pandas>=2.2",
        "numpy>=1.26",
        "requests>=2.31",
        "huggingface_hub>=0.26",
    )
)

LOCAL_VOL = modal.Volume.from_name("btc-local-data")
DATA_VOL  = modal.Volume.from_name("btc-data-cache", create_if_missing=True)

app = modal.App("btc-fetch-ob-missing-v29", image=image)


@app.function(
    cpu=4,
    memory=16384,
    timeout=3600,
    secrets=[
        modal.Secret.from_name("pmdata-api-key"),
        modal.Secret.from_name("hf-token"),
    ],
    volumes={
        "/btc_local": LOCAL_VOL,
        "/cache":     DATA_VOL,
    },
)
def fetch_ob_missing():
    import gc, io, json, logging, os, sys, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import requests
    from huggingface_hub import HfApi

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    PMDATA_KEY    = os.environ.get("PMDATA_API_KEY", "")
    HF_TOKEN      = os.environ.get("HF_TOKEN", "")
    HEADERS       = {"api_key": PMDATA_KEY}
    BASE_URL      = "https://api.pmdata.dev/download/poly_l2"
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
    LOCAL_DIR     = Path("/btc_local")
    CACHE_DIR     = Path("/cache")

    OBS_SECS = 60     # v29: janela de 60s — zero leakage

    # ── Carregar lista de faltantes do volume local ───────────────────────
    missing_path = LOCAL_DIR / "missing_ob_markets.csv"
    if not missing_path.exists():
        # Tentar no cache também
        missing_path = CACHE_DIR / "missing_ob_markets.csv"
    if not missing_path.exists():
        raise RuntimeError("missing_ob_markets.csv não encontrado. Gere localmente primeiro.")

    todo = pd.read_csv(missing_path)
    todo["market_id"] = todo["market_id"].astype(str)
    todo["slot_ts"]   = todo["slot_ts"].astype(int)
    log.info("Mercados a buscar: %d", len(todo))

    # ── Carregar OB existente ─────────────────────────────────────────────
    ob_path = LOCAL_DIR / "ob_features_full.parquet"
    if not ob_path.exists():
        ob_path = CACHE_DIR / "ob_features_full.parquet"

    if ob_path.exists():
        existing_ob = pd.read_parquet(ob_path)
        existing_ob["market_id"] = existing_ob["market_id"].astype(str)
        already_done = set(existing_ob["market_id"].tolist())
        log.info("OB existente: %d mercados", len(existing_ob))
    else:
        existing_ob = None
        already_done = set()
        log.warning("ob_features_full.parquet não encontrado — criando novo")

    todo = todo[~todo["market_id"].isin(already_done)].reset_index(drop=True)
    log.info("Após dedup: %d mercados a buscar", len(todo))

    # ── Feature extraction functions (v29 — janela 0..OBS_SECS) ──────────

    def _extract_snap(row) -> dict | None:
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

    def compute_features(book_rows: pd.DataFrame, pc_rows: pd.DataFrame) -> dict | None:
        """
        Computa todas as OB features dentro de t=[0, OBS_SECS=60s].

        Features v29 (sem leakage):
          - ob_imbalance, ob_depth_ratio, ob_total_depth, ob_spread: snapshot "open" (t=0-30s)
          - clob_spread_mean/trend, clob_mid_volatility, clob_ask_pressure: janela t=[0, 60s)

        Features legadas também computadas para manter schema compatível com treinos anteriores
        (ob_mid, ob_imbalance_end, etc) — usam snapshot "open" como fallback já que não temos
        dados além de 60s. Isso é honesto: os valores serão iguais ao open, sem leakage.
        """
        feats = {}

        if book_rows is None or len(book_rows) == 0:
            return None  # sem book data = skip

        book_rows = book_rows.sort_values("t_sec")

        # snapshot "open": primeiro disponível em t=0-30s (ou primeiro disponível)
        open_rows = book_rows[book_rows["t_sec"] <= 30]
        open_snap = _extract_snap(open_rows.iloc[0]) if len(open_rows) else _extract_snap(book_rows.iloc[0])

        if open_snap is None:
            return None

        # v29: features do snapshot open
        feats["ob_imbalance"]    = open_snap["imb"]
        feats["ob_depth_ratio"]  = open_snap["dr"]
        feats["ob_total_depth"]  = open_snap["total_depth"]
        feats["ob_spread"]       = open_snap["spread"]
        feats["ob_mid"]          = open_snap["mid"]         # legado — mesma coisa que open
        feats["ob_bid_depth_5c"] = open_snap["bd5"]         # legado
        feats["ob_ask_depth_5c"] = open_snap["ad5"]         # legado
        feats["ob_weighted_imb"] = open_snap["w_imb"]       # legado

        # Temporal features legadas — sem snapshot "end" no v29 (t=60s só)
        # Usa open como fallback → delta = 0 (honesto, sem leakage)
        feats["ob_mid_drift"]     = 0.0
        feats["ob_imbalance_end"] = open_snap["imb"]
        feats["ob_spread_end"]    = open_snap["spread"]
        feats["ob_depth_change"]  = 0.0
        feats["ob_imb_momentum"]  = 0.0

        # Windowed imbalance dentro de 60s
        for w_i, (t0, t1) in enumerate([(0, 20), (20, 40), (40, 60)]):
            w_rows = book_rows[(book_rows["t_sec"] >= t0) & (book_rows["t_sec"] < t1)]
            if len(w_rows) > 0:
                imbs = [s["imb"] for r in w_rows.itertuples() if (s := _extract_snap(r)) is not None]
                feats[f"ob_imb_w{w_i}"] = float(np.mean(imbs)) if imbs else 0.0
            else:
                feats[f"ob_imb_w{w_i}"] = 0.0

        # pc features legadas
        feats["ob_pc_up_ratio"]    = 0.5
        feats["ob_pc_volatility"]  = 0.0
        feats["ob_pc_count"]       = 0.0
        feats["ob_fill_imbalance"] = 0.0

        if pc_rows is not None and len(pc_rows) > 0:
            pc_cut = pc_rows[pc_rows["t_sec"] < OBS_SECS].copy()
            if len(pc_cut) > 0 and "pc_price" in pc_cut.columns:
                prices = pc_cut["pc_price"].dropna().values.astype(float)
                diffs  = np.diff(prices)
                n_up   = int((diffs > 0).sum())
                n_dn   = int((diffs < 0).sum())
                tot    = n_up + n_dn
                feats["ob_pc_up_ratio"]   = float(n_up / (tot + 1e-8))
                feats["ob_pc_volatility"] = float(np.std(diffs)) if len(diffs) > 1 else 0.0
                feats["ob_pc_count"]      = float(tot)
            if len(pc_cut) > 0 and "pc_side" in pc_cut.columns:
                buys     = pc_cut[pc_cut["pc_side"].str.upper() == "BUY"]
                sells    = pc_cut[pc_cut["pc_side"].str.upper() == "SELL"]
                buy_vol  = buys["pc_size"].sum()  if "pc_size"  in buys.columns  else float(len(buys))
                sell_vol = sells["pc_size"].sum() if "pc_size"  in sells.columns else float(len(sells))
                feats["ob_fill_imbalance"] = float((buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8))

        # ── CLOB features — janela t=[0, 60s) (v29) ───────────────────────
        # Usa todos os eventos dentro da janela de observação
        zeros_clob = {
            "clob_imb_mean": 0.0, "clob_imb_std": 0.0, "clob_imb_drift": 0.0,
            "clob_spread_mean": 0.0, "clob_spread_trend": 0.0,
            "clob_mid_velocity": 0.0, "clob_mid_volatility": 0.0,
            "clob_activity_rate": 0.0, "clob_depth_trend": 0.0,
            "clob_ask_pressure": 0.0,
        }

        w_books = book_rows[(book_rows["t_sec"] >= 0) & (book_rows["t_sec"] < OBS_SECS)]
        w_pcs   = pc_rows[(pc_rows["t_sec"] >= 0) & (pc_rows["t_sec"] < OBS_SECS)] if pc_rows is not None and len(pc_rows) > 0 else pd.DataFrame()

        if len(w_books) + len(w_pcs) < 3:
            feats.update(zeros_clob)
            return feats

        real_imb, real_depth, real_ts = [], [], []
        all_mid, all_spread, all_ts   = [], [], []
        ask_prices_seq = []

        for _, r in w_books.iterrows():
            s = _extract_snap(r)
            if s is None:
                continue
            real_imb.append(s["imb"])
            real_depth.append(s["total_depth"])
            real_ts.append(float(r["t_sec"]))
            all_mid.append(s["mid"])
            all_spread.append(s["spread"])
            all_ts.append(float(r["t_sec"]))

        for _, r in w_pcs.iterrows():
            try:
                ba = float(r["best_ask"]) if pd.notna(r.get("best_ask")) else None
                bb = float(r["best_bid"]) if pd.notna(r.get("best_bid")) else None
                if ba is not None and bb is not None and ba > bb > 0:
                    all_mid.append((ba + bb) / 2)
                    all_spread.append(ba - bb)
                    all_ts.append(float(r["t_sec"]))
                side = str(r.get("pc_side", "")).upper()
                if side == "SELL" and pd.notna(r.get("pc_price")):
                    ask_prices_seq.append(float(r["pc_price"]))
            except Exception:
                continue

        if len(all_mid) < 2:
            feats.update(zeros_clob)
            return feats

        order = np.argsort(all_ts)
        ts_arr  = np.array(all_ts)[order]
        mid_arr = np.array(all_mid)[order]
        sp_arr  = np.array(all_spread)[order]
        t_rel   = ts_arr - ts_arr[0]

        clob = {}
        clob["clob_imb_mean"]      = float(np.mean(real_imb))           if real_imb          else 0.0
        clob["clob_imb_std"]       = float(np.std(real_imb))            if len(real_imb) > 1 else 0.0
        clob["clob_imb_drift"]     = float(real_imb[-1] - real_imb[0]) if len(real_imb) > 1 else 0.0
        clob["clob_spread_mean"]   = float(np.mean(sp_arr))
        clob["clob_spread_trend"]  = _linslope(t_rel, sp_arr)
        clob["clob_mid_velocity"]  = _linslope(t_rel, mid_arr)
        mid_diffs = np.diff(mid_arr)
        clob["clob_mid_volatility"]  = float(np.std(mid_diffs)) if len(mid_diffs) > 0 else 0.0
        clob["clob_activity_rate"]   = float((len(w_books) + len(w_pcs)) / OBS_SECS)
        clob["clob_depth_trend"]     = _linslope(np.array(real_ts) - (real_ts[0] if real_ts else 0), np.array(real_depth)) if len(real_depth) > 1 else 0.0
        if len(ask_prices_seq) >= 2:
            diffs = np.diff(ask_prices_seq)
            moves = diffs[diffs != 0]
            clob["clob_ask_pressure"] = float((moves < 0).sum() / len(moves)) if len(moves) > 0 else 0.0
        else:
            clob["clob_ask_pressure"] = 0.0

        feats.update(clob)
        return feats

    def _fetch_one(market_id: str, slug: str, slot_ts: int) -> tuple[str, dict | None]:
        import logging as _log_module
        _log = _log_module.getLogger(__name__)
        try:
            url = f"{BASE_URL}/{slug}.parquet"
            r   = requests.get(url, headers=HEADERS, timeout=60)
            if not r.ok:
                return (market_id, None)

            df = pd.read_parquet(io.BytesIO(r.content))

            if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["ts_sec"] = df["timestamp"].astype("int64") / 1000.0
            else:
                df["ts_sec"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)
            df["t_sec"] = df["ts_sec"] - slot_ts

            # Só janela t=[0, 60s)
            df = df[(df["t_sec"] >= 0) & (df["t_sec"] < OBS_SECS)]

            books = df[df["event_type"] == "book"].copy()
            pcs   = df[df["event_type"] == "price_change"].copy()

            feats = compute_features(
                books if len(books) > 0 else None,
                pcs   if len(pcs)   > 0 else None,
            )
            if feats is not None:
                feats["market_id"] = market_id
            return (market_id, feats)
        except Exception as e:
            log.debug("fetch_one failed for %s: %s", market_id, e)
            return (market_id, None)

    # ── Fetch paralelo ────────────────────────────────────────────────────
    new_rows = []
    success  = 0
    failed   = 0
    WORKERS  = 20

    log.info("Iniciando fetch de %d mercados (workers=%d)...", len(todo), WORKERS)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(_fetch_one, r["market_id"], r["slug"], r["slot_ts"]): r["market_id"]
            for _, r in todo.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            mid, feats = fut.result()
            if feats is not None:
                new_rows.append(feats)
                success += 1
            else:
                failed += 1
            if i % 20 == 0 or i == len(todo):
                log.info("  %d/%d — ok=%d fail=%d", i, len(todo), success, failed)

    log.info("Fetch completo: %d ok, %d falhou", success, failed)

    if not new_rows:
        log.warning("Nenhuma feature nova — abortando upload")
        return {"success": 0, "failed": failed}

    # ── Merge com OB existente ────────────────────────────────────────────
    new_df = pd.DataFrame(new_rows)
    new_df["market_id"] = new_df["market_id"].astype(str)

    if existing_ob is not None:
        # Garantir schema compatível — adicionar colunas faltantes
        for col in existing_ob.columns:
            if col not in new_df.columns:
                new_df[col] = 0.0
        for col in new_df.columns:
            if col not in existing_ob.columns:
                existing_ob[col] = 0.0
        merged = pd.concat([existing_ob, new_df], ignore_index=True)
    else:
        merged = new_df

    # Dedup por market_id — manter a versão mais recente (nova)
    merged = merged.drop_duplicates(subset=["market_id"], keep="last").reset_index(drop=True)
    log.info("OB merged: %d mercados totais", len(merged))

    # ── Salvar no volume local ────────────────────────────────────────────
    out_local = LOCAL_DIR / "ob_features_full.parquet"
    merged.to_parquet(out_local, index=False)
    LOCAL_VOL.commit()
    log.info("Salvo em volume local: %s (%d rows)", out_local, len(merged))

    # ── Upload para HF (single source of truth) ───────────────────────────
    log.info("Uploading ob_features_full.parquet para HF...")
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(out_local),
        path_in_repo="data/ob_features_full.parquet",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message=f"data: add {success} missing OB features (v29, t<60s)",
    )
    log.info("Upload HF completo!")

    return {
        "success":      success,
        "failed":       failed,
        "total_merged": len(merged),
    }


@app.local_entrypoint()
def main():
    result = fetch_ob_missing.remote()
    print("\nRESULT:", result)
