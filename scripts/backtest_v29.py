"""
backtest_v29.py — OOS backtest do champion v29 (20 features, 60s obs, sem leakage)
====================================================================================
Princípios:
  - Features computadas IDENTICAMENTE ao train_v29_modal.py — fonte da verdade
    (size_disparity/vol_1h/up_ratio/lag por rank/zscores — paridade exata)
  - Nenhuma feature usa dados além de t < OBS_SECS (60s) do slot
  - Preço de entrada (fill) REAL de holdout_ticks.parquet — trade UP perto de
    t=170s (BUY=ask; senão mid+meio-spread). Sem trade UP utilizável → pula.
  - Walk-forward 90/10 (por slot_ts, não aleatório)
  - Métricas completas: AUC, Brier, Acc, Win Rate, P&L, ROI, Sharpe
  - Features lidas dinamicamente do bundle — backtest sempre espelha o modelo atual
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
        "huggingface_hub>=0.26",
    )
)

vol = modal.Volume.from_name("btc-data-cache", create_if_missing=True)
app = modal.App("btc-v29-backtest", image=image)


@app.function(
    cpu=8,
    memory=32768,
    timeout=3600,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/cache": vol},
)
def backtest_v29():
    import gc, json, logging, math, os, pickle, shutil, sys, time, warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    HF_TOKEN       = os.environ.get("HF_TOKEN", "")
    HF_MODEL_REPO  = "artbreguez/polymarket-btc-model"

    # ── Live trader filters v29 (idêntico ao live_trader.py) ─────────────
    MIN_CONFIDENCE  = 0.55
    MIN_EDGE        = 0.07
    MIN_EDGE_MID    = 0.05
    ASK_LO, ASK_HI = 0.42, 0.65
    TAKER_FEE       = 0.02
    MIN_SHARES      = 5
    MAX_SHARES      = 40
    FLOOR           = 20.0
    CEIL            = 700.0
    OBS_SECS        = 60
    SPREAD_ASSUMED  = 0.02   # spread fixo para simulação de ask (sem ob_mid no v29)

    DATA_DIR = Path("/cache")

    # ── Step 1: Download dados e modelo ──────────────────────────────────
    log.info("Step 1: Downloading data + model from HF...")
    FILES = {
        "all_markets.csv":               "data/all_markets.csv",
        "new_markets.csv":               "data/new_markets.csv",
        "ticks_btc_full_clean.parquet":  "data/ticks_btc_full_clean.parquet",
        "new_ticks_pmdata.parquet":      "data/new_ticks_pmdata.parquet",
        "binance_spot_full.parquet":     "data/binance_spot_full.parquet",
        "binance_spot_local.parquet":    "data/binance_spot_local.parquet",
        "ob_features_full.parquet":      "data/ob_features_full.parquet",
        "holdout_markets.csv":           "data/holdout_markets.csv",
        "holdout_ob_features.parquet":   "data/holdout_ob_features.parquet",
        "holdout_ticks.parquet":         "data/holdout_ticks.parquet",
        "champion.pkl":                  "champion.pkl",
    }
    # Files that must always be fresh (never use stale cache)
    NO_CACHE = {"champion.pkl", "holdout_markets.csv",
                "holdout_ob_features.parquet", "holdout_ticks.parquet"}

    for local_name, hf_path in FILES.items():
        local_path = DATA_DIR / local_name
        if local_path.exists() and local_name not in NO_CACHE:
            log.info("  %s cached", local_name)
            continue
        if local_path.exists() and local_name in NO_CACHE:
            local_path.unlink()
            log.info("  %s: forcing fresh download (no-cache)", local_name)
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO, filename=hf_path,
                token=HF_TOKEN, repo_type="model",
                local_dir=str(DATA_DIR), local_dir_use_symlinks=False,
            )
            src = DATA_DIR / hf_path
            if src.exists() and not local_path.exists():
                shutil.move(str(src), str(local_path))
            log.info("  Downloaded %s", local_name)
        except Exception as e:
            log.warning("  Could not download %s: %s", local_name, e)

    # ── Step 2: Carregar modelo ───────────────────────────────────────────
    log.info("Step 2: Loading champion model...")
    with open(DATA_DIR / "champion.pkl", "rb") as f:
        bundle = pickle.load(f)
    model    = bundle["model"]
    features = bundle["features"]   # 20 features do v29 — fonte de verdade
    version  = bundle.get("version", "unknown")
    wf_auc   = bundle.get("wf_auc", 0)
    log.info("Champion: %s | %d features | WF AUC=%.4f", version, len(features), wf_auc)
    log.info("Features: %s", features)

    # ── Step 3: Carregar holdout (mercados que o modelo NUNCA viu) ───────
    # Holdout = mercados com slot_ts > TRAIN_CUTOFF do v29 (6 jun 19:10 UTC)
    # Ver docs/holdout_policy.md
    log.info("Step 3: Loading holdout markets (genuinely OOS)...")
    import datetime as _dt
    holdout_path = DATA_DIR / "holdout_markets.csv"
    if not holdout_path.exists():
        raise RuntimeError(
            "holdout_markets.csv nao encontrado. "
            "Rode fetch_holdout_pipeline.py primeiro."
        )
    markets = pd.read_csv(holdout_path)
    markets = markets[markets["target"].notna()].sort_values("slot_ts").reset_index(drop=True)
    log.info("Holdout: %d markets (%.1f%% UP) | %s → %s",
             len(markets), 100 * markets["target"].mean(),
             _dt.datetime.utcfromtimestamp(int(markets["slot_ts"].min())),
             _dt.datetime.utcfromtimestamp(int(markets["slot_ts"].max())))

    # ── Step 4: OB features do holdout ───────────────────────────────────
    log.info("Step 4: Loading holdout OB features...")
    ob_path = DATA_DIR / "holdout_ob_features.parquet"
    if not ob_path.exists():
        raise RuntimeError(
            "holdout_ob_features.parquet nao encontrado. "
            "Rode fetch_holdout_pipeline.py primeiro."
        )
    ob_df = pd.read_parquet(str(ob_path))

    # Features com leakage temporal (capturadas em t=108-168s no training)
    # Removidas explicitamente — mesmo que o modelo v29 não as use, mantemos
    # o backtest limpo para não contaminar futuros modelos
    OB_LEAKAGE = {
        "ob_mid",           # t=108-168s — principal fonte de leakage (corr=0.60)
        "ob_imbalance_end", # snapshot "end" = slot completo
        "ob_spread_end",    # idem
        "ob_depth_change",  # end - open = depende do end
        "ob_imb_momentum",  # end - open = depende do end
        "clob_mid_velocity",# slope calculado sobre slot completo no train
        "ob_weighted_imb",  # inclui dados pós-60s
        "ob_bid_depth_5c",  # snapshot contaminado
        "ob_ask_depth_5c",  # snapshot contaminado
        # outros campos não-feature
        "ob_mid_drift", "ob_imb_w0", "ob_imb_w1", "ob_imb_w2",
        "ob_pc_up_ratio", "ob_pc_volatility",
        "clob_imb_mean", "clob_imb_std", "clob_imb_drift",
        "clob_depth_trend", "clob_activity_rate",
    }
    ob_df = ob_df.drop(columns=[c for c in OB_LEAKAGE if c in ob_df.columns])
    ob_by_id = {str(row["market_id"]): row.to_dict() for _, row in ob_df.iterrows()}
    log.info("OB features loaded: %d markets | columns: %s",
             len(ob_df), [c for c in ob_df.columns if c != "market_id"])

    # Filtrar mercados com OB
    markets["market_id"] = markets["market_id"].astype(str)
    markets_oos = markets[markets["market_id"].isin(ob_by_id.keys())].reset_index(drop=True)
    log.info("Holdout with OB: %d / %d (%.1f%% UP)",
             len(markets_oos), len(markets), 100 * markets_oos["target"].mean())
    if len(markets_oos) < len(markets):
        log.warning("  %d holdout markets sem OB features (serão skipped)",
                    len(markets) - len(markets_oos))

    all_ids = set(markets_oos["market_id"].tolist())

    # ── Step 5: Spot BTC (Binance 1m candles) ────────────────────────────
    log.info("Step 5: Loading BTC spot data...")
    spot_dfs = []
    for sp in ["binance_spot_full.parquet", "binance_spot_local.parquet"]:
        sp_path = DATA_DIR / sp
        if sp_path.exists():
            spot_dfs.append(pd.read_parquet(str(sp_path)))
    spot = pd.concat(spot_dfs).sort_values("timestamp_ms").drop_duplicates("timestamp_ms").reset_index(drop=True)
    spot_ts_arr  = spot["timestamp_ms"].values // 1000   # segundos
    spot_px_arr  = spot["close"].values.astype(np.float64)
    spot_hi_arr  = spot["high"].values.astype(np.float64)
    spot_lo_arr  = spot["low"].values.astype(np.float64)
    spot_vol_arr = spot["volume"].values.astype(np.float64)  # volume BTC por candle 1m
    log.info("Spot data: %d candles", len(spot))

    def spot_at(ts_sec):
        """Preço BTC no timestamp (candle 1m mais recente)."""
        idx = int(np.searchsorted(spot_ts_arr, ts_sec, side="right")) - 1
        return float(spot_px_arr[idx]) if 0 <= idx < len(spot_px_arr) else 0.0

    def spot_range(t_start, t_end):
        """Retorna (px_arr, hi_arr, lo_arr, vol_arr) num intervalo."""
        i0 = int(np.searchsorted(spot_ts_arr, t_start, side="left"))
        i1 = int(np.searchsorted(spot_ts_arr, t_end, side="right"))
        return (spot_px_arr[i0:i1], spot_hi_arr[i0:i1],
                spot_lo_arr[i0:i1], spot_vol_arr[i0:i1])

    def spot_vol_simple(t_start, t_end):
        """Std dos retornos SIMPLES de 1m em [t_start, t_end) — paridade train_v29 l.315-323."""
        px, *_ = spot_range(t_start, t_end)
        if len(px) < 3:
            return 0.0
        rets = np.diff(px) / px[:-1]
        return float(np.std(rets))

    def spot_vol_sum(t_start, t_end):
        """Soma de volume 1m em [t_start, t_end) — paridade train_v29 (spot_vol_at)."""
        _, _, _, v = spot_range(t_start, t_end)
        return float(np.sum(v)) if len(v) > 0 else 0.0

    # ── Step 6: Ticks do holdout ──────────────────────────────────────────
    log.info("Step 6: Loading holdout ticks...")
    tick_cols = ["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"]
    holdout_ticks_path = DATA_DIR / "holdout_ticks.parquet"
    if holdout_ticks_path.exists():
        ticks_df = pd.read_parquet(str(holdout_ticks_path), columns=tick_cols)
        ticks_df = ticks_df[ticks_df["market_id"].isin(all_ids)]
    else:
        log.warning("holdout_ticks.parquet nao encontrado — sem ticks")
        ticks_df = pd.DataFrame(columns=tick_cols)

    ticks_df["t_sec"] = ticks_df["timestamp_ms"] // 1000
    tick_by_market = {}
    for mid, grp in ticks_df.groupby("market_id"):
        tick_by_market[mid] = grp[["t_sec", "outcome", "side", "price", "size_usdc"]].to_dict("records")
    log.info("Ticks loaded: %d markets with ticks", len(tick_by_market))

    # ── Histórico por RANK (paridade train_v29 l.170-177, 265-280) ───────
    # Lag é indexado por RANK na lista global de mercados ordenada por slot_ts
    # (treino + holdout), exatamente como o treino — NÃO por timestamp contíguo.
    # slot_up_ratio = vol-weighted Up/Down sobre ticks [0,60s), por market_id.
    # Só calculamos slot_up_ratio para: (a) mercados do holdout e (b) os ~25
    # mercados de treino imediatamente antes do cutoff (predecessores por rank
    # dos primeiros mercados de holdout). market_ids treino/holdout são disjuntos
    # → sem contaminação (não usamos ticks do holdout p/ mercados de treino).
    ho_min_ts = int(markets["slot_ts"].astype(int).min())
    train_mkts = pd.read_csv(DATA_DIR / "all_markets.csv")
    train_mkts["market_id"] = train_mkts["market_id"].astype(str)
    train_mkts["slot_ts"]   = train_mkts["slot_ts"].astype(int)
    nm_path = DATA_DIR / "new_markets.csv"
    if nm_path.exists():
        nm = pd.read_csv(nm_path)
        nm["market_id"] = nm["market_id"].astype(str)
        nm["slot_ts"]   = nm["slot_ts"].astype(int)
        train_mkts = pd.concat([train_mkts[["market_id", "slot_ts"]],
                                nm[["market_id", "slot_ts"]]], ignore_index=True)
    train_mkts = train_mkts.drop_duplicates("market_id")
    train_only = train_mkts[train_mkts["slot_ts"] < ho_min_ts]

    glob = pd.concat([
        train_only[["market_id", "slot_ts"]],
        markets[["market_id", "slot_ts"]].astype({"slot_ts": int}),
    ], ignore_index=True).drop_duplicates("market_id").sort_values("slot_ts").reset_index(drop=True)
    glob["market_id"] = glob["market_id"].astype(str)
    all_mids_g  = glob["market_id"].tolist()
    all_slot_g  = glob["slot_ts"].astype(int).tolist()
    rank_by_mid = {m: i for i, m in enumerate(all_mids_g)}

    # predecessores de treino (por rank) dos mercados de holdout
    ho_ids = set(markets["market_id"].astype(str))
    first_ho_rank = min((rank_by_mid[m] for m in ho_ids if m in rank_by_mid),
                        default=len(all_mids_g))
    boundary_train_mids = set(all_mids_g[max(0, first_ho_rank - 25):first_ho_rank]) - ho_ids

    def _slot_ur_from(df, slot_map):
        """vol-weighted Up/Down sobre ticks [0,60s) por market_id (train l.273-279)."""
        if df is None or len(df) == 0:
            return {}
        d = df[["market_id", "timestamp_ms", "outcome", "size_usdc"]].copy()
        d["market_id"] = d["market_id"].astype(str)
        d = d[d["market_id"].isin(slot_map.keys())]
        if len(d) == 0:
            return {}
        d["t_rel"] = d["timestamp_ms"].astype(float) / 1000.0 - d["market_id"].map(slot_map).astype(float)
        d = d[(d["t_rel"] >= 0) & (d["t_rel"] < OBS_SECS)]
        vu = d[d["outcome"] == "Up"].groupby("market_id")["size_usdc"].sum()
        vt = d.groupby("market_id")["size_usdc"].sum()
        vu = vu.reindex(vt.index, fill_value=0.0)
        return (vu / vt.clip(lower=1e-9)).to_dict()

    # (a) holdout — ticks_df = holdout_ticks já carregado no Step 6
    ho_slot_map = dict(zip(markets["market_id"].astype(str), markets["slot_ts"].astype(int)))
    slot_up_ratio = _slot_ur_from(ticks_df, ho_slot_map)

    # (b) predecessores de treino — ticks dos arquivos de treino, filtrados
    if boundary_train_mids:
        bt_full = dict(zip(train_only["market_id"].astype(str), train_only["slot_ts"].astype(int)))
        bt_slot_map = {m: bt_full[m] for m in boundary_train_mids if m in bt_full}
        bt_chunks = []
        pf2 = pq.ParquetFile(str(DATA_DIR / "ticks_btc_full_clean.parquet"))
        for batch in pf2.iter_batches(batch_size=500_000,
                                      columns=["market_id", "timestamp_ms", "outcome", "size_usdc"]):
            b = batch.to_pandas()
            b["market_id"] = b["market_id"].astype(str)
            b = b[b["market_id"].isin(boundary_train_mids)]
            if len(b):
                bt_chunks.append(b)
        ntp = DATA_DIR / "new_ticks_pmdata.parquet"
        if ntp.exists():
            nt = pd.read_parquet(str(ntp), columns=["market_id", "timestamp_ms", "outcome", "size_usdc"])
            nt["market_id"] = nt["market_id"].astype(str)
            nt = nt[nt["market_id"].isin(boundary_train_mids)]
            if len(nt):
                bt_chunks.append(nt)
        if bt_chunks:
            slot_up_ratio.update(_slot_ur_from(pd.concat(bt_chunks, ignore_index=True), bt_slot_map))
    gc.collect()
    log.info("Rank history: %d mercados ranqueados | slot_up_ratio p/ %d mercados (%d boundary treino)",
             len(all_mids_g), len(slot_up_ratio), len(boundary_train_mids))

    # ── Step 7: Build features OOS ────────────────────────────────────────
    log.info("Step 7: Building features for %d OOS markets...", len(markets_oos))

    def _linear_slope(x, y):
        """Slope linear (replica de live_trader._linear_slope)."""
        if len(x) < 2:
            return 0.0
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        denom = np.sum((x - xm) ** 2)
        return float(np.sum((x - xm) * (y - ym)) / denom) if denom > 1e-12 else 0.0

    rows = []
    skipped = 0

    for _, row in markets_oos.iterrows():
        mid     = row["market_id"]
        slot_ts = int(row["slot_ts"])
        target  = int(row["target"])

        if mid not in ob_by_id:
            skipped += 1
            continue

        ob   = ob_by_id[mid]
        feat = {}

        obs_end_ts = slot_ts + OBS_SECS   # t=60s — limite rígido

        # ── A. SPOT features (paridade train_v29 l.336-379) ───────────────
        px_arr, hi_arr, lo_arr, vol_arr = spot_range(slot_ts, obs_end_ts)
        px_now = spot_at(obs_end_ts)   # preço em t=60s (train l.338)

        # btc_inslot_ret / range
        if len(px_arr) >= 2 and px_arr[0] > 0:
            feat["btc_inslot_ret"]   = float(px_arr[-1] / px_arr[0] - 1)
            feat["btc_inslot_range"] = float((hi_arr.max() - lo_arr.min()) / px_now) if px_now > 0 else 0.0
        else:
            feat["btc_inslot_ret"] = feat["btc_inslot_range"] = 0.0

        # btc_pre_*_ret (vs px_now em t=60s) — train l.340-347
        def pre_ret(h_sec):
            px_h = spot_at(slot_ts - h_sec)
            return float(px_now / px_h - 1) if px_h > 0 else 0.0

        feat["btc_pre_5m_ret"]  = pre_ret(300)
        feat["btc_pre_15m_ret"] = pre_ret(900)
        feat["btc_pre_30m_ret"] = pre_ret(1800)
        feat["btc_pre_1h_ret"]  = pre_ret(3600)

        # btc_dist_1k — train l.370-374
        if px_now > 0:
            px_k = px_now / 1000.0
            feat["btc_dist_1k"] = float(min(px_k - math.floor(px_k), math.ceil(px_k) - px_k))
        else:
            feat["btc_dist_1k"] = 0.5

        # btc_vol_1h — std de retornos SIMPLES de 1m na última 1h (train l.315-323, 365)
        feat["btc_vol_1h"] = spot_vol_simple(slot_ts - 3600, slot_ts)

        # btc_spot_vol_ratio — vol[t-300,t] / (vol[t-3600,t-300]/11) (train l.377-379)
        feat["btc_spot_vol_ratio"] = float(
            spot_vol_sum(slot_ts - 300, slot_ts) /
            (spot_vol_sum(slot_ts - 3600, slot_ts - 300) / 11 + 1e-9)
        )

        # ── B. TICK features (paridade train_v29 l.406-457) ───────────────
        ticks = tick_by_market.get(mid, [])
        # janela [0,60s): t_sec é absoluto (timestamp//1000), slot em [slot_ts, slot_ts+60)
        obs_ticks = [t for t in ticks if slot_ts <= t["t_sec"] < obs_end_ts]

        if obs_ticks:
            up_tks = [t for t in obs_ticks if t.get("outcome") == "Up"]
            dn_tks = [t for t in obs_ticks if t.get("outcome") == "Down"]
            vol_up = sum(t["size_usdc"] for t in up_tks)
            vol_dn = sum(t["size_usdc"] for t in dn_tks)
            total  = vol_up + vol_dn + 1e-8
            cur_up_ratio = float(vol_up / total)                     # train l.440

            # 2x30s sub-janelas → btc_up_w0 / btc_up_w1 (train l.426-429)
            sw = {}
            for i in range(2):
                w0, w1 = slot_ts + i * 30, slot_ts + (i + 1) * 30
                sub = [t for t in obs_ticks if w0 <= t["t_sec"] < w1]
                if sub:
                    vu = sum(t["size_usdc"] for t in sub if t.get("outcome") == "Up")
                    vt = sum(t["size_usdc"] for t in sub) + 1e-8
                    sw[f"btc_up_w{i}"] = float(vu / vt)
                else:
                    sw[f"btc_up_w{i}"] = 0.5
            feat["btc_up_w0"] = sw["btc_up_w0"]
            feat["btc_up_w1"] = sw["btc_up_w1"]

            feat["btc_up_ratio"]  = cur_up_ratio
            feat["btc_n_ticks"]   = float(len(obs_ticks))
            feat["btc_buy_ratio"] = float(sum(t["size_usdc"] for t in obs_ticks if t.get("side") == "BUY") / total)  # train l.445
            feat["btc_momentum"]  = float(sw["btc_up_w1"] - sw["btc_up_w0"])                # train l.432 (w1 - w0)
            feat["btc_up_ratio_stability"] = float(np.std([sw["btc_up_w0"], sw["btc_up_w1"]]))  # train l.434

            # btc_size_disparity = avg_up_sz - avg_dn_sz, por OUTCOME, SIGNED (train l.436-438)
            avg_up_sz = float(sum(t["size_usdc"] for t in up_tks) / (len(up_tks) + 1e-8))
            avg_dn_sz = float(sum(t["size_usdc"] for t in dn_tks) / (len(dn_tks) + 1e-8))
            feat["btc_size_disparity"] = float(avg_up_sz - avg_dn_sz)

            # btc_tw_up_ratio — exp recency decay sobre t relativo (train l.452-457)
            t_arr  = np.array([t["t_sec"] - slot_ts for t in obs_ticks], dtype=np.float64)
            sz_arr = np.array([t.get("size_usdc", 1.0) for t in obs_ticks], dtype=np.float64)
            up_arr = np.array([1.0 if t.get("outcome") == "Up" else 0.0 for t in obs_ticks])
            w_exp  = np.exp(-0.02 * (OBS_SECS - t_arr))
            feat["btc_tw_up_ratio"] = float(np.sum(up_arr * sz_arr * w_exp) / (np.sum(sz_arr * w_exp) + 1e-9))
        else:
            cur_up_ratio = 0.5
            feat.update({
                "btc_up_ratio": 0.5, "btc_n_ticks": 0.0, "btc_momentum": 0.0,
                "btc_buy_ratio": 0.5, "btc_size_disparity": 0.0,
                "btc_up_ratio_stability": 0.0, "btc_tw_up_ratio": 0.5,
                "btc_up_w0": 0.5, "btc_up_w1": 0.5,
            })

        # ── Fill REAL: ask do token UP perto da entrada (t≈170s) ──────────
        # Honesto — preço de trade real, não o ob_imbalance fabricado.
        # Preferência: trade UP side=BUY (executa no ASK) em t_rel ∈ [150,290s].
        # Fallback: qualquer trade UP (≈mid) + meio-spread. Sem trade UP → NaN → pula.
        up_entry = [t for t in ticks if 150 <= (t["t_sec"] - slot_ts) <= 290 and t.get("outcome") == "Up"]
        buy_up   = [t for t in up_entry if t.get("side") == "BUY"]
        pool     = buy_up if buy_up else up_entry
        if pool:
            pool.sort(key=lambda t: abs((t["t_sec"] - slot_ts) - 170))
            px_fill = float(pool[0]["price"])
            if not buy_up:
                px_fill += SPREAD_ASSUMED / 2     # mid → ask aproximado
            feat["sim_ask_real"] = float(min(max(px_fill, 0.01), 0.99))
        else:
            feat["sim_ask_real"] = float("nan")

        # ── C. OB features — apenas snapshot "open" (sem leakage) ─────────
        # Inclui: ob_imbalance, ob_depth_ratio, ob_total_depth, ob_spread
        # Exclui: ob_mid, ob_imbalance_end, ob_spread_end, ob_depth_change, etc
        OB_OK = {"ob_imbalance", "ob_depth_ratio", "ob_total_depth", "ob_spread",
                  "ob_bid_depth_5c", "ob_ask_depth_5c"}  # os _5c excluídos por leakage mas mantemos estrutura
        for col, val in ob.items():
            if col == "market_id":
                continue
            if col in OB_LEAKAGE:
                continue  # pula features com leakage
            v = val
            feat[col] = float(v) if (v is not None and not (isinstance(v, float) and math.isnan(v))) else 0.0

        # ── D. CLOB features — já presentes no OB parquet ────────────────────
        # clob_spread_mean, clob_spread_trend, clob_mid_volatility, clob_ask_pressure
        # foram gravados pelo fetch_holdout_ob_modal.py e já carregados no Step C.
        # Não zeramos — seriam zerados incorretamente aqui.
        # (Nota: clob_mid_velocity ainda é leakage e está em OB_LEAKAGE — corretamente excluída)

        # ── E. LAG / PREV_SLOT features — por RANK (paridade train_v29 l.469-534) ──
        ext_rank = rank_by_mid.get(mid)
        for lag_n in range(1, 6):
            pr = (ext_rank - lag_n) if ext_rank is not None else -1
            if pr >= 0:
                prev_mid  = all_mids_g[pr]
                prev_slot = int(all_slot_g[pr])
                if (slot_ts - prev_slot) > lag_n * 300 * 3:                      # gap guard (train l.481)
                    feat[f"prev_slot_up_ratio_{lag_n}"] = 0.5
                else:
                    feat[f"prev_slot_up_ratio_{lag_n}"] = float(slot_up_ratio.get(prev_mid, 0.5))  # train l.489
            else:
                feat[f"prev_slot_up_ratio_{lag_n}"] = 0.0                        # train l.500
            # não usadas pelas 20 (mantidas neutras p/ compatibilidade)
            feat[f"lag_{lag_n}_outcome"]       = 0.5
            feat[f"prev_slot_n_ticks_{lag_n}"] = 0.0
            feat[f"prev_slot_vol_{lag_n}"]      = 0.0
        feat["lag_streak"] = 0.0

        # Z-scores de up_ratio histórico por rank (train l.505-534)
        hist_vals = []
        if ext_rank is not None:
            for d in range(1, 21):
                pr = ext_rank - d
                if pr < 0:
                    break
                v = slot_up_ratio.get(all_mids_g[pr], None)
                if v is not None:
                    hist_vals.append(float(v))
        prev_ur_1 = feat.get("prev_slot_up_ratio_1", 0.5)
        if len(hist_vals) >= 3:
            mu20 = float(np.mean(hist_vals)); sd20 = max(float(np.std(hist_vals)), 0.01)
            feat["lag_ur_zscore_20"]        = float(np.clip((prev_ur_1 - mu20) / sd20, -5, 5))
            feat["btc_up_ratio_zscore_20s"] = float(np.clip((cur_up_ratio - mu20) / sd20, -5, 5))
        else:
            feat["lag_ur_zscore_20"] = 0.0
            feat["btc_up_ratio_zscore_20s"] = 0.0
        hist5 = hist_vals[:5]
        if len(hist5) >= 2:
            mu5 = float(np.mean(hist5)); sd5 = max(float(np.std(hist5)), 0.01)
            feat["lag_ur_zscore_5"]        = float(np.clip((prev_ur_1 - mu5) / sd5, -5, 5))
            feat["btc_up_ratio_zscore_5s"] = float(np.clip((cur_up_ratio - mu5) / sd5, -5, 5))
        else:
            feat["lag_ur_zscore_5"] = 0.0
            feat["btc_up_ratio_zscore_5s"] = 0.0

        # ── F. TEMPORAL features ──────────────────────────────────────────
        import datetime as dt_module
        dt_obj = dt_module.datetime.utcfromtimestamp(slot_ts)
        hour   = dt_obj.hour + dt_obj.minute / 60.0
        dow    = dt_obj.weekday()
        feat["hour_sin"]        = math.sin(2 * math.pi * hour / 24)
        feat["hour_cos"]        = math.cos(2 * math.pi * hour / 24)
        feat["dow_sin"]         = math.sin(2 * math.pi * dow / 7)
        feat["dow_cos"]         = math.cos(2 * math.pi * dow / 7)
        feat["hour_x_up_ratio"] = cur_up_ratio * (hour / 24.0)
        feat["hour_x_tw_ur"]    = feat.get("btc_tw_up_ratio", 0.5) * (hour / 24.0)

        # ── G. CROSS features (idêntico ao live_trader) ───────────────────
        feat["x_imb_x_ur"]       = feat.get("ob_imbalance", 0.0) * cur_up_ratio
        feat["x_depth_x_momentum"] = feat.get("ob_depth_ratio", 1.0) * feat.get("btc_momentum", 0.0)
        feat["x_imb_end_x_ret"]  = feat.get("ob_imbalance", 0.0) * feat.get("btc_inslot_ret", 0.0)
        feat["x_imb_x_inslot"]   = feat.get("ob_imbalance", 0.0) * feat.get("btc_inslot_ret", 0.0)
        feat["x_spread_x_vol"]   = feat.get("ob_spread", 0.02) * feat.get("btc_vol_1h", 0.0)
        feat["x_depth_x_vol"]    = feat.get("ob_depth_ratio", 1.0) * feat.get("btc_vol_1h", 0.0)

        feat["target"]    = target
        feat["slot_ts"]   = slot_ts
        feat["market_id"] = mid
        rows.append(feat)

    log.info("Built %d feature rows (skipped %d no-OB)", len(rows), skipped)

    df_oos = pd.DataFrame(rows)
    y_true = df_oos["target"].values

    # Garantir todas as features do modelo (fallback 0.0)
    missing = [f for f in features if f not in df_oos.columns]
    if missing:
        log.warning("Features missing from backtest (will be 0.0): %s", missing)
    for f in features:
        if f not in df_oos.columns:
            df_oos[f] = 0.0

    X_oos = df_oos[features].values.astype(np.float32)

    # ── Step 8: Predict ───────────────────────────────────────────────────
    log.info("Step 8: Predicting...")
    from sklearn.metrics import roc_auc_score, brier_score_loss

    proba = model.predict_proba(X_oos)[:, 1]
    df_oos["pred_up"] = proba

    auc_oos   = roc_auc_score(y_true, proba)
    brier_oos = brier_score_loss(y_true, proba)
    acc_oos   = float(((proba >= 0.5) == y_true).mean())
    log.info("OOS metrics: AUC=%.4f | Brier=%.4f | Acc=%.4f", auc_oos, brier_oos, acc_oos)

    # ── Step 9: Simular trades (fill REAL de holdout_ticks) ───────────────
    # sim_ask = preço real do trade UP perto da entrada (sim_ask_real, computado
    # no loop). Sem fill real → mercado é excluído da simulação (não inventa preço).
    # AUC/Brier/Acc (Step 8) já foram medidos em TODOS os mercados OOS.
    log.info("Step 9: Simulating trades (fill real)...")
    n_with_fill = int(df_oos["sim_ask_real"].notna().sum())
    log.info("Mercados com fill real (trade UP perto de t=170s): %d / %d", n_with_fill, len(df_oos))

    df_sim = df_oos[df_oos["sim_ask_real"].notna()].copy()
    if len(df_sim) == 0:
        log.warning("Nenhum mercado com fill real — sem simulação de trades.")
        return {"n_trades": 0, "oos_auc": round(auc_oos, 4), "n_with_fill": 0}

    df_sim["sim_ask"]  = df_sim["sim_ask_real"].clip(0.01, 0.99)
    df_sim["sim_mid"]  = (df_sim["sim_ask"] - SPREAD_ASSUMED / 2).clip(0.01, 0.99)
    df_sim["edge_ask"] = df_sim["pred_up"] - df_sim["sim_ask"]
    df_sim["edge_mid"] = df_sim["pred_up"] - df_sim["sim_mid"]

    mask = (
        (df_sim["pred_up"] >= MIN_CONFIDENCE) &
        (df_sim["sim_ask"] >= ASK_LO) &
        (df_sim["sim_ask"] <= ASK_HI) &
        (df_sim["edge_ask"] >= MIN_EDGE) &
        (df_sim["edge_mid"] >= MIN_EDGE_MID)
    )
    df_trades = df_sim[mask].copy()
    log.info("Trades: %d / %d com fill (%.1f%%)",
             len(df_trades), len(df_sim), len(df_trades) / max(len(df_sim), 1) * 100)

    if len(df_trades) == 0:
        log.warning("Nenhum trade passou os filtros.")
        log.info("Pred dist: min=%.3f max=%.3f mean=%.3f", proba.min(), proba.max(), proba.mean())
        log.info("Ask dist: min=%.3f max=%.3f mean=%.3f",
                 df_sim["sim_ask"].min(), df_sim["sim_ask"].max(), df_sim["sim_ask"].mean())
        return {"n_trades": 0, "oos_auc": round(auc_oos, 4), "n_with_fill": n_with_fill}

    results = []
    for _, row in df_trades.iterrows():
        ask      = row["sim_ask"]
        edge     = row["edge_ask"]
        actual_up = bool(row["target"])

        # Sizing idêntico ao live_trader (AUTO_SHARES)
        raw_shares = max(MIN_SHARES, min(MAX_SHARES, int(edge * 100)))
        cost = ask * raw_shares
        if cost < FLOOR:
            raw_shares = math.ceil(FLOOR / ask)
        elif cost > CEIL:
            raw_shares = math.floor(CEIL / ask)
        shares = max(MIN_SHARES, min(MAX_SHARES, raw_shares))
        cost   = ask * shares

        pnl = shares * (1.0 - ask) - shares * TAKER_FEE if actual_up else -cost
        results.append({
            "slot_ts": row["slot_ts"],
            "conf": row["pred_up"], "ask": ask, "edge": edge,
            "shares": shares, "cost": cost,
            "actual_up": actual_up, "pnl": pnl,
        })

    df_res = pd.DataFrame(results).sort_values("slot_ts")

    wins       = (df_res["pnl"] > 0).sum()
    total      = len(df_res)
    win_rate   = wins / total * 100
    total_pnl  = df_res["pnl"].sum()
    avg_pnl    = df_res["pnl"].mean()
    total_cost = df_res["cost"].sum()
    roi        = total_pnl / total_cost * 100 if total_cost > 0 else 0

    # Sharpe (diário — agrupa P&L por dia)
    df_res["date"] = pd.to_datetime(df_res["slot_ts"], unit="s").dt.date
    daily_pnl = df_res.groupby("date")["pnl"].sum()
    sharpe = float(daily_pnl.mean() / (daily_pnl.std() + 1e-9)) * math.sqrt(252) if len(daily_pnl) > 1 else 0.0

    # Max drawdown
    cumulative = df_res["pnl"].cumsum().values
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = float(drawdown.min())

    log.info("=" * 65)
    log.info("BACKTEST v29 — OOS (últimos 10%% | %d mercados)", len(df_oos))
    log.info("=" * 65)
    log.info("Modelo:      %s | %d features", version, len(features))
    log.info("WF AUC:      %.4f  (treino)", wf_auc)
    log.info("OOS AUC:     %.4f", auc_oos)
    log.info("OOS Brier:   %.4f", brier_oos)
    log.info("OOS Acc:     %.4f", acc_oos)
    log.info("CLOB feats:  presentes no OB parquet (clob_spread_mean/trend/mid_vol/ask_pressure)")
    log.info("---")
    log.info("Trades:      %d  (%d wins / %d losses)", total, wins, total - wins)
    log.info("Win rate:    %.1f%%", win_rate)
    log.info("Total P&L:   $%.2f", total_pnl)
    log.info("Avg P&L:     $%.2f / trade", avg_pnl)
    log.info("Total cost:  $%.2f deployed", total_cost)
    log.info("ROI:         %.1f%%", roi)
    log.info("Sharpe:      %.2f (anualizado)", sharpe)
    log.info("Max DD:      $%.2f", max_dd)
    log.info("Avg conf:    %.3f", df_res["conf"].mean())
    log.info("Avg ask:     %.3f", df_res["ask"].mean())
    log.info("Avg edge:    %.3f", df_res["edge"].mean())
    log.info("=" * 65)

    return {
        "version":   version,
        "n_features": len(features),
        "wf_auc":    wf_auc,
        "oos_auc":   round(auc_oos, 4),
        "oos_brier": round(brier_oos, 4),
        "oos_acc":   round(float(acc_oos), 4),
        "n_oos":     len(df_oos),
        "n_trades":  total,
        "win_rate":  round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl":   round(avg_pnl, 2),
        "roi":       round(roi, 1),
        "sharpe":    round(sharpe, 2),
        "max_dd":    round(max_dd, 2),
    }


@app.local_entrypoint()
def main():
    result = backtest_v29.remote()
    print("\nRESULT:", result)
