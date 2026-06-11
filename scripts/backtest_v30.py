"""
backtest_v30.py — OOS backtest do candidate v30 (15 features, 60s obs, sem leakage, -ob_total_depth)
====================================================================================
Princípios:
  - Features computadas IDENTICAMENTE ao live_trader.py — zero divergência
  - Nenhuma feature usa dados além de t < OBS_SECS (60s) do slot
  - Preço de entrada simulado via ob_imbalance do snapshot "open" (t~5-10s) + spread fixo
    (ob_mid foi removido por leakage — capturado em t=108-168s no training)
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
def backtest_v30():
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
        "candidate.pkl":                 "candidate.pkl",
    }
    # Files that must always be fresh (never use stale cache)
    NO_CACHE = {"candidate.pkl", "holdout_markets.csv", "holdout_ob_features.parquet"}

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
    with open(DATA_DIR / "candidate.pkl", "rb") as f:
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

    def spot_volatility_log(t_start, t_end):
        """Std dos log-retornos de 1m (usado em btc_vol_1h)."""
        px, *_ = spot_range(t_start, t_end)
        if len(px) < 2:
            return 0.0
        return float(np.std(np.diff(np.log(px + 1e-9))))

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

    # ── Histórico global de up_ratio por slot (para lags e zscores) ──────
    # Usa ticks de TREINO (pre-cutoff) para construir série histórica
    # Espelha o live_trader que tem histórico acumulado de slots passados
    all_ticks_dfs = []
    pf2 = pq.ParquetFile(str(DATA_DIR / "ticks_btc_full_clean.parquet"))
    for batch in pf2.iter_batches(batch_size=500_000, columns=["market_id", "timestamp_ms", "outcome", "size_usdc"]):
        all_ticks_dfs.append(batch.to_pandas())
    new_tick_path = DATA_DIR / "new_ticks_pmdata.parquet"
    if new_tick_path.exists():
        nt2 = pd.read_parquet(str(new_tick_path), columns=["market_id", "timestamp_ms", "outcome", "size_usdc"])
        all_ticks_dfs.append(nt2)
    all_ticks = pd.concat(all_ticks_dfs, ignore_index=True)
    all_ticks["t_sec"] = all_ticks["timestamp_ms"] // 1000
    all_ticks["slot_mid"] = (all_ticks["t_sec"] // 300) * 300 + 150

    slot_vol_up  = all_ticks[all_ticks["outcome"] == "Up"].groupby("slot_mid")["size_usdc"].sum()
    slot_vol_tot = all_ticks.groupby("slot_mid")["size_usdc"].sum()
    slot_up_ratio_global = (slot_vol_up / slot_vol_tot.clip(lower=1e-9)).to_dict()
    del all_ticks, all_ticks_dfs
    gc.collect()
    log.info("Global slot history: %d slots", len(slot_up_ratio_global))

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

        # ── A. SPOT features (idêntico ao live_trader.build_spot_features) ──
        px_arr, hi_arr, lo_arr, vol_arr = spot_range(slot_ts, obs_end_ts)
        px_obs_end = float(px_arr[-1]) if len(px_arr) > 0 else spot_at(obs_end_ts)

        # btc_inslot_ret / range
        if len(px_arr) >= 2 and px_arr[0] > 0:
            feat["btc_inslot_ret"]   = float(px_arr[-1] / px_arr[0] - 1)
            feat["btc_inslot_range"] = float((hi_arr.max() - lo_arr.min()) / px_obs_end) if px_obs_end > 0 else 0.0
        else:
            feat["btc_inslot_ret"] = feat["btc_inslot_range"] = 0.0

        # btc_pre_*_ret (vs ponto obs_end)
        def pre_ret(h_sec):
            px_h = spot_at(slot_ts - h_sec)
            return float(px_obs_end / px_h - 1) if px_h > 0 else 0.0

        feat["btc_pre_5m_ret"]  = pre_ret(300)
        feat["btc_pre_15m_ret"] = pre_ret(900)
        feat["btc_pre_30m_ret"] = pre_ret(1800)
        feat["btc_pre_1h_ret"]  = pre_ret(3600)

        # btc_dist_1k (distância ao múltiplo de $1k mais próximo)
        if px_obs_end > 0:
            px_k = px_obs_end / 1000.0
            feat["btc_dist_1k"] = float(min(px_k - math.floor(px_k), math.ceil(px_k) - px_k))
        else:
            feat["btc_dist_1k"] = 0.5

        # btc_vol_1h (std log-returns última 1h)
        feat["btc_vol_1h"] = spot_volatility_log(slot_ts - 3600, slot_ts)

        # btc_spot_vol_ratio = vol_5m / avg_vol_por_slot_1h
        # Espelha live_trader: vol de trades Binance no slot vs média dos slots da última 1h
        vol_5m = float(np.sum(vol_arr)) if len(vol_arr) > 0 else 0.0
        # Média de volume por janela de 5min (5 candles 1m) na última hora (12 janelas)
        vol_windows = []
        for i in range(1, 13):
            _, _, _, v_w = spot_range(slot_ts - i * 300, slot_ts - (i - 1) * 300)
            if len(v_w) > 0:
                vol_windows.append(float(np.sum(v_w)))
        avg_vol_1h = float(np.mean(vol_windows)) if vol_windows else vol_5m
        feat["btc_spot_vol_ratio"] = float(vol_5m / (avg_vol_1h + 1e-9))

        # ── B. TICK features (idêntico ao live_trader) ────────────────────
        ticks = tick_by_market.get(mid, [])
        obs_ticks = [t for t in ticks if slot_ts <= t["t_sec"] < obs_end_ts]

        if obs_ticks:
            sizes    = [t["size_usdc"] for t in obs_ticks]
            outcomes = [t.get("outcome", "") for t in obs_ticks]
            sides    = [t.get("side", "") for t in obs_ticks]

            vol_up  = sum(s for s, o in zip(sizes, outcomes) if o == "Up")
            vol_dn  = sum(s for s, o in zip(sizes, outcomes) if o == "Down")
            total   = vol_up + vol_dn

            cur_up_ratio = float(vol_up / total) if total > 0 else 0.5
            feat["btc_up_ratio"]  = cur_up_ratio
            feat["btc_n_ticks"]   = float(len(obs_ticks))
            feat["btc_momentum"]  = cur_up_ratio - 0.5
            feat["btc_buy_ratio"] = float(sum(1 for s in sides if s == "BUY") / max(len(obs_ticks), 1))

            # btc_size_disparity = abs(mean_buy - mean_sell) / mean_all
            # IDÊNTICO ao live_trader (não é std — foi corrigido no v29)
            buy_sizes  = [s for s, sd in zip(sizes, sides) if sd == "BUY"]
            sell_sizes = [s for s, sd in zip(sizes, sides) if sd == "SELL"]
            mean_buy   = float(np.mean(buy_sizes))  if buy_sizes  else 0.0
            mean_sell  = float(np.mean(sell_sizes)) if sell_sizes else 0.0
            mean_all   = float(np.mean(sizes))      if sizes      else 1.0
            feat["btc_size_disparity"] = float(abs(mean_buy - mean_sell) / (mean_all + 1e-9))

            # btc_up_ratio_stability
            w_vals = []
            for t in obs_ticks:
                sub = [x for x in obs_ticks if abs(x["t_sec"] - t["t_sec"]) <= 5]
                if sub:
                    vu2 = sum(x["size_usdc"] for x in sub if x.get("outcome") == "Up")
                    vt2 = sum(x["size_usdc"] for x in sub) or 1
                    w_vals.append(vu2 / vt2)
            feat["btc_up_ratio_stability"] = float(np.std(w_vals)) if w_vals else 0.0

            # btc_tw_up_ratio (time+volume weighted)
            sz_arr2   = np.array(sizes)
            up_arr2   = np.array([1.0 if o == "Up" else 0.0 for o in outcomes])
            w_exp     = np.exp(np.linspace(-1, 0, len(obs_ticks)))
            feat["btc_tw_up_ratio"] = float(np.sum(up_arr2 * sz_arr2 * w_exp) / (np.sum(sz_arr2 * w_exp) + 1e-9))

            # btc_up_w0, btc_up_w1 — time windows (idêntico ao live_trader)
            half = slot_ts + OBS_SECS / 2
            for label, w_start, w_end in [("w0", slot_ts, half), ("w1", half, obs_end_ts)]:
                sub = [t for t in obs_ticks if w_start <= t["t_sec"] < w_end]
                if sub:
                    vu = sum(t["size_usdc"] for t in sub if t.get("outcome") == "Up")
                    vd = sum(t["size_usdc"] for t in sub if t.get("outcome") == "Down")
                    vt = vu + vd
                    feat[f"btc_up_w{label[-1]}"] = float(vu / vt) if vt > 0 else 0.5
                else:
                    feat[f"btc_up_w{label[-1]}"] = 0.5

        else:
            cur_up_ratio = 0.5
            feat.update({
                "btc_up_ratio": 0.5, "btc_n_ticks": 0.0, "btc_momentum": 0.0,
                "btc_buy_ratio": 0.5, "btc_size_disparity": 0.0,
                "btc_up_ratio_stability": 0.0, "btc_tw_up_ratio": 0.5,
                "btc_up_w0": 0.5, "btc_up_w1": 0.5,
            })

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

        # ── E. LAG / PREV_SLOT features ───────────────────────────────────
        # Usa histórico global (não só OOS) para zscores honestos
        lag_history = []  # up_ratios dos últimos 20 slots
        for lag_n in range(1, 21):
            prev_mid_ts = slot_ts - lag_n * 300 + 150   # mid do slot lag
            v = slot_up_ratio_global.get(prev_mid_ts, None)
            lag_history.append(v if v is not None else 0.5)
            if lag_n <= 5:
                feat[f"prev_slot_up_ratio_{lag_n}"] = lag_history[-1]
                feat[f"lag_{lag_n}_outcome"]        = 0.5   # unknown no inference time
                feat[f"prev_slot_n_ticks_{lag_n}"]  = 0.0
                feat[f"prev_slot_vol_{lag_n}"]       = 0.0

        # Zscores com janela de 20 (idêntico ao live_trader)
        arr20 = np.array(lag_history)
        mu20  = float(arr20.mean())
        sd20  = float(arr20.std())
        feat["lag_ur_zscore_20"] = float(np.clip((lag_history[0] - mu20) / (sd20 + 1e-9), -5, 5))

        arr5 = arr20[:5]
        mu5  = float(arr5.mean())
        sd5  = float(arr5.std())
        feat["btc_up_ratio_zscore_5s"]  = float(np.clip((cur_up_ratio - mu5) / (sd5 + 1e-9), -5, 5))
        feat["btc_up_ratio_zscore_20s"] = float(np.clip((cur_up_ratio - mu20) / (sd20 + 1e-9), -5, 5))
        feat["lag_ur_zscore_5"]         = float(np.clip((lag_history[0] - mu5) / (sd5 + 1e-9), -5, 5))

        # lag_streak
        streak = 0
        for ur in lag_history[:5]:
            if (ur > 0.5) == (lag_history[0] > 0.5):
                streak += 1
            else:
                break
        feat["lag_streak"] = float(streak)

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

    # ── Step 9: Simular trades ────────────────────────────────────────────
    # Preço de entrada: ob_imbalance como proxy do mid do Polymarket
    # ob_imbalance = bid_vol / (bid_vol + ask_vol) ≈ probabilidade implícita
    # É capturado no snapshot "open" (t~5-10s) — dentro dos 60s, sem leakage
    # sim_ask = ob_imbalance + spread/2 (simulação conservadora)
    log.info("Step 9: Simulating trades...")

    # Se ob_imbalance não está disponível, fallback para 0.5
    if "ob_imbalance" in df_oos.columns:
        df_oos["sim_mid"] = df_oos["ob_imbalance"].clip(0.05, 0.95)
    else:
        df_oos["sim_mid"] = 0.5

    df_oos["sim_ask"]  = (df_oos["sim_mid"] + SPREAD_ASSUMED / 2).clip(0.01, 0.99)
    df_oos["edge_ask"] = df_oos["pred_up"] - df_oos["sim_ask"]
    df_oos["edge_mid"] = df_oos["pred_up"] - df_oos["sim_mid"]

    mask = (
        (df_oos["pred_up"] >= MIN_CONFIDENCE) &
        (df_oos["sim_ask"] >= ASK_LO) &
        (df_oos["sim_ask"] <= ASK_HI) &
        (df_oos["edge_ask"] >= MIN_EDGE) &
        (df_oos["edge_mid"] >= MIN_EDGE_MID)
    )
    df_trades = df_oos[mask].copy()
    log.info("Trades: %d / %d (%.1f%%)",
             len(df_trades), len(df_oos), len(df_trades) / max(len(df_oos), 1) * 100)

    if len(df_trades) == 0:
        log.warning("Nenhum trade passou os filtros.")
        log.info("Pred dist: min=%.3f max=%.3f mean=%.3f", proba.min(), proba.max(), proba.mean())
        log.info("Ask dist: min=%.3f max=%.3f mean=%.3f",
                 df_oos["sim_ask"].min(), df_oos["sim_ask"].max(), df_oos["sim_ask"].mean())
        return {"n_trades": 0, "oos_auc": round(auc_oos, 4)}

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
    result = backtest_v30.remote()
    print("\nRESULT:", result)
