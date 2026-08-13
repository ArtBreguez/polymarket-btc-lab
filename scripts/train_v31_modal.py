"""
train_v31_modal.py — BTC 5min model v31 (ZERO LOOKAHEAD + WS-PARITY)
======================================================================
Filosofia: APENAS features computáveis em tempo real via WS em t<60s.
  - Binance kline WS: price returns, vol, range, dist round numbers
  - CLOB L2 OB REST snapshot (t~5s): imbalance, spread, depth
  - CLOB WS price_change (ob60_*): spread/mid dynamics, fill imbalance
  - CLOB WS last_trade_price: up_ratio, buy_ratio, momentum, vwap
  - Lag history (ring buffer de slots anteriores)
  - Temporais + cross-features

EXCLUÍDOS por lookahead ou impossibilidade WS:
  - ob180_* features (requerem dados t=60-180s — além de OBS_SECS=60)
  - ob_mid_drift, ob_imbalance_end, ob_spread_end (snapshot t>60s)
  - ob_imb_w1, ob_imb_w2 (book real t=60-180s — WS só entrega no subscribe)
  - clob_imb_mean/std/drift, clob_depth_trend (múltiplos book reais)
  - clob_activity_rate (mismatch train vs live)

Pipeline:
  1. Download HF: markets, ticks, spot, ob_features_v31
  2. Build 71 features (grupos A-F do features.md)
  3. Noise canary + adversarial validation (anti-overfitting)
  4. Purged WF CV (gap=5) feature screening
  5. Optuna 200 trials com variance penalty
  6. Walk-forward evaluation (5-fold, gap=5)
  7. Promote se bate champion AUC
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
        "optuna>=3.6",
        "huggingface_hub>=0.26",
        "scipy>=1.13",
    )
)

vol = modal.Volume.from_name("btc-training-cache", create_if_missing=True)
app = modal.App("btc-v31-run", image=image)


@app.function(
    cpu=8,
    memory=32768,
    timeout=10800,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/cache": vol},
)
def train_v31():
    import gc, json, logging, math, os, pickle, sys, time, warnings, shutil
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np
    import optuna
    import pandas as pd
    import pyarrow.parquet as pq
    from scipy import stats
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
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
    HF_DATA_REPO  = "artbreguez/polymarket-btc-updown"
    SLOT_DURATION = 300
    OBS_SECS      = 60
    OPTUNA_TRIALS = 200
    WF_GAP        = 5
    N_SPLITS      = 5
    DATA_DIR      = Path("/tmp/btc_data")
    DATA_DIR.mkdir(exist_ok=True)

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN required")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 0: DOWNLOAD DATA
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 0: Downloading training data from HuggingFace...")

    MODEL_FILES = {
        "all_markets.csv":            "data/all_markets.csv",
        "binance_spot_full.parquet":  "data/binance_spot_full.parquet",
        "binance_spot_local.parquet": "data/binance_spot_local.parquet",
    }
    # ob_features_v31 fica separado — deve ser upado via upload_data_to_hf.py antes do treino
    OB_FILES = {
        "ob_features_v31.parquet": "data/ob_features_v31.parquet",
    }
    for local_name, hf_path in MODEL_FILES.items():
        local_path = DATA_DIR / local_name
        if local_path.exists():
            log.info("  %s already cached", local_name)
            continue
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO, filename=hf_path,
                token=HF_TOKEN, repo_type="model",
                local_dir=str(DATA_DIR), local_dir_use_symlinks=False,
            )
            src = DATA_DIR / hf_path
            if src.exists() and not local_path.exists():
                shutil.move(str(src), str(local_path))
            log.info("  Downloaded %s (%.1fMB)", local_name, local_path.stat().st_size / 1e6)
        except Exception as e:
            log.warning("  Could not download %s: %s", local_name, e)

    for local_name, hf_path in OB_FILES.items():
        local_path = DATA_DIR / local_name
        if local_path.exists():
            log.info("  %s already cached", local_name)
            continue
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO, filename=hf_path,
                token=HF_TOKEN, repo_type="model",
                local_dir=str(DATA_DIR), local_dir_use_symlinks=False,
            )
            src = DATA_DIR / hf_path
            if src.exists() and not local_path.exists():
                shutil.move(str(src), str(local_path))
            log.info("  Downloaded %s (%.1fMB)", local_name, local_path.stat().st_size / 1e6)
        except Exception as e:
            raise RuntimeError(f"ob_features_v31.parquet nao encontrado no HF! Rode upload_data_to_hf.py primeiro. Erro: {e}")

    # Ticks vem do dataset repo
    TICK_FILES = {
        "ticks_btc_full_clean.parquet": "data/ticks_btc_full_clean.parquet",
        "new_ticks_pmdata.parquet":     "data/new_ticks_pmdata.parquet",
    }
    for local_name, hf_path in TICK_FILES.items():
        local_path = DATA_DIR / local_name
        if local_path.exists():
            log.info("  %s already cached", local_name)
            continue
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO, filename=hf_path,
                token=HF_TOKEN, repo_type="model",
                local_dir=str(DATA_DIR), local_dir_use_symlinks=False,
            )
            src = DATA_DIR / hf_path
            if src.exists() and not local_path.exists():
                shutil.move(str(src), str(local_path))
            log.info("  Downloaded %s (%.1fMB)", local_name, local_path.stat().st_size / 1e6)
        except Exception as e:
            log.warning("  Could not download %s: %s (opcional)", local_name, e)

    # Champion metrics
    champion = {"version": "v26_40f_rt", "wf_auc": 0.857, "wf_brier": 0.999, "wf_acc": 0.0}
    try:
        meta_path = hf_hub_download(
            repo_id=HF_MODEL_REPO, filename="champion_meta.json",
            token=HF_TOKEN, repo_type="model"
        )
        with open(meta_path) as f:
            champion = json.load(f)
        log.info("Champion: %s AUC=%.4f", champion["version"], champion["wf_auc"])
    except Exception as e:
        log.warning("Could not load champion meta: %s — usando default", e)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD MARKETS
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 1: Loading markets...")

    markets = pd.read_csv(DATA_DIR / "all_markets.csv")
    markets["market_id"] = markets["market_id"].astype(str)
    markets["slot_ts"]   = markets["slot_ts"].astype(int)
    markets = markets.sort_values("slot_ts").reset_index(drop=True)
    log.info("Total markets: %d (%.1f%% UP)", len(markets), 100 * markets["target"].mean())

    markets["rank"] = range(len(markets))
    slot_to_rank = dict(zip(markets["slot_ts"], markets["rank"]))
    all_targets  = markets["target"].values
    all_slot_ts  = markets["slot_ts"].values
    all_mids     = markets["market_id"].values

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: LOAD OB FEATURES v31 (ob60_* apenas — sem lookahead)
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 2: Loading ob_features_v31 (ob60_* only, t<60s)...")

    ob_path = DATA_DIR / "ob_features_v31.parquet"
    if not ob_path.exists():
        raise RuntimeError("ob_features_v31.parquet não encontrado! Aguarde o fetch completar.")

    ob_df = pd.read_parquet(str(ob_path))
    ob_df["market_id"] = ob_df["market_id"].astype(str)

    # Usar APENAS ob60_* — ob180_* contém t>60s (lookahead)
    ob60_cols = [c for c in ob_df.columns if c.startswith("ob60_")]
    ob_df_60  = ob_df[["market_id"] + ob60_cols].copy()

    # REMOVER features não disponíveis no live:
    #   pc_volatility   — não existe no live
    #   clob_mid_velocity — não está no CLOB_KEEP do live (removido por divergência)
    DROP_OB60 = {"ob60_pc_volatility", "ob60_clob_mid_velocity"}
    ob60_cols = [c for c in ob60_cols if c not in DROP_OB60]
    ob_df_60  = ob_df_60[["market_id"] + ob60_cols]

    # Rename ob60_X → nome canônico do live:
    #   ob60_clob_* → clob_*  (sem prefixo ob_)
    #   ob60_*      → ob_*    (substitui ob60_ por ob_)
    def _canonical(col: str) -> str:
        col = col.replace("ob60_clob_", "clob_")
        col = col.replace("ob60_",      "ob_")
        return col

    rename_map = {c: _canonical(c) for c in ob60_cols}
    ob_df_60.rename(columns=rename_map, inplace=True)
    ob_cols = list(rename_map.values())

    ob_by_market = ob_df_60.set_index("market_id").to_dict("index")
    log.info("OB features: %d markets, %d cols: %s", len(ob_df_60), len(ob_cols), ob_cols)
    del ob_df
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: LOAD BINANCE SPOT
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 3: Loading Binance spot...")

    spot_dfs = []
    for sp in ["binance_spot_full.parquet", "binance_spot_local.parquet"]:
        sp_path = DATA_DIR / sp
        if sp_path.exists():
            spot_dfs.append(pd.read_parquet(str(sp_path)))
    spot_df = pd.concat(spot_dfs, ignore_index=True) if spot_dfs else pd.DataFrame()
    spot_df = spot_df.sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
    spot_ts_arr  = (spot_df["timestamp_ms"].values // 1000).astype(np.int64)
    spot_px_arr  = spot_df["close"].values.astype(np.float64)
    spot_vol_arr = spot_df["volume"].values.astype(np.float64) if "volume" in spot_df else np.zeros(len(spot_df))
    spot_hi_arr  = spot_df["high"].values.astype(np.float64)  if "high"   in spot_df else spot_px_arr.copy()
    spot_lo_arr  = spot_df["low"].values.astype(np.float64)   if "low"    in spot_df else spot_px_arr.copy()
    log.info("Binance spot: %d candles (%.0f days)", len(spot_ts_arr),
             (spot_ts_arr[-1] - spot_ts_arr[0]) / 86400)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: LOAD TICKS (lag history + Grupo D features)
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 4: Loading ticks (lag + Grupo D features, t<60s)...")

    all_mids_set = set(markets["market_id"].tolist())
    tick_cols    = ["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"]

    chunks = []
    tick_path = DATA_DIR / "ticks_btc_full_clean.parquet"
    if tick_path.exists():
        pf = pq.ParquetFile(str(tick_path))
        for rg_i in range(pf.metadata.num_row_groups):
            chunk = pf.read_row_group(rg_i, columns=tick_cols).to_pandas()
            chunk["market_id"] = chunk["market_id"].astype(str)
            chunk = chunk[chunk["market_id"].isin(all_mids_set)]
            if len(chunk):
                chunks.append(chunk)

    new_tick_path = DATA_DIR / "new_ticks_pmdata.parquet"
    if new_tick_path.exists():
        nt = pd.read_parquet(str(new_tick_path))
        nt["market_id"] = nt["market_id"].astype(str)
        nt = nt[nt["market_id"].isin(all_mids_set)]
        if len(nt):
            for col in tick_cols:
                if col not in nt.columns:
                    nt[col] = 0
            chunks.append(nt[tick_cols])

    gc.collect()
    btc = pd.concat(chunks, ignore_index=True)
    btc = btc[btc["timestamp_ms"] > 0]

    slot_ts_map = dict(zip(markets["market_id"], markets["slot_ts"]))
    btc["slot_ts_val"] = btc["market_id"].map(slot_ts_map).astype(float)
    btc["t_sec"] = btc["timestamp_ms"].astype(float) / 1000 - btc["slot_ts_val"]

    # CORTE ESTRITO t<60s — nunca usar dados além de OBS_SECS
    btc = btc[(btc["t_sec"] >= 0) & (btc["t_sec"] < OBS_SECS)]
    log.info("Ticks t<60s: %d para %d markets", len(btc), btc["market_id"].nunique())

    # ── Grupo D: features de order flow por mercado ───────────────────────
    log.info("Computando Grupo D (order flow features)...")

    def _safe_div(a, b, fill=0.5):
        return np.where(b > 1e-9, a / b, fill)

    up_mask  = btc["outcome"] == "Up"
    dn_mask  = btc["outcome"] == "Down"
    buy_mask = btc["side"]    == "BUY"
    sel_mask = btc["side"]    == "SELL"

    up_ticks = btc[up_mask]
    dn_ticks = btc[dn_mask]

    slot_vol_up  = up_ticks.groupby("market_id")["size_usdc"].sum()
    slot_vol_dn  = dn_ticks.groupby("market_id")["size_usdc"].sum()
    slot_vol_tot = slot_vol_up.add(slot_vol_dn, fill_value=0)
    slot_nticks  = btc.groupby("market_id").size()

    # btc_up_ratio (vol-weighted)
    slot_up_ratio = (slot_vol_up / slot_vol_tot.clip(lower=1e-9)).fillna(0.5)

    # btc_buy_ratio (count-based)
    n_buy = btc[buy_mask].groupby("market_id").size()
    n_sel = btc[sel_mask].groupby("market_id").size()
    n_tot = n_buy.add(n_sel, fill_value=0)
    slot_buy_ratio = (n_buy / n_tot.clip(lower=1)).fillna(0.5)

    # Janelas temporais w0=[0,30s) e w1=[30,60s)
    w0_mask = btc["t_sec"] <  30
    w1_mask = btc["t_sec"] >= 30

    def _win_up_ratio(mask):
        sub = btc[mask]
        vu = sub[sub["outcome"] == "Up"].groupby("market_id")["size_usdc"].sum()
        vd = sub[sub["outcome"] == "Down"].groupby("market_id")["size_usdc"].sum()
        vt = vu.add(vd, fill_value=0)
        return (vu / vt.clip(lower=1e-9)).fillna(0.5)

    slot_up_w0 = _win_up_ratio(w0_mask)
    slot_up_w1 = _win_up_ratio(w1_mask)
    slot_momentum = slot_up_w1.sub(slot_up_w0, fill_value=0).fillna(0.0)
    slot_ur_stability = pd.concat([slot_up_w0, slot_up_w1], axis=1).std(axis=1).fillna(0.0)

    # size disparity
    avg_up_size = up_ticks.groupby("market_id")["size_usdc"].mean().fillna(0)
    avg_dn_size = dn_ticks.groupby("market_id")["size_usdc"].mean().fillna(0)
    slot_size_disp = avg_up_size.sub(avg_dn_size, fill_value=0)

    # btc_vwap_up/dn/spread REMOVIDOS — não existem no live; cálculo omitido

    # time-weighted up_ratio (exp decay, recentes têm mais peso)
    def _tw_up_ratio():
        sub = btc.copy()
        sub["w"] = np.exp(-0.02 * (OBS_SECS - sub["t_sec"])) * sub["size_usdc"]
        sub["w_up"] = sub["w"] * (sub["outcome"] == "Up").astype(float)
        tw_up  = sub.groupby("market_id")["w_up"].sum()
        tw_tot = sub.groupby("market_id")["w"].sum().clip(lower=1e-9)
        return (tw_up / tw_tot).fillna(0.5)

    slot_tw_up_ratio = _tw_up_ratio()

    del btc, chunks, up_ticks, dn_ticks
    gc.collect()
    log.info("Grupo D computado. Memória liberada.")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: BUILD FEATURE MATRIX (71 features, grupos A-F)
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 5: Building feature matrix (%d markets)...", len(markets))

    def spot_at(ts_s):
        idx = np.searchsorted(spot_ts_arr, ts_s, side="right") - 1
        return float(spot_px_arr[max(0, min(idx, len(spot_px_arr) - 1))])

    def spot_vol_at(ts_start, ts_end):
        i0 = int(np.searchsorted(spot_ts_arr, ts_start, side="left"))
        i1 = int(np.searchsorted(spot_ts_arr, ts_end,   side="right"))
        return float(spot_vol_arr[i0:i1].sum()) if i1 > i0 else 0.0

    def spot_volatility(ts_start, ts_end):
        i0 = int(np.searchsorted(spot_ts_arr, ts_start, side="left"))
        i1 = int(np.searchsorted(spot_ts_arr, ts_end,   side="right"))
        if i1 - i0 >= 3:
            px   = spot_px_arr[i0:i1]
            rets = np.diff(px) / px[:-1]
            return float(np.std(rets))
        return 0.0

    rows = []
    skipped_no_ob = 0

    for rank_i, row in markets.iterrows():
        mid      = row["market_id"]
        slot_ts  = int(row["slot_ts"])
        target   = int(row["target"])
        ext_rank = slot_to_rank.get(slot_ts, rank_i)

        feat = {}

        obs_end = slot_ts + OBS_SECS  # t=60s

        # ── Grupo A: Spot Binance ─────────────────────────────────────────
        px_now = spot_at(obs_end)

        feat["btc_pre_5m_ret"]  = (px_now / spot_at(slot_ts - 300)  - 1) if spot_at(slot_ts - 300)  > 0 else 0.0
        feat["btc_pre_15m_ret"] = (px_now / spot_at(slot_ts - 900)  - 1) if spot_at(slot_ts - 900)  > 0 else 0.0
        feat["btc_pre_30m_ret"] = (px_now / spot_at(slot_ts - 1800) - 1) if spot_at(slot_ts - 1800) > 0 else 0.0
        feat["btc_pre_1h_ret"]  = (px_now / spot_at(slot_ts - 3600) - 1) if spot_at(slot_ts - 3600) > 0 else 0.0
        feat["btc_pre_4h_ret"]  = (px_now / spot_at(slot_ts - 14400)- 1) if spot_at(slot_ts - 14400)> 0 else 0.0
        # btc_pre_1h_4h_ratio REMOVIDO — warm-up 4h, não existe no live

        t0_idx = int(np.searchsorted(spot_ts_arr, slot_ts, side="left"))
        t1_idx = int(np.searchsorted(spot_ts_arr, obs_end,  side="right"))
        if t1_idx > t0_idx:
            inslot_px = spot_px_arr[t0_idx:t1_idx]
            feat["btc_inslot_ret"]   = float(inslot_px[-1] / inslot_px[0] - 1) if inslot_px[0] > 0 else 0.0
            feat["btc_inslot_range"] = float((spot_hi_arr[t0_idx:t1_idx].max() - spot_lo_arr[t0_idx:t1_idx].min()) / px_now) if px_now > 0 else 0.0
        else:
            feat["btc_inslot_ret"] = feat["btc_inslot_range"] = 0.0

        feat["btc_vol_1h"]  = spot_volatility(slot_ts - 3600,  slot_ts)
        # btc_vol_4h REMOVIDO — warm-up 4h, não existe no live

        px_k = px_now / 1000
        feat["btc_dist_1k"] = min(px_k - math.floor(px_k), math.ceil(px_k) - px_k)

        slot_vol = spot_vol_at(slot_ts - 300, slot_ts)
        h1_vol   = spot_vol_at(slot_ts - 3600, slot_ts - 300)
        feat["btc_spot_vol_ratio"] = slot_vol / (h1_vol / 11 + 1e-9)

        # ── Grupo B: L2 Orderbook (ob60_* → nomes canônicos) ─────────────
        ob = ob_by_market.get(mid)
        if ob is not None:
            for col in ob_cols:
                feat[col] = float(ob.get(col, 0.0))
        else:
            skipped_no_ob += 1
            for col in ob_cols:
                if "ratio" in col or "imbalance" in col or "imb" in col:
                    feat[col] = 0.5
                elif "spread" in col:
                    feat[col] = 0.02
                elif "depth_5c" in col:
                    feat[col] = 0.5
                elif col == "ob_mid":
                    feat[col] = 0.5
                else:
                    feat[col] = 0.0

        # ── Grupo D: Order Flow (ticks t<60s) ────────────────────────────
        feat["btc_up_ratio"]        = float(slot_up_ratio.get(mid, 0.5))
        feat["btc_n_ticks"]         = float(slot_nticks.get(mid, 0.0))
        feat["btc_buy_ratio"]       = float(slot_buy_ratio.get(mid, 0.5))
        feat["btc_tw_up_ratio"]     = float(slot_tw_up_ratio.get(mid, 0.5))
        feat["btc_momentum"]        = float(slot_momentum.get(mid, 0.0))
        feat["btc_up_w0"]           = float(slot_up_w0.get(mid, 0.5))
        feat["btc_up_w1"]           = float(slot_up_w1.get(mid, 0.5))
        feat["btc_size_disparity"]  = float(slot_size_disp.get(mid, 0.0))
        feat["btc_up_ratio_stability"] = float(slot_ur_stability.get(mid, 0.0))
        # btc_vwap_up/dn/spread REMOVIDOS — não existem no live

        # ── Grupo E: Lag History (ring buffer) ───────────────────────────
        lag_streak = 0
        streak_dir = None

        for lag_n in range(1, 6):
            prev_rank = ext_rank - lag_n
            if prev_rank >= 0:
                prev_target = int(all_targets[prev_rank])
                prev_slot   = int(all_slot_ts[prev_rank])
                prev_mid    = all_mids[prev_rank]

                # Gap muito grande → janela perdida
                if (slot_ts - prev_slot) > lag_n * SLOT_DURATION * 3:
                    feat[f"lag_{lag_n}_outcome"]       = 0.5
                    feat[f"prev_slot_up_ratio_{lag_n}"] = 0.5
                    feat[f"prev_slot_n_ticks_{lag_n}"]  = 0.0
                    feat[f"prev_slot_vol_{lag_n}"]      = 0.0
                    continue

                feat[f"lag_{lag_n}_outcome"]        = float(prev_target)
                feat[f"prev_slot_up_ratio_{lag_n}"] = float(slot_up_ratio.get(prev_mid, 0.5))
                feat[f"prev_slot_n_ticks_{lag_n}"]  = float(slot_nticks.get(prev_mid, 0.0))
                feat[f"prev_slot_vol_{lag_n}"]      = float(slot_vol_tot.get(prev_mid, 0.0) if prev_mid in slot_vol_tot else 0.0)

                if lag_n == 1:
                    streak_dir = prev_target; lag_streak = 1
                elif prev_target == streak_dir:
                    lag_streak += 1
            else:
                for k in [f"lag_{lag_n}_outcome", f"prev_slot_up_ratio_{lag_n}",
                          f"prev_slot_n_ticks_{lag_n}", f"prev_slot_vol_{lag_n}"]:
                    feat[k] = 0.0

        feat["lag_streak"] = float(lag_streak)

        def _hist_ur(lookback=20):
            vals = []
            for d in range(1, lookback + 1):
                pr = ext_rank - d
                if pr < 0:
                    break
                pm = all_mids[pr]
                v = slot_up_ratio.get(pm, None)
                if v is not None:
                    vals.append(v)
            return vals

        hist_vals = _hist_ur(20)
        cur_ur    = feat.get("prev_slot_up_ratio_1", 0.5)
        if len(hist_vals) >= 3:
            mu20 = np.mean(hist_vals); sd20 = max(np.std(hist_vals), 0.01)
            feat["lag_ur_zscore_20"] = float(np.clip((cur_ur - mu20) / sd20, -5, 5))
        else:
            feat["lag_ur_zscore_20"] = 0.0

        hist5 = hist_vals[:5]
        if len(hist5) >= 2:
            mu5 = np.mean(hist5); sd5 = max(np.std(hist5), 0.01)
            feat["lag_ur_zscore_5"] = float(np.clip((cur_ur - mu5) / sd5, -5, 5))
        else:
            feat["lag_ur_zscore_5"] = 0.0

        # btc_up_ratio z-scores — computados no Grupo E (lag_ur_zscore_5/20) — sem aliases

        # ── Grupo F: Temporais + Cross-features ──────────────────────────
        dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        dow  = dt.weekday()

        feat["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        feat["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        feat["dow_sin"]  = math.sin(2 * math.pi * dow  / 7)
        feat["dow_cos"]  = math.cos(2 * math.pi * dow  / 7)

        feat["hour_x_up_ratio"] = feat["btc_up_ratio"]    * (hour / 24)
        feat["hour_x_tw_ur"]    = feat["btc_tw_up_ratio"] * (hour / 24)

        # Cross-features: usar nomes canônicos do live (ob_imbalance/ob_spread/ob_depth_ratio)
        ob_imb = feat.get("ob_imbalance", 0.0)
        feat["x_imb_x_inslot"]     = ob_imb * feat["btc_inslot_ret"]
        feat["x_imb_x_ur"]         = ob_imb * feat["btc_up_ratio"]
        feat["x_spread_x_vol"]     = feat.get("ob_spread", 0.0) * feat["btc_vol_1h"]
        feat["x_depth_x_vol"]      = feat.get("ob_depth_ratio", 1.0) * feat["btc_vol_1h"]
        feat["x_depth_x_momentum"] = feat.get("ob_depth_ratio", 1.0) * feat["btc_momentum"]

        feat["target"] = target
        rows.append(feat)

    df = pd.DataFrame(rows)
    FEATURE_COLS = [c for c in df.columns if c != "target"]
    log.info("Feature matrix: %d rows × %d features (no_ob=%d)", len(df), len(FEATURE_COLS), skipped_no_ob)
    log.info("Features: %s", FEATURE_COLS[:10])

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["target"].values.astype(int)
    log.info("Class balance: %d UP / %d DOWN (%.1f%% UP)", y.sum(), (y==0).sum(), 100*y.mean())

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: ANTI-OVERFITTING — NOISE CANARY + ADVERSARIAL VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 6: Anti-overfitting checks...")

    # Noise canary: feature aleatória deve ter importância ~0
    rng = np.random.default_rng(42)
    canary = rng.random(len(X)).reshape(-1, 1).astype(np.float32)
    X_canary = np.hstack([X, canary])
    canary_name = "__noise_canary__"
    feat_names_canary = FEATURE_COLS + [canary_name]

    # Adversarial validation: train consegue separar primeiros 50% vs últimos 50%?
    # Se AUC > 0.6 → dataset tem drift temporal (feature criadas antes do target vazam)
    n_half = len(X) // 2
    av_labels = np.array([0] * n_half + [1] * (len(X) - n_half))
    av_model  = lgb.LGBMClassifier(n_estimators=200, max_depth=4, num_leaves=15,
                                    min_child_samples=50, random_state=42, verbose=-1)
    from sklearn.model_selection import cross_val_score
    av_auc = cross_val_score(av_model, X, av_labels, cv=5, scoring="roc_auc").mean()
    log.info("Adversarial validation AUC=%.4f (ideal < 0.55, preocupante > 0.65)", av_auc)
    if av_auc > 0.70:
        log.warning("⚠️  AV AUC=%.4f — possível data drift temporal severo! Checar features.", av_auc)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: PURGED WALK-FORWARD FEATURE SCREENING
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 7: Purged WF feature screening (gap=%d)...", WF_GAP)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP)
    screen = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=30, random_state=42, verbose=-1
    )
    feat_importances = np.zeros(len(feat_names_canary))
    for tr_idx, val_idx in tscv.split(X_canary):
        screen.fit(X_canary[tr_idx], y[tr_idx])
        feat_importances += screen.feature_importances_
    feat_importances /= N_SPLITS

    canary_imp = feat_importances[-1]
    log.info("Noise canary importance: %.2f", canary_imp)
    if canary_imp > 5:
        log.warning("⚠️  Canary importance=%.2f — modelo pode estar sofrendo overfitting", canary_imp)

    # Filtrar features com importância > canary
    feat_rank_full = np.argsort(feat_importances[:-1])[::-1]  # excluir canary
    log.info("Feature importance ranking:")
    for i, idx in enumerate(feat_rank_full):
        marker = " ← canary threshold" if feat_importances[idx] <= canary_imp else ""
        log.info("  %3d. %-40s importance=%.1f%s",
                 i+1, FEATURE_COLS[idx], feat_importances[idx], marker)

    # Features acima do canary
    selected_above_canary = [FEATURE_COLS[i] for i in feat_rank_full
                             if feat_importances[i] > canary_imp]
    log.info("%d features acima do noise canary (importance > %.2f)",
             len(selected_above_canary), canary_imp)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 8: FEATURE COUNT SELECTION
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 8: Testando contagens de features...")

    FEATURE_COUNTS = [len(selected_above_canary), 50, 40, 35, 30, 25, 20]
    FEATURE_COUNTS = sorted(set(n for n in FEATURE_COUNTS if 10 <= n <= len(FEATURE_COLS)), reverse=True)

    best_combo = None
    best_score = -1

    for n_feats in FEATURE_COUNTS:
        top_n = [FEATURE_COLS[i] for i in feat_rank_full[:n_feats]]
        X_n   = df[top_n].values.astype(np.float32)

        aucs = []
        for tr_idx, val_idx in TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP).split(X_n):
            m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                   num_leaves=20, min_child_samples=30, random_state=42, verbose=-1)
            m.fit(X_n[tr_idx], y[tr_idx])
            p = m.predict_proba(X_n[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], p))

        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)
        # Variance penalty: preferir menor variância
        penalized = mean_auc - 0.5 * std_auc
        log.info("  n=%d  WF_AUC=%.4f ± %.4f  penalized=%.4f", n_feats, mean_auc, std_auc, penalized)

        if penalized > best_score:
            best_score = penalized
            best_combo = {"n": n_feats, "feats": top_n, "auc": mean_auc, "std": std_auc}

    log.info("Melhor combo: n=%d  AUC=%.4f ± %.4f", best_combo["n"], best_combo["auc"], best_combo["std"])
    SELECTED_FEATURES = best_combo["feats"]
    X_sel = df[SELECTED_FEATURES].values.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 9: OPTUNA HYPERPARAMETER TUNING (variance-penalized)
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 9: Optuna %d trials (variance-penalized)...", OPTUNA_TRIALS)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "num_leaves":        trial.suggest_int("num_leaves", 8, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
            "random_state": 42, "verbose": -1,
        }
        aucs = []
        for tr_idx, val_idx in TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP).split(X_sel):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_sel[tr_idx], y[tr_idx],
                  eval_set=[(X_sel[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            p = m.predict_proba(X_sel[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], p))
        mean_auc = float(np.mean(aucs))
        std_auc  = float(np.std(aucs))
        # Variance penalty: punir instabilidade entre folds
        return mean_auc - 0.5 * std_auc

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["verbose"]      = -1
    log.info("Best params: %s", best_params)
    log.info("Best penalized score: %.4f", study.best_value)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 10: WALK-FORWARD EVALUATION (métricas finais)
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 10: Walk-forward evaluation...")

    wf_aucs, wf_briers, wf_accs = [], [], []
    for fold_i, (tr_idx, val_idx) in enumerate(
            TimeSeriesSplit(n_splits=N_SPLITS, gap=WF_GAP).split(X_sel)):
        m = lgb.LGBMClassifier(**best_params)
        m.fit(X_sel[tr_idx], y[tr_idx])
        p    = m.predict_proba(X_sel[val_idx])[:, 1]
        pred = (p >= 0.5).astype(int)
        auc  = roc_auc_score(y[val_idx], p)
        brier = brier_score_loss(y[val_idx], p)
        acc   = (pred == y[val_idx]).mean()
        wf_aucs.append(auc); wf_briers.append(brier); wf_accs.append(acc)
        log.info("  Fold %d: AUC=%.4f  Brier=%.4f  Acc=%.4f", fold_i+1, auc, brier, acc)

    wf_auc   = float(np.mean(wf_aucs))
    wf_brier = float(np.mean(wf_briers))
    wf_acc   = float(np.mean(wf_accs))
    wf_std   = float(np.std(wf_aucs))
    log.info("WF AUC=%.4f ± %.4f  Brier=%.4f  Acc=%.4f", wf_auc, wf_std, wf_brier, wf_acc)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 11: TREINO FINAL + CALIBRAÇÃO
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 11: Treino final + calibração isotônica...")

    from sklearn.calibration import CalibratedClassifierCV
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_sel, y)

    # Calibração isotônica (holdout = últimos 10%)
    cal_split = int(len(X_sel) * 0.90)
    # Treinar base LightGBM separado e calibrar no holdout
    base_for_cal = lgb.LGBMClassifier(**best_params)
    base_for_cal.fit(X_sel[:cal_split], y[:cal_split])
    # cv="prefit" removido no sklearn 1.4+ — usar set_params após fit manual
    cal_model = CalibratedClassifierCV(
        estimator=base_for_cal,
        method="isotonic", cv=5
    )
    cal_model.fit(X_sel[:cal_split], y[:cal_split])

    # Brier no holdout (calibrado vs não-calibrado)
    p_raw = final_model.predict_proba(X_sel[cal_split:])[:, 1]
    p_cal = cal_model.predict_proba(X_sel[cal_split:])[:, 1]
    log.info("Holdout Brier raw=%.4f  cal=%.4f",
             brier_score_loss(y[cal_split:], p_raw),
             brier_score_loss(y[cal_split:], p_cal))

    # ══════════════════════════════════════════════════════════════════════
    # STEP 12: PROMOTE IF CHAMPION
    # ══════════════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 12: Champion check...")

    champ_auc = float(champion.get("wf_auc", 0.0))
    VERSION   = f"v32_{best_combo['n']}f_ws"
    is_new_champ = wf_auc > champ_auc

    log.info("Champion AUC=%.4f | Candidate AUC=%.4f | %s",
             champ_auc, wf_auc, "🏆 NOVO CHAMPION!" if is_new_champ else "Não supera champion")

    api = HfApi(token=HF_TOKEN)

    # Sempre salva o modelo candidato (para análise)
    model_payload = {
        "model":            cal_model,
        "feature_cols":     SELECTED_FEATURES,
        "version":          VERSION,
        "wf_auc":           wf_auc,
        "wf_brier":         wf_brier,
        "wf_acc":           wf_acc,
        "wf_std":           wf_std,
        "best_params":      best_params,
        "n_features":       best_combo["n"],
        "obs_secs":         OBS_SECS,
        "av_auc":           float(av_auc),
        "canary_importance": float(canary_imp),
        "trained_at":       datetime.utcnow().isoformat(),
    }
    cand_path = Path(f"/tmp/{VERSION}.pkl")
    with open(cand_path, "wb") as f:
        pickle.dump(model_payload, f)

    api.upload_file(
        path_or_fileobj=str(cand_path),
        path_in_repo=f"models/{VERSION}.pkl",
        repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN
    )
    log.info("Candidato salvo: models/%s.pkl", VERSION)

    if is_new_champ:
        # Promove como champion.pkl
        api.upload_file(
            path_or_fileobj=str(cand_path),
            path_in_repo="champion.pkl",
            repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN
        )
        meta = {
            "version":          VERSION,
            "wf_auc":           wf_auc,
            "wf_brier":         wf_brier,
            "wf_acc":           wf_acc,
            "wf_std":           wf_std,
            "n_features":       best_combo["n"],
            "obs_secs":         OBS_SECS,
            "av_auc":           float(av_auc),
            "canary_importance": float(canary_imp),
            "promoted_at":      datetime.utcnow().isoformat(),
        }
        meta_path = Path("/tmp/champion_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo="champion_meta.json",
            repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN
        )
        log.info("🏆 CHAMPION ATUALIZADO: %s  AUC=%.4f", VERSION, wf_auc)
    else:
        log.info("Champion mantido: %s  AUC=%.4f", champion.get("version"), champ_auc)

    log.info("=" * 70)
    log.info("DONE. v31 concluído.")
    log.info("  WF AUC   = %.4f ± %.4f", wf_auc, wf_std)
    log.info("  WF Brier = %.4f", wf_brier)
    log.info("  WF Acc   = %.4f", wf_acc)
    log.info("  AV AUC   = %.4f (adversarial)", av_auc)
    log.info("  Canary   = %.2f importance", canary_imp)
    log.info("  Features = %d", best_combo["n"])
    log.info("  Champion = %s", "SIM 🏆" if is_new_champ else "NÃO")


@app.local_entrypoint()
def main():
    train_v31.remote()
