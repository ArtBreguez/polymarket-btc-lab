"""
backtest_v28.py — OOS backtest do champion v28 (73 features, 60s obs)
======================================================================
Estratégia:
  1. Baixa champion.pkl do HF
  2. Reconstrói as features dos últimos 10% dos mercados (mesmo código do treino)
  3. Prediz com o modelo, aplica filtros do live_trader (v28)
  4. Calcula P&L simulado com taxa 2% nos ganhos

Walk-forward: 90/10 split — mercados ordenados por slot_ts.
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
app = modal.App("btc-v28-backtest", image=image)


@app.function(
    cpu=8,
    memory=32768,
    timeout=3600,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/cache": vol},
)
def backtest_v28():
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

    HF_TOKEN      = os.environ.get("HF_TOKEN", "")
    HF_MODEL_REPO = "artbreguez/polymarket-btc-model"

    # ── Live trader filters v28 ───────────────────────────────────────────
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
        "champion.pkl":                  "champion.pkl",
        "champion_meta.json":            "champion_meta.json",
    }
    for local_name, hf_path in FILES.items():
        local_path = DATA_DIR / local_name
        if local_path.exists():
            log.info("  %s cached", local_name)
            continue
        try:
            downloaded = hf_hub_download(
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
    features = bundle["features"]
    version  = bundle.get("version", "unknown")
    wf_auc   = bundle.get("wf_auc", 0)
    log.info("Champion: %s | %d features | AUC=%.4f", version, len(features), wf_auc)

    # ── Step 3: Carregar mercados ─────────────────────────────────────────
    log.info("Step 3: Loading markets...")
    m1 = pd.read_csv(DATA_DIR / "all_markets.csv")
    new_path = DATA_DIR / "new_markets.csv"
    if new_path.exists():
        m2 = pd.read_csv(new_path)
        if "target" in m2.columns:
            markets = pd.concat([m1, m2], ignore_index=True)
        else:
            markets = m1
    else:
        markets = m1
    markets = markets[markets["target"].notna()].sort_values("slot_ts").reset_index(drop=True)
    log.info("Total markets: %d (%.1f%% UP)", len(markets), 100 * markets["target"].mean())

    # ── Step 4: Carregar OB features ─────────────────────────────────────
    log.info("Step 4: Loading OB features...")
    ob_df = pd.read_parquet(str(DATA_DIR / "ob_features_full.parquet"))
    OB_DROP = [
        "ob_mid_drift",        # divergência
        "ob_imb_w1",           # divergência
        "ob_imb_w2",           # divergência
        "ob_pc_up_ratio",      # fallback frequente
        "ob_pc_volatility",    # fallback frequente
        "clob_imb_mean",       # divergência
        "clob_imb_std",        # divergência
        "clob_imb_drift",      # divergência
        "clob_depth_trend",    # divergência
        "clob_activity_rate",  # fallback frequente
    ]
    ob_df = ob_df.drop(columns=[c for c in OB_DROP if c in ob_df.columns])
    ob_by_id = {str(row["market_id"]): row.to_dict() for _, row in ob_df.iterrows()}
    log.info("OB features loaded: %d markets", len(ob_df))

    # Filtrar só mercados com OB features (igual ao treino) e fazer split
    # market_id pode ser int no CSV e str no parquet — normalizar para str
    markets["market_id"] = markets["market_id"].astype(str)
    markets_with_ob = markets[markets["market_id"].isin(ob_by_id.keys())].reset_index(drop=True)
    log.info("Markets with OB: %d (%.1f%% UP)", len(markets_with_ob), 100 * markets_with_ob["target"].mean())

    # ── Walk-forward split: último 10% (só mercados com OB) ──────────────
    split_i     = int(len(markets_with_ob) * 0.90)
    markets_oos = markets_with_ob.iloc[split_i:].reset_index(drop=True)
    log.info("OOS split: %d mercados (últimos 10%%)", len(markets_oos))

    all_ids = set(markets_oos["market_id"].tolist())

    # ── Step 5: Spot BTC ──────────────────────────────────────────────────
    log.info("Step 5: Loading BTC spot data...")
    spot_dfs = []
    for sp in ["binance_spot_full.parquet", "binance_spot_local.parquet"]:
        sp_path = DATA_DIR / sp
        if sp_path.exists():
            spot_dfs.append(pd.read_parquet(str(sp_path)))
    spot = pd.concat(spot_dfs).sort_values("timestamp_ms").drop_duplicates("timestamp_ms").reset_index(drop=True)
    spot_ts_arr = spot["timestamp_ms"].values // 1000
    spot_px_arr = spot["close"].values.astype(np.float32)
    spot_hi_arr = spot["high"].values.astype(np.float32)
    spot_lo_arr = spot["low"].values.astype(np.float32)
    log.info("Spot data: %d candles", len(spot))

    def spot_at(ts_sec):
        idx = int(np.searchsorted(spot_ts_arr, ts_sec, side="right")) - 1
        return float(spot_px_arr[idx]) if 0 <= idx < len(spot_px_arr) else 0.0

    def spot_volatility(t_start, t_end):
        i0 = int(np.searchsorted(spot_ts_arr, t_start, side="left"))
        i1 = int(np.searchsorted(spot_ts_arr, t_end, side="right"))
        px = spot_px_arr[i0:i1]
        if len(px) < 2:
            return 0.0
        return float(np.std(np.diff(np.log(px + 1e-9))))

    # ── Step 6: Ticks ─────────────────────────────────────────────────────
    log.info("Step 6: Loading ticks...")
    tick_cols = ["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"]
    tick_dfs = []
    pf = pq.ParquetFile(str(DATA_DIR / "ticks_btc_full_clean.parquet"))
    for batch in pf.iter_batches(batch_size=200_000, columns=tick_cols):
        tb = batch.to_pandas()
        tb = tb[tb["market_id"].isin(all_ids)]
        if len(tb):
            tick_dfs.append(tb)
    new_tick_path = DATA_DIR / "new_ticks_pmdata.parquet"
    if new_tick_path.exists():
        nt = pd.read_parquet(str(new_tick_path), columns=tick_cols)
        nt = nt[nt["market_id"].isin(all_ids)]
        if len(nt):
            tick_dfs.append(nt)
    if not tick_dfs:
        log.warning("No ticks found for OOS markets!")
        ticks_df = pd.DataFrame(columns=tick_cols)
    else:
        ticks_df = pd.concat(tick_dfs, ignore_index=True)
    ticks_df["t_sec"] = ticks_df["timestamp_ms"] // 1000
    tick_by_market = {}
    for mid, grp in ticks_df.groupby("market_id"):
        tick_by_market[mid] = grp[["t_sec", "outcome", "side", "price", "size_usdc"]].to_dict("records")
    log.info("Ticks loaded: %d markets with ticks", len(tick_by_market))

    # Slot-level up_ratio para lags
    ticks_df["slot_mid"] = (ticks_df["t_sec"] // 300) * 300 + 150
    slot_vol_up = ticks_df[ticks_df["outcome"] == "Up"].groupby("slot_mid")["size_usdc"].sum()
    slot_vol_tot = ticks_df.groupby("slot_mid")["size_usdc"].sum()
    slot_up_ratio = slot_vol_up / slot_vol_tot.clip(lower=1e-9)

    # Rank por slot_ts
    all_mkts_sorted = markets.sort_values("slot_ts").reset_index(drop=True)
    slot_to_rank = {int(r["slot_ts"]): i for i, r in all_mkts_sorted.iterrows()}

    # ── Step 7: Build features OOS ────────────────────────────────────────
    log.info("Step 7: Building features for %d OOS markets...", len(markets_oos))

    rows = []
    skipped = 0
    for rank_i, (_, row) in enumerate(markets_oos.iterrows()):
        mid      = row["market_id"]
        slot_ts  = int(row["slot_ts"])
        target   = int(row["target"])
        ext_rank = slot_to_rank.get(slot_ts, rank_i + split_i)

        if mid not in ob_by_id:
            skipped += 1
            continue

        ob = ob_by_id[mid]
        feat = {}

        # ── A. SPOT features ──
        obs_end_ts = slot_ts + OBS_SECS
        px_now = spot_at(obs_end_ts)

        def pre_ret(h_sec):
            px_h = spot_at(slot_ts - h_sec)
            return (px_now / px_h - 1) if px_h > 0 else 0.0

        feat["btc_pre_5m_ret"]  = pre_ret(300)
        feat["btc_pre_15m_ret"] = pre_ret(900)
        feat["btc_pre_30m_ret"] = pre_ret(1800)
        feat["btc_pre_1h_ret"]  = pre_ret(3600)

        t0_idx = int(np.searchsorted(spot_ts_arr, slot_ts, side="left"))
        t1_idx = int(np.searchsorted(spot_ts_arr, obs_end_ts, side="right"))
        if t1_idx > t0_idx:
            inslot_px = spot_px_arr[t0_idx:t1_idx]
            feat["btc_inslot_ret"]   = float(inslot_px[-1] / inslot_px[0] - 1) if inslot_px[0] > 0 else 0.0
            feat["btc_inslot_vol"]   = float(np.std(inslot_px) / (np.mean(inslot_px) + 1e-8)) if len(inslot_px) > 1 else 0.0
            inslot_hi = spot_hi_arr[t0_idx:t1_idx]
            inslot_lo = spot_lo_arr[t0_idx:t1_idx]
            feat["btc_inslot_range"] = float((inslot_hi.max() - inslot_lo.min()) / px_now) if px_now > 0 else 0.0
        else:
            feat["btc_inslot_ret"] = feat["btc_inslot_vol"] = feat["btc_inslot_range"] = 0.0

        feat["btc_vol_1h"] = spot_volatility(slot_ts - 3600, slot_ts)
        feat["btc_vol_2h"] = spot_volatility(slot_ts - 7200, slot_ts)

        if px_now > 0:
            for k in [1000, 5000, 10000]:
                nearest_k = round(px_now / k) * k
                feat[f"btc_dist_{k}"] = abs(px_now - nearest_k) / k
        else:
            for k in [1000, 5000, 10000]:
                feat[f"btc_dist_{k}"] = 0.0

        if px_now > 0:
            feat["btc_spot_vol_ratio"] = spot_volatility(slot_ts - 3600, slot_ts) * px_now
        else:
            feat["btc_spot_vol_ratio"] = 0.0

        # ── B. TICK features ──
        ticks = tick_by_market.get(mid, [])
        obs_ticks = [t for t in ticks if slot_ts <= t["t_sec"] < obs_end_ts]

        if obs_ticks:
            vol_up = sum(t["size_usdc"] for t in obs_ticks if t.get("outcome") == "Up")
            vol_dn = sum(t["size_usdc"] for t in obs_ticks if t.get("outcome") == "Down")
            total = vol_up + vol_dn
            up_tks = [t for t in obs_ticks if t.get("outcome") == "Up"]
            dn_tks = [t for t in obs_ticks if t.get("outcome") == "Down"]

            # Time windows
            half = (slot_ts + obs_end_ts) / 2
            for label, w_start, w_end in [("w0", slot_ts, half), ("w1", half, obs_end_ts)]:
                subset = [t for t in obs_ticks if w_start <= t["t_sec"] < w_end]
                vu = sum(t["size_usdc"] for t in subset if t.get("outcome") == "Up")
                vd = sum(t["size_usdc"] for t in subset if t.get("outcome") == "Down")
                vt = vu + vd
                feat[f"btc_up_ratio_{label}"]    = float(vu / vt) if vt > 0 else 0.5
                feat[f"btc_n_ticks_{label}"]      = float(len(subset))
                feat[f"btc_vol_{label}"]           = float(vt)

            up_arr = np.array([1.0 if t.get("outcome") == "Up" else 0.0 for t in obs_ticks])
            sz_arr = np.array([t["size_usdc"] for t in obs_ticks])
            w_exp  = np.exp(np.linspace(-1, 0, len(obs_ticks)))
            feat["btc_tw_up_ratio"] = float(np.sum(up_arr * sz_arr * w_exp) / (np.sum(sz_arr * w_exp) + 1e-9))

            cur_up_ratio = float(vol_up / total) if total > 0 else 0.5
            feat["btc_up_ratio"]           = cur_up_ratio
            feat["btc_n_ticks"]            = float(len(obs_ticks))
            feat["btc_momentum"]           = float(vol_up / total - 0.5) if total > 0 else 0.0
            feat["btc_buy_ratio"]          = float(sum(1 for t in obs_ticks if t.get("side") == "BUY") / max(len(obs_ticks), 1))
            feat["btc_size_disparity"]     = float(np.std([t["size_usdc"] for t in obs_ticks])) if len(obs_ticks) > 1 else 0.0

            w_vals = []
            for t in obs_ticks:
                sub = [x for x in obs_ticks if abs(x["t_sec"] - t["t_sec"]) <= 5]
                if sub:
                    vu2 = sum(x["size_usdc"] for x in sub if x.get("outcome") == "Up")
                    vt2 = sum(x["size_usdc"] for x in sub) or 1
                    w_vals.append(vu2 / vt2)
            feat["btc_up_ratio_stability"] = float(np.std(w_vals)) if w_vals else 0.0
        else:
            cur_up_ratio = 0.5
            for k in ["btc_up_ratio", "btc_n_ticks", "btc_momentum", "btc_buy_ratio",
                      "btc_size_disparity", "btc_up_ratio_stability", "btc_tw_up_ratio"]:
                feat[k] = 0.5 if "ratio" in k else 0.0
            for label in ["w0", "w1"]:
                feat[f"btc_up_ratio_{label}"] = 0.5
                feat[f"btc_n_ticks_{label}"]  = 0.0
                feat[f"btc_vol_{label}"]       = 0.0

        # Zscore
        lag_up_ratios = []
        for lag_n in range(1, 6):
            prev_mid = slot_ts - lag_n * 300
            v = slot_up_ratio.get(prev_mid, None)
            lag_up_ratios.append(v if v is not None else 0.5)
        if len(lag_up_ratios) >= 20:
            mu20, sd20 = np.mean(lag_up_ratios), np.std(lag_up_ratios)
        else:
            mu20, sd20 = 0.5, 0.1
        feat["btc_up_ratio_zscore_20s"] = float(np.clip((cur_up_ratio - mu20) / (sd20 + 1e-9), -5, 5))

        if len(lag_up_ratios) >= 5:
            mu5, sd5 = np.mean(lag_up_ratios[:5]), np.std(lag_up_ratios[:5])
        else:
            mu5, sd5 = 0.5, 0.1
        feat["btc_up_ratio_zscore_5s"] = float(np.clip((cur_up_ratio - mu5) / (sd5 + 1e-9), -5, 5))
        feat["btc_up_w5_zscore"] = feat["btc_up_ratio_zscore_5s"]

        # ── C. OB features ──
        for col in ob:
            if col != "market_id":
                v = ob[col]
                feat[col] = float(v) if (v is not None and not (isinstance(v, float) and math.isnan(v))) else 0.0

        # ── D. LAG / PREV_SLOT features ──
        for lag_n in range(1, 6):
            prev_mid_ts = slot_ts - lag_n * 300
            feat[f"prev_slot_up_ratio_{lag_n}"] = float(slot_up_ratio.get(prev_mid_ts, 0.5))
            feat[f"lag_{lag_n}_outcome"]         = 0.5  # unknown at inference
            feat[f"prev_slot_n_ticks_{lag_n}"]   = 0.0
            feat[f"prev_slot_vol_{lag_n}"]        = 0.0

        prev_ur_1 = feat.get("prev_slot_up_ratio_1", 0.5)
        feat["lag_ur_zscore_5"]  = feat.get("btc_up_ratio_zscore_5s", 0.0)
        feat["lag_ur_zscore_20"] = feat.get("btc_up_ratio_zscore_20s", 0.0)

        # ── E. TEMPORAL features ──
        import datetime as dt_module
        dt_obj = dt_module.datetime.utcfromtimestamp(slot_ts)
        hour   = dt_obj.hour + dt_obj.minute / 60.0
        dow    = dt_obj.weekday()
        feat["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        feat["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        feat["dow_sin"]  = math.sin(2 * math.pi * dow / 7)
        feat["dow_cos"]  = math.cos(2 * math.pi * dow / 7)
        feat["hour_x_up_ratio"] = cur_up_ratio * (hour / 24.0)
        feat["hour_x_tw_ur"]    = feat.get("btc_tw_up_ratio", 0.5) * (hour / 24.0)

        # ── F. CROSS features ──
        feat["x_imb_x_ur"]         = feat.get("ob_imbalance", 0.0) * cur_up_ratio
        feat["x_vol_x_ur"]         = feat.get("btc_vol_1h", 0.0) * cur_up_ratio
        feat["x_spread_x_ur"]      = feat.get("ob_spread", 0.0) * cur_up_ratio
        feat["x_inslot_x_ur"]      = feat.get("btc_inslot_ret", 0.0) * cur_up_ratio
        feat["x_pre5m_x_ur"]       = feat.get("btc_pre_5m_ret", 0.0) * cur_up_ratio
        feat["x_tw_x_imb"]         = feat.get("btc_tw_up_ratio", 0.5) * feat.get("ob_imbalance", 0.0)
        feat["x_clob_vel_x_ur"]    = feat.get("clob_mid_velocity", 0.0) * cur_up_ratio

        feat["target"] = target
        rows.append(feat)

    log.info("Built %d feature rows (skipped %d no-OB)", len(rows), skipped)

    df_oos = pd.DataFrame(rows)
    y_true = df_oos["target"].values

    # Garantir todas as features do modelo
    for f in features:
        if f not in df_oos.columns:
            df_oos[f] = 0.0
    X_oos = df_oos[features].values.astype(np.float32)

    # ── Step 8: Predict ───────────────────────────────────────────────────
    log.info("Step 8: Predicting...")
    proba = model.predict_proba(X_oos)[:, 1]
    df_oos["pred_up"] = proba

    # AUC OOS
    from sklearn.metrics import roc_auc_score, brier_score_loss
    auc_oos   = roc_auc_score(y_true, proba)
    brier_oos = brier_score_loss(y_true, proba)
    acc_oos   = float(((proba >= 0.5) == y_true).mean())
    log.info("OOS metrics: AUC=%.4f | Brier=%.4f | Acc=%.4f", auc_oos, brier_oos, acc_oos)

    # ── Step 9: Simular trades ────────────────────────────────────────────
    log.info("Step 9: Simulating trades...")
    SPREAD = 0.02
    # ob_mid é o mid real do orderbook no momento da observação — melhor proxy disponível
    df_oos["sim_mid"] = df_oos["ob_mid"].clip(0.01, 0.99)
    df_oos["sim_ask"] = (df_oos["sim_mid"] + SPREAD / 2).clip(0.01, 0.99)
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
             len(df_trades), len(df_oos), len(df_trades)/len(df_oos)*100)

    results = []
    for _, row in df_trades.iterrows():
        ask  = row["sim_ask"]
        edge = row["edge_ask"]
        actual_up = bool(row["target"])

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
            "conf": row["pred_up"], "ask": ask, "edge": edge,
            "shares": shares, "cost": cost, "actual_up": actual_up, "pnl": pnl,
        })

    df_res = pd.DataFrame(results)

    if len(df_res) == 0:
        log.warning("Nenhum trade passou os filtros.")
        return {"n_trades": 0}

    wins      = (df_res["pnl"] > 0).sum()
    total     = len(df_res)
    win_rate  = wins / total * 100
    total_pnl = df_res["pnl"].sum()
    avg_pnl   = df_res["pnl"].mean()
    total_cost= df_res["cost"].sum()
    roi       = total_pnl / total_cost * 100 if total_cost > 0 else 0

    log.info("=" * 60)
    log.info("BACKTEST v28 — OOS (últimos 10%% | %d mercados)", len(df_oos))
    log.info("=" * 60)
    log.info("OOS AUC:     %.4f  (treino=%.4f)", auc_oos, wf_auc)
    log.info("OOS Brier:   %.4f", brier_oos)
    log.info("OOS Acc:     %.4f", acc_oos)
    log.info("---")
    log.info("Trades:      %d  (%d wins / %d losses)", total, wins, total - wins)
    log.info("Win rate:    %.1f%%", win_rate)
    log.info("Total P&L:   $%.2f", total_pnl)
    log.info("Avg P&L:     $%.2f / trade", avg_pnl)
    log.info("Total cost:  $%.2f deployed", total_cost)
    log.info("ROI:         %.1f%%", roi)
    log.info("Avg conf:    %.3f", df_res["conf"].mean())
    log.info("Avg ask:     %.3f", df_res["ask"].mean())
    log.info("Avg edge:    %.3f", df_res["edge"].mean())
    log.info("=" * 60)

    return {
        "version":    version,
        "wf_auc":     wf_auc,
        "oos_auc":    round(auc_oos, 4),
        "oos_brier":  round(brier_oos, 4),
        "oos_acc":    round(float(acc_oos), 4),
        "n_oos":      len(df_oos),
        "n_trades":   total,
        "win_rate":   round(win_rate, 1),
        "total_pnl":  round(total_pnl, 2),
        "avg_pnl":    round(avg_pnl, 2),
        "roi":        round(roi, 1),
    }


@app.local_entrypoint()
def main():
    result = backtest_v28.remote()
    if result:
        print("\n=== BACKTEST v28 SUMMARY ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
