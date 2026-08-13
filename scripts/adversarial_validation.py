"""
adversarial_validation.py
=========================
Detecta drift de distribuição entre treino e holdout OOS.

Metodologia:
  1. Cria dataset binário: label=0 (treino), label=1 (holdout)
  2. Treina LightGBM pra distinguir os dois
  3. Se AUC > 0.6 → distribuição divergiu → features problemáticas identificadas
  4. Reporta as features com maior importância no classificador adversarial

Uso:
    modal run scripts/adversarial_validation.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pyarrow>=18.0", "pandas>=2.2", "lightgbm==4.6.0",
        "scikit-learn==1.6.0", "numpy>=1.26", "huggingface_hub>=0.26",
    )
)
app = modal.App("btc-adversarial-validation", image=image)
vol = modal.Volume.from_name("btc-data-cache", create_if_missing=True)


@app.function(
    cpu=4, memory=16384, timeout=1800,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/cache": vol},
)
def run_adversarial():
    import gc, logging, os, pickle, shutil, sys
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    HF_REPO  = "artbreguez/polymarket-btc-model"
    DATA_DIR = __import__("pathlib").Path("/cache")

    # ── Download dados ───────────────────────────────────────────────────────
    log.info("Downloading data from HF...")
    FILES = {
        "champion.pkl":                "champion.pkl",
        "ob_features_full.parquet":    "data/ob_features_full.parquet",
        "holdout_ob_features.parquet": "data/holdout_ob_features.parquet",
        "holdout_markets.csv":         "data/holdout_markets.csv",
        "all_markets.csv":             "data/all_markets.csv",
        "ticks_btc_full_clean.parquet":"data/ticks_btc_full_clean.parquet",
        "binance_spot_full.parquet":   "data/binance_spot_full.parquet",
    }
    NO_CACHE = {"champion.pkl", "holdout_markets.csv", "holdout_ob_features.parquet"}
    for local_name, hf_path in FILES.items():
        local = DATA_DIR / local_name
        if local.exists() and local_name not in NO_CACHE:
            log.info("  %s cached", local_name)
            continue
        if local.exists():
            local.unlink()
        try:
            hf_hub_download(repo_id=HF_REPO, filename=hf_path, token=HF_TOKEN,
                            repo_type="model", local_dir=str(DATA_DIR),
                            local_dir_use_symlinks=False)
            src = DATA_DIR / hf_path
            if src.exists() and not local.exists():
                shutil.move(str(src), str(local))
            log.info("  Downloaded %s", local_name)
        except Exception as e:
            log.warning("  Could not download %s: %s", local_name, e)

    # ── Carrega modelo e features ────────────────────────────────────────────
    with open(DATA_DIR / "champion.pkl", "rb") as f:
        bundle = pickle.load(f)
    FEATURES = bundle["features"]
    log.info("Champion: %s | %d features", bundle.get("version"), len(FEATURES))

    # ── Carrega mercados de treino ───────────────────────────────────────────
    log.info("Loading training markets...")
    all_markets = pd.read_csv(DATA_DIR / "all_markets.csv")
    TRAIN_CUTOFF = 1780773000  # 6 jun 19:10 UTC
    train_mkts = all_markets[
        all_markets["slot_ts"] < TRAIN_CUTOFF
    ].dropna(subset=["target"]).copy()
    log.info("Train markets: %d", len(train_mkts))

    # ── OB features treino ──────────────────────────────────────────────────
    ob_train = pd.read_parquet(DATA_DIR / "ob_features_full.parquet")
    ob_train["market_id"] = ob_train["market_id"].astype(str)

    # ── OB features holdout ─────────────────────────────────────────────────
    ob_hold = pd.read_parquet(DATA_DIR / "holdout_ob_features.parquet")
    ob_hold["market_id"] = ob_hold["market_id"].astype(str)

    holdout_mkts = pd.read_csv(DATA_DIR / "holdout_markets.csv")
    holdout_mkts["market_id"] = holdout_mkts["market_id"].astype(str)

    # ── Spot data ────────────────────────────────────────────────────────────
    log.info("Loading spot data...")
    spot = pd.read_parquet(DATA_DIR / "binance_spot_full.parquet")
    spot = spot.sort_values("timestamp_ms").drop_duplicates("timestamp_ms").reset_index(drop=True)
    spot_ts = spot["timestamp_ms"].values // 1000
    spot_px = spot["close"].values.astype(float)

    def spot_at(ts):
        idx = int(np.searchsorted(spot_ts, ts, side="right")) - 1
        return float(spot_px[idx]) if 0 <= idx < len(spot_px) else 0.0

    def pre_ret(slot_ts, h):
        px_now = spot_at(slot_ts + 60)
        px_prev = spot_at(slot_ts - h)
        return float(px_now / px_prev - 1) if px_prev > 0 else 0.0

    # ── Build features básicas de spot + OB para ambos os conjuntos ─────────
    log.info("Building feature matrix for adversarial classifier...")

    def build_row(slot_ts, ob_row):
        import math
        f = {}
        # Spot
        px = spot_at(slot_ts + 60)
        px0 = spot_at(slot_ts)
        f["btc_inslot_ret"]  = float(px / px0 - 1) if px0 > 0 else 0.0
        f["btc_pre_5m_ret"]  = pre_ret(slot_ts, 300)
        f["btc_pre_15m_ret"] = pre_ret(slot_ts, 900)
        f["btc_pre_1h_ret"]  = pre_ret(slot_ts, 3600)
        f["btc_dist_1k"]     = min((px/1000 - math.floor(px/1000)),
                                   math.ceil(px/1000) - px/1000) if px > 0 else 0.5
        # Temporal
        import datetime as dt
        d = dt.datetime.utcfromtimestamp(slot_ts)
        hour = d.hour + d.minute / 60.0
        f["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        f["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        f["dow_sin"]  = math.sin(2 * math.pi * d.weekday() / 7)
        f["dow_cos"]  = math.cos(2 * math.pi * d.weekday() / 7)
        # OB — excluindo ob_total_depth (drift no holdout) e slot_ts_norm (temporal leak)
        for col in ["ob_imbalance", "ob_depth_ratio", "ob_spread",
                    "clob_spread_mean", "clob_spread_trend", "clob_mid_volatility",
                    "clob_ask_pressure"]:
            v = ob_row.get(col, 0.0)
            f[col] = float(v) if v is not None and v == v else 0.0
        return f

    # Treino sample (max 5000 para ser rápido)
    ob_train_dict = {str(r["market_id"]): r.to_dict()
                     for _, r in ob_train.iterrows()}
    train_sample = train_mkts[train_mkts["market_id"].astype(str).isin(ob_train_dict)].sample(
        min(5000, len(train_mkts)), random_state=42)

    rows_train, rows_hold = [], []

    for _, row in train_sample.iterrows():
        mid = str(int(row["market_id"]))
        ob = ob_train_dict.get(mid, {})
        r = build_row(int(row["slot_ts"]), ob)
        r["_label"] = 0
        rows_train.append(r)

    ob_hold_dict = {str(r["market_id"]): r.to_dict()
                    for _, r in ob_hold.iterrows()}
    for _, row in holdout_mkts.iterrows():
        mid = str(row["market_id"])
        ob = ob_hold_dict.get(mid, {})
        r = build_row(int(row["slot_ts"]), ob)
        r["_label"] = 1
        rows_hold.append(r)

    df_adv = pd.DataFrame(rows_train + rows_hold).fillna(0.0)
    y_adv = df_adv.pop("_label").values
    feat_cols = [c for c in df_adv.columns]
    X_adv = df_adv[feat_cols].values.astype(np.float32)

    log.info("Adversarial dataset: %d train + %d holdout = %d total, %d features",
             len(rows_train), len(rows_hold), len(df_adv), len(feat_cols))

    # ── Treina classificador adversarial ────────────────────────────────────
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    importances = np.zeros(len(feat_cols))

    for fold, (tr, va) in enumerate(skf.split(X_adv, y_adv)):
        clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            num_leaves=15, min_child_samples=20, random_state=42, verbose=-1
        )
        clf.fit(X_adv[tr], y_adv[tr])
        proba = clf.predict_proba(X_adv[va])[:, 1]
        auc = roc_auc_score(y_adv[va], proba)
        aucs.append(auc)
        importances += clf.feature_importances_
        log.info("  Fold %d: AUC=%.4f", fold + 1, auc)

    mean_auc = float(np.mean(aucs))
    log.info("\n=== ADVERSARIAL VALIDATION RESULT ===")
    log.info("Mean AUC: %.4f (ideal=0.50, problematic>0.60)", mean_auc)

    if mean_auc > 0.60:
        log.warning("⚠️  DISTRIBUIÇÃO DIVERGIU (AUC=%.4f > 0.60) — features com drift:", mean_auc)
    else:
        log.info("✅  Distribuições similares (AUC=%.4f ≤ 0.60)", mean_auc)

    # Top features com maior discriminação
    ranked = sorted(zip(feat_cols, importances), key=lambda x: -x[1])
    log.info("\nTop features no classificador adversarial (sinal de drift):")
    for i, (name, imp) in enumerate(ranked[:15], 1):
        flag = " ⚠️ DRIFT" if imp > importances.mean() * 2 else ""
        log.info("  %2d. %-35s imp=%.1f%s", i, name, imp, flag)

    log.info("\nFeatures com maior drift (candidatas a remoção/normalização):")
    high_drift = [name for name, imp in ranked if imp > importances.mean() * 2]
    log.info("  %s", high_drift)

    return {
        "adversarial_auc": mean_auc,
        "high_drift_features": high_drift,
        "all_importances": dict(zip(feat_cols, importances.tolist())),
    }


@app.local_entrypoint()
def main():
    result = run_adversarial.remote()
    print("\n=== RESULT ===")
    print(f"Adversarial AUC: {result['adversarial_auc']:.4f}")
    print(f"High drift features: {result['high_drift_features']}")
