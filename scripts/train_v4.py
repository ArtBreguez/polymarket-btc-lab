"""
train_v4.py — BTC 5min model v4

Goals vs v3b:
  1. class_weight='balanced' → fix DOWN calibration bias
  2. Full order flow features from local HF ticks (historical, clean)
  3. Spot price features from spot_prices parquet
  4. Isotonic calibration layer on top of LightGBM
  5. Optuna hyperparameter search
  6. Walk-forward validation
  7. Auto promote to HF if beats v3b champion (AUC 0.717)

Dataset sources (all local):
  - data/data/ticks/crypto=*/timeframe=5-minute/part-0.parquet (7 cryptos)
  - data/data/spot_prices/part-0.parquet
  - data/data/prices/crypto=BTC/timeframe=5-minute/part-0.parquet  (UP/DOWN token prices)
  - data/data/markets.parquet (via HF or btc-bot)
"""

import json
import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA        = ROOT / "data" / "data"
ARTIFACTS   = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

TICKS_DIR   = DATA / "ticks"
SPOT_PATH   = DATA / "spot_prices" / "part-0.parquet"
PRICES_PATH = DATA / "prices" / "crypto=BTC" / "timeframe=5-minute" / "part-0.parquet"

# Markets parquet — try two known locations
MARKETS_CANDIDATES = [
    ROOT / "data" / "markets.parquet",
    Path("/home/ubuntu/polymarket-btc-bot/data/raw/markets.parquet"),
    Path("/home/ubuntu/polymarket-lab/data/markets.parquet"),
]

BTC_TICKS_PATH = ROOT / "data" / "ticks_btc_5min.parquet"
MODEL_OUT   = ARTIFACTS / "btc_model_v4.pkl"
DATASET_OUT = ARTIFACTS / "btc_dataset_v4.parquet"

CRYPTOS    = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "HYPE"]
OBS_SECS   = 180        # first 3 min of each 5min slot
CHAMPION_AUC = 0.717    # v3b benchmark — must beat this to promote

HF_REPO    = "artbreguez/polymarket-btc-model"
HF_TOKEN   = os.environ.get("HF_TOKEN", "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")


# ── Step 1: Load markets ──────────────────────────────────────────────────────
def load_markets() -> pd.DataFrame:
    for p in MARKETS_CANDIDATES:
        if Path(p).exists():
            log.info("Loading markets from %s", p)
            mdf = pd.read_parquet(p)
            break
    else:
        raise FileNotFoundError("markets.parquet not found in any known path")

    btc = mdf[
        (mdf["crypto"] == "BTC") &
        (mdf["timeframe"] == "5-minute") &
        (mdf["resolution"].notna()) &
        (mdf["resolution"] != -1)
    ].copy()

    btc["slot_ts"] = btc["slug"].apply(
        lambda s: int(str(s).split("-")[-1]) if str(s).split("-")[-1].isdigit() else 0
    )
    btc = btc[btc["slot_ts"] > 0]
    log.info("BTC 5min markets: %d", len(btc))
    return btc


# ── Step 2: Load ticks per crypto ────────────────────────────────────────────
def load_ticks(crypto: str, market_ids: set = None) -> pd.DataFrame:
    if crypto == "BTC":
        # BTC uses the large consolidated file (67M rows)
        import pyarrow.parquet as pq
        filters = [("market_id", "in", list(market_ids))] if market_ids else None
        df = pq.read_table(
            str(BTC_TICKS_PATH),
            columns=["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"],
            filters=filters,
        ).to_pandas()
    else:
        p = TICKS_DIR / f"crypto={crypto}" / "timeframe=5-minute" / "part-0.parquet"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p, columns=["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"])
    df["crypto"] = crypto
    return df


# ── Step 3: Load spot prices ──────────────────────────────────────────────────
def load_spot() -> pd.DataFrame:
    log.info("Loading spot prices...")
    raw = pd.read_parquet(SPOT_PATH)
    # Format: ts_ms (epoch ms), symbol (btcusdt/ethusdt/solusdt), price, source
    # Pivot to wide: index=ts_s, cols=symbol
    raw["ts_s"] = raw["ts_ms"] // 1000
    # Use only binance source for consistency
    raw = raw[raw["source"] == "binance"].copy()
    pivot = raw.pivot_table(index="ts_s", columns="symbol", values="price", aggfunc="last")
    pivot = pivot.sort_index()
    log.info("  Spot pivot shape: %s  symbols: %s", pivot.shape, list(pivot.columns))
    return pivot


# ── Step 4: Build per-slot features ──────────────────────────────────────────
def tick_features(grp: pd.DataFrame, label: str) -> dict:
    """Order flow features for a group of ticks."""
    n      = len(grp)
    up     = grp["outcome"] == "Up"
    dn     = grp["outcome"] == "Down"
    vol_up = (grp["size_usdc"] * up).sum()
    vol_dn = (grp["size_usdc"] * dn).sum()
    total  = vol_up + vol_dn

    vwap_up = (grp.loc[up, "price"] * grp.loc[up, "size_usdc"]).sum() / (vol_up + 1e-8)
    vwap_dn = (grp.loc[dn, "price"] * grp.loc[dn, "size_usdc"]).sum() / (vol_dn + 1e-8)

    # Sub-windows
    w0 = grp[grp["t_sec"] < 60]
    w1 = grp[(grp["t_sec"] >= 60) & (grp["t_sec"] < 120)]
    w2 = grp[grp["t_sec"] >= 120]

    def ur(g):
        vu = (g["size_usdc"] * (g["outcome"] == "Up")).sum()
        t  = (g["size_usdc"]).sum()
        return vu / (t + 1e-8)

    up_ratio_w0 = ur(w0)
    up_ratio_w1 = ur(w1)
    up_ratio_w2 = ur(w2)

    return {
        f"{label}_n_ticks":     float(n),
        f"{label}_vol_up":      vol_up,
        f"{label}_vol_dn":      vol_dn,
        f"{label}_up_ratio":    vol_up / (total + 1e-8),
        f"{label}_vwap_up":     vwap_up,
        f"{label}_vwap_dn":     vwap_dn,
        f"{label}_vwap_spread": vwap_up - vwap_dn,
        f"{label}_buy_ratio":   (grp["side"] == "BUY").sum() / (n + 1e-8),
        f"{label}_avg_size":    grp["size_usdc"].mean() if n > 0 else 0.0,
        f"{label}_momentum":    up_ratio_w2 - up_ratio_w0,
        f"{label}_up_w0":       up_ratio_w0,
        f"{label}_up_w1":       up_ratio_w1,
        f"{label}_up_w2":       up_ratio_w2,
    }


def build_dataset(markets: pd.DataFrame) -> pd.DataFrame:
    slot_map   = dict(zip(markets["market_id"], markets["slot_ts"]))
    target_map = dict(zip(markets["market_id"], markets["resolution"].astype(int)))
    market_ids = set(markets["market_id"])

    # Load ticks for all cryptos
    all_ticks = {}
    for crypto in CRYPTOS:
        df = load_ticks(crypto, market_ids=market_ids if crypto == "BTC" else None)
        if df.empty:
            log.warning("No ticks for %s", crypto)
            continue
        if crypto == "BTC":
            df = df[df["market_id"].isin(market_ids)].copy()
            df["slot_ts"] = df["market_id"].map(slot_map)
            df["t_sec"] = df["timestamp_ms"] / 1000 - df["slot_ts"]
            df = df[(df["t_sec"] >= 0) & (df["t_sec"] < OBS_SECS)]
        log.info("  %s ticks: %d rows", crypto, len(df))
        all_ticks[crypto] = df

    btc_ticks = all_ticks.get("BTC", pd.DataFrame())
    if btc_ticks.empty:
        raise ValueError("No BTC ticks loaded")

    # Load spot prices — pivoted wide: index=ts_s, cols=btcusdt/ethusdt/solusdt
    spot = load_spot()
    btc_col = "btcusdt" if "btcusdt" in spot.columns else None
    eth_col = "ethusdt" if "ethusdt" in spot.columns else None
    sol_col = "solusdt" if "solusdt" in spot.columns else None
    log.info("  Spot cols available: btc=%s eth=%s sol=%s", btc_col, eth_col, sol_col)

    records = []
    btc_grps = dict(list(btc_ticks.groupby("market_id")))

    for mid, target in target_map.items():
        grp = btc_grps.get(mid)
        if grp is None or len(grp) < 5:
            continue

        slot_ts = slot_map[mid]
        feats   = {"market_id": mid, "slot_ts": slot_ts, "target": target}

        # BTC tick features
        feats.update(tick_features(grp, "btc"))

        # Time features
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        feats["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
        feats["dow_sin"]  = np.sin(2 * np.pi * dt.weekday() / 7)
        feats["dow_cos"]  = np.cos(2 * np.pi * dt.weekday() / 7)

        # Spot return features (pre-slot: 30m, 10m, 5m window)
        if btc_col:
            for window_s, label in [(1800, "30m"), (600, "10m"), (300, "5m")]:
                t0 = slot_ts - window_s
                t1 = slot_ts
                seg = spot.loc[t0:t1]
                if len(seg) >= 2:
                    p0 = float(seg[btc_col].iloc[0])
                    p1 = float(seg[btc_col].iloc[-1])
                    feats[f"btc_pre_{label}_ret"] = (p1 - p0) / (p0 + 1e-8)
                    if eth_col:
                        e0 = float(seg[eth_col].dropna().iloc[0]) if not seg[eth_col].dropna().empty else 0
                        e1 = float(seg[eth_col].dropna().iloc[-1]) if not seg[eth_col].dropna().empty else 0
                        feats[f"eth_pre_{label}_ret"] = (e1 - e0) / (e0 + 1e-8) if e0 > 0 else 0.0
                    if sol_col:
                        s0 = float(seg[sol_col].dropna().iloc[0]) if not seg[sol_col].dropna().empty else 0
                        s1 = float(seg[sol_col].dropna().iloc[-1]) if not seg[sol_col].dropna().empty else 0
                        feats[f"sol_pre_{label}_ret"] = (s1 - s0) / (s0 + 1e-8) if s0 > 0 else 0.0
                else:
                    feats[f"btc_pre_{label}_ret"] = 0.0
                    if eth_col: feats[f"eth_pre_{label}_ret"] = 0.0
                    if sol_col: feats[f"sol_pre_{label}_ret"] = 0.0

        records.append(feats)

    df = pd.DataFrame(records).sort_values("slot_ts").reset_index(drop=True)
    log.info("Dataset built: %d samples, %d features", len(df), len(df.columns) - 3)
    return df


# ── Step 5: Walk-forward evaluation ──────────────────────────────────────────
def walk_forward(df: pd.DataFrame, features: list[str], n_splits: int = 5) -> dict:
    df = df.sort_values("slot_ts").reset_index(drop=True)
    n  = len(df)
    fold_size = n // (n_splits + 1)

    aucs, accs, briers = [], [], []

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end  = min(train_end + fold_size, n)

        train = df.iloc[:train_end]
        test  = df.iloc[train_end:test_end]

        if len(test) < 20:
            continue

        X_tr = train[features].fillna(0)
        y_tr = train["target"]
        X_te = test[features].fillna(0)
        y_te = test["target"]

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]

        aucs.append(roc_auc_score(y_te, prob))
        accs.append(((prob >= 0.5) == y_te).mean())
        briers.append(brier_score_loss(y_te, prob))

    return {
        "wf_auc":    float(np.mean(aucs)),
        "wf_acc":    float(np.mean(accs)),
        "wf_brier":  float(np.mean(briers)),
        "fold_aucs": aucs,
    }


# ── Step 6: Optuna HPO ────────────────────────────────────────────────────────
def optuna_search(df: pd.DataFrame, features: list[str], n_trials: int = 40) -> dict:
    split = int(len(df) * 0.75)
    train = df.iloc[:split]
    val   = df.iloc[split:]

    X_tr = train[features].fillna(0)
    y_tr = train["target"]
    X_va = val[features].fillna(0)
    y_va = val["target"]

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "objective":        "binary",
            "class_weight":     "balanced",
            "verbose":          -1,
            "n_jobs":           -1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        prob = m.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("Optuna best AUC: %.4f  params: %s", study.best_value, study.best_params)
    return study.best_params


# ── Step 7: Train final model + calibrate ────────────────────────────────────
def train_final(df: pd.DataFrame, features: list[str], best_params: dict):
    X = df[features].fillna(0)
    y = df["target"]

    params = {**best_params, "objective": "binary", "class_weight": "balanced",
              "verbose": -1, "n_jobs": -1}
    base_model = lgb.LGBMClassifier(**params)

    # Isotonic calibration with cross-val (cv=3 time-aware)
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    cal_model = CalibratedClassifierCV(base_model, method="isotonic", cv=tscv)
    cal_model.fit(X, y)

    return cal_model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("BTC 5min Model v4 Training Pipeline")
    log.info("=" * 60)

    # 1. Load markets
    markets = load_markets()

    # 2. Build dataset
    if DATASET_OUT.exists():
        log.info("Loading cached dataset from %s", DATASET_OUT)
        df = pd.read_parquet(DATASET_OUT)
    else:
        log.info("Building dataset from scratch...")
        df = build_dataset(markets)
        df.to_parquet(DATASET_OUT, index=False)
        log.info("Dataset saved: %s", DATASET_OUT)

    # Class balance check
    vc = df["target"].value_counts()
    log.info("Target distribution: %s", dict(vc))

    NON_FEAT = {"market_id", "slot_ts", "target"}
    features = [c for c in df.columns if c not in NON_FEAT]
    log.info("Features: %d  Samples: %d", len(features), len(df))

    # 3. Walk-forward baseline (default params)
    log.info("Running walk-forward eval (default params)...")
    wf_base = walk_forward(df, features)
    log.info("Baseline WF AUC: %.4f  Acc: %.4f  Brier: %.4f",
             wf_base["wf_auc"], wf_base["wf_acc"], wf_base["wf_brier"])
    log.info("Fold AUCs: %s", [f"{x:.3f}" for x in wf_base["fold_aucs"]])

    # 4. Optuna HPO
    log.info("Running Optuna HPO (40 trials)...")
    best_params = optuna_search(df, features, n_trials=40)

    # 5. Walk-forward with best params
    log.info("Walk-forward with optimized params...")

    def wf_with_params(df, features, params):
        df = df.sort_values("slot_ts").reset_index(drop=True)
        n  = len(df)
        n_splits = 5
        fold_size = n // (n_splits + 1)
        aucs, accs, briers = [], [], []
        for i in range(n_splits):
            train_end = fold_size * (i + 1)
            test_end  = min(train_end + fold_size, n)
            train = df.iloc[:train_end]
            test  = df.iloc[train_end:test_end]
            if len(test) < 20:
                continue
            m = lgb.LGBMClassifier(**{**params, "objective": "binary",
                                       "class_weight": "balanced", "verbose": -1, "n_jobs": -1})
            m.fit(train[features].fillna(0), train["target"])
            prob = m.predict_proba(test[features].fillna(0))[:, 1]
            y_te = test["target"]
            aucs.append(roc_auc_score(y_te, prob))
            accs.append(((prob >= 0.5) == y_te).mean())
            briers.append(brier_score_loss(y_te, prob))
        return {"wf_auc": float(np.mean(aucs)), "wf_acc": float(np.mean(accs)),
                "wf_brier": float(np.mean(briers)), "fold_aucs": aucs}

    wf_opt = wf_with_params(df, features, best_params)
    log.info("Optimized WF AUC: %.4f  Acc: %.4f  Brier: %.4f",
             wf_opt["wf_auc"], wf_opt["wf_acc"], wf_opt["wf_brier"])
    log.info("Fold AUCs: %s", [f"{x:.3f}" for x in wf_opt["fold_aucs"]])

    # 6. Train final model with calibration
    log.info("Training final model with isotonic calibration...")
    final_model = train_final(df, features, best_params)

    # 7. Quick sanity check
    def predict(btc_ret):
        row = {f: 0.0 for f in features}
        row["btc_up_ratio"] = 0.5 + btc_ret * 50  # proxy: BTC up → more UP buyers
        row["btc_momentum"]  = btc_ret * 30
        X = pd.DataFrame([row], columns=features)
        return float(final_model.predict_proba(X)[0, 1])

    p_pos  = predict(0.003)
    p_zero = predict(0.0)
    p_neg  = predict(-0.003)
    log.info("Sanity check: UP(+0.3%%)=%.3f  Neutral=%.3f  DOWN(-0.3%%)=%.3f",
             p_pos, p_zero, p_neg)

    # 8. Save
    bundle = {
        "model":    final_model,
        "features": features,
        "wf_auc":   wf_opt["wf_auc"],
        "wf_acc":   wf_opt["wf_acc"],
        "wf_brier": wf_opt["wf_brier"],
        "version":  "v4",
        "n_samples": len(df),
        "class_balance": dict(vc),
        "best_params": best_params,
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Model saved: %s", MODEL_OUT)

    # 9. Compare with champion
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("  v3b (champion) WF AUC: %.4f", CHAMPION_AUC)
    log.info("  v4  (baseline) WF AUC: %.4f", wf_base["wf_auc"])
    log.info("  v4  (optimized) WF AUC: %.4f", wf_opt["wf_auc"])

    if wf_opt["wf_auc"] > CHAMPION_AUC:
        log.info("✅ v4 beats champion! Promoting to HF...")
        import subprocess, sys
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts" / "promote_champion.py"),
            "--model", str(MODEL_OUT),
            "--notes", f"v4 balanced+calibrated+hpo | AUC={wf_opt['wf_auc']:.4f}",
            "--hf-token", HF_TOKEN,
        ], capture_output=False)
        if result.returncode == 0:
            log.info("🏆 Champion promoted! CI/CD will auto-deploy.")
        else:
            log.error("Promote failed — check output above")
    else:
        log.info("❌ v4 does NOT beat champion (%.4f vs %.4f)",
                 wf_opt["wf_auc"], CHAMPION_AUC)
        log.info("   Model saved locally for analysis: %s", MODEL_OUT)

    log.info("=" * 60)
    log.info("Done.")


if __name__ == "__main__":
    main()
