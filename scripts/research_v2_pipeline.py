"""
Research v2: Advanced BTC Directional Model
- 7 cryptos x 616 slots = 4,312 samples (7x v1)
- Spot BTC/ETH/SOL features: return, vol, momentum pre-slot
- 3 sub-windows of order flow (60s each) + acceleration
- Cross-asset correlation features
- Hour-of-day bias analysis
- Walk-forward temporal validation
"""

import gc
import json
import logging
import math
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("research_v2")

REPO         = "BrockMisner/polymarket-btc-updown"
DATA_DIR     = Path("/home/ubuntu/polymarket-btc-lab/data")
ART_DIR      = Path("/home/ubuntu/polymarket-btc-lab/artifacts")
MARKETS_PATH = Path("/home/ubuntu/polymarket-btc-bot/data/raw/markets.parquet")
TICKS_BTC    = DATA_DIR / "ticks_btc_5min.parquet"
SPOT_PATH    = DATA_DIR / "data/spot_prices/part-0.parquet"
OBSERVE_SECS = 180

CRYPTOS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "HYPE"]
LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=63,
                   min_child_samples=15, colsample_bytree=0.7, subsample=0.8,
                   reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1)

# ── Phase 1: Ensure all ticks are downloaded ─────────────────────────────────
log.info("=" * 60)
log.info("PHASE 1: Checking / downloading data")
log.info("=" * 60)

tick_paths = {}
for crypto in CRYPTOS:
    if crypto == "BTC":
        tick_paths["BTC"] = TICKS_BTC
        log.info("BTC ticks: on disk")
        continue
    dest = DATA_DIR / f"data/ticks/crypto={crypto}/timeframe=5-minute/part-0.parquet"
    if dest.exists():
        log.info("%s ticks: on disk (%.0f MB)", crypto, dest.stat().st_size / 1e6)
        tick_paths[crypto] = dest
        continue
    log.info("Downloading %s ticks...", crypto)
    try:
        p = hf_hub_download(repo_id=REPO,
                            filename=f"data/ticks/crypto={crypto}/timeframe=5-minute/part-0.parquet",
                            repo_type="dataset", local_dir=str(DATA_DIR))
        tick_paths[crypto] = Path(p)
        log.info("  %s: %.0f MB", crypto, Path(p).stat().st_size / 1e6)
    except Exception as e:
        log.warning("  %s failed: %s", crypto, e)

log.info("Download phase complete.")

# ── Phase 2: Markets + spot feature lookup ────────────────────────────────────
log.info("=" * 60)
log.info("PHASE 2: Building spot feature lookup")
log.info("=" * 60)

markets = pd.read_parquet(MARKETS_PATH)
def _parse_slot(slug):
    try:
        p = str(slug).split("-")[-1].strip()
        return int(float(p)) if p else 0
    except Exception:
        return 0
markets["slot_ts"] = markets["slug"].apply(_parse_slot)
markets = markets[markets["slot_ts"] > 0]

# Collect all unique resolved slot timestamps across all cryptos
all_slots: list[int] = sorted(set(
    markets[
        (markets["timeframe"] == "5-minute") &
        (markets["resolution"] != -1) &
        (markets["crypto"].isin(CRYPTOS))
    ]["slot_ts"].tolist()
))
log.info("Unique resolved slots (all cryptos): %d", len(all_slots))

# Load spot, split by symbol, then free raw
log.info("Loading spot prices...")
spot_raw = pd.read_parquet(SPOT_PATH)
spot_raw["ts_s"] = spot_raw["ts_ms"] // 1000
spot_frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for sym, label in [("btcusdt", "btc"), ("ethusdt", "eth"), ("solusdt", "sol")]:
    sub = spot_raw[spot_raw["symbol"] == sym].sort_values("ts_s")
    spot_frames[label] = (sub["ts_s"].values.astype(np.int64),
                          sub["price"].values.astype(np.float64))
    log.info("  %s: %d rows", label, len(sub))
del spot_raw; gc.collect()

def build_spot_lookup(slots: list[int], ts_arr: np.ndarray,
                      px_arr: np.ndarray, label: str) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for slot_ts in slots:
        feat: dict[str, float] = {}
        for wname, lo, hi in [
            ("inslot_3m", slot_ts,       slot_ts + OBSERVE_SECS),
            ("pre_3m",    slot_ts - 180,  slot_ts),
            ("pre_15m",   slot_ts - 900,  slot_ts),
            ("pre_1h",    slot_ts - 3600, slot_ts),
        ]:
            mask = (ts_arr >= lo) & (ts_arr < hi)
            prices = px_arr[mask]
            if len(prices) < 2:
                feat[f"{label}_{wname}_ret"] = 0.0
                feat[f"{label}_{wname}_vol"] = 0.0
                feat[f"{label}_{wname}_mom"] = 0.0
            else:
                feat[f"{label}_{wname}_ret"] = float((prices[-1] - prices[0]) / (prices[0] + 1e-8))
                feat[f"{label}_{wname}_vol"] = float(np.std(np.diff(prices) / (prices[:-1] + 1e-8)))
                mid = prices[len(prices) // 2]
                feat[f"{label}_{wname}_mom"] = float((prices[-1] - mid) / (mid + 1e-8))
        mask1h = (ts_arr >= slot_ts - 3600) & (ts_arr < slot_ts)
        ph = px_arr[mask1h]
        rng = ph.max() - ph.min() if len(ph) > 1 else 1e-8
        feat[f"{label}_pct_of_1h_range"] = float((ph[-1] - ph.min()) / (rng + 1e-8)) if len(ph) > 0 else 0.5
        result[slot_ts] = feat
    return result

spot_lookup: dict[str, dict[int, dict]] = {}
for label, (ts_arr, px_arr) in spot_frames.items():
    log.info("  Building spot lookup: %s...", label)
    spot_lookup[label] = build_spot_lookup(all_slots, ts_arr, px_arr, label)
del spot_frames; gc.collect()
log.info("Spot lookup built, raw freed.")

# ── Phase 3: Multi-crypto dataset ─────────────────────────────────────────────
log.info("=" * 60)
log.info("PHASE 3: Building multi-crypto dataset")
log.info("=" * 60)

all_records = []

for crypto in CRYPTOS:
    tpath = tick_paths.get(crypto)
    if not tpath or not Path(tpath).exists():
        log.warning("No ticks for %s — skipping", crypto)
        continue

    log.info("--- %s ---", crypto)
    resolved = markets[
        (markets["crypto"] == crypto) &
        (markets["timeframe"] == "5-minute") &
        (markets["resolution"] != -1)
    ].copy()
    if len(resolved) == 0:
        continue

    log.info("  Resolved slots: %d  UP=%d DOWN=%d", len(resolved),
             (resolved["resolution"] == 1).sum(), (resolved["resolution"] == 0).sum())

    market_ids  = list(resolved["market_id"])
    slot_map    = dict(zip(resolved["market_id"], resolved["slot_ts"]))
    target_map  = dict(zip(resolved["market_id"], resolved["resolution"]))

    ticks = pq.read_table(
        str(tpath),
        columns=["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"],
        filters=[("market_id", "in", market_ids)],
    ).to_pandas()
    log.info("  Ticks: %d rows", len(ticks))

    ticks["slot_ts"] = ticks["market_id"].map(slot_map)
    ticks["t_sec"]   = (ticks["timestamp_ms"] / 1000) - ticks["slot_ts"]
    ticks["is_up"]   = ticks["outcome"].str.lower().isin(["up", "yes"])
    ticks["is_dn"]   = ticks["outcome"].str.lower().isin(["down", "no"])
    ticks["vol_up"]  = ticks["size_usdc"] * ticks["is_up"]
    ticks["vol_dn"]  = ticks["size_usdc"] * ticks["is_dn"]

    inslot = ticks[(ticks["t_sec"] >= 0) & (ticks["t_sec"] < OBSERVE_SECS)]
    log.info("  Inslot ticks [0-%ds): %d", OBSERVE_SECS, len(inslot))
    del ticks; gc.collect()

    n_built = 0
    for mid, grp in inslot.groupby("market_id"):
        target  = target_map.get(mid)
        if target is None or target == -1:
            continue
        slot_ts = slot_map[mid]
        n = len(grp)
        if n < 5:
            continue

        vu = grp["vol_up"].sum(); vd = grp["vol_dn"].sum(); tot = vu + vd
        nb = (grp["side"].str.upper() == "BUY").sum()
        wup = (grp["is_up"] * grp["price"] * grp["size_usdc"]).sum() / (vu + 1e-8)
        wdn = (grp["is_dn"] * grp["price"] * grp["size_usdc"]).sum() / (vd + 1e-8)

        def ws(lo, hi):
            w = grp[(grp["t_sec"] >= lo) & (grp["t_sec"] < hi)]
            wvu = w["vol_up"].sum(); wvd = w["vol_dn"].sum(); wt = wvu + wvd
            return wvu / (wt + 1e-8), wt, len(w)

        ur1, t1, n1 = ws(0, 60)
        ur2, t2, n2 = ws(60, 120)
        ur3, t3, n3 = ws(120, 180)

        px = grp["price"].values
        dt  = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        dow  = dt.weekday()

        rec: dict = {
            "market_id":       mid,
            "slot_ts":         slot_ts,
            "crypto":          crypto,
            "target":          int(target),
            # order flow
            "n_ticks":         float(n),
            "total_vol":       tot,
            "vol_up":          vu,
            "vol_dn":          vd,
            "up_ratio":        vu / (tot + 1e-8),
            "vwap_up":         wup,
            "vwap_dn":         wdn,
            "vwap_diff":       wup - wdn,
            "buy_ratio":       nb / (n + 1e-8),
            "avg_size":        grp["size_usdc"].mean(),
            "n_up":            float(grp["is_up"].sum()),
            "n_dn":            float(grp["is_dn"].sum()),
            # sub-windows
            "up_ratio_w1":     ur1, "vol_w1": t1, "n_w1": float(n1),
            "up_ratio_w2":     ur2, "vol_w2": t2, "n_w2": float(n2),
            "up_ratio_w3":     ur3, "vol_w3": t3, "n_w3": float(n3),
            "momentum_early":  ur2 - ur1,
            "momentum_late":   ur3 - ur2,
            "acceleration":    (ur3 - ur2) - (ur2 - ur1),
            "imbalance":       (vu - vd) / (tot + 1e-8),
            # token price
            "price_first":     float(px[0]),
            "price_last":      float(px[-1]),
            "price_trend":     float(px[-1] - px[0]),
            "price_vol":       float(np.std(px)) if len(px) > 1 else 0.0,
            # time
            "hour":            hour,
            "hour_sin":        math.sin(2 * math.pi * hour / 24),
            "hour_cos":        math.cos(2 * math.pi * hour / 24),
            "dow_sin":         math.sin(2 * math.pi * dow / 7),
            "dow_cos":         math.cos(2 * math.pi * dow / 7),
        }

        # Spot features (pre-computed lookup — O(1))
        for label in ["btc", "eth", "sol"]:
            slot_feats = spot_lookup[label].get(slot_ts, {})
            rec.update(slot_feats)

        all_records.append(rec)
        n_built += 1

    del inslot; gc.collect()
    log.info("  Built %d records", n_built)

df = pd.DataFrame(all_records).sort_values("slot_ts").reset_index(drop=True)
log.info("Dataset: %d rows, %d cols | UP=%d DOWN=%d",
         len(df), df.shape[1], (df.target == 1).sum(), (df.target == 0).sum())

df.to_parquet(ART_DIR / "research_v2_dataset.parquet", index=False)
log.info("Saved dataset.")

# ── Phase 4: Train & evaluate ─────────────────────────────────────────────────
log.info("=" * 60)
log.info("PHASE 4: Training & evaluation")
log.info("=" * 60)

META = {"market_id", "slot_ts", "crypto", "target"}
FEAT_COLS = [c for c in df.columns if c not in META]
X = df[FEAT_COLS].fillna(0).values
y = df["target"].values
log.info("Features: %d  Samples: %d", len(FEAT_COLS), len(X))

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs, accs, briers, all_prob, all_lbl = [], [], [], [], []
fold_preds = np.full(len(y), np.nan)

for fold, (tr, va) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(X[tr], y[tr], feature_name=FEAT_COLS,
          eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    prob = m.predict_proba(X[va])[:, 1]
    fold_preds[va] = prob
    aucs.append(roc_auc_score(y[va], prob))
    accs.append(accuracy_score(y[va], prob >= 0.5))
    briers.append(brier_score_loss(y[va], prob))
    all_prob.extend(prob); all_lbl.extend(y[va])
    log.info("  Fold %d: AUC=%.3f Acc=%.3f Brier=%.3f", fold+1, aucs[-1], accs[-1], briers[-1])

log.info("CV: AUC=%.4f±%.4f  Acc=%.4f±%.4f  Brier=%.4f",
         np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs), np.mean(briers))

# Walk-forward (time-ordered)
log.info("Walk-forward...")
n = len(df)
INIT = max(200, int(n * 0.55))
STEP = max(30, int(n * 0.04))
wf_results = []
for start in range(INIT, n - STEP, STEP):
    end = min(start + STEP, n)
    Xtr, ytr = X[:start], y[:start]
    Xva, yva = X[start:end], y[start:end]
    if len(set(yva)) < 2:
        continue
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(Xtr, ytr, feature_name=FEAT_COLS, callbacks=[lgb.log_evaluation(-1)])
    prob = m.predict_proba(Xva)[:, 1]
    wf_results.append({"n_train": start, "n_val": len(Xva),
                        "auc":   float(roc_auc_score(yva, prob)),
                        "acc":   float(accuracy_score(yva, prob >= 0.5)),
                        "brier": float(brier_score_loss(yva, prob))})
wf_df = pd.DataFrame(wf_results)
log.info("WF: AUC=%.4f±%.4f  Acc=%.4f±%.4f",
         wf_df.auc.mean(), wf_df.auc.std(), wf_df.acc.mean(), wf_df.acc.std())

# Hour-of-day bias
df["pred_prob"] = fold_preds
df["pred_correct"] = ((fold_preds >= 0.5) == y).astype(int)
hour_bias = df.groupby(df["hour"].astype(int)).agg(
    n=("target", "count"), acc=("pred_correct", "mean"), up_rate=("target", "mean")
).reset_index()
log.info("Hour bias:\n%s", hour_bias[["hour","n","acc","up_rate"]].to_string(index=False))

# Per-crypto breakdown
log.info("Per-crypto CV accuracy:")
for c in CRYPTOS:
    mask = df["crypto"] == c
    if mask.sum() == 0: continue
    preds = fold_preds[mask.values]
    trues = y[mask.values]
    valid = ~np.isnan(preds)
    if valid.sum() < 10: continue
    acc_c = accuracy_score(trues[valid], preds[valid] >= 0.5)
    auc_c = roc_auc_score(trues[valid], preds[valid]) if len(set(trues[valid])) > 1 else 0.5
    log.info("  %s: n=%d  acc=%.3f  auc=%.3f", c, mask.sum(), acc_c, auc_c)

# Final model
log.info("Training final model on all data...")
final = lgb.LGBMClassifier(**LGBM_PARAMS)
final.fit(X, y, feature_name=FEAT_COLS, callbacks=[lgb.log_evaluation(-1)])

imps = pd.DataFrame({"feature": FEAT_COLS, "importance": final.feature_importances_})
imps = imps.sort_values("importance", ascending=False)
log.info("Top 20 features:\n%s", imps.head(20).to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
bundle = {
    "model": final, "features": FEAT_COLS, "cryptos": CRYPTOS,
    "window": "first_3min", "cv_auc": float(np.mean(aucs)),
    "cv_acc": float(np.mean(accs)), "wf_auc": float(wf_df.auc.mean()),
    "wf_acc": float(wf_df.acc.mean()), "n_samples": len(df),
}
with open(ART_DIR / "btc_model_v2_research.pkl", "wb") as f:
    pickle.dump(bundle, f)

report = {
    "n_samples": int(len(df)), "n_features": int(len(FEAT_COLS)), "cryptos": CRYPTOS,
    "cv_auc": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)),
    "cv_acc": float(np.mean(accs)), "cv_brier": float(np.mean(briers)),
    "wf_auc": float(wf_df.auc.mean()), "wf_auc_std": float(wf_df.auc.std()),
    "wf_acc": float(wf_df.acc.mean()),
    "top_features": imps.head(20)[["feature","importance"]].to_dict("records"),
    "hour_bias": hour_bias.to_dict("records"),
    "wf_folds": wf_results,
}
with open(ART_DIR / "research_v2_report.json", "w") as f:
    json.dump(report, f, indent=2)

log.info("=" * 60)
log.info("DONE — n=%d  features=%d  CV AUC=%.4f  WF AUC=%.4f  WF Acc=%.4f",
         len(df), len(FEAT_COLS), np.mean(aucs), wf_df.auc.mean(), wf_df.acc.mean())
log.info("=" * 60)
