"""
STEP 3+4+5: Train v4_clean model, walk-forward validation, calibration audit.
"""
import json
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore")

ARTIFACTS = "artifacts"

print("=" * 60)
print("STEP 3: Train v3 vs v4_clean (5-fold CV)")
print("=" * 60)

v3       = pd.read_parquet(f"{ARTIFACTS}/btc_5min_dataset_v3_clean.parquet")
v4_clean = pd.read_parquet(f"{ARTIFACTS}/btc_5min_dataset_v4_clean.parquet")

V3_FEATURES = [
    "first_price","last_price","price_mean","price_std","price_min","price_max",
    "price_momentum","n_ticks","price_at_25pct","price_at_50pct","price_at_75pct",
    "hour_of_day_sin","hour_of_day_cos","day_of_week_sin","day_of_week_cos",
    "total_volume_usdc","buy_volume_usdc","sell_volume_usdc","buy_sell_imbalance",
    "up_volume_usdc","down_volume_usdc","up_down_volume_ratio",
    "n_trades","n_buy_trades","n_sell_trades","avg_trade_size",
    "spot_price_start","spot_price_end","spot_return","spot_volatility",
    "spot_price_mean","vwap_up","vwap_down",
]
TEMPORAL_FEATURES = [c for c in v4_clean.columns if c not in v3.columns and c not in ("start_ts",)]
V4_FEATURES = [f for f in V3_FEATURES if f in v4_clean.columns] + \
              [f for f in TEMPORAL_FEATURES if f in v4_clean.columns]

TARGET = "target"
y = v4_clean[TARGET].values

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=10, colsample_bytree=0.8, subsample=0.8,
    random_state=42, verbose=-1
)

def cv_eval(df, features, label):
    X = df[features].fillna(0).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, accs, briers = [], [], []
    for _, (tr, va) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(**LGBM_PARAMS)
        m.fit(X[tr], y[tr],
              eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        prob = m.predict_proba(X[va])[:, 1]
        pred = m.predict(X[va])
        aucs.append(roc_auc_score(y[va], prob))
        accs.append(accuracy_score(y[va], pred))
        briers.append(brier_score_loss(y[va], prob))
    auc_m, acc_m, bs_m = np.mean(aucs), np.mean(accs), np.mean(briers)
    print(f"  [{label}]  AUC={auc_m:.4f}±{np.std(aucs):.4f}  "
          f"Acc={acc_m:.4f}±{np.std(accs):.4f}  Brier={bs_m:.4f}")
    return auc_m, acc_m, bs_m

v3_feats = [f for f in V3_FEATURES if f in v4_clean.columns]
v4_feats = V4_FEATURES

print(f"v3 features: {len(v3_feats)}  |  v4 features: {len(v4_feats)}")
auc_v3, acc_v3, bs_v3 = cv_eval(v4_clean, v3_feats, "v3 baseline")
auc_v4, acc_v4, bs_v4 = cv_eval(v4_clean, v4_feats, "v4_clean")

print(f"\nΔ AUC: {auc_v4 - auc_v3:+.4f}  Δ Acc: {acc_v4 - acc_v3:+.4f}  Δ Brier: {bs_v4 - bs_v3:+.4f}")

# Train final v4_clean model on all data
print("\nTraining final v4_clean model on all data...")
X_all = v4_clean[v4_feats].fillna(0).values
final = lgb.LGBMClassifier(**LGBM_PARAMS)
final.fit(X_all, y)
with open(f"{ARTIFACTS}/btc_model_v4_clean.pkl", "wb") as f:
    pickle.dump({"model": final, "features": v4_feats}, f)
print(f"Saved: {ARTIFACTS}/btc_model_v4_clean.pkl")

# Feature importances
imps = pd.DataFrame({"feature": v4_feats, "importance": final.feature_importances_})
imps = imps.sort_values("importance", ascending=False)
imps.to_csv(f"{ARTIFACTS}/feature_importances_v4_clean.csv", index=False)
print(f"Saved: {ARTIFACTS}/feature_importances_v4_clean.csv")
print("\nTop 15 features:")
print(imps.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("STEP 4: Walk-Forward (Rolling Origin) Validation — v4_clean")
print("=" * 60)

# Time-ordered (no shuffle) rolling origin
v4_sorted = v4_clean.sort_values("start_ts").reset_index(drop=True)
X_sorted = v4_sorted[v4_feats].fillna(0).values
y_sorted = v4_sorted[TARGET].values

INIT_TRAIN  = 300
STEP        = 30
N_FOLDS     = 10

results = []
for fold in range(N_FOLDS):
    train_end  = INIT_TRAIN + fold * STEP
    val_start  = train_end
    val_end    = min(val_start + STEP, len(v4_sorted))
    if val_end > len(v4_sorted):
        break
    X_tr, y_tr = X_sorted[:train_end], y_sorted[:train_end]
    X_va, y_va = X_sorted[val_start:val_end], y_sorted[val_start:val_end]
    if len(X_va) < 5:
        continue
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(-1)])
    prob = m.predict_proba(X_va)[:, 1]
    pred = m.predict(X_va)
    auc  = roc_auc_score(y_va, prob) if len(set(y_va)) > 1 else 0.5
    acc  = accuracy_score(y_va, pred)
    bs   = brier_score_loss(y_va, prob)
    avg_edge = float(np.mean(np.abs(prob - 0.5)))
    results.append({"fold": fold, "train_size": train_end, "val_size": len(X_va),
                    "auc": auc, "accuracy": acc, "brier": bs, "avg_edge": avg_edge})
    print(f"  Fold {fold+1:2d}: train={train_end} val={len(X_va)} | "
          f"AUC={auc:.3f} Acc={acc:.3f} Brier={bs:.3f}")

df_res = pd.DataFrame(results)
df_res.to_csv(f"{ARTIFACTS}/rolling_origin_results_v4.csv", index=False)
summary = {
    "accuracy":  {"mean": float(df_res.accuracy.mean()), "std": float(df_res.accuracy.std()),
                  "min": float(df_res.accuracy.min()), "max": float(df_res.accuracy.max())},
    "auc_roc":   {"mean": float(df_res.auc.mean()), "std": float(df_res.auc.std()),
                  "min": float(df_res.auc.min()), "max": float(df_res.auc.max())},
    "brier_score": {"mean": float(df_res.brier.mean()), "std": float(df_res.brier.std())},
    "avg_edge":  {"mean": float(df_res.avg_edge.mean())},
    "n_folds": len(df_res),
}
with open(f"{ARTIFACTS}/rolling_origin_summary_v4.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nWalk-forward summary:")
print(f"  AUC:  {summary['auc_roc']['mean']:.4f} ± {summary['auc_roc']['std']:.4f}")
print(f"  Acc:  {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
print(f"  Brier:{summary['brier_score']['mean']:.4f}")

print("\n" + "=" * 60)
print("STEP 5: Calibration Audit")
print("=" * 60)

# 5-fold CV: collect predicted probs vs actual labels
all_probs, all_labels = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all = v4_clean[v4_feats].fillna(0).values
for tr, va in skf.split(X_all, y):
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(X_all[tr], y[tr], callbacks=[lgb.log_evaluation(-1)])
    prob = m.predict_proba(X_all[va])[:, 1]
    all_probs.extend(prob)
    all_labels.extend(y[va])

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

# Brier score
bs_overall = brier_score_loss(all_labels, all_probs)
print(f"Overall Brier Score: {bs_overall:.4f}")

# Reliability table
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
print(f"\nCalibration Table ({n_bins} bins):")
print(f"{'Bin':>14} {'N':>6} {'Mean Pred':>10} {'Actual Freq':>12} {'Calibration Err':>16}")
print("-" * 62)
ece_num = 0.0
for i in range(n_bins):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask = (all_probs >= lo) & (all_probs < hi)
    if mask.sum() == 0:
        continue
    n      = mask.sum()
    mean_p = all_probs[mask].mean()
    freq   = all_labels[mask].mean()
    err    = abs(mean_p - freq)
    ece_num += err * n
    flag = " ← big gap" if err > 0.10 else ""
    print(f"  [{lo:.1f} – {hi:.1f}): {n:>5}  {mean_p:>9.3f}  {freq:>11.3f}  {err:>14.3f}{flag}")

ece = ece_num / len(all_probs)
print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
print(f"Interpretation: model probabilities are on avg {ece*100:.1f}% off from true frequencies")

# Final comparison table
print("\n" + "=" * 60)
print("FINAL COMPARISON: v3 vs v4_dirty vs v4_clean")
print("=" * 60)
with open(f"{ARTIFACTS}/rolling_origin_summary.json") as f:
    v3_ro = json.load(f)
print(f"{'Model':<15} {'RO AUC':>8} {'RO Acc':>8} {'CV Brier':>10}")
print("-" * 45)
print(f"{'v3_clean':<15} {v3_ro['auc_roc']['mean']:>8.4f} {v3_ro['accuracy']['mean']:>8.4f}  (prev session)")
print(f"{'v4_dirty (CV)':<15}  {'0.9890':>7}  {'0.9659':>7}  (reported, LEAKED)")
print(f"{'v4_clean (RO)':<15} {summary['auc_roc']['mean']:>8.4f} {summary['accuracy']['mean']:>8.4f} {summary['brier_score']['mean']:>10.4f}")
