"""Rolling-origin (walk-forward) evaluation of LGBMForecaster on BTC 5-min dataset v3 clean.

Strategy:
- Initial training window: 300 rows
- Expand by ~30 rows per fold
- Evaluate on the next 30 rows
- No lookahead: model only sees past data at each fold
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmlab.modeling.lgbm_baseline import LGBMForecaster

DATASET_PATH = Path("/home/ubuntu/polymarket-btc-lab/artifacts/btc_5min_dataset_v3_clean.parquet")
RESULTS_CSV = Path("/home/ubuntu/polymarket-btc-lab/artifacts/rolling_origin_results.csv")
SUMMARY_JSON = Path("/home/ubuntu/polymarket-btc-lab/artifacts/rolling_origin_summary.json")

NON_FEATURE_COLS = {"market_id", "start_ts", "target"}

INITIAL_TRAIN = 300
STEP = 30


def main() -> None:
    print(f"Loading dataset from {DATASET_PATH} ...")
    df = pd.read_parquet(DATASET_PATH)
    df = df.sort_values("start_ts").reset_index(drop=True)
    print(f"  Shape: {df.shape}")
    print(f"  Target distribution:\n{df['target'].value_counts().to_string()}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    X_all = df[feature_cols].values
    y_all = df["target"].values
    n = len(df)

    # Build fold indices
    folds = []
    train_end = INITIAL_TRAIN
    while train_end + STEP <= n:
        test_end = min(train_end + STEP, n)
        folds.append((train_end, test_end))
        train_end += STEP

    print(f"\nTotal folds: {len(folds)}")
    print(f"  First fold: train=[0:{folds[0][0]}], test=[{folds[0][0]}:{folds[0][1]}]")
    print(f"  Last fold:  train=[0:{folds[-1][0]}], test=[{folds[-1][0]}:{folds[-1][1]}]")

    results = []

    for fold_idx, (train_end, test_end) in enumerate(folds):
        X_train = X_all[:train_end]
        y_train = y_all[:train_end]
        X_test = X_all[train_end:test_end]
        y_test = y_all[train_end:test_end]

        if len(np.unique(y_test)) < 2:
            print(f"  Fold {fold_idx+1:02d}: skipped (only one class in test set)")
            continue

        model = LGBMForecaster(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        model.fit(
            pd.DataFrame(X_train, columns=feature_cols),
            pd.Series(y_train),
        )

        proba = model.predict_proba(pd.DataFrame(X_test, columns=feature_cols))
        prob_up = proba[:, 1]
        preds = (prob_up >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, prob_up)
        brier = brier_score_loss(y_test, prob_up)
        avg_edge = float(np.mean(prob_up - 0.5))

        results.append({
            "fold": fold_idx + 1,
            "train_size": train_end,
            "test_start": train_end,
            "test_end": test_end,
            "test_size": test_end - train_end,
            "accuracy": acc,
            "auc_roc": auc,
            "brier_score": brier,
            "avg_edge": avg_edge,
        })

        print(
            f"  Fold {fold_idx+1:02d} | train={train_end:4d} | test=[{train_end}:{test_end}] "
            f"| acc={acc:.4f} | auc={auc:.4f} | brier={brier:.4f} | edge={avg_edge:+.4f}"
        )

    # Build summary
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("ROLLING-ORIGIN EVALUATION SUMMARY")
    print("=" * 80)
    metrics = ["accuracy", "auc_roc", "brier_score", "avg_edge"]
    summary = {}
    for m in metrics:
        vals = results_df[m].values
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        summary[m] = {"mean": mean_v, "std": std_v, "min": min_v, "max": max_v}
        print(f"  {m:<15s}  mean={mean_v:.4f}  std={std_v:.4f}  min={min_v:.4f}  max={max_v:.4f}")

    summary["n_folds"] = len(results)
    summary["initial_train_size"] = INITIAL_TRAIN
    summary["step_size"] = STEP

    print("=" * 80)
    print(f"\nTotal valid folds evaluated: {len(results)}")

    # Save outputs
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nPer-fold results saved to: {RESULTS_CSV}")

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {SUMMARY_JSON}")

    # Pretty per-fold table
    print("\nPer-fold table:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
