"""Train and evaluate LGBMForecaster on the BTC 5-min dataset v2 (with tick features)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmlab.modeling.lgbm_baseline import LGBMForecaster

DATASET_PATH = Path("/home/ubuntu/polymarket-btc-lab/artifacts/btc_5min_dataset_v2.parquet")
MODEL_PATH = Path("/home/ubuntu/polymarket-btc-lab/artifacts/btc_model_v2.pkl")

# Non-feature columns to exclude
NON_FEATURE_COLS = {"market_id", "start_ts", "target"}


def main() -> None:
    print(f"Loading dataset from {DATASET_PATH} ...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Target distribution:\n{df['target'].value_counts().to_string()}")

    # Time-ordered split (80/20)
    df = df.sort_values("start_ts").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"\nTrain size: {len(train_df):,}  |  Test size: {len(test_df):,}")

    # Determine feature columns
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    print(f"\nFitting LGBMForecaster on {len(X_train)} training samples ...")
    model = LGBMForecaster(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)

    print("Evaluating on test set ...")
    proba = model.predict_proba(X_test)  # shape (n, 2)
    prob_up = proba[:, 1]
    preds = (prob_up >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, prob_up)
    ll = log_loss(y_test, prob_up)
    auc = roc_auc_score(y_test, prob_up)

    print("\n=== Evaluation Results (v2 — with tick features) ===")
    print(f"  Test samples:      {len(y_test)}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Brier score:       {brier:.4f}")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  Log loss:          {ll:.4f}")
    print(f"  Class distribution (test):\n{y_test.value_counts().to_string()}")
    print(f"  Predicted proba range: [{prob_up.min():.4f}, {prob_up.max():.4f}]")

    # Feature importances
    try:
        importances = model._model.feature_importances_
        fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        print("\n=== Top 10 Feature Importances ===")
        for name, score in fi.head(10).items():
            print(f"  {name:<35s} {score:.1f}")
    except Exception as e:
        print(f"Could not get feature importances: {e}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
