"""Train and evaluate LGBMForecaster on the BTC 5-min Polymarket dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from btc_lab.config import DATASET_PATH
from pmlab.modeling.lgbm_baseline import LGBMForecaster

MODEL_PATH = Path(__file__).parent.parent / "artifacts" / "btc_model.pkl"

FEATURE_COLS = [
    "first_price",
    "last_price",
    "price_mean",
    "price_std",
    "price_min",
    "price_max",
    "price_momentum",
    "n_ticks",
    "price_at_25pct",
    "price_at_50pct",
    "price_at_75pct",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]


def main() -> None:
    print(f"Loading dataset from {DATASET_PATH} ...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Target distribution:\n{df['target'].value_counts()}")

    # Time-ordered split (80/20)
    df = df.sort_values("start_ts").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"\nTrain size: {len(train_df):,}  |  Test size: {len(test_df):,}")

    # Validate feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: Missing feature columns: {missing}")
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    else:
        feature_cols = FEATURE_COLS

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    print(f"\nFitting LGBMForecaster on {len(X_train)} training samples ...")
    model = LGBMForecaster(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
    )
    model.fit(X_train, y_train)

    print("Evaluating on test set ...")
    proba = model.predict_proba(X_test)  # shape (n, 2)
    prob_up = proba[:, 1]
    preds = (prob_up >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, prob_up)
    ll = log_loss(y_test, prob_up)

    print("\n=== Evaluation Results ===")
    print(f"  Test samples:      {len(y_test)}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Brier score:       {brier:.4f}")
    print(f"  Log loss:          {ll:.4f}")
    print(f"  Class distribution (test):\n{y_test.value_counts().to_string()}")
    print(f"  Predicted proba range: [{prob_up.min():.4f}, {prob_up.max():.4f}]")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
