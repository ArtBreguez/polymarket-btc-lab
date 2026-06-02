"""
validate_model.py — CI gate: validate the current champion model on HuggingFace.

Used in .github/workflows/ci.yml before any deploy.
Exits 0 if model passes all checks, non-zero otherwise.

Checks:
    1. champion.pkl exists on HF and is downloadable
    2. Feature count > 0
    3. Sanity check: BTC directional signals predict correctly
    4. WF AUC >= 0.65 (minimum bar for live deployment)
"""

import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HF_REPO = "artbreguez/polymarket-btc-model"
MIN_WF_AUC = 0.65
ERRORS = []


def fail(msg: str):
    ERRORS.append(msg)
    print(f"  ❌ {msg}")


def ok(msg: str):
    print(f"  ✅ {msg}")


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        fail("HF_TOKEN not set")
        sys.exit(1)

    # 1. Download champion.pkl
    print(f"Downloading champion.pkl from {HF_REPO}...")
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="champion.pkl",
            repo_type="model",
            token=hf_token,
            local_dir=tempfile.mkdtemp(),
            local_dir_use_symlinks=False,
        )
        ok(f"Downloaded: {path}")
    except Exception as e:
        fail(f"Download failed: {e}")
        sys.exit(1)

    # 2. Load model
    print("\nLoading model bundle...")
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        features = bundle["features"]
        wf_auc = bundle.get("wf_auc", 0.0)
        ok(f"Loaded: {len(features)} features, WF AUC={wf_auc:.4f}")
    except Exception as e:
        fail(f"Load failed: {e}")
        sys.exit(1)

    # 3. Feature count check
    if len(features) == 0:
        fail("Model has 0 features")
    else:
        ok(f"Feature count: {len(features)}")

    # 4. WF AUC gate
    if wf_auc < MIN_WF_AUC:
        fail(f"WF AUC {wf_auc:.4f} < minimum {MIN_WF_AUC}")
    else:
        ok(f"WF AUC {wf_auc:.4f} >= {MIN_WF_AUC}")

    # 5. Sanity check
    print("\nRunning sanity check...")

    def predict(btc_ret: float) -> float:
        row = {f: 0.0 for f in features}
        row["btc_inslot_3m_ret"] = btc_ret
        X = pd.DataFrame([row], columns=features)
        return float(model.predict_proba(X)[0][1])

    prob_pos = predict(0.003)
    prob_neg = predict(-0.003)
    prob_neutral = predict(0.0)

    print(f"  BTC +0.3% → P(UP) = {prob_pos:.3f}  (expect >0.65)")
    print(f"  BTC -0.3% → P(UP) = {prob_neg:.3f}  (expect <0.35)")
    print(f"  BTC  0.0% → P(UP) = {prob_neutral:.3f}  (expect 0.35-0.65)")

    if prob_pos <= 0.65:
        fail(f"UP signal too weak: {prob_pos:.3f}")
    else:
        ok("UP signal correct")

    if prob_neg >= 0.35:
        fail(f"DOWN signal too weak: {prob_neg:.3f}")
    else:
        ok("DOWN signal correct")

    if not (0.35 < prob_neutral < 0.65):
        fail(f"Neutral signal off: {prob_neutral:.3f}")
    else:
        ok("Neutral signal correct")

    # Summary
    print("\n" + "=" * 50)
    if ERRORS:
        print(f"VALIDATION FAILED — {len(ERRORS)} error(s):")
        for e in ERRORS:
            print(f"  • {e}")
        sys.exit(1)
    else:
        print("✅ ALL CHECKS PASSED — model is ready for deployment")
        sys.exit(0)


if __name__ == "__main__":
    main()
