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

    def predict(scenario: dict) -> float:
        row = {f: 0.0 for f in features}
        for k, v in scenario.items():
            if k in row:
                row[k] = v
        X = pd.DataFrame([row], columns=features)
        return float(model.predict_proba(X)[0, 1])

    # Neutral: realistic 50/50 order flow (not all-zeros which = extreme bearish for ratio features)
    neutral = {
        "btc_up_ratio": 0.50, "btc_momentum": 0.0, "btc_vwap_spread": 0.0,
        "btc_up_w0": 0.50, "btc_up_w1": 0.50, "btc_up_w2": 0.50,
        "btc_inslot_3m_ret": 0.0, "btc_pre_5m_ret": 0.0, "btc_pre_10m_ret": 0.0,
        "btc_n_ticks": 500.0, "btc_vol_up": 1000.0, "btc_vol_dn": 1000.0,
        "btc_buy_ratio": 0.50, "btc_avg_size": 50.0,
    }
    up_scene = {**neutral,
        "btc_up_ratio": 0.70, "btc_momentum": 0.15, "btc_vwap_spread": 0.05,
        "btc_up_w0": 0.65, "btc_up_w1": 0.68, "btc_up_w2": 0.72,
        "btc_inslot_3m_ret": 0.003, "btc_pre_5m_ret": 0.002,
        "btc_vol_up": 1600.0, "btc_vol_dn": 700.0,
    }
    dn_scene = {**neutral,
        "btc_up_ratio": 0.30, "btc_momentum": -0.15, "btc_vwap_spread": -0.05,
        "btc_up_w0": 0.35, "btc_up_w1": 0.32, "btc_up_w2": 0.28,
        "btc_inslot_3m_ret": -0.003, "btc_pre_5m_ret": -0.002,
        "btc_vol_up": 700.0, "btc_vol_dn": 1600.0,
    }
    strong_up = {**up_scene, "btc_up_ratio": 0.80, "btc_momentum": 0.25}

    prob_pos     = predict(up_scene)
    prob_neg     = predict(dn_scene)
    prob_neutral = predict(neutral)
    prob_sup     = predict(strong_up)

    print(f"  UP scenario    → P(UP) = {prob_pos:.3f}  (expect > P(neutral))")
    print(f"  DOWN scenario  → P(UP) = {prob_neg:.3f}  (expect < P(neutral))")
    print(f"  Neutral        → P(UP) = {prob_neutral:.3f}  (baseline)")
    print(f"  Strong UP      → P(UP) = {prob_sup:.3f}  (expect >= 0.40)")

    if prob_pos <= prob_neutral:
        fail(f"UP signal not above neutral: {prob_pos:.3f} <= {prob_neutral:.3f}")
    else:
        ok(f"UP signal directionally correct ({prob_pos:.3f} > {prob_neutral:.3f})")

    if prob_neg >= prob_neutral:
        fail(f"DOWN signal not below neutral: {prob_neg:.3f} >= {prob_neutral:.3f}")
    else:
        ok(f"DOWN signal directionally correct ({prob_neg:.3f} < {prob_neutral:.3f})")

    if prob_sup < 0.40:
        fail(f"Strong UP signal too weak: {prob_sup:.3f}")
    else:
        ok(f"Strong UP signal: {prob_sup:.3f}")

    if abs(prob_neutral - 0.5) > 0.25:
        print(f"  ⚠️  Calibration note: neutral={prob_neutral:.3f} (some bias present)")

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
