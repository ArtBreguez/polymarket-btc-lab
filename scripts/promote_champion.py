"""
promote_champion.py — Promote a trained model to champion on HuggingFace.

Usage:
    python scripts/promote_champion.py --model artifacts/btc_model_v3b_spot.pkl
    python scripts/promote_champion.py --model artifacts/my_model.pkl --notes "v4 with orderbook features"

What it does:
    1. Loads the model pkl and validates it (sanity check)
    2. Uploads it as champion.pkl to artbreguez/polymarket-btc-model on HuggingFace
    3. Uploads champion_meta.json with metadata
    4. Prints the HF URL for the new champion

The CI/CD pipeline (deploy.yml) automatically deploys to Fly.io
after a new champion is pushed to HF.
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HF_REPO = "artbreguez/polymarket-btc-model"


def sanity_check(model, features: list[str]) -> dict:
    """Run sanity check: directional signals should predict correctly.
    
    Feature-aware: detects which signal features exist in the model
    and sets them appropriately for UP/DOWN/neutral scenarios.
    """
    results = {}

    def predict(scenario: dict) -> float:
        row = {f: 0.0 for f in features}
        for k, v in scenario.items():
            if k in row:
                row[k] = v
        X = pd.DataFrame([row], columns=features)
        return float(model.predict_proba(X)[0, 1])

    # Neutral scenario: realistic 50/50 order flow
    neutral_scenario = {
        "btc_up_ratio": 0.50, "btc_momentum": 0.0,
        "btc_vwap_spread": 0.0, "btc_up_w0": 0.50,
        "btc_up_w1": 0.50, "btc_up_w2": 0.50,
        "btc_inslot_3m_ret": 0.0,
        "btc_pre_5m_ret": 0.0, "btc_pre_10m_ret": 0.0,
        "btc_n_ticks": 500.0, "btc_vol_up": 1000.0, "btc_vol_dn": 1000.0,
        "btc_buy_ratio": 0.50, "btc_avg_size": 50.0,
    }

    # UP scenario: order flow strongly bullish
    up_scenario = {**neutral_scenario,
        "btc_up_ratio": 0.70, "btc_momentum": 0.15,
        "btc_vwap_spread": 0.05,
        "btc_up_w0": 0.65, "btc_up_w1": 0.68, "btc_up_w2": 0.72,
        "btc_inslot_3m_ret": 0.003,
        "btc_pre_5m_ret": 0.002, "btc_pre_10m_ret": 0.003,
        "btc_vol_up": 1600.0, "btc_vol_dn": 700.0,
    }

    # DOWN scenario: order flow strongly bearish
    dn_scenario = {**neutral_scenario,
        "btc_up_ratio": 0.30, "btc_momentum": -0.15,
        "btc_vwap_spread": -0.05,
        "btc_up_w0": 0.35, "btc_up_w1": 0.32, "btc_up_w2": 0.28,
        "btc_inslot_3m_ret": -0.003,
        "btc_pre_5m_ret": -0.002, "btc_pre_10m_ret": -0.003,
        "btc_vol_up": 700.0, "btc_vol_dn": 1600.0,
    }

    prob_up_pos     = predict(up_scenario)
    prob_up_neg     = predict(dn_scenario)
    prob_up_neutral = predict(neutral_scenario)

    results["up_scenario"]   = round(prob_up_pos, 4)
    results["dn_scenario"]   = round(prob_up_neg, 4)
    results["neutral"]       = round(prob_up_neutral, 4)

    # Strong check: UP scenario must predict higher prob than DOWN scenario
    prob_strong_up = predict({**up_scenario,
                              "btc_up_ratio": 0.80, "btc_momentum": 0.25})
    results["strong_up"] = round(prob_strong_up, 4)

    passed = (
        prob_up_pos > prob_up_neutral
        and prob_up_neg < prob_up_neutral
        and prob_strong_up >= 0.40
    )
    results["passed"] = passed
    return results


def main():
    parser = argparse.ArgumentParser(description="Promote model to HF champion")
    parser.add_argument("--model", required=True, help="Path to model .pkl")
    parser.add_argument("--notes", default="", help="Optional release notes")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HuggingFace token")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual upload")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}")
        sys.exit(1)

    if not args.hf_token:
        print("ERROR: HF_TOKEN not set. Pass --hf-token or set HF_TOKEN env var.")
        sys.exit(1)

    # Load model
    print(f"Loading model: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    features = bundle["features"]
    wf_auc = bundle.get("wf_auc", 0.0)
    wf_acc = bundle.get("wf_acc", 0.0)

    print(f"  Features: {len(features)}")
    print(f"  WF AUC:   {wf_auc:.4f}")
    print(f"  WF Acc:   {wf_acc:.4f}")

    # Sanity check
    print("\nRunning sanity check...")
    check = sanity_check(model, features)
    print(f"  UP scenario    → P(UP) = {check['up_scenario']:.3f}  (expect > neutral)")
    print(f"  DOWN scenario  → P(UP) = {check['dn_scenario']:.3f}  (expect < neutral)")
    print(f"  Neutral        → P(UP) = {check['neutral']:.3f}  (baseline)")
    print(f"  Strong UP      → P(UP) = {check['strong_up']:.3f}  (expect >= 0.40)")

    if not check["passed"]:
        print("\nSANITY CHECK FAILED — model has wrong directional bias. Aborting.")
        sys.exit(1)

    print("  ✅ Sanity check passed\n")

    if args.dry_run:
        print("DRY RUN — skipping upload")
        return

    # Upload to HF
    from huggingface_hub import HfApi

    api = HfApi(token=args.hf_token)

    print(f"Uploading champion.pkl to {HF_REPO}...")
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="champion.pkl",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message=f"Champion: {model_path.name} | AUC={wf_auc:.3f} | {args.notes}",
    )
    print("  ✅ champion.pkl uploaded")

    # Upload metadata
    meta = {
        "model_file": model_path.name,
        "features": len(features),
        "wf_auc": wf_auc,
        "wf_acc": wf_acc,
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sanity_check": check,
        "notes": args.notes,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(meta, f, indent=2)
        tmp = f.name

    api.upload_file(
        path_or_fileobj=tmp,
        path_in_repo="champion_meta.json",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message=f"Champion metadata | AUC={wf_auc:.3f}",
    )
    os.unlink(tmp)
    print("  ✅ champion_meta.json uploaded")

    # Update HuggingFace model card
    print("\nUpdating HuggingFace model card...")
    try:
        from scripts.update_model_card import update_model_card
        update_model_card(meta, args.hf_token)
        print("  ✅ Model card updated on HuggingFace")
    except Exception as e:
        print(f"  ⚠️  Model card update failed (non-fatal): {e}")
        print("     Run manually: python scripts/update_model_card.py --hf-token $HF_TOKEN")

    print(f"\n🏆 Champion promoted!")
    print(f"   URL: https://huggingface.co/{HF_REPO}")
    print(f"   AUC: {wf_auc:.4f}  Acc: {wf_acc:.4f}")
    print(f"\nThe deploy.yml workflow will auto-deploy to Fly.io on next push to main.")


if __name__ == "__main__":
    main()
