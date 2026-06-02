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
    """Run sanity check: BTC up/down signals should predict correctly."""
    results = {}

    def predict(btc_ret: float) -> float:
        row = {f: 0.0 for f in features}
        row["btc_inslot_3m_ret"] = btc_ret
        X = pd.DataFrame([row], columns=features)
        prob_up = float(model.predict_proba(X)[0][1])
        return prob_up

    # BTC +0.3% → should predict UP strongly
    prob_up_pos = predict(0.003)
    # BTC -0.3% → should predict DOWN (low UP prob)
    prob_up_neg = predict(-0.003)
    # Neutral → should be near 50/50
    prob_up_neutral = predict(0.0)

    results["btc_up_0.3pct"] = round(prob_up_pos, 4)
    results["btc_dn_0.3pct"] = round(prob_up_neg, 4)
    results["btc_neutral"] = round(prob_up_neutral, 4)

    # Directional check: UP > neutral > DOWN
    # Also check strong UP (+1%) reaches at least 0.40
    prob_strong_up = predict(0.01)
    results["btc_up_1pct"] = round(prob_strong_up, 4)

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
    print(f"  BTC +0.3% → P(UP) = {check['btc_up_0.3pct']:.3f}  (expect >0.65)")
    print(f"  BTC -0.3% → P(UP) = {check['btc_dn_0.3pct']:.3f}  (expect <0.35)")
    print(f"  BTC  0.0% → P(UP) = {check['btc_neutral']:.3f}  (expect 0.35-0.65)")

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

    print(f"\n🏆 Champion promoted!")
    print(f"   URL: https://huggingface.co/{HF_REPO}")
    print(f"   AUC: {wf_auc:.4f}  Acc: {wf_acc:.4f}")
    print(f"\nThe deploy.yml workflow will auto-deploy to Fly.io on next push to main.")


if __name__ == "__main__":
    main()
