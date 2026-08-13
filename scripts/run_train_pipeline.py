"""
run_train_pipeline.py — Upload dados + dispara treino v28 no Modal
==================================================================
Roda depois que o fetch local termina:
  1. Upload ob_features_full.parquet para HF
  2. modal run scripts/train_v28_modal.py

Usage:
    python3 scripts/run_train_pipeline.py
"""
import os, sys, subprocess, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
DATA_DIR = Path(__file__).parent.parent / "data"
LAB_DIR  = Path(__file__).parent.parent

def upload_ob_features():
    print("=" * 60)
    print("STEP 1: Upload ob_features_full.parquet → HuggingFace")
    print("=" * 60)

    if not HF_TOKEN:
        print("ERRO: HF_TOKEN não encontrado no .env")
        sys.exit(1)

    local = DATA_DIR / "ob_features_full.parquet"
    if not local.exists():
        print(f"ERRO: {local} não existe")
        sys.exit(1)

    import pandas as pd
    df = pd.read_parquet(local)
    print(f"ob_features_full.parquet: {len(df)} mercados, {len(df.columns)} colunas")

    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    size_mb = local.stat().st_size / 1e6
    print(f"Uploading {size_mb:.1f} MB...", end=" ", flush=True)
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo="data/ob_features_full.parquet",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        token=HF_TOKEN,
    )
    print(f"OK ({time.time()-t0:.1f}s)")


def run_train():
    print()
    print("=" * 60)
    print("STEP 2: modal run train_v28_modal.py")
    print("=" * 60)

    cmd = ["modal", "run", "scripts/train_v28_modal.py"]
    print(f"CMD: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(LAB_DIR))
    if result.returncode != 0:
        print(f"\nERRO: treino saiu com código {result.returncode}")
        sys.exit(result.returncode)
    print("\nTreino concluído!")


if __name__ == "__main__":
    upload_ob_features()
    run_train()
