"""
upload_data_to_hf.py — Sobe arquivos de data/ para o HF model repo
===================================================================
Deve ser rodado ANTES do treino Modal quando ob_features_full.parquet
foi atualizado localmente.

Usage:
    python3 scripts/upload_data_to_hf.py
    python3 scripts/upload_data_to_hf.py --file ob_features_full.parquet
    python3 scripts/upload_data_to_hf.py --all
"""
import argparse
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL_REPO = "artbreguez/polymarket-btc-model"
DATA_DIR     = Path(__file__).parent.parent / "data"

# Arquivos que o train_v31 consome do HF
ALL_DATA_FILES = [
    "ob_features_v31.parquet",
    "ob_features_full.parquet",
    "all_markets.csv",
    "new_markets.csv",
    "ticks_btc_full_clean.parquet",
    "new_ticks_pmdata.parquet",
    "binance_spot_full.parquet",
    "binance_spot_local.parquet",
]


def upload_file(filename: str, api):
    local = DATA_DIR / filename
    if not local.exists():
        print(f"  SKIP {filename} — não encontrado localmente")
        return False
    size_mb = local.stat().st_size / 1e6
    print(f"  Uploading {filename} ({size_mb:.1f} MB)...", end=" ", flush=True)
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=f"data/{filename}",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        token=HF_TOKEN,
    )
    elapsed = time.time() - t0
    print(f"OK ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Subir só esse arquivo (ex: ob_features_full.parquet)")
    parser.add_argument("--all",  action="store_true", help="Subir todos os arquivos de data/")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("HF_TOKEN não encontrado no .env!")
        sys.exit(1)

    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    if args.file:
        files = [args.file]
    elif args.all:
        files = ALL_DATA_FILES
    else:
        # default: ob_features_v31 (novo fetch)
        files = ["ob_features_v31.parquet"]

    print(f"Destino: {HF_MODEL_REPO}/data/")
    print(f"Arquivos: {files}")
    print()

    for f in files:
        upload_file(f, api)

    print("\nDone. Pronto para rodar o treino no Modal.")


if __name__ == "__main__":
    main()
