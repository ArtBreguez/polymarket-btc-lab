"""
upload_dataset_hf.py — Sobe todos os arquivos de dados para o HF (single source of truth)
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.26", "pyarrow>=18.0")
)

LOCAL_VOL = modal.Volume.from_name("btc-local-data")
app = modal.App("btc-upload-dataset-hf", image=image)


@app.function(
    cpu=2,
    memory=8192,
    timeout=3600,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/btc_local": LOCAL_VOL},
)
def upload_dataset():
    import logging, os, sys
    from pathlib import Path
    from huggingface_hub import HfApi

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    HF_REPO   = "artbreguez/polymarket-btc-model"
    LOCAL_DIR = Path("/btc_local")

    api = HfApi(token=HF_TOKEN)

    # Arquivos a subir: (path_local, path_no_repo)
    FILES = [
        ("all_markets.csv",              "data/all_markets.csv"),
        ("new_markets.csv",              "data/new_markets.csv"),
        ("ticks_btc_full_clean.parquet", "data/ticks_btc_full_clean.parquet"),
        ("new_ticks_pmdata.parquet",     "data/new_ticks_pmdata.parquet"),
        ("binance_spot_full.parquet",    "data/binance_spot_full.parquet"),
        ("binance_spot_local.parquet",   "data/binance_spot_local.parquet"),
        ("ob_features_full.parquet",     "data/ob_features_full.parquet"),
    ]

    for local_name, repo_path in FILES:
        local_path = LOCAL_DIR / local_name
        if not local_path.exists():
            log.warning("  SKIP (not found): %s", local_name)
            continue
        size_mb = local_path.stat().st_size / 1024 / 1024
        log.info("Uploading %s (%.1f MB) → %s", local_name, size_mb, repo_path)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=HF_REPO,
            repo_type="model",
            commit_message=f"data: update {local_name} (2026-06-10)",
        )
        log.info("  Done: %s", repo_path)

    log.info("=" * 50)
    log.info("Upload completo!")


@app.local_entrypoint()
def main():
    upload_dataset.remote()
