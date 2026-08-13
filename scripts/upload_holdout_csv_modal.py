"""Upload holdout_markets.csv para HF via Modal (sem Mount — lê local e passa como bytes)."""
import modal
from pathlib import Path

image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub>=0.26")
app = modal.App("upload-holdout-csv", image=image)

@app.function(secrets=[modal.Secret.from_name("hf-token")])
def upload(csv_bytes: bytes):
    import os, io
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj=io.BytesIO(csv_bytes),
        path_in_repo="data/holdout_markets.csv",
        repo_id="artbreguez/polymarket-btc-model",
        repo_type="model",
        commit_message="Add holdout_markets.csv",
    )
    print(f"Uploaded holdout_markets.csv ({len(csv_bytes)} bytes) OK")

@app.local_entrypoint()
def main():
    csv_path = Path(__file__).parent.parent / "data" / "holdout_markets.csv"
    csv_bytes = csv_path.read_bytes()
    print(f"Local CSV: {len(csv_bytes)} bytes")
    upload.remote(csv_bytes)
    print("Done.")
