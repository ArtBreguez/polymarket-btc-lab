"""Upload holdout_markets.csv para o HF model repo."""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub>=0.26")
app = modal.App("upload-holdout-csv", image=image)

@app.function(secrets=[modal.Secret.from_name("hf-token")])
def upload():
    import os
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj=open("/data/holdout_markets.csv", "rb"),
        path_in_repo="data/holdout_markets.csv",
        repo_id="artbreguez/polymarket-btc-model",
        repo_type="model",
    )
    print("Uploaded holdout_markets.csv")

# Monta o volume local como arquivo
@app.local_entrypoint()
def main():
    import os
    # Sobe o CSV local para o container via stdin
    csv_path = os.path.expanduser("~/polymarket-btc-lab/data/holdout_markets.csv")
    with open(csv_path, "rb") as f:
        data = f.read()

    import tempfile, pathlib
    # Usa upload direto via HF API com o secret do Modal
    import subprocess
    result = subprocess.run(
        ["modal", "secret", "list"],
        capture_output=True, text=True
    )
    print(result.stdout[:200])

    # Upload direto — lê o token do secret via Modal e faz upload
    from huggingface_hub import HfApi
    # O token aqui é o local — mas o Modal secret é diferente
    # Melhor: incluir o CSV na imagem e subir de lá
    print(f"CSV local: {len(data)} bytes")
    print("Use modal run scripts/upload_holdout_csv_v2.py")
