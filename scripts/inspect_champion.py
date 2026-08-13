import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub>=0.26", "lightgbm==4.6.0", "scikit-learn", "numpy", "pandas"
)
app = modal.App("inspect-champion", image=image)

@app.function(secrets=[modal.Secret.from_name("hf-token")])
def inspect():
    import os, pickle
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="artbreguez/polymarket-btc-model", filename="champion.pkl",
        token=os.environ["HF_TOKEN"], repo_type="model", local_dir="/tmp"
    )
    with open(path, "rb") as f:
        b = pickle.load(f)
    print("version:", b.get("version"))
    print("wf_auc:", b.get("wf_auc"))
    feats = b["features"]
    print("n_features:", len(feats))
    for i, f in enumerate(feats):
        print(f"  [{i:02d}] {f}")
    # Importance
    import numpy as np
    imp = b["model"].feature_importances_
    ranked = sorted(zip(feats, imp), key=lambda x: -x[1])
    print("\nTop 10 importances:")
    for name, score in ranked[:10]:
        print(f"  {name}: {score:.1f}")

@app.local_entrypoint()
def main():
    inspect.remote()
