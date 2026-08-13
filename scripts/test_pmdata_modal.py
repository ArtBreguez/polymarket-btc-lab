"""Script de diagnóstico — testa uma request pmdata dentro do Modal."""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("requests", "pyarrow", "pandas")
app = modal.App("btc-pmdata-test", image=image)

@app.function(secrets=[modal.Secret.from_name("pmdata-api-key")], timeout=60)
def test_one():
    import io, os, requests, traceback
    import pandas as pd

    key = os.environ.get("PMDATA_API_KEY", "")
    print(f"Key length: {len(key)} | prefix: {key[:8]}")

    slug = "btc-updown-5m-1773969600"
    url  = f"https://api.pmdata.dev/download/poly_l2/{slug}.parquet"
    try:
        r = requests.get(url, headers={"api_key": key}, timeout=30)
        print(f"Status: {r.status_code}")
        if r.ok:
            df = pd.read_parquet(io.BytesIO(r.content))
            print(f"OK — {len(df)} rows, cols: {list(df.columns)}")
        else:
            print(f"Error: {r.text[:300]}")
    except Exception:
        traceback.print_exc()

@app.local_entrypoint()
def main():
    test_one.remote()
