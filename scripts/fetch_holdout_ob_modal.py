"""
fetch_holdout_ob_modal.py — Fetch OB features do holdout (7-10 jun) na Modal
Resume automaticamente se interrompido.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow>=18.0", "pandas>=2.2", "numpy>=1.26",
                 "requests>=2.31", "huggingface_hub>=0.26")
)

LOCAL_VOL = modal.Volume.from_name("btc-local-data")
app = modal.App("btc-holdout-ob", image=image)


@app.function(
    cpu=8, memory=16384, timeout=7200,
    secrets=[modal.Secret.from_name("pmdata-api-key"),
             modal.Secret.from_name("hf-token")],
    volumes={"/btc_local": LOCAL_VOL},
)
def fetch_holdout_ob():
    import gc, io, logging, os, sys, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import requests
    from huggingface_hub import HfApi

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    log = logging.getLogger(__name__)

    PMDATA_KEY    = os.environ.get("PMDATA_API_KEY", "")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    PMDATA_BASE   = "https://api.pmdata.dev/download/poly_l2"
    HF_REPO       = "artbreguez/polymarket-btc-model"
    LOCAL_DIR     = Path("/btc_local")
    OBS_SECS      = 60

    # Carregar markets do holdout (já fetchados no step 1)
    markets_path = LOCAL_DIR / "holdout_markets.csv"
    markets = pd.read_csv(markets_path)
    markets["market_id"] = markets["market_id"].astype(str)
    log.info("Holdout markets: %d", len(markets))

    # Resume: carregar OB já feito
    out_path = LOCAL_DIR / "holdout_ob_features.parquet"
    done_ids = set()
    existing_rows = []
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        done_ids = set(existing["market_id"].astype(str))
        existing_rows = existing.to_dict("records")
        log.info("Resume: %d já feitos", len(done_ids))

    todo = markets[~markets["market_id"].isin(done_ids)].reset_index(drop=True)
    log.info("Restantes: %d", len(todo))

    if len(todo) == 0:
        log.info("Nada a fazer!")
        return {"ok": len(done_ids), "failed": 0}

    # ── Feature extraction ────────────────────────────────────────────────
    def _snap(row):
        try:
            ap = np.array(row["ask_prices"], dtype=np.float64)
            as_ = np.array(row["ask_sizes"], dtype=np.float64)
            bp = np.array(row["bid_prices"], dtype=np.float64)
            bs = np.array(row["bid_sizes"], dtype=np.float64)
            if not len(ap) or not len(bp): return None
            mid = float((ap[0]+bp[0])/2); spread = float(ap[0]-bp[0])
            tsz = float(as_[0]+bs[0]); imb = float((bs[0]-as_[0])/tsz) if tsz>0 else 0.0
            bd5 = float(bs[bp>=mid-0.05].sum()/(bs.sum()+1e-8))
            ad5 = float(as_[ap<=mid+0.05].sum()/(as_.sum()+1e-8))
            dr = float(bd5/(ad5+1e-8)); td = float(bs.sum()+as_.sum())
            bwt = float(np.sum(bs*np.exp(-10*np.abs(bp-mid))))
            awt = float(np.sum(as_*np.exp(-10*np.abs(ap-mid))))
            wimb = float((bwt-awt)/(bwt+awt+1e-8))
            return {"mid":mid,"spread":spread,"imb":imb,"bd5":bd5,"ad5":ad5,
                    "dr":dr,"td":td,"wimb":wimb}
        except: return None

    def _slope(t, y):
        if len(t)<2: return 0.0
        t,y = np.array(t,dtype=np.float64), np.array(y,dtype=np.float64)
        tm,ym = t.mean(),y.mean(); d = np.sum((t-tm)**2)
        return float(np.sum((t-tm)*(y-ym))/d) if d>1e-12 else 0.0

    def _compute(books, pcs):
        if books is None or len(books)==0: return None
        books = books.sort_values("t_sec")
        open_r = books[books["t_sec"]<=30]
        s = _snap(open_r.iloc[0]) if len(open_r) else _snap(books.iloc[0])
        if not s: return None
        f = {"ob_imbalance":s["imb"],"ob_depth_ratio":s["dr"],"ob_total_depth":s["td"],
             "ob_spread":s["spread"],"ob_mid":s["mid"],"ob_bid_depth_5c":s["bd5"],
             "ob_ask_depth_5c":s["ad5"],"ob_weighted_imb":s["wimb"],
             "ob_mid_drift":0.0,"ob_imbalance_end":s["imb"],"ob_spread_end":s["spread"],
             "ob_depth_change":0.0,"ob_imb_momentum":0.0,
             "ob_pc_up_ratio":0.5,"ob_pc_volatility":0.0,"ob_pc_count":0.0,"ob_fill_imbalance":0.0}
        for wi,(t0,t1) in enumerate([(0,20),(20,40),(40,60)]):
            wr = books[(books["t_sec"]>=t0)&(books["t_sec"]<t1)]
            imbs = [x["imb"] for _,r in wr.iterrows() if (x:=_snap(r)) is not None]
            f[f"ob_imb_w{wi}"] = float(np.mean(imbs)) if imbs else 0.0

        # CLOB features (t=0..60s)
        wb = books[(books["t_sec"]>=0)&(books["t_sec"]<OBS_SECS)]
        wp = pcs[(pcs["t_sec"]>=0)&(pcs["t_sec"]<OBS_SECS)] if pcs is not None and len(pcs)>0 else pd.DataFrame()
        zeros = {"clob_imb_mean":0.0,"clob_imb_std":0.0,"clob_imb_drift":0.0,
                 "clob_spread_mean":0.0,"clob_spread_trend":0.0,"clob_mid_velocity":0.0,
                 "clob_mid_volatility":0.0,"clob_activity_rate":0.0,
                 "clob_depth_trend":0.0,"clob_ask_pressure":0.0}
        if len(wb)+len(wp)<3:
            f.update(zeros); return f
        ri,rd,rt,am,asp,at,ask_seq=[],[],[],[],[],[],[]
        for _,r in wb.iterrows():
            x=_snap(r)
            if x: ri.append(x["imb"]); rd.append(x["td"]); rt.append(float(r["t_sec"]))
            am.append(x["mid"] if x else 0); asp.append(x["spread"] if x else 0)
            at.append(float(r["t_sec"]))
        for _,r in wp.iterrows():
            try:
                ba=float(r["best_ask"]) if pd.notna(r.get("best_ask")) else None
                bb=float(r["best_bid"]) if pd.notna(r.get("best_bid")) else None
                if ba and bb and ba>bb>0:
                    am.append((ba+bb)/2); asp.append(ba-bb); at.append(float(r["t_sec"]))
                if str(r.get("pc_side","")).upper()=="SELL" and pd.notna(r.get("pc_price")):
                    ask_seq.append(float(r["pc_price"]))
            except: pass
        if len(am)<2: f.update(zeros); return f
        order=np.argsort(at); tA=np.array(at)[order]; mA=np.array(am)[order]; sA=np.array(asp)[order]
        tr=tA-tA[0]
        clob={"clob_imb_mean":float(np.mean(ri)) if ri else 0.0,
              "clob_imb_std":float(np.std(ri)) if len(ri)>1 else 0.0,
              "clob_imb_drift":float(ri[-1]-ri[0]) if len(ri)>1 else 0.0,
              "clob_spread_mean":float(np.mean(sA)),"clob_spread_trend":_slope(tr,sA),
              "clob_mid_velocity":_slope(tr,mA),
              "clob_mid_volatility":float(np.std(np.diff(mA))) if len(mA)>1 else 0.0,
              "clob_activity_rate":float((len(wb)+len(wp))/OBS_SECS),
              "clob_depth_trend":_slope(np.array(rt)-rt[0] if rt else [0],np.array(rd)) if len(rd)>1 else 0.0}
        if len(ask_seq)>=2:
            diffs=np.diff(ask_seq); moves=diffs[diffs!=0]
            clob["clob_ask_pressure"]=float((moves<0).sum()/len(moves)) if len(moves)>0 else 0.0
        else:
            clob["clob_ask_pressure"]=0.0
        f.update(clob); return f

    def _fetch_one(market_id, slug, slot_ts):
        try:
            r = requests.get(f"{PMDATA_BASE}/{slug}.parquet",
                             headers={"api_key": PMDATA_KEY}, timeout=30)
            if not r.ok: return market_id, None
            df = pd.read_parquet(io.BytesIO(r.content))
            if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["ts_sec"] = df["timestamp"].astype("int64")/1000.0
            else:
                df["ts_sec"] = pd.to_numeric(df["timestamp"],errors="coerce").fillna(0)
            df["t_sec"] = df["ts_sec"] - slot_ts
            df = df[(df["t_sec"]>=0)&(df["t_sec"]<OBS_SECS)]
            books = df[df["event_type"]=="book"].copy()
            pcs   = df[df["event_type"]=="price_change"].copy()
            feats = _compute(books if len(books)>0 else None,
                             pcs   if len(pcs)>0 else None)
            if feats: feats["market_id"] = market_id
            return market_id, feats
        except:
            return market_id, None

    # ── Fetch paralelo ────────────────────────────────────────────────────
    new_rows = []; ok = 0; fail = 0
    WORKERS = 40

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(_fetch_one, r["market_id"], r["slug"], r["slot_ts"]): r["market_id"]
            for _, r in todo.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            mid, feats = fut.result()
            if feats: new_rows.append(feats); ok += 1
            else: fail += 1
            if i % 100 == 0 or i == len(todo):
                log.info("  %d/%d — ok=%d fail=%d", i, len(todo), ok, fail)
                # Checkpoint
                if new_rows:
                    _df = pd.DataFrame(existing_rows + new_rows)
                    _df.to_parquet(out_path, index=False)
                    LOCAL_VOL.commit()

    all_rows = existing_rows + new_rows
    if all_rows:
        final_df = pd.DataFrame(all_rows)
        final_df.to_parquet(out_path, index=False)
        LOCAL_VOL.commit()
        log.info("OB final: %d mercados salvos", len(final_df))
    else:
        log.warning("Nenhuma feature OB computada")
        return {"ok": 0, "failed": fail}

    # ── Upload HF ─────────────────────────────────────────────────────────
    log.info("Uploading para HF...")
    api = HfApi(token=HF_TOKEN)
    for local_name, repo_path in [
        ("holdout_markets.csv",        "data/holdout_markets.csv"),
        ("holdout_ob_features.parquet","data/holdout_ob_features.parquet"),
    ]:
        p = LOCAL_DIR / local_name
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=repo_path,
                            repo_id=HF_REPO, repo_type="model",
                            commit_message=f"holdout: {local_name} (7-10 jun 2026)")
            log.info("  Uploaded %s", local_name)

    log.info("DONE — ok=%d fail=%d total=%d", ok, fail, len(all_rows))
    return {"ok": ok, "failed": fail, "total": len(all_rows)}


@app.local_entrypoint()
def main():
    result = fetch_holdout_ob.remote()
    print("\nRESULT:", result)
