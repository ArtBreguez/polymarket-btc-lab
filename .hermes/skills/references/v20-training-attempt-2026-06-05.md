# v20 Training Attempt — 2026-06-05

## Result: NOT PROMOTED (0/3)

| Metric | v19 (champion) | v20 (candidate) | Delta |
|--------|---------------|-----------------|-------|
| AUC    | 0.9000        | 0.8996          | -0.0004 |
| Brier  | 0.1291        | 0.1295          | +0.0004 |
| Acc    | 0.8127        | 0.8119          | -0.0008 |

## What Was Attempted

1. **Dataset expansion via pmdata.dev** — fetched 8027 candidate slots (Feb 15-Mar 13 gap + Jun 3-5 new). ALL returned 0 markets because **the pmdata API key expired**.
2. **4 new features**: btc_vol_regime, btc_vol_accel, ob_depth_change (fixed), btc_funding_proxy
3. **TOP_N_FEATS** increased from 40 to 45

## Why It Failed

1. **pmdata key expired**: `sk-5uX...Ijko` returns `{"error":"API key is invalid or expired"}`. Without new data, training ran on identical 22,319 markets.
2. **Feature dilution**: 45 features on same data = noise. Only `btc_vol_accel` made top 45 (rank #7). The other 3 new features didn't rank.
3. **Fundamental insight confirmed**: MORE DATA > more features. v17 (+0.04 AUC from 601→7k), v18 (+0.004 from 7k→22k). v20 with same data and more features = regression.

## Top 45 Features (v20)
1. btc_inslot_ret
2. ob_mid_drift  
3. btc_pre_5m_ret
4. btc_vwap_up
5. x_ob_drift_x_inslot
6. btc_up_w1
7. **btc_vol_accel** ← NEW, rank 7 (promising)
8. btc_pre_30m_ret
...

OB/interaction features in top 45: 13
v20 new features in top 45: only btc_vol_accel

## Sanity Check
UP → 0.916 | Neutral → 0.478 | DOWN → 0.097 ✓ (directional ordering correct)

## Fold Detail
| Fold | AUC | Brier | Acc |
|------|-----|-------|-----|
| 0 | 0.8894 | 0.1358 | 0.7978 |
| 1 | 0.8926 | 0.1366 | 0.8059 |
| 2 | 0.9032 | 0.1270 | 0.8180 |
| 3 | 0.9031 | 0.1265 | 0.8166 |
| 4 | 0.9097 | 0.1218 | 0.8212 |

## Lessons

1. **Renew pmdata key** before any dataset expansion attempt
2. **btc_vol_accel is a good feature** — add it to v21 with same TOP_N_FEATS=40 (no dilution)
3. **Don't increase feature count without more data** — dilution hurts
4. **Potential v21 approach**: keep TOP_N_FEATS=40, add btc_vol_accel (replace lowest-ranked v19 feature), same data → might marginally improve

## Script
`scripts/train_v20_modal.py` (1057 lines) — includes inline dataset expansion + OB feature computation

## Data Uploaded to HuggingFace
All training data now on HF model repo under `data/`:
- data/all_markets.csv (1MB)
- data/binance_spot_full.parquet (5MB)
- data/ob_features_full.parquet (2.8MB)
- data/ticks_btc_full_clean.parquet (867MB)
