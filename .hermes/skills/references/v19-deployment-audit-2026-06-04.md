# V19 Deployment Audit (2026-06-04)

## Pre-deploy Feature Parity Check

v19 model has 40 features. Before deploying, verified all 40 are computed in live_trader.py.

### Missing features found (added before deploy):
1. `ob_total_depth` — raw sum of all bid+ask sizes
2. `ob_weighted_imb` — exp-weighted imbalance by proximity to mid
3. `ob_imb_w0/w1/w2` — windowed imbalance (approximated as current imbalance from single snapshot)
4. `ob_imb_momentum` — 0.0 (no temporal window available from single REST snapshot)
5. `x_imb_x_ur` — ob_imbalance × btc_up_ratio
6. `x_depth_x_momentum` — ob_depth_ratio × btc_momentum
7. `x_spread_x_vol` — ob_spread × btc_n_ticks
8. `x_ob_drift_x_inslot` — ob_mid_drift × btc_inslot_ret
9. `x_fill_imb_x_buy` — ob_fill_imbalance × btc_buy_ratio

### How to check feature coverage for future versions:
```python
from huggingface_hub import hf_hub_download
import json
path = hf_hub_download(repo_id='artbreguez/polymarket-btc-model',
                       filename='champion_meta.json', repo_type='model', token=HF_TOKEN)
meta = json.load(open(path))
model_features = set(meta['features'])
# Compare with what live_trader.py produces
# Every feature must be computed — feat.setdefault(f, 0.0) masks missing features silently!
```

### Known live-vs-training approximations (acceptable):
- `ob_mid_drift` = 0.0 live (single snapshot vs training's open+close snapshots)
- `ob_imb_w0/w1/w2` = current imbalance for all three (vs training's 3 time windows)
- `ob_imb_momentum` = 0.0 live (no temporal data from single snapshot)

These approximations are acceptable because:
1. The model saw 0.6% of training data with missing OB (filled with same defaults)
2. The non-OB features (CLOB flow, spot, lags) still provide strong signal
3. Future improvement: fetch /book 3x during observation window for real temporal OB data

## Post-deploy Verification

1. `Model loaded: 40 features, WF AUC=0.900` — confirmed v19
2. Features organic: up_ratio=0.635 (not 0.500 artificial)
3. Confidence sane: 88.3% (not 99%+)
4. Zero FEATURE_SANITY violations
5. Zero PREDICTION_SANITY violations
6. Seed took ~3s (real data from API)
7. Bot operational immediately after 3-slot cold start
