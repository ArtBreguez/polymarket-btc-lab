# Full Bot Audit Protocol (2026-06-04)

Developed when user asked "bot esta na sua ultima versao com tudo auditado?" — the complete
verification protocol that ensures deployed code matches local AND model features match code.

## 3-Layer Audit

### Layer 1: Code Identity (local = deployed)
```bash
# MD5 comparison — byte-for-byte match required
/home/ubuntu/.fly/bin/flyctl ssh console -a polymarket-maker-mm \
  -C "md5sum /app/live_trader.py /app/data_quality_gate.py"
md5sum deploy/live_trader.py deploy/data_quality_gate.py
# BOTH must match. Any mismatch → redeploy.
```

### Layer 2: Model Identity (deployed loads correct version)
```bash
# Download champion from HF and verify version + metrics
export HF_TOKEN=$(grep HF_TOKEN ~/.env | cut -d= -f2)
python3 -c "
from huggingface_hub import hf_hub_download
import pickle, os
path = hf_hub_download('artbreguez/polymarket-btc-model', 'champion.pkl',
                        token=os.environ['HF_TOKEN'])
with open(path, 'rb') as f:
    data = pickle.load(f)
print(f'Version: {data[\"version\"]}')
print(f'AUC: {data[\"wf_auc\"]:.4f}')
print(f'Features ({len(data[\"features\"])}): {data[\"features\"]}')
"
```

### Layer 3: Feature Parity (all model features computed in code)
```bash
# Check each feature — but watch for dynamic/loop-generated ones!
for feat in <paste feature list>; do
  count=$(grep -c "$feat" deploy/live_trader.py)
  if [ "$count" -eq 0 ]; then
    echo "MISSING: $feat"
  fi
done

# Features that will show MISSING because they're generated in loops:
# btc_up_w{0-5}      → grep for "btc_up_w" and check range(6) loop
# prev_slot_up_ratio_{1-5} → grep for "prev_slot_up_ratio" and check lag loop
# btc_up_ratio_zscore_{5s,10s,20s} → grep for "btc_up_ratio_zscore" and check label loop
```

## Additional Checks
- `git status --short` → must be clean (no uncommitted changes)
- `git log --oneline -1 origin/main` vs `git log --oneline -1 HEAD` → must match (all pushed)
- `flyctl status -a polymarket-maker-mm` → machine state=started
- `flyctl apps list` → verify correct app name exists (was previously `polymarket-predictions-bot`, now `polymarket-maker-mm`)

## Key Discovery: champion.pkl structure
The model is a dict, NOT a bare sklearn model:
- Keys: `version`, `features`, `model`, `wf_auc`, `wf_brier`, `wf_acc`
- `model.feature_names_in_` = generic `Column_0..Column_N` (NOT real names)
- `features` list = real names in ORDER → feature ORDER is critical for parity
- Feature at position i in `features` list must correspond to column i built by `build_features()`
