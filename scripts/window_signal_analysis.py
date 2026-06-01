"""
AUDIT FINAL: Entender a estrutura temporal real dos dados e o que é "leakage" vs "signal".

Conclusões do deep audit:
1. v3 start_ts = primeiro tick registrado (~30min ANTES do slot)
2. O slot de 5min = [slug_ts, slug_ts + 300s]  
3. Os ticks cobrem a vida inteira do mercado (pré-slot + intra-slot)
4. O "leakage" não é clássico (resultado após resolução) — é temporal:
   os features usam ticks do período INTEIRO (3:13 a 3:51 para um slot 3:45-3:50)
5. Para trading real, precisamos features disponíveis ANTES do fim do slot

Estratégia correta:
- Usar APENAS ticks até t_sec < 60 dentro do slot (primeiro minuto)
- Ou usar features pré-slot (order flow antes do slot abrir)
- Comparar: early-slot features vs late-slot features vs full-window features

Este script faz:
1. Para cada mercado, divide ticks em: pre-slot, first-60s, full-slot
2. Treina modelos separados em cada janela
3. Mostra qual janela tem quanto sinal
"""
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pickle
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

TICKS_PATH   = '/home/ubuntu/polymarket-btc-lab/data/ticks_btc_5min.parquet'
MARKETS_PATH = '/home/ubuntu/polymarket-btc-bot/data/raw/markets.parquet'
V3_PATH      = '/home/ubuntu/polymarket-btc-lab/artifacts/btc_5min_dataset_v3_clean.parquet'
ARTIFACTS    = '/home/ubuntu/polymarket-btc-lab/artifacts'

print("Loading data...")
markets = pd.read_parquet(MARKETS_PATH)
btc_5m  = markets[(markets['crypto']=='BTC') & (markets['timeframe']=='5-minute') & (markets['resolution'] != -1)]
# Get slot_ts from slug
btc_5m = btc_5m.copy()
btc_5m['slot_ts'] = btc_5m['slug'].apply(lambda s: int(str(s).split('-')[-1]) if str(s).split('-')[-1].isdigit() else 0)
btc_5m = btc_5m[btc_5m['slot_ts'] > 0].set_index('market_id')
print(f"Markets with valid slots: {len(btc_5m)}")

v3 = pd.read_parquet(V3_PATH)
market_ids = v3['market_id'].tolist()
target_map = dict(zip(v3['market_id'], v3['target']))
slot_map   = {mid: btc_5m.loc[mid, 'slot_ts'] if mid in btc_5m.index else 0 for mid in market_ids}

print("Loading ticks...")
ticks = pq.read_table(
    TICKS_PATH,
    columns=["market_id", "timestamp_ms", "outcome", "side", "price", "size_usdc"],
    filters=[("market_id", "in", market_ids)]
).to_pandas()
print(f"Ticks loaded: {len(ticks):,}")

# Add slot-relative time
ticks['slot_ts'] = ticks['market_id'].map(slot_map)
ticks['t_sec']   = (ticks['timestamp_ms'] / 1000) - ticks['slot_ts']
ticks['is_up']   = ticks['outcome'] == 'Up'
ticks['is_down']  = ticks['outcome'] == 'Down'
ticks['vol_up']   = ticks['size_usdc'] * ticks['is_up']
ticks['vol_down'] = ticks['size_usdc'] * ticks['is_down']

def compute_features(grp, t_min, t_max, label):
    """Compute order-flow features for ticks in [t_min, t_max] seconds relative to slot_ts."""
    w = grp[(grp['t_sec'] >= t_min) & (grp['t_sec'] < t_max)]
    n = len(w)
    if n == 0:
        return {f"{label}_{k}": 0.0 for k in ['n', 'up_ratio', 'vol', 'vwap_up', 'vwap_dn', 'buy_ratio', 'avg_size', 'n_up', 'n_dn']}
    vol_up   = w['vol_up'].sum()
    vol_down = w['vol_down'].sum()
    total    = vol_up + vol_down
    n_buy    = (w['side'] == 'BUY').sum()
    vwap_up  = (w[w['is_up']]['price'] * w[w['is_up']]['size_usdc']).sum() / (vol_up + 1e-8)
    vwap_dn  = (w[w['is_down']]['price'] * w[w['is_down']]['size_usdc']).sum() / (vol_down + 1e-8)
    return {
        f"{label}_n":         float(n),
        f"{label}_up_ratio":  vol_up / (total + 1e-8),
        f"{label}_vol":       total,
        f"{label}_vwap_up":   vwap_up,
        f"{label}_vwap_dn":   vwap_dn,
        f"{label}_buy_ratio": n_buy / n,
        f"{label}_avg_size":  w['size_usdc'].mean(),
        f"{label}_n_up":      float(w['is_up'].sum()),
        f"{label}_n_dn":      float(w['is_down'].sum()),
    }

print("Computing features per market per window...")
records = []
for mid, grp in ticks.groupby('market_id'):
    target = target_map.get(mid)
    if target is None:
        continue
    row = {'market_id': mid, 'target': target}
    # Windows relative to slot start (t=0)
    row.update(compute_features(grp, -1800, 0,   'pre'))      # 30min before slot
    row.update(compute_features(grp, -600,  0,   'pre10'))    # 10min before slot
    row.update(compute_features(grp, 0,     60,  'first60'))  # first 60s of slot
    row.update(compute_features(grp, 0,     180, 'first3m'))  # first 3min
    row.update(compute_features(grp, 0,     300, 'full'))     # full 5min
    row.update(compute_features(grp, -1800, 300, 'all'))      # everything
    records.append(row)

df = pd.DataFrame(records)
print(f"Feature matrix: {df.shape}")

y = df['target'].values
LGBM_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15,
                   min_child_samples=10, random_state=42, verbose=-1)

def cv_score(feat_cols, label):
    X = df[feat_cols].fillna(0).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, accs = [], []
    for tr, va in skf.split(X, y):
        m = lgb.LGBMClassifier(**LGBM_PARAMS)
        m.fit(X[tr], y[tr], callbacks=[lgb.log_evaluation(-1)])
        prob = m.predict_proba(X[va])[:, 1]
        pred = m.predict(X[va])
        aucs.append(roc_auc_score(y[va], prob))
        accs.append(accuracy_score(y[va], pred))
    print(f"  {label:<30} AUC={np.mean(aucs):.4f}  Acc={np.mean(accs):.4f}")
    return np.mean(aucs), np.mean(accs)

print("\n" + "=" * 55)
print("SIGNAL ANALYSIS: what window has real predictive power?")
print("=" * 55)
pre_cols    = [c for c in df.columns if c.startswith('pre_')    and c != 'target']
pre10_cols  = [c for c in df.columns if c.startswith('pre10_')  and c != 'target']
f60_cols    = [c for c in df.columns if c.startswith('first60_') and c != 'target']
f3m_cols    = [c for c in df.columns if c.startswith('first3m_') and c != 'target']
full_cols   = [c for c in df.columns if c.startswith('full_')   and c != 'target']
all_cols    = [c for c in df.columns if c.startswith('all_')    and c != 'target']

cv_score(pre_cols,    "pre-slot (30min before)")
cv_score(pre10_cols,  "pre-slot (10min before)")
cv_score(f60_cols,    "first 60s of slot")
cv_score(f3m_cols,    "first 3min of slot")
cv_score(full_cols,   "full slot (0-300s)")
cv_score(all_cols,    "all ticks (pre+slot)")

print()
print("=" * 55)
print("INTERPRETATION:")
print("  pre-slot AUC > 0.7 → pre-slot order flow is predictive")
print("  first60 AUC > 0.8 → first minute is enough for signal")
print("  full_slot AUC >> first60 → signal builds during slot")
print()
print("For a REAL trading bot, use features available before placing the bet.")
print("If first60 or pre-slot has strong signal, we can enter early in slot.")

# Also check v3 features for comparison  
v3_feat_cols = [c for c in df.columns if c in [
    'up_down_volume_ratio','vwap_up','vwap_down','buy_sell_imbalance',
    'up_volume_usdc','down_volume_usdc','n_ticks'
]]
# Use 'all' features as proxy for v3 (which uses all ticks)
print(f"\nv3 baseline (full window) already computed above as 'all ticks'")

# Save this analysis
df.to_parquet(f'{ARTIFACTS}/btc_5min_window_analysis.parquet', index=False)
print(f"\nSaved: {ARTIFACTS}/btc_5min_window_analysis.parquet")
