"""
Pipeline final: treina modelo com first-3min features, walk-forward, calibração, scan live.

Resultados do window_signal_analysis:
  pre-slot (30min):  AUC=0.50  ← ruído puro
  first 60s:         AUC=0.61  ← sinal fraco
  first 3min:        AUC=0.80  ← SINAL REAL, utilizável em produção
  full slot (5min):  AUC=0.97  ← sinal máximo (mas requer esperar o slot acabar)
  
Estratégia: entrar no slot após 3 minutos (em torno de t=180s) com o sinal dos primeiros 3 min.
Isso dá 2 minutos de janela para executar antes do slot fechar.
"""
import json, pickle
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
import warnings
warnings.filterwarnings("ignore")

TICKS_PATH   = '/home/ubuntu/polymarket-btc-lab/data/ticks_btc_5min.parquet'
MARKETS_PATH = '/home/ubuntu/polymarket-btc-bot/data/raw/markets.parquet'
ARTIFACTS    = '/home/ubuntu/polymarket-btc-lab/artifacts'

# ── 1. Build first-3min dataset ───────────────────────────────
print("=" * 58)
print("Step 1: Building first-3min dataset")
print("=" * 58)

markets = pd.read_parquet(MARKETS_PATH)
btc_5m  = markets[
    (markets['crypto']=='BTC') & (markets['timeframe']=='5-minute') &
    (markets['resolution'] != -1)
].copy()
btc_5m['slot_ts'] = btc_5m['slug'].apply(
    lambda s: int(str(s).split('-')[-1]) if str(s).split('-')[-1].isdigit() else 0)
btc_5m = btc_5m[btc_5m['slot_ts'] > 0].set_index('market_id')

target_map = dict(zip(btc_5m.index, btc_5m['resolution']))
slot_map   = dict(zip(btc_5m.index, btc_5m['slot_ts']))
market_ids = list(btc_5m.index)

print(f"Markets: {len(market_ids)}")

print("Loading ticks...")
ticks = pq.read_table(
    TICKS_PATH,
    columns=["market_id","timestamp_ms","outcome","side","price","size_usdc"],
    filters=[("market_id","in",market_ids)]
).to_pandas()
print(f"Ticks: {len(ticks):,}")

ticks['slot_ts'] = ticks['market_id'].map(slot_map)
ticks['t_sec']   = (ticks['timestamp_ms'] / 1000) - ticks['slot_ts']
ticks['is_up']   = ticks['outcome'] == 'Up'
ticks['is_down'] = ticks['outcome'] == 'Down'
ticks['vol_up']  = ticks['size_usdc'] * ticks['is_up']
ticks['vol_dn']  = ticks['size_usdc'] * ticks['is_down']

# First-3min window = [0, 180s)
inslot = ticks[(ticks['t_sec'] >= 0) & (ticks['t_sec'] < 180)].copy()
print(f"Inslot ticks (0-180s): {len(inslot):,}")

from datetime import datetime, timezone
slot_dt_map = {mid: datetime.fromtimestamp(slot_map[mid], tz=timezone.utc)
               for mid in market_ids}

records = []
for mid, grp in inslot.groupby('market_id'):
    target = target_map.get(mid)
    if target is None or target == -1:
        continue

    n = len(grp)
    vol_up  = grp['vol_up'].sum()
    vol_dn  = grp['vol_dn'].sum()
    total   = vol_up + vol_dn
    n_buy   = (grp['side'] == 'BUY').sum()
    n_up    = grp['is_up'].sum()
    n_dn    = grp['is_down'].sum()

    up_ratio   = vol_up / (total + 1e-8)
    vwap_up    = (grp[grp['is_up']]['price'] * grp[grp['is_up']]['size_usdc']).sum() / (vol_up + 1e-8)
    vwap_dn    = (grp[grp['is_down']]['price'] * grp[grp['is_down']]['size_usdc']).sum() / (vol_dn + 1e-8)
    buy_ratio  = n_buy / (n + 1e-8)
    avg_size   = grp['size_usdc'].mean()

    # Sub-window: first 60s vs next 120s (60-180s)
    w1 = grp[grp['t_sec'] < 60]
    w2 = grp[(grp['t_sec'] >= 60) & (grp['t_sec'] < 180)]
    vu1 = w1['vol_up'].sum(); vd1 = w1['vol_dn'].sum(); t1 = vu1 + vd1
    vu2 = w2['vol_up'].sum(); vd2 = w2['vol_dn'].sum(); t2 = vu2 + vd2
    up1 = vu1 / (t1 + 1e-8)
    up2 = vu2 / (t2 + 1e-8)
    momentum = up2 - up1  # is order flow accelerating toward UP or DOWN?

    # Spot price context from Binance: not available in historical data,
    # so use price_diff (vwap_up - vwap_dn) as proxy
    price_diff = vwap_up - vwap_dn

    dt = slot_dt_map.get(mid)
    hour = dt.hour + dt.minute / 60.0 if dt else 0
    dow  = dt.weekday() if dt else 0

    records.append({
        'market_id':   mid,
        'slot_ts':     slot_map[mid],
        'target':      int(target),
        # Core order flow
        'n_ticks':     float(n),
        'total_vol':   total,
        'vol_up':      vol_up,
        'vol_dn':      vol_dn,
        'up_ratio':    up_ratio,
        'vwap_up':     vwap_up,
        'vwap_dn':     vwap_dn,
        'buy_ratio':   buy_ratio,
        'avg_size':    avg_size,
        'n_up':        float(n_up),
        'n_dn':        float(n_dn),
        # Sub-window momentum
        'up_ratio_w1': up1,
        'up_ratio_w2': up2,
        'vol_w1':      t1,
        'vol_w2':      t2,
        'momentum':    momentum,  # positive = buyers accelerating toward UP
        # Derived
        'imbalance':   (vol_up - vol_dn) / (total + 1e-8),
        'price_diff':  price_diff,
        # Time features
        'hour_sin':    np.sin(2 * np.pi * hour / 24),
        'hour_cos':    np.cos(2 * np.pi * hour / 24),
        'dow_sin':     np.sin(2 * np.pi * dow / 7),
        'dow_cos':     np.cos(2 * np.pi * dow / 7),
    })

df = pd.DataFrame(records).sort_values('slot_ts').reset_index(drop=True)
print(f"Dataset: {df.shape}  |  UP={df.target.sum()}  DOWN={(df.target==0).sum()}")

FEAT_COLS = [c for c in df.columns if c not in ('market_id','slot_ts','target')]
X = df[FEAT_COLS].fillna(0).values
y = df['target'].values

df.to_parquet(f'{ARTIFACTS}/btc_5min_3min_dataset.parquet', index=False)
print(f"Saved: {ARTIFACTS}/btc_5min_3min_dataset.parquet")

# ── 2. 5-fold CV ──────────────────────────────────────────────
print("\n" + "=" * 58)
print("Step 2: 5-fold stratified CV")
print("=" * 58)

LGBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, colsample_bytree=0.8, subsample=0.8,
            random_state=42, verbose=-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs, accs, briers, all_prob, all_lbl = [], [], [], [], []
for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(**LGBM)
    m.fit(X[tr], y[tr],
          eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    prob = m.predict_proba(X[va])[:, 1]
    pred = m.predict(X[va])
    aucs.append(roc_auc_score(y[va], prob))
    accs.append(accuracy_score(y[va], pred))
    briers.append(brier_score_loss(y[va], prob))
    all_prob.extend(prob); all_lbl.extend(y[va])

print(f"  AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"  Acc:   {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"  Brier: {np.mean(briers):.4f} ± {np.std(briers):.4f}")

# ── 3. Walk-forward (time-ordered) ────────────────────────────
print("\n" + "=" * 58)
print("Step 3: Walk-forward validation (time-ordered)")
print("=" * 58)

INIT, STEP, FOLDS = 300, 30, 10
ro_results = []
for fold in range(FOLDS):
    te = INIT + fold * STEP
    vs, ve = te, min(te + STEP, len(df))
    if ve > len(df): break
    Xtr, ytr = X[:te], y[:te]
    Xva, yva = X[vs:ve], y[vs:ve]
    if len(Xva) < 5: continue
    m = lgb.LGBMClassifier(**LGBM)
    m.fit(Xtr, ytr, callbacks=[lgb.log_evaluation(-1)])
    prob = m.predict_proba(Xva)[:, 1]
    pred = m.predict(Xva)
    auc  = roc_auc_score(yva, prob) if len(set(yva)) > 1 else 0.5
    acc  = accuracy_score(yva, pred)
    bs   = brier_score_loss(yva, prob)
    ro_results.append({'fold':fold,'auc':auc,'acc':acc,'brier':bs,'n_val':len(Xva)})
    print(f"  Fold {fold+1:2d}: n_train={te:3d} n_val={len(Xva):2d} | AUC={auc:.3f} Acc={acc:.3f}")

ro_df = pd.DataFrame(ro_results)
ro_summary = {
    'auc':    {'mean': float(ro_df.auc.mean()), 'std': float(ro_df.auc.std()),
               'min': float(ro_df.auc.min()),   'max': float(ro_df.auc.max())},
    'accuracy': {'mean': float(ro_df.acc.mean()), 'std': float(ro_df.acc.std()),
                 'min': float(ro_df.acc.min()),   'max': float(ro_df.acc.max())},
    'brier':  {'mean': float(ro_df.brier.mean())},
    'n_folds': len(ro_df),
    'window':  'first_3min_of_slot',
}
print(f"\nWalk-forward summary:")
print(f"  AUC:  {ro_summary['auc']['mean']:.4f} ± {ro_summary['auc']['std']:.4f}")
print(f"  Acc:  {ro_summary['accuracy']['mean']:.4f} ± {ro_summary['accuracy']['std']:.4f}")
print(f"  Brier:{ro_summary['brier']['mean']:.4f}")

with open(f'{ARTIFACTS}/rolling_origin_3min.json', 'w') as f:
    json.dump(ro_summary, f, indent=2)

# ── 4. Calibration audit ──────────────────────────────────────
print("\n" + "=" * 58)
print("Step 4: Calibration audit")
print("=" * 58)

all_prob = np.array(all_prob)
all_lbl  = np.array(all_lbl)
bs_overall = brier_score_loss(all_lbl, all_prob)
print(f"Overall Brier: {bs_overall:.4f}")

n_bins = 10
edges = np.linspace(0, 1, n_bins + 1)
print(f"\nCalibration Table:")
print(f"  {'Bin':>14} {'N':>5} {'PredP':>7} {'ActualP':>8} {'|Err|':>7}")
print("  " + "-" * 48)
ece_num = 0.0
for i in range(n_bins):
    lo, hi = edges[i], edges[i+1]
    mask = (all_prob >= lo) & (all_prob < hi)
    if mask.sum() == 0: continue
    n = mask.sum()
    mp = all_prob[mask].mean()
    ap = all_lbl[mask].mean()
    err = abs(mp - ap)
    ece_num += err * n
    flag = " *" if err > 0.08 else ""
    print(f"  [{lo:.1f}–{hi:.1f}): {n:>4}  {mp:>6.3f}  {ap:>7.3f}  {err:>6.3f}{flag}")

ece = ece_num / len(all_prob)
print(f"\nECE: {ece:.4f}  ({ece*100:.1f}% avg calibration error)")

# ── 5. Feature importances & final model ─────────────────────
print("\n" + "=" * 58)
print("Step 5: Final model & feature importances")
print("=" * 58)

final = lgb.LGBMClassifier(**LGBM)
final.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

imps = pd.DataFrame({'feature': FEAT_COLS, 'importance': final.feature_importances_})
imps = imps.sort_values('importance', ascending=False)
print("Top 15 features:")
print(imps.head(15).to_string(index=False))

bundle = {'model': final, 'features': FEAT_COLS, 'window': 'first_3min', 'ro_summary': ro_summary}
with open(f'{ARTIFACTS}/btc_model_3min.pkl', 'wb') as f:
    pickle.dump(bundle, f)
print(f"\nSaved: {ARTIFACTS}/btc_model_3min.pkl")
imps.to_csv(f'{ARTIFACTS}/feature_importances_3min.csv', index=False)

# ── 6. Final summary ─────────────────────────────────────────
print("\n" + "=" * 58)
print("FINAL SUMMARY — all models compared")
print("=" * 58)
print(f"  {'Window':<28} {'CV AUC':>8} {'CV Acc':>8} {'WF AUC':>8} {'WF Acc':>8}")
print("  " + "-" * 60)
print(f"  {'pre-slot (random)':<28}  {'0.5044':>7}  {'0.5000':>7}")
print(f"  {'first 60s':<28}  {'0.6084':>7}  {'0.5649':>7}")
print(f"  {'v3 (full window, ~150min)':<28}  {'0.9860':>7}  {'0.9350':>7}  {'0.9632':>7}  {'0.9033':>7}")
print(f"  {'first 3min (PRODUCTION)':<28}  {np.mean(aucs):>7.4f}  {np.mean(accs):>7.4f}  {ro_summary['auc']['mean']:>7.4f}  {ro_summary['accuracy']['mean']:>7.4f}")
print()
print("CONCLUSION:")
print(f"  First-3min model: AUC={np.mean(aucs):.3f}, Acc={np.mean(accs)*100:.1f}%")
if np.mean(aucs) > 0.70:
    print("  ✅ Signal is real and useful for directional trading.")
    print("  Strategy: observe first 3 minutes of each BTC 5m slot,")
    print("  enter YES or NO at t=180s based on model output.")
    print("  Edge threshold: only bet when model confidence > 60%.")
else:
    print("  ⚠️  Signal weaker than expected. Investigate further.")
