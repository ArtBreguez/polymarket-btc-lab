# BTC 5-min Model — Feature Brain
**Versão:** v28 (73f) | **OBS_SECS:** 60 | **SLOT_DURATION:** 300s  
**Última atualização:** 2026-06-10 | **Commit:** 6bc06fa  
**Auditoria paridade:** train=73 == live=73, diff=0 ✅  
**Auditoria viabilidade:** 73/73 ✅ OK — zero divergências, zero fallbacks estruturais

> Fonte de verdade de todas as features. Atualizar sempre que mudar `train_v28_modal.py` ou `live_trader.py`.

---

## Índice
1. [Timeline de um slot](#1-timeline-de-um-slot)
2. [Fontes de dados](#2-fontes-de-dados)
3. [Tabela canônica — 73 features (train == live == ✅ OK)](#3-tabela-canônica--73-features)
4. [Auditoria de lookahead](#4-auditoria-de-lookahead)
5. [Histório de remoções](#5-histórico-de-remoções)
6. [Grupo A — Spot Binance](#6-grupo-a--spot-binance)
7. [Grupo B — L2 Orderbook](#7-grupo-b--l2-orderbook)
8. [Grupo C — Tick-based Order Flow](#8-grupo-c--tick-based-order-flow)
9. [Grupo D — Lag History](#9-grupo-d--lag-history)
10. [Grupo E — Temporal](#10-grupo-e--temporal)
11. [Grupo F — Cross-domain Interactions](#11-grupo-f--cross-domain-interactions)
12. [Defaults quando dados ausentes](#12-defaults-quando-dados-ausentes)
13. [Seleção de features no treino](#13-seleção-de-features-no-treino)
14. [Checklist pré-treino](#14-checklist-pré-treino)
15. [Changelog de features por versão](#15-changelog-de-features-por-versão)

---

## 1. Timeline de um slot

```
slot_ts        +30s    +60s (OBS)   +108s       +168s    +170s      +240s   +300s
   |             |          |           |           |        |           |       |
   |<- tick W0 ->|<- W1 --->|           |           |        |           |       |
   |<- spot inslot ---------->|          |           |        |           |       |
   |<- CLOB WS acumulando -------------------------------->|   |           |       |
   |                          |<- lag ~90s ->|         |        |           |       |
   |                          |         clob_* janela  |        |           |       |
   |                                                   |<- entry window ->  |       |
   |                                                   | OB REST /book      |       |
   |                                                   | predict + order    |       |
                                                                        fim slot
```

- **t=0–30s**: `btc_up_w0` | **t=30–60s**: `btc_up_w1`
- **t=60s** (`obs_end = slot_ts + 60`): referência de spot e ticks
- **t=108–168s**: janela `clob_spread/mid/ask_pressure`
- **t=170s**: ticks disponíveis via data-api; OB REST chamado; predição
- **t=300s**: resultado resolvido; ring buffer atualizado

---

## 2. Fontes de dados

| Fonte | Lag live | Arquivo treino | Campos usados |
|---|---|---|---|
| Binance WS `btcusdt@kline_1m` | <2s | `binance_spot_full.parquet` | `k['c']` close, `k['h']` high, `k['l']` low, `k['v']` volume |
| Polymarket data-api `/trades` | ~90-120s | `ticks_btc_full_clean.parquet` | `outcome`, `side`, `price`, `size`, `timestamp` |
| Polymarket CLOB REST `/book` | <5s | `ob_features_full.parquet` | `asks/bids [{price,size}]` full book |
| Polymarket CLOB WS `price_change` | 0s | `ob_features_full.parquet` (`clob_*`) | `best_bid`, `best_ask`, `side`, `price` |
| Ring buffer de slots anteriores | 0s | `all_markets.csv` + tick aggregates | `target`, `up_ratio`, `n_ticks`, `vol_total` |

---

## 3. Tabela canônica — 73 features

**Todas as 73 features são ✅ OK: dado disponível, campos confirmados no código, sem divergência treino/live.**

### Grupo A — Spot Binance (10 features)

| Feature | Fórmula | Fonte live | Campos |
|---|---|---|---|
| `btc_pre_5m_ret` | `(px[obs_end] / px[slot_ts-300]) - 1` | Binance WS | `k['c']` close |
| `btc_pre_15m_ret` | `(px[obs_end] / px[slot_ts-900]) - 1` | idem | idem |
| `btc_pre_30m_ret` | `(px[obs_end] / px[slot_ts-1800]) - 1` | idem | idem |
| `btc_pre_1h_ret` | `(px[obs_end] / px[slot_ts-3600]) - 1` | idem | idem |
| `btc_inslot_ret` | `(px[-1]/px[0])-1` em `[slot_ts, obs_end]` | idem | idem |
| `btc_inslot_vol` | `std(px)/mean(px)` em `[slot_ts, obs_end]` | idem | idem |
| `btc_inslot_range` | `(max_hi - min_lo) / px_now` | Binance WS | `k['h']`, `k['l']` |
| `btc_vol_1h` | `std(diff(px)/px[:-1])` em `[slot_ts-3600, slot_ts]` | Binance WS | `k['c']` |
| `btc_dist_1k` | `min(frac(px/1k), 1-frac(px/1k))` | Binance WS | `k['c']` |
| `btc_spot_vol_ratio` | `vol[slot_ts-300:slot_ts] / (vol[slot_ts-3600:slot_ts-300]/11)` | Binance WS | `k['v']` |

### Grupo B — L2 Orderbook REST (12 features)

| Feature | Descrição | Snap | Fonte live |
|---|---|---|---|
| `ob_mid` | `(best_ask+best_bid)/2` | Open ~t60s | CLOB REST `/book` |
| `ob_spread` | `best_ask - best_bid` | Open | idem |
| `ob_imbalance` | `(bid_sz-ask_sz)/(bid_sz+ask_sz)` best-level | Open | idem |
| `ob_depth_ratio` | `bid_depth_5c / ask_depth_5c` | Close | idem |
| `ob_bid_depth_5c` | `Σbid_sz(p≥mid-0.05) / Σbid_sz` | Close | idem |
| `ob_ask_depth_5c` | `Σask_sz(p≤mid+0.05) / Σask_sz` | Close | idem |
| `ob_total_depth` | `Σbid_sz + Σask_sz` | Close | idem |
| `ob_weighted_imb` | `Σ(bid_sz·e^(-10Δp) - ask_sz·e^(-10Δp)) / Σwt` | Close | idem |
| `ob_imbalance_end` | `imbalance` do close snapshot | Temporal | idem |
| `ob_spread_end` | `spread` do close snapshot | Temporal | idem |
| `ob_depth_change` | `depth_close - depth_open` | Temporal | idem |
| `ob_imb_momentum` | `imb_close - imb_open` | Temporal | idem |

### Grupo B — CLOB WebSocket (5 features)

| Feature | Fórmula | Janela | Fonte live | Campos |
|---|---|---|---|---|
| `clob_spread_mean` | `mean(spread)` | `[108,168s)` | CLOB WS `price_change` | `best_bid`, `best_ask` → synthetic snap |
| `clob_spread_trend` | `linslope(t, spread)` | idem | idem | idem |
| `clob_mid_velocity` | `linslope(t, mid)` | idem | idem | idem |
| `clob_mid_volatility` | `std(diff(mid))` | idem | idem | idem |
| `clob_ask_pressure` | `frac(SELL price moves < 0)` | idem | CLOB WS `price_change` | `side`, `price` |

> Esses 5 usam **synthetic BookSnapshots** construídos de `best_bid`/`best_ask` presentes em cada `price_change` — disponíveis continuamente. ✓

### Grupo C — Tick-based Order Flow (9 features)

| Feature | Fórmula | Fonte live | Campos |
|---|---|---|---|
| `btc_up_ratio` | `vol_up / (vol_up+vol_dn)` | data-api `/trades` | `outcome`, `price`, `size` |
| `btc_n_ticks` | `len(ticks)` | idem | idem |
| `btc_buy_ratio` | `Σsz(side=BUY) / total` | idem | `side` |
| `btc_momentum` | `btc_up_w1 - btc_up_w0` | idem | idem |
| `btc_size_disparity` | `avg_up_size - avg_dn_size` | idem | `outcome`, `size` |
| `btc_up_ratio_stability` | `std(btc_up_w0, btc_up_w1)` | idem | idem |
| `btc_tw_up_ratio` | `Σ(up·sz·e^(-0.02·(60-t))) / Σ(sz·e^(-0.02·(60-t)))` | idem | `outcome`, `size`, `timestamp` |
| `btc_up_w0` | `up_ratio` em `[0, 30s)` | idem | idem |
| `btc_up_w1` | `up_ratio` em `[30, 60s)` | idem | idem |

### Grupo D — Lag History (25 features)

| Feature | Fonte live |
|---|---|
| `lag_{1..5}_outcome` | `_slot_history[-lag]['target']` |
| `prev_slot_up_ratio_{1..5}` | `_slot_history[-lag]['up_ratio']` |
| `prev_slot_n_ticks_{1..5}` | `_slot_history[-lag]['n_ticks']` |
| `prev_slot_vol_{1..5}` | `_slot_history[-lag]['vol_total']` |
| `lag_streak` | consecutive same-dir outcomes from ring buffer |
| `lag_ur_zscore_20` | `clip((prev_ur_1 - mean(hist_20)) / std(hist_20), -5, 5)` |
| `lag_ur_zscore_5` | `clip((prev_ur_1 - mean(hist_5)) / std(hist_5), -5, 5)` |
| `btc_up_ratio_zscore_20s` | `clip((cur_ur - mean(hist_20)) / std(hist_20), -5, 5)` |
| `btc_up_ratio_zscore_5s` | `clip((cur_ur - mean(hist_5)) / std(hist_5), -5, 5)` |

### Grupo E — Temporal (6 features)

| Feature | Fórmula |
|---|---|
| `hour_sin` | `sin(2π·hour/24)` |
| `hour_cos` | `cos(2π·hour/24)` |
| `dow_sin` | `sin(2π·weekday/7)` |
| `dow_cos` | `cos(2π·weekday/7)` |
| `hour_x_up_ratio` | `btc_up_ratio × (hour/24)` |
| `hour_x_tw_ur` | `btc_tw_up_ratio × (hour/24)` |

### Grupo F — Cross-domain Interactions (6 features)

| Feature | Fórmula |
|---|---|
| `x_imb_x_inslot` | `ob_imbalance × btc_inslot_ret` |
| `x_imb_end_x_ret` | `ob_imbalance_end × btc_inslot_ret` |
| `x_spread_x_vol` | `ob_spread × btc_vol_1h` |
| `x_depth_x_vol` | `ob_depth_ratio × btc_vol_1h` |
| `x_imb_x_ur` | `ob_imbalance × btc_up_ratio` |
| `x_depth_x_momentum` | `ob_depth_ratio × btc_momentum` |

---

## 4. Auditoria de lookahead

O bot decide em `t ∈ [170, 240]s`. Dado com `t > 60s` do slot atual = lookahead.

| Feature | Referência temporal | Lookahead? |
|---|---|---|
| `btc_pre_*_ret` | `obs_end = slot_ts + 60s` | ✅ NÃO — 60s < 170s |
| `btc_inslot_*` | `[slot_ts, slot_ts+60s]` | ✅ NÃO |
| `btc_vol_1h` | `[slot_ts-3600, slot_ts]` | ✅ NÃO |
| `btc_spot_vol_ratio` | `[slot_ts-3600, slot_ts]` | ✅ NÃO |
| `btc_dist_1k` | `px[slot_ts+60s]` | ✅ NÃO |
| `ob_*` (REST) | open ~t60s; close ~t170-240s | ✅ NÃO |
| `clob_*` (WS) | `[slot_ts+108, slot_ts+168s)` | ✅ NÃO — max 168s < 170s |
| ticks (C) | `t ∈ [0, 60s)` | ✅ NÃO — chegam via data-api em t~170s |
| `lag_*`, `prev_slot_*` | slots ANTERIORES | ✅ NÃO |
| `hour_*`, `dow_*` | `slot_ts` | ✅ NÃO |
| `x_*` cross | composição de A+B+C | ✅ NÃO |

**ZERO lookahead em todas as 73 features ✅**

---

## 5. Histórico de remoções

### Remoção 1 — 2026-06-10 (104 → 88): features inviáveis óbvias

| Feature | Motivo |
|---|---|
| `btc_up_w2/3/4/5` | DEAD — OBS=60s, janelas `[60..180s)` sempre vazias |
| `btc_vol_up`, `btc_vol_dn` | Volume absoluto não-normalizado |
| `btc_vwap_up`, `btc_vwap_dn` | VWAP extremamente ruidoso com n<5 ticks |
| `btc_vwap_spread` | Combinação linear de vwap_up − vwap_dn |
| `btc_signal_conviction` | Produto de dois features já no set |
| `btc_up_w5_zscore` | Zscore de constante 0.5 |
| `btc_dist_5k`, `btc_dist_10k` | Quase-constante para BTC ~$95k |
| `ob_imb_w0` | Janela [0,60s) — quase sempre 0.0 no live |
| `ob_pc_count` | Reset em reconexão = coverage menor |
| `ob_fill_imbalance` | Campo `size` ausente em muitos eventos WS |

### Remoção 2 — 2026-06-10 (88 → 73): auditoria rígida de fontes

| Feature | Categoria | Motivo |
|---|---|---|
| `ob_imb_w1`, `ob_imb_w2` | ❌ DIVERGÊNCIA | WS envia 1 real book snap; janelas [60,168s) sem dados reais |
| `clob_imb_mean`, `clob_imb_std`, `clob_imb_drift` | ❌ DIVERGÊNCIA | idem — janela [108,168s) quase sempre 0.0 |
| `clob_depth_trend` | ❌ DIVERGÊNCIA | idem |
| `ob_pc_up_ratio`, `ob_pc_volatility` | ⚠️ removidos | Reset WS em reconexão perde eventos do slot |
| `btc_pre_4h_ret`, `btc_vol_4h`, `btc_pre_1h_4h_ratio` | ⚠️ removidos | Warm-up 4h; gap se offline > 1min → 0.0 |
| `ob_mid_drift` | ⚠️ removido | Gap ~72s no timing close snapshot live vs treino |
| `clob_activity_rate` | ⚠️ removido | 1 real snap → contagem diferente do treino |
| `x_drift_x_ret5m`, `x_ob_drift_x_inslot` | cascata | Dependiam de ob_mid_drift |

---

## 6. Grupo A — Spot Binance

**Treino:** `scripts/train_v28_modal.py` ~l316  
**Live:** `deploy/live_trader.py` → `build_spot_features()`  
**Buffer:** `deque(maxlen=300)` = 5h candles. Seed: REST `limit=300` no boot.  
**Referência:** `obs_end = slot_ts + OBS_SECS(=60)`

```python
btc_pre_5m_ret  = (px[obs_end] / px[slot_ts - 300])  - 1   # ~7 candles
btc_pre_15m_ret = (px[obs_end] / px[slot_ts - 900])  - 1   # ~16 candles
btc_pre_30m_ret = (px[obs_end] / px[slot_ts - 1800]) - 1   # ~31 candles
btc_pre_1h_ret  = (px[obs_end] / px[slot_ts - 3600]) - 1   # ~61 candles

btc_inslot_ret   = (px[-1] / px[0]) - 1       # candles [slot_ts, obs_end]
btc_inslot_vol   = std(px) / mean(px)          # idem
btc_inslot_range = (max_hi - min_lo) / px_now  # k['h'], k['l']

btc_vol_1h = std(diff(px)/px[:-1])  # candles [slot_ts-3600, slot_ts]

btc_dist_1k = min(frac(px/1000), 1 - frac(px/1000))

btc_spot_vol_ratio = vol[slot_ts-300:slot_ts] / (vol[slot_ts-3600:slot_ts-300]/11)
# vol = k['v'] (volume acumulado da vela)
```

---

## 7. Grupo B — L2 Orderbook

**Fetch treino:** `scripts/fetch_ob_features_modal.py` (CUTOFF_SEC=168)  
**Live REST:** `deploy/live_trader.py` → `_build_ob_features()` — 2 chamadas REST por slot  
**Live WS:** apenas `clob_spread/mid/ask_pressure` via `ClobFeatureAccumulator.get_features()`

```
slot_ts       ~t60s REST (open)           ~t170-240s REST (close)
   |                  |                           |
   | ob_mid           |                           |
   | ob_spread        |                           |
   | ob_imbalance     |                           | ob_imbalance_end
   |                  |                           | ob_spread_end
   |                  |                           | ob_depth_change
   |                  |                           | ob_imb_momentum
   |    REST /book (full asks/bids) ─────────────>| ob_depth_ratio/bid/ask/total/weighted
   |
   |  CLOB WS price_change.best_bid/ask ─[108,168s)> clob_spread/mid/ask_pressure
```

### Fórmulas OB REST

```python
# Open snapshot
ob_mid        = (best_ask + best_bid) / 2
ob_spread     = best_ask - best_bid
ob_imbalance  = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-8)

# Close snapshot
ob_depth_ratio  = bid_depth_5c / (ask_depth_5c + 1e-8)
ob_bid_depth_5c = Σbid_sz(price ≥ mid-0.05) / Σbid_sz_total
ob_ask_depth_5c = Σask_sz(price ≤ mid+0.05) / Σask_sz_total
ob_total_depth  = Σbid_sz + Σask_sz
ob_weighted_imb = (Σbid_sz·e^(-10|bp-mid|) - Σask_sz·e^(-10|ap-mid|)) / Σwt

# Temporal (open → close)
ob_imbalance_end = imb_close
ob_spread_end    = spread_close
ob_depth_change  = total_depth_close - total_depth_open
ob_imb_momentum  = imb_close - imb_open
```

### Fórmulas CLOB WS (janela [slot_ts+108, slot_ts+168))

```python
# Synthetic BookSnapshots de best_bid/best_ask em cada price_change ✓
clob_spread_mean   = mean(spreads)            # all synthetic+real books
clob_spread_trend  = linslope(t, spreads)
clob_mid_velocity  = linslope(t, mids)
clob_mid_volatility = std(diff(mids))
clob_ask_pressure  = frac(SELL price moves < 0)   # de price_change.side+price
```

---

## 8. Grupo C — Tick-based Order Flow

**Janela:** `t ∈ [0, OBS_SECS=60s)`. Lag ~90-120s → disponível em t~170s.  
**Filtro crítico:** `slug = btc-updown-5m-{slot_ts}` — evita trades de outros mercados com mesmo token.

```python
vol_up = Σ(price × size) onde outcome="Up"
vol_dn = Σ(price × size) onde outcome="Down"
total  = vol_up + vol_dn + 1e-8

btc_up_ratio   = vol_up / total
btc_n_ticks    = len(ticks)
btc_buy_ratio  = Σsz_usdc(side="BUY") / total

btc_up_w0 = up_ratio em [0,  30s)
btc_up_w1 = up_ratio em [30, 60s)
btc_momentum           = btc_up_w1 - btc_up_w0
btc_up_ratio_stability = std(btc_up_w0, btc_up_w1)
btc_size_disparity     = avg_up_size - avg_dn_size

λ = 0.02
btc_tw_up_ratio = Σ(up_i·sz_i·e^(-λ·(60-t_i))) / Σ(sz_i·e^(-λ·(60-t_i)))
```

---

## 9. Grupo D — Lag History

**Ring buffer** `_slot_history`: max 20 entradas, campos `{slot_ts, up_ratio, target, n_ticks, vol_total}`.  
**Seed no boot:** `_seed_slot_history()` via data-api.  
**Staleness guard:** `time_gap > lag × 300 × 3` → fill neutro.

```python
for lag in [1..5]:
    lag_{lag}_outcome        = target[rank-lag]      (ou 0.5)
    prev_slot_up_ratio_{lag} = up_ratio[rank-lag]    (ou 0.5)
    prev_slot_n_ticks_{lag}  = n_ticks[rank-lag]     (ou 0.0)
    prev_slot_vol_{lag}      = vol_total[rank-lag]   (ou 0.0)

lag_streak = N consecutivos na mesma direção (max 5)

# Usa prev_slot_up_ratio_1 como "current"
lag_ur_zscore_20 = clip((prev_ur_1 - mean(hist_20)) / max(std(hist_20), 0.01), -5, 5)
lag_ur_zscore_5  = clip((prev_ur_1 - mean(hist_5))  / max(std(hist_5),  0.01), -5, 5)

# Usa btc_up_ratio atual como "current"
btc_up_ratio_zscore_20s = clip((cur_ur - mean(hist_20)) / max(std(hist_20), 0.01), -5, 5)
btc_up_ratio_zscore_5s  = clip((cur_ur - mean(hist_5))  / max(std(hist_5),  0.01), -5, 5)
```

---

## 10. Grupo E — Temporal

```python
dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
hour = dt.hour + dt.minute / 60.0
dow  = dt.weekday()  # 0=seg .. 6=dom

hour_sin = sin(2π·hour/24)  ;  hour_cos = cos(2π·hour/24)
dow_sin  = sin(2π·dow/7)    ;  dow_cos  = cos(2π·dow/7)
hour_x_up_ratio = btc_up_ratio    × (hour / 24.0)
hour_x_tw_ur    = btc_tw_up_ratio × (hour / 24.0)
```

---

## 11. Grupo F — Cross-domain Interactions

```python
# Computadas após A+B+C. Fallback gracioso via feat.get(k, default).
x_imb_x_inslot  = ob_imbalance     × btc_inslot_ret
x_imb_end_x_ret = ob_imbalance_end × btc_inslot_ret
x_spread_x_vol  = ob_spread        × btc_vol_1h       # default ob_spread=0.02
x_depth_x_vol   = ob_depth_ratio   × btc_vol_1h       # default 1.0
x_imb_x_ur      = ob_imbalance     × btc_up_ratio
x_depth_x_momentum = ob_depth_ratio × btc_momentum
```

---

## 12. Defaults quando dados ausentes

```python
# OB REST falha
ob_mid=0.5, ob_spread=0.02, ob_ask_depth_5c=0.5, ob_bid_depth_5c=0.5
ob_depth_ratio=1.0, ob_total_depth=1000.0
ob_imbalance=ob_imb_momentum=ob_depth_change=0.0

# clob_* (WS subscrito após t=108s)
clob_spread_mean=clob_spread_trend=clob_mid_velocity=clob_mid_volatility=clob_ask_pressure=0.0

# Ticks ausentes
btc_up_ratio=0.5, btc_n_ticks=0.0, btc_buy_ratio=0.5
btc_momentum=0.0, btc_size_disparity=0.0, btc_up_ratio_stability=0.0
btc_tw_up_ratio=0.5, btc_up_w0=0.5, btc_up_w1=0.5

# Lag history (vazio ou cold start)
lag_{1..5}_outcome=0.5, prev_slot_up_ratio_{1..5}=0.5
prev_slot_n_ticks_{1..5}=0.0, prev_slot_vol_{1..5}=0.0
lag_streak=0.0, lag_ur_zscore_*=0.0, btc_up_ratio_zscore_*=0.0
```

---

## 13. Seleção de features no treino

**Pool: 73 candidatas** — todas ✅ OK, sem divergências.

1. **Screening**: LightGBM `n_estimators=300, max_depth=4`, 5-fold `TimeSeriesSplit`
2. **Ranking**: acumula `feature_importances_` over folds ÷ N_SPLITS
3. **Teste N ∈ {40, 30, 25, 20, 15}**: avalia AUC com modelo completo
4. **Seleção**: N com melhor AUC médio

> O live computa todas as 73 mas usa apenas `features = model_data["features"]` para filtrar antes de `predict_proba`.

---

## 14. Checklist pré-treino

- [ ] `fetch_ob_features_modal.py` finalizou — todos os ~23.221 mercados
- [ ] `ob_features_full.parquet` uploaded para HF
- [ ] `all_markets.csv` + `ticks_btc_full_clean.parquet` atualizados no HF
- [ ] `binance_spot_full.parquet` cobre todo o período
- [ ] `OBS_SECS` treino == `OBSERVE_SECS` live (ambos = 60)
- [ ] `SLOT_DURATION` treino == live (ambos = 300)
- [ ] Auditoria: `python3 scripts/audit_features.py` → train=73==live=73, diff=0

---

## 15. Changelog de features por versão

| Versão | N | Mudanças |
|---|---|---|
| v21 | ~20 | Real-time only. OBS=180s causava lag. |
| v22-v23 | ~25 | `btc_up_ratio_stability`, `hour_sin/cos`. OBS→60s. |
| v24 | ~25 | Fix stability: 2 janelas reais. |
| v25 | ~35 | OB features via poly_l2. AUC 0.8575. |
| v26 | ~45 | CLOB WS. Cross-features. Fix btc_spot_vol_ratio. |
| v27 | ~50 | `ob_features_full.parquet` com clob_*. |
| v28 | 104 | Full train==live parity. Tick features no treino. |
| v28 (88f) | 88 | 1ª remoção: 16 features inviáveis óbvias. |
| **v28 (73f)** | **73** | **2ª remoção: 15 features com divergência/fallback. Auditoria rígida de fontes. 73/73 ✅ OK, zero divergências.** |

---

*Atualizar sempre que modificar features. Auditoria: `train=73==live=73 diff=0 ✅` (2026-06-10)*
