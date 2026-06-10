# BTC 5-min Model — Feature Brain
**Versão de referência:** v28 (88f) | **OBS_SECS:** 60 | **SLOT_DURATION:** 300s  
**Última atualização:** 2026-06-10 | **Commit:** 617fb86  
**Auditoria:** `train: 88 == live: 88 | diff: 0 ✅`

> Este documento é a fonte de verdade de todas as features do modelo.  
> Qualquer alteração em `train_v28_modal.py` ou `deploy/live_trader.py` deve ser refletida aqui.  
> Antes de treinar ou deployar, confirmar que a tabela de paridade está ✅ em todas as linhas.

---

## Índice
1. [Timeline de um slot](#1-timeline-de-um-slot)
2. [Fontes de dados](#2-fontes-de-dados)
3. [Tabela canônica — 88 features (train == live)](#3-tabela-canônica--88-features-train--live)
4. [Auditoria de lookahead](#4-auditoria-de-lookahead)
5. [Análise de viabilidade live](#5-análise-de-viabilidade-live)
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
slot_ts          +30s     +60s (OBS_SECS)    +108s    +168s   +170s      +240s   +300s
   |               |            |               |        |       |           |       |
   |<-- ticks W0 ->|<-- W1 ---->|               |        |       |           |       |
   |<-- spot inslot ----------->|               |        |       |           |       |
   |<-- CLOB WS acumulando ------------------------------>|       |           |       |
   |                            |<-- lag ~90s -->|        |       |           |       |
   |                            |           clob_* janela |       |           |       |
   |                            |                ob_pc_*  |       |           |       |
   |                                                      | entry window      |       |
   |                                                      | OB REST /book     |       |
   |                                                      | predict + order   |       |
                                                                              fim slot
```

- **t=0**: slot começa (`slot_ts` = múltiplo de 300s)
- **t=0–30s**: `btc_up_w0` — primeiro sub-janela de ticks
- **t=30–60s**: `btc_up_w1` — segundo sub-janela de ticks
- **t=60s** (`obs_end_ts = slot_ts + 60`): ponto de referência de spot e ticks
- **t=108–168s**: janela dos `clob_*` (60s ending at CUTOFF_SEC)
- **t=0–168s**: janela dos `ob_pc_*`; `ob_imb_w1/w2` de real WS books
- **t=170s** (`ENTER_WINDOW[0]`): ticks t<60s disponíveis via data-api; OB REST chamado
- **t=240s** (`ENTER_WINDOW[1]`): fim da janela de decisão
- **t=300s**: slot fecha, resultado resolvido, ring buffer atualizado

---

## 2. Fontes de dados

| Fonte | Lag live | Cobertura treino | Arquivo |
|---|---|---|---|
| Binance spot WebSocket | <2s | `binance_spot_full.parquet` + `binance_spot_local.parquet` | 1-min OHLCV |
| Polymarket data-api trades | ~90-120s | `ticks_btc_full_clean.parquet` + `new_ticks_pmdata.parquet` | t_sec=0-60 |
| Polymarket CLOB REST `/book` | <5s | `ob_features_full.parquet` (via `fetch_ob_features_modal.py`) | open+close snap |
| Polymarket CLOB WS book/price_change | 0s | `ob_features_full.parquet` (colunas `ob_imb_w*`, `ob_pc_*`, `clob_*`) | slot-anchored |
| Ring buffer de slots anteriores | 0s | `all_markets.csv` targets + tick aggregates | últimos 20 slots |

---

## 3. Tabela canônica — 88 features (train == live)

**Auditoria (2026-06-10): train=88, live=88, diff=0 ✅**  
Legenda: ✅ idêntico | ⚠️ divergência estrutural aceita

### Grupo A — Spot Binance (13 features)

| Feature | Fórmula | Paridade | Viabilidade |
|---|---|---|---|
| `btc_pre_5m_ret` | `(px[obs_end] / px[slot_ts-300]) - 1` | ✅ | ✅ OK |
| `btc_pre_15m_ret` | `(px[obs_end] / px[slot_ts-900]) - 1` | ✅ | ✅ OK |
| `btc_pre_30m_ret` | `(px[obs_end] / px[slot_ts-1800]) - 1` | ✅ | ✅ OK |
| `btc_pre_1h_ret` | `(px[obs_end] / px[slot_ts-3600]) - 1` | ✅ | ✅ OK |
| `btc_pre_4h_ret` | `(px[obs_end] / px[slot_ts-14400]) - 1` | ✅ | ⚠️ warm-up 4h; seed via REST no boot |
| `btc_inslot_ret` | `(px[-1]/px[0])-1` em `[slot_ts, obs_end]` | ✅ | ✅ OK |
| `btc_inslot_vol` | `std(px)/mean(px)` em `[slot_ts, obs_end]` | ✅ | ✅ OK |
| `btc_inslot_range` | `(max_hi - min_lo) / px_now` | ✅ | ✅ OK |
| `btc_vol_1h` | `std(1-min returns)` em `[slot_ts-3600, slot_ts]` | ✅ | ✅ OK |
| `btc_vol_4h` | `std(1-min returns)` em `[slot_ts-14400, slot_ts]` | ✅ | ⚠️ warm-up 4h |
| `btc_pre_1h_4h_ratio` | `(px_now - px_1h) / (px_now - px_4h + 1e-9)` | ✅ | ⚠️ warm-up 4h |
| `btc_dist_1k` | `min(frac(px/1k), 1-frac(px/1k))` | ✅ | ✅ OK |
| `btc_spot_vol_ratio` | `vol_5m / (vol_55m/11)` | ✅ | ✅ OK |

### Grupo B — L2 Orderbook REST (17 features)

| Feature | Descrição | Snap | Paridade | Viabilidade |
|---|---|---|---|---|
| `ob_mid` | `(best_ask+best_bid)/2` | Open ~t60s | ✅ | ✅ OK |
| `ob_spread` | `best_ask - best_bid` | Open | ✅ | ✅ OK |
| `ob_imbalance` | `(bid_sz-ask_sz)/(bid_sz+ask_sz)` best-level | Open | ✅ | ✅ OK |
| `ob_depth_ratio` | `bid_depth_5c / ask_depth_5c` | Close | ✅ | ✅ OK |
| `ob_bid_depth_5c` | `Σbid_sz(p≥mid-0.05) / Σbid_sz` | Close | ✅ | ✅ OK |
| `ob_ask_depth_5c` | `Σask_sz(p≤mid+0.05) / Σask_sz` | Close | ✅ | ✅ OK |
| `ob_total_depth` | `Σbid_sz + Σask_sz` | Close | ✅ | ✅ OK |
| `ob_weighted_imb` | `Σ(bid_sz·e^(-10Δp) - ask_sz·e^(-10Δp)) / Σwt` | Close | ✅ | ✅ OK |
| `ob_mid_drift` | `mid_close - mid_open` | Temporal | ⚠️ close t~168s treino / t~240s live | ✅ sinal válido |
| `ob_imbalance_end` | `imbalance` do close snapshot | Temporal | ✅ | ✅ OK |
| `ob_spread_end` | `spread` do close snapshot | Temporal | ✅ | ✅ OK |
| `ob_depth_change` | `depth_close - depth_open` | Temporal | ✅ | ✅ OK |
| `ob_imb_momentum` | `imb_close - imb_open` | Temporal | ⚠️ mesmo timing de ob_mid_drift | ✅ sinal válido |
| `ob_imb_w1` | `mean(imb)` em `[60,120s)` via WS real books | WS | ✅ | ✅ OK |
| `ob_imb_w2` | `mean(imb)` em `[120,168s)` via WS real books | WS | ✅ | ✅ OK |
| `ob_pc_up_ratio` | `n_up_moves / n_total` em `[0,168s)` | WS pc | ✅ | ✅ OK (fallback 0.5) |
| `ob_pc_volatility` | `std(price_diffs)` em `[0,168s)` | WS pc | ✅ | ✅ OK (fallback 0.0) |

### Grupo B — CLOB WebSocket (10 features)

| Feature | Fórmula | Janela | Paridade | Viabilidade |
|---|---|---|---|---|
| `clob_imb_mean` | `mean(real_imb)` | `[108,168s)` | ✅ | ⚠️ zeros se WS subscrito após t=108s |
| `clob_imb_std` | `std(real_imb)` | idem | ✅ | ⚠️ idem |
| `clob_imb_drift` | `imb[-1] - imb[0]` | idem | ✅ | ⚠️ idem |
| `clob_spread_mean` | `mean(spread)` | idem | ✅ | ⚠️ idem |
| `clob_spread_trend` | slope linear spread | idem | ✅ | ⚠️ idem |
| `clob_mid_velocity` | slope linear mid | idem | ✅ | ⚠️ idem |
| `clob_mid_volatility` | `std(diff(mid))` | idem | ✅ | ⚠️ idem |
| `clob_activity_rate` | `(n_real_books + n_pcs) / Δt` | idem | ✅ | ⚠️ idem |
| `clob_depth_trend` | slope linear depth | idem | ✅ | ⚠️ idem |
| `clob_ask_pressure` | `frac(ASK moves < 0)` | idem | ✅ | ⚠️ idem |

> **clob_* cobertura**: subscrição WS proativa em `fetch_market()`. Slots conhecidos antes de t=108s = janela cheia. Subscrição tardia = zeros (defaults neutros). Cobertura estimada ~80%.

### Grupo C — Tick-based Order Flow (9 features)

| Feature | Fórmula | Paridade | Viabilidade |
|---|---|---|---|
| `btc_up_ratio` | `vol_up / (vol_up + vol_dn)` | ✅ | ✅ OK |
| `btc_n_ticks` | `len(ticks)` | ✅ | ✅ OK |
| `btc_buy_ratio` | `Σsz(side=BUY) / total` | ✅ | ✅ OK |
| `btc_momentum` | `btc_up_w1 - btc_up_w0` | ✅ | ✅ OK |
| `btc_size_disparity` | `avg_up_size - avg_dn_size` | ✅ | ✅ OK |
| `btc_up_ratio_stability` | `std(btc_up_w0, btc_up_w1)` | ✅ | ✅ OK |
| `btc_tw_up_ratio` | `Σ(up·sz·e^(-0.02·(60-t))) / Σ(sz·e^(-0.02·(60-t)))` | ✅ | ✅ OK |
| `btc_up_w0` | `up_ratio` em `[0, 30s)` | ✅ | ✅ OK |
| `btc_up_w1` | `up_ratio` em `[30, 60s)` | ✅ | ✅ OK |

### Grupo D — Lag History (22 features)

| Feature | Paridade | Viabilidade |
|---|---|---|
| `lag_{1..5}_outcome` | ✅ | ✅ OK |
| `prev_slot_up_ratio_{1..5}` | ✅ | ✅ OK |
| `prev_slot_n_ticks_{1..5}` | ✅ | ✅ OK |
| `prev_slot_vol_{1..5}` | ✅ | ✅ OK |
| `lag_streak` | ✅ | ✅ OK |
| `lag_ur_zscore_20` | ✅ | ✅ OK |
| `lag_ur_zscore_5` | ✅ | ✅ OK |

### Grupo E — Temporal (6 features)

| Feature | Fórmula | Paridade | Viabilidade |
|---|---|---|---|
| `hour_sin` | `sin(2π·hour/24)` | ✅ | ✅ OK |
| `hour_cos` | `cos(2π·hour/24)` | ✅ | ✅ OK |
| `dow_sin` | `sin(2π·weekday/7)` | ✅ | ✅ OK |
| `dow_cos` | `cos(2π·weekday/7)` | ✅ | ✅ OK |
| `hour_x_up_ratio` | `btc_up_ratio × (hour/24)` | ✅ | ✅ OK |
| `hour_x_tw_ur` | `btc_tw_up_ratio × (hour/24)` | ✅ | ✅ OK |

### Grupo F — Cross-domain Interactions (8 features)

| Feature | Fórmula | Paridade | Viabilidade |
|---|---|---|---|
| `x_imb_x_inslot` | `ob_imbalance × btc_inslot_ret` | ✅ | ✅ OK |
| `x_imb_end_x_ret` | `ob_imbalance_end × btc_inslot_ret` | ✅ | ✅ OK |
| `x_drift_x_ret5m` | `ob_mid_drift × btc_pre_5m_ret` | ✅ | ✅ OK |
| `x_spread_x_vol` | `ob_spread × btc_vol_1h` (default 0.02) | ✅ | ✅ OK |
| `x_depth_x_vol` | `ob_depth_ratio × btc_vol_1h` (default 1.0) | ✅ | ✅ OK |
| `x_imb_x_ur` | `ob_imbalance × btc_up_ratio` | ✅ | ✅ OK |
| `x_depth_x_momentum` | `ob_depth_ratio × btc_momentum` | ✅ | ✅ OK |
| `x_ob_drift_x_inslot` | `ob_mid_drift × btc_inslot_ret` | ✅ | ✅ OK |

---

## 4. Auditoria de lookahead

O bot decide em `t ∈ [170, 240]s`. Qualquer feature com dados de `t > 60s` do slot atual é lookahead.

| Feature | Referência temporal | Lookahead? | Justificativa |
|---|---|---|---|
| `btc_pre_*_ret` | `obs_end = slot_ts + 60s` | ✅ NÃO | 60s < 170s |
| `btc_inslot_*` | `[slot_ts, slot_ts+60s]` | ✅ NÃO | fecha em t=60s |
| `btc_vol_1h/4h` | `[slot_ts-3600/14400, slot_ts]` | ✅ NÃO | pré-slot |
| `btc_spot_vol_ratio` | `[slot_ts-3600, slot_ts]` | ✅ NÃO | pré-slot |
| `btc_dist_1k` | `px[slot_ts+60s]` | ✅ NÃO | t=60s < 170s |
| `ob_mid/spread/imbalance` | open snapshot ~t60s | ✅ NÃO | pré-entry |
| `ob_mid_drift` | `mid[~168s] - mid[~60s]` | ✅ NÃO | 168s < 170s |
| `ob_imb_momentum` | idem | ✅ NÃO | idem |
| `ob_imb_w1/w2` | `[60,120s)` e `[120,168s)` | ✅ NÃO | max 168s < 170s |
| `ob_pc_*` | `[0, 168s)` | ✅ NÃO | 168s < 170s |
| `clob_*` | `[108, 168s)` | ✅ NÃO | max 168s < 170s |
| ticks Grupo C | `t ∈ [0, 60s)` | ✅ NÃO | chegam via data-api em t~170s |
| `lag_*`, `prev_slot_*` | slots ANTERIORES | ✅ NÃO | sempre passado |
| `hour_*`, `dow_*` | `slot_ts` | ✅ NÃO | timestamp público |
| `x_*` cross features | composição de A+B+C | ✅ NÃO | todos ≤168s |

**Resultado: ZERO lookahead em todas as 88 features ✅**

---

## 5. Análise de viabilidade live

### 5.1 Features removidas (histórico)

16 features removidas em 2026-06-10 por inviabilidade:

| Feature | Motivo |
|---|---|
| `btc_up_w2/3/4/5` | DEAD — OBS=60s, janelas `[60..180s)` sempre fora da observação, 100% constante 0.5 |
| `btc_vol_up`, `btc_vol_dn` | Volume absoluto USDC não-normalizado, colinear com `btc_n_ticks` |
| `btc_vwap_up`, `btc_vwap_dn` | VWAP extremamente ruidoso com n<5 ticks |
| `btc_vwap_spread` | Combinação linear de `btc_vwap_up - btc_vwap_dn` |
| `btc_signal_conviction` | Produto de `btc_up_ratio × (1 - stability)` — redundante |
| `btc_up_w5_zscore` | Zscore de constante 0.5 — sempre constante |
| `btc_dist_5k`, `btc_dist_10k` | Quase-constante para BTC ~$95k |
| `ob_imb_w0` | Live: Polymarket WS envia 1 real book snap; w0=[0,60s) quase sempre 0.0 |
| `ob_pc_count` | Coverage live sistematicamente menor (reset em reconexão WS) |
| `ob_fill_imbalance` | Campo `size` ausente em muitos eventos WS `price_change` |

### 5.2 Features com ressalvas operacionais (manter, monitorar)

| Feature(s) | Ressalva | Mitigação |
|---|---|---|
| `btc_pre_4h_ret`, `btc_vol_4h`, `btc_pre_1h_4h_ratio` | Warm-up 4h — 0.0 se bot reiniciar | `_seed_spot_buffers()` via REST no boot |
| `clob_*` (10) | Zeros se WS subscrito após t=108s | `fetch_market()` faz subscrição proativa |
| `ob_imb_w1/w2` | Depende de real book snaps chegando pelo WS | Fallback interpolado se sem dados |
| `ob_pc_up_ratio`, `ob_pc_volatility` | Perda parcial em reconexão WS | Fallback neutro (0.5 / 0.0) |
| `ob_mid_drift`, `ob_imb_momentum` | Close ~t240s live vs ~t168s treino (~72s gap) | Sinal ainda informativo; aceito |

### 5.3 Cobertura live esperada por grupo

| Grupo | Cobertura | Observação |
|---|---|---|
| A — Spot (13) | ~100% | WS + REST fallback |
| B — OB REST (17) | ~95% | Falha em books vazios/one-sided |
| B — CLOB WS (10) | ~80% | Zeros para subscrição tardia |
| C — Ticks (9) | ~90% | data-api lag; fallback neutro |
| D — Lag (22) | ~95% | `_seed_slot_history()` no boot |
| E — Temporal (6) | ~100% | Só `slot_ts` |
| F — Cross (8) | ~95% | Fallback gracioso via `.get(k, default)` |

---

## 6. Grupo A — Spot Binance

**Treino:** `scripts/train_v28_modal.py` ~l316  
**Live:** `deploy/live_trader.py` → `build_spot_features()`  
**Referência:** `obs_end_ts = slot_ts + OBS_SECS (=60)`; `px_now = px[obs_end_ts]`

```python
btc_pre_5m_ret  = (px[obs_end] / px[slot_ts - 300])   - 1
btc_pre_15m_ret = (px[obs_end] / px[slot_ts - 900])   - 1
btc_pre_30m_ret = (px[obs_end] / px[slot_ts - 1800])  - 1
btc_pre_1h_ret  = (px[obs_end] / px[slot_ts - 3600])  - 1
btc_pre_4h_ret  = (px[obs_end] / px[slot_ts - 14400]) - 1

btc_inslot_ret   = (px[-1] / px[0]) - 1          # candles [slot_ts, obs_end]
btc_inslot_vol   = std(px) / mean(px)             # idem
btc_inslot_range = (max_hi - min_lo) / px_now     # usa high/low das velas

btc_vol_1h = std(diff(px) / px[:-1])   # candles [slot_ts-3600, slot_ts]
btc_vol_4h = std(diff(px) / px[:-1])   # candles [slot_ts-14400, slot_ts]

btc_pre_1h_4h_ratio = (px_now - px_1h) / (px_now - px_4h + 1e-9)

btc_dist_1k = min(frac(px/1000), 1 - frac(px/1000))

btc_spot_vol_ratio = vol[slot_ts-300:slot_ts] / (vol[slot_ts-3600:slot_ts-300] / 11)
```

---

## 7. Grupo B — L2 Orderbook

**Fetch treino:** `scripts/fetch_ob_features_modal.py` (CUTOFF_SEC=168)  
**Live:** `deploy/live_trader.py` → `_build_ob_features()` + `ClobFeatureAccumulator`

### Timeline de snapshots

```
slot_ts     ~t60s (open REST)           t=168s (CUTOFF_SEC)    ~t240s (close REST live)
   |                |                         |                        |
   | ob_mid         |                         | ob_mid_drift           |
   | ob_spread      |                         | ob_imbalance_end       |
   | ob_imbalance   |                         | ob_spread_end          |
   |                |                         | ob_depth_change        |
   |                |                         | ob_imb_momentum        |
   |                |<-- WS real books ------->|  ob_imb_w1 [60,120s)  |
   |                |                         |  ob_imb_w2 [120,168s) |
   |<--- WS price_change events ------------->|  ob_pc_up_ratio        |
   |                                          |  ob_pc_volatility      |
   |                    [108,168s) clob_* ----+----------------------> |
```

### Fórmulas

```python
# Open snapshot (REST ~t60s)
ob_mid        = (best_ask + best_bid) / 2
ob_spread     = best_ask - best_bid
ob_imbalance  = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-8)   # best level

# Close snapshot (REST ~t168s treino / ~t240s live)
ob_depth_ratio  = bid_depth_5c / (ask_depth_5c + 1e-8)
ob_bid_depth_5c = Σbid_sz(price ≥ mid-0.05) / Σbid_sz_total
ob_ask_depth_5c = Σask_sz(price ≤ mid+0.05) / Σask_sz_total
ob_total_depth  = Σbid_sz + Σask_sz
ob_weighted_imb = (Σbid_sz·exp(-10|bp-mid|) - Σask_sz·exp(-10|ap-mid|)) / Σwt

# Temporal (open → close)
ob_mid_drift     = mid_close  - mid_open
ob_imbalance_end = imb_close
ob_spread_end    = spread_close
ob_depth_change  = total_depth_close - total_depth_open
ob_imb_momentum  = imb_close - imb_open

# Windowed imbalance (WS real book snaps, via get_windowed_imbalance)
ob_imb_w1 = mean(imb, t ∈ [60,  120s))
ob_imb_w2 = mean(imb, t ∈ [120, 168s))

# Price-change aggregates (WS events, via get_ob_pc_features, cutoff=168s)
ob_pc_up_ratio  = n_up_moves / n_total_moves
ob_pc_volatility = std(price_diffs)

# CLOB WS microstructure (via get_features, obs_secs=168, window=60 → [108,168s))
clob_imb_mean      = mean(real_book_imbalances)
clob_imb_std       = std(real_book_imbalances)
clob_imb_drift     = imb[-1] - imb[0]
clob_spread_mean   = mean(all_spreads)
clob_spread_trend  = linslope(t, spreads)
clob_mid_velocity  = linslope(t, mids)
clob_mid_volatility = std(diff(mids))
clob_activity_rate = (n_real_books + n_pc_events) / time_span
clob_depth_trend   = linslope(t, real_depths)
clob_ask_pressure  = frac(ASK price moves that went DOWN)
```

---

## 8. Grupo C — Tick-based Order Flow

**Janela:** `t ∈ [0, OBS_SECS=60s)` — ticks da data-api, lag ~90-120s, disponíveis em t~170s.

```python
vol_up = Σsize_usdc  onde outcome="Up"
vol_dn = Σsize_usdc  onde outcome="Down"
total  = vol_up + vol_dn + 1e-8

btc_up_ratio  = vol_up / total
btc_n_ticks   = len(ticks)
btc_buy_ratio = Σsz(side="BUY") / total

# Sub-janelas (apenas 2 com OBS=60s)
btc_up_w0 = up_ratio em [0,  30s)
btc_up_w1 = up_ratio em [30, 60s)

btc_momentum         = btc_up_w1 - btc_up_w0
btc_up_ratio_stability = std(btc_up_w0, btc_up_w1)
btc_size_disparity   = avg_up_size - avg_dn_size

# Time-weighted up ratio (exponential recency decay)
λ = 0.02
btc_tw_up_ratio = Σ(up_i · size_i · exp(-λ·(60-t_i))) / Σ(size_i · exp(-λ·(60-t_i)))
```

---

## 9. Grupo D — Lag History

**Ring buffer** (`_slot_history`): max 20 entradas, campos `{slot_ts, up_ratio, target, n_ticks, vol_total}`.  
Staleness guard: `time_gap > lag × 300 × 3` → fill neutro.

```python
for lag in [1..5]:
    lag_{lag}_outcome        = target[rank-lag]      (ou 0.5 se stale)
    prev_slot_up_ratio_{lag} = up_ratio[rank-lag]    (ou 0.5)
    prev_slot_n_ticks_{lag}  = n_ticks[rank-lag]     (ou 0.0)
    prev_slot_vol_{lag}      = vol_total[rank-lag]   (ou 0.0)

lag_streak = N outcomes consecutivos na mesma direção (max 5, para no primeiro divergente)

# Z-scores de up_ratio histórico (usa prev_slot_up_ratio_1 como "current")
lag_ur_zscore_20 = clip((prev_ur_1 - mean(hist_20)) / max(std(hist_20), 0.01), -5, 5)
lag_ur_zscore_5  = clip((prev_ur_1 - mean(hist_5))  / max(std(hist_5),  0.01), -5, 5)
```

> `btc_up_ratio_zscore_5s/20s` (também no set) usam `btc_up_ratio` do slot atual como "current", enquanto `lag_ur_zscore_*` usam `prev_slot_up_ratio_1`. São features distintas e não redundantes.

---

## 10. Grupo E — Temporal

```python
dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
hour = dt.hour + dt.minute / 60.0    # [0, 24)
dow  = dt.weekday()                   # 0=segunda .. 6=domingo

hour_sin = sin(2π · hour / 24)
hour_cos = cos(2π · hour / 24)
dow_sin  = sin(2π · dow / 7)
dow_cos  = cos(2π · dow / 7)

hour_x_up_ratio = btc_up_ratio    × (hour / 24.0)
hour_x_tw_ur    = btc_tw_up_ratio × (hour / 24.0)
```

---

## 11. Grupo F — Cross-domain Interactions

Computadas após grupos A+B+C. Falham graciosamente via `feat.get(k, default)` se OB ausente.

```python
x_imb_x_inslot      = ob_imbalance     × btc_inslot_ret
x_imb_end_x_ret     = ob_imbalance_end × btc_inslot_ret
x_drift_x_ret5m     = ob_mid_drift     × btc_pre_5m_ret
x_spread_x_vol      = ob_spread        × btc_vol_1h       # default ob_spread=0.02
x_depth_x_vol       = ob_depth_ratio   × btc_vol_1h       # default ob_depth_ratio=1.0
x_imb_x_ur          = ob_imbalance     × btc_up_ratio
x_depth_x_momentum  = ob_depth_ratio   × btc_momentum
x_ob_drift_x_inslot = ob_mid_drift     × btc_inslot_ret
```

---

## 12. Defaults quando dados ausentes

```python
# OB REST falha
ob_mid=0.5, ob_spread=0.02, ob_ask_depth_5c=0.5, ob_bid_depth_5c=0.5
ob_depth_ratio=1.0, ob_total_depth=1000.0
ob_imbalance=0.0, ob_mid_drift=0.0, ob_imb_momentum=0.0, ob_depth_change=0.0

# ob_pc_* (WS sem eventos ou reconexão)
ob_pc_up_ratio=0.5, ob_pc_volatility=0.0

# ob_imb_w1/w2 (sem real book snaps no WS)
ob_imb_w1=interpolado(open_snap, close_snap), ob_imb_w2=close_snap.imb

# clob_* (WS subscrito após t=108s)
todos clob_*=0.0

# Ticks (data-api falha ou slot sem ticks)
btc_up_ratio=0.5, btc_n_ticks=0.0, btc_buy_ratio=0.5
btc_momentum=0.0, btc_size_disparity=0.0, btc_up_ratio_stability=0.0
btc_tw_up_ratio=0.5, btc_up_w0=0.5, btc_up_w1=0.5

# Lag history (vazio ou stale)
lag_{1..5}_outcome=0.5, prev_slot_up_ratio_{1..5}=0.5
prev_slot_n_ticks_{1..5}=0.0, prev_slot_vol_{1..5}=0.0
lag_streak=0.0, lag_ur_zscore_*=0.0, btc_up_ratio_zscore_*=0.0
```

---

## 13. Seleção de features no treino

**Pool atual: 88 candidatas limpas** (após remoção das 16 inviáveis).

1. **Screening**: LightGBM `n_estimators=300, max_depth=4`, 5-fold `TimeSeriesSplit`
2. **Ranking**: acumula `feature_importances_` over folds, divide por N_SPLITS
3. **Teste de N**: avalia AUC para N ∈ {40, 30, 25, 20, 15} com modelo completo (400 estimators)
4. **Seleção**: N com melhor AUC médio nos folds

> O modelo só aprende o `top_features` list. O live computa todas as 88 mas usa apenas `features = model_data["features"]` para filtrar antes de `predict_proba`.

---

## 14. Checklist pré-treino

- [ ] `fetch_ob_features_modal.py` finalizou — todos os ~23.221 mercados no `ob_features_full.parquet`
- [ ] `ob_features_full.parquet` uploaded para HF em `data/ob_features_full.parquet`
- [ ] `all_markets.csv` e `ticks_btc_full_clean.parquet` atualizados no HF
- [ ] `binance_spot_full.parquet` cobre todo o período dos markets
- [ ] `OBS_SECS` no treino == `OBSERVE_SECS` no live (ambos = 60)
- [ ] `SLOT_DURATION` no treino == `SLOT_DURATION` no live (ambos = 300)
- [ ] Auditoria: `train == live, diff = 0` (rodar script abaixo)

```bash
python3 - <<'EOF'
import re, pandas as pd
from collections import defaultdict
df_ob = pd.read_parquet("data/ob_features_full.parquet")  # local copy
OB_EXCLUDED = {"ob_imb_w0","ob_pc_count","ob_fill_imbalance"}
ob_cols = set(c for c in df_ob.columns if c != "market_id" and c not in OB_EXCLUDED)
# ... (ver scripts/audit_features.py)
EOF
```

---

## 15. Changelog de features por versão

| Versão | Features | Mudanças |
|---|---|---|
| v21 | ~20 | Primeiro "real-time only". OBS=180s criava lag nos ticks. |
| v22-v23 | ~25 | `btc_up_ratio_stability`, `hour_sin/cos`, `btc_up_w5_zscore`. OBS→60s. |
| v24 | ~25 | Fix: stability usa só 2 janelas reais (n=OBS//30). |
| v25 | ~35 | OB features (ob_*) via pmdata poly_l2. AUC baseline 0.8575. |
| v26 | ~45 | CLOB WS features (clob_*). Cross-features x_imb_end_x_ret etc. |
| v27 | ~50 | `ob_features_full.parquet` com clob_*. Fix prefixo. |
| v28 | 104 | Full train==live parity: tick features no treino, dow_sin/cos, hour_x_*. lag_outcome+streak no live. Todas as divergências eliminadas. |
| **v28 (88f)** | **88** | **Remoção de 16 features inviáveis.** DEAD: btc_up_w2-5. Redundante: btc_vol_up/dn, btc_vwap_up/dn/spread, btc_signal_conviction, btc_up_w5_zscore. Quase-constante: btc_dist_5k/10k. Esparso live: ob_imb_w0, ob_pc_count, ob_fill_imbalance. btc_momentum simplificado: w1-w0. Paridade 88==88 ✅. |
| v29 (plan) | ~88 | Auto feature selection sobre 88 candidatas limpas. Purged CV. |

---

*Atualizar sempre que adicionar, remover ou modificar qualquer feature. Rodar auditoria antes de todo treino.*
