# BTC 5-min Model — Feature Brain
**Versão de referência:** v28 | **OBS_SECS:** 60 | **SLOT_DURATION:** 300s  
**Última atualização:** 2026-06-10 | **Commit:** 4880905

> Este documento é a fonte de verdade de todas as features do modelo.  
> Qualquer alteração em `train_v28_modal.py` ou `deploy/live_trader.py` deve ser refletida aqui.  
> Antes de treinar ou deployar, confirmar que a tabela de paridade está ✅ em todas as linhas.

---

## Índice
1. [Timeline de um slot](#1-timeline-de-um-slot)
2. [Fontes de dados](#2-fontes-de-dados)
3. [Tabela canônica de features (train == live)](#3-tabela-canônica-de-features)
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
slot_ts                     +60s (OBS_SECS)       +170s          +240s      +300s
   |                              |                   |              |          |
   |<--- ticks coletados -------->|                   |              |          |
   |<--- spot inslot ------------>|                   |              |          |
   |<--- CLOB WS acumulando ----->|<--continua------->|              |          |
                                  |<--- lag ~90-120s->|              |          |
                                  |                   |<--entrada -->|          |
                                  |                   |  OB REST /book          |
                                  |                   |  predict + order        |
                                                                               fim
```

- **t=0**: slot começa (slot_ts = timestamp UTC arredondado para múltiplo de 300s)
- **t=0 a 60s**: janela de observação — ticks coletados, spot inslot, CLOB WS acumulando
- **t=60s** (`obs_end_ts = slot_ts + OBS_SECS`): ponto de referência de todos os features de spot e tick
- **t=108-168s**: janela dos clob_* features no treino (e no live após fix v28)
- **t=170s**: entrada da janela de decisão (`ENTER_WINDOW[0]`)
  - data-api lag ~90-120s → ticks de t=0-60s chegam aqui ✓
  - OB REST /book chamado aqui → ob_* features ao vivo
- **t=240s**: fim da janela de decisão (`ENTER_WINDOW[1]`)
- **t=300s**: slot fecha, resultado resolvido, ring buffer atualizado

---

## 2. Fontes de dados

| Fonte | Lag live | Cobertura treino | Arquivos |
|---|---|---|---|
| Binance spot WebSocket | <2s | `binance_spot_full.parquet` + `binance_spot_local.parquet` | 1-min OHLCV candles |
| Polymarket data-api trades | ~90-120s | `ticks_btc_full_clean.parquet` + `new_ticks_pmdata.parquet` | t_sec=0-60 por mercado |
| Polymarket CLOB REST `/book` | <5s | `ob_features_full.parquet` (via `fetch_ob_features_modal.py`) | snapshot open+close por mercado |
| Polymarket CLOB WS book/price_change | 0s (stream) | `ob_features_full.parquet` (colunas `ob_imb_w*`, `ob_pc_*`, `clob_*`) | 60s window por mercado |
| Ring buffer de slots anteriores | 0s | `all_markets.csv` (targets resolvidos) + tick aggregates | últimos 20 slots |

---

## 3. Tabela canônica de features

Total: **104 features** (antes da seleção de importância).  
Legenda: ✅ idêntico | ⚠️ divergência aceita | 🚫 candidata a remoção

### Grupo A — Spot Binance (15 features)

| Feature | Train v28 | Live | Paridade | Viabilidade |
|---|---|---|---|---|
| `btc_pre_5m_ret` | `(px[obs_end] / px[slot_ts-300]) - 1` | `build_spot_features` → idem | ✅ | ✅ OK |
| `btc_pre_15m_ret` | `(px[obs_end] / px[slot_ts-900]) - 1` | idem | ✅ | ✅ OK |
| `btc_pre_30m_ret` | `(px[obs_end] / px[slot_ts-1800]) - 1` | idem | ✅ | ✅ OK |
| `btc_pre_1h_ret` | `(px[obs_end] / px[slot_ts-3600]) - 1` | idem | ✅ | ✅ OK |
| `btc_pre_4h_ret` | `(px[obs_end] / px[slot_ts-14400]) - 1` | idem | ✅ | ⚠️ warm-up 4h |
| `btc_inslot_ret` | `(px[-1]/px[0])-1` em `[slot_ts, obs_end]` | idem | ✅ | ✅ OK |
| `btc_inslot_vol` | `std(px)/mean(px)` em `[slot_ts, obs_end]` | idem | ✅ | ✅ OK |
| `btc_inslot_range` | `(max_hi - min_lo) / px_now` | idem | ✅ | ✅ OK |
| `btc_vol_1h` | `std(1-min returns)` em `[slot_ts-3600, slot_ts]` | idem | ✅ | ✅ OK |
| `btc_vol_4h` | `std(1-min returns)` em `[slot_ts-14400, slot_ts]` | idem | ✅ | ⚠️ warm-up 4h |
| `btc_pre_1h_4h_ratio` | `(px_now - px_1h) / (px_now - px_4h + 1e-9)` | idem | ✅ | ⚠️ warm-up 4h |
| `btc_dist_1k` | `min(frac(px/1k), 1-frac(px/1k))` | idem | ✅ | ✅ OK |
| `btc_dist_5k` | `abs(px%5000)/5000` | idem | ✅ | 🚫 quase-constante para BTC ~$95k |
| `btc_dist_10k` | `abs(px%10000)/10000` | idem | ✅ | 🚫 quase-constante para BTC ~$95k |
| `btc_spot_vol_ratio` | `vol_5m / (vol_55m/11)` | idem | ✅ | ✅ OK |

### Grupo B — L2 Orderbook REST (20 features)

| Feature | Descrição | Snap | Paridade | Viabilidade |
|---|---|---|---|---|
| `ob_mid` | `(best_ask+best_bid)/2` | Open ~t60s | ✅ | ✅ OK |
| `ob_spread` | `best_ask - best_bid` | Open | ✅ | ✅ OK |
| `ob_imbalance` | `(bid_sz-ask_sz)/(bid_sz+ask_sz)` best-level | Open | ✅ | ✅ OK |
| `ob_depth_ratio` | `bid_depth_5c / ask_depth_5c` | Close | ✅ | ✅ OK |
| `ob_bid_depth_5c` | `Σbid_sz(p≥mid-0.05) / Σbid_sz` | Close | ✅ | ✅ OK |
| `ob_ask_depth_5c` | `Σask_sz(p≤mid+0.05) / Σask_sz` | Close | ✅ | ✅ OK |
| `ob_total_depth` | `Σbid_sz + Σask_sz` | Close | ✅ | ✅ OK |
| `ob_weighted_imb` | `Σ(bid_sz·exp(-10·Δp) - ask_sz·exp(-10·Δp)) / Σwt` | Close | ✅ | ✅ OK |
| `ob_mid_drift` | `mid_close - mid_open` | Temporal | ⚠️ close t~168s train vs t~240s live | ✅ OK (sinal válido) |
| `ob_imbalance_end` | `imbalance` close snapshot | Temporal | ✅ | ✅ OK |
| `ob_spread_end` | `spread` close snapshot | Temporal | ✅ | ✅ OK |
| `ob_depth_change` | `depth_close - depth_open` | Temporal | ✅ | ✅ OK |
| `ob_imb_momentum` | `imb_close - imb_open` | Temporal | ⚠️ mesmo que ob_mid_drift | ✅ OK (sinal válido) |
| `ob_imb_w0` | `mean(imb, t∈[0,60s))` | WS books | ✅ | ⚠️ live: quase sempre 0.0 (1 real book snap no WS) |
| `ob_imb_w1` | `mean(imb, t∈[60,120s))` | WS books | ✅ | ✅ OK |
| `ob_imb_w2` | `mean(imb, t∈[120,168s))` | WS books | ✅ | ✅ OK |
| `ob_pc_up_ratio` | `n_up_changes / n_total` em `[0,168s)` | WS pc | ✅ | ✅ OK |
| `ob_pc_volatility` | `std(price_diffs)` em `[0,168s)` | WS pc | ✅ | ✅ OK |
| `ob_pc_count` | `n_total_changes` em `[0,168s)` | WS pc | ✅ | ⚠️ reset em reconexão WS |
| `ob_fill_imbalance` | `(buy_vol-sell_vol)/(buy_vol+sell_vol)` em `[0,168s)` | WS pc | ✅ | ⚠️ campo `size` ausente em alguns eventos WS |

### Grupo B — CLOB WebSocket (10 features)

| Feature | Fórmula | Janela | Paridade | Viabilidade |
|---|---|---|---|---|
| `clob_imb_mean` | `mean(real_imbalance)` | `[108,168s)` | ✅ | ⚠️ zeros se WS subscrito tarde (t>108s) |
| `clob_imb_std` | `std(real_imbalance)` | idem | ✅ | ⚠️ idem |
| `clob_imb_drift` | `imb[-1] - imb[0]` | idem | ✅ | ⚠️ idem |
| `clob_spread_mean` | `mean(spread)` todos snaps | idem | ✅ | ⚠️ idem |
| `clob_spread_trend` | slope linear spread vs t | idem | ✅ | ⚠️ idem |
| `clob_mid_velocity` | slope linear mid vs t | idem | ✅ | ⚠️ idem |
| `clob_mid_volatility` | `std(diff(mid))` | idem | ✅ | ⚠️ idem |
| `clob_activity_rate` | `(n_real_books + n_pcs) / time_span` | idem | ✅ | ⚠️ idem |
| `clob_depth_trend` | slope linear `total_depth` vs t | idem | ✅ | ⚠️ idem |
| `clob_ask_pressure` | `frac(ASK moves < 0)` | idem | ✅ | ⚠️ idem |

> **Nota clob_***: subscrição CLOB WS ocorre quando `fetch_market(slot_ts)` é chamado (~t=0 do slot se o slot já estava no pipeline, ou mais tarde para slots descobertos depois). Para mercados subscritos antes de t=108s, a janela estará cheia. Para subscrições tardias, zeros são retornados — modelo usa defaults neutros.

### Grupo C — Tick-based Order Flow (19 features)

| Feature | Fórmula | Paridade | Viabilidade |
|---|---|---|---|
| `btc_up_ratio` | `vol_up / total` | ✅ | ✅ OK |
| `btc_n_ticks` | `len(ticks)` | ✅ | ✅ OK |
| `btc_vol_up` | `Σsize_usdc` onde Up | ✅ | 🚫 volume absoluto não-normalizado |
| `btc_vol_dn` | `Σsize_usdc` onde Down | ✅ | 🚫 volume absoluto não-normalizado |
| `btc_vwap_up` | `Σ(price·size)/vol_up` | ✅ | 🚫 muito ruidoso com n<5 ticks |
| `btc_vwap_dn` | `Σ(price·size)/vol_dn` | ✅ | 🚫 muito ruidoso com n<5 ticks |
| `btc_vwap_spread` | `vwap_up - vwap_dn` | ✅ | 🚫 combinação linear de dois ruidosos |
| `btc_buy_ratio` | `Σsize_usdc(side=BUY) / total` | ✅ | ✅ OK |
| `btc_momentum` | `mean(w3..5) - mean(w0..2)` | ✅ | ✅ OK (w2-5 = 0.5, captura w1-w0) |
| `btc_size_disparity` | `avg_up_size - avg_dn_size` | ✅ | ✅ OK |
| `btc_up_ratio_stability` | `std(w0, w1)` | ✅ | ✅ OK |
| `btc_signal_conviction` | `btc_up_ratio × (1 - stability)` | ✅ | 🚫 combinação linear de features já no set |
| `btc_tw_up_ratio` | exponential decay weighted | ✅ | ✅ OK |
| `btc_up_w0` | `up_ratio` em `[0,30s)` | ✅ | ✅ OK |
| `btc_up_w1` | `up_ratio` em `[30,60s)` | ✅ | ✅ OK |
| `btc_up_w2` | `up_ratio` em `[60,90s)` | ✅ | 🚫 DEAD — sempre 0.5 (OBS=60s) |
| `btc_up_w3` | `up_ratio` em `[90,120s)` | ✅ | 🚫 DEAD — sempre 0.5 |
| `btc_up_w4` | `up_ratio` em `[120,150s)` | ✅ | 🚫 DEAD — sempre 0.5 |
| `btc_up_w5` | `up_ratio` em `[150,180s)` | ✅ | 🚫 DEAD — sempre 0.5 |

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

| Feature | Paridade | Viabilidade |
|---|---|---|
| `hour_sin` | ✅ | ✅ OK |
| `hour_cos` | ✅ | ✅ OK |
| `dow_sin` | ✅ | ✅ OK |
| `dow_cos` | ✅ | ✅ OK |
| `hour_x_up_ratio` | ✅ | ✅ OK |
| `hour_x_tw_ur` | ✅ | ✅ OK |

### Grupo F — Cross-domain Interactions (8 features)

| Feature | Paridade | Viabilidade |
|---|---|---|
| `x_imb_x_inslot` | ✅ | ✅ OK |
| `x_imb_end_x_ret` | ✅ | ✅ OK |
| `x_drift_x_ret5m` | ✅ | ✅ OK |
| `x_spread_x_vol` | ✅ | ✅ OK |
| `x_depth_x_vol` | ✅ | ✅ OK |
| `x_imb_x_ur` | ✅ | ✅ OK |
| `x_depth_x_momentum` | ✅ | ✅ OK |
| `x_ob_drift_x_inslot` | ✅ | ✅ OK |

---

## 4. Auditoria de lookahead

O bot decide em `t ∈ [170, 240]s`. Qualquer feature com dados de `t > 60s` do slot atual é lookahead.

| Feature | Timestamp de referência | Lookahead? | Justificativa |
|---|---|---|---|
| `btc_pre_*_ret` | `obs_end_ts = slot_ts + 60s` | ✅ NÃO | `obs_end_ts < t_entry=170s` |
| `btc_inslot_*` | `[slot_ts, slot_ts+60s]` | ✅ NÃO | janela fecha em t=60s |
| `btc_vol_1h/4h` | `[slot_ts-3600/14400, slot_ts]` | ✅ NÃO | usa só dados pré-slot |
| `btc_spot_vol_ratio` | `[slot_ts-3600, slot_ts]` | ✅ NÃO | idem |
| `btc_dist_*` | `px[slot_ts+60s]` | ✅ NÃO | t=60s < t_entry |
| `ob_mid/spread/imbalance` | open snapshot ~t60s | ✅ NÃO | pré-entry |
| `ob_mid_drift` | `mid_close(~t168s) - mid_open(~t60s)` | ✅ NÃO | t_close=168s < t_entry=170s |
| `ob_imb_momentum` | idem | ✅ NÃO | idem |
| `ob_imb_w0/w1/w2` | janelas `[0,60)/[60,120)/[120,168)` | ✅ NÃO | max=168s < t_entry |
| `ob_pc_*` | `[0, 168s)` | ✅ NÃO | 168s < t_entry |
| `clob_*` | janela `[108, 168)` | ✅ NÃO | max=168s < t_entry |
| ticks (Grupo C) | `t ∈ [0, 60s)` | ✅ NÃO | chegam em t~170s via data-api |
| `lag_*` | slots ANTERIORES | ✅ NÃO | sempre passado |
| `hour_*`, `dow_*` | `slot_ts` | ✅ NÃO | timestamp público |
| `x_*` cross features | compostos de A+B+C acima | ✅ NÃO | todos ≤168s |

---

## 5. Análise de viabilidade live

### 5.1 Features problemáticas — candidatas a remoção no próximo treino

| Feature | Categoria | Problema | Ação recomendada |
|---|---|---|---|
| `btc_up_w2` | DEAD | OBS=60s — janela `[60,90s)` sempre vazia → constante 0.5 | **REMOVER** |
| `btc_up_w3` | DEAD | OBS=60s — janela `[90,120s)` sempre vazia → constante 0.5 | **REMOVER** |
| `btc_up_w4` | DEAD | OBS=60s — janela `[120,150s)` sempre vazia → constante 0.5 | **REMOVER** |
| `btc_up_w5` | DEAD | OBS=60s — janela `[150,180s)` sempre vazia → constante 0.5 | **REMOVER** |
| `btc_vol_up` | REDUNDANTE/RUIDOSO | Volume absoluto USDC. Não-normalizado. Colinear com `btc_n_ticks`. `btc_up_ratio` já captura a direção | **REMOVER** |
| `btc_vol_dn` | REDUNDANTE/RUIDOSO | Idem `btc_vol_up` | **REMOVER** |
| `btc_vwap_spread` | REDUNDANTE | = `btc_vwap_up - btc_vwap_dn` — combinação linear | **REMOVER** |
| `btc_signal_conviction` | REDUNDANTE | = `btc_up_ratio × (1 - btc_up_ratio_stability)` — produto de dois features já no set | **REMOVER** |
| `btc_vwap_up` | RUIDOSO | VWAP com n<5 ticks (OBS=60s) é extremamente instável. 1 trade grande distorce completamente | **REMOVER** |
| `btc_vwap_dn` | RUIDOSO | Idem `btc_vwap_up` | **REMOVER** |
| `btc_dist_5k` | QUASE-CONSTANTE | `abs(px%5000)/5000`. Para BTC ~$95k: varia muito pouco. Menos informativo que `btc_dist_1k` | **REMOVER** |
| `btc_dist_10k` | QUASE-CONSTANTE | `abs(px%10000)/10000`. Para BTC ~$95k-100k: alternância binária ~0.0/0.5 | **REMOVER** |
| `ob_imb_w0` | ESPARSO LIVE | Janela `[0,60s)`. Polymarket WS envia apenas 1 real book snap ao subscribir. Se não chegar em t<60s → 0.0. Live ~50-80% das vezes = 0.0 vs valor real no treino | **REMOVER** |
| `ob_pc_count` | ESPARSO LIVE | Treino: poly_l2 histórico rico. Live: reset em reconexão WS. Coverage sistematicamente menor | **REMOVER** |
| `ob_fill_imbalance` | ESPARSO LIVE | Campo `size` nos eventos WS `price_change` frequentemente ausente → fill_imbalance=0.0 | **REMOVER** |

**Total a remover: 15 features → 104 → 89 candidatas para treino v29**

### 5.2 Features com ressalvas operacionais (manter, monitorar)

| Feature | Ressalva | Mitigação |
|---|---|---|
| `btc_pre_4h_ret`, `btc_vol_4h`, `btc_pre_1h_4h_ratio` | Buffer warm-up: precisa de 240+ candles (4h). Se bot reiniciar → 0.0 nas primeiras 4h | `_seed_spot_buffers()` preenche na inicialização via REST. Aceitar |
| `clob_*` (10 features) | Zeros se subscrição WS chegar após t=108s | Subscrição proativa em `fetch_market()`. Slots novos cobertos. Aceitar |
| `ob_imb_w1`, `ob_imb_w2` | Polymarket WS envia poucas real book snaps; dependem de eventos `book` reais chegando | Janelas w1/w2 mais prováveis de ter dados que w0. Aceitar |
| `ob_pc_up_ratio`, `ob_pc_volatility` | Reset em reconexão WS perde histórico parcial | Fallback neutro (0.5 / 0.0). Aceitar |
| `ob_mid_drift`, `ob_imb_momentum` | Close snapshot ~t240s no live vs ~t168s no treino (~72s de diferença) | Sinal de drift ainda informativo e correlacionado. Aceitar |

### 5.3 Cobertura live esperada por grupo

| Grupo | Cobertura live esperada | Observação |
|---|---|---|
| A — Spot | ~100% | Binance WS estável. REST fallback ativo |
| B — OB REST | ~95% | Falha em mercados com book vazio/one-sided |
| B — CLOB WS | ~80% | Zeros para slots subscritos após t=108s |
| C — Ticks | ~90% | data-api pode falhar; fallback neutro |
| D — Lag | ~95% | Ring buffer populado por `_seed_slot_history()` |
| E — Temporal | ~100% | Só depende de `slot_ts` |
| F — Cross | ~95% | Depende dos componentes; fallback gracioso |

---

## 6. Grupo A — Spot Binance

**Arquivo treino:** `scripts/train_v28_modal.py` linhas ~316-370  
**Arquivo live:** `deploy/live_trader.py` função `build_spot_features()` linhas ~608-760

**Referência de preço:** `px_now = spot_at(obs_end_ts)` onde `obs_end_ts = slot_ts + OBS_SECS`

```python
btc_pre_5m_ret  = (px[obs_end] / px[slot_ts - 300]) - 1
btc_pre_15m_ret = (px[obs_end] / px[slot_ts - 900]) - 1
btc_pre_30m_ret = (px[obs_end] / px[slot_ts - 1800]) - 1
btc_pre_1h_ret  = (px[obs_end] / px[slot_ts - 3600]) - 1
btc_pre_4h_ret  = (px[obs_end] / px[slot_ts - 14400]) - 1

btc_inslot_ret   = (px_arr[-1] / px_arr[0]) - 1     # candles em [slot_ts, obs_end]
btc_inslot_vol   = std(px_arr) / mean(px_arr)
btc_inslot_range = (max_high - min_low) / px_now

btc_vol_1h = std(diff(px) / px[:-1])  # candles em [slot_ts-3600, slot_ts]
btc_vol_4h = std(diff(px) / px[:-1])  # candles em [slot_ts-14400, slot_ts]

btc_pre_1h_4h_ratio = (px_now - px_1h) / (px_now - px_4h + 1e-9)

btc_dist_1k  = min(frac(px/1000), 1 - frac(px/1000))
btc_dist_5k  = abs(px % 5000) / 5000    # 🚫 candidata a remoção
btc_dist_10k = abs(px % 10000) / 10000  # 🚫 candidata a remoção

btc_spot_vol_ratio = vol_5m / (vol_55m / 11)
```

---

## 7. Grupo B — L2 Orderbook

**Fetch treino:** `scripts/fetch_ob_features_modal.py`  
**Live:** `deploy/live_trader.py` → `_build_ob_features()` + `ClobFeatureAccumulator`

### Open vs Close snapshot

```
slot_ts          ~t60s (open)           t=168s (CUTOFF_SEC)
   |                 |                       |
   | open_snap REST  |                       | close_snap REST
   | -> ob_mid       |                       | -> ob_mid_drift = close.mid - open.mid
   | -> ob_imbalance |                       | -> ob_imbalance_end
   | -> ob_spread    |                       | -> ob_spread_end
   |                 |                       | -> ob_depth_change, ob_imb_momentum
   |                 |<--- WS book snaps --->| -> ob_imb_w0/w1/w2 (real books)
   |<--- WS price_change events ----------->| -> ob_pc_* features
   |<--- WS (janela [108,168)) ------------>| -> clob_* features
```

### Fórmulas

```python
# Static (open snapshot REST ~t60s)
ob_mid         = (best_ask + best_bid) / 2
ob_spread      = best_ask - best_bid
ob_imbalance   = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-8)

# Static (close snapshot REST ~t168s)
ob_depth_ratio  = bid_depth_5c / (ask_depth_5c + 1e-8)
ob_bid_depth_5c = Σbid_sz(price ≥ mid-0.05) / Σbid_sz_total
ob_ask_depth_5c = Σask_sz(price ≤ mid+0.05) / Σask_sz_total
ob_total_depth  = Σbid_sz + Σask_sz
ob_weighted_imb = (Σbid_sz·exp(-10·|bid_p-mid|) - Σask_sz·exp(-10·|ask_p-mid|)) / Σwt

# Temporal (open → close)
ob_mid_drift     = mid_close - mid_open
ob_imbalance_end = imbalance_close
ob_spread_end    = spread_close
ob_depth_change  = total_depth_close - total_depth_open
ob_imb_momentum  = imbalance_close - imbalance_open

# Windowed imbalance (real WS book snaps)
ob_imb_w0 = mean(imbalance, t ∈ [0,   60s))   # 🚫 live: quase sempre 0.0
ob_imb_w1 = mean(imbalance, t ∈ [60, 120s))
ob_imb_w2 = mean(imbalance, t ∈ [120,168s))

# Price-change aggregates (WS price_change events)
ob_pc_up_ratio   = n_up_changes / n_total_changes   # t ∈ [0,168s)
ob_pc_volatility = std(price_diffs)                 # t ∈ [0,168s)
ob_pc_count      = n_total_changes                  # 🚫 reset em reconexão
ob_fill_imbalance = (buy_vol-sell_vol)/(buy_vol+sell_vol+1e-8)  # 🚫 size ausente

# CLOB WS microstructure (janela [slot_ts+108, slot_ts+168))
clob_imb_mean      = mean(real_book_imbalances)
clob_imb_std       = std(real_book_imbalances)
clob_imb_drift     = imb[-1] - imb[0]
clob_spread_mean   = mean(all_spreads)
clob_spread_trend  = linslope(t, spreads)
clob_mid_velocity  = linslope(t, mids)
clob_mid_volatility = std(diff(mids))
clob_activity_rate = (n_real_books + n_pcs) / time_span
clob_depth_trend   = linslope(t, real_depths)
clob_ask_pressure  = frac(ASK price moves < 0)
```

---

## 8. Grupo C — Tick-based Order Flow

**Janela:** `t ∈ [0, OBS_SECS=60s)`. Fonte: data-api trades, lag ~90-120s.

```python
vol_up = Σsize_usdc  onde outcome="Up"
vol_dn = Σsize_usdc  onde outcome="Down"
total  = vol_up + vol_dn + 1e-8

btc_up_ratio         = vol_up / total
btc_n_ticks          = len(ticks)
btc_vol_up           = vol_up          # 🚫 não-normalizado
btc_vol_dn           = vol_dn          # 🚫 não-normalizado
btc_vwap_up          = Σ(price·sz)/vol_up   # 🚫 ruidoso n<5
btc_vwap_dn          = Σ(price·sz)/vol_dn   # 🚫 ruidoso n<5
btc_vwap_spread      = vwap_up - vwap_dn    # 🚫 combinação linear
btc_buy_ratio        = Σsz(side="BUY") / total
btc_momentum         = mean(w3..5) - mean(w0..2)
btc_size_disparity   = avg_up_size - avg_dn_size
btc_up_ratio_stability = std(w0, w1)
btc_signal_conviction  = btc_up_ratio × (1 - stability)  # 🚫 redundante
btc_tw_up_ratio        = Σ(up·sz·exp(-0.02·(60-t))) / Σ(sz·exp(-0.02·(60-t)))

# Sub-janelas (OBS=60s: só w0 e w1 têm dados reais)
btc_up_w0 = up_ratio em [0,  30s)
btc_up_w1 = up_ratio em [30, 60s)
btc_up_w2 = 0.5  # 🚫 DEAD — [60, 90s) fora de OBS=60
btc_up_w3 = 0.5  # 🚫 DEAD
btc_up_w4 = 0.5  # 🚫 DEAD
btc_up_w5 = 0.5  # 🚫 DEAD
```

---

## 9. Grupo D — Lag History

**Ring buffer** (`_slot_history`): máximo 20 entradas, cada uma com `{slot_ts, up_ratio, target, n_ticks, vol_total, sw[6]}`.

```python
# Staleness guard: gap > lag * 300 * 3 → fill neutro
for lag in [1..5]:
    lag_{lag}_outcome        = target[rank - lag]      (ou 0.5)
    prev_slot_up_ratio_{lag} = up_ratio[rank - lag]    (ou 0.5)
    prev_slot_n_ticks_{lag}  = n_ticks[rank - lag]     (ou 0.0)
    prev_slot_vol_{lag}      = vol_total[rank - lag]   (ou 0.0)

lag_streak = N consecutivos na mesma direção (break no primeiro divergente, max=5)

lag_ur_zscore_20 = clip((prev_ur_1 - mean(hist_20)) / max(std(hist_20), 0.01), -5, 5)
lag_ur_zscore_5  = clip((prev_ur_1 - mean(hist_5))  / max(std(hist_5),  0.01), -5, 5)
```

> `btc_up_ratio_zscore_5s` e `btc_up_ratio_zscore_20s` usam `btc_up_ratio` atual — diferentes dos `lag_ur_zscore_*` que usam `prev_slot_up_ratio_1`.

---

## 10. Grupo E — Temporal

```python
dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
hour = dt.hour + dt.minute / 60.0
dow  = dt.weekday()

hour_sin = sin(2π · hour / 24)
hour_cos = cos(2π · hour / 24)
dow_sin  = sin(2π · dow / 7)
dow_cos  = cos(2π · dow / 7)

hour_x_up_ratio = btc_up_ratio   × (hour / 24.0)
hour_x_tw_ur    = btc_tw_up_ratio × (hour / 24.0)
```

---

## 11. Grupo F — Cross-domain Interactions

```python
x_imb_x_inslot      = ob_imbalance     × btc_inslot_ret
x_imb_end_x_ret     = ob_imbalance_end × btc_inslot_ret
x_drift_x_ret5m     = ob_mid_drift     × btc_pre_5m_ret
x_spread_x_vol      = ob_spread        × btc_vol_1h     # default ob_spread=0.02
x_depth_x_vol       = ob_depth_ratio   × btc_vol_1h     # default ob_depth_ratio=1.0
x_imb_x_ur          = ob_imbalance     × btc_up_ratio
x_depth_x_momentum  = ob_depth_ratio   × btc_momentum
x_ob_drift_x_inslot = ob_mid_drift     × btc_inslot_ret
```

---

## 12. Defaults quando dados ausentes

```python
# OB REST falha (book vazio/one-sided)
ob_mid=0.5, ob_spread=0.02, ob_ask_depth_5c=0.5, ob_bid_depth_5c=0.5
ob_depth_ratio=1.0, ob_total_depth=1000.0
ob_imbalance/drift/momentum/change → 0.0

# ob_pc_* (WS sem eventos)
ob_pc_up_ratio=0.5, ob_pc_volatility=0.0, ob_pc_count=0.0, ob_fill_imbalance=0.0

# Ticks (data-api falha ou slot sem ticks)
btc_up_ratio=0.5, btc_n_ticks=0.0, btc_vol_up/dn=0.0
btc_vwap_up/dn=0.5, btc_buy_ratio=0.5, btc_momentum=0.0
btc_tw_up_ratio=0.5, btc_up_w{0..5}=0.5

# Lag history (vazio ou stale)
lag_{1..5}_outcome=0.5, prev_slot_up_ratio_{1..5}=0.5
prev_slot_n_ticks_{1..5}=0.0, prev_slot_vol_{1..5}=0.0
lag_streak=0.0, lag_ur_zscore_*=0.0
```

---

## 13. Seleção de features no treino

### v28 (atual) — 104 candidatas, top-N por importância LightGBM

1. Screening: LightGBM `n_estimators=300, max_depth=4`, 5-fold TimeSeriesSplit
2. Ranking: acumula `feature_importances_` over folds
3. Teste N ∈ {40, 30, 25, 20, 15}: avalia AUC com modelo completo
4. Seleção: N com melhor AUC médio

### v29 (próximo) — remover 15 features problemáticas antes do screening

**Remover do candidato pool (antes do treino):**
```
DEAD (4):       btc_up_w2, btc_up_w3, btc_up_w4, btc_up_w5
REDUNDANTE (4): btc_vol_up, btc_vol_dn, btc_vwap_spread, btc_signal_conviction
RUIDOSO (2):    btc_vwap_up, btc_vwap_dn
QUASE-CONST(2): btc_dist_5k, btc_dist_10k
ESPARSO (3):    ob_imb_w0, ob_pc_count, ob_fill_imbalance
```

**104 → 89 candidatas** para screening de importância → top-N selection.

Benefícios:
- Screening foca em features com sinal real
- Menos ruído no ranking de importância
- Treino ~15% mais rápido
- Menos chance de overfitting em features que variam sistematicamente entre treino e live

---

## 14. Checklist pré-treino

Antes de rodar `modal run scripts/train_v28_modal.py` (ou v29):

- [ ] `fetch_ob_features_modal.py` finalizou — todos os ~23.221 mercados no `ob_features_full.parquet`
- [ ] `ob_features_full.parquet` uploaded para HF em `data/ob_features_full.parquet`
- [ ] `all_markets.csv` e `ticks_btc_full_clean.parquet` atualizados no HF
- [ ] `binance_spot_full.parquet` cobre todo o período dos markets
- [ ] Checar que `OBS_SECS` no treino == `OBSERVE_SECS` no live (ambos = 60)
- [ ] Checar que `SLOT_DURATION` no treino == `SLOT_DURATION` no live (ambos = 300)
- [ ] **v29+**: confirmar que as 15 features 🚫 foram removidas do candidato pool

---

## 15. Changelog de features por versão

| Versão | Mudanças principais |
|---|---|
| v21 | Primeiro modelo "real-time only". Removeu tick features — OBS=180s criava lag. |
| v22-v23 | Adicionou `btc_up_ratio_stability`, `hour_sin/cos`, `btc_up_w5_zscore`. Mudou OBS para 60s. |
| v24 | Fix: stability usa só janelas reais (n=OBS//30=2), não 6. |
| v25 | Introduziu OB features (ob_*) via pmdata poly_l2. AUC baseline 0.8575. |
| v26 | CLOB WS features (clob_*). Cross-features x_imb_end_x_ret etc. Fix btc_spot_vol_ratio. |
| v27 | `ob_features_full.parquet` com clob_* incluídos. Fix prefixo `clob_*` no treino. |
| **v28** | Full train==live parity: tick features no treino, dow_sin/cos, hour_x_up_ratio/tw_ur, todos cross-features. lag_{1..5}_outcome e lag_streak no live. 104 features, diff=0. |
| **v28 patch** | Eliminação de todas as divergências: get_features() slot-anchored [108,168); get_ob_pc_features() [0,168); get_windowed_imbalance() real book snaps; reset_token() na troca de slot; MAX_BUFFER_SECS=360. |
| **v29 (plan)** | Remover 15 features problemáticas (4 DEAD + 4 redundantes + 2 ruidosas + 2 quase-constantes + 3 esparsas live). 104→89 candidatas. Auto feature selection aprimorado. |

---

*Atualizar sempre que adicionar, remover ou modificar qualquer feature.*
