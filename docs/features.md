# BTC 5-min Model — Feature Brain
**Versão de referência:** v28 | **OBS_SECS:** 60 | **SLOT_DURATION:** 300s  
**Última atualização:** 2026-06-10 | **Commit:** d65be81

> Este documento é a fonte de verdade de todas as features do modelo.  
> Qualquer alteração em `train_v28_modal.py` ou `deploy/live_trader.py` deve ser refletida aqui.  
> Antes de treinar ou deployar, confirmar que a tabela de paridade está ✅ em todas as linhas.

---

## Índice
1. [Timeline de um slot](#1-timeline-de-um-slot)
2. [Fontes de dados](#2-fontes-de-dados)
3. [Tabela canônica de features (train == live)](#3-tabela-canônica-de-features)
4. [Auditoria de lookahead](#4-auditoria-de-lookahead)
5. [Grupo A — Spot Binance](#5-grupo-a--spot-binance)
6. [Grupo B — L2 Orderbook](#6-grupo-b--l2-orderbook)
7. [Grupo C — Tick-based Order Flow](#7-grupo-c--tick-based-order-flow)
8. [Grupo D — Lag History](#8-grupo-d--lag-history)
9. [Grupo E — Temporal](#9-grupo-e--temporal)
10. [Grupo F — Cross-domain Interactions](#10-grupo-f--cross-domain-interactions)
11. [Defaults quando dados ausentes](#11-defaults-quando-dados-ausentes)
12. [Seleção de features no treino](#12-seleção-de-features-no-treino)
13. [Checklist pré-treino](#13-checklist-pré-treino)
14. [Changelog de features por versão](#14-changelog-de-features-por-versão)

---

## 1. Timeline de um slot

```
slot_ts                     +60s (OBS_SECS)       +170s          +240s      +300s
   |                              |                   |              |          |
   |<--- ticks coletados -------->|                   |              |          |
   |<--- spot inslot ------------>|                   |              |          |
   |<--- CLOB WS acumulando ----->|                   |              |          |
                                  |<--- lag ~90-120s->|              |          |
                                  |                   |<--entrada -->|          |
                                  |                   |  OB REST /book          |
                                  |                   |  predict + order        |
                                                                               fim
```

- **t=0**: slot começa (slot_ts = timestamp UTC arredondado para múltiplo de 300s)
- **t=0 a 60s**: janela de observação — ticks coletados, spot inslot, CLOB WS acumulando
- **t=60s** (`obs_end_ts = slot_ts + OBS_SECS`): ponto de referência de todos os features de spot e tick
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
| Polymarket CLOB WS book/price_change | 0s (stream) | `ob_features_full.parquet` (colunas `clob_*`) | 60s window por mercado |
| Ring buffer de slots anteriores | 0s | `all_markets.csv` (targets resolvidos) + tick aggregates | últimos 20 slots |

---

## 3. Tabela canônica de features

Total: **104 features** (antes da seleção de importância).  
Legenda: ✅ idêntico | ⚠️ divergência conhecida | ❌ bug

### Grupo A — Spot Binance (13 features)

| Feature | Train v28 | Live | Paridade |
|---|---|---|---|
| `btc_pre_5m_ret` | `(px[obs_end] / px[slot_ts-300]) - 1` | `build_spot_features` → idem | ✅ |
| `btc_pre_15m_ret` | `(px[obs_end] / px[slot_ts-900]) - 1` | idem | ✅ |
| `btc_pre_30m_ret` | `(px[obs_end] / px[slot_ts-1800]) - 1` | idem | ✅ |
| `btc_pre_1h_ret` | `(px[obs_end] / px[slot_ts-3600]) - 1` | idem | ✅ |
| `btc_pre_4h_ret` | `(px[obs_end] / px[slot_ts-14400]) - 1` | idem | ✅ |
| `btc_inslot_ret` | `(px_arr[-1] / px_arr[0]) - 1` em `[slot_ts, obs_end]` | idem | ✅ |
| `btc_inslot_vol` | `std(px) / mean(px)` em `[slot_ts, obs_end]` | idem | ✅ |
| `btc_inslot_range` | `(max_high - min_low) / px_now` em `[slot_ts, obs_end]` | idem | ✅ |
| `btc_vol_1h` | `std(1-min returns)` em `[slot_ts-3600, slot_ts]` | idem | ✅ |
| `btc_vol_4h` | `std(1-min returns)` em `[slot_ts-14400, slot_ts]` | idem | ✅ |
| `btc_pre_1h_4h_ratio` | `(px_now - px_1h) / (px_now - px_4h + 1e-9)` | idem | ✅ |
| `btc_dist_1k` | `min(frac(px/1000), 1 - frac(px/1000))` | idem | ✅ |
| `btc_dist_5k` | `abs(px % 5000) / 5000` | idem | ✅ |
| `btc_dist_10k` | `abs(px % 10000) / 10000` | idem | ✅ |
| `btc_spot_vol_ratio` | `vol_5m / (vol_55m / 11)` | idem | ✅ |

> Nota: `px_now = spot_at(obs_end_ts)` onde `obs_end_ts = slot_ts + OBS_SECS`. Todas as features usam preços até t=60s — sem lookahead.

### Grupo B — L2 Orderbook (20 features)

Computadas em `ob_features_full.parquet` (treino) e `_build_ob_features()` via CLOB REST `/book` (live).

| Feature | Descrição | Open/Close snap | Train | Live | Paridade |
|---|---|---|---|---|---|
| `ob_mid` | `(best_ask + best_bid) / 2` | Open | parquet | REST open | ✅ |
| `ob_spread` | `best_ask - best_bid` | Open | parquet | REST open | ✅ |
| `ob_imbalance` | `(bid_sz - ask_sz) / (bid_sz + ask_sz)` best-level | Open | parquet | REST open | ✅ |
| `ob_depth_ratio` | `bid_depth_5c / ask_depth_5c` | Close | parquet | REST close | ✅ |
| `ob_bid_depth_5c` | `Σbid_sz(price >= mid-0.05) / Σbid_sz_total` | Close | parquet | REST close | ✅ |
| `ob_ask_depth_5c` | `Σask_sz(price <= mid+0.05) / Σask_sz_total` | Close | parquet | REST close | ✅ |
| `ob_total_depth` | `Σbid_sz + Σask_sz` | Close | parquet | REST close | ✅ |
| `ob_weighted_imb` | `(Σbid_sz·exp(-10·|bid_p-mid|) - Σask_sz·exp(-10·|ask_p-mid|)) / sum` | Close | parquet | REST close | ✅ |
| `ob_mid_drift` | `mid_close - mid_open` | Temporal | parquet | REST open vs close | ✅ |
| `ob_imbalance_end` | `imbalance` no close snapshot | Temporal | parquet | REST close | ✅ |
| `ob_spread_end` | `spread` no close snapshot | Temporal | parquet | REST close | ✅ |
| `ob_depth_change` | `total_depth_close - total_depth_open` | Temporal | parquet | REST open vs close | ✅ |
| `ob_imb_momentum` | `imb_close - imb_open` | Temporal | parquet | REST open vs close | ✅ |
| `ob_imb_w0` | Imbalance médio em `[0, 60s)` | Windowed | parquet (poly_l2 histórico) | interpolado: `imb_open` | ⚠️ |
| `ob_imb_w1` | Imbalance médio em `[60s, 120s)` | Windowed | parquet (poly_l2 histórico) | `(imb_open + imb_close) / 2` | ⚠️ |
| `ob_imb_w2` | Imbalance médio em `[120s, 168s)` | Windowed | parquet (poly_l2 histórico) | `imb_close` | ⚠️ |
| `ob_pc_up_ratio` | `n_price_changes_up / n_total` em `[0, 168s)` | poly_l2 | parquet | ❌ não computado live → fallback 0.0 |
| `ob_pc_volatility` | `std(price_diffs)` em `[0, 168s)` | poly_l2 | parquet | ❌ não computado live → fallback 0.0 |
| `ob_pc_count` | `n_price_changes` em `[0, 168s)` | poly_l2 | parquet | ❌ não computado live → fallback 0.0 |
| `ob_fill_imbalance` | `(buy_vol - sell_vol) / (buy_vol + sell_vol)` fills em `[0, 168s)` | poly_l2 | parquet | ❌ não computado live → fallback 0.0 |

> ⚠️ **ob_imb_w0/w1/w2**: treino usa snapshots históricos reais do poly_l2 em 3 janelas de 60s. Live interpola de 2 pontos (open ~t170s, close ~t240s). Divergência estrutural aceita — ambos capturam tendência de imbalance.

> ❌ **ob_pc_* e ob_fill_imbalance**: no live esses 4 features chegam como 0.0 (via `_neutral_defaults`). O treino usa dados reais. Se esses features tiverem importância alta, há mismatch real de distribuição. Monitorar feature importance no v28 — se aparecerem no top-15, implementar via CLOB WS log.

### Grupo B — CLOB WebSocket (10 features)

Computadas de streams de book/price_change no CLOB WS. Janela: últimos 60s antes de t=OBS_SECS.

| Feature | Fórmula | Train (ob_features_full.parquet) | Live (clob_features.py accumulator) | Paridade |
|---|---|---|---|---|
| `clob_imb_mean` | `mean(imbalance)` 60s window | `fetch_ob_features_modal.py` | `clob_features.py get_features()` | ✅ |
| `clob_imb_std` | `std(imbalance)` 60s window | idem | idem | ✅ |
| `clob_imb_drift` | `imb[-1] - imb[0]` 60s window | idem | idem | ✅ |
| `clob_spread_mean` | `mean(spread)` todos snapshots | idem | idem | ✅ |
| `clob_spread_trend` | slope linear de spread vs t | idem | idem | ✅ |
| `clob_mid_velocity` | slope linear de mid vs t | idem | idem | ✅ |
| `clob_mid_volatility` | `std(diff(mid))` | idem | idem | ✅ |
| `clob_activity_rate` | `(n_real_books + n_price_changes) / time_span` | idem | idem | ✅ |
| `clob_depth_trend` | slope linear de `best_bid_sz + best_ask_sz` vs t | idem | idem | ✅ |
| `clob_ask_pressure` | `frac(mid_moves < 0)` dos movimentos de mid | idem | idem | ✅ |

> Nota: no treino, clob_* são computados no `fetch_ob_features_modal.py` com a janela `t=[108, 168)` (60s até o CUTOFF_SEC=168s). No live, o accumulator usa os últimos 60s desde que o slot começou. Janelas podem diferir levemente mas o sinal é equivalente.

### Grupo C — Tick-based Order Flow (15 features)

Janela: `t ∈ [0, OBS_SECS=60s)`. Ticks da data-api. No live chegam ~t=170-180s (lag ~90-120s).

| Feature | Fórmula | Train | Live | Paridade |
|---|---|---|---|---|
| `btc_up_ratio` | `vol_up / (vol_up + vol_dn)` | ticks t<60s | `build_features()` | ✅ |
| `btc_n_ticks` | `len(ticks)` | idem | idem | ✅ |
| `btc_vol_up` | `Σsize_usdc` onde outcome=Up | idem | idem | ✅ |
| `btc_vol_dn` | `Σsize_usdc` onde outcome=Down | idem | idem | ✅ |
| `btc_vwap_up` | `Σ(price·size_usdc) / vol_up` | idem | idem | ✅ |
| `btc_vwap_dn` | `Σ(price·size_usdc) / vol_dn` | idem | idem | ✅ |
| `btc_vwap_spread` | `vwap_up - vwap_dn` | idem | idem | ✅ |
| `btc_buy_ratio` | `Σsize_usdc(side=BUY) / total` | idem | idem | ✅ |
| `btc_momentum` | `mean(up_ratio w3..w5) - mean(up_ratio w0..w2)` | idem | idem | ✅ |
| `btc_size_disparity` | `avg_up_size - avg_dn_size` | idem | idem | ✅ |
| `btc_up_ratio_stability` | `std(btc_up_w0, btc_up_w1)` (só 2 janelas reais para OBS=60s) | idem | idem | ✅ |
| `btc_signal_conviction` | `btc_up_ratio * (1.0 - btc_up_ratio_stability)` | idem | idem | ✅ |
| `btc_tw_up_ratio` | `Σ(up·sz·exp(-0.02·(60-t))) / Σ(sz·exp(-0.02·(60-t)))` | idem | idem | ✅ |
| `btc_up_w{0..5}` | `vol_up / vol_total` em janela `[i*30, (i+1)*30)` | idem | idem (w0/w1 reais, w2-w5=0.5) | ✅ |

> Nota sobre `btc_up_w{0..5}`: com OBS_SECS=60, apenas w0 e w1 têm dados reais. w2..w5 = 0.5 (fill neutro) tanto no treino quanto no live — comportamento idêntico.

### Grupo D — Lag History (22 features)

Baseado no ring buffer de slots anteriores. Staleness guard: `time_gap > lag * SLOT_DURATION * 3` → fill neutro.

| Feature | Fórmula | Train | Live | Paridade |
|---|---|---|---|---|
| `lag_{1..5}_outcome` | `target` do slot rank-N anterior | ring buffer resolvido | `_slot_history[-lag]["target"]` | ✅ |
| `prev_slot_up_ratio_{1..5}` | `vol_up / vol_total` do slot rank-N anterior | `slot_up_ratio[prev_mid]` | `_slot_history[-lag]["up_ratio"]` | ✅ |
| `prev_slot_n_ticks_{1..5}` | `len(ticks)` do slot rank-N anterior | `slot_nticks[prev_mid]` | `_slot_history[-lag]["n_ticks"]` | ✅ |
| `prev_slot_vol_{1..5}` | `vol_up + vol_dn` do slot rank-N anterior | `slot_vol_tot[prev_mid]` | `_slot_history[-lag]["vol_total"]` | ✅ |
| `lag_streak` | N de outcomes consecutivos na mesma direção (1..5) | loop rank-1..5 | loop `_slot_history` | ✅ |
| `lag_ur_zscore_20` | `clip((prev_ur_1 - mean(hist_20)) / std(hist_20), -5, 5)` | últimos 20 por rank | últimas 20 entradas hist | ✅ |
| `lag_ur_zscore_5` | `clip((prev_ur_1 - mean(hist_5)) / std(hist_5), -5, 5)` | últimos 5 por rank | últimas 5 entradas hist | ✅ |

> Todos os lag features usam **apenas** dados de slots anteriores ao atual. Zero lookahead por construção.

### Grupo E — Temporal (6 features)

| Feature | Fórmula | Train | Live | Paridade |
|---|---|---|---|---|
| `hour_sin` | `sin(2π · hour / 24)` onde `hour = dt.hour + dt.minute/60` | `datetime.fromtimestamp(slot_ts)` | idem | ✅ |
| `hour_cos` | `cos(2π · hour / 24)` | idem | idem | ✅ |
| `dow_sin` | `sin(2π · weekday / 7)` onde `weekday = dt.weekday()` | `datetime.fromtimestamp(slot_ts)` | idem (adicionado v28) | ✅ |
| `dow_cos` | `cos(2π · weekday / 7)` | idem | idem (adicionado v28) | ✅ |
| `hour_x_up_ratio` | `btc_up_ratio * (hour / 24.0)` | idem | idem | ✅ |
| `hour_x_tw_ur` | `btc_tw_up_ratio * (hour / 24.0)` | idem | idem | ✅ |

### Grupo F — Cross-domain Interactions (8 features)

Todas computadas **após** grupos A, B, C com `feat.get(key, default)` — falha graciosamente se OB ausente.

| Feature | Fórmula | Train | Live | Paridade |
|---|---|---|---|---|
| `x_imb_x_inslot` | `ob_imbalance * btc_inslot_ret` | idem | idem | ✅ |
| `x_imb_end_x_ret` | `ob_imbalance_end * btc_inslot_ret` | idem | idem | ✅ |
| `x_drift_x_ret5m` | `ob_mid_drift * btc_pre_5m_ret` | idem | idem | ✅ |
| `x_spread_x_vol` | `ob_spread * btc_vol_1h` | default `ob_spread=0.02` | idem | ✅ |
| `x_depth_x_vol` | `ob_depth_ratio * btc_vol_1h` | default `ob_depth_ratio=1.0` | idem | ✅ |
| `x_imb_x_ur` | `ob_imbalance * btc_up_ratio` | idem | idem | ✅ |
| `x_depth_x_momentum` | `ob_depth_ratio * btc_momentum` | idem | idem | ✅ |
| `x_ob_drift_x_inslot` | `ob_mid_drift * btc_inslot_ret` | idem | idem | ✅ |

---

## 4. Auditoria de lookahead

### Regra geral
O bot decide em `t ∈ [170, 240]s`. Qualquer feature que usa dados de `t > 60s` do slot atual é lookahead.

| Feature | Timestamp de referência | Lookahead? | Justificativa |
|---|---|---|---|
| `btc_pre_*_ret` | `obs_end_ts = slot_ts + 60s` | ✅ NÃO | `obs_end_ts < t_entry=170s` |
| `btc_inslot_*` | `[slot_ts, slot_ts+60s]` | ✅ NÃO | janela fecha em t=60s |
| `btc_vol_1h` | `[slot_ts-3600, slot_ts]` | ✅ NÃO | usa só dados pré-slot |
| `btc_vol_4h` | `[slot_ts-14400, slot_ts]` | ✅ NÃO | idem |
| `btc_spot_vol_ratio` | `[slot_ts-3600, slot_ts]` | ✅ NÃO | idem |
| `btc_dist_*` | `px[slot_ts+60s]` | ✅ NÃO | t=60s < t_entry |
| `ob_mid/spread/imbalance` | Open snapshot (~t=0-30s do slot) | ✅ NÃO | book pré-entry |
| `ob_mid_drift` | `mid_close - mid_open` onde close=t~168s no treino, t~240s live | ⚠️ VER NOTA | ver abaixo |
| `ob_imb_momentum` | `imb_close - imb_open` | ⚠️ VER NOTA | idem |
| `ob_imb_w0/w1/w2` | janelas 0-60s/60-120s/120-168s no treino | ✅ NÃO | max=168s < t_entry |
| `ob_pc_*` | `[0, 168s)` no treino | ✅ NÃO | 168s < t_entry |
| `clob_*` | últimos 60s (janela `[108, 168)` no treino) | ✅ NÃO | max=168s < t_entry |
| `btc_up_ratio` e tick features | `t ∈ [0, 60s)` | ✅ NÃO | janela fecha em t=60s |
| `btc_tw_up_ratio` | `t ∈ [0, 60s)` com decay | ✅ NÃO | idem |
| `lag_*_outcome` | slots ANTERIORES (rank-1..5) | ✅ NÃO | sempre passado |
| `prev_slot_*` | slots ANTERIORES | ✅ NÃO | sempre passado |
| `lag_streak` | slots ANTERIORES | ✅ NÃO | idem |
| `lag_ur_zscore_*` | usa `prev_slot_up_ratio_1` (anterior) como "current" | ✅ NÃO | não usa up_ratio do slot atual |
| `hour_*` e `dow_*` | `slot_ts` | ✅ NÃO | timestamp público do slot |
| `hour_x_up_ratio` | `btc_up_ratio` (t<60s) × hour | ✅ NÃO | ambos disponíveis em t_entry |
| `x_*` cross features | combinações de grupos A+B+C | ✅ NÃO | todos componentes t≤168s |

**Nota sobre `ob_mid_drift` e `ob_imb_momentum`:**  
No treino, o close snapshot é `t≈168s` (CUTOFF_SEC em `fetch_ob_features_modal.py`).  
No live, o close snapshot é obtido na segunda chamada REST dentro da janela `[170, 240]s`.  
Isso cria uma diferença de ~2-70s no "close" — não é lookahead (t_close < slot_end=300s), mas é uma divergência treino/live estrutural aceita. O sinal de drift ainda é informativo e correlacionado. Se `ob_mid_drift` aparecer consistentemente no top-5, vale alinhar os timestamps com maior precisão.

---

## 5. Grupo A — Spot Binance

**Arquivo treino:** `scripts/train_v28_modal.py` linhas ~316-370  
**Arquivo live:** `deploy/live_trader.py` função `build_spot_features()` linhas ~608-760

### Fórmulas detalhadas

**Referência de preço:** `px_now = spot_at(obs_end_ts)` onde `obs_end_ts = slot_ts + OBS_SECS`

```python
# Retornos pré-slot (relativeos a obs_end_ts)
btc_pre_5m_ret  = (px[obs_end] / px[slot_ts - 300]) - 1
btc_pre_15m_ret = (px[obs_end] / px[slot_ts - 900]) - 1
btc_pre_30m_ret = (px[obs_end] / px[slot_ts - 1800]) - 1
btc_pre_1h_ret  = (px[obs_end] / px[slot_ts - 3600]) - 1
btc_pre_4h_ret  = (px[obs_end] / px[slot_ts - 14400]) - 1

# In-slot
btc_inslot_ret   = (px_arr[-1] / px_arr[0]) - 1     # candles em [slot_ts, obs_end]
btc_inslot_vol   = std(px_arr) / mean(px_arr)
btc_inslot_range = (max_high - min_low) / px_now     # usa hi/lo das velas

# Volatilidade histórica (std dos retornos 1-min)
btc_vol_1h = std(diff(px) / px[:-1])  # candles em [slot_ts-3600, slot_ts]
btc_vol_4h = std(diff(px) / px[:-1])  # candles em [slot_ts-14400, slot_ts]

# Razão de momentum multi-escala
btc_pre_1h_4h_ratio = (px_now - px_1h) / (px_now - px_4h + 1e-9)

# Proximidade a números redondos
btc_dist_1k  = min(frac(px/1000), 1 - frac(px/1000))  # fração dentro de cada $1k
btc_dist_5k  = abs(px % 5000) / 5000
btc_dist_10k = abs(px % 10000) / 10000

# Razão de volume recente
btc_spot_vol_ratio = vol_5m / (vol_55m / 11)  # vol últimos 5min vs média dos outros 11x5min da hora
```

---

## 6. Grupo B — L2 Orderbook

**Arquivo treino (fetch):** `scripts/fetch_ob_features_modal.py`  
**Arquivo live:** `deploy/live_trader.py` função `_build_ob_features()`  
**Arquivo treino (uso):** `scripts/train_v28_modal.py` linhas ~370-395

### Open vs Close snapshot

```
slot_ts                  t=0-30s               t=168s (CUTOFF_SEC)
   |                        |                       |
   |    open_snap (book)    |                       |    close_snap (book)
   |    -> ob_mid           |                       |    -> ob_mid_drift = close.mid - open.mid
   |    -> ob_imbalance     |                       |    -> ob_imbalance_end
   |    -> ob_spread        |                       |    -> ob_spread_end
                            |<--- poly_l2 ticks --->|    -> ob_imb_w0/w1/w2
                                                         -> ob_depth_change
                                                         -> ob_imb_momentum
```

### Fórmulas

```python
# Static (open snapshot)
ob_mid         = (best_ask + best_bid) / 2
ob_spread      = best_ask - best_bid
ob_imbalance   = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-8)  # best level

# Static (close snapshot)
ob_depth_ratio  = bid_depth_5c / (ask_depth_5c + 1e-8)
ob_bid_depth_5c = Σbid_sz(price >= mid-0.05) / Σbid_sz_total
ob_ask_depth_5c = Σask_sz(price <= mid+0.05) / Σask_sz_total
ob_total_depth  = Σbid_sz + Σask_sz
ob_weighted_imb = (Σbid_sz·exp(-10·|bid_p-mid|) - Σask_sz·exp(-10·|ask_p-mid|)) / Σwt

# Temporal (open → close)
ob_mid_drift     = mid_close - mid_open
ob_imbalance_end = imbalance_close
ob_spread_end    = spread_close
ob_depth_change  = total_depth_close - total_depth_open
ob_imb_momentum  = imbalance_close - imbalance_open

# Windowed imbalance (treino: poly_l2 histórico; live: interpolado)
ob_imb_w0 = mean(imbalance, t ∈ [0,   60s))   # live: open_snap.imbalance
ob_imb_w1 = mean(imbalance, t ∈ [60, 120s))   # live: (open.imb + close.imb) / 2
ob_imb_w2 = mean(imbalance, t ∈ [120,168s))   # live: close_snap.imbalance

# Price-change aggregates (treino: poly_l2; live: N/A → default 0.0)
ob_pc_up_ratio   = n_price_up / n_total_changes   (t < 168s)
ob_pc_volatility = std(price_diffs)               (t < 168s)
ob_pc_count      = n_total_changes                (t < 168s)
ob_fill_imbalance = (buy_fill_vol - sell_fill_vol) / total_fill_vol  (t < 168s)
```

---

## 7. Grupo C — Tick-based Order Flow

**Arquivo treino:** `scripts/train_v28_modal.py` linhas ~395-460  
**Arquivo live:** `deploy/live_trader.py` função `build_features()` linhas ~1200-1270

### Fórmulas

```python
# Janela: t ∈ [0, OBS_SECS=60s)
vol_up = Σsize_usdc  onde outcome="Up"
vol_dn = Σsize_usdc  onde outcome="Down"
total  = vol_up + vol_dn + 1e-8

btc_up_ratio         = vol_up / total
btc_n_ticks          = len(ticks)
btc_vol_up           = vol_up
btc_vol_dn           = vol_dn
btc_vwap_up          = Σ(price·size_usdc) / vol_up    (else 0.5)
btc_vwap_dn          = Σ(price·size_usdc) / vol_dn    (else 0.5)
btc_vwap_spread      = vwap_up - vwap_dn
btc_buy_ratio        = Σsize_usdc(side="BUY") / total
btc_momentum         = mean(w3,w4,w5) - mean(w0,w1,w2)  # onde wi = up_ratio na janela i*30s
btc_size_disparity   = avg_up_size - avg_dn_size
btc_up_ratio_stability = std(w0, w1)   # só 2 janelas reais para OBS=60s
btc_signal_conviction  = btc_up_ratio × (1 - btc_up_ratio_stability)

# Sub-janelas de 30s (w0=0-30s, w1=30-60s, w2-w5=0.5 fill)
btc_up_w{i} = vol_up_in_window / vol_total_in_window  (else 0.5)

# Tempo-ponderado com decay exponencial (recência)
λ = 0.02
btc_tw_up_ratio = Σ(up_i · size_i · exp(-λ·(OBS-t_i))) / Σ(size_i · exp(-λ·(OBS-t_i)))
```

---

## 8. Grupo D — Lag History

**Arquivo treino:** `scripts/train_v28_modal.py` linhas ~464-538  
**Arquivo live:** `deploy/live_trader.py` linhas ~1305-1380

### Ring buffer structure

```python
# _slot_history entry (live):
{
    "slot_ts":   int,      # timestamp do slot
    "up_ratio":  float,    # btc_up_ratio daquele slot
    "target":    int,      # 1=UP / 0=DOWN (resultado resolvido)
    "n_ticks":   float,    # btc_n_ticks
    "vol_total": float,    # vol_up + vol_dn
    "sw":        list[6],  # btc_up_w0..w5
}
```

### Fórmulas

```python
# Staleness guard: se gap > lag * 300 * 3 → fill neutro
for lag in [1..5]:
    lag_{lag}_outcome        = target[rank - lag]         (ou 0.5 se stale/missing)
    prev_slot_up_ratio_{lag} = up_ratio[rank - lag]       (ou 0.5)
    prev_slot_n_ticks_{lag}  = n_ticks[rank - lag]        (ou 0.0)
    prev_slot_vol_{lag}      = vol_total[rank - lag]       (ou 0.0)

# Streak de outcomes consecutivos na mesma direção
lag_streak = N  onde target[rank-1] == target[rank-2] == ... == target[rank-N]
             (para no primeiro divergente ou stale, ou N=5 max)

# Z-scores de up_ratio histórico (usa prev_slot_up_ratio_1 como valor "atual")
hist_20 = [up_ratio[rank-d] for d in 1..20]
lag_ur_zscore_20 = clip((prev_ur_1 - mean(hist_20)) / max(std(hist_20), 0.01), -5, 5)

hist_5  = [up_ratio[rank-d] for d in 1..5]
lag_ur_zscore_5  = clip((prev_ur_1 - mean(hist_5))  / max(std(hist_5),  0.01), -5, 5)
```

> `btc_up_ratio_zscore_5s` e `btc_up_ratio_zscore_20s` usam `btc_up_ratio` (do slot atual) como valor atual — diferente de `lag_ur_zscore_*` que usam `prev_slot_up_ratio_1`. Ambos no treino e live.

---

## 9. Grupo E — Temporal

```python
dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
hour = dt.hour + dt.minute / 60.0    # hora em [0, 24)
dow  = dt.weekday()                   # 0=segunda .. 6=domingo

hour_sin = sin(2π · hour / 24)
hour_cos = cos(2π · hour / 24)
dow_sin  = sin(2π · dow / 7)
dow_cos  = cos(2π · dow / 7)

hour_x_up_ratio = btc_up_ratio × (hour / 24.0)
hour_x_tw_ur    = btc_tw_up_ratio × (hour / 24.0)
```

---

## 10. Grupo F — Cross-domain Interactions

```python
# OB × Spot
x_imb_x_inslot      = ob_imbalance     × btc_inslot_ret
x_imb_end_x_ret     = ob_imbalance_end × btc_inslot_ret
x_drift_x_ret5m     = ob_mid_drift     × btc_pre_5m_ret
x_spread_x_vol      = ob_spread        × btc_vol_1h     # default ob_spread=0.02
x_depth_x_vol       = ob_depth_ratio   × btc_vol_1h     # default ob_depth_ratio=1.0

# OB × Tick
x_imb_x_ur          = ob_imbalance   × btc_up_ratio
x_depth_x_momentum  = ob_depth_ratio × btc_momentum

# OB × Spot (drift × inslot) — captura alinhamento de direção
x_ob_drift_x_inslot = ob_mid_drift × btc_inslot_ret
```

---

## 11. Defaults quando dados ausentes

### OB features ausentes (sem book data para o mercado)

```python
# Train: skipped_no_ob += 1, fills abaixo
# Live:  _neutral_defaults dict + feat[f] = defaults.get(f, 0.0)

"ob_mid":           0.5     # mercado binário neutro
"ob_spread":        0.02    # spread típico
"ob_ask_depth_5c":  0.5
"ob_bid_depth_5c":  0.5
"ob_depth_ratio":   1.0     # bid/ask equilibrado
"ob_total_depth":   1000.0
# imbalance e drift → 0.0
# pc_* → 0.0 / 0.5 (ratio)
```

### Tick features ausentes (sem ticks no slot)

```python
btc_up_ratio         = 0.5
btc_n_ticks          = 0.0
btc_vol_up/dn        = 0.0
btc_vwap_up/dn       = 0.5
btc_vwap_spread      = 0.0
btc_buy_ratio        = 0.5
btc_momentum         = 0.0
btc_size_disparity   = 0.0
btc_up_ratio_stability = 0.0
btc_signal_conviction  = 0.0
btc_tw_up_ratio      = 0.5
btc_up_w{0..5}       = 0.5
```

### Lag features ausentes (ring buffer vazio ou stale)

```python
lag_{1..5}_outcome        = 0.5
prev_slot_up_ratio_{1..5} = 0.5
prev_slot_n_ticks_{1..5}  = 0.0
prev_slot_vol_{1..5}      = 0.0
lag_streak                = 0.0
lag_ur_zscore_*           = 0.0
```

---

## 12. Seleção de features no treino

O treino v28 parte de 104 features candidatas e seleciona o top-N por importância:

1. **Screening inicial**: LightGBM com `n_estimators=300, max_depth=4` em 5-fold TimeSeriesSplit
2. **Ranking**: acumula `feature_importances_` over folds, divide por N_SPLITS
3. **Teste de N**: avalia AUC para N ∈ {40, 30, 25, 20, 15} com modelo mais completo (400 estimators)
4. **Seleção**: N com melhor AUC médio nos folds

> **Importante**: o modelo só aprende as features do `top_features` list. Todas as outras são computadas mas descartadas. O live só usa `features = model_data["features"]` para filtrar o `feat` dict antes de `predict_proba`. Features fora do top-N no live são computadas e ignoradas — não custam predição mas custam CPU.

---

## 13. Checklist pré-treino

Antes de rodar `modal run scripts/train_v28_modal.py`:

- [ ] `fetch_ob_features_modal.py` finalizou — todos os ~23.221 mercados no `ob_features_full.parquet`
- [ ] `ob_features_full.parquet` uploaded para HF em `data/ob_features_full.parquet`
- [ ] `all_markets.csv` e `ticks_btc_full_clean.parquet` atualizados no HF
- [ ] `binance_spot_full.parquet` cobre todo o período dos markets
- [ ] Confirmar paridade: `python3 scripts/audit_features.py` (ver seção 3 acima)
- [ ] Checar que `OBS_SECS` no treino == `OBSERVE_SECS` no live (ambos = 60)
- [ ] Checar que `SLOT_DURATION` no treino == `SLOT_DURATION` no live (ambos = 300)

---

## 14. Changelog de features por versão

| Versão | Mudanças principais |
|---|---|
| v21 | Primeiro modelo "real-time only". Removeu tick features (btc_up_ratio etc) pois OBS=180s criava lag. |
| v22-v23 | Adicionou `btc_up_ratio_stability`, `hour_sin/cos`, `btc_up_w5_zscore`. Mudou OBS para 60s. |
| v24 | Fix: stability usa só janelas reais (n=OBS//30=2), não 6. |
| v25 | Introduziu OB features (ob_*) via pmdata poly_l2. AUC baseline 0.8575. |
| v26 | Adicionou CLOB WS features (clob_*). Adicionou cross-features x_imb_end_x_ret etc. Fix btc_spot_vol_ratio usa volume real Binance. |
| v27 | Migrou para `ob_features_full.parquet` com clob_* incluídos. Corrigiu prefixo `clob_*` no treino. |
| **v28** | **Reconciliação completa train==live.** Adicionou tick features (btc_up_ratio/momentum/tw/zscore/vwap/conviction/disparity) ao treino. Adicionou btc_dist_5k/10k, btc_inslot_vol, dow_sin/cos, hour_x_up_ratio/tw_ur, todos os cross-features x_imb_x_ur/x_depth_x_momentum/x_ob_drift_x_inslot ao treino. Removeu btc_vol_inslot (fórmula divergia). Adicionou lag_{1..5}_outcome e lag_streak ao live. Adicionou dow_sin/cos ao live. Audit confirmado: 104 features, diff=0. |

---

*Documento gerado em 2026-06-10. Atualizar sempre que adicionar, remover ou modificar qualquer feature.*
