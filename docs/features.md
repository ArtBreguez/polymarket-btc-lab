# BTC 5-min Model — Feature Brain
**Versão:** v28 (88f) | **OBS_SECS:** 60 | **SLOT_DURATION:** 300s  
**Última atualização:** 2026-06-10 | **Commit:** 617fb86  
**Auditoria paridade:** train=88 == live=88, diff=0 ✅  
**Auditoria viabilidade:** 75 OK | 7 atenção | 6 divergência (ver §5)

> Fonte de verdade de todas as features. Atualizar sempre que mudar `train_v28_modal.py` ou `live_trader.py`.  
> Rodar auditoria antes de todo treino.

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
slot_ts          +30s     +60s (OBS)   +108s       +168s    +170s        +240s    +300s
   |               |          |           |           |         |            |        |
   |<-- ticks W0 ->|<-- W1 -->|           |           |         |            |        |
   |<-- spot inslot ---------->|           |           |         |            |        |
   |<-- CLOB WS acumulando --------------------------------->|   |            |        |
   |                           |<- lag data-api ~90s->|         |            |        |
   |                           |           clob_* janela         |            |        |
   |                           |           ob_pc_* janela        |            |        |
   |                                                             |<- entry -->|        |
   |                                                             | OB REST /book        |
   |                                                             | predict + order      |
                                                                                  fim slot
```

- **t=0–30s**: `btc_up_w0`; **t=30–60s**: `btc_up_w1`
- **t=60s** (`obs_end = slot_ts + 60`): referência de spot e ticks
- **t=108–168s**: janela `clob_*` (60s ending at CUTOFF_SEC=168)
- **t=0–168s**: janela `ob_pc_*` e `ob_imb_w1/w2`
- **t=170s**: ticks disponíveis via data-api; OB REST chamado; predição
- **t=300s**: slot fecha; resultado resolvido; ring buffer atualizado

---

## 2. Fontes de dados

| Fonte | Lag live | Cobertura treino | Arquivo |
|---|---|---|---|
| Binance WS `btcusdt@kline_1m` | <2s | `binance_spot_full.parquet` | 1-min OHLCV, buffer 300 candles |
| Polymarket data-api `/trades` | ~90-120s | `ticks_btc_full_clean.parquet` | t_sec=0-60 por mercado |
| Polymarket CLOB REST `/book` | <5s | `ob_features_full.parquet` | open snap ~t60s + close snap ~t168s |
| Polymarket CLOB WS `book`+`price_change` | 0s | `ob_features_full.parquet` (colunas `ob_imb_w*`, `ob_pc_*`, `clob_*`) | slot-anchored |
| Ring buffer de slots anteriores | 0s | `all_markets.csv` + tick aggregates | últimos 20 slots |

---

## 3. Tabela canônica — 88 features (train == live)

Legenda: ✅ idêntico | ⚠️ divergência aceita | ❌ divergência real (ver §5.3)

### Grupo A — Spot Binance (13 features)

| Feature | Fórmula | Paridade | Viabilidade |
|---|---|---|---|
| `btc_pre_5m_ret` | `(px[obs_end] / px[slot_ts-300]) - 1` | ✅ | ✅ OK |
| `btc_pre_15m_ret` | `(px[obs_end] / px[slot_ts-900]) - 1` | ✅ | ✅ OK |
| `btc_pre_30m_ret` | `(px[obs_end] / px[slot_ts-1800]) - 1` | ✅ | ✅ OK |
| `btc_pre_1h_ret` | `(px[obs_end] / px[slot_ts-3600]) - 1` | ✅ | ✅ OK |
| `btc_pre_4h_ret` | `(px[obs_end] / px[slot_ts-14400]) - 1` | ✅ | ⚠️ warm-up 4h |
| `btc_inslot_ret` | `(px[-1]/px[0])-1` em `[slot_ts, obs_end]` | ✅ | ✅ OK |
| `btc_inslot_vol` | `std(px)/mean(px)` em `[slot_ts, obs_end]` | ✅ | ✅ OK |
| `btc_inslot_range` | `(max_hi - min_lo) / px_now` | ✅ | ✅ OK |
| `btc_vol_1h` | `std(1-min returns)` em `[slot_ts-3600, slot_ts]` | ✅ | ✅ OK |
| `btc_vol_4h` | `std(1-min returns)` em `[slot_ts-14400, slot_ts]` | ✅ | ⚠️ warm-up 4h |
| `btc_pre_1h_4h_ratio` | `(px_now - px_1h) / (px_now - px_4h + 1e-9)` | ✅ | ⚠️ warm-up 4h |
| `btc_dist_1k` | `min(frac(px/1k), 1-frac(px/1k))` | ✅ | ✅ OK |
| `btc_spot_vol_ratio` | `vol_5m / (vol_55m/11)` | ✅ | ✅ OK |

### Grupo B — L2 Orderbook REST (13 features)

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
| `ob_mid_drift` | `mid_close - mid_open` | Temporal | ✅ | ⚠️ close t~168s treino / t~240s live |
| `ob_imbalance_end` | `imbalance` do close snapshot | Temporal | ✅ | ✅ OK |
| `ob_spread_end` | `spread` do close snapshot | Temporal | ✅ | ✅ OK |
| `ob_depth_change` | `depth_close - depth_open` | Temporal | ✅ | ✅ OK |
| `ob_imb_momentum` | `imb_close - imb_open` | Temporal | ✅ | ✅ OK |

### Grupo B — OB WS (4 features)

| Feature | Janela | Paridade | Viabilidade |
|---|---|---|---|
| `ob_imb_w1` | `mean(imb)` em `[60,120s)` via WS real books | ✅ | ❌ Quase sempre 0.0 live vs poly_l2 real |
| `ob_imb_w2` | `mean(imb)` em `[120,168s)` via WS real books | ✅ | ❌ idem |
| `ob_pc_up_ratio` | `n_up_moves / n_total` em `[0,168s)` | ✅ | ⚠️ reset em reconexão WS |
| `ob_pc_volatility` | `std(price_diffs)` em `[0,168s)` | ✅ | ⚠️ reset em reconexão WS |

### Grupo B — CLOB WebSocket (10 features)

| Feature | Janela | Paridade | Viabilidade |
|---|---|---|---|
| `clob_spread_mean` | `[108,168s)` | ✅ | ✅ OK (synthetic books de best_bid/ask) |
| `clob_spread_trend` | idem | ✅ | ✅ OK |
| `clob_mid_velocity` | idem | ✅ | ✅ OK |
| `clob_mid_volatility` | idem | ✅ | ✅ OK |
| `clob_ask_pressure` | idem | ✅ | ✅ OK |
| `clob_activity_rate` | idem | ✅ | ⚠️ 1 real snap inicial levemente diferente |
| `clob_imb_mean` | idem | ✅ | ❌ Quase sempre 0.0 live vs poly_l2 real |
| `clob_imb_std` | idem | ✅ | ❌ idem |
| `clob_imb_drift` | idem | ✅ | ❌ idem |
| `clob_depth_trend` | idem | ✅ | ❌ idem |

### Grupo C — Tick-based Order Flow (9 features)

| Feature | Fórmula | Paridade | Viabilidade |
|---|---|---|---|
| `btc_up_ratio` | `vol_up / (vol_up+vol_dn)` | ✅ | ✅ OK |
| `btc_n_ticks` | `len(ticks)` | ✅ | ✅ OK |
| `btc_buy_ratio` | `Σsz(side=BUY) / total` | ✅ | ✅ OK |
| `btc_momentum` | `btc_up_w1 - btc_up_w0` | ✅ | ✅ OK |
| `btc_size_disparity` | `avg_up_size - avg_dn_size` | ✅ | ✅ OK |
| `btc_up_ratio_stability` | `std(btc_up_w0, btc_up_w1)` | ✅ | ✅ OK |
| `btc_tw_up_ratio` | `Σ(up·sz·e^(-0.02·(60-t))) / Σ(sz·e^(-0.02·(60-t)))` | ✅ | ✅ OK |
| `btc_up_w0` | `up_ratio` em `[0, 30s)` | ✅ | ✅ OK |
| `btc_up_w1` | `up_ratio` em `[30, 60s)` | ✅ | ✅ OK |

### Grupo D — Lag History (25 features)

| Feature | Paridade | Viabilidade |
|---|---|---|
| `lag_{1..5}_outcome` | ✅ | ✅ OK |
| `prev_slot_up_ratio_{1..5}` | ✅ | ✅ OK |
| `prev_slot_n_ticks_{1..5}` | ✅ | ✅ OK |
| `prev_slot_vol_{1..5}` | ✅ | ✅ OK |
| `lag_streak` | ✅ | ✅ OK |
| `lag_ur_zscore_20` | ✅ | ✅ OK |
| `lag_ur_zscore_5` | ✅ | ✅ OK |
| `btc_up_ratio_zscore_20s` | ✅ | ✅ OK |
| `btc_up_ratio_zscore_5s` | ✅ | ✅ OK |

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
| `x_spread_x_vol` | `ob_spread × btc_vol_1h` | ✅ | ✅ OK |
| `x_depth_x_vol` | `ob_depth_ratio × btc_vol_1h` | ✅ | ✅ OK |
| `x_imb_x_ur` | `ob_imbalance × btc_up_ratio` | ✅ | ✅ OK |
| `x_depth_x_momentum` | `ob_depth_ratio × btc_momentum` | ✅ | ✅ OK |
| `x_ob_drift_x_inslot` | `ob_mid_drift × btc_inslot_ret` | ✅ | ✅ OK |

---

## 4. Auditoria de lookahead

O bot decide em `t ∈ [170, 240]s`. Dado com `t > 60s` do slot atual = lookahead.

| Feature | Referência temporal | Lookahead? |
|---|---|---|
| `btc_pre_*_ret` | `obs_end = slot_ts + 60s` | ✅ NÃO — 60s < 170s |
| `btc_inslot_*` | `[slot_ts, slot_ts+60s]` | ✅ NÃO |
| `btc_vol_1h/4h` | `[slot_ts-3600/14400, slot_ts]` | ✅ NÃO |
| `btc_spot_vol_ratio` | `[slot_ts-3600, slot_ts]` | ✅ NÃO |
| `btc_dist_1k` | `px[slot_ts+60s]` | ✅ NÃO |
| `ob_mid/spread/imbalance` | open snapshot ~t60s | ✅ NÃO |
| `ob_mid_drift` | `mid[~168s] - mid[~60s]` | ✅ NÃO — 168s < 170s |
| `ob_imb_w1/w2` | `[60,120s)` e `[120,168s)` | ✅ NÃO — max 168s |
| `ob_pc_*` | `[0, 168s)` | ✅ NÃO |
| `clob_*` | `[108, 168s)` | ✅ NÃO |
| ticks (C) | `t ∈ [0, 60s)` | ✅ NÃO — chegam via data-api em t~170s |
| `lag_*`, `prev_slot_*` | slots ANTERIORES | ✅ NÃO |
| `hour_*`, `dow_*` | `slot_ts` | ✅ NÃO |
| `x_*` cross | composição de A+B+C | ✅ NÃO |

**ZERO lookahead em todas as 88 features ✅**

---

## 5. Análise de viabilidade live

**Auditoria de fontes executada em 2026-06-10 — 88 features:**

| Veredito | Qtd | Significado |
|---|---|---|
| ✅ OK | 75 | Dado disponível, campos presentes, sem problema real |
| ⚠️ ATENÇÃO | 7 | Disponível mas com caveat operacional; valor ainda informativo |
| ❌ DIVERGÊNCIA | 6 | Dado live estruturalmente diferente do treino (distribuição diferente) |

### 5.1 ✅ OK (75 features)

**A — Spot (10):** `btc_pre_5m/15m/30m/1h_ret`, `btc_inslot_ret/vol/range`, `btc_vol_1h`, `btc_dist_1k`, `btc_spot_vol_ratio`  
Fonte: Binance WS `k['c']` (close), `k['h']`/`k['l']` (high/low), `k['v']` (volume). Todos os campos confirmados no handler `_spot_on_message`.

**B — OB REST (12):** todas exceto `ob_mid_drift`  
Fonte: CLOB REST `/book` → `asks:[{price,size}]`, `bids:[{price,size}]`. Full book retornado sempre. ✓

**B — CLOB WS (5):** `clob_spread_mean/trend`, `clob_mid_velocity/volatility`, `clob_ask_pressure`  
Fonte: `price_change.best_bid` + `price_change.best_ask` → synthetic BookSnapshot com mid/spread reais. Polymarket envia esses campos em cada `price_change`. ✓

**C — Ticks (9):** todas  
Fonte: data-api `/trades` → `outcome`, `side`, `price`, `size`, `timestamp`. Filtro por `slug=btc-updown-5m-{slot_ts}`. ✓

**D — Lag (25):** todas  
Fonte: `_slot_history` ring buffer. `_seed_slot_history()` popula via REST no boot. ✓

**E — Temporal (6):** todas. Determinístico a partir de `slot_ts`. ✓

**F — Cross (8):** todas. `feat.get(k, default)` — fallback gracioso. ✓

### 5.2 ⚠️ ATENÇÃO (7 features) — manter, monitorar

| Feature | Caveat | Mitigação |
|---|---|---|
| `btc_pre_4h_ret` | Precisa 241 candles (4h). Se bot offline >1min → gap → 0.0 | `_seed_spot_buffers()` REST `limit=300` no boot ✓ |
| `btc_vol_4h` | Precisa 240 candles | idem |
| `btc_pre_1h_4h_ratio` | Depende de `px_4h` | idem |
| `ob_mid_drift` | open_snap ~t60s REST vs close_snap ~t240s live (treino: ~t168s). Gap ~72s | Sinal ainda informativo e correlacionado. Aceito. |
| `ob_pc_up_ratio` | Reset WS em reconexão perde eventos do slot → fallback 0.5 | Fallback neutro. Reconexão rara em produção. |
| `ob_pc_volatility` | idem → fallback 0.0 | idem |
| `clob_activity_rate` | 1 real snap inicial → `(1 + n_pcs) / 60` ligeiramente diferente do treino | Sinal correlacionado. Aceito. |

### 5.3 ❌ DIVERGÊNCIA (6 features) — problema estrutural

**Causa raiz:** Polymarket CLOB WS envia **exatamente 1 full book snapshot** ao subscribir (t≈0). Depois envia apenas `price_change` (sem full book). Features que dependem de múltiplos full book snaps dentro de janelas específicas recebem quase sempre **0.0** no live, enquanto no treino (poly_l2) existem ~4 snaps/min.

| Feature | Janela | Treino (poly_l2) | Live | Impacto |
|---|---|---|---|---|
| `ob_imb_w1` | `[60,120s)` | ~4 snaps reais | 0 real snaps → fallback interpolado REST | Distribuição completamente diferente |
| `ob_imb_w2` | `[120,168s)` | ~4 snaps reais | 0 real snaps → fallback = close_snap.imb | 1 ponto vs média N |
| `clob_imb_mean` | `[108,168s)` | N snaps com imb real | Quase sempre 0.0 | Treino: N(μ≠0,σ>0); Live: ≈0.0 |
| `clob_imb_std` | `[108,168s)` | std de N snaps | Quase sempre 0.0 | idem |
| `clob_imb_drift` | `[108,168s)` | imb[-1]−imb[0] | Quase sempre 0.0 | idem |
| `clob_depth_trend` | `[108,168s)` | slope de N depth snaps | Quase sempre 0.0 | idem |

**Opções de resolução:**

| Opção | Ação | Custo | Status |
|---|---|---|---|
| **A** | **Remover as 6 do pool** (→ 82 candidatas) | Baixo | Pendente decisão |
| **B** | Poll REST adicional em t=60s, 120s, 168s para forçar real book snaps no accumulator | Médio | Resolve todas as 6 |
| C | Aceitar 0.0 como "neutro estrutural" (modelo já viu muitos 0.0 no treino em mercados pouco ativos) | Zero | Opção de fallback |

### 5.4 Features removidas do pool (histórico)

16 features removidas em 2026-06-10:

| Feature | Motivo |
|---|---|
| `btc_up_w2/3/4/5` | DEAD — janelas `[60..180s)` sempre fora de OBS=60s |
| `btc_vol_up`, `btc_vol_dn` | Volume absoluto USDC não-normalizado |
| `btc_vwap_up`, `btc_vwap_dn` | VWAP extremamente ruidoso com n<5 ticks |
| `btc_vwap_spread` | Combinação linear de `vwap_up − vwap_dn` |
| `btc_signal_conviction` | Produto de dois features já no set |
| `btc_up_w5_zscore` | Zscore de constante 0.5 |
| `btc_dist_5k`, `btc_dist_10k` | Quase-constante para BTC ~$95k |
| `ob_imb_w0` | Janela [0,60s) — 1 real snap do WS, quase sempre 0.0 |
| `ob_pc_count` | Reset em reconexão = coverage sistematicamente menor |
| `ob_fill_imbalance` | Campo `size` ausente em muitos eventos WS `price_change` |

### 5.5 Cobertura live por grupo

| Grupo | N | Cobertura | Observação |
|---|---|---|---|
| A — Spot | 13 | ~100% | WS + REST fallback |
| B — OB REST | 13 | ~95% | Falha em books vazios/one-sided |
| B — OB WS imb | 2 | ~60% real; ~95% fallback | Fallback interpolado REST |
| B — OB PC | 2 | ~90% | Reset em reconexão → fallback neutro |
| B — CLOB WS (OK) | 5 | ~80% | Zeros se subscrito após t=108s |
| B — CLOB WS (div) | 4 | ~0% real | Quase sempre 0.0 live |
| C — Ticks | 9 | ~90% | data-api lag ~90-120s |
| D — Lag | 25 | ~95% | `_seed_slot_history()` no boot |
| E — Temporal | 6 | ~100% | Determinístico |
| F — Cross | 8 | ~95% | Fallback gracioso |

---

## 6. Grupo A — Spot Binance

**Treino:** `scripts/train_v28_modal.py` ~l316  
**Live:** `deploy/live_trader.py` → `build_spot_features()`  
**Buffer:** `deque(maxlen=300)` → 5h de candles. Seed: REST `limit=300` no boot. Fallback REST poll a cada 30s.  
**Referência:** `obs_end = slot_ts + OBS_SECS(=60)`; `px_now = px[obs_end]`

```python
btc_pre_5m_ret  = (px[obs_end] / px[slot_ts - 300])   - 1   # ~7 candles lookback
btc_pre_15m_ret = (px[obs_end] / px[slot_ts - 900])   - 1   # ~16 candles
btc_pre_30m_ret = (px[obs_end] / px[slot_ts - 1800])  - 1   # ~31 candles
btc_pre_1h_ret  = (px[obs_end] / px[slot_ts - 3600])  - 1   # ~61 candles
btc_pre_4h_ret  = (px[obs_end] / px[slot_ts - 14400]) - 1   # ~241 candles ⚠️

btc_inslot_ret   = (px[-1] / px[0]) - 1        # candles [slot_ts, obs_end]
btc_inslot_vol   = std(px) / mean(px)           # idem
btc_inslot_range = (max_hi - min_lo) / px_now  # hi/lo de cada vela (c[2], c[3])

btc_vol_1h = std(diff(px)/px[:-1])  # [slot_ts-3600, slot_ts]
btc_vol_4h = std(diff(px)/px[:-1])  # [slot_ts-14400, slot_ts] ⚠️

btc_pre_1h_4h_ratio = (px_now - px_1h) / (px_now - px_4h + 1e-9)  # ⚠️

btc_dist_1k = min(frac(px/1000), 1 - frac(px/1000))

btc_spot_vol_ratio = vol[slot_ts-300:slot_ts] / (vol[slot_ts-3600:slot_ts-300]/11)
# vol = candle volume (c[4], campo k['v'] do WS kline)
```

---

## 7. Grupo B — L2 Orderbook

**Fetch treino:** `scripts/fetch_ob_features_modal.py` (CUTOFF_SEC=168)  
**Live REST:** `deploy/live_trader.py` → `_build_ob_features()` — 2 chamadas por slot  
**Live WS:** `ClobFeatureAccumulator` — `get_windowed_imbalance()` + `get_ob_pc_features()` + `get_features()`

### Sequência de snapshots

```
slot_ts      ~t60s REST (open)         t=168s CUTOFF       ~t170-240s REST (close)
   |                 |                      |                         |
   | ob_mid          |                      |                         |
   | ob_spread       |                      | ob_mid_drift ⚠️          |
   | ob_imbalance    |                      | ob_imbalance_end         |
   |                 |                      | ob_spread_end            |
   |                 |                      | ob_depth_change          |
   |                 |                      | ob_imb_momentum          |
   |                 |<-- WS real books ---->| ob_imb_w1/w2 ❌         |
   |<--- WS price_change events ----------->| ob_pc_up_ratio ⚠️        |
   |                    [108,168s) clob_*  ->| clob_* (5 OK + 4 ❌)    |
```

### Fórmulas

```python
# Open snapshot (REST ~t60s)
ob_mid         = (best_ask + best_bid) / 2
ob_spread      = best_ask - best_bid
ob_imbalance   = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-8)

# Close snapshot (REST ~t168s treino / ~t240s live)
ob_depth_ratio  = bid_depth_5c / (ask_depth_5c + 1e-8)
ob_bid_depth_5c = Σbid_sz(price ≥ mid-0.05) / Σbid_sz_total
ob_ask_depth_5c = Σask_sz(price ≤ mid+0.05) / Σask_sz_total
ob_total_depth  = Σbid_sz + Σask_sz
ob_weighted_imb = (Σbid_sz·e^(-10|bp-mid|) - Σask_sz·e^(-10|ap-mid|)) / Σwt

# Temporal (open → close)
ob_mid_drift     = mid_close  - mid_open      # ⚠️ timing gap ~72s
ob_imbalance_end = imb_close
ob_spread_end    = spread_close
ob_depth_change  = depth_close - depth_open
ob_imb_momentum  = imb_close  - imb_open

# Windowed imbalance (WS real book snaps)  ❌ divergência
ob_imb_w1 = mean(imb, t ∈ [60,  120s))  # ~0.0 live (sem real snaps na janela)
ob_imb_w2 = mean(imb, t ∈ [120, 168s))  # fallback = close_snap.imb

# Price-change aggregates (WS price_change events, cutoff=168s)  ⚠️ atenção
ob_pc_up_ratio  = n_up_moves / n_total_moves
ob_pc_volatility = std(price_diffs)

# CLOB WS microstructure (obs_secs=168, window=60 → [slot_ts+108, slot_ts+168))
clob_spread_mean   = mean(all_spreads)          # ✅ synthetic books
clob_spread_trend  = linslope(t, spreads)       # ✅
clob_mid_velocity  = linslope(t, mids)          # ✅
clob_mid_volatility = std(diff(mids))           # ✅
clob_ask_pressure  = frac(SELL moves < 0)       # ✅
clob_activity_rate = (n_real_books+n_pcs)/Δt    # ⚠️ levemente diferente
clob_imb_mean      = mean(real_imb)             # ❌ ~0.0 live
clob_imb_std       = std(real_imb)              # ❌ ~0.0 live
clob_imb_drift     = imb[-1] - imb[0]          # ❌ ~0.0 live
clob_depth_trend   = linslope(t, real_depths)   # ❌ ~0.0 live
```

> **Por que clob_spread/mid são OK mas clob_imb não?**  
> `spread` e `mid` vêm de `best_bid`/`best_ask` que chegam em cada `price_change` → synthetic BookSnapshot.  
> `imbalance` e `depth` precisam do full book (todos os levels) → só disponível em real `book` events → 1x ao subscribir → janela [108,168s) fica sem dados.

---

## 8. Grupo C — Tick-based Order Flow

**Janela:** `t ∈ [0, OBS_SECS=60s)`. Fonte: data-api `/trades`, lag ~90-120s, disponível em t~170s.  
**Campos usados:** `outcome` (Up/Down), `side` (BUY/SELL), `price` (float), `size` (shares), `timestamp` (unix s ou ms).  
**Filtro:** `slug = btc-updown-5m-{slot_ts}` — evita contaminação de outros mercados com mesmo token.

```python
vol_up = Σ(price × size)  onde outcome="Up"      # size_usdc = price × size
vol_dn = Σ(price × size)  onde outcome="Down"
total  = vol_up + vol_dn + 1e-8

btc_up_ratio  = vol_up / total
btc_n_ticks   = len(ticks)
btc_buy_ratio = Σsz_usdc(side="BUY") / total

btc_up_w0 = up_ratio em [0,  30s)
btc_up_w1 = up_ratio em [30, 60s)
btc_momentum         = btc_up_w1 - btc_up_w0
btc_up_ratio_stability = std(btc_up_w0, btc_up_w1)
btc_size_disparity   = avg_up_size - avg_dn_size

λ = 0.02
btc_tw_up_ratio = Σ(up_i·sz_i·e^(-λ·(60-t_i))) / Σ(sz_i·e^(-λ·(60-t_i)))
```

---

## 9. Grupo D — Lag History

**Ring buffer** (`_slot_history`): max 20 entradas, campos `{slot_ts, up_ratio, target, n_ticks, vol_total}`.  
**Seed no boot:** `_seed_slot_history()` busca últimos 20 slots resolvidos via data-api.  
Staleness guard: `time_gap > lag × 300 × 3` → fill neutro.

```python
for lag in [1..5]:
    lag_{lag}_outcome        = target[rank-lag]      (ou 0.5 se stale)
    prev_slot_up_ratio_{lag} = up_ratio[rank-lag]    (ou 0.5)
    prev_slot_n_ticks_{lag}  = n_ticks[rank-lag]     (ou 0.0)
    prev_slot_vol_{lag}      = vol_total[rank-lag]   (ou 0.0)

lag_streak = N outcomes consecutivos na mesma direção (max 5)

# Z-scores: usa prev_slot_up_ratio_1 como "current"
lag_ur_zscore_20 = clip((prev_ur_1 - mean(hist_20)) / max(std(hist_20), 0.01), -5, 5)
lag_ur_zscore_5  = clip((prev_ur_1 - mean(hist_5))  / max(std(hist_5),  0.01), -5, 5)

# Z-scores: usa btc_up_ratio atual como "current"
btc_up_ratio_zscore_20s = clip((cur_ur - mean(hist_20)) / max(std(hist_20), 0.01), -5, 5)
btc_up_ratio_zscore_5s  = clip((cur_ur - mean(hist_5))  / max(std(hist_5),  0.01), -5, 5)
```

---

## 10. Grupo E — Temporal

```python
dt   = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
hour = dt.hour + dt.minute / 60.0    # [0, 24)
dow  = dt.weekday()                   # 0=seg .. 6=dom

hour_sin = sin(2π · hour / 24)
hour_cos = cos(2π · hour / 24)
dow_sin  = sin(2π · dow / 7)
dow_cos  = cos(2π · dow / 7)
hour_x_up_ratio = btc_up_ratio    × (hour / 24.0)
hour_x_tw_ur    = btc_tw_up_ratio × (hour / 24.0)
```

---

## 11. Grupo F — Cross-domain Interactions

Computadas após A+B+C via `feat.get(k, default)`. Falham graciosamente se OB ausente.

```python
x_imb_x_inslot      = ob_imbalance     × btc_inslot_ret
x_imb_end_x_ret     = ob_imbalance_end × btc_inslot_ret
x_drift_x_ret5m     = ob_mid_drift     × btc_pre_5m_ret
x_spread_x_vol      = ob_spread        × btc_vol_1h       # default ob_spread=0.02
x_depth_x_vol       = ob_depth_ratio   × btc_vol_1h       # default 1.0
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
ob_imbalance=ob_mid_drift=ob_imb_momentum=ob_depth_change=0.0

# ob_pc_* (WS sem eventos ou reconexão)
ob_pc_up_ratio=0.5, ob_pc_volatility=0.0

# ob_imb_w1/w2 (sem real book snaps — divergência §5.3)
ob_imb_w1 = (open_snap.imb + close_snap.imb) / 2  # fallback interpolado
ob_imb_w2 = close_snap.imb

# clob_* (subscrito após t=108s ou sem real snaps)
todos clob_* = 0.0

# Ticks ausentes (data-api falha ou slot sem ticks)
btc_up_ratio=0.5, btc_n_ticks=0.0, btc_buy_ratio=0.5
btc_momentum=0.0, btc_size_disparity=0.0, btc_up_ratio_stability=0.0
btc_tw_up_ratio=0.5, btc_up_w0=0.5, btc_up_w1=0.5

# Lag history (vazio, stale ou cold start)
lag_{1..5}_outcome=0.5, prev_slot_up_ratio_{1..5}=0.5
prev_slot_n_ticks_{1..5}=0.0, prev_slot_vol_{1..5}=0.0
lag_streak=0.0, lag_ur_zscore_*=0.0, btc_up_ratio_zscore_*=0.0
```

---

## 13. Seleção de features no treino

**Pool atual: 88 candidatas** (após remoção das 16 inviáveis).

1. **Screening**: LightGBM `n_estimators=300, max_depth=4`, 5-fold `TimeSeriesSplit`
2. **Ranking**: acumula `feature_importances_` over folds ÷ N_SPLITS
3. **Teste N ∈ {40, 30, 25, 20, 15}**: avalia AUC com modelo completo (400 estimators)
4. **Seleção**: N com melhor AUC médio

> O modelo aprende somente o `top_features` list. O live computa todas as 88 mas filtra por `features = model_data["features"]` antes de `predict_proba`.

---

## 14. Checklist pré-treino

- [ ] `fetch_ob_features_modal.py` finalizou — todos os ~23.221 mercados
- [ ] `ob_features_full.parquet` uploaded para HF em `data/ob_features_full.parquet`
- [ ] `all_markets.csv` + `ticks_btc_full_clean.parquet` atualizados no HF
- [ ] `binance_spot_full.parquet` cobre todo o período dos markets
- [ ] `OBS_SECS` treino == `OBSERVE_SECS` live (ambos = 60)
- [ ] `SLOT_DURATION` treino == live (ambos = 300)
- [ ] Decidir sobre as 6 features com ❌ DIVERGÊNCIA (§5.3): remover ou aceitar

---

## 15. Changelog de features por versão

| Versão | N | Mudanças |
|---|---|---|
| v21 | ~20 | Real-time only. OBS=180s criava lag nos ticks. |
| v22-v23 | ~25 | `btc_up_ratio_stability`, `hour_sin/cos`. OBS→60s. |
| v24 | ~25 | Fix stability: só 2 janelas reais. |
| v25 | ~35 | OB features via poly_l2. AUC 0.8575. |
| v26 | ~45 | CLOB WS features. Cross-features. Fix btc_spot_vol_ratio volume. |
| v27 | ~50 | `ob_features_full.parquet` com clob_*. Fix prefixo. |
| v28 | 104 | Full train==live parity. Tick features no treino. Todas divergências eliminadas. |
| **v28 (88f)** | **88** | Remoção de 16 inviáveis: DEAD w2-5, redundantes vwap/vol/conviction, quase-constantes dist_5k/10k, esparsos ob_imb_w0/pc_count/fill_imb. btc_momentum simplificado. |
| v29 (plan) | ~82-88 | Decidir sobre 6 features ❌ (§5.3). Auto feature selection sobre pool limpo. |

---

*Atualizar sempre que modificar features. Auditoria: `train=88==live=88 diff=0 ✅` (2026-06-10)*
