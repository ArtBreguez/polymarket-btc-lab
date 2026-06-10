# BTC 5-min Model — Feature Brain
**Versão:** v29_20f_rt | **OBS_SECS:** 60 | **SLOT_DURATION:** 300s
**Última atualização:** 2026-06-10 | Auditoria paridade: train == live ✅
**Features ativas:** 20 (seleção automática por permutation importance, threshold=15% mediana)

> Fonte de verdade de todas as features. Atualizar sempre que mudar `train_v29_modal.py` ou `live_trader.py`.

---

## Índice
1. [Timeline de um slot](#1-timeline-de-um-slot)
2. [Fontes de dados](#2-fontes-de-dados)
3. [Tabela canônica — 20 features ativas (train == live ✅)](#3-tabela-canônica--20-features-ativas)
4. [Auditoria de lookahead](#4-auditoria-de-lookahead)
5. [Histórico de remoções](#5-histórico-de-remoções)
6. [Grupo A — Spot Binance](#6-grupo-a--spot-binance)
7. [Grupo B — L2 Orderbook](#7-grupo-b--l2-orderbook)
8. [Grupo C — CLOB Real-time](#8-grupo-c--clob-real-time)
9. [Grupo D — Order Flow (ticks)](#9-grupo-d--order-flow-ticks)
10. [Grupo E — Lag History](#10-grupo-e--lag-history)
11. [Grupo F — Cross features](#11-grupo-f--cross-features)

---

## 1. Timeline de um slot

```
slot_ts (t=0)
  │
  ├─ t=0..60s  → OBS_SECS=60: coleta spot/tick/CLOB
  │               ob snapshot "open" capturado em t~5-10s
  │
  ├─ t=60s     → predict_proba() com as 20 features
  │
  └─ t=300s    → resolução (target = BTC subiu ou caiu?)
```

**Regra inviolável:** nenhuma feature pode usar dados após t=60s.

---

## 2. Fontes de dados

| Fonte | Dados | Latência |
|---|---|---|
| Binance WebSocket | BTC/USDT klines + trades | ~real-time |
| Polymarket CLOB WS | book snapshots, price_change | ~real-time |
| Polymarket data-api | ticks históricos do slot | ~120s lag |
| CLOB REST /book | ob snapshot on-demand | ~100-300ms |

---

## 3. Tabela canônica — 20 features ativas

Todas computadas em t<60s. Paridade train==live verificada.

| # | Feature | Grupo | Fonte | Descrição |
|---|---|---|---|---|
| 1 | `btc_inslot_ret` | A | Binance spot | Retorno BTC no slot (px_end/px_start - 1) |
| 2 | `btc_inslot_range` | A | Binance spot | (max-min)/px_end do slot |
| 3 | `btc_pre_5m_ret` | A | Binance spot | Retorno dos 5min anteriores ao slot |
| 4 | `btc_dist_1k` | A | Binance spot | Distância ao múltiplo de $1k mais próximo |
| 5 | `btc_spot_vol_ratio` | A | Binance spot | Vol 5m / média de vol 5m na última 1h |
| 6 | `ob_imbalance` | B | CLOB REST | Imbalance do OB no snapshot "open" (t~5-10s) |
| 7 | `ob_depth_ratio` | B | CLOB REST | bid_depth / ask_depth no snapshot "open" |
| 8 | `ob_total_depth` | B | CLOB REST | Profundidade total do OB no snapshot "open" |
| 9 | `clob_spread_mean` | C | CLOB WS | Spread médio no período t=0..60s |
| 10 | `clob_spread_trend` | C | CLOB WS | Tendência linear do spread no período |
| 11 | `clob_mid_volatility` | C | CLOB WS | Desvio padrão das variações do mid-price |
| 12 | `clob_ask_pressure` | C | CLOB WS | Pressão vendedora (asks dominam o book) |
| 13 | `btc_up_w1` | D | data-api ticks | Up-ratio ponderado da segunda metade do slot |
| 14 | `btc_size_disparity` | D | data-api ticks | Dispersão de tamanhos entre BUY e SELL |
| 15 | `btc_up_ratio_zscore_5s` | D | data-api ticks | Z-score do up-ratio vs janela 5 slots |
| 16 | `prev_slot_up_ratio_3` | E | histórico | Up-ratio do slot t-3 |
| 17 | `prev_slot_up_ratio_5` | E | histórico | Up-ratio do slot t-5 |
| 18 | `lag_ur_zscore_20` | E | histórico | Z-score do up-ratio do slot anterior vs janela 20 |
| 19 | `x_imb_x_ur` | F | cross | ob_imbalance × btc_up_ratio |
| 20 | `x_depth_x_vol` | F | cross | ob_depth_ratio × btc_vol_1h |

---

## 4. Auditoria de lookahead

**Regra:** feature deve usar exclusivamente dados de t < OBS_SECS (60s).

Todas as 20 features acima passaram na auditoria. Ver seção 5 para features removidas por violação.

---

## 5. Histórico de remoções

### v29 — removidas por leakage temporal (2026-06-10)

Estas features eram coletadas em t=108-168s no training (média do slot), mas o live as capturava em t<60s. Correlação artificial com o target inflou AUC de ~0.79 para 0.85.

| Feature | Correlação c/ target | Motivo |
|---|---|---|
| `ob_mid` | 0.60 | OB capturado em t=108-168s no train |
| `ob_imbalance_end` | — | snapshot "end" usa dados do slot completo |
| `ob_spread_end` | — | idem |
| `ob_imb_momentum` | — | end - open = depende do end |
| `ob_depth_change` | — | end - open = depende do end |
| `clob_mid_velocity` | — | slope do mid calculado sobre slot completo no train |
| `ob_weighted_imb` | — | inclui dados pós-60s |
| `ob_bid_depth_5c` | — | snapshot contaminado |
| `ob_ask_depth_5c` | — | snapshot contaminado |

**Efeito:** win rate caiu de 96.8% → baseline honesto. AUC v28=0.848 (inflado) → v29=0.792 (real).

### v26-v28 — features de versões anteriores
Features como `btc_pre_4h_ret`, `btc_vol_1h`, variantes de lag (1-20), `hour_sin/cos`, `dow_sin/cos`, `btc_up_ratio`, etc. foram computadas mas não selecionadas pelo permutation importance no v29. Presentes no código mas não usadas pelo modelo.

---

## 6. Grupo A — Spot Binance

Fonte: WebSocket Binance `btcusdt@kline_1m` + `btcusdt@trade`.

| Feature | Cálculo |
|---|---|
| `btc_inslot_ret` | `px_end / px_start - 1` onde start/end são os ticks dentro do slot |
| `btc_inslot_range` | `(high - low) / px_end` dos ticks do slot |
| `btc_pre_5m_ret` | `px_obs_end / px_5m_ago - 1` |
| `btc_dist_1k` | `min(px % 1000, 1000 - px % 1000) / 1000` |
| `btc_spot_vol_ratio` | `vol_5m / mean(vol_5m_per_slot_last_1h)` |

---

## 7. Grupo B — L2 Orderbook

Fonte: CLOB REST `/book` — snapshot "open" capturado em t~5-10s após inicio do slot.

**Importante:** apenas o snapshot "open" é usado. Snapshots "end" (t>60s) foram removidos por leakage.

| Feature | Cálculo |
|---|---|
| `ob_imbalance` | `bid_vol / (bid_vol + ask_vol)` top-10 levels |
| `ob_depth_ratio` | `bid_depth / ask_depth` top-5 levels |
| `ob_total_depth` | `bid_depth + ask_depth` total |

---

## 8. Grupo C — CLOB Real-time

Fonte: WebSocket CLOB — eventos `book_snapshot` e `price_change` acumulados em t=0..60s.

| Feature | Cálculo |
|---|---|
| `clob_spread_mean` | Média do spread (ask-bid) durante o período |
| `clob_spread_trend` | Slope linear do spread vs tempo |
| `clob_mid_volatility` | `std(diff(mid_prices))` — volatilidade do mid |
| `clob_ask_pressure` | Razão de eventos onde ask domina o volume |

---

## 9. Grupo D — Order Flow (ticks)

Fonte: Polymarket data-api `/trades` — ticks do slot atual com lag ~120s.

| Feature | Cálculo |
|---|---|
| `btc_up_w1` | Up-ratio ponderado por volume na segunda metade do slot |
| `btc_size_disparity` | `abs(mean_buy_size - mean_sell_size) / mean_size` |
| `btc_up_ratio_zscore_5s` | `(up_ratio - mean_5slots) / std_5slots` |

---

## 10. Grupo E — Lag History

Fonte: histórico de slots resolvidos em memória.

| Feature | Cálculo |
|---|---|
| `prev_slot_up_ratio_3` | Up-ratio do slot t-3 |
| `prev_slot_up_ratio_5` | Up-ratio do slot t-5 |
| `lag_ur_zscore_20` | `(up_ratio[t-1] - mean_20) / std_20` |

---

## 11. Grupo F — Cross features

Interações multiplicativas entre grupos.

| Feature | Cálculo |
|---|---|
| `x_imb_x_ur` | `ob_imbalance × btc_up_ratio` |
| `x_depth_x_vol` | `ob_depth_ratio × btc_vol_1h` |
