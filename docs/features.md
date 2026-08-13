# BTC 5-min Model — Feature Brain

**Versão ativa:** v29_20f_rt | **OBS_SECS:** 60 | **SLOT_DURATION:** 300s
**Última atualização:** 2026-06-13 | **WF AUC:** 0.792

> Fonte de verdade de todas as features. Só entram features 100% computáveis em
> tempo real com os dados que chegam nos WebSockets. Cada feature tem o campo
> de dados exato validado ao vivo (13/13 sanity checks passando, 2026-06-13).

---

## 1. Features do modelo ativo (v29 — 20 features)

| # | Feature | Grupo | Fonte |
|---|---------|-------|-------|
| 1 | `btc_inslot_ret` | A | Binance kline (normalizado por vol_1h) |
| 2 | `btc_inslot_range` | A | Binance kline (normalizado por vol_1h) |
| 3 | `btc_pre_5m_ret` | A | Binance kline buffer (normalizado por vol_1h) |
| 4 | `btc_dist_1k` | A | Binance kline (close) |
| 5 | `btc_spot_vol_ratio` | A | Binance kline (volume) |
| 6 | `ob_total_depth` | B | CLOB REST /book (snapshot open t~60s) |
| 7 | `ob_imbalance` | B | CLOB REST /book (snapshot open) |
| 8 | `ob_depth_ratio` | B | CLOB REST /book (snapshot open) |
| 9 | `clob_spread_mean` | C | CLOB WS price_change (janela t=[108,168s) — treino v29) |
| 10 | `clob_spread_trend` | C | CLOB WS price_change (janela t=[108,168s) — treino v29) |
| 11 | `clob_mid_volatility` | C | CLOB WS price_change (janela t=[108,168s) — treino v29) |
| 12 | `clob_ask_pressure` | C | CLOB WS price_change (janela t=[108,168s) — treino v29) |
| 13 | `btc_up_w1` | D | Data-API trades (t=30-60s) |
| 14 | `btc_size_disparity` | D | Data-API trades (avg_up - avg_dn) |
| 15 | `btc_up_ratio_zscore_5s` | D | Ring buffer (zscore 5 slots) |
| 16 | `prev_slot_up_ratio_3` | E | Ring buffer (t-3) |
| 17 | `prev_slot_up_ratio_5` | E | Ring buffer (t-5) |
| 18 | `lag_ur_zscore_20` | E | Ring buffer (zscore 20 slots) |
| 19 | `x_imb_x_ur` | F | ob_imbalance × btc_up_ratio |
| 20 | `x_depth_x_vol` | F | ob_depth_ratio × btc_vol_1h |

---

## 2. Dados reais observados nos WebSockets

### Binance WS (`btcusdt@kline_1m`)
Mensagem: `{stream, data: {k: {...}}}`
Campos do kline:
- `t` — open time (ms)
- `o` — open price
- `c` — close price (atualizado em tempo real)
- `h` — high do candle
- `l` — low do candle
- `v` — volume base (BTC)
- `n` — número de trades no candle
- `V` — taker buy volume (BTC)
- `q` — quote volume (USDT)
- `x` — candle fechado? (bool)

### CLOB WS (`wss://ws-subscriptions-clob.polymarket.com/ws/market`)

**Evento `book`** — chega ~1x por slot (no subscribe)
```json
{
  "event_type": "book",
  "asset_id": "...",
  "bids": [{"price": "0.76", "size": "51543.9"}, ...],
  "asks": [{"price": "0.77", "size": "49798.9"}, ...]
}
```

**Evento `price_change`** — alta frequência (dezenas de milhares por slot)
```json
{
  "event_type": "price_change",
  "price_changes": [{
    "asset_id": "...",
    "price": "0.76",
    "size": "49803.91",
    "side": "BUY",
    "best_bid": "0.76",
    "best_ask": "0.77"
  }]
}
```
> Nota: `side` usa "BUY"/"SELL" (não "BID"/"ASK"). Mapeado internamente para BID/ASK.

### CLOB REST (`/book?token_id=...`)
- Snapshot estático com asks e bids ordenados DESC por preço
- Melhor ask = `min(asks)` (não `asks[0]`, que é o mais caro)
- Usado para open snapshot (t~60s) e hydration do acumulador no reconnect

---

## 3. Timeline de um slot

```
slot_ts (t=0)
  │
  ├─ t=0: reset_token() no acumulador CLOB — limpa buffers do slot anterior
  │        CLOB WS re-subscrito → recebe book snapshot inicial (1 evento real)
  │        Acumulador começa a receber price_change events
  │
  ├─ t=0..168s: acumula price_change via CLOB WS → janela [0, 168s)
  │              Binance kline buffer atualiza continuamente
  │
  ├─ t~60s: OB early snapshot via CLOB REST /book (open_snap)
  │          Usado para ob_imbalance, ob_depth_ratio, ob_total_depth
  │
  ├─ t~170s: ENTER_WINDOW — executa predict_proba() com features de [0, 168s)
  │           Data-API ticks chegam com ~120s lag, disponíveis aqui
  │           CLOB features: janela [0, 168s) (window_secs=168)
  │
  └─ t=300s: resolução do mercado
```

---

## 4. Detalhamento por grupo

### Grupo A — Spot Binance (kline buffer)

> **IMPORTANTE:** `btc_inslot_ret`, `btc_pre_5m_ret`, `btc_inslot_range` são
> **normalizados por `btc_vol_1h`** antes de entrar no modelo.
> `btc_vol_1h = std(returns_last_12_candles)`.
> Isso significa que o DataQualityGate deve usar limites em torno de ±50 (não ±0.05).

| Feature | Fórmula |
|---------|---------|
| `btc_inslot_ret` | `(close_at_obs / open_at_slot - 1) / vol_1h` |
| `btc_inslot_range` | `((high - low) / close) / vol_1h` |
| `btc_pre_5m_ret` | `(close_now / close_5m_ago - 1) / vol_1h` |
| `btc_dist_1k` | `min(px % 1000, 1000 - px % 1000) / 1000` |
| `btc_spot_vol_ratio` | `vol_slot_5m / mean(vol_12_slots_last_1h)` |
| `btc_vol_1h` | `std(returns_last_12_candles)` — usado como normalizador, não feature |

### Grupo B — L2 Orderbook (CLOB REST snapshot, t~60s)

Open snapshot capturado em t~60s (OB early snapshot). Retry 3x com 0.5s backoff.
Se `asks=0` → mercado skewed (ask >= 0.97), não retenta (correto não entrar).

| Feature | Fórmula |
|---------|---------|
| `ob_total_depth` | `sum(all bid_sz) + sum(all ask_sz)` |
| `ob_imbalance` | `bid_sz / (bid_sz + ask_sz)` top-1 level |
| `ob_depth_ratio` | `bid_depth_top5 / ask_depth_top5` |

### Grupo C — CLOB Real-time (price_change WS, janela t=[0,168s))

Accumulator (`ClobFeatureAccumulator`) bufferiza eventos `price_change` e cria
synthetic `BookSnapshot` com `best_bid`/`best_ask` de cada evento.

`get_features(obs_secs=168, window_secs=168)` → janela `[slot_ts, slot_ts+168s)`.

> **Bug histórico (corrigido 2026-06-13):** `window_secs` era 60, cobrindo apenas
> [108s, 168s). Excluía ~64% dos eventos. Corrigido para 168 (slot inteiro).

| Feature | Fórmula |
|---------|---------|
| `clob_spread_mean` | `mean(best_ask - best_bid)` sobre todos os eventos |
| `clob_spread_trend` | slope linear do spread vs tempo |
| `clob_mid_volatility` | `std(diff(mid_sequence))` |
| `clob_ask_pressure` | fração de ASK moves consecutivos que foram DOWN |

### Grupo D — Order Flow (Data-API ticks, t=[0,60s))

Ticks do Polymarket CLOB fetched via data-api com ~120s de lag (disponíveis em t~170s).
`side` = "Up"/"Down" (outcome label).

| Feature | Fórmula |
|---------|---------|
| `btc_up_w1` | `vol_Up / total_vol` na janela t=30-60s |
| `btc_size_disparity` | `mean_size_Up - mean_size_Dn` |
| `btc_up_ratio_zscore_5s` | `(up_ratio - mean_5slots) / std_5slots` |

### Grupo E — Lag History (ring buffer)

Computado de slots anteriores resolvidos. Nunca usa dados futuros.

| Feature | Fonte |
|---------|-------|
| `prev_slot_up_ratio_3` | up_ratio do slot t-3 |
| `prev_slot_up_ratio_5` | up_ratio do slot t-5 |
| `lag_ur_zscore_20` | zscore do up_ratio atual vs janela 20 slots |

### Grupo F — Cross-features

| Feature | Fórmula |
|---------|---------|
| `x_imb_x_ur` | `ob_imbalance × btc_up_ratio` |
| `x_depth_x_vol` | `ob_depth_ratio × btc_vol_1h` |

---

## 5. DataQualityGate — limites corretos

| Check | Limite | Motivo |
|-------|--------|--------|
| `RETURN_RANGE` | `(-50, 50)` | Retornos normalizados por vol_1h; ±0.05 era falso positivo |
| `ASK_RANGE` | `(0.1, 0.95)` | Exclui mercados já decididos |
| `MIN_TICKS` | 50 | Mínimo de ticks para sinal confiável |
| `SPOT_MAX_AGE` | 300s | Buffer Binance não pode ser muito stale |
| `WARMUP_SLOTS` | 3 | Cold start protection |

---

## 6. Features do pipeline completo (71 features candidatas para versões futuras)

Ver histórico completo em `docs/features_full_history.md` (versões v18–v31).

---

## 7. Features REMOVIDAS por impossibilidade live ou mismatch train/live

| Feature | Motivo |
|---------|--------|
| `ob_mid_drift` | Precisa snapshot close (t~168s) — lookahead vs OBS_SECS=60 |
| `ob_imbalance_end` | Idem |
| `ob_spread_end` | Idem |
| `ob_imb_w1`, `ob_imb_w2` | Requer book reais em t=60-168s — WS só entrega 1 por slot |
| `clob_imb_mean/std/drift` | Sempre 0.0 live — apenas 1 book real por slot |
| `clob_depth_trend` | Idem |
| `clob_activity_rate` | Diverge entre train (janela fixa) e live (wall-clock) |
| `clob_mid_velocity` | Computada mas não inclusa no CLOB_KEEP (v29) |
