# 06 — Live Trading

## Strategy

Each BTC 5-minute market on Polymarket has a 300-second slot.

| Phase | Time Window | Action |
|-------|-------------|--------|
| Subscribe | t=0s | CLOB WS subscribe → recebe book snapshot, começa a acumular price_change |
| OB Early Snap | t~60s | CLOB REST /book → open snapshot (ob_imbalance, ob_depth_ratio, ob_total_depth) |
| Observe | 0–168s | Acumula CLOB price_change + Binance kline continuamente |
| Decide | 170–240s | Fetch data-api ticks, build features, predict, check gates |
| Enter | 170–240s | Place order se todos os critérios OK |
| Settle | 300–360s | Wait for resolution, record P&L |

Entry criteria (todos devem ser verdadeiros):
- Model confidence > 55% (`MIN_CONFIDENCE = 0.55`)
- Edge vs ask price >= 7% (`MIN_EDGE = 0.07`)
- Edge vs market mid >= 5% (`MIN_EDGE_MID = 0.05`)
- Ask no range `[0.42, 0.65]` (exclui mercados decididos)
- Auto-sizing shares: 5–40 shares proporcional ao saldo ($20–$700)
- PREDICTION_SANITY gate: rejeita prob > 99% (sinal de dado problemático)

---

## live_trader.py Architecture

```
┌─────────────────────────────────────────┐
│  Binance Spot Daemon (background WS)    │
│  btcusdt@kline_1m → _spot_buffers       │
│  300 candles (5h buffer)                │
└─────────────────────────────────────────┘
          │
┌─────────────────────────────────────────┐
│  CLOB WS Daemon (background)            │
│  price_change → ClobFeatureAccumulator  │
│  book → _clob_prices cache              │
│  Hydration REST no reconnect            │
│  Reconnect automático (WebSocketManager)│
└─────────────────────────────────────────┘
          │
┌─────────────────────────────────────────┐
│  Main Loop (a cada 10s)                 │
│  1. Detect slot change → reset_token()  │
│  2. t~60s: fetch OB early snapshot      │
│  3. t=170-240s: ENTER_WINDOW            │
│     a. fetch_inslot_trades (data-api)   │
│     b. build_features()                 │
│     c. DataQualityGate checks           │
│     d. predict_proba()                  │
│     e. PREDICTION_SANITY gate           │
│     f. check ask price + edge           │
│     g. place_order() se tudo OK         │
└─────────────────────────────────────────┘
```

**Model**: Baixado do HuggingFace (`artbreguez/polymarket-btc-model`) no startup.
**Versão ativa:** v29_20f_rt — 20 features, WF AUC=0.792.
**Slot history**: Ring buffer de outcomes recentes para lag features.

---

## Feature Computation

Ver `docs/features.md` para a referência completa.

Features do modelo v29 (20 features, todas computáveis em RT):

| Grupo | Features | Fonte |
|-------|---------|-------|
| A — Spot | btc_inslot_ret, btc_inslot_range, btc_pre_5m_ret, btc_dist_1k, btc_spot_vol_ratio | Binance WS |
| B — OB | ob_total_depth, ob_imbalance, ob_depth_ratio | CLOB REST /book |
| C — CLOB RT | clob_spread_mean, clob_spread_trend, clob_mid_volatility, clob_ask_pressure | CLOB WS price_change |
| D — Ticks | btc_up_w1, btc_size_disparity, btc_up_ratio_zscore_5s | Data-API |
| E — Lag | prev_slot_up_ratio_3, prev_slot_up_ratio_5, lag_ur_zscore_20 | Ring buffer |
| F — Cross | x_imb_x_ur, x_depth_x_vol | Computed |

**Regra crítica:** Retornos do Grupo A são **normalizados por `btc_vol_1h`**.
O DataQualityGate usa `RETURN_RANGE=(-50, 50)` — não ±0.05 (que era falso positivo).

---

## CLOB Feature Accumulator

`deploy/clob_features.py` — `ClobFeatureAccumulator` singleton.

- `feed_event(token_id, event)` — chamado pelo WS daemon para cada evento
- `reset_token(token_id, slot_ts)` — chamado no slot change (t=0), limpa buffer
- `get_features(token_id, slot_ts, obs_secs=168, window_secs=168)` — janela `[0, 168s)`
- Hydration REST: no reconnect, injeta 1 BookSnapshot via `/book` para cada token

**Armadilha conhecida:** Usar `window_secs < 168` descarta eventos acumulados no início
do slot, pois o acumulador é resetado em t=0. Janela padrão deve ser `168s` (slot inteiro).

---

## WebSocket Management (`WebSocketManager`)

**Binance spot** (`btcusdt@kline_1m`):
- Background thread
- 300 candles buffer (5h)
- Auto-reconnect com exponential backoff
- Zombie detection: 60s sem mensagem = reconecta

**Polymarket CLOB WS** (`wss://ws-subscriptions-clob.polymarket.com/ws/market`):
- Background daemon com `WebSocketManager`
- `_clob_on_connect`: prune tokens stale + re-subscribe + hydration REST
- `_clob_on_message`: atualiza `_clob_prices` + `ClobFeatureAccumulator`
- Zombie detection: 60s sem mensagem = reconecta
- Reconnect hydra o acumulador via REST `/book` para dados imediatos

---

## DataQualityGate

Gates executados em sequência antes da predição:

1. **Cold start** — warmup 3 slots após boot
2. **Data completeness** — min 50 ticks, spot buffer age < 300s
3. **Feature sanity** — `RETURN_RANGE=(-50, 50)` (normalizados por vol_1h)
4. **Prediction sanity** — `prob_up < 0.99` (>99% = dado problemático)
5. **Ask range** — ask em `[0.42, 0.65]`
6. **Edge** — `edge_ask >= 7%` e `edge_mid >= 5%`

---

## Order Placement

Usa Polymarket CLOB API via `py-clob-client`:
- **Endpoint**: `https://clob.polymarket.com`
- **Auth**: Builder API credentials (key, secret, passphrase)
- **Wallet**: Proxy wallet (`POLY_SAFE_ADDRESS`), `signature_type=1` (POLY_GNOSIS_SAFE)
- **Order type**: Market buy no token UP
- **Auto-sizing**: `AUTO_SHARES=true`, min=5, max=40, floor=$20, ceil=$700

---

## P&L Tracking

Trades logados em `/tmp/live_trades.json`.
Histórico reconstruído do chain no boot via `get_trades()`.

---

## Deploy

```bash
# Deploy (usa deploy3.py com token completo)
python3 /tmp/deploy3.py

# Start/stop machine
python3 /tmp/start_machine.py   # start
python3 /tmp/stop_machine.py    # stop (criar se necessário)
```

**App Fly.io:** `polymarket-maker-mm` | **Machine:** `5683e451b695e8` | **Region:** AMS

---

## Monitoring

```bash
# Logs em tempo real (via deploy3.py token)
python3 - <<'EOF'
import subprocess, os
with open('/home/ubuntu/.env') as f:
    for line in f:
        if 'FLY_ARTBREGUEZ_UCL_TOKEN' in line:
            tok = line.strip().split('=', 1)[1]; break
env = {**os.environ, 'PATH': os.environ['PATH'] + ':/home/ubuntu/.fly/bin', 'FLY_API_TOKEN': tok}
subprocess.run(['flyctl', 'logs', '-a', 'polymarket-maker-mm', '--no-tail'], env=env)
EOF
```

Key log lines:
- `build_features OK — 20/20 features non-zero` ✅
- `CLOB features: X/5 non-zero` — deve ser 4-5/5
- `CLOB features: no data yet (accumulator empty)` ⚠️ — acumulador vazio
- `Prediction: UP conf=XX.X%` — sinal do modelo
- `PREDICTION_SANITY: extreme probability` ⚠️ — features ruins (CLOB zeradas)
- `Skip — ask $0.000 outside [0.42, 0.65]` — mercado decidido (normal)
- `ENTER — UP @ $X.XX` — trade executado

---

## Common Issues

| Issue | Sintoma | Fix |
|-------|---------|-----|
| CLOB features always zero | `accumulator empty` em todo slot | Verificar se `window_secs=168` no `get_features()` |
| Prediction > 99% | `PREDICTION_SANITY: extreme probability` | Causado por CLOB features zeradas → fix do acumulador |
| DataQualityGate bloqueando | `RETURN_RANGE` reject | `RETURN_RANGE=(-50, 50)` — retornos normalizados por vol_1h |
| Market decided | `all asks >= 0.97` | Normal — mercado já resolvido antes do entry window |
| CLOB WS disconnects | `Connection closed: code=1006` | Auto-reconnect OK; hydration REST repopula acumulador |
| OB asks=0 (skewed) | `ask=0.97` ou superior | Não retenta (correto). Retry só para falha de rede. |
| Fly.io token auth | `missing third-party discharge tokens` | Usar ambas as partes do token (vírgula-separated) |
