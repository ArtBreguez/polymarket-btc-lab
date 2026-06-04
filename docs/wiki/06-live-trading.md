# 06 — Live Trading

## Strategy

Each BTC 5-minute market on Polymarket has a 300-second slot.

| Phase | Time Window | Action |
|-------|-------------|--------|
| Observe | 0–180s | Collect CLOB trades + Binance spot data, build features |
| Decide | 170–240s | Compute prediction, check entry criteria |
| Enter | 170–240s | Place order if criteria met |
| Settle | 300–360s | Wait for resolution, record P&L |

Entry criteria (all must be true):
- Model confidence > 60% (`MIN_CONFIDENCE = 0.60`)
- Edge vs ask price >= 10% (`MIN_EDGE = 0.10`)
- Edge vs market mid >= 5% (`MIN_EDGE_MID = 0.05`)
- Flat stake: $1.50 USDC per trade

---

## live_trader.py Architecture

```
┌─────────────────────────────────────────┐
│  Spot Daemon (background thread)        │
│  Binance WS → _spot_buffers (deque)     │
│  btcusdt: 300 candles (5h buffer)       │
│  Writes /tmp/spot_buffer.json           │
└─────────────────────────────────────────┘
          │
┌─────────────────────────────────────────┐
│  Main Loop                              │
│  1. Discover active BTC 5-min markets   │
│  2. Wait for observation window         │
│  3. Fetch inslot trades (data-api)      │
│  4. Build features (build_features())   │
│  5. Model prediction                    │
│  6. Check entry criteria                │
│  7. Place order via CLOB API            │
│  8. Track P&L                           │
└─────────────────────────────────────────┘
```

**Model**: Downloaded from HuggingFace (`artbreguez/polymarket-btc-model`) on startup, saved to `/tmp/champion.pkl`.

**Slot history**: Ring buffer of recent slot outcomes for lag features (lag_1/2/3_outcome, lag_streak).

---

## Feature Computation — MUST Match Training

This is the single most important rule in the system:

**`build_features()` in live_trader.py MUST produce identical features to `tick_features_v8()` in training scripts.**

The feature set (v18, 30 features):
- 6x30s sub-windows: `btc_up_w0..w5` + per-window zscores
- Multi-scale up_ratio zscore (5/10/20 slots)
- Time-weighted order flow, VWAP trend, volume-weighted momentum
- Lag outcomes + lag streak from slot history ring buffer
- BTC spot: inslot_ret/vol, pre_5m/15m/30m/1h/4h_ret/vol, dist_1k/5k/10k
- OB features filled with neutral defaults (not available live)

If training adds/removes/renames a feature and live_trader.py isn't updated, the model will silently produce garbage predictions. See "Feature Parity" in 08-troubleshooting.md.

---

## WebSocket Management

**Binance spot stream** (`btcusdt@kline_1m`):
- Runs in background thread
- Writes to `_spot_buffers` deque (maxlen=300 for BTC)
- Periodically persists to `/tmp/spot_buffer.json`
- Buffer staleness check: if last update > 120s ago, skip trading

**Polymarket CLOB WS** (`wss://ws-subscriptions-clob.polymarket.com/ws/market`):
- Used for real-time trade observation during slots
- Trades also fetched via REST (data-api) as fallback

---

## Order Placement

Uses Polymarket CLOB API:
- **Endpoint**: `https://clob.polymarket.com`
- **Auth**: Builder API credentials (key, secret, passphrase)
- **Wallet**: Proxy wallet (`POLY_SAFE_ADDRESS`)
- **Order type**: Market buy on UP or DOWN token
- **Taker fee**: ~2%

---

## P&L Tracking

Trades logged to `/tmp/live_trades.json`:
- Entry price, side (UP/DOWN), model probability, edge
- Resolution outcome, profit/loss
- Cumulative P&L

---

## Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Stale asks | Large spread, model sees edge that doesn't exist | `MIN_EDGE_MID` check catches this — edge must exist vs mid too |
| WS disconnects | Missing candles in spot buffer | Auto-reconnect in spot daemon; staleness check skips trading if buffer too old |
| Cold start | No spot history on first boot | Buffer needs ~5h to fill for `btc_pre_4h_ret`; uses available data with graceful degradation |
| Binance 451 | Binance blocks Fly.io region | Spot data fetched via WS stream (not REST); if WS also blocked, pre-fetch locally |
| Feature mismatch | Model predictions are nonsensical | Verify build_features() output matches training script exactly |

---

## Monitoring

```bash
# Live logs
fly logs -a polymarket-maker-mm

# App status
fly status -a polymarket-maker-mm

# SSH into running machine
fly ssh console -a polymarket-maker-mm

# Check spot buffer
fly ssh console -a polymarket-maker-mm -C "cat /tmp/spot_buffer.json | python3 -m json.tool | head -20"

# Check recent trades
fly ssh console -a polymarket-maker-mm -C "cat /tmp/live_trades.json"
```

Key log lines to watch:
- `ENTER` — order placed (shows side, price, edge, confidence)
- `SKIP` — entry criteria not met (shows why)
- `SETTLE` — slot resolved (shows P&L)
- `STALE` — spot buffer too old, skipping
- `WS reconnect` — WebSocket recovered
