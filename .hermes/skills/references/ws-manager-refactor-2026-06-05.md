# WebSocket Manager Refactor — 2026-06-05

## Problem
Both WS daemons (Binance spot + Polymarket CLOB) had identical reliability issues:

1. **Fixed 5s reconnect delay** — no exponential backoff. Under instability, rapid reconnect loops could trigger rate limiting.
2. **No zombie detection** — TCP could stay "alive" with no data flowing. Daemon would sit connected but receiving nothing. All WS prices would go stale, triggering HTTP fallback on every ask price check.
3. **CLOB daemon used `recv(timeout=1.0)` polling** — generated thousands of `asyncio.TimeoutError` per minute (silently caught). Inefficient vs Binance daemon's `async for` pattern.
4. **No health metrics** — impossible to diagnose WS stability from logs. "CLOB WS error: reconnecting in 5s" was the only signal.
5. **Cache invalidation on price_change worsening** — when best ask rose, the entire cache entry was popped, forcing HTTP fallback even if the new price was still valid.

## Solution: `deploy/ws_manager.py`

### Architecture
Single `WebSocketManager` class, used by both daemons:

```
WebSocketManager(name, url, on_message, on_connect, config)
  ├─ _connect_loop()      — main reconnect loop with backoff
  │   ├─ websockets.connect() with ping/pong
  │   ├─ on_connect(ws)   — callback for (re)subscriptions
  │   └─ _message_loop()  — async for + zombie watchdog
  │       ├─ on_message(msg) — callback for each parsed JSON msg
  │       └─ _zombie_watchdog() — background asyncio.Task
  ├─ BackoffCalculator     — exponential with jitter, resets on success
  ├─ WSMetrics            — thread-safe counters + rolling rate
  └─ WSConfig             — all timeouts/intervals configurable
```

### Key Design Decisions

1. **Backoff resets after first message, not after connect** — connecting to a WS that immediately disconnects shouldn't reset the backoff counter.

2. **Zombie watchdog runs as asyncio.Task alongside message loop** — the `async for` loop blocks while waiting for messages, so zombie detection runs concurrently via `asyncio.create_task()`.

3. **Active zombie detection (v5 fix)** — the zombie watchdog does NOT blindly kill connections with no data. On one-sided/illiquid markets (asks >=0.97, empty book), the CLOB WS legitimately sends zero data messages for minutes while the connection is perfectly healthy. Server pings/pongs happen at the protocol level (RFC 6455) and do NOT appear in the `async for raw in ws` loop. **Fix**: before force-closing, the watchdog sends a protocol-level `ws.ping()` and waits for PONG (10s timeout). If PONG comes back → connection alive, reset `last_message_at`, continue. If PING fails → true zombie, kill. This eliminated false-positive kills (~1/min on quiet markets → near-zero).

4. **`send_sync()` for cross-thread subscriptions** — the CLOB daemon needs to accept subscription requests from the main trading thread. Uses `asyncio.run_coroutine_threadsafe()` to safely inject into the WS event loop.

5. **Don't clear price cache on reconnect** — `PRICE_MAX_AGE` (15s) handles staleness. Clearing on reconnect forces HTTP fallback for ALL tokens even if prices were updated <5s ago.

6. **Keepalive moved into on_message callback** — instead of a separate polling loop, the CLOB keepalive (re-subscribe every 30s) piggybacks on the message handler, simplifying the architecture.

### Configuration per Daemon

| Setting | Binance Spot | CLOB |
|---------|-------------|------|
| ping_interval | 20s | None (server controls) |
| ping_timeout | 10s | None |
| zombie_timeout | 60s | 120s (raised from 60s for quiet markets) |
| health_log_interval | 300s | 300s |
| initial_backoff | 5s | 5s |
| max_backoff | 60s | 60s |

### Refactor Changes in live_trader.py

**Before**: Two separate `_*_daemon_thread()` functions, each with inline `asyncio.run()`, manual `websockets.connect()`, manual exception handling, hardcoded `await asyncio.sleep(5)` on error.

**After**: Two `WebSocketManager` instances initialized in `start_*_daemon()`. Message handling extracted into `_spot_on_message()` and `_clob_on_message()` async callbacks. CLOB subscription logic extracted into `_clob_on_connect()`, `_clob_drain_and_subscribe()`, `_clob_prune_stale()`.

### Tests: `tests/test_ws_manager.py` (41 tests)

- **BackoffCalculator** (7): exponential growth, cap, jitter variance, reset, minimum floor
- **WSMetrics** (7): initial state, counters, rate window, thread safety (4 threads × 100 ops)
- **WSConfig** (2): defaults, custom overrides
- **WebSocketManager** (9): health before start, disconnect state, stop without start, send_sync disconnected, message dispatch, invalid JSON skip, callback error resilience, zombie trigger/no-false-positive, double-start warning, stale price on reconnect
- **CLOB handler** (7): book event best ask, filter >=0.97, empty asks, price_change tighten/widen/bid-ignore/high-reject
- **Spot handler** (5): kline append, same-ts update, new-ts append, unknown stream, missing data

### Post-Deploy Bugs Found & Fixed

**Bug A: `send_sync()` deadlock inside async event loop**
- **Symptom**: `[clob] send_sync() failed:` on every keepalive (every 30s). Subscriptions and keepalive silently broken.
- **Root cause**: `_clob_on_message()` is called from within the WS manager's async event loop. It called `send_sync()` which uses `asyncio.run_coroutine_threadsafe()` — but that deadlocks when called FROM the same event loop (it tries to schedule on the loop that's already blocked waiting for the coroutine to complete).
- **Fix**: Changed keepalive and in-message subscription drain to use `await _clob_ws_manager.send()` (async) instead of `send_sync()` (sync wrapper). `send_sync()` is only for cross-thread calls (e.g., main trading thread queuing subscriptions).
- **Rule**: Inside `on_message` callbacks, always use `await send()`. `send_sync()` is for external threads only.

**Bug B: Binance WS geo-blocked (HTTP 451) from Fly.io Amsterdam**
- **Symptom**: `[binance-spot] Connection rejected: HTTP 451` on every connect attempt. Spot buffer had 0 candles despite seed working.
- **Root cause**: Binance blocks WebSocket connections from certain cloud regions (Amsterdam/EU) for regulatory compliance. REST API (`/api/v3/klines`) is NOT blocked — only the WebSocket stream.
- **Fix**: Added `_spot_rest_poll()` — a background thread that polls Binance REST `/klines` every 30s as fallback. The WS manager still tries to connect (for low-latency in non-blocked regions), but the REST poller ensures spot data stays fresh regardless.
- **Architecture**: Seed (REST, 300 candles) → WS manager (tries to connect) → REST fallback poller (every 30s, 5 latest candles). All three write to the same `_spot_buffers` deque.
- **Rule**: For geo-sensitive data sources, always implement REST polling as a fallback alongside WS. Don't assume WS availability.

**Bug C: False-positive zombie kills on quiet markets (2026-06-05)**
- **Symptom**: CLOB WS disconnects every ~65s with "Zombie detected — no message for 65s". 16 disconnects, 7 zombie kills in logs. Bot constantly reconnecting, invalidating price cache, falling back to HTTP.
- **Root cause**: On one-sided markets (asks >=0.97 or empty book), no CLOB trades happen, so zero data messages flow over WS. The server still sends protocol-level PINGs every ~30s, and the client responds with PONGs — but these happen inside the `websockets` library and do NOT appear in the `async for raw in ws` loop. The zombie detector only tracked data messages, so it saw "no messages for 60s" and killed a perfectly healthy connection.
- **Fix**: Active ping/pong probe before killing. Zombie watchdog now does: `pong = await ws.ping(); await asyncio.wait_for(pong, timeout=10.0)`. If PONG comes back → connection alive, log "market quiet", reset `last_message_at`. If PING fails → true zombie, force close. Also raised zombie_timeout from 60s to 120s.
- **Impact**: Disconnects dropped from ~1/min to near-zero on quiet markets. Cache invalidation churn eliminated. HTTP fallback only triggered by genuinely stale data.

### Deploy Note
**IMPORTANT**: `ws_manager.py` must be added to `deploy/Dockerfile` COPY list. Missing = ImportError crash loop (10 restarts → machine stops, needs manual `fly machine start`).

```dockerfile
COPY ws_manager.py .
```
