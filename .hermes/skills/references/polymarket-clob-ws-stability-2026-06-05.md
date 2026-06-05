# Polymarket CLOB WebSocket Stability — Research & Fix (2026-06-05)

## Problem
CLOB WS (`wss://ws-subscriptions-clob.polymarket.com/ws/market`) dropped every ~60s with code 1006 (abnormal closure — no close frame). This happened on Fly.io Amsterdam with Python `websockets` library.

## Root Cause: Double-Ping Problem
Confirmed by @poly-rodr (Polymarket team member) in [py-clob-client issue #82](https://github.com/Polymarket/py-clob-client/issues/82):

> "try setting the ping_interval to None, to let the server control the ping/pong mechanism"
> "There's some incompatibility between the default ping_interval set by websockets and the server-side ping mechanism."

What happens:
1. Polymarket server sends WebSocket PING frames every ~30s
2. Python `websockets` lib (default `ping_interval=20`) also sends client PINGs every 20s
3. Server does NOT properly handle unsolicited client-initiated PINGs
4. Server drops connection after ~60s with code 1006

## Fix
```python
async with websockets.connect(
    ws_url,
    ping_interval=None,   # CRITICAL: disable client-side pings
    ping_timeout=None,     # CRITICAL: disable client-side ping timeout
    close_timeout=5,
) as ws:
```

The `websockets` library STILL auto-responds to server PING frames with PONG even with `ping_interval=None`. This is protocol-level behavior (RFC 6455) and cannot be disabled.

## Results
- Before (ping_interval=20): ~3 drops per 3 minutes, avg uptime ~60s
- After (ping_interval=None): 1 drop in 5+ minutes, uptime 98-236s+
- Local testing (from Ubuntu, not Fly.io): 0 drops in 90s with 8 tokens

## Additional Findings

### Official Docs (docs.polymarket.com)
- ZERO mention of keepalive, ping, pong, heartbeat, or connection lifetime
- Only says: "Connections that are idle for too long will be terminated"
- Recommends: "implement reconnection logic to handle unexpected disconnections"
- No specific timeout values documented

### Code 1006 Meaning (RFC 6455)
- "Abnormal Closure" — TCP dropped without WebSocket close handshake
- Never set by an endpoint — generated locally when underlying TCP dies
- Causes: proxy/CDN timeout, server crash, firewall, network path issue

### Cloudflare Factor
- Polymarket likely uses Cloudflare as reverse proxy
- Cloudflare default WebSocket idle timeout: 100s
- With server pings every ~30s, idle timeout shouldn't trigger
- Remaining occasional drops (~1 per 5 min) are likely Cloudflare proxy cycling

### Fly.io Notes
- Outbound WS connections have no specific idle timeout from Fly.io side
- Amsterdam region is required — US blocks Polymarket by regulation
- Binance WS is geo-blocked (HTTP 451) from Amsterdam — use REST fallback

## Stale Price Protection Stack (4 layers)
After each reconnect, prices may be stale. Four layers of protection:

1. **PRICE_MAX_AGE=15s** — `get_ask_price_ws()` rejects WS cache older than 15s, falls back to HTTP `/book`
2. **Cache invalidation on reconnect** — `_clob_on_connect()` calls `_clob_prices.clear()`, forcing HTTP fallback until fresh book snapshot arrives
3. **Ask vs mid divergence >$0.20** — catches stale/deep-book asks that slipped through layers 1-2
4. **price_change invalidation** — when best ask rises (book worsened), cache is cleared for that token

## WSConfig for Polymarket CLOB
```python
WSConfig(
    ping_interval=None,      # Server controls ping/pong
    ping_timeout=None,       # Server controls ping/pong
    close_timeout=5.0,
    zombie_timeout=60.0,     # Server pings every ~30s → 60s without data = zombie
    health_log_interval=300.0,
)
```

## WSConfig for Binance Spot
```python
WSConfig(
    ping_interval=20.0,      # Binance handles client pings fine
    ping_timeout=10.0,
    zombie_timeout=60.0,     # Binance sends klines every ~2s
    health_log_interval=300.0,
)
```
Note: Binance WS is fine with client pings — the double-ping problem is Polymarket-specific.
