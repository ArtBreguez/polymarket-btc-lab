# 3 Critical Bugs Fixed — 2026-06-04 Session 2

## Bug 1: OB Temporal Features Always Zero (CRITICAL — top-5 feature blind)

**Symptom:** Model predicted UP at 73.5% conf when CLOB signals were bearish (up_ratio=0.459, momentum=-0.140). Trade LOST.

**Root cause:** `_build_ob_features()` took a SINGLE snapshot. Three temporal features were hardcoded to 0.0:
- `ob_mid_drift` (feature importance rank #2) — always 0.0
- `ob_imb_momentum` (rank #28) — always 0.0
- `x_ob_drift_x_inslot` (rank #5) — always 0.0 (because ob_mid_drift × anything = 0)
- `ob_imb_w0/w1/w2` — all set to same current imbalance (model sees no temporal variation)

**Impact:** Top-5 feature providing ZERO signal. Calibration trained WITH real drift values produces shifted probabilities when drift=0 at inference. Model over-relies on remaining features and can flip direction.

**Fix:** Cache "open" OB snapshot at first poll of entry window (`_ob_open_cache`). Compute:
- `ob_mid_drift = close_mid - open_mid`
- `ob_imb_momentum = close_imb - open_imb`
- `ob_imb_w0 = open_imb`, `ob_imb_w1 = (open+close)/2`, `ob_imb_w2 = close_imb`
Cache cleared per-slot via `_ob_last_slot` global.

## Bug 2: Data-API Cloudflare Cache (Frozen Tick Count)

**Symptom:** Tick count was IDENTICAL across all polls within one entry window: 886 at t=171s, 182s, 192s, 203s. Should increase as data-api lag catches up.

**Root cause:** `data-api.polymarket.com` returns `Cache-Control: public, max-age=300` + Cloudflare CDN. All requests to same URL within 5 minutes return cached response. No cache-busting param.

**Impact:** Model makes 7 decisions on same stale snapshot. No new information between polls.

**Fix:** Add `_t=int(time.time())` cache-buster to params in `fetch_inslot_trades()`. After fix: 1018 → 1018 → 1156 → 1156 → 1624 ticks (real progression).

## Bug 3: WebSocket Instability (No Ping/Pong + Long Keepalive)

**Symptom:** "CLOB WS error: no close frame received or sent" every few minutes. Frequent HTTP /book fallbacks. Stale ask prices.

**Root cause:** `ping_interval=None` disabled RFC 6455 ping/pong. Keepalive was only every 480s (8min). Intermediate proxies/LBs kill idle TCP after 60-120s. Also, reconnect cleared ALL cached prices (`_clob_prices.clear()`), forcing HTTP fallback for every token even if prices were <5s old.

**Fix:**
1. `ping_interval=20, ping_timeout=10` (match Binance WS config)
2. Keepalive interval 480s → 30s
3. Removed `_clob_prices.clear()` on reconnect (PRICE_MAX_AGE=15s handles staleness)

After fix: 0 WS disconnects observed. Keepalive every 30s visible in logs. Ask prices from WS (0.1-0.3s age) instead of HTTP fallback.

## Validation

All 3 fixes validated in production logs:
- Tick count now increases between polls ✓
- market_mid correct ($0.885 vs ask $0.890 = $0.005 spread) ✓
- WS stable (keepalive every 30s, 0 disconnects) ✓
- Ask from WS (0.1s age) not HTTP fallback ✓
