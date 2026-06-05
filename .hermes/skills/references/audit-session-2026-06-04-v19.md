# Full Audit Session — 2026-06-04 (v19 deployment)

## Context
After v19 training (AUC=0.9000) and initial deployment, a comprehensive audit was run
covering WebSockets, feature parity, hardcoded values, and live behavior.

## Audit Methodology
Three parallel subagent audits were dispatched:
1. WebSocket health (CLOB + Binance)
2. Feature parity (all 40 features, train vs live)
3. Hardcoded values scan

## Critical Finding: Auditor Used Wrong Training Script
The hardcoded-values auditor compared live code against `train_v15_modal.py` (old version)
and reported 3 "critical" feature formula mismatches:
- btc_tw_up_ratio: "different weighting scheme" — FALSE (v19 matches live)
- btc_buy_ratio: "volume-based vs count-based" — FALSE (v19 matches live)
- btc_size_disparity: "difference vs ratio" — FALSE (v19 matches live)

All three were changed between v15→v17. The v19 training script matches live perfectly.

**Lesson**: When delegating audit tasks, ALWAYS specify the exact training script version
in the context. Include: `champion.pkl version = v19, reference = scripts/train_v19_modal.py`.

## Bugs Found and Fixed This Session

### 1. Gamma outcomePrices Stale (CRITICAL)
- Gamma returns ~$0.50/$0.50 for new BTC 5-min markets, rarely updates
- Real book mid was $0.12 while Gamma said $0.505 (delta = 0.38)
- Fix: get_market_mid() now uses CLOB /book midpoint
- Verified: market_mid=$0.685 vs ask=$0.690 (delta $0.005) — correct

### 2. Data-API Cloudflare Cache (CRITICAL)
- 5-min CDN TTL froze tick count at 886 across all entry window polls
- Fix: Added `_t=int(time.time())` cache-buster
- Verified: ticks now change between polls (976→986→1418)

### 3. OB Temporal Features Always Zero (CRITICAL)
- ob_mid_drift (#2 importance), ob_imb_momentum, ob_imb_w0/w1/w2
- Fix: _ob_open_cache stores first snapshot, computes real drift
- Verified: features now produce non-zero values

### 4. WS Stability (MODERATE)
- "no close frame" every few minutes, wiped price cache
- Fix: ping_interval=20, keepalive 480s→30s, don't clear cache on reconnect
- Verified: zero WS errors over 1+ hour monitoring

### 5. OB Fallback Defaults Wrong (MODERATE)
- ob_mid defaulted to 0.0 (should be 0.5 for binary market)
- Fix: Context-aware neutral defaults dict

### 6. Spot Stale Data Leaked Through (MODERATE)
- BUFFER_STALE=120s only logged warning, still used stale data
- Fix: Hard gate — return zeros if stale

### 7. Shares Cap (MINOR)
- Configured to always use CLOB minimum of 5 shares (testing phase)

## Live Behavior Validation Post-Fixes
- Trade executed: BUY DOWN @ $0.690, 5 shares, $3.45, FILLED
- market_mid=$0.685 (correct, $0.005 from ask)
- edge_ask=11.0%, edge_mid=11.5% (above minimums)
- DATA GATE correctly blocking slots with <3/6 sub-windows
- WS keepalive every 30s, zero disconnects
- Tick counts updating between polls (cache-bust working)

## WS Health Summary
- CLOB WS: Healthy. ping/pong active, keepalive 30s, 18-22 tokens subscribed
- Binance Spot WS: Healthy. 300 BTC candles, ping_interval=20
- No Coinbase references found (Binance only)
- PRICE_MAX_AGE=15s staleness guard working correctly
