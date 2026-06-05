---
name: polymarket-btc-pipeline
description: "Comprehensive quant operations guide for Polymarket BTC 5-min prediction pipeline. Covers architecture, deployment, training, monitoring, bug database, feature parity audit, and operational runbooks. An agent loading this skill can diagnose issues, run training, deploy, and monitor without guessing."
version: 3.0.0
author: Arthur + Hermes
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [ML, Trading, LightGBM, Modal, Fly.io, Polymarket, BTC, Quant]
    related_skills: []
---

# Polymarket BTC 5-min Prediction Pipeline — Quant Operations Guide

## Quick Reference

| Resource | Value |
|----------|-------|
| Repo | `/home/ubuntu/polymarket-btc-lab` |
| Copytrade repo | `/home/ubuntu/polymarket-copytrade` |
| Fly.io app | `polymarket-maker-mm` (AMS region) |
| Fly.io copytrade | `polymarket-copytrade-amber-woodland-5363` (AMS region) |
| Fly.io CLI | `/home/ubuntu/.fly/bin/flyctl` |
| HuggingFace model | `artbreguez/polymarket-btc-model` (private) |

### HuggingFace Repo Contents
| File | Content |
|------|---------|
| `champion.pkl` | Current model bundle (CalibratedClassifierCV + metadata) |
| `champion_meta.json` | Version, metrics, feature list |
| `README.md` | Model card: architecture, features, training config, reproduction |
| `EXPERIMENTS.md` | Full experiment log (v4→v19) with lessons learned |
| `training_config.json` | Machine-readable Optuna ranges, WF settings, promotion gate |
| `feature_definitions.json` | All 40 features: formula, neutral value, range, category |
| `fetch_training_data.py` | Script to reconstruct training data from original APIs |
Note: Raw training data lives on Modal Volume `btc-local-data` (~2GB+).
| Modal profile | `artbreguez/main` |
| Modal volume | `btc-local-data` |
| Wallet | `0x362095ED373A63d3AA58091E1d74CeB634129b33` (BTC bot) / `0xd0E6f59F7dE8Ba2DfA1289C46Ab4809538974cBb` (copytrade) |
| Secrets file | `~/.env` (HF_TOKEN, MODAL_TOKEN_COMMAND, FLY_ARTBREGUEZ_UCL_TOKEN) |
| pmdata API key | `sk-5uX...Ijko` — **EXPIRED as of 2026-06-05**. Needs renewal at pmdata.dev. |
| Monitor crons | REMOVED 2026-06-05 (were: btc-bot-monitor 10min, btc-bot-monitor 2h, btc-repo-autopush 6h) |
| Current model | v21: 30 features, AUC=0.9002, Brier=0.1290, Acc=81.3% |
| v21 result | PROMOTED (3/3 gate). Ablation: 30-feat variant beat v19 on all metrics with 25% fewer features. live_trader.py pruned (-137 lines). See `references/v21-feature-pruning-2026-06-05.md` |
| v20 attempt | NOT PROMOTED (0/3). Same data + 45 features = dilution. See `references/v20-training-attempt-2026-06-05.md` |
| HF data | Training data now on HF under `data/` (all_markets.csv, ticks, OB, spot) |

---

## 1. ARCHITECTURE

### System Diagram
```
DATA COLLECTION                    TRAINING                         DEPLOYMENT
───────────────                    ────────                         ──────────
[pmdata.dev API]                   [Modal: train_v19_modal.py]      [Fly.io: polymarket-maker-mm]
  → poly_l2 parquet per market       ↑ reads from Volume              ↑ downloads champion.pkl
  → ticks, OB snapshots              ↓ writes champion.pkl            ↑ from HuggingFace on startup
  → Modal Volume: btc-local-data     → HuggingFace upload             │
                                                                       │
[Binance API]                      [Modal: fetch_ob_features.py]    [live_trader.py ~1900 lines]
  → 1m BTCUSDT candles               → pre-compute OB features       ├─ Spot daemon (Binance WS)
  → Modal Volume                     → ob_features_full.parquet      ├─ CLOB WS daemon (prices)
                                                                      ├─ DataQualityGate (5-layer)
[Polymarket data-api]              [GitHub Actions]                   ├─ build_features()
  → inslot CLOB trades               ├─ ci.yml (tests)               ├─ Model predict
  → market resolution                └─ deploy.yml (Fly.io)          └─ Order execution
```

### Key Files
| File | Purpose | Lines |
|------|---------|-------|
| `deploy/live_trader.py` | Main bot: spot daemon, CLOB WS, features, trading loop | ~1930 |
| `deploy/ws_manager.py` | Resilient WS connection manager (backoff, zombie detection, metrics) | ~450 |
| `deploy/data_quality_gate.py` | 5-layer circuit breaker system | ~510 |
| `deploy/Dockerfile` | Container: python:3.12-slim + libgomp1 | 15 |
| `deploy/fly.toml` | Fly.io config: AMS region, performance-1x, 2GB RAM | 14 |
| `scripts/train_v21_modal.py` | Current training: ablation + Optuna + walk-forward on Modal | ~730 |
| `scripts/fetch_ob_features_modal.py` | L2 orderbook feature pre-computation | - |
| `scripts/monitor_bot.sh` | Health check: log parsing + alerting | - |
| `scripts/fetch_spot_full.py` | Binance spot data fetch (run locally, NOT on Modal) | - |
| `tests/test_ws_manager.py` | 41 unit tests for WS manager + handler logic + reconnect cache invalidation | ~600 |
| `tests/test_features.py` | Unit tests for feature computation | ~145 |
| `src/btc_lab/features.py` | Shared feature computation library | - |

### Bot Startup Sequence
1. `start_spot_daemon()` — seeds 300 candles via REST, starts `WebSocketManager("binance-spot")` + REST fallback poller (every 30s)
2. `load_model()` — downloads champion.pkl from HuggingFace
3. `SecureClient.create()` — Polymarket CLOB auth
4. `start_clob_daemon()` — starts `WebSocketManager("clob")`, pre-subscribes 4 slots
5. `_seed_slot_history()` — fetches REAL up_ratio for 25 past slots (~3s)
6. `DataQualityGate` — 3-slot warmup before first trade
7. Main loop: settle → backfill → entry check → predict → trade

### Performance Optimizations (live_trader.py)
- **Parallel trade fetch**: `fetch_inslot_trades()` uses `ThreadPoolExecutor(max_workers=2)` to fetch yes_token and no_token trades simultaneously. Cuts ~4-6s sequential to ~2-3s.
- **Parallel price fetch**: `get_ask_price()` and `get_market_mid()` run in `ThreadPoolExecutor(max_workers=2)` — measured 78-113ms parallel vs ~2-10s sequential.
- **Progressive fill polling**: delays `[1,2,3,4,5,5]` = 20s total (vs old `[5]*6` = 30s). First poll at 1s catches fast fills.
- **Pre-window volatility**: `btc_pre_{5m,15m,30m,1h,4h}_vol` computed from spot buffer segments (was hardcoded 0.0).
- **OB depth change**: `ob_depth_change = total_depth(close) - total_depth(open)` using cached open snapshot (was hardcoded 0.0).

### Background Threads
- **Spot daemon**: `WebSocketManager("binance-spot")` → `_spot_buffers` dict → `/tmp/spot_buffer.json`
  - btcusdt: 300 candles (5h, needed for btc_pre_4h_ret)
  - ethusdt/solusdt: 75 candles (kept for future use, not used by v19)
  - Zombie timeout: 60s (Binance sends klines every ~2s)
  - **REST fallback poller** (`_spot_rest_poll()`): polls Binance REST `/klines` every 30s in a separate thread. Ensures spot data stays fresh when WS is geo-blocked (HTTP 451 from Fly.io Amsterdam). The WS manager still tries to connect for low-latency in non-blocked regions.
- **CLOB WS daemon**: `WebSocketManager("clob")` → `_clob_prices` dict
  - Subscribes to token IDs for current slot
  - `PRICE_MAX_AGE = 15s` — stale prices trigger HTTP fallback
  - **CRITICAL**: `ping_interval=None, ping_timeout=None` — lets the Polymarket server control ping/pong. Their server sends pings every ~30s; the websockets lib auto-responds with pong. Client-initiated pings (`ping_interval=20`) cause "double-ping" conflict → server drops connection every ~60s (code 1006). Ref: https://github.com/Polymarket/py-clob-client/issues/82 (confirmed by @poly-rodr, Polymarket team).
  - Cache invalidation on reconnect: ALL cached prices cleared when WS reconnects, forcing HTTP `/book` fallback until fresh data arrives from the new connection.
  - Zombie timeout: 120s (raised from 60s — one-sided markets can go minutes without data msgs; active ping/pong probe prevents true zombies)

### WebSocket Manager (deploy/ws_manager.py)
Both WS daemons use a shared `WebSocketManager` class that provides:
- **Exponential backoff with jitter**: 5s → 10s → 20s → max 60s (±30% jitter). Resets after first successful message.
- **Active zombie detection with ping/pong probe**: If no *data* message received for N seconds, sends a protocol-level PING and waits for PONG (10s timeout). If PONG comes back, the connection is alive — just no data (common on illiquid/one-sided markets where asks are >=0.97 or empty). Resets `last_message_at` to avoid re-checking every 5s. Only force-closes if PING also fails (true zombie: silently dead TCP connection). This prevents false-positive zombie kills on quiet markets — the old approach killed healthy connections every ~65s when no trades were flowing.
- **Health metrics**: Thread-safe counters for connects, disconnects, errors, zombie kills, msgs/min (60s rolling window), last_msg_age. Logged every 5 min.
- **`async for` message loop**: Uses websockets' native async iterator (not `recv(timeout=1.0)` polling, which generated thousands of silent TimeoutErrors).
- **`send_sync()`**: Thread-safe synchronous send for cross-thread subscription requests.
- **Structured logging**: Every lifecycle event (connect, disconnect, backoff, zombie kill, health) logged with `[name]` prefix.
- **Graceful shutdown**: `stop_event` threading.Event allows clean exit during backoff waits.
- Tests: `tests/test_ws_manager.py` — 41 unit tests covering backoff, metrics, config, lifecycle, zombie (active ping/pong probe), message dispatch, handler logic, and reconnect cache invalidation.

### Zombie Detection Log Patterns
| Log message | Meaning |
|-------------|---------|
| `[clob] No data for Ns but ping/pong OK — connection alive (market quiet)` | Active probe passed — connection healthy, just no CLOB data (one-sided market). Normal. |
| `[clob] Zombie confirmed — no message for Ns AND ping failed (...). Force closing.` | True zombie — TCP alive but server unresponsive. Connection killed, will reconnect with backoff. |

---

## 2. CRITICAL BUG DATABASE

These are hard-won lessons. Every one caused real money loss or silent model degradation.

### Bug #1: Gamma outcomePrices are STALE
**Symptom**: All trades blocked by ask-mid divergence check (>0.20).
**Root cause**: Gamma API returns outcomePrices ~$0.50/$0.50 for BTC 5-min markets at creation and RARELY updates during the active slot. Real CLOB book mid can be $0.12 while Gamma says $0.505.
**Fix**: `get_market_mid()` computes mid = `(best_bid + best_ask) / 2` from CLOB `/book` endpoint. Falls back to binary-market identity (`up + dn ≈ 1`) when one side has no book.
**Rule**: NEVER use Gamma outcomePrices for active BTC 5-min slots. ALWAYS use CLOB /book midpoint.

### Bug #2: Data-API Cloudflare CDN Cache (5-min freeze)
**Symptom**: Tick count identical across all polls in entry window. Model makes 7 decisions on same stale data.
**Root cause**: `data-api.polymarket.com` responses cached by Cloudflare CDN for 5 minutes.
**Fix**: Add `_t=int(time.time())` cache-buster param to ALL data-api requests in `fetch_inslot_trades()`.
**Rule**: Always cache-bust data-api requests.

### Bug #3: OB Temporal Features Need 2+ Snapshots
**Symptom**: `ob_mid_drift` (top-5 feature), `ob_imb_momentum`, `ob_imb_w0/w1/w2` always zero.
**Root cause**: Single OB snapshot gives drift=0 by definition. Need open snapshot (first poll) and close snapshot (current poll).
**Fix**: `_ob_open_cache` dict stores first OB snapshot per token_id. `_ob_last_slot` global tracks current slot — cache clears when slot changes. Drift = close.mid - open.mid. Windowed imbalance: w0=open, w1=interpolated, w2=close.
**Implementation details**:
- `_fetch_ob_snapshot()` — raw CLOB REST fetch, returns parsed dict
- `_build_ob_features()` — caches first snapshot as "open", computes drift vs current "close"
- Cache cleanup: `if cur_slot != _ob_last_slot: _ob_open_cache.clear(); _ob_last_slot = cur_slot` in main loop
- First poll of entry window: open==close, so drift=0 (correct — no change yet)
- Later polls: real drift emerges as book changes over 10-70s of entry window
**Rule**: Temporal features require at minimum 2 data points. Never default temporal features to 0.0 when they're top-ranked.

### Bug #4: CLOB WS Ping/Pong + Keepalive + Zombie Connections
**Symptom**: Frequent "no close frame" disconnects, wiping price cache → all prices become None → trades blocked. Also: silent zombie connections (TCP alive but no data flowing).
**Root cause (original)**: `ping_interval=None` + 480s keepalive. Proxies/LBs kill idle TCP connections after 60-120s.
**Root cause (zombie)**: Even with ping/pong, TCP can stay "alive" with no application data. The old daemon had no way to detect this — it would sit connected but receiving nothing, causing all WS prices to go stale and triggering HTTP fallback for every ask.
**Root cause (reconnect)**: Fixed 5s reconnect delay. If Polymarket WS is unstable, rapid reconnect loops could trigger rate limiting.
**Fix (v1)**: `ping_interval=20`, `ping_timeout=10`, keepalive reduced to 30s. Don't clear price cache on reconnect (PRICE_MAX_AGE handles staleness).
**Fix (v2, 2026-06-05)**: Refactored both daemons to use `WebSocketManager` (`deploy/ws_manager.py`):
- Exponential backoff: 5s → 10s → 20s → max 60s with ±30% jitter (prevents reconnect storms)
- Zombie detection: no message for 45s (CLOB) / 60s (Binance) → force close + reconnect
- `async for` message loop replaces `recv(timeout=1.0)` polling (eliminated thousands of silent TimeoutErrors per minute)
- Health metrics logged every 5 min (uptime, disconnect count, msgs/min, zombie kills)
- Cache invalidation on reconnect: ALL `_clob_prices` cleared in `_clob_on_connect()`, forcing HTTP `/book` fallback until fresh book snapshot arrives. Without this, stale prices from the ~4-6s disconnect gap could pass `PRICE_MAX_AGE` (15s) and be used for trading.
**Fix (v3, 2026-06-05)**: Removed 30s re-subscribe keepalive — was generating ~30k+ unnecessary msgs/min.
**Fix (v4, 2026-06-05)**: ROOT CAUSE FOUND. "Double-ping problem" confirmed by @poly-rodr (Polymarket team) in py-clob-client issue #82:
- Polymarket server sends its own WebSocket PING frames every ~30s
- Python `websockets` library (with `ping_interval=20`) ALSO sends client PINGs every 20s
- Server does NOT handle unsolicited client PINGs → drops connection every ~60s with code 1006
- **FIX**: Set `ping_interval=None, ping_timeout=None` on the CLOB WS connection
- The `websockets` library still auto-responds to server PING with PONG even with `ping_interval=None`
- Result: connection uptime went from ~60s to 98s+ (still occasional drops from Cloudflare proxy, but much rarer)
- **Local testing confirmed**: 0 drops in 90s with 8 tokens subscribed when using `ping_interval=None`
- Ref: https://github.com/Polymarket/py-clob-client/issues/82
**Fix (v5, 2026-06-05)**: Active zombie detection — zombie watchdog now sends protocol-level PING before killing. On one-sided markets (asks >=0.97, empty book), CLOB sends zero data messages for minutes but server pings/pongs still flow at protocol level. Old zombie detector (v2-v4) killed these healthy connections every ~65s (zombie_timeout=60s). Fix: PING probe before kill — if PONG comes back, connection is alive (reset last_message_at, don't kill). Also raised zombie_timeout from 60s to 120s. Disconnects dropped from ~1/min to near-zero on quiet markets.
**Stale price protection stack (4 layers)**:
1. `PRICE_MAX_AGE=15s` — rejects WS cache older than 15s, falls back to HTTP `/book`
2. Cache invalidation on reconnect — clears ALL prices in `_clob_on_connect()`, forces HTTP fallback
3. Ask vs mid divergence >$0.20 — catches stale/deep-book asks that passed layers 1-2
4. `price_change` invalidation — clears cache when best ask rises (book worsened)
**Rule**: For Polymarket CLOB WS: set `ping_interval=None, ping_timeout=None` (server controls ping/pong). For other WS servers: check if they send their own pings before enabling client pings. Always invalidate price caches on reconnect. Use exponential backoff, not fixed delays. Monitor for zombie connections. Don't use app-level messages as keepalive — RFC 6455 PING/PONG from the server is sufficient.

### Bug #5: Cross-Market Trade Contamination (~67%)
**Symptom**: up_ratio ≈ 0.5 for all markets. Model predictions random.
**Root cause**: `data-api.polymarket.com/trades?asset=<token_id>` returns trades from ALL markets sharing that token ID — ETH, SOL, BNB, daily BTC, tennis, weather, elections. API's `slug` query param is SILENTLY IGNORED server-side.
**Fix**: Client-side filter: `slug == f"btc-updown-5m-{slot_ts}"` on every trade. Also: data-api returns IDENTICAL trades for both YES and NO token queries — do NOT force outcome_label from token loop.
**Rule**: Never trust API query params. Always verify with exploratory queries. Filter client-side.

### Bug #6: Synthetic Seed Data → Zscore Explosion
**Symptom**: PREDICTION_SANITY blocks all trades for ~25 minutes after restart.
**Root cause**: Seeding `_slot_history` with constant up_ratio=0.5 creates std=0 → zscore = inf. Random/synthetic values also violate no-synthetic-data principle.
**Fix**: `_fetch_seed_up_ratio()` fetches REAL up_ratio from data-api trades for each of 25 past slots (~3s total). Use epsilon=1e-6 (not 1e-8) and clamp zscores to [-5, 5].
**Rule**: No synthetic data in production. If a feature needs history, FETCH it.

### Bug #7: ob_mid Default Should Be 0.5, Not 0.0
**Symptom**: OB features biased when orderbook fetch fails.
**Root cause**: Default ob_mid=0.0 is maximally bearish. Neutral for a binary market is 0.5.
**Fix**: All OB ratio/mid defaults use 0.5. Drift/momentum defaults use 0.0. Spread default 0.02.
**Rule**: Default values must be semantically neutral, not numerically zero.

### Bug #8: Spot Buffer Staleness Hard Gate
**Symptom**: Model uses stale 4-hour-old spot data, predicts on garbage.
**Root cause**: Soft handling of stale spot buffer — used old data instead of blocking.
**Fix**: If spot buffer is >120s old (`BUFFER_STALE`), return zeros for all spot features. DataQualityGate layer 1 catches this.
**Rule**: Stale data is worse than no data. Hard-gate, don't soft-degrade.

### Bug #9: Feature Parity Drift (6 bugs, -$40 P&L)
**Symptom**: 42% live win rate vs 81% backtest.
**Root cause**: 6 feature computation mismatches between training and live:
1. `btc_size_disparity`: subtraction vs division
2. `btc_buy_ratio`: dollar-weighted vs count-based
3. `btc_dist_1k`: distance to nearest $1k vs floor
4. `btc_pre_*_ret`: obs-end price vs slot-start price
5. `btc_pre_1h_4h_ratio`: wrong price reference
6. `btc_up_w5_zscore`: per-window stats vs overall stats
**Fix**: Line-by-line audit of every feature computation.
**Rule**: Feature parity is sacred. Run full audit before every deploy.

### Bug #10: Audit Against Wrong Training Version

### Bug #12: Copytrade Balance Returns $0 (signature_type for proxy/safe wallets)
**Symptom**: Copytrade bot logs `Balance $0.00 below $1.0` despite wallet having $162 USDC.
**Root cause**: `py_clob_client.get_balance_allowance()` defaults to `signature_type=0` (EOA). For proxy/safe wallets where the funder address differs from the SDK-derived key address, sig_type=0 queries the wrong on-chain address and returns 0.
**Diagnosis**: SDK-derived address (`clob.get_address()`) != `POLY_SAFE_ADDRESS` → wallet is a proxy/safe, needs sig_type=1.
**Fix**: In `polymarket.py` shim, try `signature_type=1` (POLY_GNOSIS_SAFE) first, fall back to 0:
```python
for sig_type in [1, 0]:
    resp = self._clob.get_balance_allowance(
        BalanceAllowanceParams(asset_type=at, signature_type=sig_type))
    if int(resp.get("balance", 0)) > 0:
        return BalanceResult(balance=raw_balance)
### Bug #13: Copytrade FOK Orders Return order_id=? status=unknown
**Symptom**: All copytrade orders log `order_id=? status=unknown`, balance unchanged after orders.
**Root cause**: `py_clob_client.create_market_order()` returns a dict whose keys vary by SDK version. The shim expected `orderID` / `status` but the SDK may return `success` / `orderIds` / no status field. Additionally, FOK orders on stale/historical trades (seed cycle) don't fill because live prices have moved.
**Fix**: (1) Log raw SDK response for diagnosis. (2) Parse multiple key patterns: `orderID`, `orderIds[0]`, `id`. Derive status from `success` boolean when no `status` key. (3) Don't panic if seed-cycle FOK orders don't fill — it's expected for historical trades.
**Rule**: Always log raw SDK responses from third-party trading APIs. Response schemas change between versions without notice.

### Bug #10: Audit Against Wrong Training Version
**Symptom**: Subagent flags 3 "critical" feature mismatches that don't actually exist.
**Root cause**: Audit compared live code against `train_v15_modal.py` instead of `train_v19_modal.py`. The v15→v19 training scripts changed feature formulas (e.g., btc_buy_ratio changed from count-based to dollar-weighted in v17+). Live code matches v19 perfectly.
**Rule**: ALWAYS verify which training script version produced the current champion before auditing. Check `champion_meta.json` or `bundle["version"]` in the pkl. Never compare against an old training script.

### Bug #11: DATA GATE Sub-Window Coverage vs Data-API Lag

### Bug #14: pmdata.dev API Key Expiry → Silent Dataset Expansion Failure
**Symptom**: v20 training script reported "0 new markets found from 8027 candidates" — expansion appeared to work but found nothing.
**Root cause**: pmdata API key `sk-5uX...Ijko` expired. The API returns HTTP 403 (Cloudflare error 1010) or 401 `{"error":"API key is invalid or expired"}`. The v20 script's fetch loop treated 403 as "no data for this slug" instead of "auth failed", so it silently skipped all 8027 candidates without raising an error.
**Impact**: Trained v20 on identical 22k dataset as v19 but with 45 features instead of 40 → feature dilution → 0/3 gate.
**Fix**: (1) Test pmdata key with a known-good slug BEFORE starting batch fetch. (2) Distinguish 401/403 from 404 — 401/403 = auth failure (abort), 404 = no data (skip). (3) Renew key at pmdata.dev.
**Rule**: Always validate API credentials with a single test call before batch operations. Don't mask auth errors as "no data found".
**Symptom**: "only 2/6 sub-windows have ticks" blocks entire slots despite 1000+ ticks fetched.
**Root cause**: Data-api lag is ~120s. At t=179s (entry window start), available ticks cover t=0-59s = windows w0+w1 only (2/6). The 1000+ ticks are all concentrated in the first 60s. Gate requires min 3/6 windows.
**Expected behavior**: Bot enters later in the entry window (t=187s+) when lag allows 3+ windows. If data-api lag is unusually high (>140s), entire slot gets skipped — this is CORRECT protection.
**Rule**: Don't lower the sub-window gate to fix this. The gate exists because partial-window features (w2-w5 = 0.5 neutral) degrade model accuracy. Wait for natural lag to catch up.

---

## 3. TRADING RULES & CONFIGURATION

### Active Parameters (Testing Phase)
```
SLOT_DURATION   = 300s (5 min)
OBSERVE_SECS    = 180s (first 3 min: collect order flow)
ENTER_WINDOW    = (170s, 240s) — predict + trade
SETTLE_GRACE    = 60s after slot end
MIN_CONFIDENCE  = 0.60 (60%)
MIN_EDGE        = 0.10 (10% edge vs ask price)
MIN_EDGE_MID    = 0.05 (5% edge vs market mid)
TAKER_FEE       = 0.02 (2%)
SHARES          = 5 (CLOB minimum, testing phase)
ASK_RANGE       = [0.38, 0.90]
ASK_MID_DIVERGE = 0.20 max
BUFFER_STALE    = 120s
PRICE_MAX_AGE   = 15s (WS price freshness)
```

### Order Flow
1. t=0s: New 5-min slot starts
2. t=0-180s: Observe CLOB trades + Binance spot
3. t=170s: Entry window opens — fetch trades, build features, predict
4. t=170-240s: If confidence >60% AND edge >10% AND ask in range → place order
5. t=300s: Slot ends
6. t=360s: Settle — check resolution via Gamma API

### Effective Minimum Balance
Do NOT tell user "$1.50 is enough". CLOB enforces 5-share minimum:
- At ask=$0.69: 5 shares = $3.45. With 5% fee buffer = $3.62
- At ask=$0.90: 5 shares = $4.50. With buffer = $4.73
- **Real minimum: ~$5.00 USDC** to handle any ask in valid range

---

## 4. TRAINING PIPELINE

### Runbook: Train a New Model Version

```bash
# 1. Ensure Binance spot data is current (Binance blocks Modal US region)
cd /home/ubuntu/polymarket-btc-lab
python3 scripts/fetch_spot_full.py
modal volume put btc-local-data /tmp/binance_spot_full.parquet binance_spot_full.parquet

# 2. If adding new features requiring per-market API calls, pre-compute first
# Example: OB features (Stage 1)
modal run scripts/fetch_ob_features_modal.py

# 3. Create new training script (copy latest version)
cp scripts/train_v19_modal.py scripts/train_v20_modal.py
# Edit: update version string, champion baseline, feature changes

# 4. Run training on Modal
modal run scripts/train_v20_modal.py

# 5. Check output: promotion gate results
# Gate: 2/3 metrics must beat champion
# AUC > champion.wf_auc
# Brier < champion.wf_brier  (lower is better)
# Acc > champion.wf_acc

# 6. If promoted, verify on HuggingFace
# champion.pkl and champion_meta.json should be updated

# 7. Deploy (see Deployment section)
```

### Training Configuration
| Parameter | Value | Notes |
|-----------|-------|-------|
| Algorithm | LightGBM + CalibratedClassifierCV | Isotonic calibration, cv=3 |
| Calibration | ALWAYS isotonic | Sigmoid underfits at this sample size |
| Walk-forward | 5 folds, gap=5 | Never change (validated in v12) |
| Optuna trials | 150 | Diminishing returns past this |
| Features (v21) | Top 30 by ablation study | Was 40 in v19 (pruned 10 low-value features) |
| Modal resources | 8 CPU, 32GB RAM, 2h timeout | |
| Modal secret | `hf-token` (env: HF_TOKEN) | |

### Data Sources (Modal Volume `btc-local-data`)
| File | Content | Size |
|------|---------|------|
| `ticks_btc_full_clean.parquet` | 68.3M clean ticks, 22,237 markets | Main training data |
| `all_markets.csv` | 22,319 markets with slot_ts, slug, targets | Labels |
| `binance_spot_full.parquet` | 119k 1m BTCUSDT candles | Spot features |
| `ob_features_full.parquet` | Pre-computed L2 OB features per market | v19+ |
| `ob_progress.json` | Resume tracker for OB feature fetch | |

### Champion PKL Structure
```python
{
    "version": "v21-v21_30feat",
    "features": ["btc_inslot_ret", "ob_mid_drift", ...],  # 30 feature names
    "model": CalibratedClassifierCV,  # wraps LightGBM
    "wf_auc": 0.9002,
    "wf_brier": 0.1290,
    "wf_acc": 0.8134
}
```
**CRITICAL**: `model.feature_names_in_` contains generic `Column_0..Column_N`. Feature ORDER is determined by the `features` list — model maps by position, not name. If you reorder features, the model breaks silently.

### Champion Progression
v17 AUC=0.8925 → v18 AUC=0.8966 → v19 AUC=0.9000 → v20 AUC=0.8996 (NOT PROMOTED) → v21 AUC=0.9002 (PROMOTED, 30 features via ablation)

### Key Insight: Data > Features > Fewer Features
The biggest AUC gains came from dataset expansion, NOT feature engineering:
- v17: +0.04 AUC (601→7k samples) — **largest gain ever**
- v18: +0.004 AUC (7k→22k samples)
- v19: +0.003 AUC (L2 OB features, same data)
- v20: -0.0004 AUC (4 new features, same data → feature dilution)
- v21: +0.0002 AUC (feature PRUNING: 40→30 features, same data → reduced noise)
**Rule**: Prioritize dataset expansion over feature engineering. Adding features without more data hurts. PRUNING low-importance features can improve performance by reducing noise. Use ablation studies to validate pruning before deploying.

### Feature Categories (v21, 30 features)
- **CLOB flow** (10): up_ratio, w1/w2 sub-windows, buy_ratio, VWAP up/dn/spread, momentum, size_disparity, signal_conviction
- **Spot** (5): inslot_ret, pre_5m/30m/1h/4h_ret
- **OB** (7): ob_mid, ob_mid_drift, ob_weighted_imb, ob_imb_w0/w2, ob_imb_momentum, ob_ask_depth_5c
- **Cross-domain** (3): x_imb_x_ur, x_depth_x_momentum, x_ob_drift_x_inslot
- **History** (4): prev_slot_up_ratio_1/2/3/5
- **Temporal** (1): hour_x_up_ratio

### Features PRUNED (v19→v21, 10 removed)
Ablation study confirmed these 10 features added noise without improving AUC/Brier/Acc:
- `btc_dist_1k`, `btc_pre_1h_4h_ratio` (spot)
- `btc_up_ratio_zscore_5s`, `btc_up_ratio_zscore_20s` (zscores)
- `btc_up_w0` (sub-window — w1/w2 sufficient)
- `hour_cos`, `hour_x_tw_ur` (temporal)
- `ob_imb_w1`, `ob_total_depth` (OB)
- `prev_slot_up_ratio_4` (history)

Additionally, live_trader.py was pruned of ~74 dead feature computations (-137 lines):
- Removed: tw_up, vwap_trend, vwmom, tick_accel, vol_accel, momentum_vol_sync
- Removed: all zscore features, realized_vol, lag_outcomes, lag_streak
- Removed: calendar features (hour_sin/cos, dow_sin/cos)
- Removed: cross features x_spread_x_vol, x_fill_imb_x_buy
- Removed: counters n_ticks, vol_up, vol_dn, vol_ratio, avg_size
- Kept: intermediate computations needed by remaining features (e.g., up_ratio_stability for signal_conviction)

### Top 5 Features (v21)
1. `btc_inslot_ret` — BTC spot return during observation window
2. `ob_mid_drift` — orderbook mid price drift (open→close snapshot)
3. `btc_pre_5m_ret` — 5-min pre-slot spot return
4. `btc_vwap_up` — Up token VWAP
5. `x_ob_drift_x_inslot` — OB mid drift × BTC inslot return

### Two-Stage Modal Pattern (for expensive per-market features)
When features require per-market API calls (e.g., L2 orderbook data):
1. **Stage 1**: Pre-compute features → save to Modal Volume parquet
2. **Stage 2**: Training script loads pre-computed features, joins by market_id
- Markets missing data get neutral defaults (not excluded)
- Resume-safe: progress tracking + periodic flush

### Anti-Patterns in Training
- ❌ Fetch Binance spot inline from Modal (HTTP 451 geo-block)
- ❌ Use sigmoid calibration (underfits at this sample size)
- ❌ Use >5 WF folds (test sets too small)
- ❌ Use gap<5 with lag features (leakage risk)
- ❌ Ensemble LightGBM+LR (tested v7, hurt performance)
- ❌ Multi-crypto features ETH/SOL (noise, not signal)
- ❌ OB Down token features (best_bid_size always 0)
- ❌ Wrong Modal secret name (`pmdata-api-key`, NOT `pmdata-key`; env var = `PMDATA_API_KEY`)
- ❌ Fetching per-market API data inline during training (use two-stage pre-compute)
- ❌ Increasing TOP_N_FEATS without proportionally more data (dilution, v20 lesson)
- ❌ Assuming pmdata API key is valid — check with a test fetch BEFORE running 8000-slot expansion
- ❌ Dropping features without ablation study validation — always test multiple subsets and compare with walk-forward before deploying
- ✅ Feature pruning via ablation study (train_v21): test multiple feature subsets (40/35/30) in same run, compare all vs champion, promote the best that passes gate. Prefer fewer features when scores are equal. Never just drop features — validate with walk-forward first.
- ✅ Extract importances from champion.pkl: `model.calibrated_classifiers_[0].estimator.feature_importances_` (CalibratedClassifierCV wraps LightGBM). Cross-reference with live data quality to identify pruning candidates (low importance + bad live data = prime target).
- ✅ After promoting a pruned model, also prune the live feature computation code to remove dead computations. Keep intermediate values needed by remaining features (e.g., up_ratio_stability for signal_conviction).
- ✅ When pruning live_trader features, verify the model was trained with the pruned set (force_download from HF, check `bundle['features']` count and `model.n_features_in_`). HF cache can serve stale files.

---

## 5. DEPLOYMENT

### Runbook: Deploy to Fly.io

```bash
cd /home/ubuntu/polymarket-btc-lab/deploy

# Option A: Manual deploy
/home/ubuntu/.fly/bin/flyctl deploy --app polymarket-maker-mm

# Option B: GitHub Actions (auto-triggers on deploy/ changes)
cd /home/ubuntu/polymarket-btc-lab
git add -A && git commit -m "deploy: description" && git push
# Or manually: gh workflow run deploy.yml

# MANDATORY: Post-deploy verification
/home/ubuntu/.fly/bin/flyctl machines list -a polymarket-maker-mm
# → Verify state=started. If stopped:
/home/ubuntu/.fly/bin/flyctl machines start <machine_id> -a polymarket-maker-mm

# Wait for startup (model download + spot seed + warmup)
sleep 90

# Verify model version
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "Model loaded"
# → Check version, feature count, AUC match expected

# Verify predictions are sane
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "Prediction"
# → conf should NOT be 99%+ consistently, up_ratio should vary

# Verify no safety violations
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "FEATURE_SANITY\|PREDICTION_SANITY"
# → Should be EMPTY after warmup (first 3 slots)

# Byte-for-byte code verification (MD5)
/home/ubuntu/.fly/bin/flyctl ssh console -a polymarket-maker-mm -C "md5sum /app/live_trader.py /app/data_quality_gate.py"
md5sum deploy/live_trader.py deploy/data_quality_gate.py
# → Must match exactly
```

### Deploy Infrastructure
- **Dockerfile**: `deploy/Dockerfile` — `python:3.12-slim` + `libgomp1` (for LightGBM)
- **Files copied**: `live_trader.py`, `data_quality_gate.py`, `ws_manager.py`, `requirements.txt`
- **Model**: NOT bundled in image. Downloaded from HuggingFace at startup via HF_TOKEN secret
- **VM**: `performance-1x`, 2GB RAM, AMS region
- **Note**: Binance WS is geo-blocked (HTTP 451) in AMS. REST fallback poller handles this automatically.

### Fly.io Secrets (set via `flyctl secrets set`)
```
POLY_PRIVATE_KEY      — EOA private key
POLY_SAFE_ADDRESS     — Proxy wallet (0x362095...)
MM_BUILDER_KEY        — Builder API key
MM_BUILDER_SECRET     — Builder API secret
MM_BUILDER_PASSPHRASE — Builder API passphrase
HF_TOKEN              — HuggingFace access token
```

### Deploy Failure Modes

**Dockerfile missing COPY**: If you add a new .py import to live_trader.py, add `COPY <file>.py .` to Dockerfile. Missing = crash loop (10 restarts → machine stops). **Current required files**: `live_trader.py`, `data_quality_gate.py`, `ws_manager.py`, `requirements.txt`.

**Machine stopped after rolling deploy**: Fly rolling deploy FREQUENTLY leaves machine in `stopped` state (SIGINT during model loading race). This happens ~50% of the time. The deploy command reports success but the machine is dead. ALWAYS run `flyctl machines list` after deploy and `flyctl machines start <id>` if stopped. Machine ID can CHANGE between deploys (old machine replaced by new one) — never hardcode IDs.

**Worker bot auto-suspend**: Fly.io suspends machines with no inbound HTTP traffic after ~5-7 minutes. Both the BTC bot and copytrade bot are worker processes with no HTTP port. They get suspended even while actively polling/trading. After manual `fly machine start`, monitor that it stays running. If it keeps getting suspended, the machine may need `auto_stop_machines = false` in fly.toml or a dummy HTTP health check endpoint.

**Docker layer cache on Depot**: Fly's Depot builder caches layers aggressively. Code changes in COPY steps may appear cached if file hashes match previous builds. ALWAYS add a cache-buster ARG as the LAST step before CMD:
```dockerfile
ARG BUILD_DATE=unknown
RUN echo "Build: $BUILD_DATE"
```
Deploy with: `fly deploy --build-arg BUILD_DATE="$(date -u +%Y%m%dT%H%M%S)"`

**MANDATORY post-deploy sequence** (never skip):
```bash
/home/ubuntu/.fly/bin/flyctl status -a polymarket-maker-mm
# If STATE=stopped:
/home/ubuntu/.fly/bin/flyctl machine start <machine_id> -a polymarket-maker-mm
sleep 30
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "Model loaded\|Wallet balance"
```

**HuggingFace rate limit (429)**: Rapid CI+deploy runs exhaust HF limits. Wait 30s, rerun: `gh run rerun <id> --failed`.

**HuggingFace auth (401)**: GitHub Secret `HF_TOKEN` must match `~/.env`. Update: `gh secret set HF_TOKEN < <(grep -oP 'HF_TOKEN=\K.*' ~/.env)`

**Deploy must run from deploy/ directory**: `flyctl deploy` uses CWD as Docker build context. Running from repo root fails with "app does not have a Dockerfile or buildpacks configured" because the Dockerfile is at `deploy/Dockerfile`, not repo root. Always `cd /home/ubuntu/polymarket-btc-lab/deploy` before running `fly deploy`.

**Binance WS geo-blocked (HTTP 451)**: Fly.io Amsterdam region is blocked by Binance WebSocket. The REST API (`/api/v3/klines`) still works. The bot handles this via `_spot_rest_poll()` — a background thread polling REST every 30s. If the WS connects (e.g., in a non-blocked region), it takes priority via real-time updates. Both paths write to the same `_spot_buffers`. No action needed — the fallback is automatic.

**Machine replacement**: Rolling deploys can REPLACE machine (new ID). Old ID becomes invalid. Always re-run `flyctl status` to get current ID.

### CI/CD Workflows
- `ci.yml`: every push → unit tests + validate champion model from HF
- `deploy.yml`: push to `deploy/` or manual `workflow_dispatch` → Fly.io deploy
- GitHub Secrets needed: `HF_TOKEN`, `FLY_API_TOKEN`
- Note: model updates on HF alone don't trigger deploy. Use `gh workflow run deploy.yml` manually.

---

## 6. MONITORING

### Automated Monitoring
**Cron job** `btc-bot-monitor` (ID: `2ba5781a7d0a`) runs every 10 minutes:
- Executes `bash /home/ubuntu/polymarket-btc-lab/scripts/monitor_bot.sh`
- Parses last ~200 lines of Fly.io logs
- Reports to WhatsApp in Portuguese
- Checks: predictions, trades, fills, skips, errors, WS stability, balance

### Manual Health Check
```bash
# Quick status
/home/ubuntu/.fly/bin/flyctl status -a polymarket-maker-mm

# Live logs (streaming — default, runs indefinitely)
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm

# Recent logs (static snapshot — use --no-tail, NOT -n <number>; fly logs has no -n flag)
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | tail -200

# Full monitor script
bash /home/ubuntu/polymarket-btc-lab/scripts/monitor_bot.sh

# SSH into container
/home/ubuntu/.fly/bin/flyctl ssh console -a polymarket-maker-mm
```

### What Log Patterns Mean

| Pattern | Meaning | Action |
|---------|---------|--------|
| `Insufficient balance $X.XX < required $Y.YY` | Balance too low (need ~$5 USDC) | Deposit USDC |
| `diverges 0.XX from mid` | Ask/mid spread too wide | Check get_market_mid() uses CLOB /book |
| `ask $0.9XX outside [0.38, 0.90]` | Market priced in, no edge | Normal — efficient market |
| `get_ask_price FAILED all 3 attempts` | Returns $0.000, WS + HTTP both failed | Check CLOB WS health |
| `CLOB WS error: no close frame` | WS disconnect | Should auto-reconnect. If >5/window, investigate |
| `FEATURE_SANITY` after warmup | Feature computation bug | STOP. Investigate feature parity |
| `PREDICTION_SANITY` after warmup | Model outputting extreme values | Check for stale/corrupt features |
| `Skip — DATA_COMPLETENESS` | <50 ticks or stale spot | May be normal (low activity slot) |
| `Warming up...` | Cold start, first 3 slots | Normal after restart |
| `CLOB WS reconnect — invalidated N cached prices` | Price cache cleared on WS reconnect | Normal — forces HTTP fallback until fresh data |
| `WS-HEALTH [name] connected=... uptime=...` | WS health report (every ~5min) | Check disconnects, zombie kills |
| `SPOT-BUFFER age=Ns btc_candles=N` | Spot data freshness | age>120s or candles=0 = problem |
| `build_features OK (Nms) — M/N features non-zero` | Feature build timing, M/N counts MODEL features only (not all computed) | High zero count = data issue. Denominator must be `len(features)` from model bundle, NOT `len(feat)` dict (which includes unused spot/OB extras). Bug: counting dict values showed "50/54" instead of "29/30" — misleading. |
| `get_ask_price: $X.XXX (took Nms)` | Ask price fetch timing | >500ms = WS stale, using HTTP fallback |
| `Order placed` / `FILL` | Trade executed | Monitor outcome |
| `SETTLED WIN/LOSS` | Trade resolved | Track P&L |

### Alert Escalation
1. **Sanity/Feature violations > 0 (after warmup)**: IMMEDIATE — feature parity regression
2. **Bot crashed / machine stopped**: Check `flyctl status`, restart if needed
3. **Zero trades over 4+ hours**: Check if balance issue or all markets efficiently priced
4. **Win rate <40% over 20 trades**: DataQualityGate auto-pauses. Investigate features.
5. **WS disconnects >5 per window**: Network instability. May need backoff increase.

---

## 7. FEATURE PARITY AUDIT CHECKLIST

**Run BEFORE every deploy and every new model version.**

### Automated Audit Steps
```bash
cd /home/ubuntu/polymarket-btc-lab

# 1. Get feature list from current champion
python3 -c "
import pickle
with open('/tmp/champion.pkl','rb') as f: d=pickle.load(f)
for i,f in enumerate(d['features']): print(f'{i}: {f}')
"

# 2. For each feature, verify computation matches between:
#    scripts/train_v19_modal.py (tick_features_v19 / build_features_v19)
#    deploy/live_trader.py (build_features)
```

### Manual Checklist (17 items)

**PREREQUISITE**: Before starting audit, confirm which training script produced the current champion:
```bash
python3 -c "import pickle; d=pickle.load(open('/tmp/champion.pkl','rb')); print(d['version'])"
# Use scripts/train_v{VERSION}_modal.py as the reference. NEVER compare against old versions.
```
1. ☐ `btc_size_disparity`: SUBTRACTION (avg_up - avg_dn), NOT division
2. ☐ `btc_buy_ratio`: DOLLAR-WEIGHTED (sum size_usdc for BUY / total), NOT count
3. ☐ `btc_dist_1k`: `min(frac, 1-frac)` — distance to NEAREST $1k, NOT floor
4. ☐ `btc_pre_*_ret`: use `spot_at(slot_ts + OBS_SECS)` as px_now, NOT slot_ts
5. ☐ `btc_pre_1h_4h_ratio`: use obs-end price, NOT slot-start
6. ☐ `btc_up_w5_zscore`: use overall up_ratio mu20/sd20, NOT per-window stats
7. ☐ Feature ORDER in model matches feature ORDER in live prediction
8. ☐ Outcome labels: use API's `t.get("outcome", outcome_label)` — NOT forced
9. ☐ Zscore epsilon: 1e-6 (match training), clamp to [-5, 5]
10. ☐ buy_ratio denominator: total dollar volume (vol_up + vol_dn), never ~0
11. ☐ Cross-market filter: slug == `f"btc-updown-5m-{slot_ts}"` in fetch_inslot_trades
12. ☐ Zscore clip bounds: align with DataQualityGate ZSCORE_RANGE (-5, 5)
13. ☐ Seed history: `_fetch_seed_up_ratio()` uses REAL data, not constant/synthetic
14. ☐ Dynamic features: loops generate all indices (w0-w5, ratio_1-5, zscore_5s/10s/20s)
15. ☐ `get_market_mid()`: uses CLOB `/book` midpoints, NOT Gamma outcomePrices
16. ☐ OB temporal features: open/close snapshot caching, NOT single-snapshot 0.0 defaults
17. ☐ Data-API cache busting: `_t=int(time.time())` param on all data-api requests

### Dynamic Feature Audit Note
Some features are generated in loops and won't match literal grep:
- `btc_up_w0..w5` → `for i in range(6): sw[f"btc_up_w{i}"]`
- `prev_slot_up_ratio_1..5` → `for lag in range(1, 6): feat[f"prev_slot_up_ratio_{lag}"]`
- `btc_up_ratio_zscore_5s/10s/20s` → loop over `[(5, "5s"), (10, "10s"), (20, "20s")]`
Search for pattern root and verify loop covers all indices.

---

## 8. DataQualityGate — 5-Layer Circuit Breaker

File: `deploy/data_quality_gate.py`

### Layer 0: COLD START PROTECTION
- 3-slot warmup after restart
- Builds lag feature history before trading
- `gate.start_warmup(n_slots=3)` / `gate.is_warm()`

### Layer 1: DATA COMPLETENESS
- Minimum 50 ticks required
- Spot buffer must be <120s old (hard gate — returns zeros, not stale data)
- At least 3 of 6 sub-windows must have data
- `gate.check_data_completeness(ticks, spot_buffer_path, slot_ts)`

### Layer 2: FEATURE SANITY
- All features must be finite (not NaN/inf)
- Return features: [-0.05, 0.05] (5% move in 5min is extreme)
- Ratio features: [0.0, 1.0]
- Z-score features: [-5.0, 5.0]
- Absolute max: 1e6 for any feature
- `gate.check_feature_sanity(feat_dict, features_list)`

### Layer 3: PREDICTION SANITY
- Flags predictions >99% or <1% (likely feature corruption)
- Detects stale/repeated feature vectors
- `gate.check_prediction_sanity(prob_up, feat_dict)`

### Layer 4: EXECUTION GATE
- Ask price must be in [0.10, 0.95]
- Auto-pause if win rate <40% over last 20 trades
- `gate.check_execution_gate(ask_price, trades_list)`

---

## 9. OPERATIONAL RUNBOOKS

### Runbook: Bot Is Down
```bash
# 1. Check machine status
/home/ubuntu/.fly/bin/flyctl machines list -a polymarket-maker-mm

# 2. If stopped, start it
/home/ubuntu/.fly/bin/flyctl machines start <machine_id> -a polymarket-maker-mm

# 3. If crash-looping (10 restarts), fix the bug first, then:
cd /home/ubuntu/polymarket-btc-lab/deploy
/home/ubuntu/.fly/bin/flyctl deploy --app polymarket-maker-mm
# Then start if needed:
/home/ubuntu/.fly/bin/flyctl machines start <machine_id> -a polymarket-maker-mm

# 4. Verify recovery
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | tail -50
```

### Runbook: Feature Parity Regression
```bash
# 1. Identify which feature is failing
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "FEATURE_SANITY\|PREDICTION_SANITY"

# 2. Compare live feature values with expected ranges
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "feat="

# 3. Run full audit (see Section 7)

# 4. Fix in deploy/live_trader.py
# 5. Re-deploy (see Section 5)
```

### Runbook: Retrain Model
```bash
# 1. Update data if needed
python3 scripts/fetch_spot_full.py
modal volume put btc-local-data /tmp/binance_spot_full.parquet binance_spot_full.parquet

# 2. Create new version script
cp scripts/train_v19_modal.py scripts/train_v20_modal.py
# Edit version, champion baseline, features

# 3. Train
modal run scripts/train_v20_modal.py

# 4. Check promotion gate output
# 5. If promoted, deploy new model
gh workflow run deploy.yml
# Or: cd deploy && /home/ubuntu/.fly/bin/flyctl deploy --app polymarket-maker-mm
```

### Runbook: Expand Training Dataset
```bash
# 1. Fetch new ticks from pmdata.dev
python3 scripts/fetch_pmdata_ticks.py

# 2. Upload to Modal Volume
modal volume put btc-local-data /tmp/new_ticks.parquet ticks_btc_full_clean.parquet

# 3. Update market labels
modal volume put btc-local-data /tmp/all_markets.csv all_markets.csv

# 4. If OB features needed for new markets:
modal run scripts/fetch_ob_features_modal.py

# 5. Retrain (see above)
```

### Runbook: Check Wallet Balance
```bash
# From logs
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "balance"

# Direct check (requires bot running)
/home/ubuntu/.fly/bin/flyctl ssh console -a polymarket-maker-mm -C "python3 -c \"
from polymarket import SecureClient
from polymarket.auth import BuilderApiKey
import os
c = SecureClient.create(
    private_key=os.environ['POLY_PRIVATE_KEY'],
    wallet=os.environ['POLY_SAFE_ADDRESS'],
    api_key=BuilderApiKey(key=os.environ['MM_BUILDER_KEY'],
                          secret=os.environ['MM_BUILDER_SECRET'],
                          passphrase=os.environ['MM_BUILDER_PASSPHRASE']))
b = c.get_balance_allowance(asset_type='COLLATERAL')
print(f'Balance: \${float(b.balance)/1e6:.2f} USDC')
\""
```

### Runbook: Force Model Update Without Code Change
```bash
# Model is downloaded from HF at startup. Restart the machine:
/home/ubuntu/.fly/bin/flyctl machines list -a polymarket-maker-mm
/home/ubuntu/.fly/bin/flyctl machines restart <machine_id> -a polymarket-maker-mm
# Verify new model loaded:
sleep 60
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "Model loaded"
```

---

## 10. QUANT PRINCIPLES (Always Apply)

1. **Feature parity is sacred.** Every feature in training MUST be computed identically live. No approximations. A single mismatch silently destroys predictions.

2. **Data quality over quantity.** 1000 clean ticks beat 3000 contaminated ones. Always verify data provenance.

3. **Never trust APIs blindly.** The Polymarket data-api ignores query params server-side. Always verify with exploratory queries.

4. **No synthetic data in production.** If a feature needs historical data, FETCH it. Never simulate or randomize.

5. **Every prediction must be explainable.** If the model says 99%, ask: what feature drives this? Is that feature receiving clean data?

6. **Monitor P&L relentlessly.** Feature bugs are silent killers — the model still predicts, just wrongly.

7. **Deploy is not done until verified.** Push → Deploy → Check logs → Verify features → Confirm sane predictions → DONE.

8. **Default values must be semantically neutral.** Returns default to 0.0, ratios to 0.5, z-scores to 0.0. Never use 0.0 for everything.

---

## 11. OB FEATURES REFERENCE

### Neutral Defaults (when OB fetch fails or no data)
| Feature | Default | Rationale |
|---------|---------|-----------|
| ob_mid | 0.5 | Neutral for binary market |
| ob_spread | 0.02 | Typical market spread |
| ob_imbalance | 0.0 | No directional bias |
| ob_depth_ratio | 1.0 | Balanced depth |
| ob_bid/ask_depth_5c | 0.5 | Equal depth |
| ob_total_depth | 1000.0 | Typical total |
| ob_weighted_imb | 0.0 | No bias |
| ob_mid_drift | 0.0 | No movement |
| ob_imb_momentum | 0.0 | No change |
| ob_imb_w0/w1/w2 | 0.0 | Neutral windows |
| ob_pc_up_ratio | 0.5 | Equal up/down |
| ob_pc_volatility | 0.0 | No vol |
| ob_fill_imbalance | 0.0 | Balanced fills |
| Cross-domain (x_*) | 0.0 | No signal |

### Sanity Probe Values
- **Bullish**: imbalance=+0.3, mid_drift=+0.02, depth_ratio=1.3, fill_imbalance=+0.2
- **Bearish**: imbalance=-0.3, mid_drift=-0.02, depth_ratio=0.7, fill_imbalance=-0.2
- Return features: neutral = 0.0 (NOT 0.5)
- Ratio features: neutral = 0.5
- Z-scores: neutral = 0.0

---

## 12. REFERENCE FILES

### Skill References
| File | Content |
|------|---------|
| `references/market-mid-gamma-bug-2026-06-04.md` | Gamma stale prices, CLOB book mid fix |
| `references/3-critical-bugs-2026-06-04-session2.md` | OB temporal, CDN cache, WS keepalive |
| `references/feature-parity-bugs-2026-06-04.md` | 6-bug post-mortem (-$40 P&L) |
| `references/cross-market-contamination-2026-06-04.md` | 67% trade contamination |
| `references/polymarket-data-api-token-sharing.md` | 3 surprising API behaviors |
| `references/v19-ob-features-design.md` | L2 OB feature definitions |
| `references/v19-deployment-audit-2026-06-04.md` | Pre/post-deploy audit template |
| `references/full-audit-protocol-2026-06-04.md` | 3-layer audit protocol |
| `references/audit-session-2026-06-04-v19.md` | v19 deploy audit: 7 bugs fixed, WS health, false alarm lesson |
| `references/copytrade-bot-v9-multi-wallet.md` | Copytrade bot v9.0: multi-wallet arch, SecureClient shim, Fly.io deploy |
| `references/v20-training-attempt-2026-06-05.md` | v20 result: 0/3, pmdata key expired, feature dilution lesson |
| `references/bot-audit-2026-06-05.md` | Full audit: Gate 4 not integrated, edge too high, WS cache ineffective |
| `references/ws-manager-refactor-2026-06-05.md` | WS infra refactor: ws_manager.py design, zombie detection, backoff, 40 tests |
| `references/polymarket-clob-ws-stability-2026-06-05.md` | CLOB WS drops research: code 1006, double-ping root cause, ping_interval=None fix, stale price protection stack |
| `references/live-trading-perf-audit-2026-06-05.md` | Latency audit + data quality fixes: parallel fetches, progressive polling, 6 dead features revived |
| `references/v21-feature-pruning-2026-06-05.md` | v21 ablation study: feature importances from champion, pruning rationale, 40/35/30 variants |
| `references/v21-live-deployment-2026-06-05.md` | v21 live deployment: active zombie fix, feature pruning, count logging bug, copytrade diagnosis |

### Wiki (in repo: `docs/wiki/`)
| Page | Content |
|------|---------|
| `00-architecture.md` | System architecture, data flow, costs |
| `01-data-pipeline.md` | Data sources, pmdata API, dataset expansion |
| `02-feature-engineering.md` | All features, parity requirements |
| `03-training-pipeline.md` | Training steps, Modal config, Optuna |
| `04-model-evaluation.md` | Metrics, champion progression |
| `05-deployment.md` | CI/CD, Fly.io, GitHub Actions |
| `06-live-trading.md` | Trading strategy, live_trader architecture |
| `08-troubleshooting.md` | Common issues and fixes |
| `09-anti-patterns.md` | Everything tried and failed |
| `10-roadmap.md` | Planned improvements |

### Git Workflow
```bash
cd /home/ubuntu/polymarket-btc-lab
git add -A && git commit -m "description" && git push
```
Pushes to `deploy/` auto-trigger Fly.io deployment via GitHub Actions.

---

## 13. END-TO-END WORKFLOW: Training → Live

### Complete Version Release Workflow

When releasing a new model version (e.g., v22), follow this complete checklist:

#### Phase 1: Data Preparation
```bash
cd /home/ubuntu/polymarket-btc-lab

# 1a. Update Binance spot data (MUST run locally — Binance blocks Modal US)
python3 scripts/fetch_spot_full.py
modal volume put btc-local-data /tmp/binance_spot_full.parquet binance_spot_full.parquet

# 1b. If expanding CLOB ticks dataset (requires pmdata API key)
# First test key: curl -H "X-Api-Key: $PMDATA_KEY" https://pmdata.dev/api/...
modal run scripts/fetch_ticks_modal.py

# 1c. If adding OB features for new markets
modal run scripts/fetch_ob_features_modal.py

# 1d. Update market labels
modal volume put btc-local-data /tmp/all_markets.csv all_markets.csv
```

#### Phase 2: Training
```bash
# 2a. Create new training script (copy latest)
cp scripts/train_v21_modal.py scripts/train_v22_modal.py
# Edit: version string, feature changes, champion baseline

# 2b. Run training on Modal (~40 min)
modal run scripts/train_v22_modal.py

# 2c. Check promotion gate output in logs:
# Gate: 2/3 metrics must beat champion
# If PROMOTED → champion.pkl + champion_meta.json uploaded to HF
# If NOT PROMOTED → stop here, analyze why
```

#### Phase 3: Live Code Update (IF features changed)
```bash
# 3a. Compare new vs old feature list
python3 -c "
import pickle
from huggingface_hub import hf_hub_download
path = hf_hub_download('artbreguez/polymarket-btc-model', 'champion.pkl', force_download=True, token='...')
with open(path, 'rb') as f:
    bundle = pickle.load(f)
print(f'Features ({len(bundle[\"features\"])}):')
for f in bundle['features']: print(f'  {f}')
"

# 3b. If features changed: update deploy/live_trader.py build_features()
# - Add computation for new features
# - Remove computation for pruned features (keep intermediaries!)
# - Verify neutral defaults for new features

# 3c. Update non-zero feature count logging to use model feature list:
#   sum(1 for f in features if feat.get(f, 0.0) != 0.0), len(features)
#   NOT: sum(1 for v in feat.values() if v != 0.0), len(feat)

# 3d. Run tests
python -m pytest tests/ -v
```

#### Phase 4: Deploy
```bash
# 4a. Deploy to Fly.io (MUST run from deploy/ directory)
cd /home/ubuntu/polymarket-btc-lab/deploy
/home/ubuntu/.fly/bin/flyctl deploy --app polymarket-maker-mm --remote-only

# 4b. If just model update (no code changes), restart instead:
/home/ubuntu/.fly/bin/flyctl machines list -a polymarket-maker-mm
/home/ubuntu/.fly/bin/flyctl machines restart <machine_id> -a polymarket-maker-mm

# 4c. MANDATORY post-deploy verification (wait 30-60s first)
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "Model loaded"
# → Verify: correct version, feature count, AUC
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "build_features OK"
# → Verify: X/N features non-zero (N = model features, NOT dict size)
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep -E "disconnect|zombie|error"
# → Should be empty (no errors)
```

#### Phase 5: Documentation Update (NEVER SKIP)
```bash
# 5a. Update repo docs
# - README.md: version, metrics, feature count
# - docs/EXPERIMENTS.md: add new version entry
# - docs/wiki/04-model-evaluation.md: champion progression table
# - docs/wiki/README.md: current state section
# - feature_definitions.json: if features changed
# - training_config.json: if hyperparams changed

# 5b. Update HuggingFace docs
# Upload: README.md, EXPERIMENTS.md, feature_definitions.json, training_config.json

# 5c. Update HuggingFace dataset (if data changed)
# Upload new data files to data/ in the model repo

# 5d. Update this skill
# hermes skill_manage(action='patch', name='polymarket-btc-pipeline', ...)
# - Quick Reference table (version, metrics)
# - Feature categories section
# - Champion progression line
# - Training script references
# - Roadmap items

# 5e. Commit and push everything
cd /home/ubuntu/polymarket-btc-lab
git add -A && git commit -m "v22: description" && git push
```

#### Phase 6: Monitoring (first 30 min)
```bash
# 6a. Watch for feature/prediction sanity violations
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep -E "SANITY|ERROR|FAIL"

# 6b. Verify predictions are reasonable (not all 99% or all 50%)
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "Prediction"

# 6c. Check WS stability
/home/ubuntu/.fly/bin/flyctl logs -a polymarket-maker-mm --no-tail | grep "WS-HEALTH"
# → disconnects=0, zombies=0 is ideal

# 6d. If issues found: roll back by restarting with old model on HF
# or re-deploy previous commit
```

### Quick Decision Tree: What Changed?

```
Only hyperparameters (same features)?
  → Phase 2 + 4b (restart) + 5
  → No code changes needed

Features added?
  → Phase 2 + 3 (add to build_features) + 4a (full deploy) + 5

Features removed/pruned?
  → Phase 2 + 3 (remove from build_features, keep intermediaries) + 4a + 5

Dataset expanded?
  → Phase 1 + 2 + 4b (restart) + 5

Just monitoring/no changes?
  → Phase 6 only
```

---

## 14. IMPROVEMENT ROADMAP

| Priority | Item | Status |
|----------|------|--------|
| 1 | Daily data expansion cron | TODO |
| 2 | Rolling retraining automation | TODO |
| 3 | Regime detection (vol-based thresholds) | TODO |
| 4 | Multi-snapshot OB in live | DONE (2026-06-04) |
| 5 | Expand dataset (Feb 15 → Mar 13 gap, ~16% more) | TODO |
| 6 | CLOB book mid instead of Gamma | DONE (2026-06-04) |
| 7 | Adaptive stake sizing (exactly 5 shares × ask) | TODO |
| 8 | Exponential backoff on WS reconnect + zombie detection | DONE (2026-06-05, ws_manager.py) |
| 9 | HuggingFace dataset artifacts + model card | DONE (2026-06-04) |
| 10 | Binance spot re-seed on WS reconnect (gap protection) | DONE (REST fallback poller every 30s + WS manager reconnect) |
| 11 | Copytrade bot multi-wallet (YatSen + beachboy4) | DONE → REPLACED (2026-06-05) |
| 12 | Copytrade balance fix: signature_type=1 for proxy/safe wallets | DONE (2026-06-05) |
| 13 | Copytrade wallet swap: phdcapital + pako → REPLACED by Respectful-Clan | DONE (Respectful-Clan is current sole target, burst trader, last active 2026-06-01, balance $197.81) |
| 14 | Copytrade: validate order fill parsing with real-time target trade | TODO |
| 15 | Weather market traders: negligible on Polymarket (no dedicated traders found) | WONTFIX |
| 16 | v20 training: dataset expansion + new features | DONE — NOT PROMOTED (pmdata key expired, feature dilution) |
| 17 | Renew pmdata.dev API key for dataset expansion | BLOCKED (user action) |
| 18 | Lower MIN_EDGE to 0.05 so bot actually trades | TODO (safe with v19 calibration) |
| 19 | Integrate Gate 4 (execution/win-rate circuit breaker) into live_trader.py | TODO |
| 20 | Upload training data to HuggingFace data/ | DONE (2026-06-05) |
| 21 | Remove monitoring crons (btc-bot-monitor, btc-repo-autopush) | DONE (2026-06-05) |
| 22 | ~~v21: btc_vol_accel + same TOP_N_FEATS=40~~ → v21 is feature pruning ablation | SUPERSEDED |
| 23 | WS reconnect cache invalidation (stale price protection layer 2) | DONE (2026-06-05) |
| 24 | Startup diagnostics: package versions, config values, WS health logging | DONE (2026-06-05) |
| 25 | Debug instrumentation: feature build timing, ask price timing, spot buffer freshness | DONE (2026-06-05) |
| 26 | Fix CLOB WS double-ping: ping_interval=None (py-clob-client issue #82) | DONE (2026-06-05) |
| 27 | Parallel trade fetch (yes+no tokens via ThreadPoolExecutor) | DONE (2026-06-05) |
| 28 | Parallel price fetch (get_ask_price + get_market_mid) | DONE (2026-06-05) |
| 29 | Progressive fill polling (1,2,3,4,5,5s = 20s vs old 30s) | DONE (2026-06-05) |
| 30 | Fix 6 dead features: btc_pre_*_vol (5) + ob_depth_change (1) | DONE (2026-06-05) |
| 31 | Multi-snapshot OB polling during observation window (fix ob_imb_w1 interpolation) | TODO |
| 32 | v21: Feature pruning ablation study (40/35/30 features) | DONE — PROMOTED (2026-06-05). 30 features, AUC=0.9002, 3/3 gate. live_trader pruned -137 lines. |
| 33 | Prune live_trader.py to match v21's 30 features (remove dead code) | DONE (2026-06-05). Removed ~74 unused feature computations. |

---

## 15. COPYTRADE BOT DIAGNOSTICS

### Quick Reference
| Resource | Value |
|----------|-------|
| Repo | `/home/ubuntu/polymarket-copytrade` |
| Fly.io app | `polymarket-copytrade-amber-woodland-5363` |
| Wallet | `0xd0E6f59F7dE8Ba2DfA1289C46Ab4809538974cBb` |
| Current target | Respectful-Clan (`0x6e1d5040d0ac73709b0621f620d2a60b80d2d0fa`) |
| Target analysis | `TARGET_WALLET.md` in copytrade repo |

### Runbook: Copytrade Not Making Trades

**Symptom**: Logs show only "Balance: $X | checking activity..." + "Open positions: N markets (blocking re-entry)" with zero "Skip" or "NEW trade" lines.

**Diagnosis steps** (in order):
1. **Check target activity**: The most common cause is the target simply hasn't traded recently.
   ```bash
   curl -s "https://data-api.polymarket.com/activity?user=<TARGET_WALLET>&limit=5&type=TRADE" | python3 -m json.tool
   ```
   Convert timestamp to check recency: `datetime.fromtimestamp(ts, tz=timezone.utc)`

2. **Check if all trades are already seen**: If the loop processes activities but all pass `trade_seen()`, no "Skip" or "NEW" lines are logged — only "checking activity" appears. This is the EXPECTED behavior when the target hasn't made new trades since last seen.

3. **Check open positions blocking re-entry**: 9+ open positions doesn't mean the bot is broken — it means we already hold those markets and the bot correctly won't re-enter.
   ```bash
   curl -s "https://data-api.polymarket.com/positions?user=<COPYTRADE_WALLET>&limit=100" | python3 -c "
   import sys, json
   data = json.load(sys.stdin)
   positions = data if isinstance(data, list) else data.get('data', [])
   for p in positions:
       size = float(p.get('size') or 0)
       cur_price = float(p.get('curPrice') or 0)
       if size > 0.001 and cur_price > 0.001:
           print(f'OPEN | size={size:.3f} | price={cur_price:.3f} | {p.get(\"title\",\"\")[:60]}')
   "
   ```

4. **Check for errors in startup/seed**: Startup logs may be rotated off Fly.io. If the bot recently restarted, errors during `seed_seen_trades()` could cause all historical trades to be left unseen (triggering a flood of old copies) or all to be marked seen (blocking new copies in the same markets).

**Key insight**: The current target (Respectful-Clan) is a **burst trader** — operates in intensive 2-3h sessions every few days, not continuously. Days without trades are NORMAL. The bot is designed to wait patiently and copy when the target becomes active.

### Copytrade Anti-Patterns
- ❌ Assuming "no trades" = bug. Check target activity FIRST.
- ❌ Lowering `already_open` check to "fix" no trades — it prevents doubling into the same market.
- ❌ Restarting the bot to "fix" no trades — restart clears the SQLite DB and re-runs seed, which may cause duplicate copies of historical trades.

### Testing Phase → Production Transition
Currently shares are hardcoded to CLOB minimum of 5. To go production:
```python
# Current (testing):
shares = 5.0
# Production:
shares = max(5.0, round(STAKE_USDC / ask_price, 2))
```
