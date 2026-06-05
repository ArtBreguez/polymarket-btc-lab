# Copytrade Bot v9.0 — Multi-Wallet Architecture

## Quick Reference

| Resource | Value |
|----------|-------|
| Repo | `/home/ubuntu/polymarket-copytrade` (GitHub: `ArtBreguez/polymarket-copytrade`) |
| Fly.io app | `polymarket-copytrade-amber-woodland-5363` (AMS region) |
| Machine ID | Changes on deploy — always check `fly status` |
| Version | v9.0 (multi-wallet) |
| Wallet | `0xd0E6f59F7dE8Ba2DfA1289C46Ab4809538974cBb` (POLY_SAFE_ADDRESS) |
| Proxy (SDK derived) | `0x8d4ad7Fbc36D47Da99a7ff497bF55571c5906e92` |
| Storage | MongoDB (primary) + SQLite fallback |

## Architecture

Copy-trade bot monitors target wallets on Polymarket, detects new BUY trades, and mirrors them at minimum size ($1-3 per trade).

```
[Target Wallets]         [Polymarket APIs]        [Our Wallet]
 phdcapital (0x3b7e...)  →   data-api /activity  →   place_order (FOK)
 pako (0x71ed...)            CLOB /price             → MongoDB trade log
                             Gamma /markets
```

### Key Files
| File | Purpose |
|------|---------|
| `bot.py` | Main copytrade bot (v9.0, ~800 lines) |
| `polymarket.py` | SecureClient shim wrapping py-clob-client |
| `monitor.py` | Health check for Hermes cron monitoring |
| `Dockerfile` | python:3.12-slim based container |
| `fly.toml` | Fly.io config (shared-cpu-1x, 512MB) |
| `requirements.txt` | aiohttp, python-dotenv, py-clob-client, pymongo |

## Target Wallets (current — 2026-06-05)

| Wallet | Username | Focus | PnL | Why |
|--------|----------|-------|-----|-----|
| `0x3b7ed1242417f4b8f6992b5dd53aa9415a2c23eb` | phdcapital | Crypto + finance | $585, ROI 25% | 60% crypto, 0% both_sides, very active |
| `0x71edffd0d70a1da823ff07a3c6fc81457294d338` | pako | Crypto + macro | $567K, ROI 5.22x | 27% crypto + Fed/macro, 0% both_sides |

### Previous wallets (removed 2026-06-05)
- YatSen (0x5bff...) — politics/economics/culture, $2.3M PnL
- beachboy4 (0xc2e7...) — sports, $5.1M PnL

**Selection criteria**: Low `both_pct` (not a market maker), high PnL, fundamentalist (not HFT), active recently, focus on crypto/finance markets.

### Wallet Research Methodology
To find fundamentalist wallets from the Polymarket leaderboard:
1. Dashboard at `https://artbreguez.github.io/polymarket-lab/research.html` has `LOW_FREQ_DATA` JS variable with pre-analyzed traders
2. Filter by `cat_breakdown` for desired categories (crypto_lt, finance, politics, sports, etc.)
3. Filter OUT market makers: `both_pct > 15%` = likely MM, `tpd > 50` = likely HFT
4. Verify via data-api: `GET /activity?user=WALLET&limit=100&type=TRADE`
5. Check positions: `GET /positions?user=WALLET&limit=200`
6. Key metrics: PnL, ROI, trades_per_day, avg_size, both_pct, last_trade_ts

## Multi-Wallet Implementation

- `TARGET_WALLETS` env var: comma-separated wallet addresses
- Backward compat: falls back to `TARGET_WALLET` (single) if `TARGET_WALLETS` not set
- Each activity record tagged with `_source_wallet` for attribution in logs/DB
- All wallets polled each cycle, results merged + sorted by timestamp
- Dedup by `conditionId + outcome` — one copy per market side ever, regardless of which wallet triggered it

## SecureClient Shim (`polymarket.py`)

The original `polymarket` SDK package that provided `SecureClient` is unavailable. A thin shim module wraps `py_clob_client.ClobClient`:

```python
from polymarket import SecureClient

client = SecureClient.create(private_key="0x...", wallet="0x...")
balance = client.get_balance_allowance(asset_type="COLLATERAL")
# balance.balance is raw micro-USDC (divide by 1e6 for dollars)

result = client.place_market_order(
    token_id="...", side="BUY", amount=Decimal("1.50"), order_type="FOK"
)
# result.order_id, result.status
```

Key details:
- `get_balance_allowance()` returns `BalanceResult(balance=int)` — micro-USDC raw integer
- `place_market_order()` accepts Decimal amount, converts to float for py-clob-client
- `create()` calls `clob.create_or_derive_api_creds()` + `clob.set_api_creds()` for L2 auth
- CLOB host hardcoded to `https://clob.polymarket.com`, chain_id=137 (Polygon)

### Order Response Parsing (Bug: order_id=? status=unknown)

`py_clob_client.create_market_order()` returns a dict whose keys vary by SDK version and order outcome. Known response shapes:
- `{"orderID": "0xabc...", "status": "matched"}` — filled
- `{"success": true, "errorMsg": "", "orderID": "0xabc..."}` — success without status field
- `{"orderIds": ["0xabc..."], ...}` — batch format
- Signed order object (not a dict) — rare, for pre-signed orders

The shim logs the raw SDK response (`log.info(f"SDK raw response: {resp}")`) to diagnose parsing failures. Parse order:
```python
oid = resp.get("orderID") or resp.get("orderIds", [None])[0] or resp.get("id") or "?"
status = resp.get("status", "")
if not status:
    status = "matched" if resp.get("success") else "submitted"
```

**Pitfall**: If balance doesn't change after ORDER BUY logs, the FOK orders are likely not filling. This is normal for stale/historical trades where the live price has moved. Real-time trades from active target wallets have much better fill rates.

**Pitfall**: `fly ssh console -C` does NOT support shell redirections (`<`, `>`, `|`, `&&`). The `-C` argument is passed as a single exec command, not interpreted by a shell. Use a single `python3 -c "..."` command or write a script file first.

### CRITICAL BUG: Balance signature_type for proxy/safe wallets

**Symptom**: `get_balance_allowance()` returns `balance=0` even when the wallet holds USDC.

**Root cause**: `py_clob_client` defaults to `signature_type=0` (EOA wallet). For proxy/safe wallets where funder != SDK-derived address, the balance query hits the wrong address. The SDK derives a proxy address (`0x8d4ad7Fb...`) from the private key, but the actual USDC sits in the safe/funder address (`0xd0E6f59F...`).

**Fix**: Use `signature_type=1` (POLY_GNOSIS_SAFE) in `BalanceAllowanceParams`:
```python
# Try signature_type=1 (proxy/safe) first, fallback to 0 (EOA)
for sig_type in [1, 0]:
    resp = self._clob.get_balance_allowance(
        BalanceAllowanceParams(asset_type=at, signature_type=sig_type)
    )
    raw_balance = int(resp.get("balance", 0))
    if raw_balance > 0:
        return BalanceResult(balance=raw_balance)
```

**Diagnosis**: Run inside the container to determine which sig_type works:
```python
for sig_type in [0, 1, 2]:
    resp = clob.get_balance_allowance(
        BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=sig_type)
    )
    print(f'sig_type={sig_type}: ${int(resp.get("balance","0"))/1e6:.2f}')
```

**Rule**: Always try signature_type=1 first for proxy wallets. If the SDK-derived address differs from POLY_SAFE_ADDRESS, the wallet is a proxy/safe and needs sig_type=1.

## Fly.io Secrets

```
TARGET_WALLETS        — comma-separated wallet addresses
POLY_PRIVATE_KEY      — EOA private key
POLY_SAFE_ADDRESS     — Proxy wallet (0xd0E6f59F...)
MONGO_URI             — MongoDB connection string
MONGO_DB              — Database name (default: copytrade)
INFURA_WS_URL         — Optional Polygon WS (empty = poll fallback)
MAX_ORDER_SIZE_USD    — Max spend per trade
MIN_BALANCE_USD       — Pause threshold
MIN_COPY_PRICE        — Lower price bound (default 0.03)
MAX_COPY_PRICE        — Upper price bound (default 0.93)
CHAIN_ID              — 137 (Polygon)
RPC_URL               — Polygon RPC URL
```

## Fly.io Worker Bot Pitfalls

1. **Auto-suspend**: Fly.io suspends idle machines (no HTTP traffic). Worker bots with no HTTP port get suspended after ~5-7 minutes of "no requests". The bot keeps running in cycles but Fly sees no inbound connections.

2. **Deploy leaves machine stopped**: `fly deploy` with rolling strategy frequently results in machine in `stopped` state. ALWAYS run `fly machine start <id>` after deploy.

3. **Image cache**: Depot builder caches Docker layers aggressively. Add a `BUILD_DATE` ARG to bust cache:
   ```dockerfile
   ARG BUILD_DATE=unknown
   RUN echo "Build: $BUILD_DATE"
   ```
   Deploy with: `fly deploy --build-arg BUILD_DATE="$(date -u +%Y%m%dT%H%M%S)"`

4. **Machine ID changes**: Deploy can replace machine (new ID). Never hardcode IDs.

5. **Swap wallets without redeploy**: Just update the secret, stop, and start:
   ```bash
   fly secrets set TARGET_WALLETS="0xaaa...,0xbbb..." -a polymarket-copytrade-amber-woodland-5363
   fly machine stop <id> -a polymarket-copytrade-amber-woodland-5363
   fly machine start <id> -a polymarket-copytrade-amber-woodland-5363
   ```

## Deploy Command

```bash
cd /home/ubuntu/polymarket-copytrade
export PATH="$HOME/.fly/bin:$PATH"
fly deploy --strategy immediate --wait-timeout 120 --build-arg BUILD_DATE="$(date -u +%Y%m%dT%H%M%S)"
# ALWAYS verify + start:
fly status -a polymarket-copytrade-amber-woodland-5363
fly machine start <id> -a polymarket-copytrade-amber-woodland-5363
# Validate logs:
sleep 15
fly logs -a polymarket-copytrade-amber-woodland-5363 --no-tail | tail -30
# Confirm balance reads correctly (should NOT be $0.00):
fly logs ... | grep "Balance:"
```

## Trading Logic

1. Poll all target wallets for recent activity (50 trades each)
2. For each BUY trade not yet seen:
   - Check dedup (conditionId + outcome)
   - Check we don't already hold that market
   - Check daily budget
   - Check market expiry (>6h remaining)
   - Check live price in copy range (0.03–0.93)
   - Place FOK market BUY order
3. Track in MongoDB/SQLite

## Key Differences from BTC Bot

| Aspect | BTC Bot | Copytrade Bot |
|--------|---------|---------------|
| Strategy | ML prediction (LightGBM) | Mirror other wallets |
| Markets | BTC 5-min up/down only | Any market target trades |
| Data | Binance + CLOB + OB | Only Polymarket data-api |
| Sizing | 5 shares (testing) | $1-3 per trade (compute_spend) |
| Fly.io VM | performance-1x, 2GB | shared-cpu-1x, 512MB |
| Model | HuggingFace champion.pkl | None (rule-based) |
