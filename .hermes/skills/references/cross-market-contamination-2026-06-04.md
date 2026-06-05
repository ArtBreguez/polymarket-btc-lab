# Cross-Market Trade Contamination Bug (2026-06-04)

## Summary
The Polymarket data-api returns trades from ALL markets sharing the same token ID,
not just the requested BTC 5-min market. This caused ~24% of trades used for feature
computation to come from completely unrelated markets.

## Root Cause
Polymarket reuses CLOB token IDs across multiple markets. A single token ID can appear in:
- `btc-updown-5m-{slot}` (our target)
- `eth-updown-15m-{slot}` (Ethereum markets!)
- `sol-updown-5m-{slot}` (Solana)
- `bnb-updown-5m-{slot}` (Binance Coin)
- `bitcoin-up-or-down-on-june-4-2026` (daily Bitcoin)
- `bitcoin-above-64k-on-june-4-2026-539` (threshold markets)
- `2026-balance-of-power-r-senate-r-house-537` (completely unrelated!)
- `elon-musk-of-tweets-may-29-june-5-240-259` (completely unrelated!)

The `data-api /trades?asset=<token_id>` endpoint returns all trades for that token
across ALL markets. Neither `slug` nor `conditionId` query parameters filter results
(server-side filtering appears to be ignored).

## Impact on Features
With ~24% contaminated trades:
- `btc_up_ratio`: diluted by non-BTC 5m trade flow
- `btc_buy_ratio`: contaminated by different market's BUY/SELL distribution
- `btc_vwap_up/dn`: VWAP computed over mixed-market prices
- `btc_momentum`: sub-window ratios affected
- `btc_tw_up_ratio`: time-weighted flow corrupted
- All downstream features (zscores, interactions)

Note: the timestamp filter `[slot_ts, slot_ts+180s)` removes trades from other time
slots, but trades from other markets happening in the SAME 180s window pass through.

## Fix
Client-side filter in `fetch_inslot_trades()`:
```python
expected_slug = f"btc-updown-5m-{slot_ts}"
for t in batch:
    trade_slug = t.get("slug", "")
    if trade_slug and trade_slug != expected_slug:
        continue  # skip cross-market trades
```

## Verification
Before fix: `Fetched 3000 inslot ticks`
After fix:  `Fetched 2000 inslot ticks` (25% reduction = contamination removed)

## Why Training Wasn't Affected
Training uses `ticks_btc_full_clean.parquet` which is pre-filtered by `market_id`.
Each row belongs to exactly one market. No cross-market contamination possible.

## Lesson
NEVER assume an API endpoint returns only the data you asked for.
Always verify with exploratory queries that check for unexpected slugs/markets/outcomes.
The data-api is essentially a "search" that returns all matching tokens, not a "filter"
by market.
