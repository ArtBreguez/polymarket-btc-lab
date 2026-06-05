# Polymarket Data-API Token Sharing Behavior

## Summary
The Polymarket data-api has three surprising behaviors that caused multiple
critical bugs in the live trading pipeline. Any code consuming this API
must account for all three.

## Behavior 1: Token IDs Are Shared Across Markets
A single CLOB token ID appears in dozens to hundreds of unrelated markets.
Example token `62270355...` appeared in:
- btc-updown-5m-1780581600 (our target: 38 trades)
- btc-updown-15m-1780585200 (62 trades)
- eth-updown-5m-1780585500 (19 trades)
- sol-updown-5m-1780585500 (9 trades)
- bnb-updown-5m-1780585500 (9 trades)
- bitcoin-up-or-down-june-4-2026-11am-et (15 trades)
- bitcoin-above-64k-on-june-4-2026-539 (4 trades)
- wta-bartunk-knutson-2026-06-03 (9 trades — tennis!)
- cs2-mibr-lvg-2026-06-04 (8 trades — CS2 esports!)
- elon-musk-of-tweets-may-29-june-5-240-259 (1 trade)
- will-china-invade-taiwan-before-2027 (1 trade)
- highest-temperature-in-helsinki-on-june-5-2026-20c (2 trades — weather!)
- ...and 80+ more slugs

**Impact**: ~67% of trades returned for a BTC 5m token query are from wrong markets.

## Behavior 2: `/trades?asset=` Does Not Filter by Token
The `asset` parameter filters by condition/market group, NOT by specific token.
Both YES and NO token queries for the same market return the IDENTICAL set of trades.

**Impact**: If you force outcome labels based on which token you queried
(YES→"Up", NO→"Down"), you get vol_up = vol_dn for every slot → up_ratio = 0.5 always.

## Behavior 3: Server-Side Filters Are Ignored
The `slug` query parameter appears to be accepted but does NOT filter results server-side.
`/trades?asset=X&slug=btc-updown-5m-123` returns the same results as `/trades?asset=X`.

**Impact**: All filtering must be done client-side.

## Required Client-Side Filters
```python
expected_slug = f"btc-updown-5m-{slot_ts}"
for t in api_trades:
    # Filter 1: correct market only
    if t.get("slug", "") != expected_slug:
        continue
    # Filter 2: correct time window
    ts = int(t.get("timestamp", 0))
    if ts > 1e12: ts //= 1000
    if not (0 <= ts - slot_ts < 180):
        continue
    # Filter 3: use API's outcome field (correct for BTC 5m after slug filter)
    outcome = t.get("outcome", fallback_label)
```

## Validation
Before slug filter: `Fetched 3000 inslot ticks`, up_ratio=0.500 (artificial)
After slug filter:  `Fetched 1048 inslot ticks`, up_ratio=0.238 (real bearish signal)
