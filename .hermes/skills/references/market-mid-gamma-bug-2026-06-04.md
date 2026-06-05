# Market Mid Bug: Gamma outcomePrices vs CLOB Book (2026-06-04)

## Summary
`get_market_mid()` used Gamma API `outcomePrices` to compute market midpoints. For BTC 5-min markets, these prices are essentially useless — they start at ~$0.50/$0.50 on market creation and rarely (if ever) update during the active 5-minute slot.

## Impact
The `ask-vs-mid divergence check` (`abs(ask - mid) > 0.20`) blocked virtually ALL trades because:
- Real book ask for DOWN token: $0.86 (market correctly priced DOWN)
- Gamma mid for DOWN: $0.495 (stuck at default)
- Divergence: $0.365 > $0.20 threshold → **SKIP**

Over 20+ minutes of observation, ZERO trades executed despite high-confidence predictions (83-96%).

## Evidence

### Gamma API returns stale data for active slots
```
Slot 1780592100 (active):  outcomePrices = ["0.505", "0.495"]   ← default
Slot 1780591800 (active):  outcomePrices = ["0.545", "0.455"]   ← barely moved
Slot 1780591500 (resolved): outcomePrices = ["0",     "1"]      ← correct post-resolution
Slot 1780591200 (resolved): outcomePrices = ["1",     "0"]      ← correct post-resolution
```

### CLOB book tells the real story
For the same active slot (1780592100):
```
UP token book:  best_ask=$0.170  best_bid=$0.160  mid=$0.165
Gamma says UP:  $0.505
DELTA:          $0.380 (!!)
```

The market actually prices UP at ~16.5%, not 50.5%.

## Root Cause
Gamma `outcomePrices` appears to be trade-weighted or VWAP-based, updated only when trades execute through the Gamma orderbook. For BTC 5-min binary markets:
- Markets are created every 5 minutes
- Initial outcomePrices default to ~0.50/0.50
- Most trading volume goes through the CLOB (not Gamma directly)
- By the time trades start flowing, the observation window is already open
- outcomePrices may update to ~0.545 at best, still far from true book mid

## Fix Applied
Replaced Gamma outcomePrices with CLOB `/book` midpoints:

```python
# OLD (broken): Gamma outcomePrices
op = m.get("outcomePrices", "[]")
up_mid = float(op[0])  # always ~0.505

# NEW (correct): CLOB book midpoint
for each token:
    book = GET /book?token_id=<tid>
    asks = [float(a["price"]) for a in book["asks"] if 0 < price < 0.99]
    bids = [float(b["price"]) for b in book["bids"] if 0 < price < 0.99]
    mid = (min(asks) + max(bids)) / 2.0
```

Fallbacks:
- If only asks exist → use best ask as proxy
- If only bids exist → use best bid as proxy
- If only one token has a book → derive other via binary identity (up + dn ≈ 1)

## Verification
First trade post-fix:
```
BUY UP | ask=$0.450 | market_mid=$0.405 (delta=0.045 ← CORRECT)
edge_ask=26.2% | edge_mid=30.7%
Order FILLED: 5.00 shares @ $0.450 = $2.25
```

Pre-fix, this same trade would have shown `market_mid=$0.505` and divergence=$0.055 — it would have passed, but only by accident. The real danger was the DOWN-side trades where asks were $0.86+ and Gamma mid was $0.495 → divergence $0.365 → blocked.

## Lesson
Never trust Gamma `outcomePrices` for short-lived markets. The CLOB order book is the only reliable source of current market state for BTC 5-min binaries. This may also affect other short-duration Polymarket markets (10-min, hourly).
