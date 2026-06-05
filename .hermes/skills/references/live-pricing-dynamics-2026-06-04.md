# Live Pricing Dynamics — Observations from v19 First Session (2026-06-04)

## Context
v19 deployed with 40 features, AUC=0.900. Monitored 5+ consecutive slots in real-time.

## Observation 1: Market Already Priced In (No Edge at Tail)
When model predicts DOWN with 83-96% confidence, the DOWN token ask is $0.86-$0.96.
This means:
- $0.96 ask → win $1.00 → only 4% max return
- After the edge_ask check (model_prob - ask >= 10%), these are correctly skipped
- The market is **efficient at the tail** — when it's obvious, everyone knows

The tradeable edge window is narrow: model conf high + ask hasn't caught up yet ($0.50-$0.75).

## Observation 2: Gamma outcomePrices Stale
`get_market_mid()` returned $0.495/$0.505 across 3+ consecutive slots during a strongly bearish period (up_ratio=0.26-0.45). Meanwhile, the orderbook had best asks at $0.86-$0.96 for DOWN.

This triggers the divergence check: `abs(0.900 - 0.495) = 0.405 > 0.20` → skip.

Possible causes:
- Gamma API caches outcomePrices and updates slowly for 5-min markets
- outcomePrices reflects last trade, not current orderbook state
- Low-volume slots may not update outcomePrices at all

Impact: good-edge trades skipped when ask is reasonable but mid is stale.
Fix ideas: compute mid from orderbook `(best_bid + best_ask) / 2` as fallback.

## Observation 3: CLOB Minimum 5 Shares Inflates Required Capital
The Polymarket CLOB enforces a minimum of 5 shares per order.
At $1.50 stake / $0.69 ask = 2.17 shares → bumped to 5 shares → $3.45 cost.
With 5% fee buffer: $3.45 × 1.05 = $3.62 required.

Real minimum balance needed: ~$5.00 USDC (not $1.50 as the STAKE_USDC constant suggests).

## Observation 4: ask=$0.000 from WS + HTTP Failure
When WS disconnects (frequent "no close frame"), price cache is wiped.
If HTTP /book fallback also fails (all asks >= $0.97), returns $0.000.
The [0.38, 0.90] range filter correctly catches this.
But it means 1-2 entry attempts per slot are wasted on failed price lookups.

## Observation 5: Data-API Returns Same Ticks Across Entry Window
Within a single slot, all entry attempts (t=177s through t=237s) show the same 1264-1400 ticks.
This is expected — data-api has ~120s lag, so ticks from t=0-60s are what's available at t=180s.
The model prediction can still vary across attempts because:
- OB features are fetched fresh each attempt from /book
- WS price updates change the ask price
- But CLOB flow features (up_ratio, momentum, tw) are stable within a slot

## Observation 6: Model Prediction Instability (UP/DOWN Flip)
In slot 1780589700, predictions flipped between UP (52-53%) and DOWN (54-56%) across consecutive 10s attempts, all with same ticks (1400 ticks, up_ratio=0.651).
This indicates the model is near its decision boundary (prob_up ≈ 0.50).
All were correctly skipped due to conf < 60%.
The instability comes from OB features varying between attempts (fresh /book fetch each time).
