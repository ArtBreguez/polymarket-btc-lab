# Outcome Label Bug — Corrected Root Cause Analysis (2026-06-04)

## Original Diagnosis (WRONG)
Initially believed data-api returned `outcome="Yes"/"No"` instead of `"Up"/"Down"`,
causing buy_ratio/zscore explosion. Fix was to force `outcome_label` from the token loop.

## Actual Root Cause
The data-api returns trades from ALL markets sharing the same token ID (~67% contamination).
Non-BTC markets (tennis, weather, elections, etc.) use "Yes"/"No" outcomes — these were
the source of the non-Up/Down values. BTC 5m markets correctly return "Up"/"Down".

## Why the "Fix" Made Things Worse
The data-api returns IDENTICAL trade sets for both YES and NO token queries — it does
NOT filter by token, only by condition/market. When outcome_label was forced:
- YES token query → all trades forced to "Up" → vol_up = $X
- NO token query → SAME trades forced to "Down" → vol_dn = $X
- Result: vol_up = vol_dn ALWAYS → up_ratio = 0.500 for EVERY slot

This created:
- btc_up_ratio = 0.500 (constant, no signal)
- btc_momentum = 0.000 (no signal)
- btc_tw_up_ratio = 0.500 (no signal)
- Model producing extreme predictions (99%+) from spot features alone

## Correct Fix (Applied)
1. Add slug filter: `if t.get("slug") != f"btc-updown-5m-{slot_ts}": continue`
   This removes all cross-market contamination, including the Yes/No outcomes.
2. Use API outcome with fallback: `t.get("outcome", outcome_label)`
   API returns correct "Up"/"Down" for BTC 5m markets after slug filter.

## Key Insight
The Polymarket data-api has three surprising behaviors:
1. Token IDs are shared across hundreds of unrelated markets
2. `/trades?asset=<token_id>` returns trades from ALL markets using that token
3. Both YES and NO token queries return the SAME set of trades

NEVER assume an API endpoint returns only what you asked for. Always verify
with exploratory queries checking slugs, outcomes, and trade distributions.
