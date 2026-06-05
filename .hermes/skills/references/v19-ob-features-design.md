# V19 L2 Orderbook Features Design

## Data Source

pmdata.dev `poly_l2` API returns 3 event types per market slot:
- `book` (~2,700 events/slot): full orderbook snapshots with ask_prices, ask_sizes, bid_prices, bid_sizes arrays
- `price_change` (~68,000 events/slot): BBO price movements
- `last_trade_price` (~900 events/slot): individual trades (already used in v15-v18 tick data)

## Pre-computation Script: `scripts/fetch_ob_features_modal.py`

Modal app `btc-fetch-ob`, 4 CPU, 16GB RAM, 3h timeout.
- Secret: `pmdata-api-key` (env var: `PMDATA_API_KEY`)
- Output: `/btc_local/ob_features_full.parquet` on Modal Volume
- Progress: `/btc_local/ob_progress.json`
- Resume-safe: flushes every 200 markets, commits volume after each batch
- 20 parallel HTTP workers

## Feature Definitions

### From Book Snapshots

**Open snapshot** = first book event where t_sec <= 30
**Close snapshot** = last book event where t_sec >= 150

| Feature | Formula | Neutral |
|---------|---------|---------|
| ob_mid | (best_ask + best_bid) / 2 | 0.5 |
| ob_spread | best_ask - best_bid | 0.02 |
| ob_imbalance | (best_bid_sz - best_ask_sz) / (bid + ask + 1e-8) | 0.0 |
| ob_depth_ratio | bid_depth_5c / ask_depth_5c | 1.0 |
| ob_bid_depth_5c | sum(bid_sizes where price >= mid-0.05) / total_bid | 0.5 |
| ob_ask_depth_5c | sum(ask_sizes where price <= mid+0.05) / total_ask | 0.5 |
| ob_total_depth | sum(all bid + ask sizes) | 1000.0 |
| ob_weighted_imb | exp-weighted imbalance (closer to mid = higher weight, decay=10) | 0.0 |
| ob_mid_drift | close.mid - open.mid | 0.0 |
| ob_imbalance_end | close snapshot imbalance | 0.0 |
| ob_spread_end | close snapshot spread | 0.02 |
| ob_depth_change | close.total_depth - open.total_depth | 0.0 |
| ob_imb_momentum | close.imb - open.imb | 0.0 |
| ob_imb_w0 | mean imbalance in [0, 60)s | 0.0 |
| ob_imb_w1 | mean imbalance in [60, 120)s | 0.0 |
| ob_imb_w2 | mean imbalance in [120, 180)s | 0.0 |

### From Price Change Events

| Feature | Formula | Neutral |
|---------|---------|---------|
| ob_pc_up_ratio | count(price_diff > 0) / total_changes | 0.5 |
| ob_pc_volatility | std(price_diffs) | 0.0 |
| ob_pc_count | total price changes | 0.0 |
| ob_fill_imbalance | (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8) | 0.0 |

### Cross-Domain Interactions (OB × CLOB)

| Feature | Formula | Neutral |
|---------|---------|---------|
| x_imb_x_ur | ob_imbalance × btc_up_ratio | 0.0 |
| x_depth_x_momentum | ob_depth_ratio × btc_momentum | 0.0 |
| x_spread_x_vol | ob_spread × btc_n_ticks | 0.0 |
| x_ob_drift_x_inslot | ob_mid_drift × btc_inslot_ret | 0.0 |
| x_fill_imb_x_buy | ob_fill_imbalance × btc_buy_ratio | 0.0 |

## Live Trading Compatibility

When v19 is promoted, live_trader.py must compute OB features from the CLOB REST endpoint:
```
GET /book?token_id=<token_id>
```
This returns current orderbook with bids/asks arrays — same structure as poly_l2 book events.
The live trader already has `fetch_order_book()` which returns bid/ask arrays.

Key mapping:
- Training: poly_l2 book snapshots (historical, multiple per slot)
- Live: CLOB REST /book endpoint (real-time, one snapshot at observation end)
- Mismatch risk: training has open+close snapshots, live may only have one
- Mitigation: use close-only features for live (ob_imbalance_end, ob_spread_end), or take 2+ snapshots during observation window
