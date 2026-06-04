# 01 — Data Pipeline

## Overview

The model trains on tick-level CLOB data from Polymarket BTC 5-minute binary markets, combined with BTC/USDT spot candles from Binance. Data comes from two sources: local CLOB collection (early markets, Mar–Apr 2026) and the pmdata.dev API (retroactive ticks, Apr 12 – Jun 3 2026). All data is stored on a Modal Volume for cloud training.

---

## Data Sources

### 1. pmdata.dev API (Primary — retroactive tick data)

| Property | Value |
|----------|-------|
| Endpoint | `https://api.pmdata.dev/get-download-url/poly_l2/{slug}` |
| Auth | Header `api_key: sk-5uX...Ijko` |
| Format | `poly_l2` — Parquet files with L2 order book + trade events |
| Slug pattern | `btc-updown-5m-{ts}` where `{ts}` is the slot Unix timestamp |
| Coverage | Apr 12, 2026 – Jun 3, 2026 (~15,257 markets) |
| Fetcher script | `scripts/fetch_pmdata_ticks.py` |
| Rate limit | ~10 req/s (use `--workers 12` with backoff) |

**How it works:**
1. Script reads `data/new_markets.csv` for market IDs and slot timestamps
2. For each market, constructs the slug: `btc-updown-5m-{slot_ts}`
3. Calls pmdata.dev to get a signed download URL for the poly_l2 parquet
4. Downloads the parquet, extracts `last_trade_price` events
5. Filters to the observation window `[slot_ts, slot_ts + 180s)`
6. Maps to internal tick schema (see below)
7. Saves progress to `data/new_ticks_pmdata_progress.json` (resumable)

**poly_l2 to internal tick conversion** (`fetch_pmdata_ticks.py`):
```
poly_l2 fields          →  Internal tick schema
─────────────────────       ─────────────────────
(derived from slug)     →  market_id
timestamp               →  timestamp_ms (× 1000 if seconds)
trade_price             →  price
trade_size              →  size_usdc
trade_side              →  side ("BUY" / "SELL")
winning_outcome / price →  outcome ("Up" if price > 0.5, else "Down")
```

### 2. Local CLOB Collection (Legacy — early markets)

| Property | Value |
|----------|-------|
| Source | Direct Polymarket CLOB WebSocket collection |
| Coverage | Mar 2026 – Apr 2026 (~7,062 markets) |
| Format | Parquet (`ticks_btc_5min.parquet`) |
| Status | Merged into `ticks_btc_full_clean.parquet` |

### 3. Binance BTC Spot Candles

| Property | Value |
|----------|-------|
| Source | Binance REST API (`/api/v3/klines`, symbol `BTCUSDT`, interval `1m`) |
| Coverage | Feb 15, 2026 – present |
| Format | CSV (columns: open_time, open, high, low, close, volume, ...) |
| Geo-block | Binance API is blocked from Modal's US servers |
| Fetcher script | `scripts/fetch_binance_spot.py` |

**Important:** Binance spot data must be pre-fetched from a non-geo-blocked machine and uploaded to the Modal Volume before training. The training script fetches it inline from the volume, NOT from the Binance API.

---

## Modal Volume Contents

Volume name: `btc-local-data`
Mount point in Modal: `/btc_local/`

```
/btc_local/
├── ticks_btc_full_clean.parquet   # 22,237 markets, 68.3M ticks
│                                   # Merged: local CLOB + pmdata.dev
│                                   # Zero-timestamp rows pre-cleaned
│
├── all_markets.csv                 # 22,319 markets metadata
│                                   # Columns: market_id, slot_ts,
│                                   #   condition_id, question, outcome,
│                                   #   resolution_price, resolved_at
│
└── spot_1m_*.csv                   # Binance 1m candles
                                    # Pre-fetched locally
```

### Tick Schema (`ticks_btc_full_clean.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `market_id` | string | Polymarket condition ID |
| `timestamp_ms` | int64 | Trade timestamp in milliseconds |
| `outcome` | string | "Up" or "Down" |
| `side` | string | "BUY" or "SELL" |
| `price` | float64 | Trade price (0.0 – 1.0, probability) |
| `size_usdc` | float64 | Trade size in USDC |

### Markets Schema (`all_markets.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `market_id` | string | Polymarket condition ID |
| `slot_ts` | int64 | Slot start Unix timestamp |
| `question` | string | Market question text |
| `outcome` | string | Resolved outcome ("Up" / "Down") |
| `resolution_price` | float64 | 1.0 (Up won) or 0.0 (Down won) |

---

## Dataset Expansion Process

When new markets become available (Polymarket creates ~288 BTC 5-minute markets per day):

### Step 1: Fetch market metadata
```bash
python scripts/fetch_new_markets.py
# Output: data/new_markets.csv
```

### Step 2: Fetch tick data from pmdata.dev
```bash
python scripts/fetch_pmdata_ticks.py --workers 12
# Output: data/new_ticks_pmdata.parquet
# Progress: data/new_ticks_pmdata_progress.json (resumable)
```

### Step 3: Fetch Binance spot candles (from non-blocked host)
```bash
python scripts/fetch_binance_spot.py
# Output: data/spot_1m_YYYYMMDD.csv
```

### Step 4: Merge and clean
```bash
# Combine old + new ticks, remove zero-timestamp rows
# Upload merged parquet to Modal Volume:
modal volume put btc-local-data ticks_btc_full_clean.parquet /ticks_btc_full_clean.parquet
modal volume put btc-local-data all_markets.csv /all_markets.csv
```

### Step 5: Train new model version
```bash
modal run scripts/train_v18_modal.py
```

---

## Data Coverage Timeline

```
Feb 15, 2026  ─────────────────────────────────────────────── Present
     |                                                           |
     |  Binance spot candles (continuous)                       |
     |  ═══════════════════════════════════════════════════════ |
     |                                                           |
     |        Mar 2026        Apr 12         Jun 3               |
     |        |── local ──|── pmdata.dev ──|                    |
     |        | CLOB coll.|| retroactive   |                    |
     |        | 7,062 mkts|| 15,257 mkts   |                    |
     |        |───────────||───────────────|                    |
     |                     |                                     |
     |                     └── Merged into ticks_btc_full_clean  |
     |                         22,237 markets total              |
```

---

## Data Quality Notes

- **Zero-timestamp rows:** Some pmdata.dev ticks had `timestamp_ms = 0`. These are pre-cleaned in `ticks_btc_full_clean.parquet`.
- **Observation window:** Only ticks in `[slot_ts, slot_ts + 180s)` are used (first 3 minutes of 5-minute slot).
- **Missing ticks:** Some low-liquidity markets have <10 ticks. Feature engineering handles this with neutral defaults.
- **Duplicate markets:** `all_markets.csv` has 22,319 entries vs 22,237 in ticks — 82 markets had no ticks and are excluded from training.

---

## See Also

- [Architecture](00-architecture.md) — system overview and costs
- [Feature Engineering](02-feature-engineering.md) — how ticks become features
- [Experiment Log](../EXPERIMENTS.md) — data expansion impact on model quality
