# polymarket-btc-lab

ML research and live paper trading for Polymarket BTC 5-minute binary markets.

Predicts whether BTC will go **UP** or **DOWN** in each 5-minute slot by reading order flow from the Polymarket CLOB during the first 3 minutes of the slot, enriched with live spot price context from BTC, ETH, and SOL.

---

## Dataset

**HuggingFace:** [BrockMisner/polymarket-btc-updown](https://huggingface.co/datasets/BrockMisner/polymarket-btc-updown)

Each Polymarket crypto market is a 5-minute binary: *"Will BTC go up in the next 5 minutes?"*  
Participants buy YES/NO tokens during the slot. Their aggregate order flow is the signal.

| File | Size | Description |
|---|---|---|
| `ticks/{CRYPTO}/5-minute/` | 2.3 GB (BTC) | All CLOB trades per market — price, size, side, outcome |
| `spot_prices/` | 68 MB | BTC/ETH/SOL/XRP/BNB/DOGE/HYPE spot at 1s resolution |
| `orderbook/{CRYPTO}/5-minute/` | 3.2 GB (BTC) | Full orderbook snapshots |
| `prices/BTC/5-minute/` | 9.7 MB | CLOB UP/DOWN token prices per market |
| `markets.parquet` | 9.5 MB | Market metadata (slug, resolution, timestamps) |

**Resolved markets:** 616 per crypto × 7 cryptos = **4,312 total**  
Cryptos: BTC, ETH, SOL, XRP, BNB, DOGE, HYPE

---

## Research

### Why order flow works

Participants who buy YES/NO tokens *during* the slot are directionally voting before resolution. Their aggregate behavior — buy/sell imbalance, VWAP spread, momentum across sub-windows — carries real predictive signal. This is **not** data leakage: we only use trades from `[t=0, t=180s)` of the slot, predict at `t=180s`, and settle at `t=300s+`.

### Model versions

#### v1 — BTC only, 22 features (deprecated)

| Metric | Value |
|---|---|
| Walk-forward AUC | 0.824 |
| Walk-forward Accuracy | 74.7% |
| Brier Score | 0.210 |
| Training samples | 616 (BTC only) |

**Paper trading result: 1W/3L, -$14.95**

Root causes of failure:
1. **No edge filter** — entered trades where market already priced in the signal (e.g. model 84% UP, market $0.99 → edge = -15%)
2. **Hour-of-day bias** — all losses at 17–18h UTC (NY open, historically bullish for BTC). 616-sample model couldn't capture regime
3. **No spot context** — model had zero knowledge of where BTC was trending macro

#### v2 — 7 cryptos, 73 features (production)

| Metric | Value |
|---|---|
| Walk-forward AUC | **0.853** |
| Walk-forward Accuracy | **77.0%** |
| CV AUC (5-fold) | 0.914 ± 0.004 |
| CV Accuracy | 83.0% |
| Brier Score | 0.119 |
| Training samples | **3,900** (7 cryptos) |
| Walk-forward folds | 11 (AUC range: 0.692–0.943) |

**Improvements over v1:**
- 7× more data via multi-crypto training (all cryptos share the same directional order flow dynamic)
- 3 sub-windows (0–60s, 60–120s, 120–180s) + momentum + acceleration features
- Live BTC/ETH/SOL spot price context (return, volatility, momentum) over 4 lookback windows
- Edge filter: `model_prob − market_price ≥ 10%` required before any entry

### Feature engineering

All features computed from trades in `[t=0, t=180s)` of each slot:

**Order flow (29 features)**
| Feature | Description |
|---|---|
| `up_ratio` | % of volume on YES side |
| `vwap_up / vwap_dn` | Volume-weighted avg price, YES vs NO |
| `vwap_diff` | VWAP spread — key edge signal |
| `buy_ratio` | % of trades that were maker buys |
| `avg_size` | Average trade size (USDC) |
| `up_ratio_w1/w2/w3` | YES ratio per 60s sub-window |
| `momentum_early/late` | Change in YES ratio across windows |
| `acceleration` | Is buying momentum accelerating or decelerating? |
| `price_first/last/trend/vol` | YES token price trajectory |

**Spot context (39 features)**  
For each of BTC, ETH, SOL — 4 windows × 3 metrics + pct_of_1h_range:

| Window | Metrics |
|---|---|
| `inslot_3m` (slot to slot+180s) | ret, vol, mom |
| `pre_3m` (3 min before slot) | ret, vol, mom |
| `pre_15m` (15 min before) | ret, vol, mom |
| `pre_1h` (1 hour before) | ret, vol, mom |
| — | pct_of_1h_range |

**Time encoding (5 features)**  
`hour`, `hour_sin/cos`, `dow_sin/cos`

### Top 20 features by LightGBM importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | vwap_up | 825 |
| 2 | price_vol | 825 |
| 3 | vwap_dn | 787 |
| 4 | price_trend | 738 |
| 5 | price_last | 726 |
| 6 | buy_ratio | 710 |
| 7 | vwap_diff | 697 |
| 8 | sol_inslot_3m_mom | 679 |
| 9 | up_ratio_w3 | 677 |
| 10 | avg_size | 672 |
| 11 | momentum_late | 651 |
| 12 | sol_inslot_3m_ret | 628 |
| 13 | up_ratio_w1 | 608 |
| 14 | price_first | 603 |
| 15 | momentum_early | 586 |
| 16 | up_ratio | 574 |
| 17 | acceleration | 572 |
| 18 | sol_pre_3m_vol | 539 |
| 19 | btc_inslot_3m_mom | 504 |
| 20 | up_ratio_w2 | 502 |

SOL inslot momentum appears before BTC in the ranking — cross-asset correlation during the slot is a real signal.

### Hour-of-day accuracy bias

| Best hours (UTC) | Accuracy |
|---|---|
| 22h | 90.1% |
| 13h | 86.2% |
| 19h | 85.5% |
| 14h | 85.0% |
| 1h | 84.9% |

| Worst hours (UTC) | Accuracy |
|---|---|
| 15h | 74.8% |
| 0h | 78.7% |
| 2h | 79.2% |

15h UTC (NY pre-open) is the hardest to predict — high volatility, regime unclear.

---

## Architecture

```
┌──────────────────────────────┐
│  spot_daemon.py              │  wss://stream.binance.com
│  (always-on background proc) │ ──────────────────────────►
│                              │  btcusdt@kline_1m
│  Rolling 75-min buffer of    │  ethusdt@kline_1m
│  1m close prices per symbol  │  solusdt@kline_1m
│  → /tmp/spot_buffer.json     │
└──────────────────────────────┘
           │ file read (~1ms)
           ▼
┌──────────────────────────────┐     ┌──────────────────┐
│  paper_trader.py             │────►│ Polymarket CLOB  │
│  (cron, every 1 minute)      │     │ data-api (ticks) │
│                              │     └──────────────────┘
│  1. Settle expired trades    │
│  2. At t=170–240s: predict   │
│     - Fetch 3min order flow  │
│     - Read spot from buffer  │
│     - model.predict_proba()  │
│     - Edge filter ≥10%       │
│  3. Record to paper_trades   │
└──────────────────────────────┘
```

**Why WebSocket for spot data:**  
The alternative (REST polling Binance klines) required 15 HTTP calls per inference adding 3–7 seconds of latency. The entry window is only 70 seconds wide (t=170s to t=240s). The daemon streams ticks continuously and writes to a local file — inference reads from disk in ~1ms.

---

## Repo structure

```
scripts/
  paper_trader.py          # Production paper trader (cron, every 1m)
  spot_daemon.py           # Binance WS daemon → /tmp/spot_buffer.json
  research_v2_pipeline.py  # Full training pipeline for v2 model
  final_pipeline.py        # v1 training pipeline (reference)
  rolling_origin_eval.py   # Walk-forward evaluation
  window_signal_analysis.py # Signal AUC by time window

src/btc_lab/
  plugin.py                # BtcUpDownPlugin (pmlab MarketPlugin)
  features.py              # Feature computation
  config.py                # Paths and constants

artifacts/
  btc_model_v2_research.pkl  # Production model (gitignored — large)
  research_v2_report.json    # Full evaluation report
  paper_trades.json          # Live paper trade log
  rolling_origin_3min.json   # v1 walk-forward results

tests/
  test_features.py
```

> **Note:** Model `.pkl` files and `.parquet` datasets are gitignored (too large).  
> Download the dataset from HuggingFace and retrain with `scripts/research_v2_pipeline.py`.

---

## Running

**Requirements:** Python 3.11+, uv

```bash
git clone https://github.com/ArtBreguez/polymarket-btc-lab
cd polymarket-btc-lab
uv sync
```

**Start the spot daemon (keep running):**
```bash
uv run python scripts/spot_daemon.py
```

**Run the paper trader once:**
```bash
uv run python scripts/paper_trader.py
```

**Retrain v2 model** (requires dataset from HuggingFace):
```bash
# Download dataset first (requires ~5GB disk)
uv run python scripts/research_v2_pipeline.py
```

---

## Trading strategy

- **Entry:** at `t=170–240s` of each 5-minute slot
- **Signal:** model confidence > 60%
- **Edge filter:** `model_prob − market_ask ≥ 10%` (prevents entering when market has already priced the signal)
- **Stake:** flat $5 USDC per trade (paper)
- **Settlement:** 60s after slot end, using `outcomePrices` field (not `resolution` — authoritative for closed markets)
- **Live capital gate:** 30 settled trades, win rate ≥ 60%, P&L > 0, avg edge ≥ 10%

---

## Known pitfalls

- **Polymarket data-api pagination:** trades are returned newest-first. Time filter params (`after=`, `since=`) are accepted but silently ignored. Must paginate via `offset` until reaching intra-slot timestamps.
- **Orderbook parquet type mismatch:** `market_id` is STRING in orderbook files but INT in `markets.parquet`. PyArrow cannot push down the filter, reads all 1.48B rows → OOM. Cast to strings before filtering or skip orderbook entirely.
- **outcomePrices vs resolution:** `market.resolution` returns `None` for many settled markets. Use `outcomePrices`: if `op[0] >= 0.99` → UP won, if `op[1] >= 0.99` → DOWN won.
