"""BTC 5-minute Up/Down market plugin for pmlab.

Handles Polymarket BTC 5-minute prediction markets (Will BTC be up/down in 5 minutes?).
Computes tick-based and price-based features for ML inference and training.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import pandas as pd

from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.plugins.base import MarketPlugin

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_API_BASE = "https://data-api.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Feature columns (in training order) — must match btc_5min_dataset_v3_clean.parquet
FEATURE_COLS: list[str] = [
    "first_price",
    "last_price",
    "price_mean",
    "price_std",
    "price_min",
    "price_max",
    "price_momentum",
    "n_ticks",
    "price_at_25pct",
    "price_at_50pct",
    "price_at_75pct",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "total_volume_usdc",
    "buy_volume_usdc",
    "sell_volume_usdc",
    "buy_sell_imbalance",
    "up_volume_usdc",
    "down_volume_usdc",
    "up_down_volume_ratio",
    "n_trades",
    "n_buy_trades",
    "n_sell_trades",
    "avg_trade_size",
    "spot_price_start",
    "spot_price_end",
    "spot_return",
    "spot_volatility",
    "spot_price_mean",
    "vwap_up",
    "vwap_down",
]

# BTC 5-min market filters
# Real Polymarket BTC 5-min markets are titled "Bitcoin Up or Down - [Date], [TimeRange] ET"
# or daily ones like "Bitcoin Up or Down - September 13, 11PM ET"
_BTC_UPDOWN_PHRASES = (
    "bitcoin up or down",
    "btc up or down",
    "will btc be up",
    "will bitcoin be up",
    "will btc be higher",
    "will bitcoin be higher",
)
# Legacy/broader keywords kept for completeness but not used for slug matching
_BTC_KEYWORDS = _BTC_UPDOWN_PHRASES + ("will btc be", "btc higher", "btc lower", "bitcoin higher", "bitcoin lower")
# Slug hints must be specific enough to avoid false positives like "bitcoin hit $1m"
_BTC_SLUG_HINTS = ("bitcoin-up-or-down", "btc-up-or-down", "btc-updown", "bitcoin-updown")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_cyclical(value: float, period: float) -> tuple[float, float]:
    """Return (sin, cos) for a cyclical feature."""
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def _is_btc_5min_market(raw: dict[str, Any]) -> bool:
    """Return True if raw market dict looks like a BTC 5-minute Up/Down market."""
    question = (raw.get("question") or "").lower()
    slug = (raw.get("slug") or "").lower()
    # Must specifically match BTC up/down phrasing (not just any bitcoin market)
    btc_match = any(phrase in question for phrase in _BTC_UPDOWN_PHRASES)
    btc_slug = any(h in slug for h in _BTC_SLUG_HINTS)
    if not (btc_match or btc_slug):
        return False
    # Outcomes must be binary Yes/No or Up/Down
    outcomes = raw.get("outcomes") or []
    if isinstance(outcomes, str):
        import json
        try:
            outcomes = json.loads(outcomes)
        except Exception:
            outcomes = []
    if len(outcomes) != 2:
        return False
    return True


def _build_spec(raw: dict[str, Any]) -> MarketSpec:
    """Parse a raw Gamma market dict into a MarketSpec."""
    import json

    outcomes_raw = raw.get("outcomes") or []
    if isinstance(outcomes_raw, str):
        try:
            outcomes_raw = json.loads(outcomes_raw)
        except Exception:
            outcomes_raw = ["Yes", "No"]

    prices_raw = raw.get("outcomePrices") or []
    if isinstance(prices_raw, str):
        try:
            prices_raw = json.loads(prices_raw)
        except Exception:
            prices_raw = []

    prices: dict[str, float] = {}
    for label, p in zip(outcomes_raw, prices_raw):
        try:
            prices[str(label)] = float(p)
        except (TypeError, ValueError):
            prices[str(label)] = 0.5

    bins = [
        OutcomeBin(label=str(lbl))
        for lbl in outcomes_raw
    ]

    # Market price = Yes (Up) token price
    market_price = prices.get("Yes", prices.get(outcomes_raw[0] if outcomes_raw else "Yes", 0.5))

    condition_id = str(raw.get("conditionId") or raw.get("id") or "")

    return MarketSpec(
        market_id=condition_id,
        slug=str(raw.get("slug") or ""),
        question=str(raw.get("question") or ""),
        outcome_bins=bins,
        close_time=str(raw.get("endDate") or ""),
        market_family="binary",
        tags=["crypto", "btc", "5min"],
        metadata={
            "condition_id": condition_id,
            "market_price": market_price,
            "outcome_labels": [str(l) for l in outcomes_raw],
            "outcome_prices": prices,
            "raw": raw,
        },
    )


def _fetch_trades(condition_id: str, limit: int = 500, timeout: float = 15.0) -> list[dict[str, Any]]:
    """Fetch trades from Polymarket Data API for a given conditionId."""
    url = f"{DATA_API_BASE}/trades"
    params = {"market": condition_id, "limit": limit}
    resp = httpx.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    # Some endpoints wrap in {"data": [...]}
    if isinstance(data, dict):
        return data.get("data", data.get("trades", []))
    return []


def _fetch_btc_spot_price(timeout: float = 10.0) -> float | None:
    """Fetch current BTC spot price from Binance."""
    try:
        resp = httpx.get(BINANCE_PRICE_URL, timeout=timeout)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception as exc:
        _log.warning("Failed to fetch BTC spot price: %s", exc)
        return None


def _fetch_btc_klines(
    start_ms: int,
    end_ms: int,
    interval: str = "1m",
    timeout: float = 10.0,
) -> list[list[Any]]:
    """Fetch BTC OHLCV klines from Binance for a time window."""
    try:
        resp = httpx.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": "BTCUSDT",
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 10,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[return-value]
    except Exception as exc:
        _log.warning("Failed to fetch BTC klines: %s", exc)
        return []


def _compute_features_from_trades(
    trades: list[dict[str, Any]],
    reference_dt: datetime,
) -> dict[str, float]:
    """
    Compute ML features from a list of raw trade dicts.

    Expected trade fields:
        - type / side: "BUY" or "SELL"
        - outcome: "Yes" (Up) or "No" (Down)
        - price: float in [0,1]
        - usdcSize / size: USDC size of trade
        - timestamp: unix seconds or ISO string
    """
    if not trades:
        _log.debug("No trades available — returning NaN features")
        return _nan_features(reference_dt)

    rows = []
    for t in trades:
        try:
            side = (t.get("type") or t.get("side") or "").upper()
            outcome = (t.get("outcome") or t.get("outcomeIndex") or "")
            price = float(t.get("price", 0.0))
            size = float(t.get("usdcSize") or t.get("size") or 0.0)
            ts_raw = t.get("timestamp") or t.get("createdAt") or 0
            if isinstance(ts_raw, str):
                try:
                    ts = pd.Timestamp(ts_raw).timestamp()
                except Exception:
                    ts = 0.0
            else:
                ts = float(ts_raw)
            rows.append({
                "side": side,
                "outcome": str(outcome),
                "price": price,
                "size_usdc": size,
                "ts": ts,
            })
        except Exception as exc:
            _log.debug("Skipping malformed trade: %s", exc)
            continue

    if not rows:
        return _nan_features(reference_dt)

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

    # Classify Up / Down tokens (Yes=Up, No=Down; also handle "0"/"1" index)
    def _is_up(outcome: str) -> bool:
        return outcome.lower() in ("yes", "up", "0")

    def _is_down(outcome: str) -> bool:
        return outcome.lower() in ("no", "down", "1")

    buy_mask = df["side"] == "BUY"
    sell_mask = df["side"] == "SELL"
    up_mask = df["outcome"].apply(_is_up)
    down_mask = df["outcome"].apply(_is_down)

    buy_vol = df.loc[buy_mask, "size_usdc"].sum()
    sell_vol = df.loc[sell_mask, "size_usdc"].sum()
    total_vol = buy_vol + sell_vol
    up_vol = df.loc[up_mask, "size_usdc"].sum()
    down_vol = df.loc[down_mask, "size_usdc"].sum()

    n_trades = len(df)
    n_buy = buy_mask.sum()
    n_sell = sell_mask.sum()
    avg_trade_size = total_vol / n_trades if n_trades > 0 else 0.0
    buy_sell_imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
    up_down_ratio = up_vol / (up_vol + down_vol) if (up_vol + down_vol) > 0 else 0.5

    up_df = df[up_mask]
    down_df = df[down_mask]
    vwap_up = float(
        (up_df["price"] * up_df["size_usdc"]).sum() / up_vol
        if up_vol > 0 else float("nan")
    )
    vwap_down = float(
        (down_df["price"] * down_df["size_usdc"]).sum() / down_vol
        if down_vol > 0 else float("nan")
    )

    # Price features: treat "Up" token price as the market price series
    up_prices = up_df["price"].values if len(up_df) > 0 else df["price"].values
    if len(up_prices) == 0:
        up_prices = np.array([float("nan")])

    first_price = float(up_prices[0])
    last_price = float(up_prices[-1])
    price_mean = float(np.nanmean(up_prices))
    price_std = float(np.nanstd(up_prices, ddof=0))
    price_min = float(np.nanmin(up_prices))
    price_max = float(np.nanmax(up_prices))
    price_momentum = last_price - first_price
    n_ticks = len(up_prices)

    def _price_at_q(q: float) -> float:
        if len(up_prices) == 0:
            return float("nan")
        idx = int(round(q * (len(up_prices) - 1)))
        idx = max(0, min(idx, len(up_prices) - 1))
        return float(up_prices[idx])

    price_at_25 = _price_at_q(0.25)
    price_at_50 = _price_at_q(0.50)
    price_at_75 = _price_at_q(0.75)

    # Temporal features from reference_dt (market start / current time)
    hour = reference_dt.hour + reference_dt.minute / 60.0
    dow = reference_dt.weekday()
    hour_sin, hour_cos = _encode_cyclical(hour, 24.0)
    dow_sin, dow_cos = _encode_cyclical(dow, 7.0)

    # Spot price — try to derive from klines around trade window
    spot_start = float("nan")
    spot_end = float("nan")
    spot_return_val = float("nan")
    spot_volatility = 0.0
    spot_price_mean = float("nan")

    if len(df) > 0 and df["ts"].iloc[0] > 0:
        start_ms = int(df["ts"].iloc[0] * 1000)
        end_ms = int(df["ts"].iloc[-1] * 1000) + 60_000
        klines = _fetch_btc_klines(start_ms, end_ms)
        if klines:
            opens = [float(k[1]) for k in klines]
            closes = [float(k[4]) for k in klines]
            spot_start = opens[0]
            spot_end = closes[-1]
            spot_return_val = (spot_end - spot_start) / spot_start if spot_start != 0 else float("nan")
            spot_price_mean = float(np.mean(closes))
            spot_volatility = float(np.std(closes, ddof=0)) if len(closes) > 1 else 0.0

    return {
        "first_price": first_price,
        "last_price": last_price,
        "price_mean": price_mean,
        "price_std": price_std,
        "price_min": price_min,
        "price_max": price_max,
        "price_momentum": price_momentum,
        "n_ticks": float(n_ticks),
        "price_at_25pct": price_at_25,
        "price_at_50pct": price_at_50,
        "price_at_75pct": price_at_75,
        "hour_of_day_sin": hour_sin,
        "hour_of_day_cos": hour_cos,
        "day_of_week_sin": dow_sin,
        "day_of_week_cos": dow_cos,
        "total_volume_usdc": float(total_vol),
        "buy_volume_usdc": float(buy_vol),
        "sell_volume_usdc": float(sell_vol),
        "buy_sell_imbalance": float(buy_sell_imbalance),
        "up_volume_usdc": float(up_vol),
        "down_volume_usdc": float(down_vol),
        "up_down_volume_ratio": float(up_down_ratio),
        "n_trades": float(n_trades),
        "n_buy_trades": float(n_buy),
        "n_sell_trades": float(n_sell),
        "avg_trade_size": float(avg_trade_size),
        "spot_price_start": spot_start,
        "spot_price_end": spot_end,
        "spot_return": spot_return_val,
        "spot_volatility": spot_volatility,
        "spot_price_mean": spot_price_mean,
        "vwap_up": vwap_up,
        "vwap_down": vwap_down,
    }


def _nan_features(reference_dt: datetime) -> dict[str, float]:
    """Return a feature dict filled with NaN / zeros for a market with no data."""
    hour = reference_dt.hour + reference_dt.minute / 60.0
    dow = reference_dt.weekday()
    hour_sin, hour_cos = _encode_cyclical(hour, 24.0)
    dow_sin, dow_cos = _encode_cyclical(dow, 7.0)
    feats = {col: float("nan") for col in FEATURE_COLS}
    feats["hour_of_day_sin"] = hour_sin
    feats["hour_of_day_cos"] = hour_cos
    feats["day_of_week_sin"] = dow_sin
    feats["day_of_week_cos"] = dow_cos
    feats["n_ticks"] = 0.0
    feats["n_trades"] = 0.0
    feats["n_buy_trades"] = 0.0
    feats["n_sell_trades"] = 0.0
    feats["avg_trade_size"] = 0.0
    feats["buy_sell_imbalance"] = 0.0
    feats["up_down_volume_ratio"] = 0.5
    feats["total_volume_usdc"] = 0.0
    feats["buy_volume_usdc"] = 0.0
    feats["sell_volume_usdc"] = 0.0
    feats["up_volume_usdc"] = 0.0
    feats["down_volume_usdc"] = 0.0
    return feats


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class BtcUpDownPlugin(MarketPlugin):
    """Plugin for Polymarket BTC 5-minute Up/Down prediction markets.

    Discover active markets, compute tick-based ML features, resolve outcomes.
    """

    family = "btc_updown"

    def __init__(
        self,
        gamma_base_url: str = GAMMA_API_BASE,
        data_api_base_url: str = DATA_API_BASE,
        http_timeout: float = 20.0,
    ) -> None:
        self._gamma_base = gamma_base_url
        self._data_api_base = data_api_base_url
        self._timeout = http_timeout

    # ------------------------------------------------------------------
    # MarketPlugin interface
    # ------------------------------------------------------------------

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        """Discover active BTC 5-minute Up/Down markets from Gamma API.

        Searches multiple pages of the Gamma API without tag filtering to find
        markets with 'Bitcoin Up or Down' in the question (the real Polymarket
        5-min BTC market title format: "Bitcoin Up or Down - [Date], [Time] ET").

        Returns:
            List of MarketSpec for active BTC 5-min markets.
        """
        active = kwargs.get("active", True)
        max_pages = kwargs.get("max_pages", 20)  # Up to 2000 markets
        page_size = 100

        _log.info(
            "Discovering BTC 5-min markets (active=%s, scanning up to %d pages)…",
            active, max_pages,
        )

        all_raw: list[dict[str, Any]] = []
        for page in range(max_pages):
            offset = page * page_size
            try:
                resp = httpx.get(
                    f"{self._gamma_base}/markets",
                    params={
                        "limit": page_size,
                        "offset": offset,
                        "active": active,
                        "closed": not active,
                    },
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                batch: list[dict[str, Any]] = resp.json()
            except Exception as exc:
                _log.warning("Gamma API call failed at offset %d: %s", offset, exc)
                break

            if not batch:
                _log.debug("No more markets at offset %d — stopping", offset)
                break

            all_raw.extend(batch)
            _log.debug("Fetched offset=%d: %d markets", offset, len(batch))

            if len(batch) < page_size:
                break

        specs = []
        for raw in all_raw:
            if _is_btc_5min_market(raw):
                try:
                    specs.append(_build_spec(raw))
                except Exception as exc:
                    _log.debug("Failed to build spec for market %s: %s", raw.get("id"), exc)

        _log.info(
            "Scanned %d total markets, found %d BTC 5-min Up/Down markets",
            len(all_raw), len(specs),
        )
        return specs

    def fetch_features(
        self,
        spec: MarketSpec,
        horizon: str = "live",
        **kwargs: Any,
    ) -> dict[str, float]:
        """Fetch live tick data and compute ML features for *spec*.

        Hits the Polymarket Data API for trades in this market, then
        computes the same 33-dimensional feature vector used at training time.

        Args:
            spec: The market to featurize.
            horizon: Unused for live markets; kept for interface compatibility.

        Returns:
            Dict of feature_name → float, matching FEATURE_COLS order.
        """
        condition_id = spec.metadata.get("condition_id") or spec.market_id
        reference_dt = kwargs.get("reference_dt") or datetime.now(tz=timezone.utc)
        limit = kwargs.get("trade_limit", 500)

        _log.debug("Fetching trades for market %s", condition_id)
        try:
            trades = _fetch_trades(
                condition_id,
                limit=limit,
                timeout=self._timeout,
            )
        except Exception as exc:
            _log.warning("Data API call failed for %s: %s — using NaN features", condition_id, exc)
            trades = []

        return _compute_features_from_trades(trades, reference_dt)

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> float | None:
        """Check if the market is resolved and return 1.0 (UP) or 0.0 (DOWN).

        Queries the Gamma API for the market's resolution status.
        Returns None if the market is still open / unresolved.

        Args:
            spec: The market to check.

        Returns:
            1.0 if resolved UP (Yes wins), 0.0 if resolved DOWN (No wins), None otherwise.
        """
        condition_id = spec.metadata.get("condition_id") or spec.market_id
        _log.debug("Fetching truth for market %s", condition_id)

        try:
            resp = httpx.get(
                f"{self._gamma_base}/markets/{condition_id}",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            raw: dict[str, Any] = resp.json()
        except Exception as exc:
            _log.warning("Could not fetch market status for %s: %s", condition_id, exc)
            return None

        # Check resolved flag
        resolved = raw.get("resolved") or raw.get("closed") or False
        if not resolved:
            return None

        # Determine which outcome won
        resolution_price = raw.get("resolutionPrice")
        if resolution_price is not None:
            # Polymarket sets resolutionPrice=1.0 for YES winner
            try:
                return 1.0 if float(resolution_price) >= 0.5 else 0.0
            except (TypeError, ValueError):
                pass

        # Try outcome prices — the winning outcome has price ~1.0
        import json
        prices_raw = raw.get("outcomePrices") or []
        if isinstance(prices_raw, str):
            try:
                prices_raw = json.loads(prices_raw)
            except Exception:
                prices_raw = []
        outcomes_raw = raw.get("outcomes") or []
        if isinstance(outcomes_raw, str):
            try:
                outcomes_raw = json.loads(outcomes_raw)
            except Exception:
                outcomes_raw = []

        for label, price in zip(outcomes_raw, prices_raw):
            try:
                p = float(price)
            except (TypeError, ValueError):
                continue
            if p >= 0.99:
                label_str = str(label).lower()
                if label_str in ("yes", "up"):
                    return 1.0
                elif label_str in ("no", "down"):
                    return 0.0

        _log.debug("Could not determine resolution for %s — returning None", condition_id)
        return None

    def build_training_row(
        self,
        spec: MarketSpec,
        horizon: str = "live",
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Build a labeled training row combining features and truth.

        Args:
            spec: The resolved market.
            horizon: Decision point (unused for BTC 5-min).

        Returns:
            Dict with features + metadata + target, or None if truth unavailable.
        """
        truth = self.fetch_truth(spec, **kwargs)
        if truth is None:
            _log.debug("Market %s not resolved — skipping training row", spec.market_id)
            return None

        features = self.fetch_features(spec, horizon, **kwargs)

        winning_label = "Yes" if truth == 1.0 else "No"
        market_price = spec.metadata.get("market_price", 0.5)

        return {
            "market_id": spec.market_id,
            "decision_horizon": horizon,
            "winning_label": winning_label,
            "outcome_label": "Yes",   # the bin this row represents (UP)
            "market_price": market_price,
            "target": int(truth),
            **features,
        }

    # ------------------------------------------------------------------
    # Extra helpers
    # ------------------------------------------------------------------

    def is_truth_final(self, spec: MarketSpec, **kwargs: Any) -> bool:
        """BTC 5-min markets resolve quickly — truth is final once resolved."""
        return self.fetch_truth(spec, **kwargs) is not None

    def features_to_frame(self, feature_dicts: list[dict[str, float]]) -> pd.DataFrame:
        """Convert a list of feature dicts into a DataFrame with correct column order."""
        df = pd.DataFrame(feature_dicts, columns=FEATURE_COLS)
        return df
