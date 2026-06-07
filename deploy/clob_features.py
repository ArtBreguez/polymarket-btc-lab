"""CLOB Feature Accumulator for Polymarket BTC Trading Bot.

Buffers CLOB WebSocket events (book and price_change) and computes
real-time microstructure features with zero lag for LightGBM prediction.

Usage:
    from clob_features import get_accumulator
    acc = get_accumulator()
    acc.feed_event(token_id, event_dict)  # called from WS thread
    features = acc.get_features(token_id, window_secs=60)  # called from main thread
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# Feature names returned by get_features (all prefixed with clob_)
FEATURE_NAMES: List[str] = [
    "clob_imb_mean",
    "clob_imb_std",
    "clob_imb_drift",
    "clob_spread_mean",
    "clob_spread_trend",
    "clob_mid_velocity",
    "clob_mid_volatility",
    "clob_activity_rate",
    "clob_depth_trend",
    "clob_ask_pressure",
]

# Maximum buffer duration in seconds
MAX_BUFFER_SECS: float = 180.0


@dataclass
class BookSnapshot:
    """Parsed book event data."""
    timestamp: float
    best_ask: float
    best_bid: float
    mid: float
    spread: float
    imbalance: float
    total_depth: float


@dataclass
class PriceChangeEvent:
    """Parsed price_change event data."""
    timestamp: float
    price: float
    side: str  # "ASK" or "BID"


@dataclass
class TokenBuffer:
    """Thread-safe event buffer for a single token."""
    book_events: List[BookSnapshot] = field(default_factory=list)
    price_events: List[PriceChangeEvent] = field(default_factory=list)


class ClobFeatureAccumulator:
    """Accumulates CLOB WebSocket events and computes real-time features.

    Thread-safe: feed_event() is called from the WebSocket thread,
    get_features() is called from the main prediction thread.

    Events are stored in a time-windowed buffer (last 180s) and pruned
    on each feed call to prevent unbounded memory growth.
    """

    def __init__(self, max_buffer_secs: float = MAX_BUFFER_SECS) -> None:
        self._max_buffer_secs = max_buffer_secs
        self._buffers: Dict[str, TokenBuffer] = defaultdict(TokenBuffer)
        self._lock = threading.Lock()

    def feed_event(self, token_id: str, event: Dict[str, Any]) -> None:
        """Ingest a raw CLOB WebSocket event into the buffer.

        Args:
            token_id: The asset/token identifier.
            event: Raw event dict from the WebSocket. Must have 'event_type'.
        """
        event_type = event.get("event_type", "")
        now = time.time()

        with self._lock:
            buf = self._buffers[token_id]

            if event_type == "book":
                snapshot = self._parse_book_event(event, now)
                if snapshot is not None:
                    buf.book_events.append(snapshot)

            elif event_type == "price_change":
                pc_events = self._parse_price_change_event(event, token_id, now)
                buf.price_events.extend(pc_events)

            # Prune old events beyond max buffer
            self._prune_buffer(buf, now)

    def get_features(self, token_id: str, window_secs: float = 60.0) -> Dict[str, float]:
        """Compute features over the last window_secs seconds.

        Args:
            token_id: The asset/token identifier.
            window_secs: Lookback window in seconds (default 60).

        Returns:
            Dict mapping feature names to float values. Returns all zeros
            if insufficient data (< 5 events or < 5 seconds of history).
        """
        zeros = {name: 0.0 for name in FEATURE_NAMES}
        now = time.time()
        cutoff = now - window_secs

        with self._lock:
            buf = self._buffers.get(token_id)
            if buf is None:
                return zeros

            # Filter book events within window
            books = [b for b in buf.book_events if b.timestamp >= cutoff]
            prices = [p for p in buf.price_events if p.timestamp >= cutoff]

        # Check minimum data requirements
        total_events = len(books) + len(prices)
        if total_events < 5:
            return zeros
        if len(books) < 2:
            return zeros

        time_span = books[-1].timestamp - books[0].timestamp
        if time_span < 5.0:
            return zeros

        # Compute features from book snapshots
        imbalances = np.array([b.imbalance for b in books], dtype=np.float64)
        spreads = np.array([b.spread for b in books], dtype=np.float64)
        mids = np.array([b.mid for b in books], dtype=np.float64)
        depths = np.array([b.total_depth for b in books], dtype=np.float64)
        timestamps = np.array([b.timestamp for b in books], dtype=np.float64)

        # Relative timestamps for regression (seconds from first event)
        t_rel = timestamps - timestamps[0]

        features: Dict[str, float] = {}

        # Imbalance features
        features["clob_imb_mean"] = float(np.mean(imbalances))
        features["clob_imb_std"] = float(np.std(imbalances))
        features["clob_imb_drift"] = float(imbalances[-1] - imbalances[0])

        # Spread features
        features["clob_spread_mean"] = float(np.mean(spreads))
        features["clob_spread_trend"] = self._linear_slope(t_rel, spreads)

        # Mid-price features
        features["clob_mid_velocity"] = self._linear_slope(t_rel, mids)
        mid_diffs = np.diff(mids)
        features["clob_mid_volatility"] = float(np.std(mid_diffs)) if len(mid_diffs) > 0 else 0.0

        # Activity rate (all events per second)
        features["clob_activity_rate"] = float(total_events / time_span)

        # Depth trend
        features["clob_depth_trend"] = self._linear_slope(t_rel, depths)

        # Ask pressure: fraction of ASK price_change events that moved DOWN
        features["clob_ask_pressure"] = self._compute_ask_pressure(prices)

        return features

    def reset_token(self, token_id: str) -> None:
        """Clear all buffered data for a token (e.g., on slot change).

        Args:
            token_id: The asset/token identifier to clear.
        """
        with self._lock:
            if token_id in self._buffers:
                del self._buffers[token_id]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_book_event(event: Dict[str, Any], timestamp: float) -> Optional[BookSnapshot]:
        """Parse a raw book event into a BookSnapshot."""
        asks = event.get("asks", [])
        bids = event.get("bids", [])

        if not asks or not bids:
            return None

        # Best ask = lowest ask price, best bid = highest bid price
        best_ask = float(asks[0]["price"])
        best_bid = float(bids[0]["price"])

        # Sizes at top of book
        ask_sz = float(asks[0].get("size", "0"))
        bid_sz = float(bids[0].get("size", "0"))

        mid = (best_ask + best_bid) / 2.0
        spread = best_ask - best_bid

        # Imbalance: positive = more bid pressure (bullish)
        total_sz = bid_sz + ask_sz
        imbalance = (bid_sz - ask_sz) / total_sz if total_sz > 0 else 0.0

        # Total depth across all levels
        total_depth = sum(float(a.get("size", "0")) for a in asks) + \
                      sum(float(b.get("size", "0")) for b in bids)

        return BookSnapshot(
            timestamp=timestamp,
            best_ask=best_ask,
            best_bid=best_bid,
            mid=mid,
            spread=spread,
            imbalance=imbalance,
            total_depth=total_depth,
        )

    @staticmethod
    def _parse_price_change_event(
        event: Dict[str, Any], token_id: str, timestamp: float
    ) -> List[PriceChangeEvent]:
        """Parse a raw price_change event into PriceChangeEvent list."""
        results: List[PriceChangeEvent] = []
        for pc in event.get("price_changes", []):
            asset_id = pc.get("asset_id", "")
            if asset_id != token_id:
                continue
            price = float(pc.get("price", "0"))
            side = pc.get("side", "UNKNOWN")
            results.append(PriceChangeEvent(timestamp=timestamp, price=price, side=side))
        return results

    def _prune_buffer(self, buf: TokenBuffer, now: float) -> None:
        """Remove events older than max_buffer_secs."""
        cutoff = now - self._max_buffer_secs
        if buf.book_events and buf.book_events[0].timestamp < cutoff:
            # Binary-style prune: find first event within window
            idx = 0
            for i, b in enumerate(buf.book_events):
                if b.timestamp >= cutoff:
                    idx = i
                    break
            else:
                idx = len(buf.book_events)
            buf.book_events = buf.book_events[idx:]

        if buf.price_events and buf.price_events[0].timestamp < cutoff:
            idx = 0
            for i, p in enumerate(buf.price_events):
                if p.timestamp >= cutoff:
                    idx = i
                    break
            else:
                idx = len(buf.price_events)
            buf.price_events = buf.price_events[idx:]

    @staticmethod
    def _linear_slope(t: np.ndarray, y: np.ndarray) -> float:
        """Compute linear regression slope (units of y per second).

        Uses numpy polyfit degree 1 for efficiency.
        """
        if len(t) < 2 or t[-1] == t[0]:
            return 0.0
        try:
            coeffs = np.polyfit(t, y, 1)
            return float(coeffs[0])
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    @staticmethod
    def _compute_ask_pressure(prices: List[PriceChangeEvent]) -> float:
        """Compute ask pressure: fraction of consecutive ASK moves that went DOWN.

        A declining ask = tighter market = bullish pressure.
        Returns 0.5 (neutral) if no ASK price changes available.
        """
        ask_prices = [p.price for p in prices if p.side == "ASK"]
        if len(ask_prices) < 2:
            return 0.0

        moves_down = 0
        total_moves = 0
        for i in range(1, len(ask_prices)):
            diff = ask_prices[i] - ask_prices[i - 1]
            if diff != 0:
                total_moves += 1
                if diff < 0:
                    moves_down += 1

        return float(moves_down / total_moves) if total_moves > 0 else 0.0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_accumulator = ClobFeatureAccumulator()


def get_accumulator() -> ClobFeatureAccumulator:
    """Return the module-level singleton ClobFeatureAccumulator instance."""
    return _accumulator
