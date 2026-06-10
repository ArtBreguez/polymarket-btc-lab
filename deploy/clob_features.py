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
    synthetic: bool = False  # True = built from price_change best_bid/best_ask (no imbalance/depth data)


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
                pc_events, synth_books = self._parse_price_change_event(event, token_id, now)
                buf.price_events.extend(pc_events)
                # Synthetic BookSnapshots built from best_bid/best_ask in price_change.
                # This gives us a mid/spread time-series even when Polymarket only
                # sends 1 full book snapshot per subscription.
                buf.book_events.extend(synth_books)

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

        # Check minimum data requirements (count only real events, no double-counting)
        real_books_all = [b for b in books if not b.synthetic]
        total_events = len(real_books_all) + len(prices)
        if total_events < 5:
            return zeros

        # Polymarket WS sends 1 book snapshot on subscribe then only price_change events.
        # Relax: allow 1 book snapshot as long as we have enough price_change events (>=5).
        # Use price_events timestamps for time_span when only 1 book snapshot available.
        if len(books) < 2:
            if len(prices) < 5:
                return zeros
            time_span = prices[-1].timestamp - prices[0].timestamp
            if time_span < 5.0:
                return zeros
        else:
            time_span = books[-1].timestamp - books[0].timestamp
            if time_span < 5.0:
                # Fall back to price_events span
                if len(prices) >= 2:
                    time_span = prices[-1].timestamp - prices[0].timestamp
                if time_span < 5.0:
                    return zeros

        # Compute features from book snapshots.
        # Synthetic snapshots (built from price_change best_bid/best_ask) have real
        # mid/spread but NO imbalance or depth data — separate them to avoid pollution.
        real_books  = [b for b in books if not b.synthetic]
        all_books   = books  # includes synthetic — used for mid/spread time-series

        # Use all books for mid/spread (synthetic have real best_bid/best_ask)
        spreads    = np.array([b.spread for b in all_books], dtype=np.float64)
        mids       = np.array([b.mid    for b in all_books], dtype=np.float64)
        timestamps = np.array([b.timestamp for b in all_books], dtype=np.float64)
        t_rel      = timestamps - timestamps[0]

        # Use ONLY real books for imbalance and depth (synthetic have fabricated zeros)
        imbalances = np.array([b.imbalance    for b in real_books], dtype=np.float64)
        depths     = np.array([b.total_depth  for b in real_books], dtype=np.float64)
        t_real     = np.array([b.timestamp    for b in real_books], dtype=np.float64)
        t_real_rel = t_real - t_real[0] if len(t_real) > 1 else t_real

        features: Dict[str, float] = {}

        # Imbalance features — from real book snapshots only
        features["clob_imb_mean"]  = float(np.mean(imbalances))  if len(imbalances) > 0 else 0.0
        features["clob_imb_std"]   = float(np.std(imbalances))   if len(imbalances) > 1 else 0.0
        features["clob_imb_drift"] = float(imbalances[-1] - imbalances[0]) if len(imbalances) > 1 else 0.0

        # Spread features — from all books (synthetic have real spread)
        features["clob_spread_mean"]  = float(np.mean(spreads))
        features["clob_spread_trend"] = self._linear_slope(t_rel, spreads) if len(spreads) > 1 else 0.0

        # Mid-price features — from all books (synthetic have real mid)
        features["clob_mid_velocity"]   = self._linear_slope(t_rel, mids) if len(mids) > 1 else 0.0
        mid_diffs = np.diff(mids)
        features["clob_mid_volatility"] = float(np.std(mid_diffs)) if len(mid_diffs) > 0 else 0.0

        # Activity rate: real events per second.
        # Synthetic book snapshots are derived from price_change events — counting
        # them would double-count every price_change and inflate activity_rate ~2x
        # vs training (where synthetics didn't exist). Use only real book snapshots
        # + price_change events to match the training distribution.
        real_event_count = len(real_books) + len(prices)
        features["clob_activity_rate"] = float(real_event_count / time_span)

        # Depth trend — from real book snapshots only
        features["clob_depth_trend"] = self._linear_slope(t_real_rel, depths) if len(depths) > 1 else 0.0

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
    ) -> tuple:
        """Parse a raw price_change event into (PriceChangeEvent list, BookSnapshot list).

        Polymarket WS sends `side` as "BUY"/"SELL" (not "BID"/"ASK").
        Each item also includes `best_bid` and `best_ask` — we use those to build
        synthetic BookSnapshots so features can be computed even with only 1 real
        book snapshot (which is all Polymarket sends on subscribe).
        """
        results: List[PriceChangeEvent] = []
        synth_books: List[BookSnapshot] = []

        for pc in event.get("price_changes", []):
            asset_id = pc.get("asset_id", "")
            if asset_id != token_id:
                continue
            price = float(pc.get("price", "0"))
            raw_side = pc.get("side", "UNKNOWN")
            # Polymarket uses BUY/SELL — map to BID/ASK for our feature logic
            side = "BID" if raw_side == "BUY" else ("ASK" if raw_side == "SELL" else raw_side)
            results.append(PriceChangeEvent(timestamp=timestamp, price=price, side=side))

            # Build synthetic BookSnapshot from best_bid / best_ask if present.
            # These are sent on every price_change and give us a real time-series of
            # mid / spread without waiting for a new full book event.
            # IMPORTANT: imbalance and total_depth are NOT available in price_change
            # events — we set them to 0.0 (neutral) rather than fabricating values.
            # The model was trained with real imbalance from the full book; injecting
            # fake ±0.3 values causes distribution shift on clob_imb_* features.
            # Zero is the least-bad neutral value (model treats it as no imbalance signal).
            raw_bid = pc.get("best_bid")
            raw_ask = pc.get("best_ask")
            if raw_bid is not None and raw_ask is not None:
                try:
                    best_bid = float(raw_bid)
                    best_ask = float(raw_ask)
                    if best_ask > best_bid > 0:
                        mid = (best_ask + best_bid) / 2.0
                        spread = best_ask - best_bid
                        synth_books.append(BookSnapshot(
                            timestamp=timestamp,
                            best_ask=best_ask,
                            best_bid=best_bid,
                            mid=mid,
                            spread=spread,
                            imbalance=0.0,    # unknown — not in price_change events
                            total_depth=0.0,  # unknown — not in price_change events
                            synthetic=True,
                        ))
                except (ValueError, TypeError):
                    pass

        return results, synth_books

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
