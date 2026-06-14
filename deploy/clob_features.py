"""CLOB Feature Accumulator for Polymarket BTC Trading Bot.

Buffers CLOB WebSocket events (book and price_change) and computes
real-time microstructure features with zero lag for LightGBM prediction.

All time-windowed methods accept slot_ts so windows are anchored to the
slot start — exactly matching how fetch_ob_features_modal.py computes
training features. This eliminates train/live divergence.

Usage:
    from clob_features import get_accumulator
    acc = get_accumulator()
    acc.feed_event(token_id, event_dict)  # called from WS thread

    # At prediction time (t≈170-240s into slot):
    clob_feats  = acc.get_features(token_id, slot_ts, obs_secs=60, window_secs=60)
    pc_feats    = acc.get_ob_pc_features(token_id, slot_ts, cutoff_secs=168)
    imb_windows = acc.get_windowed_imbalance(token_id, slot_ts, windows=[(0,60),(60,120),(120,168)])
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Feature names returned by get_features — only the 5 OK ones
# Removed: clob_imb_mean/std/drift (❌ quase sempre 0.0 live, 1 WS book snap)
#          clob_depth_trend (❌ idem)
#          clob_activity_rate (⚠️ levemente diferente do treino)
FEATURE_NAMES: List[str] = [
    "clob_spread_mean",
    "clob_spread_trend",
    "clob_mid_velocity",
    "clob_mid_volatility",
    "clob_ask_pressure",
]

# Feature names returned by get_ob_pc_features
OB_PC_FEATURE_NAMES: List[str] = [
    "ob_pc_up_ratio",
    "ob_pc_volatility",
    "ob_pc_count",
    "ob_fill_imbalance",
]

# Maximum buffer duration in seconds — must be >= slot duration (300s) so
# we retain enough events to cover the full ob_pc window [0, 168s).
MAX_BUFFER_SECS: float = 360.0


@dataclass
class BookSnapshot:
    """Parsed book event data."""
    timestamp: float      # wall-clock time (time.time())
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
    timestamp: float      # wall-clock time (time.time())
    price: float
    side: str             # "ASK" or "BID"
    size: float = 0.0     # fill size (pc_size from price_change, if available)


@dataclass
class TokenBuffer:
    """Thread-safe event buffer for a single token."""
    book_events: List[BookSnapshot] = field(default_factory=list)
    price_events: List[PriceChangeEvent] = field(default_factory=list)
    slot_ts: int = 0      # slot this buffer was last reset for


class ClobFeatureAccumulator:
    """Accumulates CLOB WebSocket events and computes real-time features.

    Thread-safe: feed_event() is called from the WebSocket thread,
    get_features() is called from the main prediction thread.

    All windowed methods use slot_ts as the epoch so [t0, t1) is
    [slot_ts+t0, slot_ts+t1) in wall-clock time — matching training.
    """

    def __init__(self, max_buffer_secs: float = MAX_BUFFER_SECS) -> None:
        self._max_buffer_secs = max_buffer_secs
        self._buffers: Dict[str, TokenBuffer] = defaultdict(TokenBuffer)
        self._lock = threading.Lock()

    def feed_event(self, token_id: str, event: Dict[str, Any]) -> None:
        """Ingest a raw CLOB WebSocket event into the buffer."""
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

    def reset_token(self, token_id: str, slot_ts: int = 0) -> None:
        """Reanchor buffer to a new slot_ts WITHOUT clearing events.

        Eventos acumulados são mantidos — o obs_secs dinâmico no live_trader.py
        garante que só eventos do slot atual sejam usados via janela [slot_ts, now+5).
        Limpar o buffer aqui causava CLOB features sempre zero: o WS levava 60-170s
        para remandar o book snapshot do novo token, e a janela [slot_ts, slot_ts+60)
        ficava vazia no momento da decisão.

        Args:
            token_id: The asset/token identifier to reanchor.
            slot_ts: New slot timestamp (stored for reference).
        """
        with self._lock:
            buf = self._buffers[token_id]
            # Não limpar — só reancorar o slot_ts.
            # Eventos antigos são automaticamente excluídos pela janela dinâmica
            # [slot_ts, now+5) em get_features() e pelo prune do MAX_BUFFER_SECS.
            buf.slot_ts = slot_ts

    # ------------------------------------------------------------------
    # Primary feature methods (all slot_ts-anchored)
    # ------------------------------------------------------------------

    def get_features(
        self,
        token_id: str,
        slot_ts: int,
        obs_secs: int = 60,
        window_secs: float = 60.0,
    ) -> Dict[str, float]:
        """Compute clob_* features over a slot-anchored window.

        Window: wall-clock [slot_ts + obs_secs - window_secs, slot_ts + obs_secs).
        Matches training: fetch_ob_features_modal._compute_clob_features uses
        t=[108, 168) (window_secs=60, obs_secs=168 by default → same semantics).

        At prediction time (t≈170s), obs_secs=168 gives window [108, 168) —
        identical to training. Default obs_secs=60 is kept for backward compat.

        Args:
            token_id: The asset/token identifier.
            slot_ts: Slot start timestamp (UTC seconds).
            obs_secs: Observation cutoff relative to slot_ts (default 60; use 168 for
                      full parity with training window).
            window_secs: How many seconds before obs_secs to look back (default 60).

        Returns:
            Dict of clob_* features. All zeros if insufficient data.
        """
        zeros = {name: 0.0 for name in FEATURE_NAMES}

        cutoff_wall   = slot_ts + obs_secs          # wall-clock upper bound (exclusive)
        window_wall   = cutoff_wall - window_secs    # wall-clock lower bound (inclusive)

        with self._lock:
            buf = self._buffers.get(token_id)
            if buf is None:
                return zeros
            books  = [b for b in buf.book_events  if window_wall <= b.timestamp < cutoff_wall]
            prices = [p for p in buf.price_events if window_wall <= p.timestamp < cutoff_wall]

        real_books = [b for b in books if not b.synthetic]
        total_events = len(real_books) + len(prices)
        if total_events < 5:
            return zeros

        # Time span
        all_ts_wall = sorted([b.timestamp for b in books] + [p.timestamp for p in prices])
        time_span = max(all_ts_wall[-1] - all_ts_wall[0], 1e-3) if len(all_ts_wall) >= 2 else 1.0

        # Real books for imbalance/depth
        imbalances = np.array([b.imbalance    for b in real_books], dtype=np.float64)
        depths     = np.array([b.total_depth  for b in real_books], dtype=np.float64)
        t_real     = np.array([b.timestamp    for b in real_books], dtype=np.float64)

        # All events (real + synthetic books + pc synthetic mid/spread) for velocity
        all_mid:    List[float] = []
        all_spread: List[float] = []
        all_ts:     List[float] = []
        ask_prices_seq: List[float] = []

        for b in books:
            all_mid.append(b.mid)
            all_spread.append(b.spread)
            all_ts.append(b.timestamp)

        for p in prices:
            # price_change events don't carry best_bid/best_ask directly here
            # (those are already captured in the synthetic BookSnapshot above).
            # We use them only for ask_pressure.
            if p.side == "ASK":
                ask_prices_seq.append(p.price)

        if len(all_mid) < 2:
            return zeros

        order          = np.argsort(all_ts)
        all_ts_arr     = np.array(all_ts)[order]
        all_mid_arr    = np.array(all_mid)[order]
        all_spread_arr = np.array(all_spread)[order]
        t_rel          = all_ts_arr - all_ts_arr[0]

        features: Dict[str, float] = {}

        features["clob_imb_mean"]  = float(np.mean(imbalances)) if len(imbalances) > 0 else 0.0
        features["clob_imb_std"]   = float(np.std(imbalances))  if len(imbalances) > 1 else 0.0
        features["clob_imb_drift"] = float(imbalances[-1] - imbalances[0]) if len(imbalances) > 1 else 0.0

        features["clob_spread_mean"]  = float(np.mean(all_spread_arr))
        features["clob_spread_trend"] = self._linear_slope(t_rel, all_spread_arr) if len(t_rel) > 1 else 0.0

        features["clob_mid_velocity"]   = self._linear_slope(t_rel, all_mid_arr) if len(t_rel) > 1 else 0.0
        mid_diffs = np.diff(all_mid_arr)
        features["clob_mid_volatility"] = float(np.std(mid_diffs)) if len(mid_diffs) > 0 else 0.0

        # Activity rate: real book events + price_change events (no double-count)
        features["clob_activity_rate"] = float((len(real_books) + len(prices)) / time_span)

        # Depth trend — real books only
        if len(depths) > 1:
            t_real_rel = t_real - t_real[0]
            features["clob_depth_trend"] = self._linear_slope(t_real_rel, depths)
        else:
            features["clob_depth_trend"] = 0.0

        # Ask pressure: fraction of ASK price changes that went DOWN
        features["clob_ask_pressure"] = self._compute_ask_pressure_list(ask_prices_seq)

        return features

    def get_ob_pc_features(
        self,
        token_id: str,
        slot_ts: int,
        cutoff_secs: float = 168.0,
    ) -> Dict[str, float]:
        """Compute ob_pc_* features from price_change events since slot_ts.

        Window: wall-clock [slot_ts, slot_ts + cutoff_secs).
        Matches training: fetch_ob_features_modal uses t=[0, CUTOFF_SEC=168).

        ob_pc_up_ratio   = n_up_moves / n_total_moves   (price changes going UP)
        ob_pc_volatility = std(price_diffs)
        ob_pc_count      = n_total price change moves
        ob_fill_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)
                           where BUY = side=="BID", SELL = side=="ASK"

        Args:
            token_id: The asset/token identifier.
            slot_ts: Slot start timestamp (UTC seconds).
            cutoff_secs: Upper bound relative to slot_ts (default 168s = CUTOFF_SEC).

        Returns:
            Dict with ob_pc_* features. Neutral fills if no data.
        """
        neutral = {
            "ob_pc_up_ratio":    0.5,
            "ob_pc_volatility":  0.0,
            "ob_pc_count":       0.0,
            "ob_fill_imbalance": 0.0,
        }

        t_lo = float(slot_ts)
        t_hi = float(slot_ts) + cutoff_secs

        with self._lock:
            buf = self._buffers.get(token_id)
            if buf is None:
                return neutral
            prices = [p for p in buf.price_events if t_lo <= p.timestamp < t_hi]

        if not prices:
            return neutral

        price_vals = np.array([p.price for p in prices], dtype=np.float64)
        diffs      = np.diff(price_vals)
        changes    = diffs[diffs != 0]

        if len(changes) == 0:
            result = dict(neutral)
            result["ob_pc_count"] = float(len(prices))
            return result

        n_up = int((changes > 0).sum())
        n_total = len(changes)

        # Fill imbalance: BID-side fills = buys, ASK-side = sells
        buy_vol  = sum(p.size for p in prices if p.side == "BID")
        sell_vol = sum(p.size for p in prices if p.side == "ASK")
        fill_imb = float((buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8))

        return {
            "ob_pc_up_ratio":    float(n_up / n_total),
            "ob_pc_volatility":  float(np.std(changes)),
            "ob_pc_count":       float(n_total),
            "ob_fill_imbalance": fill_imb,
        }

    def get_windowed_imbalance(
        self,
        token_id: str,
        slot_ts: int,
        windows: List[Tuple[float, float]] = ((0, 60), (60, 120), (120, 168)),
    ) -> Dict[str, float]:
        """Compute ob_imb_w0/w1/w2 from real book snapshots in slot-anchored windows.

        Window i covers wall-clock [slot_ts + t0_i, slot_ts + t1_i).
        Uses ONLY real (non-synthetic) book snapshots — matching training.

        Returns mean imbalance in each window, or 0.0 if no real books present.
        Falls back to 0.0 (not interpolated) so behaviour is honest when data absent.

        Args:
            token_id: The asset/token identifier.
            slot_ts: Slot start timestamp (UTC seconds).
            windows: List of (t_start, t_end) tuples relative to slot_ts.

        Returns:
            Dict with ob_imb_w{i} keys.
        """
        result: Dict[str, float] = {}
        t_base = float(slot_ts)

        with self._lock:
            buf = self._buffers.get(token_id)
            if buf is None:
                for i in range(len(windows)):
                    result[f"ob_imb_w{i}"] = 0.0
                return result
            real_books = [b for b in buf.book_events if not b.synthetic]

        for i, (t0, t1) in enumerate(windows):
            lo = t_base + t0
            hi = t_base + t1
            imbs = [b.imbalance for b in real_books if lo <= b.timestamp < hi]
            result[f"ob_imb_w{i}"] = float(np.mean(imbs)) if imbs else 0.0

        return result

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

        best_ask = float(asks[0]["price"])
        best_bid = float(bids[0]["price"])
        ask_sz   = float(asks[0].get("size", "0"))
        bid_sz   = float(bids[0].get("size", "0"))

        mid    = (best_ask + best_bid) / 2.0
        spread = best_ask - best_bid

        total_sz  = bid_sz + ask_sz
        imbalance = (bid_sz - ask_sz) / total_sz if total_sz > 0 else 0.0

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
        synthetic BookSnapshots so mid/spread time-series are available even with
        only 1 real book snapshot per subscription.

        We also capture `size` from the price_change for ob_fill_imbalance.
        """
        results: List[PriceChangeEvent] = []
        synth_books: List[BookSnapshot] = []

        for pc in event.get("price_changes", []):
            asset_id = pc.get("asset_id", "")
            if asset_id != token_id:
                continue
            price    = float(pc.get("price", "0"))
            raw_side = pc.get("side", "UNKNOWN")
            # Polymarket uses BUY/SELL — map to BID/ASK
            side = "BID" if raw_side == "BUY" else ("ASK" if raw_side == "SELL" else raw_side)
            size = float(pc.get("size", "0") or "0")

            results.append(PriceChangeEvent(
                timestamp=timestamp,
                price=price,
                side=side,
                size=size,
            ))

            # Synthetic BookSnapshot from best_bid / best_ask.
            # IMPORTANT: imbalance and total_depth are NOT available in price_change
            # events — set to 0.0 so synthetic books are excluded from imbalance/depth
            # features (handled by the synthetic flag).
            raw_bid = pc.get("best_bid")
            raw_ask = pc.get("best_ask")
            if raw_bid is not None and raw_ask is not None:
                try:
                    best_bid = float(raw_bid)
                    best_ask = float(raw_ask)
                    if best_ask > best_bid > 0:
                        synth_books.append(BookSnapshot(
                            timestamp=timestamp,
                            best_ask=best_ask,
                            best_bid=best_bid,
                            mid=(best_ask + best_bid) / 2.0,
                            spread=best_ask - best_bid,
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

        def _trim(lst):
            if lst and lst[0].timestamp < cutoff:
                idx = next((i for i, e in enumerate(lst) if e.timestamp >= cutoff), len(lst))
                del lst[:idx]

        _trim(buf.book_events)
        _trim(buf.price_events)

    @staticmethod
    def _linear_slope(t: np.ndarray, y: np.ndarray) -> float:
        """Compute linear regression slope (units of y per second)."""
        if len(t) < 2 or t[-1] == t[0]:
            return 0.0
        try:
            return float(np.polyfit(t, y, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    @staticmethod
    def _compute_ask_pressure_list(ask_prices: List[float]) -> float:
        """Compute ask pressure: fraction of consecutive ASK moves that went DOWN."""
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
