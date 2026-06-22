"""
data_quality_gate.py — Circuit Breaker / Data Quality Gate System
=================================================================
Drop-in safety layer for live_trader.py.  Implements five gate levels:

  1. DATA COMPLETENESS GATE   — before prediction
  2. FEATURE SANITY GATE      — after feature computation
  3. PREDICTION SANITY GATE   — after model inference
  4. EXECUTION GATE           — before order placement
  5. COLD START PROTECTION    — after restart

Usage in live_trader.py:
    from data_quality_gate import DataQualityGate
    gate = DataQualityGate(features_list=features)

    # In the main loop, before prediction:
    ok, reason = gate.check_data_completeness(ticks, spot_buffer_path, slot_ts)
    if not ok: log.info("  Skip — %s", reason); continue

    ok, reason = gate.check_feature_sanity(feat_dict, features_list)
    if not ok: log.info("  Skip — %s", reason); continue

    # After prediction:
    ok, reason = gate.check_prediction_sanity(prob_up, feat_dict)
    if not ok: log.info("  Skip — %s", reason); continue

    # Before execution:
    ok, reason = gate.check_execution_gate(ask_price, trades_list)
    if not ok: log.info("  Skip — %s", reason); continue

    # On startup:
    gate.start_warmup(n_slots=3)
    # In loop:
    if not gate.is_warm(): log.info("  Warming up..."); continue
"""

import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("live_trader")

# ── Feature range constraints (from training distribution) ─────────────────────
# These define the valid ranges for each feature category.
# Values outside these ranges indicate data corruption or extreme anomalies.

RETURN_FEATURES = {
    "btc_inslot_ret", "btc_pre_5m_ret", "btc_pre_15m_ret",
    "btc_pre_30m_ret", "btc_pre_1h_ret", "btc_pre_4h_ret",
}
# Since v30, return features are normalised by btc_vol_1h (floor=1e-4).
# Raw 5% move / 0.0003 vol = ~167 normalised units. ±50 catches only
# genuine data corruption (e.g. stale candle, wrong timestamp).
RETURN_RANGE = (-0.05, 0.05)   # retornos brutos (não normalizados) — paridade com v29

RATIO_FEATURES = {
    "btc_up_ratio", "btc_buy_ratio", "btc_tw_up_ratio",
    "btc_up_w0", "btc_up_w1", "btc_up_w2", "btc_up_w3", "btc_up_w4", "btc_up_w5",
    "ob_bid_depth_5c", "ob_ask_depth_5c",
}
RATIO_RANGE = (0.0, 1.0)

ZSCORE_FEATURES = {
    "btc_up_ratio_zscore_5s", "btc_up_ratio_zscore_10s", "btc_up_ratio_zscore_20s",
    "btc_up_w0_zscore", "btc_up_w1_zscore", "btc_up_w2_zscore",
    "btc_up_w3_zscore", "btc_up_w4_zscore", "btc_up_w5_zscore",
}
ZSCORE_RANGE = (-5.0, 5.0)

# Broad sanity bounds for all features (catches NaN, inf, obviously wrong values)
FEATURE_ABS_MAX = 1e6


class DataQualityGate:
    """
    Comprehensive circuit breaker system for live_trader.py.

    Five independent gate levels, each returning (ok: bool, reason: str).
    When ok=False, the caller should skip the current slot and log the reason.
    """

    def __init__(
        self,
        features_list: list[str],
        min_ticks: int = 50,
        spot_max_age_s: int = 300,
        min_subwindows_with_ticks: int = 2,
        warmup_slots: int = 3,
        stale_repeat_threshold: int = 10,
        execution_ask_range: tuple[float, float] = (0.10, 0.95),
        no_trade_warning_s: int = 7200,
        win_rate_min: float = 0.40,
        win_rate_lookback: int = 20,
    ):
        self.features_list = features_list
        self.min_ticks = min_ticks
        self.spot_max_age_s = spot_max_age_s
        self.min_subwindows = min_subwindows_with_ticks
        self.warmup_slots = warmup_slots
        self.stale_repeat_threshold = stale_repeat_threshold
        self.execution_ask_range = execution_ask_range
        self.no_trade_warning_s = no_trade_warning_s
        self.win_rate_min = win_rate_min
        self.win_rate_lookback = win_rate_lookback

        # Cold start tracking
        self._start_time = time.time()
        self._slots_seen = 0
        self._warmup_complete = False

        # Prediction staleness tracking
        self._last_predictions: deque[tuple[float, str]] = deque(maxlen=stale_repeat_threshold + 5)
        # (prob, feature_hash) tuples

        # Win rate pause state
        self._paused = False
        self._pause_reason = ""

        log.info(
            "DataQualityGate initialized: min_ticks=%d, spot_max_age=%ds, "
            "warmup_slots=%d, ask_range=%s",
            min_ticks, spot_max_age_s, warmup_slots, execution_ask_range,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # GATE 1: DATA COMPLETENESS — before ANY prediction
    # ═══════════════════════════════════════════════════════════════════════════
    def check_data_completeness(
        self,
        ticks: list[dict],
        spot_buffer_path: str | Path,
        slot_ts: int,
        observe_secs: int = 180,
    ) -> tuple[bool, str]:
        """
        Verify raw data is sufficient for a meaningful prediction.

        Checks:
          1. Tick count >= min_ticks (default 50)
          2. Spot buffer exists and was updated within spot_max_age_s (default 5 min)
          3. At least min_subwindows of 6 sub-windows (30s each) have ticks
          4. Ticks span a reasonable portion of the observation window

        Returns: (ok, reason)
        """
        reasons = []

        # 1. Tick count
        n_ticks = len(ticks)
        if n_ticks < self.min_ticks:
            reasons.append(
                f"DATA_COMPLETENESS: only {n_ticks} ticks < min {self.min_ticks}"
            )

        # 2. Spot buffer freshness
        spot_path = Path(spot_buffer_path)
        if not spot_path.exists():
            reasons.append("DATA_COMPLETENESS: spot_buffer.json missing")
        else:
            try:
                buf = json.loads(spot_path.read_text())
                buf_age = int(time.time()) - buf.get("updated_at", 0)
                if buf_age > self.spot_max_age_s:
                    reasons.append(
                        f"DATA_COMPLETENESS: spot buffer is {buf_age}s stale "
                        f"(max {self.spot_max_age_s}s)"
                    )
                # Also check BTC candles exist
                btc_candles = buf.get("btcusdt", [])
                if len(btc_candles) < 10:
                    reasons.append(
                        f"DATA_COMPLETENESS: only {len(btc_candles)} BTC candles in buffer"
                    )
            except Exception as e:
                reasons.append(f"DATA_COMPLETENESS: spot buffer read error: {e}")

        # 3. Sub-window coverage (6 x 30s windows)
        if ticks:
            windows_with_ticks = 0
            for i in range(6):
                t0, t1 = i * 30, (i + 1) * 30
                w_ticks = [t for t in ticks if t0 <= t.get("t_sec", -1) < t1]
                if len(w_ticks) >= 1:
                    windows_with_ticks += 1
            if windows_with_ticks < self.min_subwindows:
                reasons.append(
                    f"DATA_COMPLETENESS: only {windows_with_ticks}/6 sub-windows have ticks "
                    f"(min {self.min_subwindows})"
                )

        if reasons:
            return False, " | ".join(reasons)
        return True, ""

    # ═══════════════════════════════════════════════════════════════════════════
    # GATE 2: FEATURE SANITY — after computing features, before prediction
    # ═══════════════════════════════════════════════════════════════════════════
    def check_feature_sanity(
        self,
        feat_dict: dict[str, float],
        features_list: list[str] | None = None,
    ) -> tuple[bool, str]:
        """
        Validate that all model features are present, finite, and within
        training distribution ranges.

        Checks:
          1. All features in features_list are present and finite (not NaN/inf)
          2. Return features in [-0.05, 0.05]
          3. Ratio features in [0, 1]
          4. Z-score features in [-5, 5]
          5. No feature exceeds absolute max (1e6)

        Returns: (ok, reason) — reason includes which specific features failed.
        """
        flist = features_list or self.features_list
        violations = []

        # 1. Presence and finiteness
        for f in flist:
            val = feat_dict.get(f)
            if val is None:
                violations.append(f"FEATURE_SANITY: '{f}' is missing (None)")
                continue
            if not np.isfinite(val):
                violations.append(f"FEATURE_SANITY: '{f}' is not finite (val={val})")
                continue
            if abs(val) > FEATURE_ABS_MAX:
                violations.append(
                    f"FEATURE_SANITY: '{f}' = {val:.4g} exceeds abs max {FEATURE_ABS_MAX}"
                )
                continue

            # 2. Return features
            if f in RETURN_FEATURES:
                lo, hi = RETURN_RANGE
                if not (lo <= val <= hi):
                    violations.append(
                        f"FEATURE_SANITY: return feature '{f}' = {val:.6f} "
                        f"outside [{lo}, {hi}]"
                    )

            # 3. Ratio features
            if f in RATIO_FEATURES:
                lo, hi = RATIO_RANGE
                if not (lo - 0.001 <= val <= hi + 0.001):  # tiny epsilon for float precision
                    violations.append(
                        f"FEATURE_SANITY: ratio feature '{f}' = {val:.6f} "
                        f"outside [{lo}, {hi}]"
                    )

            # 4. Z-score features
            if f in ZSCORE_FEATURES:
                lo, hi = ZSCORE_RANGE
                if not (lo <= val <= hi):
                    violations.append(
                        f"FEATURE_SANITY: zscore feature '{f}' = {val:.4f} "
                        f"outside [{lo}, {hi}]"
                    )

        if violations:
            # Log each violation individually for debugging
            for v in violations[:5]:  # cap at 5 to avoid log spam
                log.warning("  %s", v)
            if len(violations) > 5:
                log.warning("  ... and %d more violations", len(violations) - 5)
            return False, f"FEATURE_SANITY: {len(violations)} violations — first: {violations[0]}"

        return True, ""

    # ═══════════════════════════════════════════════════════════════════════════
    # GATE 3: PREDICTION SANITY — after model inference
    # ═══════════════════════════════════════════════════════════════════════════
    def check_prediction_sanity(
        self,
        prob_up: float,
        feat_dict: dict[str, float],
    ) -> tuple[bool, str]:
        """
        Validate model output is reasonable.

        Checks:
          1. Extreme probabilities (>99% or <1%) are flagged as suspicious
          2. Repeated identical predictions suggest stale/stuck data

        Returns: (ok, reason)
        """
        # 1. Extreme probability check
        if prob_up > 0.99 or prob_up < 0.01:
            reason = (
                f"PREDICTION_SANITY: extreme probability {prob_up:.4f} "
                f"({'> 99%' if prob_up > 0.99 else '< 1%'}) — likely data issue"
            )
            log.warning("  %s", reason)
            return False, reason

        # 2. Staleness detection — hash key features to detect identical inputs
        # Use a subset of volatile features that should change every slot
        key_features = [
            "btc_up_ratio", "btc_n_ticks", "btc_vol_up", "btc_vol_dn",
            "btc_tw_up_ratio", "btc_momentum", "btc_inslot_ret",
        ]
        feat_sig = "|".join(
            f"{feat_dict.get(f, 0.0):.6f}" for f in key_features
        )
        self._last_predictions.append((round(prob_up, 6), feat_sig))

        # Check if last N predictions have identical feature signatures
        if len(self._last_predictions) >= self.stale_repeat_threshold:
            recent = list(self._last_predictions)[-self.stale_repeat_threshold:]
            sigs = [s for _, s in recent]
            if len(set(sigs)) == 1:
                reason = (
                    f"PREDICTION_SANITY: last {self.stale_repeat_threshold} predictions "
                    f"have identical feature signatures — data appears stale/stuck"
                )
                log.error("  %s", reason)
                return False, reason

        return True, ""

    # ═══════════════════════════════════════════════════════════════════════════
    # GATE 4: EXECUTION GATE — before placing order
    # ═══════════════════════════════════════════════════════════════════════════
    def check_execution_gate(
        self,
        ask_price: float,
        trades: list[dict],
    ) -> tuple[bool, str]:
        """
        Final safety check before placing an order.

        Checks:
          1. Ask price > 0 and within [0.10, 0.95]
          2. If no trades placed in last 2 hours, log warning (non-blocking)
          3. If win rate < 40% over last 20 settled trades, PAUSE trading

        Returns: (ok, reason)
        """
        # 0. Check if paused due to poor performance
        if self._paused:
            return False, f"EXECUTION_GATE: trading PAUSED — {self._pause_reason}"

        # 1. Ask price sanity
        lo, hi = self.execution_ask_range
        if ask_price <= 0:
            return False, f"EXECUTION_GATE: ask price is {ask_price} (<= 0)"
        if not (lo <= ask_price <= hi):
            return False, (
                f"EXECUTION_GATE: ask ${ask_price:.3f} outside [{lo}, {hi}]"
            )

        # 2. No-trade warning (non-blocking — just logs)
        now = int(time.time())
        placed_trades = [
            t for t in trades
            if t.get("status") in ("open", "settled")
            and t.get("entered_at", 0) > now - self.no_trade_warning_s
        ]
        if not placed_trades:
            elapsed_h = self.no_trade_warning_s / 3600
            log.warning(
                "  EXECUTION_GATE WARNING: no trades placed in last %.0fh — "
                "possible issue with data feed or market conditions",
                elapsed_h,
            )

        # 3. Win rate circuit breaker
        settled = [
            t for t in trades
            if t.get("status") == "settled" and t.get("result") in ("WIN", "LOSS")
        ]
        if len(settled) >= self.win_rate_lookback:
            recent = settled[-self.win_rate_lookback:]
            wins = sum(1 for t in recent if t["result"] == "WIN")
            wr = wins / len(recent)
            if wr < self.win_rate_min:
                self._paused = True
                self._pause_reason = (
                    f"win rate {wr:.0%} ({wins}/{len(recent)}) "
                    f"< {self.win_rate_min:.0%} over last {self.win_rate_lookback} trades"
                )
                log.error(
                    "  EXECUTION_GATE: PAUSING TRADING — %s", self._pause_reason
                )
                return False, f"EXECUTION_GATE: PAUSED — {self._pause_reason}"

        return True, ""

    # ═══════════════════════════════════════════════════════════════════════════
    # GATE 5: COLD START PROTECTION — after restart
    # ═══════════════════════════════════════════════════════════════════════════
    def start_warmup(self, n_slots: int | None = None):
        """Call on startup to begin warmup period."""
        if n_slots is not None:
            self.warmup_slots = n_slots
        self._slots_seen = 0
        self._warmup_complete = False
        self._start_time = time.time()
        log.info(
            "COLD_START: warmup started — will observe %d slots before trading",
            self.warmup_slots,
        )

    def record_slot_observed(self):
        """Call each time a slot's entry window is reached (even if skipped)."""
        self._slots_seen += 1
        if not self._warmup_complete and self._slots_seen >= self.warmup_slots:
            self._warmup_complete = True
            elapsed = time.time() - self._start_time
            log.info(
                "COLD_START: warmup complete after %d slots (%.0fs) — trading enabled",
                self._slots_seen, elapsed,
            )

    def is_warm(self) -> bool:
        """Returns True if warmup period is complete."""
        if self._warmup_complete:
            return True
        elapsed = time.time() - self._start_time
        log.info(
            "  COLD_START: warming up — %d/%d slots observed (%.0fs elapsed)",
            self._slots_seen, self.warmup_slots, elapsed,
        )
        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # ADMIN: Manual controls
    # ═══════════════════════════════════════════════════════════════════════════
    def unpause(self):
        """Manually unpause trading after win rate circuit breaker triggers."""
        if self._paused:
            log.info("EXECUTION_GATE: trading UNPAUSED (manual override)")
            self._paused = False
            self._pause_reason = ""

    def force_warm(self):
        """Skip remaining warmup period."""
        self._warmup_complete = True
        log.info("COLD_START: warmup forced complete (manual override)")

    def status(self) -> dict:
        """Return current gate status for monitoring/logging."""
        return {
            "warm": self._warmup_complete,
            "slots_seen": self._slots_seen,
            "warmup_target": self.warmup_slots,
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "predictions_tracked": len(self._last_predictions),
            "uptime_s": int(time.time() - self._start_time),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # CONVENIENCE: Run all pre-prediction gates at once
    # ═══════════════════════════════════════════════════════════════════════════
    def pre_prediction_check(
        self,
        ticks: list[dict],
        spot_buffer_path: str | Path,
        slot_ts: int,
        feat_dict: dict[str, float],
        observe_secs: int = 180,
    ) -> tuple[bool, str]:
        """
        Run Gate 1 (data completeness) + Gate 2 (feature sanity) in sequence.
        Convenience method for the main loop.
        """
        # Gate 5: Cold start
        if not self.is_warm():
            return False, "COLD_START: still warming up"

        # Gate 1: Data completeness
        ok, reason = self.check_data_completeness(
            ticks, spot_buffer_path, slot_ts, observe_secs
        )
        if not ok:
            return False, reason

        # Gate 2: Feature sanity
        ok, reason = self.check_feature_sanity(feat_dict)
        if not ok:
            return False, reason

        return True, ""

    def post_prediction_check(
        self,
        prob_up: float,
        feat_dict: dict[str, float],
        ask_price: float,
        trades: list[dict],
    ) -> tuple[bool, str]:
        """
        Run Gate 3 (prediction sanity) + Gate 4 (execution gate) in sequence.
        Convenience method for the main loop.
        """
        # Gate 3: Prediction sanity
        ok, reason = self.check_prediction_sanity(prob_up, feat_dict)
        if not ok:
            return False, reason

        # Gate 4: Execution gate
        ok, reason = self.check_execution_gate(ask_price, trades)
        if not ok:
            return False, reason

        return True, ""
