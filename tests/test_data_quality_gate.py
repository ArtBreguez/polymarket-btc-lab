"""
Comprehensive tests for deploy/data_quality_gate.py — DataQualityGate
"""

import json
import math
import time

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))

from data_quality_gate import DataQualityGate, FEATURE_ABS_MAX


# ── Helpers ──────────────────────────────────────────────────────────────────

MINIMAL_FEATURES = [
    "btc_inslot_ret",   # return feature
    "btc_up_ratio",     # ratio feature
    "btc_up_ratio_zscore_5s",  # zscore feature
    "btc_n_ticks",      # generic feature
]


def _make_gate(**kw) -> DataQualityGate:
    """Create a gate with minimal features and force-warm it."""
    g = DataQualityGate(features_list=MINIMAL_FEATURES, **kw)
    g.force_warm()
    return g


def _good_feat_dict() -> dict:
    return {
        "btc_inslot_ret": 0.001,
        "btc_up_ratio": 0.52,
        "btc_up_ratio_zscore_5s": 0.3,
        "btc_n_ticks": 200.0,
    }


def _make_ticks(n: int = 100) -> list[dict]:
    """Generate ticks spread across 6 sub-windows (0-180s)."""
    return [{"t_sec": (i * 170) / n, "price": 100.0} for i in range(n)]


def _write_spot_buffer(path, updated_at=None, n_candles=50):
    """Write a valid spot buffer JSON file."""
    if updated_at is None:
        updated_at = int(time.time())
    data = {
        "updated_at": updated_at,
        "btcusdt": [{"o": 100, "h": 101, "l": 99, "c": 100.5}] * n_candles,
    }
    path.write_text(json.dumps(data))


# ═════════════════════════════════════════════════════════════════════════════
# GATE 1: check_data_completeness
# ═════════════════════════════════════════════════════════════════════════════

class TestCheckDataCompleteness:

    def test_sufficient_data_passes(self, tmp_path):
        """Good ticks, fresh buffer, good coverage → (True, '')"""
        g = _make_gate()
        buf = tmp_path / "spot_buffer.json"
        _write_spot_buffer(buf)
        ticks = _make_ticks(100)
        ok, reason = g.check_data_completeness(ticks, str(buf), int(time.time()))
        assert ok is True
        assert reason == ""

    def test_too_few_ticks_fails(self, tmp_path):
        """< 50 ticks → fails"""
        g = _make_gate(min_ticks=50)
        buf = tmp_path / "spot_buffer.json"
        _write_spot_buffer(buf)
        ticks = _make_ticks(10)
        ok, reason = g.check_data_completeness(ticks, str(buf), int(time.time()))
        assert ok is False
        assert "only 10 ticks" in reason

    def test_missing_spot_buffer_fails(self, tmp_path):
        g = _make_gate()
        missing = tmp_path / "nonexistent.json"
        ticks = _make_ticks(100)
        ok, reason = g.check_data_completeness(ticks, str(missing), int(time.time()))
        assert ok is False
        assert "missing" in reason.lower()

    def test_stale_spot_buffer_fails(self, tmp_path):
        g = _make_gate(spot_max_age_s=300)
        buf = tmp_path / "spot_buffer.json"
        _write_spot_buffer(buf, updated_at=int(time.time()) - 600)
        ticks = _make_ticks(100)
        ok, reason = g.check_data_completeness(ticks, str(buf), int(time.time()))
        assert ok is False
        assert "stale" in reason.lower()

    def test_too_few_btc_candles_fails(self, tmp_path):
        g = _make_gate()
        buf = tmp_path / "spot_buffer.json"
        _write_spot_buffer(buf, n_candles=5)
        ticks = _make_ticks(100)
        ok, reason = g.check_data_completeness(ticks, str(buf), int(time.time()))
        assert ok is False
        assert "candles" in reason.lower()

    def test_corrupt_spot_buffer_fails(self, tmp_path):
        g = _make_gate()
        buf = tmp_path / "spot_buffer.json"
        buf.write_text("NOT VALID JSON {{{")
        ticks = _make_ticks(100)
        ok, reason = g.check_data_completeness(ticks, str(buf), int(time.time()))
        assert ok is False
        assert "read error" in reason.lower()

    def test_insufficient_subwindow_coverage_fails(self, tmp_path):
        """All ticks in one 30s window → only 1/6 coverage < min 3"""
        g = _make_gate(min_subwindows_with_ticks=3)
        buf = tmp_path / "spot_buffer.json"
        _write_spot_buffer(buf)
        # All ticks in first 30s window
        ticks = [{"t_sec": 5.0, "price": 100.0} for _ in range(100)]
        ok, reason = g.check_data_completeness(ticks, str(buf), int(time.time()))
        assert ok is False
        assert "sub-windows" in reason.lower()

    def test_multiple_failures_combined(self, tmp_path):
        """Few ticks + missing buffer → both failures reported"""
        g = _make_gate()
        missing = tmp_path / "nonexistent.json"
        ticks = _make_ticks(5)
        ok, reason = g.check_data_completeness(ticks, str(missing), int(time.time()))
        assert ok is False
        assert "ticks" in reason.lower()
        assert "missing" in reason.lower()


# ═════════════════════════════════════════════════════════════════════════════
# GATE 2: check_feature_sanity
# ═════════════════════════════════════════════════════════════════════════════

class TestCheckFeatureSanity:

    def test_all_features_valid_passes(self):
        g = _make_gate()
        ok, reason = g.check_feature_sanity(_good_feat_dict())
        assert ok is True
        assert reason == ""

    def test_missing_feature_none_fails(self):
        g = _make_gate()
        fd = _good_feat_dict()
        del fd["btc_n_ticks"]
        ok, reason = g.check_feature_sanity(fd)
        assert ok is False
        assert "missing" in reason.lower()

    def test_nan_value_fails(self):
        g = _make_gate()
        fd = _good_feat_dict()
        fd["btc_n_ticks"] = float("nan")
        ok, reason = g.check_feature_sanity(fd)
        assert ok is False
        assert "not finite" in reason.lower()

    def test_inf_value_fails(self):
        g = _make_gate()
        fd = _good_feat_dict()
        fd["btc_n_ticks"] = float("inf")
        ok, reason = g.check_feature_sanity(fd)
        assert ok is False
        assert "not finite" in reason.lower()

    def test_feature_exceeds_abs_max_fails(self):
        g = _make_gate()
        fd = _good_feat_dict()
        fd["btc_n_ticks"] = FEATURE_ABS_MAX + 1
        ok, reason = g.check_feature_sanity(fd)
        assert ok is False
        assert "abs max" in reason.lower()

    def test_return_feature_out_of_range_fails(self):
        g = _make_gate()
        fd = _good_feat_dict()
        fd["btc_inslot_ret"] = 0.10  # > 0.05
        ok, reason = g.check_feature_sanity(fd)
        assert ok is False
        assert "return feature" in reason.lower()

    def test_ratio_feature_out_of_range_fails(self):
        g = _make_gate()
        fd = _good_feat_dict()
        fd["btc_up_ratio"] = 1.5  # > 1.0 + epsilon
        ok, reason = g.check_feature_sanity(fd)
        assert ok is False
        assert "ratio feature" in reason.lower()

    def test_boundary_values_pass(self):
        """Boundary values 0.0 and 1.0 for ratio features should pass."""
        g = _make_gate()
        fd = _good_feat_dict()
        fd["btc_up_ratio"] = 0.0
        ok, _ = g.check_feature_sanity(fd)
        assert ok is True

        fd["btc_up_ratio"] = 1.0
        ok, _ = g.check_feature_sanity(fd)
        assert ok is True

    def test_float_epsilon_passes(self):
        """1.0005 should still pass for ratio features (epsilon 0.001)."""
        g = _make_gate()
        fd = _good_feat_dict()
        fd["btc_up_ratio"] = 1.0005  # within 0.001 epsilon
        ok, _ = g.check_feature_sanity(fd)
        assert ok is True

    def test_zscore_out_of_range_fails(self):
        g = _make_gate()
        fd = _good_feat_dict()
        fd["btc_up_ratio_zscore_5s"] = 6.0  # > 5.0
        ok, reason = g.check_feature_sanity(fd)
        assert ok is False
        assert "zscore" in reason.lower()


# ═════════════════════════════════════════════════════════════════════════════
# GATE 3: check_prediction_sanity
# ═════════════════════════════════════════════════════════════════════════════

class TestCheckPredictionSanity:

    def test_normal_prob_passes(self):
        g = _make_gate()
        ok, reason = g.check_prediction_sanity(0.55, _good_feat_dict())
        assert ok is True
        assert reason == ""

    def test_extreme_high_fails(self):
        g = _make_gate()
        ok, reason = g.check_prediction_sanity(0.995, _good_feat_dict())
        assert ok is False
        assert "extreme" in reason.lower()
        assert "> 99%" in reason

    def test_extreme_low_fails(self):
        g = _make_gate()
        ok, reason = g.check_prediction_sanity(0.005, _good_feat_dict())
        assert ok is False
        assert "extreme" in reason.lower()
        assert "< 1%" in reason

    def test_boundary_099_passes(self):
        """0.99 is exactly at boundary → should pass (> 0.99 fails)."""
        g = _make_gate()
        ok, _ = g.check_prediction_sanity(0.99, _good_feat_dict())
        assert ok is True

    def test_boundary_001_passes(self):
        """0.01 is exactly at boundary → should pass (< 0.01 fails)."""
        g = _make_gate()
        ok, _ = g.check_prediction_sanity(0.01, _good_feat_dict())
        assert ok is True

    def test_staleness_detection(self):
        """Identical feature signatures repeated → stale detection."""
        g = _make_gate(stale_repeat_threshold=5)
        fd = _good_feat_dict()
        # Feed identical predictions
        for i in range(4):
            ok, _ = g.check_prediction_sanity(0.55, fd)
            assert ok is True
        # 5th identical should trigger staleness
        ok, reason = g.check_prediction_sanity(0.55, fd)
        assert ok is False
        assert "stale" in reason.lower()

    def test_varying_features_no_staleness(self):
        """Different feature values each time → no staleness."""
        g = _make_gate(stale_repeat_threshold=5)
        for i in range(10):
            fd = _good_feat_dict()
            fd["btc_up_ratio"] = 0.50 + i * 0.01  # vary
            ok, _ = g.check_prediction_sanity(0.55, fd)
            assert ok is True


# ═════════════════════════════════════════════════════════════════════════════
# GATE 4: check_execution_gate
# ═════════════════════════════════════════════════════════════════════════════

class TestCheckExecutionGate:

    def test_normal_ask_passes(self):
        g = _make_gate()
        ok, reason = g.check_execution_gate(0.55, [])
        assert ok is True
        assert reason == ""

    def test_ask_lte_zero_fails(self):
        g = _make_gate()
        ok, reason = g.check_execution_gate(0.0, [])
        assert ok is False
        assert "<= 0" in reason

    def test_negative_ask_fails(self):
        g = _make_gate()
        ok, reason = g.check_execution_gate(-0.5, [])
        assert ok is False
        assert "<= 0" in reason

    def test_ask_below_range_fails(self):
        g = _make_gate(execution_ask_range=(0.10, 0.95))
        ok, reason = g.check_execution_gate(0.05, [])
        assert ok is False
        assert "outside" in reason.lower()

    def test_ask_above_range_fails(self):
        g = _make_gate(execution_ask_range=(0.10, 0.95))
        ok, reason = g.check_execution_gate(0.98, [])
        assert ok is False
        assert "outside" in reason.lower()

    def test_trading_paused_fails(self):
        g = _make_gate()
        g._paused = True
        g._pause_reason = "manual pause"
        ok, reason = g.check_execution_gate(0.55, [])
        assert ok is False
        assert "PAUSED" in reason

    def test_win_rate_below_threshold_pauses(self):
        """Win rate < 40% over 20 settled trades → pauses trading."""
        g = _make_gate(win_rate_min=0.40, win_rate_lookback=20)
        # Create 20 settled trades: 5 wins, 15 losses = 25% WR
        trades = []
        for i in range(5):
            trades.append({"status": "settled", "result": "WIN", "entered_at": int(time.time())})
        for i in range(15):
            trades.append({"status": "settled", "result": "LOSS", "entered_at": int(time.time())})
        ok, reason = g.check_execution_gate(0.55, trades)
        assert ok is False
        assert "PAUSED" in reason
        assert g._paused is True

    def test_fewer_than_lookback_trades_no_wr_check(self):
        """With < 20 settled trades, win rate check is skipped."""
        g = _make_gate(win_rate_lookback=20)
        # Only 10 settled trades, all losses
        trades = [
            {"status": "settled", "result": "LOSS", "entered_at": int(time.time())}
            for _ in range(10)
        ]
        ok, reason = g.check_execution_gate(0.55, trades)
        assert ok is True

    def test_ask_at_boundaries_passes(self):
        """Ask at exact boundary values should pass."""
        g = _make_gate(execution_ask_range=(0.10, 0.95))
        ok, _ = g.check_execution_gate(0.10, [])
        assert ok is True
        ok, _ = g.check_execution_gate(0.95, [])
        assert ok is True


# ═════════════════════════════════════════════════════════════════════════════
# GATE 5: Cold Start
# ═════════════════════════════════════════════════════════════════════════════

class TestColdStart:

    def test_initial_state_not_warm(self):
        g = DataQualityGate(features_list=MINIMAL_FEATURES, warmup_slots=3)
        assert g.is_warm() is False

    def test_warm_after_n_slots(self):
        g = DataQualityGate(features_list=MINIMAL_FEATURES, warmup_slots=3)
        g.start_warmup(n_slots=3)
        for _ in range(3):
            g.record_slot_observed()
        assert g.is_warm() is True

    def test_not_warm_before_n_slots(self):
        g = DataQualityGate(features_list=MINIMAL_FEATURES, warmup_slots=3)
        g.start_warmup(n_slots=3)
        g.record_slot_observed()
        g.record_slot_observed()
        assert g.is_warm() is False

    def test_force_warm(self):
        g = DataQualityGate(features_list=MINIMAL_FEATURES, warmup_slots=10)
        g.start_warmup(n_slots=10)
        assert g.is_warm() is False
        g.force_warm()
        assert g.is_warm() is True

    def test_start_warmup_resets(self):
        g = DataQualityGate(features_list=MINIMAL_FEATURES)
        g.force_warm()
        assert g.is_warm() is True
        g.start_warmup(n_slots=5)
        assert g.is_warm() is False


# ═════════════════════════════════════════════════════════════════════════════
# ADMIN
# ═════════════════════════════════════════════════════════════════════════════

class TestAdmin:

    def test_unpause_clears_state(self):
        g = _make_gate()
        g._paused = True
        g._pause_reason = "test reason"
        g.unpause()
        assert g._paused is False
        assert g._pause_reason == ""

    def test_unpause_when_not_paused_is_noop(self):
        g = _make_gate()
        g.unpause()
        assert g._paused is False

    def test_status_returns_dict(self):
        g = _make_gate()
        s = g.status()
        assert isinstance(s, dict)
        assert "warm" in s
        assert "slots_seen" in s
        assert "warmup_target" in s
        assert "paused" in s
        assert "pause_reason" in s
        assert "predictions_tracked" in s
        assert "uptime_s" in s
        assert s["warm"] is True  # force_warm was called
        assert s["paused"] is False


# ═════════════════════════════════════════════════════════════════════════════
# COMPOSITE: pre_prediction_check + post_prediction_check
# ═════════════════════════════════════════════════════════════════════════════

class TestComposite:

    def test_pre_prediction_all_good(self, tmp_path):
        g = _make_gate()
        buf = tmp_path / "spot_buffer.json"
        _write_spot_buffer(buf)
        ticks = _make_ticks(100)
        fd = _good_feat_dict()
        ok, reason = g.pre_prediction_check(ticks, str(buf), int(time.time()), fd)
        assert ok is True
        assert reason == ""

    def test_pre_prediction_cold_start_blocks(self, tmp_path):
        g = DataQualityGate(features_list=MINIMAL_FEATURES, warmup_slots=5)
        buf = tmp_path / "spot_buffer.json"
        _write_spot_buffer(buf)
        ticks = _make_ticks(100)
        fd = _good_feat_dict()
        ok, reason = g.pre_prediction_check(ticks, str(buf), int(time.time()), fd)
        assert ok is False
        assert "COLD_START" in reason

    def test_pre_prediction_data_completeness_blocks(self, tmp_path):
        g = _make_gate()
        missing = tmp_path / "nonexistent.json"
        ticks = _make_ticks(5)  # too few
        fd = _good_feat_dict()
        ok, reason = g.pre_prediction_check(ticks, str(missing), int(time.time()), fd)
        assert ok is False
        assert "DATA_COMPLETENESS" in reason

    def test_pre_prediction_feature_sanity_blocks(self, tmp_path):
        g = _make_gate()
        buf = tmp_path / "spot_buffer.json"
        _write_spot_buffer(buf)
        ticks = _make_ticks(100)
        fd = _good_feat_dict()
        fd["btc_n_ticks"] = float("nan")
        ok, reason = g.pre_prediction_check(ticks, str(buf), int(time.time()), fd)
        assert ok is False
        assert "FEATURE_SANITY" in reason

    def test_post_prediction_all_good(self):
        g = _make_gate()
        ok, reason = g.post_prediction_check(0.55, _good_feat_dict(), 0.55, [])
        assert ok is True
        assert reason == ""

    def test_post_prediction_extreme_prob_blocks(self):
        g = _make_gate()
        ok, reason = g.post_prediction_check(0.999, _good_feat_dict(), 0.55, [])
        assert ok is False
        assert "PREDICTION_SANITY" in reason

    def test_post_prediction_bad_ask_blocks(self):
        g = _make_gate()
        ok, reason = g.post_prediction_check(0.55, _good_feat_dict(), 0.0, [])
        assert ok is False
        assert "EXECUTION_GATE" in reason
