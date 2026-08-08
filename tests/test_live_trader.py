"""
Comprehensive unit tests for build_features() and build_spot_features()
in deploy/live_trader.py.

These test the feature computation logic in isolation by mocking:
  - _build_ob_features (network-dependent OB fetcher)
  - build_spot_features (when testing build_features; tested directly in its own suite)
  - _slot_history (module-level ring buffer)
  - SPOT_BUFFER (file path for spot data)
  - time.time (for staleness checks)
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# ── Patch env vars and heavy imports BEFORE importing live_trader ──────────────
# live_trader reads env vars at import time; provide dummies.
_ENV_PATCH = {
    "POLY_PRIVATE_KEY": "0x" + "ab" * 32,
    "POLY_SAFE_ADDRESS": "0x" + "cd" * 20,
    "MM_BUILDER_KEY": "fake-key",
    "MM_BUILDER_SECRET": "fake-secret",
    "MM_BUILDER_PASSPHRASE": "fake-pass",
}
for k, v in _ENV_PATCH.items():
    os.environ.setdefault(k, v)

# Add deploy/ to sys.path so live_trader can find its sibling modules
DEPLOY_DIR = str(Path(__file__).resolve().parent.parent / "deploy")
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

# Mock heavy/network modules that live_trader imports at module level
sys.modules.setdefault("py_clob_client.client", mock.MagicMock())
sys.modules.setdefault("py_clob_client.clob_types", mock.MagicMock())

import live_trader  # noqa: E402

# ── Champion feature list (v29_20f_rt) ────────────────────────────────────────
# This is the exact list the deployed model was trained on. Keep it in sync with
# champion.pkl["features"]; build_features() must be able to produce every one of
# them from data available live.
V29_FEATURES = [
    "btc_inslot_ret", "ob_depth_ratio", "ob_imbalance", "btc_pre_5m_ret",
    "clob_spread_mean", "clob_spread_trend", "btc_inslot_range", "ob_total_depth",
    "x_imb_x_ur", "btc_up_w1", "x_depth_x_vol", "clob_mid_volatility",
    "lag_ur_zscore_20", "prev_slot_up_ratio_3", "prev_slot_up_ratio_5",
    "btc_size_disparity", "btc_dist_1k", "clob_ask_pressure",
    "btc_up_ratio_zscore_5s", "btc_spot_vol_ratio",
]



# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_tick(t_sec, size_usdc, outcome, price, side="BUY"):
    return {
        "t_sec": t_sec,
        "size_usdc": size_usdc,
        "outcome": outcome,
        "price": price,
        "side": side,
    }


def _make_ticks_balanced(n=60):
    """Generate balanced Up/Down ticks spread across 180s observation window."""
    ticks = []
    for i in range(n):
        t_sec = (i / n) * 180.0
        outcome = "Up" if i % 2 == 0 else "Down"
        ticks.append(_make_tick(t_sec, 10.0, outcome, 0.55 if outcome == "Up" else 0.45, "BUY"))
    return ticks


def _make_ticks_all_up(n=30):
    ticks = []
    for i in range(n):
        t_sec = (i / n) * 180.0
        ticks.append(_make_tick(t_sec, 10.0, "Up", 0.60, "BUY"))
    return ticks


def _make_ticks_all_down(n=30):
    ticks = []
    for i in range(n):
        t_sec = (i / n) * 180.0
        ticks.append(_make_tick(t_sec, 10.0, "Down", 0.40, "SELL"))
    return ticks


def _make_ticks_6_windows():
    """One tick per 30s sub-window with varying up-ratios."""
    ticks = []
    for w in range(6):
        t_sec = w * 30 + 15  # middle of each window
        outcome = "Up" if w < 3 else "Down"
        ticks.append(_make_tick(t_sec, 10.0, outcome, 0.55, "BUY"))
    return ticks


def _spot_buffer_data(slot_ts, px=100000.0, n_candles=300, updated_at=None):
    """Generate a valid spot buffer dict."""
    if updated_at is None:
        updated_at = slot_ts + 180  # fresh buffer
    candles = []
    for i in range(n_candles):
        ts = slot_ts - (n_candles - i) * 60
        # slight uptrend
        p = px + (i - n_candles // 2) * 0.5
        candles.append([ts, p])
    return {"updated_at": updated_at, "btcusdt": candles}


def _ob_neutral():
    """Neutral OB features dict."""
    return {
        "ob_mid": 0.50,
        "ob_mid_drift": 0.0,
        "ob_imbalance_end": 0.0,
        "ob_spread_end": 0.02,
        "ob_depth_change": 0.0,
        "ob_imb_momentum": 0.0,
        "ob_imb_w0": 0.0,
        "ob_imb_w1": 0.0,
        "ob_imb_w2": 0.0,
        "ob_weighted_imb": 0.0,
        "ob_ask_depth_5c": 0.5,
        "ob_bid_depth_5c": 0.5,
        "ob_depth_ratio": 1.0,
        "ob_imbalance": 0.0,
    }


def _mock_slot_history(n=5, base_ts=1700000000):
    """Build a list of slot history entries."""
    entries = []
    for i in range(n):
        entries.append({
            "slot_ts": base_ts + i * 300,
            "up_ratio": 0.5 + 0.02 * i,
            "sw": [0.5] * 6,
            "pre_ret": 0.001 * i,
            "target": 1 if i % 2 == 0 else 0,
            "n_ticks": 30.0,
            "vol_total": 300.0,
        })
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
# ██ build_spot_features Tests ██
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildSpotFeatures:
    """Tests for build_spot_features(slot_ts)."""

    def test_missing_file_returns_zeros(self, tmp_path):
        """When SPOT_BUFFER file doesn't exist, return zero dict."""
        fake_path = tmp_path / "nonexistent.json"
        with mock.patch.object(live_trader, "SPOT_BUFFER", fake_path):
            result = live_trader.build_spot_features(1700000000)

        assert isinstance(result, dict)
        assert result["btc_inslot_ret"] == 0.0
        assert result["btc_pre_5m_ret"] == 0.0
        assert result["btc_pre_4h_ret"] == 0.0
        # Round-number proximity defaults to 0.5.
        # v29 keeps only btc_dist_1k; _5k/_10k were dropped from the feature set.
        assert result["btc_dist_1k"] == 0.5

    def test_corrupt_json_returns_zeros(self, tmp_path):
        """Corrupt JSON should return zeros gracefully."""
        buf_path = tmp_path / "spot.json"
        buf_path.write_text("{corrupt json!@#$")
        with mock.patch.object(live_trader, "SPOT_BUFFER", buf_path):
            result = live_trader.build_spot_features(1700000000)

        assert result["btc_inslot_ret"] == 0.0
        assert result["btc_dist_1k"] == 0.5

    def test_stale_buffer_returns_zeros(self, tmp_path):
        """If buffer updated_at is too old, return zeros."""
        slot_ts = 1700000000
        buf_path = tmp_path / "spot.json"
        data = _spot_buffer_data(slot_ts)
        data["updated_at"] = slot_ts - 9999  # very old
        buf_path.write_text(json.dumps(data))

        with mock.patch.object(live_trader, "SPOT_BUFFER", buf_path), \
             mock.patch("time.time", return_value=slot_ts + 200):
            result = live_trader.build_spot_features(slot_ts)

        assert result["btc_inslot_ret"] == 0.0

    def test_empty_candles_returns_zeros(self, tmp_path):
        """Empty candles list returns zeros."""
        slot_ts = 1700000000
        buf_path = tmp_path / "spot.json"
        data = {"updated_at": int(slot_ts + 180), "btcusdt": []}
        buf_path.write_text(json.dumps(data))

        with mock.patch.object(live_trader, "SPOT_BUFFER", buf_path), \
             mock.patch("time.time", return_value=slot_ts + 180):
            result = live_trader.build_spot_features(slot_ts)

        assert result["btc_inslot_ret"] == 0.0

    def test_normal_computation(self, tmp_path):
        """Normal case: candles available, features computed."""
        slot_ts = 1700000000
        buf_path = tmp_path / "spot.json"
        # Build candles covering inslot and pre-windows
        candles = []
        px_base = 100000.0
        for i in range(300):
            ts = slot_ts - (300 - i) * 60 + 60  # covers slot_ts-5h .. slot_ts+5m
            px = px_base + i * 1.0  # slight uptrend: 99701 .. 100000
            candles.append([ts, px])
        data = {"updated_at": int(slot_ts + 180), "btcusdt": candles}
        buf_path.write_text(json.dumps(data))

        with mock.patch.object(live_trader, "SPOT_BUFFER", buf_path), \
             mock.patch("time.time", return_value=slot_ts + 180):
            result = live_trader.build_spot_features(slot_ts)

        # v29 contract: returns/range + dist_1k. The per-window *_vol features and
        # dist_5k/_10k were dropped from the feature set, so they are no longer
        # computed on the happy path.
        for key in ["btc_inslot_ret", "btc_inslot_vol", "btc_inslot_range",
                     "btc_pre_5m_ret", "btc_pre_15m_ret",
                     "btc_pre_30m_ret", "btc_pre_1h_ret",
                     "btc_dist_1k"]:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], float), f"{key} not float"

        # With uptrend, every computed pre-window return should be positive
        assert result["btc_pre_5m_ret"] > 0, "Expected positive 5m return with uptrend"
        assert result["btc_pre_1h_ret"] > 0, "Expected positive 1h return with uptrend"

    def test_round_number_proximity(self, tmp_path):
        """Test round-number proximity features at known price levels."""
        slot_ts = 1700000000
        buf_path = tmp_path / "spot.json"

        # Price exactly at $100,000 — dist_1k should be ~0 (at round number)
        candles = [[slot_ts + i, 100000.0] for i in range(-300, 200)]
        data = {"updated_at": int(slot_ts + 180), "btcusdt": candles}
        buf_path.write_text(json.dumps(data))

        with mock.patch.object(live_trader, "SPOT_BUFFER", buf_path), \
             mock.patch("time.time", return_value=slot_ts + 180):
            result = live_trader.build_spot_features(slot_ts)

        # At exactly 100000, dist to nearest 1k boundary = 0.
        # dist_5k/_10k are not part of the v29 feature set any more.
        assert result["btc_dist_1k"] == pytest.approx(0.0, abs=0.01)

    def test_round_number_proximity_midrange(self, tmp_path):
        """Price at $100,500 should be 0.5 from 1k boundaries."""
        slot_ts = 1700000000
        buf_path = tmp_path / "spot.json"

        candles = [[slot_ts + i, 100500.0] for i in range(-300, 200)]
        data = {"updated_at": int(slot_ts + 180), "btcusdt": candles}
        buf_path.write_text(json.dumps(data))

        with mock.patch.object(live_trader, "SPOT_BUFFER", buf_path), \
             mock.patch("time.time", return_value=slot_ts + 180):
            result = live_trader.build_spot_features(slot_ts)

        # At 100500, dist_1k = min(0.5, 0.5) = 0.5
        assert result["btc_dist_1k"] == pytest.approx(0.5, abs=0.01)

    def test_pre_window_returns_single_candle(self, tmp_path):
        """With only one candle per segment, returns should be 0."""
        slot_ts = 1700000000
        buf_path = tmp_path / "spot.json"

        # Only 2 candles — not enough for vol but enough for ret
        candles = [
            [slot_ts - 100, 100000.0],
            [slot_ts + 90, 100100.0],
        ]
        data = {"updated_at": int(slot_ts + 180), "btcusdt": candles}
        buf_path.write_text(json.dumps(data))

        with mock.patch.object(live_trader, "SPOT_BUFFER", buf_path), \
             mock.patch("time.time", return_value=slot_ts + 180):
            result = live_trader.build_spot_features(slot_ts)

        # Should still return a valid dict
        assert isinstance(result, dict)
        assert "btc_inslot_ret" in result

    def test_spot_vol_ratio_present(self, tmp_path):
        """btc_spot_vol_ratio is the v29 volatility-regime feature.

        (It replaced btc_pre_1h_4h_ratio, which was dropped along with its 4h
        warm-up dependency.)
        """
        slot_ts = 1700000000
        buf_path = tmp_path / "spot.json"
        candles = []
        for i in range(300):
            ts = slot_ts - (300 - i) * 60
            candles.append([ts, 100000.0 + i])
        data = {"updated_at": int(slot_ts + 180), "btcusdt": candles}
        buf_path.write_text(json.dumps(data))

        with mock.patch.object(live_trader, "SPOT_BUFFER", buf_path), \
             mock.patch("time.time", return_value=slot_ts + 180):
            result = live_trader.build_spot_features(slot_ts)

        assert "btc_pre_1h_4h_ratio" not in result, "feature was removed in v26"
        assert "btc_inslot_range" in result


# ═══════════════════════════════════════════════════════════════════════════════
# ██ build_features Tests ██
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildFeatures:
    """Tests for build_features(ticks, slot_ts, features, up_token_id)."""

    @pytest.fixture(autouse=True)
    def _patch_externals(self):
        """Mock OB features and spot features for all build_features tests."""
        with mock.patch.object(live_trader, "_build_ob_features", return_value=_ob_neutral()), \
             mock.patch.object(live_trader, "build_spot_features", return_value={
                 "btc_inslot_ret": 0.001, "btc_inslot_vol": 0.0002,
                 "btc_pre_5m_ret": 0.0005, "btc_pre_5m_vol": 0.0001,
                 "btc_pre_15m_ret": 0.001, "btc_pre_15m_vol": 0.0002,
                 "btc_pre_30m_ret": 0.002, "btc_pre_30m_vol": 0.0003,
                 "btc_pre_1h_ret": 0.003, "btc_pre_1h_vol": 0.0004,
                 "btc_pre_4h_ret": 0.01, "btc_pre_4h_vol": 0.001,
                 "btc_dist_1k": 0.3, "btc_dist_5k": 0.2, "btc_dist_10k": 0.1,
                 "btc_pre_1h_4h_ratio": 0.3,
             }):
            # Reset _slot_history for each test
            live_trader._slot_history = []
            yield

    def test_normal_ticks_returns_all_features(self):
        """Normal balanced ticks should produce all v21 features."""
        slot_ts = 1700000000
        ticks = _make_ticks_balanced(60)
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        assert result is not None
        for f in V29_FEATURES:
            assert f in result, f"Missing feature: {f}"
            assert isinstance(result[f], (int, float)), f"{f} is not numeric: {type(result[f])}"
            assert not math.isnan(result[f]), f"{f} is NaN"
            assert not math.isinf(result[f]), f"{f} is inf"

    def test_empty_ticks_returns_neutral_defaults(self):
        """No ticks should fill neutral defaults (0.5 up_ratio, 0.0 momentum, etc.)."""
        slot_ts = 1700000000
        result = live_trader.build_features([], slot_ts, V29_FEATURES, "tok123")

        assert result is not None
        assert result["btc_up_ratio"] == 0.5
        assert result["btc_momentum"] == 0.0
        assert result["btc_buy_ratio"] == 0.5
        assert result["btc_size_disparity"] == 0.0
        # v29 observes [0, OBSERVE_SECS=60) as 2x30s sub-windows (w0, w1).
        # w2..w5 existed only while OBSERVE_SECS was 180.
        for i in range(2):
            assert result[f"btc_up_w{i}"] == 0.5
        assert "btc_up_w2" not in result

    def test_all_up_ticks(self):
        """All-Up ticks inside the observation window: up_ratio ~1.0."""
        slot_ts = 1700000000
        ticks = _make_ticks_all_up(30)
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        assert result is not None
        assert result["btc_up_ratio"] == pytest.approx(1.0, abs=0.01)
        assert result["btc_buy_ratio"] == pytest.approx(1.0, abs=0.01)
        assert result["btc_size_disparity"] > 0, "all-Up flow should skew disparity positive"

    def test_all_down_ticks(self):
        """All-Down ticks: up_ratio ~0.0, vwap_up defaults to 0.5."""
        slot_ts = 1700000000
        ticks = _make_ticks_all_down(30)
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        assert result is not None
        assert result["btc_up_ratio"] == pytest.approx(0.0, abs=0.01)
        # buy_ratio: all SELL
        assert result["btc_buy_ratio"] == pytest.approx(0.0, abs=0.01)
        assert result["btc_size_disparity"] < 0, "all-Down flow should skew disparity negative"

    def test_two_window_coverage(self):
        """v29 observes [0,60) as two 30s sub-windows: w0 = [0,30), w1 = [30,60)."""
        slot_ts = 1700000000
        ticks = [
            _make_tick(15, 10.0, "Up", 0.55, "BUY"),    # w0
            _make_tick(45, 10.0, "Down", 0.45, "BUY"),  # w1
        ]
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        assert result is not None
        assert result["btc_up_w0"] == pytest.approx(1.0, abs=0.01)
        assert result["btc_up_w1"] == pytest.approx(0.0, abs=0.01)

    def test_momentum_is_w1_minus_w0(self):
        """v29 momentum = btc_up_w1 - btc_up_w0 (was a 3-vs-3 window mean at 180s)."""
        slot_ts = 1700000000
        ticks = [
            _make_tick(15, 10.0, "Up", 0.55, "BUY"),    # w0 -> up_ratio 1.0
            _make_tick(45, 10.0, "Down", 0.45, "BUY"),  # w1 -> up_ratio 0.0
        ]
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        assert result["btc_momentum"] == pytest.approx(-1.0, abs=0.01)

    def test_subwindows_only_cover_the_observation_window(self):
        """btc_up_w0/w1 span [0,30) and [30,60); later ticks fall in neither.

        NOTE on the contract: build_features() aggregates btc_up_ratio over the
        WHOLE ticks list it is handed — it does not re-filter by t_sec. The
        [0, OBSERVE_SECS) cut happens in the caller (live_trader.py:1001), which
        mirrors train_v29_modal.py's `btc[btc.t_sec < OBS_SECS]` pre-filter.
        Both sides therefore see the same window.
        """
        slot_ts = 1700000000
        ticks = [
            _make_tick(10, 10.0, "Up", 0.60, "BUY"),     # w0
            _make_tick(40, 10.0, "Up", 0.60, "BUY"),     # w1
            _make_tick(200, 10.0, "Down", 0.40, "SELL"),  # outside both windows
        ]
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        # The late tick lands in no sub-window, so both stay all-Up.
        assert result["btc_up_w0"] == pytest.approx(1.0, abs=0.01)
        assert result["btc_up_w1"] == pytest.approx(1.0, abs=0.01)

    def test_hour_x_up_ratio(self):
        """hour_x_up_ratio = up_ratio * (hour / 24.0)."""
        # Use slot_ts that corresponds to a known UTC hour
        # 1700000000 = 2023-11-14 22:13:20 UTC → hour ≈ 22.22
        from datetime import datetime as dt, timezone as tz
        slot_ts = 1700000000
        utc_dt = dt.fromtimestamp(slot_ts, tz=tz.utc)
        hour = utc_dt.hour + utc_dt.minute / 60.0

        ticks = _make_ticks_balanced(60)
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        expected_ur = 0.5  # balanced ticks
        expected = expected_ur * (hour / 24.0)
        assert result["hour_x_up_ratio"] == pytest.approx(expected, abs=0.02)

    def test_lag_features_with_history(self):
        """prev_slot_up_ratio_{1,2,3,5} should come from _slot_history."""
        live_trader._slot_history = _mock_slot_history(n=6, base_ts=1700000000)
        slot_ts = 1700000000 + 6 * 300  # next slot after history

        ticks = _make_ticks_balanced(20)
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        # History entries have up_ratio = 0.5, 0.52, 0.54, 0.56, 0.58, 0.60
        # lag 1 → last entry (index -1) = 0.60
        assert result["prev_slot_up_ratio_1"] == pytest.approx(0.60, abs=0.01)
        # lag 2 → index -2 = 0.58
        assert result["prev_slot_up_ratio_2"] == pytest.approx(0.58, abs=0.01)
        # lag 3 → index -3 = 0.56
        assert result["prev_slot_up_ratio_3"] == pytest.approx(0.56, abs=0.01)
        # lag 5 → index -5 = 0.52
        assert result["prev_slot_up_ratio_5"] == pytest.approx(0.52, abs=0.01)

    def test_lag_features_no_history(self):
        """Without slot history, lag features should default to 0.5."""
        live_trader._slot_history = []
        slot_ts = 1700000000
        ticks = _make_ticks_balanced(20)
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        assert result["prev_slot_up_ratio_1"] == 0.5
        assert result["prev_slot_up_ratio_2"] == 0.5
        assert result["prev_slot_up_ratio_3"] == 0.5
        assert result["prev_slot_up_ratio_5"] == 0.5

    def test_lag_features_partial_history(self):
        """With 2 slots of history, lag 3 and 5 should default."""
        live_trader._slot_history = _mock_slot_history(n=2, base_ts=1700000000)
        slot_ts = 1700000000 + 2 * 300
        ticks = _make_ticks_balanced(10)
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        # lag 1, 2 available
        assert result["prev_slot_up_ratio_1"] != 0.5 or result["prev_slot_up_ratio_2"] != 0.5
        # lag 3, 5 should be 0.5 (not enough history)
        assert result["prev_slot_up_ratio_3"] == 0.5
        assert result["prev_slot_up_ratio_5"] == 0.5

    def test_cross_domain_interactions(self):
        """The two cross features in the v29 set are products of their parts."""
        slot_ts = 1700000000
        ticks = _make_ticks_balanced(20)
        result = live_trader.build_features(
            ticks, slot_ts, V29_FEATURES + ["btc_vol_1h"], "tok123")

        assert result["x_imb_x_ur"] == pytest.approx(
            result["ob_imbalance"] * result["btc_up_ratio"], abs=1e-9)
        assert result["x_depth_x_vol"] == pytest.approx(
            result["ob_depth_ratio"] * result["btc_vol_1h"], abs=1e-9)

    def test_neutral_defaults_for_missing_features(self):
        """Features not computed should get context-aware defaults."""
        slot_ts = 1700000000
        ticks = _make_ticks_balanced(10)
        # Add a feature that isn't computed anywhere
        features_with_extra = V29_FEATURES + ["some_unknown_feature"]
        result = live_trader.build_features(ticks, slot_ts, features_with_extra, "tok123")

        assert result is not None
        assert result["some_unknown_feature"] == 0.0

    def test_ob_neutral_defaults_when_orderbook_fails(self):
        """A failed OB fetch must yield context-aware defaults, never 0.0 mids.

        Only features in the requested list get filled, so ask for them explicitly.
        """
        slot_ts = 1700000000
        ticks = _make_ticks_balanced(10)
        feats = V29_FEATURES + ["ob_mid", "ob_ask_depth_5c", "ob_bid_depth_5c", "ob_spread"]

        with mock.patch.object(live_trader, "_build_ob_features", return_value={}):
            result = live_trader.build_features(ticks, slot_ts, feats, "tok123")

        assert result["ob_mid"] == 0.5
        assert result["ob_ask_depth_5c"] == 0.5
        assert result["ob_bid_depth_5c"] == 0.5
        assert result["ob_spread"] == 0.02
        assert result["ob_depth_ratio"] == 1.0
        assert result["ob_total_depth"] == 1000.0

    def test_size_disparity(self):
        """btc_size_disparity = avg_up_size - avg_dn_size."""
        slot_ts = 1700000000
        ticks = [
            _make_tick(10, 200, "Up", 0.60, "BUY"),    # avg_up = 200
            _make_tick(20, 100, "Down", 0.40, "BUY"),   # avg_dn = 100
        ]
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        assert result["btc_size_disparity"] == pytest.approx(100.0, abs=1.0)

    def test_buy_ratio(self):
        """btc_buy_ratio = BUY volume / total volume."""
        slot_ts = 1700000000
        ticks = [
            _make_tick(10, 100, "Up", 0.60, "BUY"),
            _make_tick(20, 100, "Down", 0.40, "SELL"),
            _make_tick(30, 100, "Up", 0.55, "BUY"),
        ]
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        # BUY volume = 200, total = 300 → buy_ratio ≈ 0.667
        assert result["btc_buy_ratio"] == pytest.approx(200.0 / 300.0, abs=0.02)

    def test_all_features_numeric(self):
        """Every returned feature value should be a finite number."""
        slot_ts = 1700000000
        ticks = _make_ticks_balanced(60)
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        assert result is not None
        for k, v in result.items():
            if k in V29_FEATURES:
                assert isinstance(v, (int, float)), f"{k}: expected number, got {type(v)}"
                assert math.isfinite(v), f"{k}: value {v} is not finite"

    def test_spot_features_integrated(self):
        """Spot features from mocked build_spot_features should appear in result."""
        slot_ts = 1700000000
        ticks = _make_ticks_balanced(20)
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        # These come from the mocked build_spot_features
        assert result["btc_inslot_ret"] == pytest.approx(0.001, abs=0.0001)
        assert result["btc_pre_5m_ret"] == pytest.approx(0.0005, abs=0.0001)
        assert result["btc_pre_4h_ret"] == pytest.approx(0.01, abs=0.001)

    def test_ob_features_integrated(self):
        """OB features from mocked _build_ob_features should appear in result."""
        slot_ts = 1700000000
        ticks = _make_ticks_balanced(20)
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        assert result["ob_mid"] == pytest.approx(0.50, abs=0.01)
        assert result["ob_mid_drift"] == pytest.approx(0.0, abs=0.01)
        assert result["ob_imb_w0"] == pytest.approx(0.0, abs=0.01)

    def test_single_tick(self):
        """A single tick should not crash."""
        slot_ts = 1700000000
        ticks = [_make_tick(90, 50, "Up", 0.55, "BUY")]
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        assert result is not None
        assert result["btc_up_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_large_tick_volume(self):
        """Large volumes should not cause overflow or NaN."""
        slot_ts = 1700000000
        ticks = [
            _make_tick(10, 1e8, "Up", 0.99, "BUY"),
            _make_tick(20, 1e8, "Down", 0.01, "SELL"),
        ]
        result = live_trader.build_features(ticks, slot_ts, V29_FEATURES, "tok123")

        assert result is not None
        for f in V29_FEATURES:
            assert math.isfinite(result[f]), f"{f} is not finite with large volume"


# ═══════════════════════════════════════════════════════════════════════════════
# ██ _update_slot_history Tests ██
# ═══════════════════════════════════════════════════════════════════════════════

class TestUpdateSlotHistory:
    """Tests for _update_slot_history()."""

    def setup_method(self):
        live_trader._slot_history = []

    def test_append_new_entry(self):
        live_trader._update_slot_history(1700000000, 0.55, [0.5] * 6, 0.001, None)
        assert len(live_trader._slot_history) == 1
        assert live_trader._slot_history[0]["slot_ts"] == 1700000000
        assert live_trader._slot_history[0]["up_ratio"] == 0.55

    def test_update_existing_target(self):
        live_trader._update_slot_history(1700000000, 0.55, [0.5] * 6, 0.001, None)
        live_trader._update_slot_history(1700000000, 0.55, [0.5] * 6, 0.001, 1)
        assert len(live_trader._slot_history) == 1
        assert live_trader._slot_history[0]["target"] == 1

    def test_ring_buffer_max_size(self):
        for i in range(30):
            live_trader._update_slot_history(1700000000 + i * 300, 0.5, [0.5] * 6)
        assert len(live_trader._slot_history) <= live_trader._HIST_MAX

    def test_multiple_entries_order(self):
        for i in range(5):
            live_trader._update_slot_history(1700000000 + i * 300, 0.5 + 0.01 * i, [0.5] * 6)
        assert live_trader._slot_history[0]["slot_ts"] == 1700000000
        assert live_trader._slot_history[-1]["slot_ts"] == 1700000000 + 4 * 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
