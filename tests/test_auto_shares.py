"""Tests for compute_shares auto-sizing logic."""
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# ── Patch env vars and heavy imports BEFORE importing live_trader ──────────────
_ENV_PATCH = {
    "POLY_PRIVATE_KEY": "0x" + "ab" * 32,
    "POLY_SAFE_ADDRESS": "0x" + "cd" * 20,
    "MM_BUILDER_KEY": "fake-key",
    "MM_BUILDER_SECRET": "fake-secret",
    "MM_BUILDER_PASSPHRASE": "fake-pass",
}
for k, v in _ENV_PATCH.items():
    os.environ.setdefault(k, v)

DEPLOY_DIR = str(Path(__file__).resolve().parent.parent / "deploy")
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

sys.modules.setdefault("py_clob_client.client", mock.MagicMock())
sys.modules.setdefault("py_clob_client.clob_types", mock.MagicMock())

import live_trader as lt  # noqa: E402


# ── Fixed mode (AUTO_SHARES=False) ────────────────────────────────────────────

class TestFixedShares:
    def test_fixed_returns_fixed_shares(self):
        """When AUTO_SHARES=False, always returns FIXED_SHARES."""
        with mock.patch.object(lt, "AUTO_SHARES", False), \
             mock.patch.object(lt, "FIXED_SHARES", 8.0), \
             mock.patch.object(lt, "AUTO_SHARES_MIN", 5.0):
            assert lt.compute_shares(100.0, 0.50) == 8.0

    def test_fixed_respects_min(self):
        """FIXED_SHARES below MIN gets bumped to MIN."""
        with mock.patch.object(lt, "AUTO_SHARES", False), \
             mock.patch.object(lt, "FIXED_SHARES", 3.0), \
             mock.patch.object(lt, "AUTO_SHARES_MIN", 5.0):
            assert lt.compute_shares(100.0, 0.50) == 5.0

    def test_fixed_ignores_balance(self):
        """Balance doesn't affect fixed mode."""
        with mock.patch.object(lt, "AUTO_SHARES", False), \
             mock.patch.object(lt, "FIXED_SHARES", 10.0), \
             mock.patch.object(lt, "AUTO_SHARES_MIN", 5.0):
            assert lt.compute_shares(1000.0, 0.50) == 10.0
            assert lt.compute_shares(10.0, 0.50) == 10.0

    def test_fixed_ignores_ask_price(self):
        """Ask price doesn't affect fixed mode."""
        with mock.patch.object(lt, "AUTO_SHARES", False), \
             mock.patch.object(lt, "FIXED_SHARES", 8.0), \
             mock.patch.object(lt, "AUTO_SHARES_MIN", 5.0):
            assert lt.compute_shares(100.0, 0.10) == 8.0
            assert lt.compute_shares(100.0, 0.90) == 8.0


# ── Auto mode (AUTO_SHARES=True) ──────────────────────────────────────────────

def _auto_ctx():
    """Context manager with default auto-sizing config."""
    return mock.patch.multiple(
        lt,
        AUTO_SHARES=True,
        AUTO_SHARES_MIN=5.0,
        AUTO_SHARES_MAX=40.0,
        AUTO_SHARES_BAL_FLOOR=20.0,
        AUTO_SHARES_BAL_CEIL=700.0,
    )


class TestAutoShares:
    def test_at_floor_returns_min(self):
        with _auto_ctx():
            assert lt.compute_shares(20.0, 0.50) == 5.0

    def test_below_floor_returns_min(self):
        with _auto_ctx():
            assert lt.compute_shares(10.0, 0.50) == 5.0

    def test_at_ceil_returns_max(self):
        with _auto_ctx():
            assert lt.compute_shares(700.0, 0.50) == 40.0

    def test_above_ceil_returns_max(self):
        with _auto_ctx():
            assert lt.compute_shares(1000.0, 0.50) == 40.0

    def test_midpoint_linear(self):
        """Balance at midpoint -> roughly midpoint shares."""
        # midpoint balance = (20 + 700) / 2 = 360
        # expected = 5 + 0.5 * (40 - 5) = 22.5 -> int = 22
        with _auto_ctx():
            assert lt.compute_shares(360.0, 0.50) == 22.0

    def test_quarter_point(self):
        # 25% = 20 + 0.25 * 680 = 190
        # expected = 5 + 0.25 * 35 = 13.75 -> int = 13
        with _auto_ctx():
            assert lt.compute_shares(190.0, 0.50) == 13.0

    def test_three_quarter_point(self):
        # 75% = 20 + 0.75 * 680 = 530
        # expected = 5 + 0.75 * 35 = 31.25 -> int = 31
        with _auto_ctx():
            assert lt.compute_shares(530.0, 0.50) == 31.0

    def test_returns_integer(self):
        with _auto_ctx():
            result = lt.compute_shares(200.0, 0.50)
            assert result == int(result)

    def test_returns_float_type(self):
        with _auto_ctx():
            result = lt.compute_shares(200.0, 0.50)
            assert isinstance(result, float)

    def test_monotonically_increasing(self):
        with _auto_ctx():
            prev = 0
            for bal in [20, 50, 100, 200, 300, 400, 500]:
                shares = lt.compute_shares(float(bal), 0.50)
                assert shares >= prev, f"shares={shares} < prev={prev} at bal={bal}"
                prev = shares


# ── Risk cap (10% of balance) ─────────────────────────────────────────────────

class TestRiskCap:
    def test_risk_cap_limits_shares(self):
        """With high ask price and low balance, risk cap kicks in."""
        with _auto_ctx():
            result = lt.compute_shares(50.0, 0.90)
            cost = result * 0.90
            assert cost <= 50.0 * 0.10 + 0.01

    def test_risk_cap_with_expensive_ask(self):
        with _auto_ctx():
            result = lt.compute_shares(100.0, 0.90)
            cost = result * 0.90
            assert cost <= 100.0 * 0.10 + 0.01

    def test_no_risk_cap_with_cheap_ask(self):
        """With cheap asks, risk cap doesn't interfere."""
        with _auto_ctx():
            # balance=200, ask=0.10 -> 10% = $20 -> max 200 shares (above MAX)
            # linear: 5 + ((200-20)/680) * 35 = 5 + 9.26 = 14.26 -> 14
            result = lt.compute_shares(200.0, 0.10)
            assert result == 14.0

    def test_risk_cap_never_exceeds_10pct(self):
        """For any balance/ask combo above floor, cost doesn't exceed 10% of balance.
        Exception: at MIN shares (5), cost can exceed 10% because CLOB requires min 5."""
        with _auto_ctx():
            for bal in [50, 100, 200, 500]:
                for ask in [0.10, 0.30, 0.50, 0.70, 0.90]:
                    shares = lt.compute_shares(float(bal), ask)
                    cost = shares * ask
                    if shares > 5.0:  # above minimum, risk cap must hold
                        assert cost <= bal * 0.10 + 0.01, \
                            f"cost={cost:.2f} > 10% of bal={bal} at ask={ask}"


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_balance(self):
        with _auto_ctx():
            assert lt.compute_shares(0.0, 0.50) == 5.0

    def test_negative_balance(self):
        with _auto_ctx():
            assert lt.compute_shares(-10.0, 0.50) == 5.0

    def test_very_high_balance(self):
        with _auto_ctx():
            assert lt.compute_shares(10000.0, 0.50) == 40.0

    def test_very_small_ask(self):
        with _auto_ctx():
            result = lt.compute_shares(100.0, 0.001)
            assert 5.0 <= result <= 50.0

    def test_zero_ask(self):
        with _auto_ctx():
            result = lt.compute_shares(100.0, 0.0)
            assert result >= 5.0

    def test_ask_price_1(self):
        """Ask = $1.00 (max in binary market) -> risk cap applies."""
        with _auto_ctx():
            result = lt.compute_shares(200.0, 1.0)
            assert result <= 20.0  # 10% of 200 = $20 max cost
            assert result >= 5.0


# ── Custom config ─────────────────────────────────────────────────────────────

class TestCustomConfig:
    def test_custom_fixed_shares(self):
        with mock.patch.object(lt, "AUTO_SHARES", False), \
             mock.patch.object(lt, "FIXED_SHARES", 15.0), \
             mock.patch.object(lt, "AUTO_SHARES_MIN", 5.0):
            assert lt.compute_shares(100.0, 0.50) == 15.0

    def test_custom_min_max(self):
        with mock.patch.multiple(lt, AUTO_SHARES=True, AUTO_SHARES_MIN=10.0,
                                  AUTO_SHARES_MAX=30.0, AUTO_SHARES_BAL_FLOOR=50.0,
                                  AUTO_SHARES_BAL_CEIL=300.0):
            assert lt.compute_shares(50.0, 0.50) == 10.0
            assert lt.compute_shares(300.0, 0.50) == 30.0

    def test_custom_floor_ceil(self):
        with mock.patch.multiple(lt, AUTO_SHARES=True, AUTO_SHARES_MIN=5.0,
                                  AUTO_SHARES_MAX=50.0, AUTO_SHARES_BAL_FLOOR=100.0,
                                  AUTO_SHARES_BAL_CEIL=1000.0):
            assert lt.compute_shares(50.0, 0.50) == 5.0
            assert lt.compute_shares(1000.0, 0.50) == 50.0

    def test_tight_range(self):
        """Floor == Ceil -> bal <= floor triggers MIN."""
        with mock.patch.multiple(lt, AUTO_SHARES=True, AUTO_SHARES_MIN=5.0,
                                  AUTO_SHARES_MAX=50.0, AUTO_SHARES_BAL_FLOOR=100.0,
                                  AUTO_SHARES_BAL_CEIL=100.0):
            # bal <= floor -> MIN
            assert lt.compute_shares(100.0, 0.10) == 5.0
            # bal > ceil -> MAX
            assert lt.compute_shares(101.0, 0.10) == 50.0
