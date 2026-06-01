"""Unit tests for btc_lab.features module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_lab.features import build_price_features, encode_cyclical


# ---------------------------------------------------------------------------
# encode_cyclical tests
# ---------------------------------------------------------------------------


def test_encode_cyclical_zero():
    """At position 0, sin should be 0 and cos should be 1."""
    s = pd.Series([0.0])
    sin, cos = encode_cyclical(s, period=24.0)
    assert abs(float(sin.iloc[0])) < 1e-10
    assert abs(float(cos.iloc[0]) - 1.0) < 1e-10


def test_encode_cyclical_half_period():
    """At half period, sin should be 0 and cos should be -1."""
    s = pd.Series([12.0])
    sin, cos = encode_cyclical(s, period=24.0)
    assert abs(float(sin.iloc[0])) < 1e-10
    assert abs(float(cos.iloc[0]) + 1.0) < 1e-10


def test_encode_cyclical_quarter_period():
    """At quarter period, sin should be 1 and cos should be 0."""
    s = pd.Series([6.0])
    sin, cos = encode_cyclical(s, period=24.0)
    assert abs(float(sin.iloc[0]) - 1.0) < 1e-10
    assert abs(float(cos.iloc[0])) < 1e-10


def test_encode_cyclical_output_range():
    """Output values must be within [-1, 1]."""
    s = pd.Series(np.linspace(0, 24, 100))
    sin, cos = encode_cyclical(s, period=24.0)
    assert sin.between(-1.0, 1.0).all()
    assert cos.between(-1.0, 1.0).all()


# ---------------------------------------------------------------------------
# build_price_features tests
# ---------------------------------------------------------------------------


def _make_mock_data(
    n_ticks: int = 10,
    resolution: int = 1,
    market_id: str = "mkt-001",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create minimal mock prices_df and market_meta for testing."""
    timestamps = pd.date_range("2024-01-15 10:00", periods=n_ticks, freq="30s", tz="UTC")
    prices = pd.DataFrame(
        {
            "market_id": market_id,
            "timestamp": timestamps,
            "up_price": np.linspace(0.4, 0.6, n_ticks),
        }
    )
    meta = pd.DataFrame(
        {
            "market_id": [market_id],
            "resolution": [resolution],
        }
    )
    return prices, meta


def test_build_price_features_shape():
    """Should return exactly one row per resolved market."""
    prices, meta = _make_mock_data(n_ticks=10, resolution=1)
    result = build_price_features(prices, meta)
    assert len(result) == 1


def test_build_price_features_target_up():
    """resolution=1 should map to target=1."""
    prices, meta = _make_mock_data(resolution=1)
    result = build_price_features(prices, meta)
    assert result.iloc[0]["target"] == 1


def test_build_price_features_target_down():
    """resolution=0 should map to target=0."""
    prices, meta = _make_mock_data(resolution=0)
    result = build_price_features(prices, meta)
    assert result.iloc[0]["target"] == 0


def test_build_price_features_momentum():
    """price_momentum should equal last_price - first_price."""
    prices, meta = _make_mock_data(n_ticks=10)
    result = build_price_features(prices, meta)
    row = result.iloc[0]
    assert abs(row["price_momentum"] - (row["last_price"] - row["first_price"])) < 1e-10


def test_build_price_features_columns():
    """Result should contain all expected feature columns."""
    prices, meta = _make_mock_data(n_ticks=10)
    result = build_price_features(prices, meta)
    expected_cols = {
        "market_id",
        "start_ts",
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
        "target",
    }
    assert expected_cols.issubset(set(result.columns)), (
        f"Missing columns: {expected_cols - set(result.columns)}"
    )


def test_build_price_features_multiple_markets():
    """Should handle multiple markets and return one row each."""
    prices1, meta1 = _make_mock_data(n_ticks=8, resolution=1, market_id="mkt-A")
    prices2, meta2 = _make_mock_data(n_ticks=5, resolution=0, market_id="mkt-B")
    combined_prices = pd.concat([prices1, prices2], ignore_index=True)
    combined_meta = pd.concat([meta1, meta2], ignore_index=True)
    result = build_price_features(combined_prices, combined_meta)
    assert len(result) == 2
    targets = set(result["target"].tolist())
    assert targets == {0, 1}
