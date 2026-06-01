"""Feature engineering functions for BTC 5-minute Polymarket markets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def encode_cyclical(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Encode a cyclical feature as sin/cos pair."""
    sin = np.sin(2 * np.pi * series / period)
    cos = np.cos(2 * np.pi * series / period)
    return sin, cos


def _price_at_quantile(group: pd.DataFrame, q: float) -> float:
    """Return the up_price at a fractional elapsed time within a market window."""
    if len(group) == 0:
        return float("nan")
    t_min = group["timestamp"].min()
    t_max = group["timestamp"].max()
    if t_min == t_max:
        return group["up_price"].iloc[0]
    t_target = t_min + q * (t_max - t_min)
    # Find closest row by timestamp
    idx = (group["timestamp"] - t_target).abs().idxmin()
    return group.loc[idx, "up_price"]


def build_price_features(
    prices_df: pd.DataFrame,
    market_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each resolved BTC 5-min market, build features from its price series.

    The market window is 5 minutes. We look at prices during the window and extract:
    - first_price: up_price at the first tick
    - last_price: up_price at the last tick before end
    - price_mean: average up_price across all ticks
    - price_std: std of up_price (volatility)
    - price_min, price_max: range
    - price_momentum: last_price - first_price
    - n_ticks: number of price updates
    - price_at_25pct, price_at_50pct, price_at_75pct: price at 1/4, 1/2, 3/4 of elapsed time
    - hour_of_day_sin/cos: sin/cos encoded hour (cyclical)
    - day_of_week_sin/cos: sin/cos encoded day (cyclical)

    Returns DataFrame with one row per market, including 'target' (resolution: 1=UP, 0=DOWN).
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(prices_df["timestamp"]):
        prices_df = prices_df.copy()
        prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"], unit="s", utc=True)

    # Build market_id index for fast lookup
    meta_indexed = market_meta.set_index("market_id")

    rows = []
    for market_id, group in prices_df.groupby("market_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)

        if market_id not in meta_indexed.index:
            continue

        meta = meta_indexed.loc[market_id]
        resolution = meta["resolution"] if "resolution" in meta.index else None

        first_price = group["up_price"].iloc[0]
        last_price = group["up_price"].iloc[-1]
        price_mean = group["up_price"].mean()
        price_std = group["up_price"].std(ddof=0)
        price_min = group["up_price"].min()
        price_max = group["up_price"].max()
        price_momentum = last_price - first_price
        n_ticks = len(group)

        price_at_25pct = _price_at_quantile(group, 0.25)
        price_at_50pct = _price_at_quantile(group, 0.50)
        price_at_75pct = _price_at_quantile(group, 0.75)

        # Temporal features from the first timestamp of the market
        t0 = group["timestamp"].iloc[0]
        hour = t0.hour + t0.minute / 60.0
        dow = t0.dayofweek

        hour_sin, hour_cos = encode_cyclical(pd.Series([hour]), 24.0)
        dow_sin, dow_cos = encode_cyclical(pd.Series([dow]), 7.0)

        # Derive target: resolution == 1 → UP (1), resolution == 0 → DOWN (0)
        if resolution is None:
            continue
        target = int(resolution)

        rows.append(
            {
                "market_id": market_id,
                "start_ts": t0,
                "first_price": first_price,
                "last_price": last_price,
                "price_mean": price_mean,
                "price_std": price_std,
                "price_min": price_min,
                "price_max": price_max,
                "price_momentum": price_momentum,
                "n_ticks": n_ticks,
                "price_at_25pct": price_at_25pct,
                "price_at_50pct": price_at_50pct,
                "price_at_75pct": price_at_75pct,
                "hour_of_day_sin": float(hour_sin.iloc[0]),
                "hour_of_day_cos": float(hour_cos.iloc[0]),
                "day_of_week_sin": float(dow_sin.iloc[0]),
                "day_of_week_cos": float(dow_cos.iloc[0]),
                "target": target,
            }
        )

    return pd.DataFrame(rows)


def build_tick_features(ticks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build order flow and microstructure features from tick data.
    One row per market_id.
    """
    rows = []
    for market_id, grp in ticks_df.groupby('market_id'):
        grp = grp.sort_values('timestamp_ms')
        buy = grp[grp['side'] == 'BUY']
        sell = grp[grp['side'] == 'SELL']
        up = grp[grp['outcome'] == 'Up']
        down = grp[grp['outcome'] == 'Down']

        buy_vol = buy['size_usdc'].sum()
        sell_vol = sell['size_usdc'].sum()
        total_vol = buy_vol + sell_vol
        up_vol = up['size_usdc'].sum()
        down_vol = down['size_usdc'].sum()

        spot = grp['spot_price_usdt'].dropna()
        spot_start = spot.iloc[0] if len(spot) > 0 else float('nan')
        spot_end = spot.iloc[-1] if len(spot) > 0 else float('nan')
        spot_return = (spot_end - spot_start) / spot_start if spot_start and spot_start != 0 else float('nan')

        vwap_up = (up['price'] * up['size_usdc']).sum() / up_vol if up_vol > 0 else float('nan')
        vwap_down = (down['price'] * down['size_usdc']).sum() / down_vol if down_vol > 0 else float('nan')

        rows.append({
            'market_id': str(market_id),
            'total_volume_usdc': total_vol,
            'buy_volume_usdc': buy_vol,
            'sell_volume_usdc': sell_vol,
            'buy_sell_imbalance': (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0,
            'up_volume_usdc': up_vol,
            'down_volume_usdc': down_vol,
            'up_down_volume_ratio': up_vol / (up_vol + down_vol) if (up_vol + down_vol) > 0 else 0.5,
            'n_trades': len(grp),
            'n_buy_trades': len(buy),
            'n_sell_trades': len(sell),
            'avg_trade_size': total_vol / len(grp) if len(grp) > 0 else 0.0,
            'spot_price_start': spot_start,
            'spot_price_end': spot_end,
            'spot_return': spot_return,
            'spot_volatility': spot.std() if len(spot) > 1 else 0.0,
            'spot_price_mean': spot.mean() if len(spot) > 0 else float('nan'),
            'vwap_up': vwap_up,
            'vwap_down': vwap_down,
        })
    return pd.DataFrame(rows)
