
import numpy as np
import pandas as pd

from .data import normalize_utc_timestamps


DEFAULT_GAP_TOLERANCE = pd.Timedelta("90s")


def indicator_segment_ids(
    timestamps: pd.Series,
    *,
    gap_tolerance: str | pd.Timedelta = DEFAULT_GAP_TOLERANCE,
) -> pd.Series:
    """Return causal segment IDs, starting a segment only when a gap exceeds tolerance.

    A gap exactly equal to the tolerance stays in the same segment. Input times
    must be valid, UTC-normalizable, strictly increasing, and unique.
    """
    tolerance = pd.Timedelta(gap_tolerance)
    if tolerance < pd.Timedelta(0):
        raise ValueError("gap_tolerance cannot be negative")
    times = normalize_utc_timestamps(timestamps)
    if times.isna().any():
        raise ValueError("timestamps must be valid to determine indicator segments")
    differences = times.diff()
    if (differences.iloc[1:] <= pd.Timedelta(0)).any():
        raise ValueError("timestamps must be strictly increasing and unique")
    starts = differences.gt(tolerance)
    if not starts.empty:
        starts.iloc[0] = False
    return starts.cumsum().astype("int64")


def segment_bar_counts(segment_ids: pd.Series) -> pd.Series:
    """Return one-based observed-bar counts within each indicator segment."""
    return segment_ids.groupby(segment_ids, sort=False).cumcount() + 1


def build_features(
    df: pd.DataFrame,
    *,
    gap_tolerance: str | pd.Timedelta = DEFAULT_GAP_TOLERANCE,
) -> pd.DataFrame:
    """Build causal features from candles with unambiguous timezone-aware times.

    Input timestamps must be timezone-aware; they are normalized to UTC. Naive
    timestamps are rejected here because this function has no source-specific
    timezone context. A timestamp gap greater than ``gap_tolerance`` starts a
    new indicator segment: EMA and RSI state reset, while returns and rolling
    windows cannot use rows from the preceding segment. Gaps equal to the
    tolerance remain within the segment. No session calendar is applied, so
    this reset policy also treats weekends and other closures as boundaries.
    """

    result = df.copy()
    result["time"] = normalize_utc_timestamps(result["time"])
    segments = indicator_segment_ids(
        result["time"], gap_tolerance=gap_tolerance
    )
    close = result["close"]

    # Price momentum over different lookback periods
    for period in [1, 5, 15, 60]:
        result[f"return_{period}"] = close.groupby(
            segments, sort=False
        ).transform(lambda values: values.pct_change(period))

    # Trend indicators
    for span in (9, 21, 50):
        result[f"ema_{span}"] = close.groupby(
            segments, sort=False
        ).transform(lambda values: values.ewm(span=span, adjust=False).mean())

    result["ema_gap"] = (
        result["ema_9"] - result["ema_21"]
    ) / close

    result["ema_long_gap"] = (
        result["ema_21"] - result["ema_50"]
    ) / close

    # RSI (14 periods)
    change = close.groupby(segments, sort=False).diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    average_gain = gains.groupby(segments, sort=False).transform(
        lambda values: values.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    )

    average_loss = losses.groupby(segments, sort=False).transform(
        lambda values: values.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    )

    relative_strength = average_gain / average_loss

    result["rsi_14"] = 100 - (
        100 / (1 + relative_strength)
    )

    # Recent volatility
    result["volatility_15"] = result["return_1"].groupby(
        segments, sort=False
    ).transform(lambda values: values.rolling(15).std())

    result["volatility_60"] = result["return_1"].groupby(
        segments, sort=False
    ).transform(lambda values: values.rolling(60).std())

    # Candle structure
    result["candle_range"] = (
        result["high"] - result["low"]
    ) / close

    result["candle_body"] = (
        result["close"] - result["open"]
    ) / close

    # Time-of-day features
    timestamp = pd.to_datetime(result["time"])
    minute_of_day = timestamp.dt.hour * 60 + timestamp.dt.minute

    result["time_sin"] = np.sin(
        2 * np.pi * minute_of_day / 1440
    )

    result["time_cos"] = np.cos(
        2 * np.pi * minute_of_day / 1440
    )

    return result.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    from data import load_market_data

    market_data = load_market_data()
    featured_data = build_features(market_data)

    feature_columns = [
        column for column in featured_data.columns
        if column not in ["time", "open", "high", "low", "close", "volume"]
    ]

    print("Feature engineering completed")
    print(f"Market rows: {len(featured_data):,}")
    print(f"Features created: {len(feature_columns)}")
    print("Feature names:", feature_columns)
    print("\nLast five rows:")
    print(featured_data[["time"] + feature_columns].tail())
    print("\nMissing values per feature:")
    print(featured_data[feature_columns].isna().sum())
