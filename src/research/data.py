
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "eurusd_clean.csv"

REQUIRED_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
]
OHLC_COLUMNS = ["open", "high", "low", "close"]


def validate_m1_cadence(timestamps: pd.Series) -> dict[str, int]:
    """Validate UTC M1 bar timestamps while allowing minute-aligned market gaps.

    Consecutive one-minute bars are valid. A gap is valid only when it is an
    integer multiple of one minute and strictly greater than the engine's
    90-second segment tolerance. Intermediate/ambiguous intervals (including
    exactly 90 seconds) and non-minute-aligned stamps are rejected.
    """
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("M1 cadence requires timezone-aware timestamps")
    times = timestamps.dt.tz_convert("UTC").dt.as_unit("ns")
    if times.isna().any():
        raise ValueError("M1 cadence requires valid timestamps")
    minute_ns = pd.Timedelta("1min").value
    if ((times.dt.second != 0) | (times.dt.microsecond != 0) | (times.dt.nanosecond != 0)).any():
        raise ValueError("M1 timestamps must align to whole UTC minutes")
    delta_ns = times.diff().dropna().astype("int64")
    if (delta_ns <= 0).any():
        raise ValueError("M1 timestamps must be strictly increasing and unique")
    ordinary = delta_ns.eq(minute_ns)
    gaps = delta_ns.gt(pd.Timedelta("90s").value)
    aligned_gaps = delta_ns.mod(minute_ns).eq(0)
    if not (ordinary | (gaps & aligned_gaps)).all():
        raise ValueError(
            "timestamps are not M1 cadence: intervals must be 1 minute or "
            "minute-aligned gaps greater than 90 seconds"
        )
    if len(delta_ns) and not ordinary.any():
        raise ValueError(
            "cannot establish M1 cadence: no adjacent one-minute bars distinguish "
            "market gaps from a regularly sampled non-M1 series"
        )
    return {"rows": int(len(times)), "market_gaps": int(gaps.sum())}


def normalize_utc_timestamps(
    values: pd.Series,
    *,
    assume_naive_timezone: str | None = None,
) -> pd.Series:
    """Normalize homogeneous timestamps to UTC without guessing mixed input.

    Naive values are localized only when the caller supplies their known
    timezone. Aware values are converted to UTC by instant. A mixture of naive
    and aware values is rejected because it has no single safe interpretation.
    Unparseable values become ``NaT`` so callers can count/drop those rows.
    """
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    parsed: list[object] = []
    awareness: set[bool] = set()

    for value in series:
        if value is None or value is pd.NaT:
            parsed.append(pd.NaT)
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            if pd.isna(value):
                parsed.append(pd.NaT)
                continue
            raise ValueError("Numeric timestamps are ambiguous; provide datetime values or strings")
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            parsed.append(pd.NaT)
            continue
        if pd.isna(timestamp):
            parsed.append(pd.NaT)
            continue
        is_aware = timestamp.tzinfo is not None
        awareness.add(is_aware)
        parsed.append(timestamp)

    if len(awareness) > 1:
        raise ValueError("Mixed timezone-aware and naive timestamps are ambiguous")

    if awareness == {True}:
        normalized = pd.to_datetime(parsed, errors="coerce", utc=True)
    elif awareness == {False}:
        if assume_naive_timezone is None:
            raise ValueError("Naive timestamps require an explicitly known timezone")
        naive = pd.DatetimeIndex(parsed)
        localized = naive.tz_localize(
            assume_naive_timezone,
            ambiguous="raise",
            nonexistent="raise",
        )
        normalized = (
            localized
            if assume_naive_timezone == "UTC"
            else localized.tz_convert("UTC")
        )
    else:
        # An empty or entirely invalid input still gets a stable UTC dtype.
        normalized = pd.to_datetime(parsed, errors="coerce", utc=True)

    return pd.Series(normalized, index=series.index, name=series.name)


def validate_ohlc_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric OHLC columns or reject any malformed price row.

    Every OHLC value must be finite and strictly positive, and each candle must
    satisfy its high/low bounds. Rejecting the input prevents a bad row from
    silently shifting which candle supplies a later entry or exit.
    """
    missing = [column for column in OHLC_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    result = df.copy()
    for column in OHLC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    prices = result[OHLC_COLUMNS].to_numpy(dtype=float)
    invalid_value = (~np.isfinite(prices) | (prices <= 0)).any(axis=1)
    invalid_shape = (
        (result["high"] < result["low"])
        | (result["high"] < result[["open", "close"]].max(axis=1))
        | (result["low"] > result[["open", "close"]].min(axis=1))
    ).to_numpy()
    invalid = invalid_value | invalid_shape
    if invalid.any():
        bad_indices = result.index[invalid].tolist()[:10]
        raise ValueError(
            f"Found {int(invalid.sum())} rows with missing, non-finite, "
            f"non-positive, or inconsistent OHLC prices; sample indices: {bad_indices}"
        )
    return result


def load_market_data(path=DEFAULT_DATA_PATH, *, naive_timezone: str | None = None):
    """Load validated OHLCV data and normalize timestamps to UTC.

    Naive timestamps in the default project CSV are localized to UTC based on
    the parquet/CSV evidence. Naive timestamps from another path are rejected
    unless ``naive_timezone`` is explicitly supplied. Aware timestamps are
    converted by instant; mixed aware and naive values are always rejected.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Market data not found: {path}")

    df = pd.read_csv(path)
    input_rows = len(df)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    # The project CSV contains UTC clock values with its timezone marker
    # removed by export. Localize naive values to UTC; aware inputs are
    # normalized to UTC without changing their represented instant.
    timezone_for_naive = naive_timezone
    if timezone_for_naive is None and path.resolve() == DEFAULT_DATA_PATH.resolve():
        timezone_for_naive = "UTC"
    df["time"] = normalize_utc_timestamps(
        df["time"], assume_naive_timezone=timezone_for_naive
    )

    price_columns = ["open", "high", "low", "close", "volume"]

    for column in price_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_timestamp = df["time"].isna()
    dropped_missing_timestamps = int(missing_timestamp.sum())
    df = df.loc[~missing_timestamp].copy()
    df = validate_ohlc_prices(df)

    missing_volume = df["volume"].isna()
    dropped_missing_volume = int(missing_volume.sum())
    df = df.loc[~missing_volume].copy()

    duplicate_count = int(df["time"].duplicated(keep="last").sum())
    df = df.sort_values("time")
    df = df.drop_duplicates(subset="time", keep="last")
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid market data remains after cleaning.")

    df.attrs["load_report"] = {
        "input_rows": input_rows,
        "dropped_missing_timestamps": dropped_missing_timestamps,
        "dropped_missing_volume": dropped_missing_volume,
        "dropped_duplicate_timestamps": duplicate_count,
        "output_rows": len(df),
    }

    return df


if __name__ == "__main__":
    data = load_market_data()

    print("EUR/USD data loaded successfully")
    print(f"Rows: {len(data):,}")
    print(f"First candle: {data['time'].iloc[0]}")
    print(f"Last candle:  {data['time'].iloc[-1]}")
    print(f"Missing values: {int(data.isna().sum().sum())}")
    print(f"Duplicate timestamps: {int(data['time'].duplicated().sum())}")
