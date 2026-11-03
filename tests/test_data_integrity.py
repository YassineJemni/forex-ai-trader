import pandas as pd

from src.research.data_integrity import audit_frame


def frame(times, rows):
    return pd.DataFrame({
        "time": pd.to_datetime(times, utc=True),
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
    })


def test_flags_pre_euro_eurusd_and_keeps_exact_timestamp():
    df = frame(["1971-01-03T22:00:00Z"], [(0.53, 0.53, 0.53, 0.53)])
    result = audit_frame(df, "EURUSD", "H4")
    assert result["date_plausibility_findings"][0]["timestamp"] == "1971-01-03T22:00:00+00:00"
    assert result["timezone_aware_utc"]


def test_detects_duplicate_nonmonotonic_invalid_ohlc_and_gap_boundaries():
    df = frame(["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T00:30:00Z"],
               [(1, 1.1, .9, 1), (1, 1.1, .9, 1), (1, .8, .9, 1)])
    result = audit_frame(df, "EURUSD", "M15")
    assert result["duplicate_timestamp_count"] == 1
    assert result["invalid_ohlc_count"] == 1
    assert result["gaps"][0]["from"] == "2024-01-01T00:00:00+00:00"
    assert result["gaps"][0]["to"] == "2024-01-01T00:30:00+00:00"


def test_flags_long_flat_runs_and_extreme_close_jump():
    df = frame([f"2024-01-01T0{i}:00:00Z" for i in range(4)],
               [(1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1.2, 1.2, 1.2, 1.2)])
    result = audit_frame(df, "GBPUSD", "H1")
    assert result["exact_flat_runs"][0]["candles"] == 3
    assert result["price_jumps_over_10pct"][0]["to"] == "2024-01-01T03:00:00+00:00"


def test_naive_timestamp_is_not_certified_as_utc():
    df = frame(["2024-01-01T00:00:00Z"], [(1, 1, 1, 1)])
    df["time"] = df["time"].dt.tz_localize(None)
    assert not audit_frame(df, "AUDUSD", "H1")["timezone_aware_utc"]


def test_sustained_daily_sampling_in_h4_is_reported_as_cadence_anomaly():
    days = pd.date_range("1971-01-01 22:00", periods=240, freq="D", tz="UTC")
    df = frame(days, [(1, 1, 1, 1)] * len(days))
    result = audit_frame(df, "USDJPY", "H4")
    year = result["sustained_daily_sampling_years_in_h4"][0]
    assert year["year"] == 1971
    assert year["daily_intervals"] >= 200
