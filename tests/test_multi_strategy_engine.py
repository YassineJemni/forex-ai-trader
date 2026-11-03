import numpy as np
import pandas as pd
import pytest

from src.research.multi_strategy_engine import (
    ResearchConfig, build_features, chronological_splits, generate_signals,
    load_research_frame, simulate_trades, strategy_library,
)


def market(n=360):
    times = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 1.1 + np.cumsum(np.sin(np.arange(n) / 8) * 0.0002 + 0.00001)
    return pd.DataFrame({"time": times, "open": close - 0.00003, "high": close + 0.00015,
                         "low": close - 0.00015, "close": close})


def test_strategy_library_is_independent_and_covers_requested_families():
    strategies = strategy_library()
    assert len({s.strategy_id for s in strategies}) == len(strategies)
    assert {"trend_following", "breakout", "mean_reversion", "market_structure",
            "liquidity_smc_hypothesis", "price_action"} <= {s.family for s in strategies}


def test_every_strategy_has_causal_timestamped_signal_output():
    data = market()
    features = build_features(data, "M15")
    for strategy in strategy_library():
        signals = generate_signals(features, strategy)
        assert set(("source_index", "decision_timestamp", "direction", "entry_logic",
                    "invalidation_logic", "target_logic", "evidence")) <= set(signals.columns)
        if len(signals):
            assert signals.source_index.min() >= 100
            assert set(signals.direction) <= {-1, 1}


def test_prefix_outputs_do_not_change_when_future_candles_are_appended():
    base = market()
    extended = pd.concat([base, market(40).assign(time=lambda d: pd.date_range(base.time.iloc[-1] + pd.Timedelta("15min"), periods=40, freq="15min", tz="UTC"),
                                                       close=lambda d: d.close + 0.5, open=lambda d: d.open + 0.5,
                                                       high=lambda d: d.high + 0.5, low=lambda d: d.low + 0.5)], ignore_index=True)
    strategy = next(s for s in strategy_library() if s.strategy_id == "donchian_breakout")
    left = strategy.signal_fn(build_features(base, "M15"))
    right = strategy.signal_fn(build_features(extended, "M15")).iloc[:len(base)]
    pd.testing.assert_series_equal(left, right)


def test_indicators_keep_state_across_weekend_closure():
    week1 = pd.date_range("2024-01-01", periods=30, freq="4h", tz="UTC")
    week2 = pd.date_range(week1[-1] + pd.Timedelta("52h"), periods=30, freq="4h")
    times = week1.append(week2)
    close = np.linspace(1.0, 1.1, len(times))
    frame = pd.DataFrame({"time": times, "open": close, "high": close + .001,
                          "low": close - .001, "close": close})
    features = build_features(frame, "H4")
    assert features.segment.nunique() == 1
    assert features.ema21.iloc[-1] > features.ema21.iloc[0]


def test_stop_wins_when_stop_and_target_touch_same_bar_and_costs_are_explicit():
    data = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
                         "open": [1.0, 1.0, 1.0], "high": [1.001, 1.003, 1.001],
                         "low": [.999, .997, .999], "close": [1.0, 1.0, 1.0]})
    features = data.copy()
    features["atr14"] = .001
    signals = pd.DataFrame([{"source_index": 0, "decision_timestamp": data.time.iloc[0],
                             "direction": 1, "regime": "ranging", "evidence": "test"}])
    result, _ = simulate_trades(data, features, signals, "EURUSD", "M15", ResearchConfig())
    trade = result.iloc[0]
    assert trade.exit_reason == "stop_and_target_same_bar_stop_first"
    assert trade.gross_pnl_pips == pytest.approx(-15)
    assert pd.isna(trade.net_pnl_pips)
    configured = ResearchConfig(spread_pips=1.0, commission_pips_round_trip=.5, slippage_pips_round_trip=.5)
    result, _ = simulate_trades(data, features, signals, "EURUSD", "M15", configured)
    assert result.iloc[0].costs_pips == pytest.approx(2.0)
    assert result.iloc[0].net_pnl_pips == pytest.approx(-17)


def test_position_size_uses_risk_budget_and_quote_conversion():
    data = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
                         "open": [150, 150, 150], "high": [151, 151, 151], "low": [149, 149, 149], "close": [150, 150, 150]})
    features = data.copy(); features["atr14"] = 1.0
    signals = pd.DataFrame([{"source_index": 0, "decision_timestamp": data.time.iloc[0], "direction": 1, "regime": "trending", "evidence": "test"}])
    result, _ = simulate_trades(data, features, signals, "USDJPY", "H1", ResearchConfig(account_balance=10_000, risk_fraction=.01))
    # $100 risk / 1.5 JPY stop / (1/150 USD per JPY) = 10,000 USD units.
    assert result.iloc[0].position_units == pytest.approx(10_000)
    assert result.iloc[0].risk_amount == pytest.approx(100)


def test_position_limit_respected_when_overlap_is_enabled():
    data = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC"),
                         "open": [1.0]*8, "high": [1.0001]*8, "low": [.9999]*8, "close": [1.0]*8})
    features = data.copy(); features["atr14"] = .001
    signals = pd.DataFrame([{"source_index": i, "decision_timestamp": data.time.iloc[i], "direction": 1,
                             "regime": "ranging", "evidence": "test"} for i in range(5)])
    config = ResearchConfig(allow_overlapping=True, max_open_positions=2)
    trades, rejected = simulate_trades(data, features, signals, "EURUSD", "H1", config)
    assert len(trades) <= 2
    assert rejected["overlap_skipped"] > 0


def test_splits_are_chronological_and_disjoint():
    splits = chronological_splits(market(100))
    assert splits["train"][1] == splits["validation"][0]
    assert splits["validation"][1] == splits["test"][0]
    assert splits["train"][0] < splits["train"][1] < splits["test"][1]


def test_h4_eurusd_exclusion_and_usdjpy_quarantine_preserve_source(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    derived = tmp_path / "derived"; quarantine = tmp_path / "quarantine"
    dates = pd.to_datetime(["1998-12-28T22:00Z", "1998-12-29T22:00Z", "1999-01-04T06:00Z", "1999-01-04T10:00Z"])
    frame = pd.DataFrame({"time": dates, "open": [1, 1, 1, 1], "high": [1.1]*4, "low": [.9]*4, "close": [1]*4})
    frame.to_parquet(source / "EURUSD_H4.parquet", index=False)
    frame.to_parquet(source / "USDJPY_H4.parquet", index=False)
    original = (source / "USDJPY_H4.parquet").read_bytes()
    eur, meta = load_research_frame("EURUSD", "H4", source, derived_dir=derived, quarantine_dir=quarantine)
    jpy, jmeta = load_research_frame("USDJPY", "H4", source, derived_dir=derived, quarantine_dir=quarantine)
    assert len(eur) == len(jpy) == 2
    assert meta["excluded_rows"] == jmeta["excluded_rows"] == 2
    assert len(pd.read_parquet(quarantine / "USDJPY_H4_pre1999.parquet")) == 2
    assert (source / "USDJPY_H4.parquet").read_bytes() == original
    assert (derived / "EURUSD_H4.provenance.json").exists()
    assert (quarantine / "USDJPY_H4_pre1999.provenance.json").exists()


def test_invalid_or_missing_timeseries_is_rejected(tmp_path):
    frame = market(10).iloc[::-1]
    path = tmp_path / "GBPUSD_M15.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate or unordered"):
        load_research_frame("GBPUSD", "M15", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_research_frame("AUDUSD", "H1", tmp_path)
    good = market(20)
    good.loc[3, "high"] = good.loc[3, "low"] - .1
    good.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="invalid OHLC"):
        load_research_frame("GBPUSD", "M15", tmp_path)
