import joblib
import numpy as np
import pandas as pd

from src.trader.multi_training import (BASE_FEATURES, build_model_features,
                                       build_multi_pair_samples,
                                       train_multi_pair_candidates)


def market(seed=4, count=1000, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.0001, count)
    close = 1.1 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.full(count, 0.00004)
    return pd.DataFrame({"time": pd.date_range(start, periods=count, freq="5min", tz="UTC"),
                         "open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close})


def test_model_features_are_causal_and_include_required_indicators():
    data = market()
    baseline = build_model_features(data, timeframe="M5")
    changed = data.copy()
    changed.loc[700:, ["open", "high", "low", "close"]] *= 1.2
    after = build_model_features(changed, timeframe="M5")
    pd.testing.assert_series_equal(baseline.loc[:699, "macd_pct"], after.loc[:699, "macd_pct"])
    assert {"macd_pct", "macd_signal_pct", "atr_pct", "trend_strength",
            "volatility_regime", "london_session", "new_york_session"}.issubset(baseline)


def test_multi_pair_labels_are_exact_bar_and_peer_features_are_present():
    eur = market(seed=10)
    gbp = market(seed=11)
    # Missing bars are kept missing and prevent labels that cross the gap.
    gbp = gbp.drop(index=350).reset_index(drop=True)
    samples = build_multi_pair_samples({"EURUSD": eur, "GBPUSD": gbp},
                                       timeframe="M5", horizon_bars=3,
                                       round_trip_cost_pips=1.5)
    assert set(samples.pair) == {"EURUSD", "GBPUSD"}
    assert samples.target.isin([0, 1]).all()
    assert samples.target_exceeds_cost.isin([0, 1]).all()
    assert samples.label_end_time.gt(samples.time).all()
    assert samples.peer_return_mean.notna().any()
    assert all(name in samples for name in BASE_FEATURES)
    for _, pair_rows in samples.groupby("pair"):
        ordered = pair_rows.sort_values("time")
        assert (ordered.label_end_time.iloc[:-1].reset_index(drop=True)
                < ordered.entry_time.iloc[1:].reset_index(drop=True)).all()
    gap_start = gbp.time.iloc[350]
    around_gap = samples[(samples.pair == "GBPUSD") &
                         (samples.time.between(gap_start - pd.Timedelta(minutes=20),
                                               gap_start + pd.Timedelta(minutes=20)))]
    assert len(around_gap) < 9


def test_training_creates_shared_and_pair_specific_versioned_candidates(tmp_path):
    reports = train_multi_pair_candidates(
        {"EURUSD": market(20, 1000), "GBPUSD": market(21, 1000)},
        timeframe="M5", output_dir=tmp_path / "candidates",
        round_trip_cost_pips=2.0, horizon_bars=3,
        minimum_train_rows=100, minimum_holdout_rows=20, minimum_trades=2,
        minimum_walk_forward_validation_rows=15,
    )
    created = [row for row in reports if row["status"] == "candidate_created"]
    assert {row["kind"] for row in created} == {"shared", "pair"}
    assert {row["algorithm"] for row in created} == {
        "logistic_regression", "hist_gradient_boosting"}
    for report in created:
        bundle = joblib.load(report["candidate_path"])
        assert bundle["status"] == "candidate"
        assert bundle["metadata"]["gate_passed"] in (True, False)
        assert bundle["metadata"]["dataset_fingerprint"]
        assert bundle["metadata"]["walk_forward_folds"]
        assert bundle["metadata"]["holdout_start_utc"] > bundle["metadata"]["training_end_utc"]
    assert (tmp_path / "candidate_registry.json").exists()
