import joblib
import numpy as np
import pandas as pd
import pytest

from src.trader.model_registry import ModelRegistry, promote_candidate
from src.trader.retraining import retrain_from_history
from src.trader.history import ScanHistory
from src.trader.training import (make_samples, train_candidate,
                                 train_candidate_from_labeled_samples,
                                 expected_calibration_error, FEATURES)


def market(n=1800, start="2025-01-01"):
    rng = np.random.default_rng(42)
    shocks = rng.normal(.000002, .00008, n)
    close = 1.1 + np.cumsum(shocks)
    times = pd.date_range(start, periods=n, freq="min", tz="UTC")
    open_ = close - shocks / 2
    return pd.DataFrame({"time": times, "open": open_,
                         "high": np.maximum(open_, close) + .00005,
                         "low": np.minimum(open_, close) - .00005,
                         "close": close, "volume": np.ones(n)})


def test_feature_samples_are_labeled_on_exact_elapsed_times_and_chronological():
    data = market(800)
    samples = make_samples(data)
    assert len(samples) > 500
    assert samples.target.isin([0, 1]).all()
    assert (samples.exit_time - samples.entry_time == pd.Timedelta(minutes=15)).all()
    assert samples.time.is_monotonic_increasing


def test_training_saves_candidate_but_never_marks_it_promoted(tmp_path):
    path = tmp_path / "candidate.joblib"
    result = train_candidate(market(), path, minimum_validation_samples=100,
                             minimum_validation_trades=10)
    bundle = joblib.load(path)
    assert bundle["status"] == "candidate"
    assert bundle["metadata"]["timeframe"] == "M1"
    assert bundle["metadata"]["final_20_percent"] == "not read for features, labels, or metrics"
    assert result.promotion_status == "candidate_only"
    assert result.validation_rows >= 100
    assert not ModelRegistry(tmp_path / "current.joblib").available


def test_promotion_requires_independent_later_data_and_passing_metrics(tmp_path):
    data = market()
    candidate = tmp_path / "candidate.joblib"
    train_candidate(data, candidate, minimum_validation_samples=100,
                    minimum_validation_trades=10)
    with pytest.raises(ValueError, match="development validation gate|too few labeled|matches the candidate training fingerprint|must occur after"):
        promote_candidate(candidate, tmp_path / "current.joblib",
                          confirmation_candles=data, approved_by="test reviewer",
                          minimum_sample_count=100, minimum_trades=10)


def test_promoted_model_inference_is_timeframe_locked(tmp_path):
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy="constant", constant=1).fit([[0], [1]], [0, 1])
    path = tmp_path / "current.joblib"
    joblib.dump({"status": "promoted", "model": model,
                 "feature_columns": ["x"],
                 "metadata": {"version": "test-version", "timeframe": "M1"}}, path)
    registry = ModelRegistry(path)
    assert registry.predict_probability({"x": 0.5}, timeframe="M1") == pytest.approx(1.0)
    assert registry.predict_probability({"x": 0.5}, timeframe="M5") is None
    assert registry.predict_probability({"wrong": 0.5}, timeframe="M1") is None


def test_calibration_error_distinguishes_calibrated_and_overconfident_predictions():
    y = np.array([0, 1, 0, 1, 0, 1])
    calibrated = np.array([.1, .9, .1, .9, .1, .9])
    overconfident = np.array([.4, .6, .4, .6, .4, .6])
    assert expected_calibration_error(y, calibrated) == pytest.approx(.1)
    assert expected_calibration_error(y, overconfident) == pytest.approx(.4)


def test_retraining_from_new_labels_creates_only_an_inert_candidate(tmp_path):
    rng = np.random.default_rng(7)
    samples = []
    times = pd.date_range("2026-01-01", periods=1600, freq="min", tz="UTC")
    for stamp in times:
        features = {name: float(rng.normal(0, .001)) for name in FEATURES}
        target = int(features["return_1"] > 0)
        samples.append({**features, "time": stamp, "target": target,
                        "future_pips": 2.0 if target else -2.0})
    output = tmp_path / "candidate.joblib"
    result = train_candidate_from_labeled_samples(samples, output, symbol="EURUSD",
                                                   minimum_validation_samples=100,
                                                   minimum_validation_trades=10)
    assert result.validation_rows > 100
    assert joblib.load(output)["status"] == "candidate"


def test_scheduled_retraining_skips_until_enough_labels_exist(tmp_path):
    history = ScanHistory(tmp_path / "history.sqlite3")
    result = retrain_from_history(history_path=history.path, symbol="EURUSD",
                                  timeframe="M1", output_path=tmp_path / "candidate.joblib",
                                  minimum_samples=1000)
    assert result["status"] == "not_enough_labels"
    assert result["promoted"] is False


def test_explicit_promotion_archives_a_versioned_active_artifact(tmp_path, monkeypatch):
    from src.trader import training
    from sklearn.dummy import DummyClassifier
    candidate_path = tmp_path / "candidate.joblib"
    current_path = tmp_path / "models" / "current.joblib"
    model = DummyClassifier(strategy="constant", constant=1).fit([[0], [1]], [0, 1])
    joblib.dump({"status": "candidate", "model": model,
                 "feature_columns": list(FEATURES),
                 "metadata": {"gate_passed": True, "horizon_minutes": 15,
                              "data_fingerprint": "training-data", "timeframe": "M1",
                              "training_end_utc": "2025-01-01T00:00:00+00:00",
                              "development_end_utc": "2025-01-02T00:00:00+00:00",
                              "training_prevalence": .5}}, candidate_path)
    samples = pd.DataFrame({name: np.zeros(600) for name in FEATURES})
    samples["time"] = pd.date_range("2026-01-01", periods=600, freq="min", tz="UTC")
    samples["target"] = np.tile([0, 1], 300)
    samples["future_pips"] = np.where(samples.target > 0, 3.0, -3.0)
    samples["entry_time"] = samples.time + pd.Timedelta(minutes=1)
    samples["exit_time"] = samples.time + pd.Timedelta(minutes=16)
    samples["exit_index"] = np.arange(600)
    monkeypatch.setattr(training, "make_samples", lambda *a, **k: samples)
    monkeypatch.setattr(training, "_score_non_overlapping", lambda *a, **k: (200, 1.0))
    monkeypatch.setattr(training, "expected_calibration_error", lambda *a, **k: .01)
    monkeypatch.setattr("sklearn.metrics.roc_auc_score", lambda *a, **k: .8)
    monkeypatch.setattr("sklearn.metrics.brier_score_loss",
                        lambda y, p: .1 if np.mean(p) > .6 else .25)
    from src.trader.model_registry import promote_candidate
    metadata = promote_candidate(candidate_path, current_path,
                                 confirmation_candles=pd.DataFrame({"time": ["2026-01-01"]}),
                                 approved_by="unit test", minimum_sample_count=500,
                                 minimum_trades=100)
    assert metadata["approved_by"] == "unit test"
    assert joblib.load(current_path)["status"] == "promoted"
    archived = list((current_path.parent / "versions").glob("*.joblib"))
    assert len(archived) == 1
    assert joblib.load(archived[0])["metadata"]["version"] == metadata["version"]
