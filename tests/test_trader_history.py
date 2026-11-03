import pytest

from src.trader.history import ScanHistory


def test_records_scans_predictions_and_one_observed_outcome(tmp_path):
    history = ScanHistory(tmp_path / "history.sqlite3")
    scan_id = history.record_scan({"status": "opportunity", "observations": [
        {"symbol": "EURUSD", "timeframe": "M1", "probability": .7,
         "direction": "BUY", "model_version": "v1", "candle_time": "2026-01-01T00:00:00+00:00",
         "features": {"return_1": .01}}]})
    snapshot = history.recent()
    assert snapshot["scans"][0]["scan_id"] == scan_id
    prediction_id = snapshot["predictions"][0]["prediction_id"]
    history.record_outcome(prediction_id, 3.25)
    assert history.recent()["predictions"][0]["outcome_return"] == 3.25
    assert history.labeled_samples(symbol="EURUSD", timeframe="M1")[0]["target"] == 1
    assert history.performance()["labeled_predictions"] == 1
    with pytest.raises(ValueError, match="already has an outcome"):
        history.record_outcome(prediction_id, -1)
