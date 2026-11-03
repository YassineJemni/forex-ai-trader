import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.labels import TrainingBoundarySpec
from src.research.mission4_runner import (
    _read_file_timestamps,
    _read_training_prices,
    run_mission4,
)


def market_data(rows=750, *, seed=17):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=rows, freq="min", tz="UTC")
    mid = 1.1 + np.cumsum(rng.normal(0.0, 0.00002, size=rows))
    bid = pd.DataFrame({
        "time": times,
        "open": mid,
        "high": mid + 0.00004,
        "low": mid - 0.00004,
        "close": mid + rng.normal(0.0, 0.00001, size=rows),
        "volume": np.ones(rows),
    })
    ask = bid.copy()
    ask[["open", "high", "low", "close"]] += 0.00002
    return bid, ask


class FakeBatch:
    def __init__(self, frame):
        self.frame = frame
        self.num_rows = len(frame)

    def to_pandas(self):
        return self.frame.copy()


class FakeColumn:
    def __init__(self, values):
        self.values = values

    def to_pandas(self):
        return self.values.copy()


class FakeTable:
    def __init__(self, frame):
        self.frame = frame
        self.column_names = list(frame.columns)

    def __getitem__(self, column):
        return FakeColumn(self.frame[column])


class FakeRowGroup:
    def __init__(self, num_rows):
        self.num_rows = num_rows


class FakeMetadata:
    def __init__(self, num_rows):
        self.num_row_groups = 1
        self._row_group = FakeRowGroup(num_rows)

    def row_group(self, index):
        if index != 0:
            raise IndexError(index)
        return self._row_group


class FakeParquetFile:
    def __init__(self, frame, batch_requests):
        self.frame = frame
        self.metadata = FakeMetadata(len(frame))
        self.batch_requests = batch_requests

    def read(self, *, columns):
        return FakeTable(self.frame.loc[:, columns].copy())

    def iter_batches(self, *, row_groups, batch_size, columns):
        if row_groups != [0]:
            raise IndexError(row_groups)
        self.batch_requests.append(batch_size)
        yield FakeBatch(self.frame.loc[:, columns].iloc[:batch_size].copy())


class Mission4RunnerTests(unittest.TestCase):
    def test_runner_streams_training_prefix_and_writes_complete_reports(self):
        bid, ask = market_data()
        train_rows = 700
        boundary = bid.loc[train_rows, "time"]
        spec = TrainingBoundarySpec(
            training_rows=train_rows,
            boundary_timestamp_utc=boundary.isoformat().replace("+00:00", "Z"),
        )

        # Invalid future prices would fail OHLC validation if the runner emitted
        # them to the label builder.
        bid.loc[train_rows + 1:, ["open", "high", "low", "close"]] = -1.0
        ask.loc[train_rows + 1:, ["open", "high", "low", "close"]] = -1.0
        sources = {"bid": bid, "ask": ask}
        batch_requests = {"bid": [], "ask": []}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = {}
            for side in ("bid", "ask"):
                directories[side] = root / side.upper()
                directories[side].mkdir()
                (directories[side] / "2025-01.parquet").touch()

            def open_fake(path):
                side = Path(path).parent.name.lower()
                return FakeParquetFile(sources[side], batch_requests[side])

            with (
                patch("src.research.mission4_runner._open_parquet", side_effect=open_fake),
                patch(
                    "src.research.mission4_runner._utc_now",
                    side_effect=[
                        datetime(2026, 11, 3, 5, 0, tzinfo=timezone.utc),
                        datetime(2026, 11, 3, 5, 1, tzinfo=timezone.utc),
                    ],
                ),
            ):
                report, json_path, markdown_path = run_mission4(
                    bid_data_dir=directories["bid"],
                    ask_data_dir=directories["ask"],
                    output_dir=root / "reports",
                    boundary_spec=spec,
                    report_stamp="20261103T120000Z",
                )

            self.assertEqual(report["run_id"], "20261103T120000Z")
            self.assertEqual(
                report["run_started_at_utc"],
                "2026-11-03T05:00:00+00:00",
            )
            self.assertEqual(
                report["generated_at_utc"],
                "2026-11-03T05:01:00+00:00",
            )
            self.assertNotEqual(
                report["generated_at_utc"],
                datetime.strptime(report["run_id"], "%Y%m%dT%H%M%SZ")
                .replace(tzinfo=timezone.utc)
                .isoformat(),
            )
            self.assertEqual(batch_requests, {"bid": [train_rows], "ask": [train_rows]})
            self.assertEqual(report["inputs"]["post_boundary_price_rows_emitted_or_analyzed"], 0)
            self.assertEqual(report["inputs"]["boundary_timestamp_prices_loaded"], False)
            self.assertEqual(report["labels"]["boundary"]["verified_index"], train_rows)
            self.assertEqual(report["labels"]["boundary"]["verified_timestamp_utc"], boundary.isoformat())
            self.assertEqual(report["labels"]["training_bid_rows"], train_rows)
            self.assertEqual(report["labels"]["training_ask_rows"], train_rows)
            self.assertEqual(
                len(report["rejected_samples"]),
                report["sample_accounting"]["rejected"],
            )
            self.assertEqual(
                report["inputs"]["bid"]["training_last_timestamp_utc"],
                bid.loc[train_rows - 1, "time"].isoformat(),
            )
            self.assertEqual(len(report["evaluation"]["results"]), 64)
            self.assertEqual(report["evaluation"]["family"]["size"], 64)
            self.assertEqual(
                report["inputs"]["training_data_fingerprint_sha256"],
                report["labels"]["training_data_fingerprint_sha256"],
            )
            self.assertEqual(json_path.name, "mission4_predictability_20261103T120000Z.json")
            self.assertEqual(markdown_path.name, "mission4_predictability_20261103T120000Z.md")
            self.assertIn('"feature_horizon_tests": 64', json_path.read_text(encoding="utf-8"))
            self.assertIn("| Shift FWER p |", markdown_path.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                from src.research.mission4_runner import write_mission4_reports

                write_mission4_reports(report, output_dir=root / "reports")

    def test_boundary_mismatch_fails_before_any_price_batch_is_requested(self):
        bid, ask = market_data(rows=30)
        train_rows = 20
        expected = bid.loc[train_rows, "time"]
        spec = TrainingBoundarySpec(
            training_rows=train_rows,
            boundary_timestamp_utc=(expected + pd.Timedelta("1min")).isoformat().replace(
                "+00:00", "Z",
            ),
        )
        sources = {"bid": bid, "ask": ask}
        batch_requests = {"bid": [], "ask": []}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = {}
            for side in ("bid", "ask"):
                directories[side] = root / side.upper()
                directories[side].mkdir()
                (directories[side] / "2025-01.parquet").touch()

            def open_fake(path):
                side = Path(path).parent.name.lower()
                return FakeParquetFile(sources[side], batch_requests[side])

            with patch("src.research.mission4_runner._open_parquet", side_effect=open_fake):
                with self.assertRaisesRegex(ValueError, "timestamp at original index 20"):
                    run_mission4(
                        bid_data_dir=directories["bid"],
                        ask_data_dir=directories["ask"],
                        output_dir=root / "reports",
                        boundary_spec=spec,
                    )

        self.assertEqual(batch_requests, {"bid": [], "ask": []})

    def test_runner_fails_closed_when_training_bid_ask_timestamps_differ(self):
        bid, ask = market_data()
        train_rows = 700
        boundary = bid.loc[train_rows, "time"]
        spec = TrainingBoundarySpec(
            training_rows=train_rows,
            boundary_timestamp_utc=boundary.isoformat().replace("+00:00", "Z"),
        )
        ask = ask.drop(index=100).reset_index(drop=True)
        sources = {"bid": bid, "ask": ask}
        batch_requests = {"bid": [], "ask": []}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = {}
            for side in ("bid", "ask"):
                directories[side] = root / side.upper()
                directories[side].mkdir()
                (directories[side] / "2025-01.parquet").touch()

            def open_fake(path):
                side = Path(path).parent.name.lower()
                return FakeParquetFile(sources[side], batch_requests[side])

            with patch("src.research.mission4_runner._open_parquet", side_effect=open_fake):
                with self.assertRaisesRegex(ValueError, "exact BID/ASK training alignment"):
                    run_mission4(
                        bid_data_dir=directories["bid"],
                        ask_data_dir=directories["ask"],
                        output_dir=root / "reports",
                        boundary_spec=spec,
                    )

    def test_partial_parquet_row_group_yields_only_its_training_prefix(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("PyArrow is not installed")

        bid, _ = market_data(rows=12)
        boundary_index = 7
        boundary = bid.loc[boundary_index, "time"]
        bid.loc[boundary_index:, ["open", "high", "low", "close"]] = -1.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "single_row_group.parquet"
            pq.write_table(
                pa.Table.from_pandas(bid, preserve_index=False),
                path,
                row_group_size=len(bid),
            )
            timestamps = _read_file_timestamps(path)
            prefix = _read_training_prices(path, timestamps, boundary)

        self.assertEqual(len(prefix), boundary_index)
        self.assertTrue(prefix["time"].lt(boundary).all())
        self.assertTrue((prefix["open"] > 0).all())
        self.assertEqual(prefix["time"].iloc[-1], bid.loc[boundary_index - 1, "time"])

    def test_report_stamp_is_required_to_be_a_valid_utc_timestamp(self):
        from src.research.mission4_runner import _validate_report_stamp

        self.assertEqual(_validate_report_stamp("20261103T120000Z"), "20261103T120000Z")
        with self.assertRaisesRegex(ValueError, "YYYYMMDDTHHMMSSZ"):
            _validate_report_stamp("2026-11-03")


if __name__ == "__main__":
    unittest.main()
