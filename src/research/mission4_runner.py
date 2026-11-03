"""Training-only Mission 4 predictability-map runner and report writer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

from .data import normalize_utc_timestamps, validate_m1_cadence
from .execution_models import MarketDataMetadata
from .experiment_store import PROJECT_ROOT, canonical_json
from .labels import (
    FROZEN_TRAINING_BOUNDARY,
    TrainingBoundarySpec,
    build_mid_return_samples,
)
from .predictability import evaluate_predictability_map


DEFAULT_BID_DATA_DIR = (
    PROJECT_ROOT / "market-data" / "m1" / "Forex" / "Majors" / "EUR-USD" / "BID"
)
DEFAULT_ASK_DATA_DIR = (
    PROJECT_ROOT / "market-data" / "m1" / "Forex" / "Majors" / "EUR-USD" / "ASK"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
PRICE_COLUMNS = ("time", "open", "high", "low", "close", "volume")
TIMESTAMP_COLUMN = "time"
REPORT_STAMP_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")


def _open_parquet(path: Path) -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "PyArrow is required for bounded Parquet batch reads in Mission 4"
        ) from exc
    return parquet.ParquetFile(path)


def _monthly_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("????-??.parquet"))
    if not files:
        raise FileNotFoundError(f"no monthly parquet files found under {data_dir}")
    return files


def _read_file_timestamps(path: Path) -> pd.Series:
    table = _open_parquet(path).read(columns=[TIMESTAMP_COLUMN])
    if TIMESTAMP_COLUMN not in table.column_names:
        raise ValueError(f"{path} has no {TIMESTAMP_COLUMN!r} column")
    values = table[TIMESTAMP_COLUMN].to_pandas()
    timestamps = normalize_utc_timestamps(values, assume_naive_timezone="UTC")
    if timestamps.empty:
        raise ValueError(f"{path} contains no timestamps")
    if timestamps.isna().any():
        raise ValueError(f"{path} contains invalid timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{path} timestamps are not in chronological order")
    if timestamps.duplicated().any():
        raise ValueError(f"{path} contains duplicate timestamps")
    return timestamps.reset_index(drop=True)


def _verify_original_boundary(
    bid_timestamps: pd.Series,
    boundary_spec: TrainingBoundarySpec,
) -> tuple[pd.Timestamp, pd.Series]:
    ordered = bid_timestamps.sort_values(kind="stable").reset_index(drop=True)
    if len(ordered) <= boundary_spec.training_rows:
        raise ValueError(
            "cannot verify frozen training boundary: original BID timeline "
            f"ends before index {boundary_spec.training_rows}"
        )
    prefix_through_boundary = ordered.iloc[:boundary_spec.training_rows + 1]
    if prefix_through_boundary.isna().any():
        raise ValueError(
            "cannot verify frozen training boundary: invalid timestamp before boundary"
        )
    if prefix_through_boundary.duplicated().any():
        raise ValueError(
            "cannot verify frozen training boundary: duplicate timestamp before boundary"
        )
    try:
        validate_m1_cadence(prefix_through_boundary)
    except ValueError as exc:
        raise ValueError(f"cannot verify frozen training boundary: {exc}") from exc

    expected = pd.Timestamp(boundary_spec.boundary_timestamp_utc)
    observed = ordered.iloc[boundary_spec.training_rows]
    if observed != expected:
        raise ValueError(
            "cannot verify frozen training boundary: timestamp at original "
            f"index {boundary_spec.training_rows} is {observed.isoformat()}, "
            f"expected {expected.isoformat()}"
        )
    return expected, ordered.iloc[:boundary_spec.training_rows].reset_index(drop=True)


def _read_training_prices(
    path: Path,
    file_timestamps: pd.Series,
    boundary: pd.Timestamp,
) -> pd.DataFrame:
    """Read only pre-boundary rows from each Parquet row group."""
    parquet_file = _open_parquet(path)
    row_group_count = parquet_file.metadata.num_row_groups
    frames: list[pd.DataFrame] = []
    timestamp_offset = 0

    for row_group_index in range(row_group_count):
        row_group_rows = parquet_file.metadata.row_group(row_group_index).num_rows
        group_timestamps = file_timestamps.iloc[
            timestamp_offset:timestamp_offset + row_group_rows
        ]
        if len(group_timestamps) != row_group_rows:
            raise ValueError(f"{path} row-group metadata does not match its timestamps")
        eligible_rows = int(group_timestamps.lt(boundary).sum())
        if eligible_rows:
            batches = parquet_file.iter_batches(
                row_groups=[row_group_index],
                batch_size=eligible_rows,
                columns=list(PRICE_COLUMNS),
            )
            batch = next(iter(batches), None)
            if batch is None or batch.num_rows != eligible_rows:
                raise ValueError(
                    f"{path} did not return the expected pre-boundary row-group prefix"
                )
            frames.append(batch.to_pandas())
        timestamp_offset += row_group_rows

    if timestamp_offset != len(file_timestamps):
        raise ValueError(f"{path} row-group metadata does not cover its timestamps")
    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    if not set(PRICE_COLUMNS).issubset(result.columns):
        raise ValueError(f"{path} is missing one or more required price columns")
    result["time"] = normalize_utc_timestamps(
        result["time"], assume_naive_timezone="UTC",
    )
    if result["time"].ge(boundary).any():
        raise ValueError(f"{path} returned a price row at or after the boundary")
    expected = file_timestamps.loc[file_timestamps < boundary].reset_index(drop=True)
    if not result["time"].reset_index(drop=True).equals(expected):
        raise ValueError(f"{path} price-row timestamps do not match its timestamp scan")
    return result.loc[:, list(PRICE_COLUMNS)]


def _load_training_side(
    data_dir: Path,
    *,
    boundary: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
    files = _monthly_files(data_dir)
    per_file_timestamps = [(path, _read_file_timestamps(path)) for path in files]
    all_timestamps = pd.concat(
        [timestamps for _, timestamps in per_file_timestamps],
        ignore_index=True,
    )
    if not all_timestamps.is_monotonic_increasing:
        raise ValueError(f"monthly timestamps under {data_dir} are not chronological")
    if all_timestamps.duplicated().any():
        raise ValueError(f"monthly timestamps under {data_dir} contain duplicates")

    price_frames = [
        _read_training_prices(path, timestamps, boundary)
        for path, timestamps in per_file_timestamps
        if timestamps.lt(boundary).any()
    ]
    training_prices = pd.concat(price_frames, ignore_index=True)
    training_prices["time"] = normalize_utc_timestamps(
        training_prices["time"], assume_naive_timezone="UTC",
    )
    if not training_prices["time"].is_monotonic_increasing:
        raise ValueError(f"training prices under {data_dir} are not chronological")

    source = {
        "directory": str(data_dir.resolve()),
        "files": [str(path.resolve()) for path in files],
        "timestamp_rows_scanned": int(len(all_timestamps)),
        "training_price_rows_emitted": int(len(training_prices)),
        "first_timestamp_utc": all_timestamps.iloc[0].isoformat(),
        "last_timestamp_utc": all_timestamps.iloc[-1].isoformat(),
        "training_first_timestamp_utc": (
            all_timestamps.loc[all_timestamps < boundary].iloc[0].isoformat()
        ),
        "training_last_timestamp_utc": (
            all_timestamps.loc[all_timestamps < boundary].iloc[-1].isoformat()
        ),
    }
    return training_prices, source, all_timestamps


def _validate_report_stamp(stamp: str | None) -> str:
    if stamp is None:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not REPORT_STAMP_PATTERN.fullmatch(stamp):
        raise ValueError("report stamp must use YYYYMMDDTHHMMSSZ format")
    try:
        datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ValueError("report stamp must be a valid UTC timestamp") from exc
    return stamp


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _markdown_report(report: dict[str, Any]) -> str:
    labels = report["labels"]
    evaluation = report["evaluation"]
    boundary = labels["boundary"]
    lines = [
        "# Mission 4: Training-Only Predictability Map",
        "",
        f"- Run: `{report['run_id']}`",
        "- Scope: frozen training data only; no validation or historical-test prices, features, labels, or metrics",
        f"- Frozen boundary: index `{boundary['verified_index']}`, "
        f"`{boundary['verified_timestamp_utc']}`",
        f"- Input training-frame fingerprint (BID/ASK): "
        f"`{labels['training_data_fingerprint_sha256']}`",
        f"- Accepted labels: {labels['accepted']}; rejected labels: {labels['rejected']}",
        f"- Rejections by reason: `{canonical_json(labels['rejected_by_reason'])}`",
        f"- Rejected sample records included: {len(report['rejected_samples'])}",
        "",
        "## Fixed Evaluation Protocol",
        "",
        f"- Family: `{evaluation['family_id']}` "
        f"({evaluation['family']['size']} feature-horizon tests)",
        f"- Features: {', '.join(evaluation['family']['features'])}",
        f"- Horizons: {', '.join(evaluation['family']['horizons'])}",
        f"- Effect: {evaluation['inference']['effect']}",
        f"- Uncertainty: {evaluation['inference']['uncertainty']}",
        f"- Multiple testing: {evaluation['inference']['multiple_testing']}",
        f"- Null control: {evaluation['inference']['null_control']}",
        "",
        "## Results",
        "",
        "| Feature | Horizon | N | Correlation | HAC SE | HAC 95% CI | HAC p | Holm p | Shift FWER p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in evaluation["results"]:
        interval = result["hac_95_ci"]
        interval_text = (
            "n/a" if interval is None
            else f"[{interval[0]:.6g}, {interval[1]:.6g}]"
        )
        values = (
            result["correlation"],
            result["hac_standard_error"],
            result["hac_p_value"],
            result["holm_adjusted_p_value"],
            result["shift_max_statistic_fwer_p_value"],
        )
        formatted = [
            "n/a" if value is None else f"{value:.6g}"
            for value in values
        ]
        lines.append(
            f"| {result['feature']} | {result['horizon']} | {result['n']} | "
            f"{formatted[0]} | {formatted[1]} | {interval_text} | "
            f"{formatted[2]} | {formatted[3]} | {formatted[4]} |"
        )
    lines.extend([
        "",
        "## Sample Accounting",
        "",
        f"```json\n{json.dumps(report['sample_accounting'], indent=2, sort_keys=True)}\n```",
        "",
        "## Input and Provenance",
        "",
        f"```json\n{json.dumps(report['inputs'], indent=2, sort_keys=True)}\n```",
        "",
        f"```json\n{json.dumps(report['provenance'], indent=2, sort_keys=True)}\n```",
        "",
        "This exploratory association map is not a trading strategy or evidence of executable profitability.",
        "",
    ])
    return "\n".join(lines)


def write_mission4_reports(
    report: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"mission4_predictability_{report['run_id']}"
    json_path = destination / f"{stem}.json"
    markdown_path = destination / f"{stem}.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(f"Mission 4 report already exists for run {report['run_id']}")

    json_content = json.dumps(
        json.loads(canonical_json(report)),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    markdown_content = _markdown_report(report)
    created: list[Path] = []
    try:
        for path, content in (
            (json_path, json_content),
            (markdown_path, markdown_content),
        ):
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            created.append(path)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return json_path, markdown_path


def run_mission4(
    *,
    bid_data_dir: str | Path = DEFAULT_BID_DATA_DIR,
    ask_data_dir: str | Path = DEFAULT_ASK_DATA_DIR,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
    boundary_spec: TrainingBoundarySpec = FROZEN_TRAINING_BOUNDARY,
    report_stamp: str | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Verify the frozen boundary, evaluate training-only data, and write reports."""
    bid_dir = Path(bid_data_dir)
    ask_dir = Path(ask_data_dir)
    run_id = _validate_report_stamp(report_stamp)
    run_started_at = _utc_now()
    output_path = Path(output_dir)
    report_stem = f"mission4_predictability_{run_id}"
    if (
        (output_path / f"{report_stem}.json").exists()
        or (output_path / f"{report_stem}.md").exists()
    ):
        raise FileExistsError(f"Mission 4 report already exists for run {run_id}")

    # Timestamp-only scans establish the boundary before any price data is read.
    bid_files = _monthly_files(bid_dir)
    bid_file_timestamps = [(path, _read_file_timestamps(path)) for path in bid_files]
    bid_timestamps = pd.concat(
        [timestamps for _, timestamps in bid_file_timestamps],
        ignore_index=True,
    )
    if not bid_timestamps.is_monotonic_increasing:
        raise ValueError(f"monthly BID timestamps under {bid_dir} are not chronological")
    if bid_timestamps.duplicated().any():
        raise ValueError(f"monthly BID timestamps under {bid_dir} contain duplicates")
    boundary, expected_training_times = _verify_original_boundary(
        bid_timestamps, boundary_spec,
    )

    bid_prices = pd.concat([
        _read_training_prices(path, timestamps, boundary)
        for path, timestamps in bid_file_timestamps
        if timestamps.lt(boundary).any()
    ], ignore_index=True)
    bid_prices["time"] = normalize_utc_timestamps(
        bid_prices["time"], assume_naive_timezone="UTC",
    )
    if not bid_prices["time"].reset_index(drop=True).equals(expected_training_times):
        raise ValueError("loaded BID training prices do not match the verified timeline")

    ask_prices, ask_source, ask_timestamps = _load_training_side(
        ask_dir, boundary=boundary,
    )
    expected_ask_times = ask_timestamps.loc[ask_timestamps < boundary].reset_index(drop=True)
    if not expected_ask_times.equals(expected_training_times):
        raise ValueError(
            "cannot verify exact BID/ASK training alignment: timestamps differ before boundary"
        )
    if not ask_prices["time"].reset_index(drop=True).equals(expected_training_times):
        raise ValueError("loaded ASK training prices do not match the verified timeline")

    boundary_row = pd.DataFrame([{
        "time": boundary,
        "open": np.nan,
        "high": np.nan,
        "low": np.nan,
        "close": np.nan,
        "volume": np.nan,
    }])
    bid_label_input = pd.concat([bid_prices, boundary_row], ignore_index=True)
    bid_metadata = MarketDataMetadata(
        "BID", "EUR/USD", "M1", "bar_open", "UTC", "Dukascopy",
    )
    ask_metadata = MarketDataMetadata(
        "ASK", "EUR/USD", "M1", "bar_open", "UTC", "Dukascopy",
    )
    samples, label_audit = build_mid_return_samples(
        bid_label_input,
        ask_prices,
        bid_metadata=bid_metadata,
        ask_metadata=ask_metadata,
        boundary_spec=boundary_spec,
    )
    if label_audit["boundary"]["verified_timestamp_utc"] != boundary.isoformat():
        raise ValueError("label builder boundary verification disagrees with timestamp scan")
    evaluation = evaluate_predictability_map(
        samples,
        training_boundary_utc=boundary.isoformat(),
    )

    bid_source = {
        "directory": str(bid_dir.resolve()),
        "files": [str(path.resolve()) for path, _ in bid_file_timestamps],
        "timestamp_rows_scanned": int(len(bid_timestamps)),
        "training_price_rows_emitted": int(len(bid_prices)),
        "first_timestamp_utc": bid_timestamps.iloc[0].isoformat(),
        "last_timestamp_utc": bid_timestamps.iloc[-1].isoformat(),
        "training_first_timestamp_utc": expected_training_times.iloc[0].isoformat(),
        "training_last_timestamp_utc": expected_training_times.iloc[-1].isoformat(),
    }
    rejected_samples = [
        {
            "decision_timestamp": row.decision_timestamp.isoformat(),
            "entry_timestamp": row.entry_timestamp.isoformat(),
            "exit_timestamp": row.exit_timestamp.isoformat(),
            "horizon": row.horizon,
            "rejection_reason": row.rejection_reason,
        }
        for row in samples.loc[~samples["accepted"]].itertuples(index=False)
    ]
    generated_at = _utc_now()
    report = {
        "schema_version": "mission4_training_predictability_report_v1",
        "run_id": run_id,
        "run_started_at_utc": run_started_at.isoformat(),
        "generated_at_utc": generated_at.isoformat(),
        "scope": "frozen_training_only",
        "inputs": {
            "bid": bid_source,
            "ask": ask_source,
            "training_data_fingerprint_sha256": (
                label_audit["training_data_fingerprint_sha256"]
            ),
            "fingerprint_scope": "normalized BID and ASK price frames strictly before the frozen boundary",
            "price_loading_method": (
                "Parquet row-group batches capped at the pre-boundary timestamp "
                "count; no post-boundary price rows are emitted to validation, "
                "features, labels, or the evaluator"
            ),
            "post_boundary_price_rows_emitted_or_analyzed": 0,
            "boundary_timestamp_prices_loaded": False,
        },
        "labels": label_audit,
        "rejected_samples": rejected_samples,
        "sample_accounting": {
            "accepted": label_audit["accepted"],
            "rejected": label_audit["rejected"],
            "rejected_by_reason": label_audit["rejected_by_reason"],
            "evaluator": evaluation["sample_accounting"],
        },
        "evaluation": evaluation,
        "provenance": {
            "label_builder": label_audit["software_provenance"],
            "evaluator": evaluation["software_provenance"],
            "invocation": {
                "command": "python -m src.research.mission4_runner",
                "arguments": {
                    "--bid-data-dir": str(bid_dir.resolve()),
                    "--ask-data-dir": str(ask_dir.resolve()),
                    "--output-dir": str(output_path.resolve()),
                    "--report-stamp": run_id,
                },
            },
        },
    }
    json_path, markdown_path = write_mission4_reports(report, output_dir=output_dir)
    return report, json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen training-only Mission 4 predictability map."
    )
    parser.add_argument("--bid-data-dir", type=Path, default=DEFAULT_BID_DATA_DIR)
    parser.add_argument("--ask-data-dir", type=Path, default=DEFAULT_ASK_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--report-stamp",
        help="optional UTC filename stamp in YYYYMMDDTHHMMSSZ format",
    )
    args = parser.parse_args(argv)
    _, json_path, markdown_path = run_mission4(
        bid_data_dir=args.bid_data_dir,
        ask_data_dir=args.ask_data_dir,
        output_dir=args.output_dir,
        report_stamp=args.report_stamp,
    )
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
