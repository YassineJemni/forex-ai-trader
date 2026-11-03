"""Coordinate selected, registered research runs without strategy search.

Example::

    python -m src.research.experiment_orchestrator --strategy ema_crossover --dry-run
    python -m src.research.experiment_orchestrator --strategy ema_crossover \
        --strategy rsi_mean_reversion --summary reports/orchestration.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from .compare_strategies import (
    ComparisonRunResult,
    comparison_request_args,
    prepare_comparison_context,
    run_comparison,
    validate_selected_strategy_ids,
)
from .experiment_store import (
    DEFAULT_DB_PATH,
    ExecutionClaimBusy,
    ExecutionClaimLost,
    ExperimentStore,
    FailedRunRequiresRetry,
)


@dataclass
class OrchestrationResult:
    orchestration_id: str | None
    status: str
    strategy_ids: list[str]
    strategies: list[dict[str, Any]]
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "orchestration_id": self.orchestration_id,
            "status": self.status,
            "strategy_ids": list(self.strategy_ids),
            "strategies": self.strategies,
            "dry_run": self.dry_run,
        }


def _disposition(handle: Any) -> str:
    if handle.reused_completed:
        return "reused_completed"
    if handle.resumed:
        return "resumed"
    return "new"


def _stored_strategy_summary(
    store: ExperimentStore,
    strategy_id: str,
    name: str,
    disposition: str,
    status: str,
    experiment_id: str | None,
    *,
    attempt: int | None = None,
    error: str | None = None,
    identity_differences: list[str] | None = None,
    historical_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "strategy_id": strategy_id,
        "name": name,
        "disposition": disposition,
        "status": status,
        "experiment_id": experiment_id,
        "attempt": attempt,
        "error": error,
        "identity_differences": identity_differences or [],
        "historical_candidates": historical_candidates or [],
        "periods": {},
    }
    if experiment_id is None:
        return output
    run = store.get_run(experiment_id)
    output.update({
        "strategy_version": run["strategy_version"],
        "parameters": run["parameters"],
        "dataset_fingerprint": run["dataset_fingerprint"],
        "dataset_identity": run["dataset_identity"],
        "date_start_utc": run["date_start_utc"],
        "date_end_utc": run["date_end_utc"],
        "split_metadata": run["split_metadata"],
        "execution_assumptions": run["execution_assumptions"],
        "gap_policy": run["gap_policy"],
        "data_quality": run["data_quality"],
        "software": run["software"],
    })
    # Train and validation are the only periods returned by the store's
    # selection-facing API. Test is attached separately with its guard flags.
    for period_name in ("train", "validation"):
        try:
            record = store.get_period_result(experiment_id, period_name)
        except KeyError:
            continue
        output["periods"][period_name] = {
            "metrics": record["metrics"],
            "costs": record["costs"],
            "rejection_counts": record["rejection_counts"],
            "counts": record["counts"],
            "data_quality": record["data_quality"],
            "confirmatory": bool(record["confirmatory"]),
            "selection_eligible": bool(record["selection_eligible"]),
        }
    try:
        test_record = store.get_period_result(experiment_id, "test")
    except KeyError:
        pass
    else:
        output["periods"]["test"] = {
            "metrics": test_record["metrics"],
            "costs": test_record["costs"],
            "rejection_counts": test_record["rejection_counts"],
            "counts": test_record["counts"],
            "data_quality": test_record["data_quality"],
            "confirmatory": False,
            "selection_eligible": False,
            "previously_inspected": True,
        }
    return output


def render_summary(result: OrchestrationResult) -> str:
    """Render a requested-order descriptive summary without ranking strategies."""
    lines = [
        "# Research orchestration summary",
        "",
        f"- Orchestration ID: `{result.orchestration_id or 'dry-run'}`",
        f"- Status: **{result.status}**",
        f"- Requested strategy IDs: {', '.join(f'`{item}`' for item in result.strategy_ids)}",
        "- Results are descriptive research records; they do not establish profitability.",
        "- Previously inspected test results are non-confirmatory and selection-ineligible.",
        "",
        "| Strategy | Experiment ID | Disposition | Status | Train trades / net pips | Validation trades / net pips |",
        "|---|---|---|---|---:|---:|",
    ]
    for item in result.strategies:
        periods = item.get("periods", {})
        train = periods.get("train", {}).get("metrics", {})
        validation = periods.get("validation", {}).get("metrics", {})
        train_cell = f"{train.get('number_of_trades', 'n/a')} / {train.get('total_net_pips', 'n/a')}"
        validation_cell = (
            f"{validation.get('number_of_trades', 'n/a')} / "
            f"{validation.get('total_net_pips', 'n/a')}"
        )
        lines.append(
            f"| `{item['strategy_id']}` | `{item.get('experiment_id') or 'n/a'}` | "
            f"{item['disposition']} | {item['status']} | {train_cell} | {validation_cell} |"
        )
    for item in result.strategies:
        lines.extend(["", f"## {item['strategy_id']} details", ""])
        if item.get("error"):
            lines.append(f"- Error: {item['error']}")
        if item.get("identity_differences"):
            lines.append(
                "- A new experiment is required because stored request identity differs in: "
                + ", ".join(item["identity_differences"])
                + "."
            )
        for candidate in item.get("historical_candidates", []):
            line = (
                f"- Historical candidate `{candidate['experiment_id']}`: "
                f"status={candidate['status']}, identity={candidate['identity_status']}"
            )
            if candidate.get("identity_differences"):
                line += "; differing/unrecorded fields: " + ", ".join(candidate["identity_differences"])
            lines.append(line + ".")
        if item.get("data_quality"):
            lines.append(f"- Data quality: `{item['data_quality']}`")
        if item.get("execution_assumptions"):
            lines.append(f"- Execution assumptions: `{item['execution_assumptions']}`")
        if item.get("gap_policy"):
            lines.append(f"- Gap policy: `{item['gap_policy']}`")
        for period_name, period in item.get("periods", {}).items():
            label = period_name
            if period_name == "test":
                label = "test (previously inspected; non-confirmatory; selection-ineligible)"
            lines.extend([
                f"### {label}",
                "",
                f"- Metrics: `{period['metrics']}`",
                f"- Costs: `{period['costs']}`",
                f"- Rejection counts: `{period['rejection_counts']}`",
                f"- Counts: `{period['counts']}`",
                f"- Data quality: `{period['data_quality']}`",
                f"- Confirmatory: `{period['confirmatory']}`; selection eligible: `{period['selection_eligible']}`",
                "",
            ])
    return "\n".join(lines)


def run_research(
    strategy_ids: Sequence[str],
    *,
    dry_run: bool = False,
    retry_failed: bool = False,
    summary_path: str | Path | None = None,
) -> OrchestrationResult:
    """Run a bounded set of registered strategies through the existing runner.

    A dry run reads/fingerprints market data and previews exact store identities,
    but does not initialize or write to SQLite and does not generate signals.
    """
    requested = list(strategy_ids)
    specs = validate_selected_strategy_ids(requested)  # Fails before data/store work.
    context = prepare_comparison_context()
    db_store = ExperimentStore(DEFAULT_DB_PATH, initialize=not dry_run)
    previews = {
        spec.strategy_id: db_store.preview_run(
            **comparison_request_args(spec.strategy_id, context),
            retry_failed=retry_failed,
        )
        for spec in specs
    }
    if dry_run:
        outcomes = []
        for spec in specs:
            preview = previews[spec.strategy_id]
            outcomes.append({
                "strategy_id": spec.strategy_id,
                "name": spec.name,
                "disposition": preview["disposition"],
                "status": preview["status"],
                "experiment_id": preview["experiment_id"],
                "attempt": preview["attempt"],
                "error": None,
                "identity_differences": preview.get("identity_differences", []),
                "historical_candidates": preview.get("historical_candidates", []),
                "periods": {},
            })
        result = OrchestrationResult(None, "dry_run", requested, outcomes, dry_run=True)
        return result

    orchestration_id = db_store.start_orchestration(requested, {
        "retry_failed": retry_failed,
        "dry_run": False,
        "dataset_fingerprint": context["fingerprint"],
        "dataset_identity": context["identity"],
        "date_start_utc": context["date_start"],
        "date_end_utc": context["date_end"],
        "split_metadata": comparison_request_args(specs[0].strategy_id, context)["split_metadata"],
        "code_provenance": context["software"],
        "strategy_configurations": {
            spec.strategy_id: {
                "version": spec.version,
                "parameters": dict(spec.parameters),
                "execution_assumptions": comparison_request_args(spec.strategy_id, context)["execution_assumptions"],
                "gap_policy": comparison_request_args(spec.strategy_id, context)["gap_policy"],
            }
            for spec in specs
        },
    })
    outcomes_by_id: dict[str, dict[str, Any]] = {}
    interrupted = False
    for spec in specs:
        preview = previews[spec.strategy_id]
        disposition = preview["disposition"]
        handle_seen: dict[str, Any] = {}

        def register_handle(strategy_id: str, handle: Any) -> None:
            actual_disposition = _disposition(handle)
            handle_seen[strategy_id] = handle
            db_store.update_orchestration_strategy(
                orchestration_id, strategy_id, disposition=actual_disposition,
                status=handle.status, experiment_id=handle.experiment_id,
            )

        owner_id = str(uuid4())
        try:
            run_result = run_comparison(
                report_path=None,
                retry_failed=retry_failed,
                strategy_ids=[spec.strategy_id],
                store=db_store,
                context=context,
                execution_owner=owner_id,
                on_handle=register_handle,
                write_report=False,
                return_details=True,
            )
            assert isinstance(run_result, ComparisonRunResult)
            handle = run_result.handles[spec.strategy_id]
            outcome = _stored_strategy_summary(
                db_store, spec.strategy_id, spec.name, _disposition(handle),
                "completed", handle.experiment_id, attempt=handle.attempt,
                identity_differences=preview.get("identity_differences", []),
                historical_candidates=preview.get("historical_candidates", []),
            )
            db_store.update_orchestration_strategy(
                orchestration_id, spec.strategy_id,
                disposition=outcome["disposition"], status="completed",
                experiment_id=handle.experiment_id,
            )
        except KeyboardInterrupt:
            interrupted = True
            handle = handle_seen.get(spec.strategy_id)
            experiment_id = handle.experiment_id if handle is not None else None
            db_store.update_orchestration_strategy(
                orchestration_id, spec.strategy_id, disposition="interrupted",
                status="interrupted", experiment_id=experiment_id,
                error_message="Interrupted; an active experiment remains resumable after its lease expires.",
            )
            outcome = _stored_strategy_summary(
                db_store, spec.strategy_id, spec.name, "interrupted", "interrupted",
                experiment_id, attempt=handle.attempt if handle is not None else None,
                error="Interrupted; active execution remains resumable after its lease expires.",
                identity_differences=preview.get("identity_differences", []),
                historical_candidates=preview.get("historical_candidates", []),
            )
        except Exception as exc:
            handle = handle_seen.get(spec.strategy_id)
            if isinstance(exc, FailedRunRequiresRetry):
                failure_disposition = "retry_required"
                status = "retry_required"
            elif isinstance(exc, (ExecutionClaimBusy, ExecutionClaimLost)):
                failure_disposition = "busy"
                status = "busy"
            else:
                failure_disposition = "failed"
                status = "failed"
            experiment_id = (
                handle.experiment_id if handle is not None else preview.get("experiment_id")
            )
            attempt = handle.attempt if handle is not None else preview.get("attempt")
            db_store.update_orchestration_strategy(
                orchestration_id, spec.strategy_id, disposition=failure_disposition,
                status=status, experiment_id=experiment_id, error_message=str(exc),
            )
            outcome = _stored_strategy_summary(
                db_store, spec.strategy_id, spec.name, failure_disposition, status,
                experiment_id, attempt=attempt,
                error=str(exc), identity_differences=preview.get("identity_differences", []),
                historical_candidates=preview.get("historical_candidates", []),
            )
        outcomes_by_id[spec.strategy_id] = outcome
        if interrupted:
            break

    if interrupted:
        for spec in specs:
            if spec.strategy_id not in outcomes_by_id:
                db_store.update_orchestration_strategy(
                    orchestration_id, spec.strategy_id, disposition="interrupted",
                    status="not_started", error_message="Not started because the orchestration was interrupted.",
                )
                outcomes_by_id[spec.strategy_id] = {
                    "strategy_id": spec.strategy_id, "name": spec.name,
                    "disposition": "interrupted", "status": "not_started",
                    "experiment_id": None, "attempt": None,
                    "error": "Not started because the orchestration was interrupted.",
                    "identity_differences": previews[spec.strategy_id].get("identity_differences", []),
                    "historical_candidates": previews[spec.strategy_id].get("historical_candidates", []),
                    "periods": {},
                }
    outcomes = [outcomes_by_id[sid] for sid in requested]
    failure_count = sum(item["status"] in {"failed", "retry_required", "busy"} for item in outcomes)
    success_count = sum(item["status"] == "completed" for item in outcomes)
    status = "interrupted" if interrupted else (
        "completed" if not failure_count else "partial_failed" if success_count else "failed"
    )
    result = OrchestrationResult(orchestration_id, status, requested, outcomes)
    db_store.finish_orchestration(orchestration_id, status, summary=result.as_dict())
    if summary_path is not None:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_summary(result), encoding="utf-8")
    return result


def main() -> None:
    from .strategy_registry import list_strategies

    allowed = [spec.strategy_id for spec in list_strategies()]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", action="append", choices=allowed, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    result = run_research(
        args.strategy, dry_run=args.dry_run,
        retry_failed=args.retry_failed, summary_path=args.summary,
    )
    print(render_summary(result))
    if result.status == "interrupted":
        raise SystemExit(130)


if __name__ == "__main__":
    main()
