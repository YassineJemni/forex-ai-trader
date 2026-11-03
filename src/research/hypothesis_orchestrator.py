"""Persistent, bounded orchestration for explicitly registered hypotheses.

This module is intentionally separate from the baseline-strategy orchestrator:
it dispatches only through the registered hypothesis runner and never selects by results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import pandas as pd

from .data import load_market_data, validate_m1_cadence
from .experiment_ema import split_metadata, timeline_periods
from .experiment_store import (
    DEFAULT_DB_PATH, ExecutionClaimBusy, ExperimentStore, FailedRunRequiresRetry,
    canonical_json, code_provenance, fingerprint_market_frame, utc_date_range,
)
from .execution_models import (
    BidAskExecutionModel,
    MarketDataMetadata,
    prepare_bid_ask_inputs,
)
from .hypothesis_runner import run_registered_hypothesis
from .hypothesis_schema import StrategyHypothesis
from .hypothesis_signals import prepare_hypothesis_market_data
from .hypothesis_validator import validate_hypothesis

MAX_HYPOTHESES_PER_BATCH = 10


@dataclass
class HypothesisOrchestrationResult:
    orchestration_id: str | None
    status: str
    hypotheses: list[dict[str, Any]]
    lifecycle_counts: dict[str, int]


M1_QUALITY_POLICY = {
    "policy_id": "hypothesis_market_data_v1",
    "timestamps": "UTC-aware, sorted, unique; invalid timestamps dropped under compiler policy",
    "prices": "finite positive OHLC with consistent candle bounds; invalid prices rejected",
    "duplicates": "sort by UTC time and keep last input row for duplicate timestamps",
    "cadence": "one-minute intervals or integer-minute gaps strictly greater than 90 seconds",
    "gap_indicators": "permitted gaps are retained and segmented by existing >90s policy",
    "volume": "retained and fingerprinted; unused by current hypothesis feature allowlist",
}


def prepare_hypothesis_context(market_data: pd.DataFrame | None = None) -> dict[str, Any]:
    """Prepare and fingerprint the exact canonical frame the runner receives."""
    source = load_market_data() if market_data is None else market_data
    market, preparation = prepare_hypothesis_market_data(source)
    cadence = validate_m1_cadence(market["time"])
    fingerprint, identity = fingerprint_market_frame(market, quality_policy=M1_QUALITY_POLICY)
    identity["rows_used"] = int(len(market))
    date_start, date_end = utc_date_range(market)
    return {
        "market": market,
        "fingerprint": fingerprint,
        "identity": identity,
        "date_start": date_start,
        "date_end": date_end,
        "quality": {
            "source_load_report": dict(source.attrs.get("load_report", {})),
            "normalization": preparation,
            "cadence": cadence,
            "quality_policy": M1_QUALITY_POLICY,
        },
        "software": code_provenance(),
    }


def _reference(value: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, str) and "@" in value:
        return tuple(value.rsplit("@", 1))  # type: ignore[return-value]
    raise ValueError("hypothesis selection must be (hypothesis_id, version) or 'id@version'")


def _execution_identity(
    hypothesis: StrategyHypothesis,
    execution_model: BidAskExecutionModel | None = None,
    bid_ask_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    execution = hypothesis.execution_assumptions
    assumptions = {
        "experiment_kind": "registered_hypothesis",
        "timestamp_convention": execution.timestamp_convention,
        "bar_duration": execution.bar_duration,
        "holding_period": hypothesis.exit_rule.max_holding_period,
        "execution_delay": execution.execution_delay,
        "allow_overlapping": execution.allow_overlapping,
        "transaction_cost_pips_round_trip": execution.transaction_cost_pips,
        "pip_size": execution.pip_size,
    }
    if execution_model is not None:
        if execution.transaction_cost_pips != 0.0:
            raise ValueError(
                "BID/ASK hypotheses must set transaction_cost_pips to zero; "
                "spread is embedded in side-specific prices"
            )
        if bid_ask_identity is None:
            raise ValueError("BID/ASK execution identity requires both side fingerprints and metadata")
        assumptions["execution_model"] = execution_model.identity_payload(**bid_ask_identity)
        assumptions["fixed_transaction_cost_pips_round_trip"] = 0.0
    gap = {"tolerance": execution.gap_tolerance,
           "indicator_reset_above": execution.gap_tolerance,
           "exact_boundary_resets": False}
    return assumptions, gap


def _period_record(period: dict[str, Any]) -> dict[str, Any]:
    attribution = period["attribution"]
    counts = {
        "candidate_events": period["candidate_events"],
        "warmup_excluded": period["warmup_excluded"],
        "backtester_inputs": period["backtester_inputs"],
        "accepted_trades": period["accepted_trades"],
        "backtester_rejected": period["backtester_rejected"],
        "accepted_attributed": attribution["accepted_verified"],
        "accepted_attribution_unavailable": attribution["accepted_unavailable"],
        "rejected_attributed": attribution["rejected_verified"],
        "rejected_attribution_unavailable": attribution["rejected_unavailable"],
        "candidate_attribution_verified": attribution["candidates_verified"],
        "candidate_attribution_unavailable": attribution["candidates_unavailable"],
    }
    return {
        "summary": period["summary"],
        "rejected": period["rejected"],
        "trades": period["trades"],
        "rejected_signals": period["rejected_signals"],
        "data_quality": period["data_quality"],
        "counts": counts,
    }


def _stored_summary(store: ExperimentStore, experiment_id: str) -> dict[str, Any]:
    run = store.get_run(experiment_id)
    periods = {name: store.get_period_result(experiment_id, name)
               for name in ("train", "validation")}
    test = {
        "status": "not_evaluated", "previously_inspected": True,
        "confirmatory": False, "selection_eligible": False, "metrics": None,
    }
    return {"experiment_id": experiment_id, "run_status": run["status"],
            "periods": periods, "test": test,
            "split_metadata": run["split_metadata"],
            "dataset_identity": run["dataset_identity"],
            "execution_assumptions": run["execution_assumptions"],
            "data_quality": run["data_quality"]}


def _lifecycle_counts(rows: list[dict[str, Any]], *, registered_count: int) -> dict[str, int]:
    statuses = [row.get("status") for row in rows]
    dispositions = [row.get("disposition") for row in rows]
    return {
        "selected": len(rows),
        "registered": registered_count,
        "executed": sum(bool(row.get("execution_started")) for row in rows),
        "completed": sum(status == "completed" for status in statuses),
        "reused": sum(disposition == "reused_completed" for disposition in dispositions),
        "failed": sum(status == "failed" for status in statuses),
        "skipped": sum(status in {"retry_required", "busy", "skipped", "not_started"}
                       for status in statuses),
    }


def run_hypothesis_research(
    selections: Sequence[str | tuple[str, str]], *, dry_run: bool = False,
    retry_failed: bool = False, summary_path: str | Path | None = None,
    market_data: pd.DataFrame | None = None,
    execution_model: BidAskExecutionModel | None = None,
    ask_market_data: pd.DataFrame | None = None,
    bid_metadata: MarketDataMetadata | None = None,
    ask_metadata: MarketDataMetadata | None = None,
    store_path: str | Path = DEFAULT_DB_PATH,
) -> HypothesisOrchestrationResult:
    """Run explicitly registered hypotheses in deterministic ID/version order.

    Dry runs use SQLite read-only connections and do not compile signals. Test
    periods are never passed to the runner; stored test results are not selected
    or summarized as metrics.
    """
    refs = [_reference(item) for item in selections]
    if not refs:
        raise ValueError("at least one registered hypothesis is required")
    if len(set(refs)) != len(refs):
        raise ValueError("duplicate hypothesis selections are not allowed")
    if len(refs) > MAX_HYPOTHESES_PER_BATCH:
        raise ValueError(f"batch exceeds maximum of {MAX_HYPOTHESES_PER_BATCH} hypotheses")
    refs.sort()
    if execution_model is None:
        if ask_market_data is not None or bid_metadata is not None or ask_metadata is not None:
            raise ValueError("BID/ASK inputs require an explicit side-aware execution_model")
    elif not isinstance(execution_model, BidAskExecutionModel):
        raise TypeError("execution_model must be a BidAskExecutionModel")

    store = ExperimentStore(store_path, initialize=not dry_run)
    registered: list[tuple[tuple[str, str], StrategyHypothesis, dict[str, Any]]] = []
    for ref in refs:
        record = store.get_registered_hypothesis(*ref)
        if record["validation_status"] != "valid" or record["registration_status"] != "registered":
            raise ValueError(f"hypothesis {ref[0]}@{ref[1]} is not valid and registered")
        hypothesis = StrategyHypothesis.from_dict(record["hypothesis"])
        validation = validate_hypothesis(hypothesis)
        if validation.semantic_digest != record["semantic_digest"]:
            raise ValueError(f"registered hypothesis {ref[0]}@{ref[1]} failed digest verification")
        if pd.Timedelta(hypothesis.execution_assumptions.bar_duration) != pd.Timedelta("1min"):
            raise ValueError("the current EUR/USD research dataset is M1; bar_duration must be 1min")
        if execution_model is not None and hypothesis.execution_assumptions.transaction_cost_pips != 0.0:
            raise ValueError(
                "BID/ASK hypotheses must set transaction_cost_pips to zero; "
                "spread is embedded in side-specific prices"
            )
        registered.append((ref, hypothesis, record))

    context = prepare_hypothesis_context(market_data)
    data = context["market"]
    normalized_ask = None
    bid_ask_identity = None
    if execution_model is not None:
        if ask_market_data is None or bid_metadata is None or ask_metadata is None:
            raise ValueError("BID/ASK execution requires ASK data and metadata for both sides")
        conventions = {hypothesis.execution_assumptions.timestamp_convention
                       for _, hypothesis, _ in registered}
        if len(conventions) != 1:
            raise ValueError("one BID/ASK orchestration request cannot mix timestamp conventions")
        timestamp_convention = next(iter(conventions))
        data, normalized_ask, alignment_report = prepare_bid_ask_inputs(
            data, ask_market_data, bid_metadata=bid_metadata, ask_metadata=ask_metadata,
            timestamp_convention=timestamp_convention, bar_duration="1min",
        )
        ask_fingerprint, _ = fingerprint_market_frame(
            normalized_ask,
            quality_policy={**M1_QUALITY_POLICY, "market_side": "ASK",
                            "alignment_policy": execution_model.alignment_policy},
        )
        bid_ask_identity = {
            "bid_metadata": bid_metadata,
            "ask_metadata": ask_metadata,
            "bid_fingerprint": context["fingerprint"],
            "ask_fingerprint": ask_fingerprint,
            "alignment_report": alignment_report,
        }
        context["quality"]["bid_ask_alignment"] = alignment_report
    outcomes: list[dict[str, Any]] = []
    for ref, hypothesis, record in registered:
        assumptions, gap_policy = _execution_identity(
            hypothesis, execution_model=execution_model,
            bid_ask_identity=bid_ask_identity,
        )
        periods = timeline_periods(data, bar_duration=hypothesis.execution_assumptions.bar_duration)
        start, end = utc_date_range(data)
        request = {
            "hypothesis": hypothesis,
            "dataset_fingerprint": context["fingerprint"],
            "dataset_identity": context["identity"],
            "date_start_utc": start,
            "date_end_utc": end,
            "execution_assumptions": assumptions,
            "gap_policy": gap_policy,
            "software": context["software"],
            "data_quality": context["quality"],
            "split_metadata": split_metadata(periods),
        }
        preview = store.preview_hypothesis_run(**request, retry_failed=retry_failed)
        outcomes.append({
            "hypothesis_id": ref[0], "version": ref[1],
            "semantic_digest": record["semantic_digest"],
            "disposition": preview["disposition"], "experiment_id": preview["experiment_id"],
            "status": preview["status"],
            "identity_differences": preview["identity_differences"],
            "historical_candidates": preview["historical_candidates"],
            "request": request,
        })

    if dry_run:
        for row in outcomes:
            row.pop("request", None)
        counts = _lifecycle_counts(outcomes, registered_count=len(registered))
        return HypothesisOrchestrationResult(None, "dry_run", outcomes, counts)

    orchestration_id = store.start_orchestration(
        [f"hypothesis:{row['hypothesis_id']}@{row['version']}" for row in outcomes],
        {"kind": "registered_hypothesis_batch", "selected": [f"{a}@{b}" for a, b in refs],
         "retry_failed": retry_failed, "batch_limit": MAX_HYPOTHESES_PER_BATCH},
    )
    completed = 0
    interrupted = False
    owner_id = f"hypothesis-orchestrator-{uuid4()}"
    processed: set[str] = set()
    for row, (ref, hypothesis, _) in zip(outcomes, registered):
        child_id = f"hypothesis:{row['hypothesis_id']}@{row['version']}"
        processed.add(child_id)
        try:
            handle = store.create_or_resume_hypothesis_run(
                **row["request"], retry_failed=retry_failed,
            )
            row["experiment_id"] = handle.experiment_id
            row["disposition"] = "reused_completed" if handle.reused_completed else (
                "resumed" if handle.resumed else "new")
            if handle.reused_completed:
                row.update(_stored_summary(store, handle.experiment_id))
                row["status"] = "completed"
                store.update_orchestration_strategy(
                    orchestration_id, child_id, disposition="reused_completed",
                    status="completed", experiment_id=handle.experiment_id,
                )
                completed += 1
                continue
            try:
                with store.maintain_execution_claim(handle.experiment_id, owner_id):
                    row["execution_started"] = True
                    store.update_orchestration_strategy(
                        orchestration_id, child_id,
                        disposition="resumed" if handle.resumed else "new",
                        status="running", experiment_id=handle.experiment_id,
                    )
                    runner_kwargs = {"store": store, "market_data": data}
                    # Keep the established legacy invocation unchanged. Side-aware
                    # arguments are passed only when that separately identified
                    # execution model was explicitly requested.
                    if execution_model is not None:
                        runner_kwargs.update(
                            execution_model=execution_model,
                            ask_market_data=normalized_ask,
                            bid_metadata=bid_metadata,
                            ask_metadata=ask_metadata,
                        )
                    evaluated = run_registered_hypothesis(ref[0], ref[1], **runner_kwargs)
                    if evaluated["semantic_digest"] != row["semantic_digest"]:
                        raise ValueError("runner semantic digest differs from registered hypothesis")
                    persisted = {name: _period_record(evaluated["periods"][name])
                                 for name in ("train", "validation")}
                    store.complete_run(handle.experiment_id, persisted, claim_owner=owner_id)
                row.update(_stored_summary(store, handle.experiment_id))
                row["status"] = "completed"
                store.update_orchestration_strategy(
                    orchestration_id, child_id,
                    disposition="resumed" if handle.resumed else "new",
                    status="completed", experiment_id=handle.experiment_id,
                )
                completed += 1
            except KeyboardInterrupt:
                interrupted = True
                row["status"] = "interrupted"
                store.release_execution_claim(handle.experiment_id, owner_id)
                store.update_orchestration_strategy(
                    orchestration_id, child_id, disposition="interrupted", status="interrupted",
                    experiment_id=handle.experiment_id, error_message="interrupted; run remains resumable",
                )
                raise
            except ExecutionClaimBusy as exc:
                row.update({"disposition": "busy", "status": "busy", "error": str(exc)})
                store.update_orchestration_strategy(
                    orchestration_id, child_id, disposition="busy", status="busy",
                    experiment_id=handle.experiment_id, error_message=str(exc),
                )
            except Exception as exc:
                # Only this invocation's claimed execution is failed. A run that
                # failed during create/resume or claim remains untouched.
                try:
                    store.record_failure(handle.experiment_id, "execution_error", str(exc),
                                         claim_owner=owner_id)
                except Exception:
                    pass
                store.update_orchestration_strategy(
                    orchestration_id, child_id, disposition="failed", status="failed",
                    experiment_id=handle.experiment_id, error_message=str(exc),
                )
                row.update({"run_status": "failed", "status": "failed", "error": str(exc)})
        except KeyboardInterrupt:
            interrupted = True
            break
        except FailedRunRequiresRetry as exc:
            row.update({"disposition": "retry_required", "status": "retry_required", "error": str(exc)})
            store.update_orchestration_strategy(
                orchestration_id, child_id, disposition="retry_required", status="retry_required",
                error_message=str(exc),
            )
        except ExecutionClaimBusy as exc:
            row.update({"disposition": "busy", "status": "busy", "error": str(exc)})
            store.update_orchestration_strategy(
                orchestration_id, child_id, disposition="busy", status="busy",
                experiment_id=row.get("experiment_id"), error_message=str(exc),
            )
        except Exception as exc:
            row.update({"disposition": "failed", "status": "failed", "error": str(exc)})
            store.update_orchestration_strategy(
                orchestration_id, child_id, disposition="failed", status="failed",
                experiment_id=row.get("experiment_id"), error_message=str(exc),
            )
    for row in outcomes:
        row.pop("request", None)
    if interrupted:
        for row in outcomes:
            child_id = f"hypothesis:{row['hypothesis_id']}@{row['version']}"
            if child_id not in processed:
                row.update({"disposition": "skipped", "status": "skipped",
                            "error": "not started because this orchestration was interrupted"})
                store.update_orchestration_strategy(
                    orchestration_id, child_id, disposition="interrupted", status="not_started",
                    error_message=row["error"],
                )
    counts = _lifecycle_counts(outcomes, registered_count=len(registered))
    status = "interrupted" if interrupted else (
        "completed" if completed == len(outcomes) else "partial_failed" if completed else "failed")
    store.finish_orchestration(orchestration_id, status,
                               summary={"lifecycle_counts": counts, "hypotheses": outcomes})
    result = HypothesisOrchestrationResult(orchestration_id, status, outcomes, counts)
    if summary_path is not None:
        _write_summary(Path(summary_path), result)
    return result


def _write_summary(path: Path, result: HypothesisOrchestrationResult) -> None:
    lines = ["# Registered hypothesis orchestration", "",
             "Descriptive research records only; no profitability claim is made.",
             "Previously inspected test results are not evaluated or selection-eligible.",
             f"Lifecycle counts: `{canonical_json(result.lifecycle_counts)}`.",
             "Hypothesis generation was not performed here; generation, rejection, and deduplication counts are unavailable.", ""]
    for item in result.hypotheses:
        lines.extend([f"## {item['hypothesis_id']} v{item['version']}", "",
                      f"- Experiment: `{item.get('experiment_id')}`",
                      f"- Status/disposition: {item.get('run_status', item.get('status'))} / {item['disposition']}"])
        split = item.get("split_metadata", {}).get("periods", {})
        for name in ("train", "validation"):
            if name in split:
                lines.append(f"- {name} boundary (UTC): `{split[name]}`")
        for name in ("train", "validation"):
            period = item.get("periods", {}).get(name)
            if period:
                lines.extend([
                    f"- {name} metrics: `{canonical_json(period['metrics'])}`",
                    f"- {name} costs: `{canonical_json(period['costs'])}`",
                    f"- {name} rejections: `{canonical_json(period['rejection_counts'])}`",
                    f"- {name} counts: `{canonical_json(period['counts'])}`",
                    f"- {name} data quality: `{canonical_json(period['data_quality'])}`",
                ])
        lines.append("- Test: not evaluated; previously inspected, non-confirmatory, selection-ineligible.")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run explicitly registered hypotheses")
    parser.add_argument("hypotheses", nargs="+", help="registered hypothesis ID@version")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    result = run_hypothesis_research(args.hypotheses, dry_run=args.dry_run,
                                     retry_failed=args.retry_failed, summary_path=args.summary)
    print(canonical_json({"orchestration_id": result.orchestration_id,
                          "status": result.status, "hypotheses": result.hypotheses}))
    return 0 if result.status in {"completed", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
