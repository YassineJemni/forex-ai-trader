"""Bounded, deterministic research cycles around the registered-run pipeline.

The cycle chooses proposals from an explicit template configuration before any
metrics exist. It never mutates a registered hypothesis and never reads test
period metrics. The existing registered-hypothesis orchestrator remains the
only execution path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import numpy as np

from .data import load_market_data
from .experiment_store import (
    DEFAULT_DB_PATH, ExperimentStore, canonical_json, code_provenance, fingerprint_market_frame,
    utc_date_range,
)
from .execution_models import BidAskExecutionModel, MarketDataMetadata, prepare_bid_ask_inputs
from .experiment_ema import split_metadata, timeline_periods
from .hypothesis_orchestrator import (
    M1_QUALITY_POLICY, prepare_hypothesis_context, run_hypothesis_research,
)
from .hypothesis_schema import ExecutionAssumptions
from .hypothesis_templates import TemplateConfig, generate_hypotheses
from .hypothesis_validator import validate_hypothesis
from .hypothesis_runner import run_registered_hypothesis_windows
from .research_evidence import (
    ResearchEvidenceStore, build_walk_forward_folds, training_screen,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASK_DATA_DIR = PROJECT_ROOT / "market-data" / "m1" / "Forex" / "Majors" / "EUR-USD" / "ASK"
MAX_CYCLE_CANDIDATES = 5
MAX_CYCLE_EXPERIMENTS = 10


@dataclass(frozen=True)
class CostScenario:
    """Explicit round-trip cost assumptions, not calibrated execution costs."""

    name: str
    commission_pips: float
    slippage_pips: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("cost scenario name must be nonempty")
        BidAskExecutionModel(self.commission_pips, self.slippage_pips)


DEFAULT_COST_SCENARIOS = (
    CostScenario("provisional_base", 0.0, 0.5),
    CostScenario("commission_sensitivity", 0.5, 0.5),
    CostScenario("higher_slippage_sensitivity", 0.5, 1.0),
)

BID_METADATA = MarketDataMetadata("BID", "EUR/USD", "M1", "bar_open", "UTC", "Dukascopy")
ASK_METADATA = MarketDataMetadata("ASK", "EUR/USD", "M1", "bar_open", "UTC", "Dukascopy")


def load_ask_m1(data_dir: str | Path = ASK_DATA_DIR) -> pd.DataFrame:
    """Read local monthly ASK parquet partitions without filling or shifting."""
    path = Path(data_dir)
    files = sorted(path.glob("????-??.parquet"))
    if not files:
        raise FileNotFoundError(f"no monthly ASK parquet files found under {path}")
    frames = [pd.read_parquet(file, columns=["time", "open", "high", "low", "close", "volume"])
              for file in files]
    result = pd.concat(frames, ignore_index=True)
    result["time"] = pd.to_datetime(result["time"], utc=True, errors="raise")
    result = result.sort_values("time", kind="stable").reset_index(drop=True)
    if result["time"].duplicated().any():
        raise ValueError("ASK monthly files contain duplicate timestamps")
    return result


def select_predeclared_candidates(
    proposals: Sequence[Any], *, budget: int,
) -> tuple[list[Any], list[dict[str, str]]]:
    """Select in generator order, independent of all market and result metrics."""
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= MAX_CYCLE_CANDIDATES:
        raise ValueError(f"candidate budget must be between 1 and {MAX_CYCLE_CANDIDATES}")
    selected: list[Any] = []
    decisions: list[dict[str, str]] = []
    for proposal in proposals:
        if len(selected) < budget:
            selected.append(proposal)
            decisions.append({"hypothesis_id": proposal.hypothesis_id,
                              "decision": "selected_within_predeclared_budget",
                              "reason": "deterministic template order; no metrics consulted"})
        else:
            decisions.append({"hypothesis_id": proposal.hypothesis_id,
                              "decision": "skipped_budget",
                              "reason": "candidate budget exhausted before evaluation"})
    return selected, decisions


def analyze_period_results(periods: dict[str, dict[str, Any]], *, minimum_trades: int = 100) -> list[dict[str, Any]]:
    """Return evidence-limited observations from train/validation only."""
    if minimum_trades < 1:
        raise ValueError("minimum_trades must be positive")
    observations: list[dict[str, Any]] = []
    for period_name in ("train", "validation"):
        period = periods.get(period_name)
        if not period or period.get("status") == "not_evaluated":
            continue
        metrics, costs = period.get("metrics", {}), period.get("costs", {})
        trades = int(metrics.get("number_of_trades", 0))
        notes: list[str] = []
        mid_gross = costs.get("mid_price_gross_total_pips")
        net = metrics.get("total_net_pips")
        if mid_gross is not None and mid_gross < 0:
            notes.append("mid-price gross aggregate is negative before side spread and explicit assumed costs")
        if mid_gross is not None and net is not None and mid_gross >= 0 > net:
            notes.append("net aggregate is negative after embedded spread and explicit assumed costs")
        if trades < minimum_trades:
            notes.append(f"trade count below the descriptive minimum ({minimum_trades}); uncertainty is high")
        observations.append({"period": period_name, "trade_count": trades, "notes": notes,
                             "classification": "descriptive_only"})
    if len(observations) == 2:
        train = periods.get("train", {}).get("metrics", {}).get("total_net_pips")
        validation = periods.get("validation", {}).get("metrics", {}).get("total_net_pips")
        if train is not None and validation is not None and (train < 0) != (validation < 0):
            observations.append({"period": "train_vs_validation", "trade_count": None,
                                 "notes": ["net-P&L signs differ across development periods; this is instability evidence, not a selection test"],
                                 "classification": "descriptive_only"})
    return observations


def recommend_next_steps(observations: Sequence[dict[str, Any]]) -> list[str]:
    """Map supported diagnostics to process recommendations, never parameter edits."""
    notes = [note for item in observations for note in item.get("notes", [])]
    recommendations: list[str] = []
    if any("mid-price gross aggregate is negative" in note for note in notes):
        recommendations.append(
            "Do not expand this exact rule on the same data; any new mechanism should be predeclared and evaluated on future observations."
        )
    if any("after embedded spread and explicit assumed costs" in note for note in notes):
        recommendations.append(
            "Obtain account-specific commission and execution records before treating the cost sensitivity as calibrated."
        )
    if any("trade count below" in note for note in notes):
        recommendations.append(
            "Prioritize additional untouched observations and uncertainty estimation rather than threshold tuning."
        )
    if any("signs differ" in note for note in notes):
        recommendations.append(
            "Treat train/validation disagreement as instability; defer selection until an untouched forward period is available."
        )
    if not recommendations:
        recommendations.append(
            "Keep the hypothesis exploratory and require a preregistered, purged walk-forward protocol before any selection decision."
        )
    return recommendations


def _development_history(store: ExperimentStore) -> list[dict[str, Any]]:
    """Read only prior train/validation summaries; never query test metrics."""
    import json

    with store._readonly_connection() as db:
        rows = db.execute("""
            SELECT e.experiment_id, e.hypothesis_id, e.hypothesis_version,
                   e.hypothesis_digest, e.status, p.period, p.metrics_json,
                   p.costs_json, p.counts_json, p.rejection_counts_json
            FROM experiment_runs AS e
            LEFT JOIN period_results AS p ON p.experiment_id=e.experiment_id
                AND p.period IN ('train', 'validation')
            WHERE e.hypothesis_id IS NOT NULL
            ORDER BY e.started_at_utc, p.period
        """).fetchall()
    history: dict[str, dict[str, Any]] = {}
    for row in rows:
        record = history.setdefault(row["experiment_id"], {
            "experiment_id": row["experiment_id"], "hypothesis_id": row["hypothesis_id"],
            "version": row["hypothesis_version"], "semantic_digest": row["hypothesis_digest"],
            "status": row["status"], "periods": {},
            "history_status": "included for audit after candidate selection; not used for proposal generation or selection",
        })
        if row["period"] in {"train", "validation"}:
            record["periods"][row["period"]] = {
                "metrics": json.loads(row["metrics_json"]),
                "costs": json.loads(row["costs_json"]),
                "counts": json.loads(row["counts_json"]),
                "rejection_counts": json.loads(row["rejection_counts_json"]),
            }
    return list(history.values())


def _proposal_config() -> TemplateConfig:
    # One predeclared, interpretable addition: EMA(9/21) cross confirmed by
    # volatility(15) below 0.0005. Existing standalone proposal is included in
    # generation for duplicate accounting, but is not evaluated by this cycle.
    return TemplateConfig(
        families=("trend_following",), ema_pairs=((9, 21),),
        include_trend_volatility_filters=True,
        trend_volatility_periods=(15,), trend_volatility_thresholds=(0.0005,),
        trend_volatility_modes=("below",), candidate_limit=2,
        execution_assumptions=ExecutionAssumptions(transaction_cost_pips=0.0),
    )


def _evidence_proposal_config() -> TemplateConfig:
    """One train-evidence-motivated interaction, fixed before this cycle runs."""
    return TemplateConfig(
        families=("mean_reversion",), rsi_threshold_pairs=((30.0, 70.0),),
        include_mean_reversion_volatility_filters=True,
        mean_reversion_volatility_periods=(15,),
        mean_reversion_volatility_thresholds=(0.0005,),
        mean_reversion_volatility_modes=("below",), candidate_limit=2,
        execution_assumptions=ExecutionAssumptions(transaction_cost_pips=0.0),
    )


def _training_only_cost_drag_evidence(store: ExperimentStore) -> list[dict[str, Any]]:
    """Read only prior mean-reversion TRAIN rows for proposal rationale."""
    import json

    if not store.path.is_file():
        return []
    with store._readonly_connection() as db:
        rows = db.execute("""
            SELECT e.experiment_id, e.hypothesis_id, e.hypothesis_version,
                   e.hypothesis_digest, p.metrics_json, p.costs_json, p.counts_json
            FROM experiment_runs e
            JOIN registered_hypotheses h
              ON h.hypothesis_id=e.hypothesis_id
             AND h.hypothesis_version=e.hypothesis_version
            JOIN period_results p ON p.experiment_id=e.experiment_id
            WHERE h.strategy_family='mean_reversion' AND p.period='train'
            ORDER BY e.started_at_utc, e.experiment_id
        """).fetchall()
    evidence = []
    for row in rows:
        metrics, costs, counts = (json.loads(row[key]) for key in
                                  ("metrics_json", "costs_json", "counts_json"))
        evidence.append({
            "experiment_id": row["experiment_id"], "hypothesis_id": row["hypothesis_id"],
            "version": row["hypothesis_version"], "semantic_digest": row["hypothesis_digest"],
            "trade_count": metrics.get("number_of_trades"),
            "mid_price_gross_total_pips": costs.get("mid_price_gross_total_pips"),
            "net_total_pips": metrics.get("total_net_pips"),
            "assumed_cost_total_pips": costs.get("total_assumed_cost_pips"),
            "used_period": "train",
        })
    return evidence


def _training_scorecard(fold_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Descriptive fold-aware scorecard; it does not rank candidates."""
    rows: dict[str, Any] = {}
    for scenario, folds in fold_results.items():
        nonempty = [fold["trades"] for fold in folds
                    if isinstance(fold.get("trades"), pd.DataFrame) and not fold["trades"].empty]
        trades = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(
            columns=["net_pnl_pips"])
        net = trades["net_pnl_pips"].to_numpy(dtype=float) if not trades.empty else np.array([])
        gains = float(net[net > 0].sum()) if len(net) else 0.0
        losses = float(-net[net < 0].sum()) if len(net) else 0.0
        fold_totals = [float(fold["summary"].get("total_net_pips", 0.0)) for fold in folds]
        costs = [fold.get("costs", {}) for fold in folds]
        mid_gross = [item.get("mid_price_gross_total_pips") for item in costs]
        notes = []
        if mid_gross and all(value is not None for value in mid_gross) and sum(mid_gross) < 0:
            notes.append("negative_mid_price_gross")
        elif (mid_gross and all(value is not None for value in mid_gross)
              and sum(mid_gross) >= 0 and sum(fold_totals) < 0):
            notes.append("net_negative_after_embedded_spread_and_assumed_costs")
        if len(net) < 100:
            notes.append("insufficient_trade_observations_for_predeclared_minimum")
        if fold_totals and any(value <= 0 for value in fold_totals):
            notes.append("nonpositive_training_fold")
        rows[scenario] = {
            "trade_count": int(len(net)),
            "expectancy_net_pips_per_trade": float(net.mean()) if len(net) else None,
            "total_net_pips": float(net.sum()) if len(net) else 0.0,
            "profit_factor": gains / losses if losses > 0 else None,
            "max_fold_drawdown_pips": max(
                (float(fold["summary"].get("max_drawdown_pips", 0.0)) for fold in folds),
                default=0.0,
            ),
            "fold_net_pips": fold_totals,
            "fold_sign_consistent_positive": bool(fold_totals and all(value > 0 for value in fold_totals)),
            "mid_price_gross_total_pips": sum(mid_gross) if mid_gross and all(
                value is not None for value in mid_gross) else None,
            "assumed_cost_total_pips": sum(float(cost.get("total_assumed_cost_pips", 0.0))
                                           for cost in costs),
            "failure_attribution": notes,
        }
    return {"scenarios": rows, "automatic_ranking": False}


def _side_aware_cost_components(trades: pd.DataFrame) -> dict[str, float]:
    """Summarize persisted side-aware fills without deducting spread twice."""
    def total(column: str) -> float:
        return float(trades[column].sum()) if column in trades else 0.0

    gross = total("gross_pnl_pips")
    explicit_cost = total("transaction_costs_pips")
    net = total("net_pnl_pips")
    if not np.isclose(gross - explicit_cost, net, rtol=0.0, atol=1e-7):
        raise AssertionError("trade gross less explicit costs does not reconcile to net P&L")
    mid = total("mid_price_pnl_pips")
    spread = total("spread_cost_pips")
    if not np.isclose(mid - spread, gross, rtol=0.0, atol=1e-7):
        raise AssertionError("side-aware spread decomposition does not reconcile to fill gross")
    return {
        "gross_total_pips": gross,
        "mid_price_gross_total_pips": mid,
        "spread_total_pips_embedded_in_fill_prices": spread,
        "commission_assumption_total_pips": total("commission_costs_pips"),
        "slippage_assumption_total_pips": total("slippage_costs_pips"),
        "total_assumed_cost_pips": explicit_cost,
        "net_total_pips": net,
        "spread_deducted_again": 0.0,
    }


def run_evidence_driven_cycle(
    *, store_path: str | Path = DEFAULT_DB_PATH,
    market_data: pd.DataFrame | None = None,
    ask_market_data: pd.DataFrame | None = None,
    cost_scenarios: Sequence[CostScenario] = DEFAULT_COST_SCENARIOS,
    report_path: str | Path | None = None,
    bid_metadata: MarketDataMetadata = BID_METADATA,
    ask_metadata: MarketDataMetadata = ASK_METADATA,
    n_folds: int = 3,
    bootstrap_resamples: int = 1000,
) -> dict[str, Any]:
    """Run a bounded train-screen then (only if passed) one validation gate.

    Generation is a fixed declarative response to *training-only* cost-drag
    evidence from prior RSI experiments. The final 20% is timestamp-partitioned
    only and never feature-built, queried for metrics, or sent to an evaluator.
    """
    if not cost_scenarios or len(cost_scenarios) != 3 or any(
        not isinstance(value, CostScenario) for value in cost_scenarios
    ):
        raise ValueError("the predeclared pilot requires exactly three CostScenario values")
    if n_folds != 3:
        raise ValueError("the first evidence cycle fixes exactly three walk-forward folds")
    if isinstance(bootstrap_resamples, bool) or bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")

    experiment_store = ExperimentStore(store_path)
    prior_train_evidence = _training_only_cost_drag_evidence(experiment_store)
    qualifying_evidence = [row for row in prior_train_evidence
                          if row["mid_price_gross_total_pips"] is not None
                          and row["mid_price_gross_total_pips"] > 0
                          and row["net_total_pips"] is not None
                          and row["net_total_pips"] < 0]
    if not qualifying_evidence:
        raise RuntimeError(
            "No prior training-only RSI record documents positive mid-price gross and "
            "negative net after execution costs; refusing to claim an evidence-driven proposal."
        )

    # The fixed follow-up is RSI(14) recrossing conditioned on low vol(15),
    # motivated by training-only evidence that costs erase positive mid-price gross.
    generation = generate_hypotheses(_evidence_proposal_config())
    registry = experiment_store.list_registered_hypotheses()
    registered_digests: dict[str, list[str]] = {}
    for item in registry:
        registered_digests.setdefault(item["semantic_digest"], []).append(
            f"{item['hypothesis_id']}@{item['hypothesis_version']}"
        )
    composites = [proposal for proposal in generation.hypotheses
                  if len(proposal.indicators) == 2]
    if len(composites) != 1:
        raise AssertionError("fixed template configuration must produce one RSI/volatility proposal")
    proposal = composites[0]
    validation = validate_hypothesis(proposal)
    if validation.semantic_digest != proposal.semantic_digest():
        raise ValueError("proposal digest differs from formal validator")
    prior_matches = registered_digests.get(validation.semantic_digest, [])
    exact_prior: dict[str, Any] | None = None
    try:
        exact_prior = experiment_store.get_registered_hypothesis(
            proposal.hypothesis_id, proposal.version)
    except KeyError:
        pass
    if prior_matches and exact_prior is None:
        raise RuntimeError(
            "The candidate behavior is already registered as " + ", ".join(prior_matches)
            + "; refusing duplicate evaluation in this first bounded cycle."
        )
    if exact_prior is not None and (
        exact_prior.get("semantic_digest") != validation.semantic_digest
        or exact_prior.get("canonical_config") != proposal.canonical_behavior()
    ):
        raise RuntimeError("generated candidate conflicts with existing registered ID/version")

    bid = load_market_data() if market_data is None else market_data.copy()
    ask = load_ask_m1() if ask_market_data is None else ask_market_data.copy()
    bid_context = prepare_hypothesis_context(bid)
    bid, ask, alignment = prepare_bid_ask_inputs(
        bid_context["market"], ask, bid_metadata=bid_metadata,
        ask_metadata=ask_metadata, timestamp_convention="bar_open", bar_duration="1min",
    )
    ask_fingerprint, ask_identity = fingerprint_market_frame(
        ask, quality_policy={**M1_QUALITY_POLICY, "market_side": "ASK",
                             "alignment_policy": "exact_utc_timestamp_join_v1"},
    )
    # Use only the complete timestamp index to establish frozen boundaries.
    periods = timeline_periods(bid, bar_duration="1min")
    fixed_splits = split_metadata(periods)
    software = code_provenance(PROJECT_ROOT)
    full_times = bid["time"]
    folds = build_walk_forward_folds(
        full_times, outer_start=periods["train"]["start"],
        outer_end=periods["train"]["end"], maximum_outcome_horizon="16min",
        n_folds=n_folds, initial_history_fraction=0.50, embargo="16min",
    )
    models = [BidAskExecutionModel(s.commission_pips, s.slippage_pips)
              for s in cost_scenarios]
    cycle_store = ResearchEvidenceStore(store_path)
    cycle_id = cycle_store.start_cycle({
        "cycle_type": "training_screen_then_validation_gate_v1",
        "hypothesis_id": proposal.hypothesis_id, "hypothesis_version": proposal.version,
        "semantic_digest": validation.semantic_digest,
        "generation": {"generated": generation.proposals_generated,
                       "validated": generation.proposals_validated,
                       "rejected": generation.proposals_rejected,
                       "deduplicated": generation.proposals_deduplicated},
        "cost_scenarios": [asdict(s) for s in cost_scenarios],
        "split_metadata": fixed_splits, "test_evaluation": "prohibited",
        "candidate_budget": 1, "validation_budget": 1,
        "purge": "16min (15-minute max holding plus one M1 bar for strict later entry)",
        "embargo": "16min (same conservative horizon)",
        "source_fingerprint": bid_context["fingerprint"],
        "ask_fingerprint": ask_fingerprint,
        "software_provenance": software,
    })
    events = []
    executions: list[dict[str, Any]] = []
    validation_outcomes: list[dict[str, Any]] = []
    status = "failed"
    try:
        registration = experiment_store.register_hypothesis(proposal)
        if (registration["semantic_digest"] != validation.semantic_digest
                or registration["registration_status"] != "registered"):
            raise RuntimeError("registered candidate identity did not match validated proposal")
        generated_proposals: list[dict[str, Any]] = []
        for generated in generation.hypotheses:
            selected_behavior = generated.semantic_digest() == proposal.semantic_digest()
            equivalent_registered = registered_digests.get(generated.semantic_digest(), [])
            disposition = (
                "selected_for_train_screen" if selected_behavior else
                "duplicate_registered_behavior" if equivalent_registered else
                "outside_predeclared_pilot_scope"
            )
            generated_proposals.append({
                "hypothesis_id": generated.hypothesis_id, "version": generated.version,
                "semantic_digest": generated.semantic_digest(), "disposition": disposition,
                "registered_equivalents": equivalent_registered,
                "indicators": generated.to_dict()["indicators"],
                "long_entry": generated.to_dict()["long_entry"],
                "short_entry": generated.to_dict()["short_entry"],
            })
            if not selected_behavior:
                cycle_store.record_candidate_event(
                    cycle_id, generated.hypothesis_id, generated.version,
                    stage="candidate_scope_exclusion",
                    status=("duplicate_registered_behavior" if equivalent_registered
                            else "not_selected_by_predeclared_scope"),
                    reason=("semantic behavior was already registered" if equivalent_registered
                            else "the bounded pilot predeclared the RSI/volatility interaction only"),
                    details={"semantic_digest": generated.semantic_digest(),
                             "registered_equivalents": equivalent_registered},
                )
        cycle_store.record_candidate_event(
            cycle_id, proposal.hypothesis_id, proposal.version, stage="generated_validated_deduplicated",
            status="accepted", details={"semantic_digest": validation.semantic_digest,
                                         "generation": {
                                             "generated": generation.proposals_generated,
                                             "validated": generation.proposals_validated,
                                             "rejected": generation.proposals_rejected,
                                             "deduplicated": generation.proposals_deduplicated,
                                             "rejections": [asdict(row) for row in generation.rejections],
                                         }},
        )
        cycle_store.record_candidate_event(
            cycle_id, proposal.hypothesis_id, proposal.version, stage="registered",
            status="registered" if exact_prior is None else "exact_registration_reused",
            details={"registered_at_utc": registration["registered_at_utc"],
                     "prior_failed_attempts_preserved": exact_prior is not None},
        )
        training = run_registered_hypothesis_windows(
            proposal.hypothesis_id, proposal.version, folds,
            execution_models=models, store=experiment_store, market_data=bid,
            ask_market_data=ask, bid_metadata=bid_metadata, ask_metadata=ask_metadata,
        )
        scenario_fold_results: dict[str, list[dict[str, Any]]] = {
            f"commission_{model.commission_pips_round_trip:g}_slippage_{model.slippage_pips_round_trip:g}": []
            for model in models
        }
        for fold_result in training["folds"]:
            fold_meta = {key: value for key, value in fold_result.items() if key != "scenarios"}
            for scenario, result in fold_result["scenarios"].items():
                result["costs"] = _side_aware_cost_components(result["trades"])
                fold_meta_with_quality = {
                    **fold_meta,
                    "data_quality": result.get("data_quality", {}),
                    "attribution": result.get("attribution", {}),
                }
                scenario_fold_results[scenario].append({
                    **fold_meta_with_quality, "summary": result["summary"], "costs": result["costs"],
                    "trades": result["trades"], "rejected_signals": result["rejected_signals"],
                    "counts": {key: result.get(key) for key in (
                        "candidate_events", "warmup_excluded", "backtester_inputs",
                        "backtester_rejected", "accepted_trades")},
                    "rejection_counts": result["rejected"],
                })
                cycle_store.record_fold_result(
                    cycle_id, proposal.hypothesis_id, proposal.version, scenario,
                    {**fold_meta_with_quality, "trades": result["trades"],
                     "rejected_signals": result["rejected_signals"]},
                    metrics=result["summary"], costs=result["costs"],
                    counts={key: result.get(key) for key in (
                        "candidate_events", "warmup_excluded", "backtester_inputs",
                        "backtester_rejected", "accepted_trades")},
                    rejection_counts=result["rejected"],
                )
        family_size = max(1, generation.proposals_validated * len(models))
        screen = training_screen(
            scenario_fold_results, family_size=family_size, minimum_trades=100,
            block_length=5, n_resamples=bootstrap_resamples,
        )
        cycle_store.record_candidate_event(
            cycle_id, proposal.hypothesis_id, proposal.version, stage="training_screen",
            status="passed" if screen["passed"] else "rejected",
            reason=None if screen["passed"] else ";".join(screen["reasons"]),
            details=screen,
        )
        executions.append({"stage": "training_screen", "status": "passed" if screen["passed"] else "rejected",
                           "folds": len(folds), "family_size_configs": family_size})
        if screen["passed"]:
            # Validation consumes exactly one candidate budget and only after
            # the frozen training gate. Test is disabled in the registered runner.
            for scenario in cost_scenarios:
                model = BidAskExecutionModel(scenario.commission_pips, scenario.slippage_pips)
                try:
                    outcome = run_hypothesis_research(
                        [f"{proposal.hypothesis_id}@{proposal.version}"],
                        market_data=bid, ask_market_data=ask,
                        bid_metadata=bid_metadata, ask_metadata=ask_metadata,
                        execution_model=model, store_path=store_path,
                    )
                    row = outcome.hypotheses[0]
                    periods_out = row.get("periods", {})
                    val = periods_out.get("validation", {})
                    validation_outcomes.append({
                        "scenario": asdict(scenario), "experiment_id": row.get("experiment_id"),
                        "status": row.get("run_status", row.get("status")),
                        "validation": val, "test": "not evaluated",
                    })
                    cycle_store.record_candidate_event(
                        cycle_id, proposal.hypothesis_id, proposal.version,
                        stage="validation", status=str(row.get("run_status", row.get("status"))),
                        reason=row.get("error"), details={
                            "scenario": asdict(scenario), "experiment_id": row.get("experiment_id"),
                            "validation_metrics": val.get("metrics"),
                            "validation_counts": val.get("counts"),
                            "validation_rejection_counts": val.get("rejection_counts"),
                            "confirmatory": False, "selection_eligible": False,
                        },
                    )
                except Exception as exc:
                    validation_outcomes.append({"scenario": asdict(scenario), "status": "failed",
                                                "error": str(exc)})
                    cycle_store.record_candidate_event(
                        cycle_id, proposal.hypothesis_id, proposal.version,
                        stage="validation", status="failed", reason=str(exc),
                        details={"scenario": asdict(scenario)},
                    )
            status = "partial_failed" if any(x.get("status") == "failed" for x in validation_outcomes) else "completed"
        else:
            cycle_store.record_candidate_event(
                cycle_id, proposal.hypothesis_id, proposal.version, stage="validation_gate",
                status="skipped", reason="training criteria were not met; validation budget remains unused",
                details={"validation_budget_used": 0},
            )
            status = "completed"
        events = cycle_store.get_cycle(cycle_id)["candidate_events"]
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        try:
            cycle_store.record_candidate_event(
                cycle_id, proposal.hypothesis_id, proposal.version, stage="cycle_failure",
                status=status, reason=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        raise
    finally:
        # completion update must not overwrite an earlier terminal state.
        try:
            cycle_store.finish_cycle(cycle_id, status, {
                "training_screen": locals().get("screen"),
                "validation_outcomes": validation_outcomes,
                "test_period": {"evaluated": False, "selection_eligible": False},
            })
        except ValueError:
            pass

    report: dict[str, Any] = {
        "cycle_id": cycle_id, "cycle_status": status,
        "proposal": {"hypothesis_id": proposal.hypothesis_id, "version": proposal.version,
                     "semantic_digest": validation.semantic_digest,
                     "indicators": proposal.to_dict()["indicators"],
                     "long_entry": proposal.to_dict()["long_entry"],
                     "short_entry": proposal.to_dict()["short_entry"],
                     "rationale": proposal.rationale},
        "generated_proposals": generated_proposals,
        "generation": {"proposals_generated": generation.proposals_generated,
                       "proposals_validated": generation.proposals_validated,
                       "proposals_rejected": generation.proposals_rejected,
                       "proposals_deduplicated": generation.proposals_deduplicated,
                       "rejections": [asdict(row) for row in generation.rejections]},
        "training_only_prior_evidence": prior_train_evidence,
        "training_screen": screen,
        "training_scorecard": _training_scorecard(scenario_fold_results),
        "advance_decision": {
            "validation_advanced": bool(screen["passed"]),
            "reason": ("training gate passed" if screen["passed"] else
                       "predeclared training gate failed; validation budget unused"),
        },
        "training_folds": [
            {"fold": {key: value for key, value in fold.items() if key != "scenarios"},
             "scenarios": {scenario: {
                 "metrics": result["summary"], "costs": result["costs"],
                 "counts": {key: result.get(key) for key in (
                     "candidate_events", "warmup_excluded", "backtester_inputs",
                     "backtester_rejected", "accepted_trades")},
                 "rejection_counts": result["rejected"],
             } for scenario, result in fold["scenarios"].items()}}
            for fold in training["folds"]
        ],
        "validation_outcomes": validation_outcomes,
        "dataset": {"bid_fingerprint": bid_context["fingerprint"],
                    "ask_fingerprint": ask_fingerprint, "alignment": alignment,
                    "bid_rows": len(bid), "ask_rows": len(ask),
                    "start_utc": str(bid["time"].iloc[0]), "end_utc": str(bid["time"].iloc[-1]),
                    "period_boundaries": fixed_splits},
        "software_provenance": software,
        "protocol": {
            "selection": "single predeclared RSI(14) recross x volatility(15) filter, motivated by prior TRAIN-only cost-drag evidence; no prior validation or test metrics used",
            "cost_scenarios": [asdict(s) for s in cost_scenarios],
            "purge": "16 minutes (15-minute horizon + one M1 bar)",
            "embargo": "16 minutes",
            "fold_count": len(folds), "validation_budget": 1,
            "validation_status": "exploratory only; not independent confirmation",
            "test_period": "not evaluated; previously inspected; non-confirmatory and selection-ineligible",
        },
        "ledger_events": events,
        "limitations": [
            "Validation has been inspected in earlier work and is exploratory.",
            "The previously inspected final test period was not evaluated or queried by this cycle.",
            "Commission/slippage values are provisional assumptions; latency, market impact, liquidity and fill uncertainty remain unmodeled.",
            "Moving-block bootstrap blocks trades in entry order; block length five is a predeclared heuristic and does not eliminate data snooping.",
            "Bonferroni family size conservatively counts generated valid proposals times cost scenarios; this does not correct all researcher degrees of freedom.",
            "No parameter or strategy winner is selected by this cycle.",
        ],
        "next_milestone": "Reserve new future observations for pre-registered independent confirmation after execution assumptions are calibrated.",
    }
    if report_path is not None:
        write_evidence_cycle_report(report, report_path)
    return report


def write_evidence_cycle_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a standalone, reproducible report without replacing older reports."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior report: {output}")
    lines = [
        "# Evidence-driven strategy discovery pilot", "",
        "Exploratory research only. No profitability, robustness, or significance claim is made.",
        "The previously inspected test period was not evaluated, queried for metrics, or used for selection.", "",
        f"- Cycle ID: `{report['cycle_id']}`", f"- Status: `{report['cycle_status']}`",
        f"- Candidate: `{report['proposal']['hypothesis_id']}@{report['proposal']['version']}`",
        f"- Semantic digest: `{report['proposal']['semantic_digest']}`", "",
        "## Predeclared proposal and screening protocol", "",
        f"- Rationale: {report['proposal']['rationale']}",
        f"- Long rule: `{canonical_json(report['proposal']['long_entry'])}`",
        f"- Short rule: `{canonical_json(report['proposal']['short_entry'])}`",
        f"- Generation counts: `{canonical_json(report['generation'])}`",
        f"- Protocol: `{canonical_json(report['protocol'])}`", "",
        "## Generated candidates and fixed exclusions", "",
        *[f"- `{canonical_json(item)}`" for item in report["generated_proposals"]], "",
        "## Training-only prior evidence used to motivate the proposal", "",
        f"`{canonical_json(report['training_only_prior_evidence'])}`", "",
        "## Dataset and chronological boundaries", "",
        f"`{canonical_json(report['dataset'])}`", "",
        "## Software provenance", "",
        f"`{canonical_json(report['software_provenance'])}`", "",
        "## Purged training folds and cost sensitivity", "",
        f"- Training screen: `{canonical_json(report['training_screen'])}`", "",
        f"- Training scorecard (descriptive; not ranked): `{canonical_json(report['training_scorecard'])}`", "",
    ]
    for fold in report["training_folds"]:
        lines.append(f"- Fold: `{canonical_json(fold)}`")
    lines.extend(["", "## Validation outcomes", "",
                  ("Validation was permitted only after the fixed training gate; it remains exploratory."
                   if report["training_screen"]["passed"] else
                   "Validation was not run because the predeclared training gate failed; the validation budget remains unused."),
                  f"- Advance decision: `{canonical_json(report['advance_decision'])}`", ""])
    lines.extend(f"- `{canonical_json(item)}`" for item in report["validation_outcomes"])
    lines.extend(["", "## Persisted lifecycle ledger", ""])
    lines.extend(f"- `{canonical_json(item)}`" for item in report["ledger_events"])
    lines.extend(["", "## Limitations and next milestone", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append(f"- Next milestone: {report['next_milestone']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research_cycle(
    *, store_path: str | Path = DEFAULT_DB_PATH, market_data: pd.DataFrame | None = None,
    ask_market_data: pd.DataFrame | None = None, candidate_budget: int = 1,
    cost_scenarios: Sequence[CostScenario] = DEFAULT_COST_SCENARIOS,
    dry_run: bool = False, report_path: str | Path | None = None,
    bid_metadata: MarketDataMetadata = BID_METADATA,
    ask_metadata: MarketDataMetadata = ASK_METADATA,
) -> dict[str, Any]:
    """Generate one bounded batch and optionally run its predeclared scenarios.

    Actual runs use the production registered-only orchestrator. Dry runs do
    not register hypotheses or write SQLite records.
    """
    if isinstance(cost_scenarios, (str, bytes)) or not cost_scenarios:
        raise ValueError("at least one explicit cost scenario is required")
    if len(cost_scenarios) > 3 or any(not isinstance(x, CostScenario) for x in cost_scenarios):
        raise ValueError("at most three CostScenario values are supported")
    if candidate_budget * len(cost_scenarios) > MAX_CYCLE_EXPERIMENTS:
        raise ValueError(f"cycle exceeds experiment budget {MAX_CYCLE_EXPERIMENTS}")

    generation = generate_hypotheses(_proposal_config())
    store = ExperimentStore(store_path, initialize=not dry_run)
    prior_registry = store.list_registered_hypotheses()
    prior_digests: dict[str, list[str]] = {}
    for record in prior_registry:
        prior_digests.setdefault(record["semantic_digest"], []).append(
            f"{record['hypothesis_id']}@{record['hypothesis_version']}")
    # Candidate choice and duplicate screening depend only on frozen rule
    # structure and registry identity, never on market or experiment metrics.
    composite = [h for h in generation.hypotheses if len(h.indicators) == 3]
    novel_composite = [h for h in composite if h.semantic_digest() not in prior_digests]
    selected, selection_decisions = select_predeclared_candidates(
        novel_composite, budget=candidate_budget,
    )
    for proposal in composite:
        if proposal.semantic_digest() in prior_digests:
            selection_decisions.append({
                "hypothesis_id": proposal.hypothesis_id,
                "decision": "skipped_previously_registered_behavior",
                "reason": "semantic digest already registered as " + ", ".join(
                    prior_digests[proposal.semantic_digest()]),
            })
    for proposal in generation.hypotheses:
        if len(proposal.indicators) != 3:
            selection_decisions.append({
                "hypothesis_id": proposal.hypothesis_id,
                "decision": "not_in_predeclared_pilot",
                "reason": "bounded demonstration is scoped to the composite trend/volatility rule",
            })
    proposal_registry_status: list[dict[str, Any]] = []
    for proposal in generation.hypotheses:
        matches = prior_digests.get(proposal.semantic_digest(), [])
        proposal_registry_status.append({
            "hypothesis_id": proposal.hypothesis_id,
            "version": proposal.version,
            "semantic_digest": proposal.semantic_digest(),
            "registered_equivalents": matches,
            "status": "previous_behavior_already_registered" if matches else "new_behavior",
            "indicators": proposal.to_dict()["indicators"],
            "long_entry": proposal.to_dict()["long_entry"],
            "short_entry": proposal.to_dict()["short_entry"],
            "holding_period": proposal.exit_rule.max_holding_period,
            "warmup_bars": proposal.required_warmup_bars,
            "rationale": proposal.rationale,
        })
    registrations: list[dict[str, Any]] = []
    executable: list[Any] = []
    for hypothesis in selected:
        validation = validate_hypothesis(hypothesis)
        if validation.semantic_digest != hypothesis.semantic_digest():
            raise ValueError("generated hypothesis semantic digest failed validation")
        if dry_run:
            try:
                existing = store.get_registered_hypothesis(hypothesis.hypothesis_id, hypothesis.version)
            except KeyError:
                registrations.append({"hypothesis_id": hypothesis.hypothesis_id,
                                      "version": hypothesis.version,
                                      "status": "would_require_explicit_registration"})
                continue
            if existing["semantic_digest"] != validation.semantic_digest:
                registrations.append({"hypothesis_id": hypothesis.hypothesis_id,
                                      "version": hypothesis.version, "status": "identity_conflict"})
                continue
            registrations.append({"hypothesis_id": hypothesis.hypothesis_id,
                                  "version": hypothesis.version, "status": "registered_exact_match"})
            executable.append(hypothesis)
        else:
            record = store.register_hypothesis(hypothesis)
            if record["semantic_digest"] != validation.semantic_digest:
                raise ValueError("registration returned a different semantic digest")
            registrations.append({"hypothesis_id": hypothesis.hypothesis_id,
                                  "version": hypothesis.version, "status": record["registration_status"]})
            executable.append(hypothesis)

    bid = load_market_data() if market_data is None else market_data
    ask = load_ask_m1() if ask_market_data is None else ask_market_data
    bid_context = prepare_hypothesis_context(bid)
    bid, ask, alignment = prepare_bid_ask_inputs(
        bid_context["market"], ask, bid_metadata=bid_metadata,
        ask_metadata=ask_metadata, timestamp_convention="bar_open", bar_duration="1min",
    )
    ask_fingerprint, ask_identity = fingerprint_market_frame(
        ask, quality_policy={**M1_QUALITY_POLICY, "market_side": "ASK",
                             "alignment_policy": "exact_utc_timestamp_join_v1"},
    )
    bid_start, bid_end = utc_date_range(bid)
    fixed_periods = timeline_periods(bid, bar_duration="1min")
    fixed_split_metadata = split_metadata(fixed_periods)
    previews: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for hypothesis in executable:
        for scenario in cost_scenarios:
            model = BidAskExecutionModel(scenario.commission_pips, scenario.slippage_pips)
            preview = run_hypothesis_research(
                [f"{hypothesis.hypothesis_id}@{hypothesis.version}"], dry_run=True,
                market_data=bid, ask_market_data=ask, bid_metadata=bid_metadata,
                ask_metadata=ask_metadata, execution_model=model, store_path=store_path,
            )
            item = preview.hypotheses[0]
            preview_row = {"hypothesis_id": hypothesis.hypothesis_id, "version": hypothesis.version,
                           "scenario": asdict(scenario), "disposition": item["disposition"],
                           "experiment_id": item.get("experiment_id"),
                           "identity_differences": item.get("identity_differences", [])}
            previews.append(preview_row)
            if dry_run:
                continue
            if item["disposition"] not in {"new", "resumed", "reused_completed", "new_retry"}:
                executions.append({**preview_row, "status": "skipped_identity_or_lifecycle_safeguard",
                                   "reason": "dry-run disposition is not safely executable"})
                continue
            actual = run_hypothesis_research(
                [f"{hypothesis.hypothesis_id}@{hypothesis.version}"],
                market_data=bid, ask_market_data=ask, bid_metadata=bid_metadata,
                ask_metadata=ask_metadata, execution_model=model, store_path=store_path,
            )
            row = actual.hypotheses[0]
            persisted_identity: dict[str, Any] = {}
            experiment_id = row.get("experiment_id")
            if experiment_id is not None:
                stored_run = store.get_run(experiment_id)
                persisted_identity = {
                    key: stored_run.get(key) for key in (
                        "dataset_fingerprint", "dataset_identity", "date_start_utc",
                        "date_end_utc", "execution_assumptions", "gap_policy",
                        "software", "split_metadata",
                    )
                }
            executions.append({**preview_row, "orchestration_id": actual.orchestration_id,
                               "status": row.get("run_status", row.get("status")),
                               "disposition": row.get("disposition"), "experiment_id": row.get("experiment_id"),
                               "periods": row.get("periods", {}),
                               "identity": persisted_identity,
                               "failure": row.get("error")})

    observations = [
        {"experiment_id": execution.get("experiment_id"),
         "hypothesis_id": execution.get("hypothesis_id"), "scenario": execution.get("scenario"),
         "items": analyze_period_results(execution.get("periods", {}))}
        for execution in executions if execution.get("periods")
    ]
    # Load aggregate historical evidence only after the candidate set and
    # selection decisions are fixed and execution has finished.
    prior_history = _development_history(store)
    lifecycle = {
        "selected": len(selected),
        "registered": sum(item["status"] in {"registered", "already_registered"} for item in registrations),
        "executed": sum(item.get("disposition") in {"new", "resumed", "new_retry"}
                         and item.get("status") == "completed" for item in executions),
        "completed": sum(item.get("status") == "completed" for item in executions),
        "reused": sum(item.get("disposition") == "reused_completed" for item in executions),
        "failed": sum(item.get("status") in {"failed", "partial_failed", "interrupted"} for item in executions),
        "skipped": sum(item.get("status", "").startswith("skipped") for item in executions),
    }
    failed_count = sum(item.get("status") in {"failed", "partial_failed", "interrupted"}
                       for item in executions)
    blocked_count = sum(str(item.get("status", "")).startswith("skipped") for item in executions)
    cycle_status = "dry_run" if dry_run else (
        "partial_failed" if failed_count else "blocked" if blocked_count else "completed"
    )
    report = {
        "cycle_status": cycle_status,
        "generation": {
            "proposals_generated": generation.proposals_generated,
            "proposals_validated": generation.proposals_validated,
            "proposals_rejected": generation.proposals_rejected,
            "proposals_deduplicated": generation.proposals_deduplicated,
            "candidate_space_size": generation.candidate_space_size,
            "not_generated_due_to_limit": generation.not_generated_due_to_limit,
            "rejections": [asdict(item) for item in generation.rejections],
        },
        "selection_policy": "one three-indicator trend/volatility rule selected by fixed template structure before evaluation; no metrics consulted",
        "research_budget": {
            "candidate_budget": candidate_budget,
            "maximum_candidates": MAX_CYCLE_CANDIDATES,
            "cost_scenario_count": len(cost_scenarios),
            "maximum_experiments_per_cycle": MAX_CYCLE_EXPERIMENTS,
            "prior_registered_hypothesis_versions": len(prior_registry),
            "current_generation_proposals_examined": generation.proposals_generated,
            "total_hypothesis_versions_seen_or_generated": len(prior_registry) + generation.proposals_generated,
        },
        "dataset_preflight": {
            "symbol": bid_metadata.symbol, "timeframe": bid_metadata.timeframe,
            "timestamp_convention": bid_metadata.timestamp_convention,
            "timezone": bid_metadata.timezone, "provider": bid_metadata.provider,
            "bid_rows": len(bid), "ask_rows": len(ask), "start_utc": bid_start,
            "end_utc": bid_end,
            "bid_fingerprint_sha256": bid_context["fingerprint"],
            "ask_fingerprint_sha256": ask_fingerprint,
            "bid_identity": bid_context["identity"], "ask_identity": ask_identity,
            "alignment": alignment,
            "split_metadata": fixed_split_metadata,
            "test_evaluation": "disabled; previously inspected and selection-ineligible",
            "execution_policy": {
                "timestamp_convention": "bar_open", "bar_duration": "1min",
                "entry_delay": "0s", "holding_period": "15min",
                "gap_tolerance": "90s", "overlapping_positions": False,
                "legacy_fixed_cost_pips": 0.0,
                "cost_scenarios": [asdict(scenario) for scenario in cost_scenarios],
            },
        },
        "selection_decisions": selection_decisions,
        "generated_hypotheses": proposal_registry_status,
        "registrations": registrations,
        "previous_development_history": prior_history,
        "dry_run_previews": previews,
        "executions": executions,
        "lifecycle_counts": lifecycle,
        "failure_analysis": observations,
        "next_step_recommendations": recommend_next_steps(
            [item for observation in observations for item in observation.get("items", [])]
        ),
        "test_period": {"status": "not_evaluated", "previously_inspected": True,
                        "confirmatory": False, "selection_eligible": False},
        "limitations": [
            "The existing official runner evaluates train and validation together; validation is already inspected and exploratory.",
            "No clean untouched historical holdout is available; old test remains excluded.",
            "Commission and slippage scenarios are assumptions, not calibrated fills.",
            "The current store persists aggregate period metrics, not trade-level series needed for block-bootstrap uncertainty estimates.",
            "The existing runner has fixed 60/20/20 periods and does not expose purged rolling folds; no walk-forward claims are made.",
        ],
    }
    if report_path is not None:
        write_cycle_report(report, report_path)
    return report


def write_cycle_report(report: dict[str, Any], path: str | Path) -> None:
    lines = ["# Bounded autonomous research cycle", "",
             "Descriptive exploratory research only. No profitability, robustness, or significance claim is made.",
             "The previously inspected test period was not evaluated or used for selection.", "",
             f"- Cycle status: `{report['cycle_status']}`",
             f"- Research budget: `{canonical_json(report.get('research_budget', {}))}`",
             f"- Dataset preflight: `{canonical_json(report.get('dataset_preflight', {}))}`",
             f"- Lifecycle counts: `{canonical_json(report.get('lifecycle_counts', {}))}`",
             f"- Generation accounting: `{canonical_json(report['generation'])}`",
             f"- Selection policy: {report['selection_policy']}", ""]
    lines.append(f"- Research budget: `{canonical_json(report.get('research_budget', {}))}`")
    lines.extend(["", "## Generated proposals", ""])
    for proposal in report.get("generated_hypotheses", []):
        lines.extend([
            f"### {proposal['hypothesis_id']} v{proposal['version']}",
            f"- Semantic digest: `{proposal['semantic_digest']}`",
            f"- Registry status: `{proposal['status']}`; equivalents: `{canonical_json(proposal['registered_equivalents'])}`",
            f"- Indicators: `{canonical_json(proposal['indicators'])}`",
            f"- Long rule: `{canonical_json(proposal['long_entry'])}`",
            f"- Short rule: `{canonical_json(proposal['short_entry'])}`",
            f"- Exit/warm-up: {proposal['holding_period']} / {proposal['warmup_bars']} bars",
            f"- Rationale: {proposal['rationale']}", "",
        ])
    for decision in report["selection_decisions"]:
        lines.append(f"- Selection: `{decision['hypothesis_id']}` — {decision['decision']}: {decision['reason']}")
    lines.extend(["", "## Registration and execution", ""])
    for item in report["registrations"]:
        lines.append(f"- `{item['hypothesis_id']}@{item['version']}`: {item['status']}")
    lines.extend(["", "## Stored train/validation experiment history", "",
                  "Prior records and current-cycle results are listed for audit context. This history is loaded only after candidate selection and does not affect proposal generation or selection.", ""])
    for record in report.get("previous_development_history", []):
        lines.append(f"- `{record['experiment_id']}` {record['hypothesis_id']}@{record['version']} ({record['status']}): `{canonical_json(record['periods'])}`")
    for item in report["executions"]:
        lines.extend(["", f"### {item.get('hypothesis_id')} / {item.get('scenario', {}).get('name', 'preview')}",
                      f"- Experiment ID: `{item.get('experiment_id')}`",
                      f"- Status/disposition: `{item.get('status')}` / `{item.get('disposition')}`"])
        if item.get("identity"):
            lines.append(f"- Dataset fingerprint: `{item['identity'].get('dataset_fingerprint')}`")
            lines.append(f"- Period boundaries: `{canonical_json(item['identity'].get('split_metadata', {}))}`")
            lines.append(f"- Code provenance fingerprint: `{item['identity'].get('software', {}).get('source_fingerprint_sha256')}`")
            lines.append(f"- Execution assumptions: `{canonical_json(item['identity'].get('execution_assumptions', {}))}`")
        for period_name in ("train", "validation"):
            period = item.get("periods", {}).get(period_name)
            if period:
                lines.append(f"- {period_name} metrics: `{canonical_json(period.get('metrics', {}))}`")
                lines.append(f"- {period_name} costs: `{canonical_json(period.get('costs', {}))}`")
                lines.append(f"- {period_name} counts/rejections: `{canonical_json(period.get('counts', {}))}` / `{canonical_json(period.get('rejection_counts', {}))}`")
        lines.append("- Test: not evaluated; previously inspected, non-confirmatory, selection-ineligible.")
    lines.extend(["", "## Evidence-limited observations", ""])
    lines.extend(f"- `{canonical_json(item)}`" for item in report["failure_analysis"])
    lines.extend(["", "## Limitations and next step", ""])
    lines.extend(f"- Recommendation: {item}" for item in report.get("next_step_recommendations", []))
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["- Highest-priority next milestone: add a separately versioned, purged walk-forward evaluator that persists trade-level returns and uncertainty inputs, then reserve genuinely new future data for confirmation.", ""])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Run the bounded evidence-driven hypothesis pilot")
    parser.add_argument("--store", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report_path = args.report or (
        PROJECT_ROOT / "reports" /
        f"evidence_discovery_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md"
    )
    report = run_evidence_driven_cycle(
        store_path=args.store, bootstrap_resamples=args.bootstrap_resamples,
        report_path=report_path,
    )
    print(canonical_json({key: report[key] for key in (
        "cycle_id", "cycle_status", "proposal", "generation", "training_screen",
        "validation_outcomes", "dataset", "protocol",
    )}))
    return 0 if report["cycle_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
