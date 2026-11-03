# Strategy research orchestration

The orchestration API coordinates the existing registered strategy runners. It does not generate strategy rules, search parameter combinations, rank strategies, or make profitability claims. The fixed EMA, RSI, and Donchian definitions, execution rules, costs, gap policy, and 60/20/20 timeline split remain owned by `compare_strategies.py` and the shared backtester.

## Usage

Use stable strategy IDs from `strategy_registry.py`:

```powershell
python -m src.research.experiment_orchestrator --strategy ema_crossover --dry-run
python -m src.research.experiment_orchestrator --strategy ema_crossover --strategy rsi_mean_reversion
python -m src.research.experiment_orchestrator --strategy donchian_breakout --retry-failed
```

The Python API is:

```python
from src.research.experiment_orchestrator import run_research

result = run_research(
    ["ema_crossover", "rsi_mean_reversion"],
    dry_run=False,
    retry_failed=False,
    summary_path="reports/orchestration.md",
)
```

The selection must be nonempty, contain registered IDs only, and contain no duplicates. Parameters and execution assumptions are not CLI options in this phase.

## Dry runs and identities

A dry run validates strategy IDs and runner/registry configuration, loads the data needed to calculate exact run identities, and reads the SQLite database in read-only mode to preview `new`, `resumed`, `reused_completed`, or `retry_required` disposition. It does not initialize or write the database, create experiments or claims, generate signals, or run backtests. A requested summary path is not written during a dry run.

Run identity remains strict. It includes the strategy/version, fixed parameters, dataset fingerprint and identity, date range, execution and gap assumptions, code provenance, data quality, and exact split metadata. If an earlier strategy record differs in any identity field, the preview reports the differing fields and says a new experiment is required. The orchestrator does not silently adopt a result with different provenance or assumptions.

## Execution, retries, and interruption

Actual orchestrations are recorded in `orchestration_runs`; each requested strategy has a corresponding row in `orchestration_strategy_runs` with its disposition, experiment ID when available, status, and error. Successful results remain in the existing `experiment_runs` and `period_results` tables. An individual failure is recorded for that strategy and the orchestrator continues with later requested strategies. A failed experiment is retried only when `retry_failed=True` (CLI `--retry-failed`), which preserves the earlier failed attempt.

Before evaluating an incomplete run, the runner acquires a SQLite execution lease. A live lease prevents a second orchestration from computing the same experiment concurrently. The lease is renewed during evaluation. Completion and failure updates verify the lease owner; a stale worker cannot overwrite results after losing its claim. If interrupted, the orchestration is recorded as `interrupted`, the active experiment remains `running`, and its lease expires naturally so a later invocation can resume it. Completed experiment records are immutable.

Database initialization adds three tables with `CREATE TABLE IF NOT EXISTS`: `experiment_execution_claims`, `orchestration_runs`, and `orchestration_strategy_runs`. No existing experiment rows or result payloads are rewritten. Existing migrations for split metadata remain additive. Dry runs avoid even these schema migrations.

## Results and test-period safeguards

Summaries are reconstructed from persisted run and period records. They preserve the user's requested strategy order and show train/validation metrics, trade counts, costs, rejection counts, and data-quality metadata. The summary does not sort or choose a strategy.

Previously inspected test results, when present, are shown only as descriptive historical output and are labeled `previously inspected`, `non-confirmatory`, and `selection-ineligible`. The store's selection-facing API exposes train and validation only. No orchestration step uses test metrics to choose or filter strategies.

The fixed one-pip round-trip cost remains an assumption rather than a verified spread/fill model. Timestamp and BID-only historical price limitations remain as described in the experiment reports.

## Evidence-driven hypothesis cycle

The original `run_research_cycle()` remains for backwards compatibility. The
CLI now runs `run_evidence_driven_cycle()`, a bounded staged protocol. It
generates and validates a fixed RSI(14) recross plus 15-bar volatility filter
proposal. The proposal is motivated only by prior registered mean-reversion
**training** records where positive mid-price gross was erased by spread and
explicit assumed costs. It does not read prior validation or test metrics to
generate or select the proposal. Existing standalone RSI behavior is included
in generator accounting; only the composite, if semantically new, is
registered and screened.

The registered runner creates signals causally on the training candles. It
evaluates three expanding-history folds contained in the existing outer 60%
training period. The first half is initial history; each evaluation window is
preceded by a 16-minute purge and separated from the next evaluation window by
a 16-minute embargo. Sixteen minutes is the fixed 15-minute maximum holding
horizon plus one M1 bar for strict post-decision entry. The exact row counts
and UTC boundaries are stored. These are walk-forward evaluation windows; the
purge/embargo reduce adjacent-window outcome overlap but do not make the
historical observations independent.

The hard training gate is fixed before the cycle: at least 100 trades across
folds for every declared cost scenario; positive net P&L in every fold; and a
95% familywise Bonferroni-adjusted circular moving-block-bootstrap percentile
interval for mean trade return whose lower bound is above zero. The bootstrap
uses five-trade blocks in entry order, a deterministic seed and 1,000
resamples by default. Family size is the number of valid generated proposals
times the three cost scenarios. This correction does not remove all data
snooping or researcher degrees of freedom. Fold scorecards also preserve
expectancy, profit factor, drawdown, trade counts, rejection accounting and
cost decomposition; no candidate is ranked automatically.

Only if the candidate passes this training-only gate does the cycle spend its
one-candidate validation budget through the existing registered orchestrator.
Validation is exploratory because it has been inspected in earlier work. The
test period is not evaluated, sent to feature construction, or queried for
metrics; it remains previously inspected, non-confirmatory, and
selection-ineligible. No candidate proceeds adaptively after validation.

Three fixed side-aware cost scenarios are evaluated: 0.0 commission/0.5 pip
round-trip slippage, 0.5/0.5, and 0.5/1.0. They are provisional assumptions,
not calibrated execution costs. Side-specific bid/ask fills embed spread;
the legacy fixed cost is zero. Latency, market impact, liquidity, and actual
fill quality remain unmodeled.

The additive `research_cycles`, `research_candidate_events`,
`research_fold_results`, and `research_trade_evidence` tables preserve
proposal lifecycle, fold metrics, cost/rejection accounting, and attributed
accepted/rejected signal records. Prior experiment rows are not rewritten.
Failures and a failed training gate remain recorded. Reproduce the bounded
demonstration with:

```powershell
.\.venv\Scripts\python.exe -m src.research.research_cycle --bootstrap-resamples 1000 --report reports/evidence_discovery_YYYYMMDDTHHMMSSZ.md
```

Use a new report filename; the writer refuses to overwrite an existing report.
The cycle deliberately stops after one candidate. A clean independent
confirmation requires future observations that were not used in prior work.
