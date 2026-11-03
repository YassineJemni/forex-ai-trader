# Local experiment tracking

The research runners use a small Python strategy registry and a local SQLite database at `results/experiments.sqlite3`. No cloud service or dashboard is involved. The database is generated output; the source registry and this document define the human-maintained metadata.

## Registered strategies

| Stable ID | Family | Version | Fixed configuration |
|---|---|---|---|
| `ema_crossover` | Trend following | `1.0.0` | EMA 9/21, 15-minute hold, 105-bar segment warm-up |
| `rsi_mean_reversion` | Mean reversion | `1.0.0` | RSI(14), recross 30/70, 15-minute hold, 105-bar segment warm-up |
| `donchian_breakout` | Breakout | `1.0.0` | Prior 20-candle channel, 15-minute hold, 105-bar segment warm-up |

Rules, hypotheses, and parameters live in `src/research/strategy_registry.py`. Bump a strategy version when its rule or parameter meaning changes; do not reassign an existing ID to a different strategy.

## Database records

`experiment_runs` stores the UUID experiment ID, attempt, strategy/version, canonical parameters, raw-file SHA-256, dataset facts, full UTC date range, execution and gap assumptions, data-quality counts, source-code fingerprint, Git state when available, timestamps, status, failure details, validation outcome classification, and `split_metadata_json`. Split metadata records the ordered-timeline policy, train/validation/test fractions, and exact UTC start/end plus row-index boundaries for each period. `period_results` stores separate structured JSON fields for metrics, rejection counts, costs, counts, and period-specific data-quality details.

No current result is labeled confirmatory: the existing experiments have no fresh confirmatory protocol. Train and validation are development results and are selection-eligible. Previously inspected test results are recorded with `confirmatory=0` and `selection_eligible=0`; `selection_results()` returns only train and validation, and `assert_selection_eligible()` rejects test access for selection.

The run request key hashes the strategy/version, parameters, dataset fingerprint and identity, date range, assumptions, code provenance, data-quality metadata, and split metadata. An identical completed request reuses its prior record and reconstructs Markdown output from SQLite without regenerating signals or rerunning backtests. In a comparison run, completed strategies are reused independently while only active strategies are evaluated. An incomplete identical request resumes with the same ID. A failed run stays in the database; retry it explicitly with `--retry-failed`, which starts a distinct attempt and experiment ID. A comparison startup error does not mark previously resumed runs failed; a run is failed only after its actual evaluation begins and that execution encounters an error.

Only explicitly named duration fields are normalized before hashing: `execution_assumptions.bar_duration`, `holding_period`, and `execution_delay`, plus `gap_policy.tolerance` and `indicator_reset_above`. Their canonical ISO duration form makes equivalent inputs such as `0s` and `0 days 00:00:00` identical; arbitrary strings and `experiment_kind` are not normalized. The full source fingerprint over `src/**/*.py` remains part of identity. It intentionally changes when any Python source file changes, including unrelated source files, so code differences cannot silently reuse a result. Historical candidates are previewed individually with run ID, status, field-level differences, and an `exact_match`, `mismatch`, or `unverifiable` disposition. Missing historical split metadata is labeled unrecorded; old boundaries are never inferred. Missing split metadata or source fingerprint prevents safe reuse, without rewriting historical rows or keys.

Both runners validate their fixed signal/evaluation configuration against the strategy registry before execution. A mismatch fails clearly rather than silently recording metadata that differs from the implementation. Existing databases are migrated by adding the split metadata column; pre-migration rows receive `{}` because their exact split boundaries are not inferred retroactively. New experiment requests include split metadata in their idempotency key.

The orchestration entry point is documented in `docs/research_orchestration.md`. Actual orchestration requests and per-strategy outcomes are stored in `orchestration_runs` and `orchestration_strategy_runs`; dry runs are read-only and are not persisted as executions. `experiment_execution_claims` provides expiring SQLite leases so a second orchestrator cannot evaluate the same incomplete run concurrently. Completed results remain immutable, and an interrupted run remains resumable after its lease expires.

Failures use `insufficient_data`, `invalid_configuration`, `execution_error`, or `persistence_error`. Negative validation net P&L is recorded as `validation_outcome=negative_net_pips` on a completed run, not as a claim about future results and not as an execution failure.

## Reproducibility limits

The dataset fingerprint hashes exact input-file bytes, so a changed export gets a different identity even if its displayed name is unchanged. Code provenance records the Git commit and clean/dirty state only when the project is the recognized Git root. This checkout has no project-local `.git` directory, so its Git commit and dirty state are reported as unavailable; a SHA-256 fingerprint over the Python files in `src/` is stored instead. Python, pandas, NumPy, SQLite, and platform versions are also recorded. The source fingerprint does not capture the full Python environment, system libraries, or untracked non-Python configuration, so the record does not claim bit-for-bit reproducibility.

The fixed transaction cost remains an assumption, not a verified spread or fill model. Timestamp, timezone, BID-only price source, execution quality, and historical test-period exposure retain the limitations documented in the experiment reports. Stored results are descriptive research records, not evidence of profitability.

## Versioned side-aware execution

The backtester's default remains the legacy fixed round-trip-cost model. The
opt-in `bid_ask_side_aware` execution model version `1.0.0` uses ASK open for
long entries, BID close for long exits, BID open for short entries, and ASK
close for short exits. It pairs observations by exact UTC timestamp and
rejects unavailable required quotes; it never joins by row position or fills
quotes. Source metadata must agree on symbol, M1 timeframe, bar convention,
UTC, and provider. A side-aware hypothesis must set the legacy
`transaction_cost_pips` field to zero so the old fixed deduction cannot be
applied in addition to spread.

Spread is reflected in the side-specific entry and exit prices. Commission
and slippage are separate configurable round-trip pip assumptions; they are
not estimated from OHLC data. The run identity records the execution model ID
and version, both source metadata records and normalized data fingerprints,
timestamp alignment policy/report, commission and slippage assumptions, and
that the fixed cost was not applied. This makes side-aware runs distinct from
legacy fixed-cost runs. No local ASK historical data is currently available,
so this model is infrastructure only until a matching verified ASK dataset is
provided.

## Registered hypothesis experiments

Declarative proposals are not runnable until explicitly registered through
`ExperimentStore.register_hypothesis()`. Registration validates the structure
and stores the ID/version, semantic digest, canonical executable configuration,
and registration time. An identical ID/version registration is idempotent;
different content under that same key is rejected. Separate versions are
retained independently.

The hypothesis path uses `run_hypothesis_research()` and
`python -m src.research.hypothesis_orchestrator id@version`. It does not route
through the baseline EMA/RSI/Donchian name mapping. Dry runs are read-only and
do not compile signals or call the backtester. Batches are capped at ten and
ordered by hypothesis ID/version. New runs persist the canonical hypothesis
configuration and digest alongside ordinary data, software, execution, and
split identity. The schema migration is additive: historical run records and
idempotency keys are not rewritten, and missing historical hypothesis
identity, split boundaries, or provenance are never inferred.

Hypothesis runs reuse the SQLite expiring execution lease, immutable completed
result handling, and deliberate retry-attempt behavior. Only train and
validation are evaluated. Test is intentionally omitted; previously inspected
test-period results remain non-confirmatory and selection-ineligible. The
summary is descriptive and does not rank strategies or imply profitability.

## Evidence-driven discovery ledger

`research_evidence.py` adds four SQLite tables using `CREATE TABLE IF NOT
EXISTS`; it does not rewrite or backfill prior experiment rows. `research_cycles`
stores the request and terminal summary; `research_candidate_events` is an
append-only lifecycle trail; `research_fold_results` stores UTC fold boundaries,
metrics, costs, counts, rejection counts and uncertainty settings;
`research_trade_evidence` stores attributed accepted trades and rejected
signals in sequence. Existing `experiment_runs` and report records are
preserved.

The bounded CLI proposal is RSI(14) threshold recrossing conditioned on 15-bar
volatility below 0.0005. Its motivation is computed from prior registered
**train-only** records where mid-price gross was positive but side-aware net
was negative after embedded spread and explicit assumed costs. No validation
or test result enters proposal generation. The proposal must pass the formal
schema validator and explicit registration before the registered walk-forward
runner can evaluate it.

The training screen uses three chronological folds inside the original 60%
training partition, 16-minute purge and embargo intervals, three fixed
side-aware commission/slippage assumptions, at least 100 trades in every
scenario, positive fold net P&L throughout, and a deterministic circular
moving-block bootstrap interval with Bonferroni familywise adjustment. The
five-trade block length is a declared heuristic, not a calibrated dependence
model. This correction does not eliminate all multiple-testing or data-snooping
risk. Validation runs only after the training gate and remains exploratory;
the previously inspected final 20% test interval is not evaluated or queried.

The report preserves fold expectancy, profit factor, drawdown,
trade/rejection counts, cost decomposition and uncertainty settings. There is
no automated winner ranking. Commission and slippage remain provisional
assumptions; latency, market impact, liquidity and actual execution quality
are not modeled. Reproduce with the command in
`docs/research_orchestration.md`, writing to a new timestamped report path.

## Inspect locally

Use Python's built-in SQLite client or `sqlite3`:

```sql
SELECT experiment_id, strategy_id, strategy_version, status, validation_outcome,
       started_at_utc, completed_at_utc
FROM experiment_runs
ORDER BY started_at_utc DESC;

SELECT experiment_id, period, metrics_json, rejection_counts_json,
       costs_json, confirmatory, selection_eligible
FROM period_results
ORDER BY experiment_id, period;

SELECT experiment_id, split_metadata_json
FROM experiment_runs
ORDER BY started_at_utc DESC;

SELECT hypothesis_id, hypothesis_version, semantic_digest, strategy_family,
       validation_status, registration_status, registered_at_utc
FROM registered_hypotheses
ORDER BY hypothesis_id, hypothesis_version;

SELECT experiment_id, hypothesis_id, hypothesis_version, hypothesis_digest, status
FROM experiment_runs
WHERE hypothesis_id IS NOT NULL
ORDER BY started_at_utc DESC;

SELECT orchestration_id, selected_strategy_ids_json, status, started_at_utc,
       completed_at_utc, summary_json
FROM orchestration_runs
ORDER BY started_at_utc DESC;

SELECT orchestration_id, strategy_id, experiment_id, disposition, status, error_message
FROM orchestration_strategy_runs
ORDER BY orchestration_id, rowid;
```
