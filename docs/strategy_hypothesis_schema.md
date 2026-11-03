# Strategy hypothesis schema (Stage 1)

`src/research/hypothesis_schema.py` defines immutable, versioned JSON-compatible
records for declarative strategy hypotheses. Rules are typed trees; the schema
does not evaluate strings or execute generated code. `hypothesis_validator.py`
performs structural checks only. A valid record is not an evaluated strategy
and validation makes no claim about performance.

## Rule and execution limits

Rule operators are `gt`, `gte`, `lt`, `lte`, `eq`, `ne`, `crosses_above`,
`crosses_below`, `and`, `or`, and `not`. A feature reference uses
`offset_bars=0` for the current completed signal candle and a positive offset
for older candles. Negative offsets are future references and are rejected.
Price fields `open`, `high`, `low`, and `close` may be referenced because all
are known once the decision candle closes. Volume is excluded because its
meaning is not established for this data.

The current validator supports only `fixed_duration` exits. Holding duration,
bar duration, execution delay, gap tolerance, overlap setting, pip size, fixed
transaction cost, and timestamp convention map to fields accepted by the
existing OHLC backtester. The fixed cost is an assumption; it is not a model
of historical spread, slippage, commission, or fill quality. Rule-based exits,
stops, targets, and trailing exits are rejected because the current backtester
does not implement them.

## Indicator allowlist and warm-up

Only columns currently emitted by `build_features()` are advertised. Listed
periods/spans are the choices supported by this schema; the general feature
builder's existing calculations remain unchanged.

| Indicator input | Supported configuration | Output | Minimum segment bars |
|---|---|---|---:|
| `ema` | span 9, 21, or 50 | `ema_9`, `ema_21`, `ema_50` | 5 × span: 45, 105, or 250 |
| `ema_gap` | none | `ema_gap` (EMA 9/21) | 105 |
| `ema_long_gap` | none | `ema_long_gap` (EMA 21/50) | 250 |
| `return` | period 1, 5, 15, or 60 | matching `return_N` | period + 1 bars |
| `rsi` | period 14 | `rsi_14` | 15 |
| `volatility` | period 15 or 60 | matching `volatility_N` | period + 1 bars |
| `candle_range`, `candle_body` | none | same-named feature | 1 |
| `time_of_day` | component `sin` or `cos` | `time_sin` or `time_cos` | 1 |

The EMA threshold is a practical stabilization convention: five spans do not
mathematically eliminate recursive seed influence. The 105-bar convention is
therefore not universal. RSI(14) needs 15 observed bars because it uses 14
valid close changes; rolling volatility needs one initial close plus its
window of returns. A crossing rule needs a preceding observation in addition
to the referenced feature's minimum, and explicit lag offsets add the required
history. The declared warm-up must meet the greatest requirement across its
indicators and rules. Warm-up is measured within one indicator segment; feature
state resets after gaps greater than 90 seconds, including weekends under the
current conservative policy.

Donchian levels are computed inside the existing strategy comparison code,
not emitted by `build_features()`, so Donchian is intentionally not in this
Stage 1 indicator allowlist. No signal compiler or runner integration is part
of this stage.

## Canonical identity

`canonical_behavior()` sorts JSON keys and normalizes duration strings. It
includes indicators and parameters, entry trees, decision timing, fixed exit,
warm-up, and execution assumptions. It excludes hypothesis ID, version,
family classification, rationale, assumptions text, and limitation text. Thus the semantic digest
detects behaviorally identical records despite different labels, while ID and
version remain available for tracking. Boolean `and`/`or` child order is
canonicalized; the digest is a deterministic structural identity, not a
general-purpose theorem prover for all logically equivalent expressions.

The validator never reads candles, labels, returns, experiment results, or
test-period metrics. Future template generation and runner/store integration
must remain separate and preserve the existing experiment identity and
provenance safeguards.

## Stage 2 deterministic templates

`src/research/hypothesis_templates.py` enumerates a finite grammar over
predeclared choices. Its defaults generate 14 raw proposals: three EMA pairs,
three RSI threshold pairs, one prior-candle breakout, three return-momentum
periods, and four volatility-filter combinations. The breakout template is a
one-prior-candle high/low break with `return_1` direction confirmation; it is
not a Donchian channel strategy. No historical data or experiment results are
inputs to generation.

`TemplateConfig` permits subsets of the explicit option lists. Enumeration
order is fixed by family and then parameter order. `candidate_limit` defaults
to 50 and cannot exceed the hard maximum of 100; it limits raw proposals
examined. The result reports raw proposals generated, valid proposals,
rejections with reasons, semantic duplicates, total finite candidate-space
size, and proposals not visited because of the limit. For duplicates, the
first validated proposal is retained. No proposal is ranked or screened using
performance. Generated hypotheses remain unevaluated and are not executable:
signal compilation, backtesting, registry/store registration, and orchestration
integration are later stages.

## Stage 3 causal compiler and isolated runner

`src/research/hypothesis_signals.py` accepts only a structurally validated
`StrategyHypothesis`. It evaluates the allowlisted rule-tree operators using
features from `build_features()` and completed-bar OHLC fields. `offset_bars=0`
means the current completed candle; positive offsets and crossing history are
shifted within indicator segments. Missing observations make a rule unavailable
and cannot become true through `not`, `and`, or `or`. Crosses lack a previous
value at a segment start. Signals include deterministic IDs, source-row index,
source candle time, close-time decision, segment information, and warm-up
readiness. Source attribution is checked against the same normalized ordered
market frame used by the compiler.

`src/research/hypothesis_runner.py` evaluates one supplied hypothesis through
the existing sequential backtester, with execution assumptions taken from the
validated hypothesis. It uses the existing full-timeline 60/20/20 boundaries,
filters candidates by their assigned decision period, and truncates each
period's candles so accepted exit timestamps remain strictly inside that
period. Only train and validation are evaluated. The previously inspected test
interval is returned as `not_evaluated`, non-confirmatory, and
selection-ineligible; test-period signals and metrics are not returned. The
runner has no database or orchestrator integration and never chooses between
hypotheses. Its fixed-cost field remains an assumption, not historical spread,
commission, slippage, or verified fill data.

## Stage 4 registration and persistent evaluation

Generated proposals remain inert until `ExperimentStore.register_hypothesis()`
explicitly validates and records them. Registration is keyed by hypothesis ID
and version. Repeating an identical registration is idempotent; conflicting
content under the same ID/version is rejected. New versions are separate
records and preserve their own semantic digests.

`run_hypothesis_research()` accepts only registered `id@version` selections
(maximum batch size 10), orders them by ID/version, and dispatches through the
isolated `run_hypothesis()` runner. Its CLI is
`python -m src.research.hypothesis_orchestrator id@version [id@version ...]`;
`--dry-run` previews identity without database writes, signal compilation, or
backtesting. For example, register from Python with
`store.register_hypothesis(hypothesis)`, then preview with
`python -m src.research.hypothesis_orchestrator --dry-run example@1.0.0`.

The additive SQLite migration creates `registered_hypotheses` and nullable
hypothesis ID/version/digest columns on new/legacy experiment rows. Historical
rows are not rewritten; absent hypothesis identity remains NULL and absent
split/provenance metadata remains unrecorded. New request identity includes the
full canonical executable configuration, semantic digest, ID/version, full
source fingerprint, dataset fingerprint, execution assumptions, and exact
split metadata. Completed results are reused only for an exact request with
recorded provenance and splits. Existing leases prevent concurrent execution;
interrupted work remains resumable and deliberate failed retries create new
attempts.

Only train and validation are evaluated and persisted by this runner. The
previously inspected test period is not run, is marked non-confirmatory and
selection-ineligible, and is not used to rank hypotheses. Summaries are
descriptive; they do not establish a robust edge or profitability. The current
fixed transaction cost remains an assumption rather than verified historical
spread, slippage, commissions, or fill quality.
