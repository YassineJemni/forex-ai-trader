# Mission 4: causal predictability map protocol

This protocol is fixed before any real-data predictability results are
calculated. It is an exploratory training-only association map, not a trading
strategy, profitability test, or evidence of executable returns. Validation
and historical-test observations are excluded.

## Frozen timeline and label definition

The sorted BID M1 timeline must have a valid, unique minute-cadence prefix
through index 404,889, and that index must equal
`2026-02-03T18:36:00Z`. A missing prefix or timestamp mismatch fails closed.
This boundary comes from the existing reported split and must not be
re-derived after filtering. The reported complete source has 674,815 rows, but
the implementation does not validate/count its post-boundary tail.

Boundary verification uses the sorted original BID timestamps to locate the
fixed index; it reads no post-boundary quote prices and computes no post-
boundary performance or quality statistics. After verification, BID and ASK
prices are restricted to bar-open timestamps strictly before the boundary.
Features use only these training rows. Each source bar at `t` closes and produces its decision at
`t + 1 minute`. Entry is the exact next M1 open at `t + 2 minutes`; the
existing backtester and its strict-later convention are unchanged. For elapsed
horizon `h`, exit uses the exact close at `entry + h`, belonging to the bar
opening at `entry + h - 1 minute`. Missing entry/exit timestamps, missing
matching-side quotes, gaps above 90 seconds, or entry/exit bars whose close is
at or beyond the boundary are rejected with reason counts.

The continuous target is
`ln(exit_mid / entry_mid)`, where entry midpoint is the average BID/ASK open
and exit midpoint is the average BID/ASK close. No quote is interpolated,
forward-filled, or paired by row position.

## Frozen test family

All 16 predictors are tested at each of the four horizons (`1min`, `5min`,
`15min`, `60min`), for exactly 64 feature-horizon tests:

1. `return_1`, `return_5`, `return_15`, `return_60`
2. `ema_gap`, `ema_long_gap`, `rsi_14`
3. `volatility_15`, `volatility_60`
4. `candle_range`, `candle_body`
5. `time_sin`, `time_cos`
6. `spread_mid_ratio`, `spread_mid_ratio_mean_15`,
   `spread_mid_ratio_std_15`

Spread predictors use close-side spread divided by close midpoint; the two
rolling statistics use 15 observed bars within the existing gap-reset
segments. No feature selection, fitted threshold, or family expansion is
permitted after results are seen.

## Frozen statistical procedure

- Effect: Pearson correlation between the feature and continuous forward
  log-return.
- All four horizons use the intersection of accepted, complete-feature
  decision timestamps. This keeps the tested observation set identical across
  horizons and retains shared dependence in the family-level null.
- Uncertainty: Newey-West HAC standard error of the correlation influence
  function with a Bartlett kernel and fixed 120 one-minute lags. Gaps split
  series; autocovariances are never carried across a discontinuity. Report a
  normal-reference 95% interval and two-sided HAC Wald p-value.
- Primary family-wise correction: Holm step-down correction at alpha 0.05 over
  all 64 tests, including any constant feature test as p=1 rather than
  shrinking the family.
- Null control: 999 deterministic circular feature shifts (seed `20261027`)
  within each contiguous M1 segment, using the same shift for every feature
  and horizon within a segment. For each draw, offsets are restricted to at
  least 120 minutes in either circular direction; a maximum absolute
  correlation across the full 64-test family yields family-wise empirical
  shift p-values. Segments shorter than 241 rows are explicitly excluded from
  the shift control and the reported map statistics.
- Every result includes counts, segment exclusions, the fixed family and
  method identity. The label builder fingerprints only normalized training
  BID/ASK frames and records source-code/software provenance. No validation or
  test prices, features, labels, or metrics enter the map.

The HAC lag and circular-shift controls are prespecified approximations for
temporal dependence, not guarantees against all nonstationarity or researcher
degrees of freedom. Bar-derived midpoint values are not tick-level quotes.
This protocol does not authorize evaluating the real dataset.
