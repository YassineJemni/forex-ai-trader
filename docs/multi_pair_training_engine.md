# Multi-pair training engine

This pipeline collects completed MT5 candles into its own Parquet directory,
builds causal features and exact-bar labels, evaluates shared and pair-specific
candidates, and leaves all candidates inert until independent later data and
an explicit promotion command pass the validation gates.

## Collect candles

Open the intended MT5 terminal and run from the repository root. Set the server
timezone only after verifying the broker's timestamp convention. On the
MetaQuotes-Demo terminal observed during the timestamp investigation, the
measured current offset was UTC+2:

```powershell
$env:MT5_SERVER_TIMEZONE = 'Etc/GMT-2'
.\.venv\Scripts\python.exe -m src.trader.data_collection `
  --pairs EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD `
  --timeframes M5,M15,H1,H4 --bars 10000 `
  --data-dir market-data/mt5-history
```

If the broker adds symbol suffixes and automatic resolution is ambiguous, pass
a JSON mapping file with `--symbol-map broker-symbols.json`, for example:

```json
{"EURUSD": "EURUSD.a", "GBPUSD": "GBPUSD.a"}
```

Each pair/timeframe gets a Parquet file and a provenance JSON sidecar. Updates
merge overlapping recent windows, retain old rows, and record replaced overlap
counts and detected gaps. Missing intervals are never filled. Invalid OHLC or
timestamps reject that update and leave the previous file intact. MT5 history
availability still depends on terminal settings and broker retention.

Multi-pair training applies the existing research-data filter to EURUSD and
USDJPY H4 files: EURUSD observations before 1999 are excluded, and USDJPY
observations before 1999 are quarantined. Original collected Parquet files are
preserved; filtered derivatives and quarantine files retain source and content
fingerprints in provenance sidecars.

## Train candidates

Supply an explicit total round-trip cost in pips. Include your estimated spread,
commission and slippage in that assumption; the report records the assumption.
It is not automatically inferred from broker spread fields.

```powershell
.\.venv\Scripts\python.exe -m src.trader.multi_training `
  --data-dir market-data/mt5-history `
  --output-dir models/candidates `
  --pairs EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD `
  --timeframes M5,M15,H1,H4 --horizon-bars 3 `
  --round-trip-cost-pips 2.0
```

For each adequately sampled timeframe, training creates logistic-regression
and histogram-gradient-boosting candidates for a pooled shared model and each
pair-specific model. Features include returns, moving-average relationships,
RSI, MACD, volatility/ATR, candle structure, session, a volatility regime,
same-time peer returns, pair identity and timeframe identity. Labels enter at
the next bar open and exit at the configured horizon close; labels crossing a
missing bar are discarded. Walk-forward folds are chronological and purge any
training label whose outcome reaches the validation start. The final 15% of
available timestamps is excluded from fitting and walk-forward model selection
and is reported as a one-time candidate holdout.

Metrics include AUC, Brier score versus prevalence, calibration error,
thresholded precision/recall and directional accuracy, plus a cost-adjusted
fixed-notional trade simulation. The simulation is intentionally limited: it
uses a single explicit pip cost, fixed thresholds and no overlapping trade per
pair; financing, market impact, dynamic spreads and account-level concurrent
position sizing are not modeled. A candidate failing any gate remains stored
and cannot be promoted. Candidate metadata and paths are indexed in
`models/candidate_registry.json`.

## Explicit promotion and rollback

Promotion uses data timestamped after the candidate's final holdout. Collect
new bars after training, then run:

```powershell
.\.venv\Scripts\python.exe -m src.trader.model_management promote `
  --candidate models/candidates/CANDIDATE_FILE.joblib `
  --data-dir market-data/mt5-history `
  --round-trip-cost-pips 2.0 --approved-by "local operator"
```

For pair-specific candidates, the independent confirmation must improve on an
active shared model for that timeframe when one exists. No candidate is
promoted automatically. Active model pointers are in
`models/active_models.json`; promoted artifacts are retained in
`models/versions/`. Roll back one slot by its model key and archived version:

```powershell
.\.venv\Scripts\python.exe -m src.trader.model_management rollback `
  --model-key pair:EURUSD:M5 --version VERSION_FROM_METADATA
```

FETCH selects a promoted pair-specific model first and otherwise falls back to
the promoted shared model for the scanned timeframe. An absent or unvalidated
model continues to produce **NO TRADE** without a probability.
