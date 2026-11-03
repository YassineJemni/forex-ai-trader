# Forex AI Trader MVP

The local dashboard uses a read-only MetaTrader 5 adapter, scans configured
symbols/timeframes, calculates features with `src.research.features`, and logs
scan/prediction records in a local SQLite database. There is no order API in
the adapter or application.

## Start

On the Windows computer that runs MT5, install project dependencies and the
optional terminal bridge:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-mt5.txt
```

Open MetaTrader 5, sign in to the intended account, and leave the terminal
running. The dashboard initializes the local terminal connection when FETCH is
pressed. If multiple terminals are installed, set `MT5_TERMINAL_PATH` to the
terminal executable, for example:

```powershell
$env:MT5_TERMINAL_PATH = 'C:\Program Files\MetaTrader 5\terminal64.exe'
```

Start the app from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.trader.app
```

Open `http://127.0.0.1:8765`. Keep it bound to localhost unless protected by a
trusted reverse proxy and authentication. The service has no authentication.

## Configuration

Environment variables:

- `FOREX_PAIRS`: comma-separated MT5 symbols; defaults to seven major pairs.
- `FOREX_TIMEFRAMES`: `M1`, `M5`, and/or `M15`; defaults to all three.
- `FOREX_BARS_PER_SCAN`: historical completed bars per pair/timeframe; min 260.
- `FOREX_ACCOUNT_BALANCE`, `FOREX_ACCOUNT_CURRENCY`.
- `FOREX_RISK_PERCENT`, `FOREX_MAX_OPEN_RISK_PERCENT`.
- `FOREX_MAX_QUOTE_AGE_SECONDS`.
- `MT5_TERMINAL_PATH`.
- `MT5_SERVER_TIMEZONE`: IANA timezone used by the broker server for MT5. The current MetaQuotes-Demo configuration defaults to `Etc/GMT-2` based on live tick/candle comparison; recheck and override it if the broker changes its server offset, especially around daylight-saving changes.
  tick/bar timestamps; defaults to `UTC` as specified by MetaQuotes. If live
  timestamps appear future-dated, verify the server's convention independently
  before setting this (for example, `Etc/GMT-2` means UTC+2 year-round).

MT5 symbol names must match the broker's Market Watch names (some brokers add
suffixes). Adjust `FOREX_PAIRS` accordingly. Symbols are selected in Market
Watch by the MT5 bridge; no orders are submitted.

## FETCH and decision behavior

FETCH connects to MT5, checks fresh bid/ask ticks, downloads completed bars,
rejects invalid/stale inputs, and calculates the existing causal research
features. The paper-only market-intelligence coordinator uses one UTC as-of
cutoff for all markets and only higher-timeframe candles already closed at a
signal's decision time. Its evidence gate requires exact pair/timeframe
validation plus pair/timeframe stability evidence; the available Mission 9
report does not provide that stability certification, so this coordinator
fails closed rather than treating pooled screening as pair-specific evidence.
It then applies only an explicitly promoted model whose metadata
matches the scanned timeframe. With no promoted model, it displays market
observations and returns **NO TRADE**; it does not invent probabilities.

The training command builds an inert candidate artifact using the existing
feature builder, exact elapsed-time outcomes, chronological splits, and a
purge around split boundaries:

```powershell
.\.venv\Scripts\python.exe -m src.trader.training --data eurusd_clean.csv --output models/candidates/eurusd-m1.joblib
```

Training never activates a candidate. Promotion requires independent later
candles, enough labeled observations, AUC above chance, Brier score better
than the training-prevalence baseline, expected calibration error no greater
than 5 percentage points on fixed bins, sufficient fixed-threshold trades,
positive expectancy after the supplied assumed cost, and improvement over an
incumbent model when one exists. Call `promote_candidate()` explicitly with
the independent data and an approval identity. Promotion gates are research
controls, not a profitability claim; cost assumptions must be validated for
the intended venue.
Promoted artifacts are version-stamped and retained under `models/versions/`;
`models/current.joblib` is the active pointer.

Position sizing requires a currency-conversion rate; missing conversion blocks
the plan. While a model is active, any existing MT5 position blocks new
recommendations because aggregate open risk is not yet fully calculated. The
local history database records scans and model predictions; outcome labels
can be entered from the history table or through `POST /api/outcomes/{prediction_id}`
as an observed forward return in pips (positive for an upward move). The
dashboard performance card shows descriptive accuracy, Brier score, and mean
observed directional return; it does not subtract costs or imply profitability.

For isolated MT5 history collection, multi-pair candidate training, purged
walk-forward validation, independent model promotion, and rollback, see
[the multi-pair training engine guide](multi_pair_training_engine.md).

For controlled retraining from those labels, schedule this command with Windows
Task Scheduler after enough outcomes have accumulated:

```powershell
.\.venv\Scripts\python.exe -m src.trader.retraining --symbol EURUSD --timeframe M1 --minimum-samples 1000
```

It creates an inert candidate only; repeated runs with no new labeled data are
skipped, and no scheduled run can promote a model.

## Current state

No promoted model artifact is included in the repository. Therefore the
working default is a multi-pair read-only scan with diagnostics and **NO
TRADE**. Historical research results do not qualify as a validated inference
model.
