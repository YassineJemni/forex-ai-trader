
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    brier_score_loss,
)

PIP = 0.0001
HORIZON_MINUTES = 15

df = pd.read_csv("eurusd_clean.csv", parse_dates=["time"])
df = (
    df.sort_values("time")
    .drop_duplicates("time")
    .reset_index(drop=True)
)

for col in ["open", "high", "low", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["time", "open", "close"]).reset_index(drop=True)

# Features use only current and past information.
for period in [5, 15, 60]:
    df[f"return_{period}"] = df["close"].pct_change(period)

df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
df["ema_gap"] = (df["ema9"] - df["ema21"]) / df["close"]

delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df["rsi"] = 100 - 100 / (1 + rs)

df["volatility"] = df["return_5"].rolling(30).std()
minute_of_day = df["time"].dt.hour * 60 + df["time"].dt.minute
df["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
df["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

features = [
    "return_5", "return_15", "return_60",
    "ema_gap", "rsi", "volatility",
    "time_sin", "time_cos",
]

# Build the 15-minute target using elapsed time, not row count.
times = df["time"].to_numpy(dtype="datetime64[ns]")
opens = df["open"].to_numpy()
closes = df["close"].to_numpy()

targets = np.full(len(df), np.nan)
future_return_pips = np.full(len(df), np.nan)

for i in range(len(df) - 1):
    entry = i + 1

    # Signal is known at candle i close; entry is next candle open.
    if (times[entry] - times[i]) / np.timedelta64(1, "m") > 1.5:
        continue

    target_time = times[entry] + np.timedelta64(
        HORIZON_MINUTES, "m"
    )
    exit_idx = np.searchsorted(times, target_time, side="left")

    if exit_idx >= len(df):
        continue

    segment = times[i:exit_idx + 1]
    gaps = np.diff(segment) / np.timedelta64(1, "m")

    # Exclude trades crossing missing minutes or market closures.
    if len(gaps) and np.any(gaps > 1.5):
        continue

    move = (closes[exit_idx] - opens[entry]) / PIP

    # Exclude exact-flat outcomes for this first binary classifier.
    if move == 0:
        continue

    future_return_pips[i] = move
    targets[i] = int(move > 0)

df["target"] = targets
df["future_pips"] = future_return_pips

data = df.dropna(subset=features + ["target"]).copy()
data["target"] = data["target"].astype(int)

# Chronological split: 60% train, 20% validation, 20% final test.
n = len(data)
train_end = int(n * 0.60)
validation_end = int(n * 0.80)

train = data.iloc[:train_end]
validation = data.iloc[train_end:validation_end]
test = data.iloc[validation_end:]

X_train, y_train = train[features], train["target"]
X_val, y_val = validation[features], validation["target"]
X_test, y_test = test[features], test["target"]

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)

model.fit(X_train, y_train)

def report(name, X, y):
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    print(f"\n--- {name} ---")
    print(f"Samples: {len(y):,}")
    print(f"Majority-class baseline: "
          f"{max(y.mean(), 1 - y.mean()):.2%}")
    print(f"Accuracy: {accuracy_score(y, prediction):.2%}")
    print(f"Balanced accuracy: "
          f"{balanced_accuracy_score(y, prediction):.2%}")
    print(f"ROC AUC: {roc_auc_score(y, probability):.4f}")
    print(f"Brier score: {brier_score_loss(y, probability):.4f}")

print("EUR/USD 15-MINUTE ML EXPERIMENT")
print(f"Training: {train['time'].min()} to {train['time'].max()}")
print(f"Validation: {validation['time'].min()} to "
      f"{validation['time'].max()}")
print(f"Final test: {test['time'].min()} to {test['time'].max()}")

report("VALIDATION", X_val, y_val)
report("FINAL TEST", X_test, y_test)

# Economic evaluation: fixed confidence thresholds.
# These thresholds are diagnostic, not optimized for profit.
COST_PIPS = 1.0
THRESHOLDS = [0.50, 0.52, 0.55, 0.60]

test_probability = model.predict_proba(X_test)[:, 1]
actual_pips = test["future_pips"].to_numpy()

print("\n--- FINAL TEST: SIMULATED TRADE RESULTS ---")
print(f"Assumed round-trip cost: {COST_PIPS:.1f} pips")
print("Signals are evaluated at fixed thresholds.")

for threshold in THRESHOLDS:
    direction = np.zeros(len(test_probability), dtype=int)
    direction[test_probability >= threshold] = 1
    direction[test_probability <= 1 - threshold] = -1

    selected = direction != 0
    if not selected.any():
        print(f"\nThreshold {threshold:.2f}: no trades")
        continue

    net = (
        direction[selected] * actual_pips[selected]
        - COST_PIPS
    )

    winners = int((net > 0).sum())
    losers = int((net < 0).sum())
    gross_profit = net[net > 0].sum()
    gross_loss = abs(net[net < 0].sum())
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else np.inf
    )

    # Illustrative drawdown using the ordered sequence of trade returns.
    equity = np.r_[0, np.cumsum(net)]
    drawdown = equity - np.maximum.accumulate(equity)

    print(f"\nConfidence threshold: {threshold:.2f}")
    print(f"Trades: {len(net):,} ({selected.mean():.2%} of samples)")
    print(f"Winning trades: {winners:,}")
    print(f"Losing trades: {losers:,}")
    print(f"Average net pips/trade: {net.mean():.4f}")
    print(f"Total net pips: {net.sum():.2f}")
    print(f"Profit factor: {profit_factor:.3f}")
    print(f"Max drawdown in cumulative pips: {abs(drawdown.min()):.2f}")

print(
    "\nCaution: this is a preliminary simulation. "
    "Signals overlap in time, so these figures are not a valid "
    "live portfolio backtest. Execution costs are assumptions."
)
