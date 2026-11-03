
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FILE = "eurusd_clean.csv"
PIP = 0.0001
HORIZON = 15
COST_PIPS = 1.0
THRESHOLDS = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]

# Load and clean data.
df = pd.read_csv(FILE, parse_dates=["time"])
df = (
    df.sort_values("time")
    .drop_duplicates("time")
    .reset_index(drop=True)
)

for col in ["open", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["time", "open", "close"]).reset_index(drop=True)

# Features use current and past data only.
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

minute = df["time"].dt.hour * 60 + df["time"].dt.minute
df["time_sin"] = np.sin(2 * np.pi * minute / 1440)
df["time_cos"] = np.cos(2 * np.pi * minute / 1440)

features = [
    "return_5", "return_15", "return_60", "ema_gap",
    "rsi", "volatility", "time_sin", "time_cos",
]

# Build a target using elapsed minutes and reject market/data gaps.
times = df["time"].to_numpy(dtype="datetime64[ns]")
opens = df["open"].to_numpy()
closes = df["close"].to_numpy()

target = np.full(len(df), np.nan)
future_pips = np.full(len(df), np.nan)
exit_indices = np.full(len(df), -1, dtype=int)

for i in range(len(df) - 1):
    entry = i + 1
    if (times[entry] - times[i]) / np.timedelta64(1, "m") > 1.5:
        continue

    exit_time = times[entry] + np.timedelta64(HORIZON, "m")
    exit_idx = np.searchsorted(times, exit_time, side="left")
    if exit_idx >= len(df):
        continue

    segment = times[i:exit_idx + 1]
    gaps = np.diff(segment) / np.timedelta64(1, "m")
    if len(gaps) and np.any(gaps > 1.5):
        continue

    move = (closes[exit_idx] - opens[entry]) / PIP
    if move == 0:
        continue

    target[i] = int(move > 0)
    future_pips[i] = move
    exit_indices[i] = exit_idx

df["target"] = target
df["future_pips"] = future_pips
df["exit_idx"] = exit_indices

data = df.dropna(subset=features + ["target"]).copy()
data["target"] = data["target"].astype(int)

# Split chronologically: 60% train, 20% validation, 20% final test.
n = len(data)
a, b = int(n * 0.60), int(n * 0.80)
train, val, test = data.iloc[:a], data.iloc[a:b], data.iloc[b:]

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
model.fit(train[features], train["target"])

def get_probabilities(part):
    return model.predict_proba(part[features])[:, 1]

def choose_threshold(part, probabilities):
    # Greedy, non-overlapping trades. Threshold chosen on validation only.
    best = None
    for threshold in THRESHOLDS:
        selected_returns = []
        next_allowed_time = None

        for pos, (_, row) in enumerate(part.iterrows()):
            probability = probabilities[pos]
            direction = 1 if probability >= threshold else (
                -1 if probability <= 1 - threshold else 0
            )
            if direction == 0:
                continue

            entry_idx = row.name + 1
            exit_idx = int(row["exit_idx"])
            if exit_idx < 0:
                continue

            entry_time = times[entry_idx]
            if next_allowed_time is not None and entry_time < next_allowed_time:
                continue

            gross = direction * row["future_pips"]
            selected_returns.append(gross - COST_PIPS)
            next_allowed_time = times[exit_idx]

        if len(selected_returns) < 100:
            continue

        r = np.asarray(selected_returns)
        score = r.mean()
        if best is None or score > best["avg_net"]:
            best = {
                "threshold": threshold,
                "returns": r,
                "avg_net": score,
            }
    return best

val_prob = get_probabilities(val)
chosen = choose_threshold(val, val_prob)

if chosen is None:
    raise SystemExit(
        "No validation threshold produced 100 non-overlapping trades. "
        "Do not lower the bar based on test results."
    )

print("\n--- THRESHOLD SELECTED ON VALIDATION ---")
print(f"Threshold: {chosen['threshold']:.2f}")
print(f"Validation trades: {len(chosen['returns'])}")
print(f"Validation average net pips: {chosen['avg_net']:.4f}")

def evaluate_test(part, probabilities, threshold):
    returns = []
    next_allowed_time = None

    for pos, (_, row) in enumerate(part.iterrows()):
        probability = probabilities[pos]
        direction = 1 if probability >= threshold else (
            -1 if probability <= 1 - threshold else 0
        )
        if direction == 0:
            continue

        entry_idx = row.name + 1
        exit_idx = int(row["exit_idx"])
        if exit_idx < 0:
            continue

        entry_time = times[entry_idx]
        if next_allowed_time is not None and entry_time < next_allowed_time:
            continue

        returns.append(direction * row["future_pips"] - COST_PIPS)
        next_allowed_time = times[exit_idx]

    return np.asarray(returns)

test_prob = get_probabilities(test)
result = evaluate_test(test, test_prob, chosen["threshold"])

print("\n--- FINAL TEST: THRESHOLD FROZEN ---")
print(f"Threshold: {chosen['threshold']:.2f}")
print(f"Trades: {len(result)}")

if len(result):
    winners = result[result > 0]
    losers = result[result < 0]
    profit_factor = (
        winners.sum() / abs(losers.sum())
        if len(losers) else np.inf
    )
    equity = np.r_[0, np.cumsum(result)]
    drawdown = equity - np.maximum.accumulate(equity)

    print(f"Win rate: {(result > 0).mean():.2%}")
    print(f"Average net pips: {result.mean():.4f}")
    print(f"Total net pips: {result.sum():.2f}")
    print(f"Profit factor: {profit_factor:.3f}")
    print(f"Max drawdown in pips: {abs(drawdown.min()):.2f}")

print(
    "\nResearch only. Costs are assumptions; this simulation "
    "does not model full bid/ask execution or slippage."
)
