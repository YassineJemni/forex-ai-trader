
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    brier_score_loss,
)

PIP = 0.0001
HORIZON_MINUTES = 15

df = pd.read_csv("eurusd_clean.csv", parse_dates=["time"])
df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

for col in ["open", "high", "low", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["time", "open", "close"]).reset_index(drop=True)

# Features from present and past prices only.
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

# Target: direction over 15 elapsed minutes, with gap filtering.
times = df["time"].to_numpy(dtype="datetime64[ns]")
opens = df["open"].to_numpy()
closes = df["close"].to_numpy()

target = np.full(len(df), np.nan)

for i in range(len(df) - 1):
    entry = i + 1
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
    if len(gaps) and np.any(gaps > 1.5):
        continue

    move = closes[exit_idx] - opens[entry]
    if move != 0:
        target[i] = int(move > 0)

df["target"] = target
data = df.dropna(subset=features + ["target"]).copy()
data["target"] = data["target"].astype(int)

# Chronological splits.
n = len(data)
a, b = int(n * 0.60), int(n * 0.80)
train = data.iloc[:a]
validation = data.iloc[a:b]
test = data.iloc[b:]

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,
    min_child_weight=20,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=4,
    random_state=42,
)

model.fit(train[features], train["target"])

def report(name, part):
    y = part["target"].to_numpy()
    p = model.predict_proba(part[features])[:, 1]
    pred = (p >= 0.5).astype(int)

    print(f"\n--- {name} ---")
    print(f"Samples: {len(part):,}")
    print(f"Majority baseline: {max(y.mean(), 1-y.mean()):.2%}")
    print(f"Accuracy: {accuracy_score(y, pred):.2%}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y, pred):.2%}")
    print(f"ROC AUC: {roc_auc_score(y, p):.4f}")
    print(f"Brier score: {brier_score_loss(y, p):.4f}")

print("EUR/USD XGBOOST EXPERIMENT")
print(f"Train: {train['time'].min()} to {train['time'].max()}")
print(f"Validation: {validation['time'].min()} to {validation['time'].max()}")
print(f"Test: {test['time'].min()} to {test['time'].max()}")

report("VALIDATION", validation)
report("FINAL TEST — DIAGNOSTIC ONLY", test)

print("\nFeature importance:")
importance = pd.Series(
    model.feature_importances_, index=features
).sort_values(ascending=False)
print(importance.round(4).to_string())

print("\nThis is a research model, not a live-trading forecast.")
