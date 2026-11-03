
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

PIP = 0.0001
HORIZON = 15

df = pd.read_csv("eurusd_clean.csv", parse_dates=["time"])
df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

for col in ["open", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["time", "open", "close"]).reset_index(drop=True)

for period in [5, 15, 60]:
    df[f"return_{period}"] = df["close"].pct_change(period)

ema9 = df["close"].ewm(span=9, adjust=False).mean()
ema21 = df["close"].ewm(span=21, adjust=False).mean()
df["ema_gap"] = (ema9 - ema21) / df["close"]

delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df["volatility"] = df["return_5"].rolling(30).std()

minute = df["time"].dt.hour * 60 + df["time"].dt.minute
df["time_sin"] = np.sin(2 * np.pi * minute / 1440)
df["time_cos"] = np.cos(2 * np.pi * minute / 1440)

features = [
    "return_5", "return_15", "return_60", "ema_gap",
    "rsi", "volatility", "time_sin", "time_cos",
]
no_time_features = [f for f in features if f not in ["time_sin", "time_cos"]]

times = df["time"].to_numpy(dtype="datetime64[ns]")
opens = df["open"].to_numpy()
closes = df["close"].to_numpy()
target = np.full(len(df), np.nan)

for i in range(len(df) - 1):
    entry = i + 1
    if (times[entry] - times[i]) / np.timedelta64(1, "m") > 1.5:
        continue

    target_time = times[entry] + np.timedelta64(HORIZON, "m")
    exit_idx = np.searchsorted(times, target_time, side="left")
    if exit_idx >= len(df):
        continue

    gaps = np.diff(times[i:exit_idx + 1]) / np.timedelta64(1, "m")
    if len(gaps) and np.any(gaps > 1.5):
        continue

    move = closes[exit_idx] - opens[entry]
    if move != 0:
        target[i] = int(move > 0)

df["target"] = target
data = df.dropna(subset=features + ["target"]).copy()
data["target"] = data["target"].astype(int)

# Split by timestamp and purge 15 minutes at each boundary
# so labels cannot reach across a split.
t1 = data["time"].iloc[int(len(data) * 0.60)]
t2 = data["time"].iloc[int(len(data) * 0.80)]

train = data[data["time"] < t1 - pd.Timedelta(minutes=15)]
val = data[
    (data["time"] >= t1 + pd.Timedelta(minutes=15))
    & (data["time"] < t2 - pd.Timedelta(minutes=15))
]

def make_model():
    return XGBClassifier(
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

def evaluate(label, columns):
    model = make_model()
    model.fit(train[columns], train["target"])
    prob = model.predict_proba(val[columns])[:, 1]
    pred = (prob >= 0.5).astype(int)

    print(f"\n--- {label} ---")
    print(f"Validation samples: {len(val):,}")
    print(f"Accuracy: {accuracy_score(val['target'], pred):.2%}")
    print(f"ROC AUC: {roc_auc_score(val['target'], prob):.4f}")
    print(f"Brier score: {brier_score_loss(val['target'], prob):.4f}")

print(f"Training samples: {len(train):,}")
print(f"Validation samples: {len(val):,}")
print("Validation only; no final-test tuning.")

evaluate("ALL FEATURES", features)
evaluate("WITHOUT TIME OF DAY", no_time_features)
