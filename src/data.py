"""Loading, cleaning, splitting, scaling and windowing (leak-free)."""
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

NUM_COLS = [
    "Usage_kWh",                              # target, always column 0
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
]


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw(path):
    """Read the CSV, parse dates (day first), sort, drop duplicate timestamps."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def build_features(df):
    """Numeric columns + cyclical time features. Column 0 is Usage_kWh."""
    cols = [c for c in NUM_COLS if c in df.columns]
    out = df[cols].astype("float64").copy()
    hour = df["date"].dt.hour + df["date"].dt.minute / 60
    dow = df["date"].dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype("float64")
    return out


def chronological_split(arr, train_frac, val_frac):
    """Split by time order, never shuffled."""
    n = len(arr)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return arr[:i1], arr[i1:i2], arr[i2:]


def make_windows(arr, input_len, output_len, stride):
    """arr: (T, F), target in column 0. Returns X (N, input_len, F), y (N, output_len)."""
    n = (len(arr) - input_len - output_len) // stride + 1
    if n <= 0:
        raise ValueError("Split too short for the requested window sizes.")
    idx = np.arange(n) * stride
    X = np.stack([arr[i:i + input_len] for i in idx]).astype("float32")
    y = np.stack([arr[i + input_len:i + input_len + output_len, 0] for i in idx]).astype("float32")
    return X, y


def build_datasets(cfg):
    """Full pipeline. Scaler is fitted on the training split only."""
    df = load_raw(cfg["data"]["raw_path"])
    feats = build_features(df)
    train, val, test = chronological_split(
        feats.values, cfg["data"]["train_frac"], cfg["data"]["val_frac"])

    scaler = StandardScaler().fit(train)          # train only -> no leakage
    train, val, test = (scaler.transform(a) for a in (train, val, test))

    w = cfg["window"]
    out = {
        "train": make_windows(train, w["input_len"], w["output_len"], w["stride_train"]),
        "val": make_windows(val, w["input_len"], w["output_len"], w["stride_eval"]),
        "test": make_windows(test, w["input_len"], w["output_len"], w["stride_eval"]),
        "scaler": scaler,
        "feature_names": list(feats.columns),
        # to convert scaled Usage_kWh back to kWh: y * target_std + target_mean
        "target_mean": float(scaler.mean_[0]),
        "target_std": float(scaler.scale_[0]),
    }
    return out
