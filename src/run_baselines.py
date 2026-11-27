"""Evaluate baselines on validation and log to experiments/experiment_log.csv.

Usage: python -m src.run_baselines --author "Your Name"
"""
import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np

from src.data import build_datasets, load_config
from src.evaluate import all_metrics
from src.models import LastValue, RidgeBaseline, SeasonalNaive

LOG = Path("experiments/experiment_log.csv")
FIELDS = [
    "run_id", "timestamp_utc", "author", "experiment_type", "model", "split",
    "seed", "input_steps", "forecast_steps", "train_stride",
    "evaluation_stride", "feature_count", "scaler", "scaler_fit", "loss",
    "MAE_kWh", "RMSE_kWh", "R2", "status", "notes",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args()

    cfg = load_config(a.config)
    np.random.seed(cfg["seed"])
    d = build_datasets(cfg)
    Xtr, ytr = d["train"]
    Xva, yva = d["val"]
    print(f"train {Xtr.shape}  val {Xva.shape}  features: {d['feature_names']}")

    b = cfg["baselines"]
    models = [SeasonalNaive(), LastValue(cfg["window"]["output_len"]),
              RidgeBaseline(b["ridge_alpha"], b["ridge_train_subsample"])]

    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    n_prev = 0 if new else sum(1 for _ in open(LOG)) - 1
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        print(f"\n{'Model':38s} {'MAE':>8s} {'RMSE':>8s} {'R2':>7s}")
        for i, m in enumerate(models, 1):
            m.fit(Xtr, ytr)
            r = all_metrics(yva, m.predict(Xva), d["target_mean"], d["target_std"])
            print(f"{m.name:38s} {r['MAE']:8.3f} {r['RMSE']:8.3f} {r['R2']:7.3f}")
            w.writerow({
                "run_id": f"B{n_prev + i:03d}",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "author": a.author,
                "experiment_type": "baseline",
                "model": m.name,
                "split": "validation",
                "seed": cfg["seed"],
                "input_steps": cfg["window"]["input_len"],
                "forecast_steps": cfg["window"]["output_len"],
                "train_stride": cfg["window"]["stride_train"],
                "evaluation_stride": cfg["window"]["stride_eval"],
                "feature_count": len(d["feature_names"]),
                "scaler": "StandardScaler",
                "scaler_fit": "training_only",
                "loss": "N/A",
                "MAE_kWh": round(r["MAE"], 4),
                "RMSE_kWh": round(r["RMSE"], 4),
                "R2": round(r["R2"], 4),
                "status": "completed",
                "notes": "Baseline evaluated on validation only; test set reserved.",
            })
    print(f"\nLogged to {LOG}")


if __name__ == "__main__":
    main()
