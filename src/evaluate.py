"""Metrics, always computed in original units (kWh)."""
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot)


def to_kwh(y_scaled, mean, std):
    return y_scaled * std + mean


def all_metrics(y_scaled, p_scaled, mean, std):
    y, p = to_kwh(y_scaled, mean, std), to_kwh(p_scaled, mean, std)
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "R2": r2(y, p)}
