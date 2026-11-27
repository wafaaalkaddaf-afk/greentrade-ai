"""Forecast evaluation helpers."""
import numpy as np


def _paired_arrays(y, predictions):
    y_array = np.asarray(y, dtype=np.float64).reshape(-1)
    prediction_array = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if y_array.size == 0:
        raise ValueError("y and predictions must contain at least one value.")
    if y_array.shape != prediction_array.shape:
        raise ValueError(
            "y and predictions must contain the same number of values "
            f"(got {y_array.size} and {prediction_array.size})."
        )
    if not np.isfinite(y_array).all() or not np.isfinite(prediction_array).all():
        raise ValueError("y and predictions must contain only finite numeric values.")
    return y_array, prediction_array


def mae(y, predictions):
    y, predictions = _paired_arrays(y, predictions)
    return float(np.mean(np.abs(y - predictions)))


def rmse(y, predictions):
    y, predictions = _paired_arrays(y, predictions)
    return float(np.sqrt(np.mean((y - predictions) ** 2)))


def r2(y, predictions):
    y, predictions = _paired_arrays(y, predictions)
    ss_res = float(np.sum((y - predictions) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def to_kwh(values_scaled, mean, std):
    if std <= 0:
        raise ValueError("std must be greater than zero.")
    return np.asarray(values_scaled, dtype=np.float64) * std + mean


def evaluate_predictions(y_kwh, predictions_kwh):
    """Calculate all reported metrics in original kWh units."""
    return {
        "MAE": mae(y_kwh, predictions_kwh),
        "RMSE": rmse(y_kwh, predictions_kwh),
        "R2": r2(y_kwh, predictions_kwh),
    }


def all_metrics(y_scaled, predictions_scaled, mean, std):
    """Backward-compatible helper for scaled targets and predictions."""
    y_kwh = to_kwh(y_scaled, mean, std)
    predictions_kwh = to_kwh(predictions_scaled, mean, std)
    return evaluate_predictions(y_kwh, predictions_kwh)


# ---------------------------------------------------------------------------
# Added by Evaluation & Deployment: comparison, split guard, error analysis.
# Everything above this line is unchanged.
# ---------------------------------------------------------------------------

def check_split(split, final_report=False):
    """Protect the test set: it is only allowed for the single final report."""
    split = str(split).lower()
    if split not in ("validation", "test"):
        raise ValueError("split must be 'validation' or 'test'.")
    if split == "test" and not final_report:
        raise ValueError(
            "The test split is reserved for the final report. Select models on "
            "validation MAE, then call with final_report=True once."
        )
    return split


def evaluate_model(model, X, y_scaled, mean, std, split="validation", final_report=False):
    """Predict with any model that has .predict(X) and return MAE/RMSE/R2 in kWh."""
    check_split(split, final_report)
    return all_metrics(y_scaled, model.predict(X), mean, std)


def mae_per_horizon(y_kwh, predictions_kwh):
    """MAE of each forecast step. Inputs (N, H) in kWh, returns (H,)."""
    y = np.asarray(y_kwh, dtype=np.float64)
    p = np.asarray(predictions_kwh, dtype=np.float64)
    if y.ndim != 2 or y.shape != p.shape:
        raise ValueError(f"Expected two equal 2-D arrays (N, H); got {y.shape} and {p.shape}.")
    return np.mean(np.abs(y - p), axis=0)


def skill_score(model_mae, reference_mae):
    """1 - MAE_model / MAE_reference. Positive means better than the reference."""
    if reference_mae <= 0:
        raise ValueError("reference_mae must be greater than zero.")
    return 1.0 - model_mae / reference_mae


def compare_models(y_kwh, predictions_by_model, reference):
    """Same targets for every model. Returns rows sorted by MAE (best first).

    predictions_by_model: {name: predictions in kWh, same shape as y_kwh}
    reference: name of the model used for the skill score (e.g. seasonal naive)
    """
    if reference not in predictions_by_model:
        raise ValueError(f"reference '{reference}' is not in predictions_by_model.")
    rows = []
    for name, preds in predictions_by_model.items():
        row = {"model": name, **evaluate_predictions(y_kwh, preds)}
        rows.append(row)
    ref_mae = next(r["MAE"] for r in rows if r["model"] == reference)
    for r in rows:
        r["skill_vs_" + "reference"] = skill_score(r["MAE"], ref_mae)
    return sorted(rows, key=lambda r: r["MAE"])


def to_markdown(rows):
    """Render compare_models() output as a Markdown table."""
    lines = ["| Model | MAE (kWh) | RMSE (kWh) | R2 | Skill vs reference |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['MAE']:.3f} | {r['RMSE']:.3f} | "
                     f"{r['R2']:.3f} | {r['skill_vs_reference']:+.3f} |")
    return "\n".join(lines)


def plot_mae_per_horizon(y_kwh, predictions_by_model, path=None, step_minutes=15):
    """Line plot: how the error grows across the 24-hour horizon."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, preds in predictions_by_model.items():
        m = mae_per_horizon(y_kwh, preds)
        ax.plot(np.arange(1, len(m) + 1) * step_minutes / 60, m, label=name)
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("MAE per forecast step")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_pred_vs_actual(y_kwh, predictions_kwh, window_index=0, title="", path=None,
                        step_minutes=15):
    """One validation day: actual vs predicted Usage_kWh."""
    import matplotlib.pyplot as plt
    y = np.asarray(y_kwh, dtype=np.float64)[window_index]
    p = np.asarray(predictions_kwh, dtype=np.float64)[window_index]
    hours = np.arange(1, len(y) + 1) * step_minutes / 60
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, y, label="Actual")
    ax.plot(hours, p, label="Predicted")
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("Usage_kWh")
    ax.set_title(title or f"Predicted vs actual (window {window_index})")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig