# GreenTrade AI — Shared Fixed Settings

The following values are taken from the team's supplied `configs/config.yaml` and project code.

| Setting | Fixed value |
|---|---|
| Random seed | `42` |
| Dataset path | `data/raw/Steel_industry_data.csv` |
| Target | `Usage_kWh` |
| Train / validation / test | `70% / 15% / 15%` |
| Split method | Chronological; no shuffle |
| Input window | `96` readings (24 hours) |
| Forecast horizon | `96` readings (next 24 hours) |
| Sampling interval | 15 minutes |
| Training window stride | `1` |
| Validation / test window stride | `96` (non-overlapping windows) |
| Scaler | `StandardScaler`, fitted on training split only |
| Primary metric | MAE, reported in original kWh |
| Secondary metrics | RMSE and R² |
| Ridge alpha | `10.0` |
| Ridge training subsample | `4` |

## Feature pipeline
- Numeric columns: `Usage_kWh`, `Lagging_Current_Reactive.Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `CO2(tCO2)`, `Lagging_Current_Power_Factor`, `Leading_Current_Power_Factor`.
- Time features: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`.
- Total input features: 11.
- Dates are parsed day-first, rows sorted chronologically, duplicate timestamps dropped.
- Target is column 0 in the feature matrix.

## Baseline evaluation protocol
- Fit baselines on the training windows.
- Report baseline metrics on the validation split.
- Convert scaled target values back to kWh before calculating metrics.
- Preserve the test split for final evaluation.

## Reproducibility notes
- The configured seed is 42; `src/run_baselines.py` applies `np.random.seed(cfg["seed"])`.
- Record author, timestamp, model, split and seed for each baseline run.
- The supplied logger records baseline validation runs in `experiments/experiment_log.csv`.
- For neural-network runs, also record framework-level seeds and any relevant deterministic settings when the training code is implemented; these are not established by the supplied baseline runner.
