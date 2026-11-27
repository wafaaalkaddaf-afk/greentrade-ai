# Member 4 - Experimentation Report

## 1. Role and Responsibilities
Member 4 is responsible for experiment tracking, reproducibility,
evaluation consistency, and comparison of model results.

## 2. Experiment Setup
- Dataset: UCI Steel Industry Energy Consumption.
- Forecasting task: 96 previous readings to predict the next 96 readings.
- Random seed: 42.
- Data split: chronological 70% training, 15% validation, 15% testing.
- Scaling: StandardScaler fitted on training data only.
- Current evaluation stride: 96 for validation and test.
- Primary metric: MAE.
- Secondary metrics: RMSE and R2.

## 3. Baseline Results

| Model | MAE (kWh) | RMSE (kWh) | R2 |
|---|---:|---:|---:|
| Seasonal Naive | 15.4552 | 29.0388 | 0.1072 |
| Last Value | 35.0723 | 46.2315 | -1.2631 |
| Ridge Regression | 13.2578 | 19.1908 | 0.6101 |

These are recovered historical validation results, not newly executed
experiments. The original run timestamps were not available.

## 4. Initial Findings
Ridge Regression achieved the lowest MAE and RMSE and the highest R2
among the three baseline models. It is therefore the strongest baseline
in the currently available results.

The Last Value baseline performed worst, with a negative R2.

## 5. Reproducibility and Logging
- Experiment records are stored in experiment_log.csv.
- Unified baseline results are stored in baseline_results.csv.
- Evaluation metric definitions are documented in metrics_guide.md.
- Shared experiment settings are documented in fixed_settings.md.
- The random seed is fixed at 42.

## 6. Protocol Alignment Issue
The current README and recovered baseline results use 11 features
and validation stride 96, resulting in 53 validation windows.

An earlier model-design discussion mentioned 12 features and validation
stride 1. The team must confirm the shared protocol before comparing
results from different models.

If the protocol changes, all models and baselines must be evaluated
using the same feature set, data split, and evaluation windows.

## 7. Next Steps
1. Confirm the feature count and validation stride with the team.
2. Record each model experiment and its hyperparameters.
3. Compare all models using the same validation protocol.
4. Select the best model using validation MAE.
5. Evaluate the selected model on the test set only for final reporting.
6. Document tuning and ablation experiments when completed.

## 8. Deliverable Status
- [x] Experiment log created.
- [x] Unified baseline results created.
- [x] Metrics guide created.
- [x] Fixed settings documented.
- [x] Experimentation report created.
- [ ] Team confirmation of the shared evaluation protocol.
- [ ] Additional model experiments and final test evaluation.
