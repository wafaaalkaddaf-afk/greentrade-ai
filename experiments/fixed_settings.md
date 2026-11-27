# Fixed Experiment Settings

## Dataset
- Dataset: UCI Steel Industry Energy Consumption.
- Total records: 35,040.
- Sampling interval: 15 minutes.
- Target variable: Usage_kWh.

## Forecasting Task
- Input window: 96 previous readings.
- Forecast horizon: 96 future readings.
- Input shape: (samples, 96, 11).
- Output shape: (samples, 96).

## Features
- Total features: 11.
- Six numeric features, including Usage_kWh and electrical measurements.
- Five time features: hour sin/cos, day-of-week sin/cos, and weekend flag.

## Data Preparation
- Chronological split: 70% training, 15% validation, 15% testing.
- No random shuffling.
- StandardScaler is fitted on the training data only.
- Random seed: 42.

## Window Stride
- Training stride: 1.
- Validation stride: 96.
- Test stride: 96.

## Evaluation
- Primary metric: MAE (kWh).
- Secondary metrics: RMSE (kWh) and R2.
- Metrics are calculated in the original kWh scale.
- Validation is used for model selection.
- Test data is reserved for final evaluation.

## Baseline Models
1. Seasonal Naive: tomorrow's forecast repeats today's readings.
2. Last Value: repeats the last observed value.
3. Ridge Regression: linear regression with L2 regularization.

## Important Alignment Note
An earlier model-design discussion mentioned 12 features and validation
with stride 1. The current project README and recovered baseline results
use 11 features and validation stride 96 (53 validation windows).

The team should confirm one shared protocol before comparing all models.
If the feature count or evaluation stride changes, all models and baselines
must be evaluated again using the same settings.

Model-specific hyperparameters should be recorded separately for each run.
