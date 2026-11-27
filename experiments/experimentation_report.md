
# GreenTrade AI - Experimentation Report

## 1. Objective
The objective of the experimentation phase is to evaluate baseline forecasting models
for predicting the next 24 hours of electricity consumption in steel plants.

## 2. Experimental Setup
- Dataset: Steel Industry Energy Consumption Dataset
- Target variable: Usage_kWh
- Input window: Previous 24 hours (96 time steps)
- Forecast horizon: Next 24 hours (96 time steps)
- Train/Validation/Test split: 70% / 15% / 15%
- Random seed: 42
- Evaluation split: Validation Set (non-overlapping 24-hour windows)
- Metrics calculated in original kWh units.
- Primary metric: MAE (kWh)
- Secondary metrics: RMSE and R²

## 3. Baseline Models and Results

| Model | MAE (kWh) | RMSE | R² |
|---|---:|---:|---:|
| Seasonal Naive | 15.455 | 29.039 | 0.107 |
| Last Value | 35.072 | 46.231 | -1.263 |
| Ridge Regression | 13.258 | 19.191 | 0.610 |

## 4. Results Analysis
Ridge Regression achieved the best baseline performance, with the lowest MAE
and RMSE and the highest R² score.

Seasonal Naive performed better than Last Value, while Last Value had the
weakest performance.

## 5. Conclusion
Ridge Regression is the strongest baseline model, achieving an MAE of
13.258 kWh. Neural network models should be compared against this baseline
to determine whether they improve forecasting performance.

## 6. Pending Experiments
RNN, LSTM, and GRU experiments will be added after the neural network
models and training procedures are completed by Member 3.
