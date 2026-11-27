# GreenTrade AI — Metrics Guide

All metrics are calculated on predictions after converting the scaled target back to its original unit (kWh), as implemented in `src/evaluate.py`.

## Primary metric: MAE (kWh)
Mean Absolute Error is the average absolute difference between actual and predicted consumption.

`MAE = mean(abs(y_true - y_pred))`

Lower is better. It is the project's primary metric and is directly interpretable in kWh.

## Secondary metric: RMSE
Root Mean Squared Error is the square root of the mean squared prediction error.

`RMSE = sqrt(mean((y_true - y_pred) ** 2))`

Lower is better. Large errors have a stronger influence on RMSE than on MAE.

## Secondary metric: R²
R-squared compares prediction error with the variance of the actual values.

`R2 = 1 - SS_res / SS_tot`

A value closer to 1 is generally better. 0 means no improvement over predicting the mean of the evaluated targets; a negative value means worse performance than that reference.

## Comparison rules
- Compare models on the same split (the supplied baseline results are validation results).
- Use the same target (`Usage_kWh`), preprocessing, windows, forecast horizon and evaluation timestamps.
- Metrics must be calculated in original kWh units, not standardized units.
- Use validation results for model selection. Keep the test split for final evaluation.
- Do not compare validation metrics with test metrics as if they were the same evaluation.
