# Evaluation Metrics Guide

All evaluation metrics are calculated on the original Usage_kWh scale.

## 1. MAE (Mean Absolute Error)
MAE is the average absolute difference between actual and predicted values.

- Unit: kWh
- Lower is better.
- Primary metric used to compare forecasting models.

## 2. RMSE (Root Mean Squared Error)
RMSE is the square root of the average squared prediction errors.

- Unit: kWh
- Lower is better.
- Penalizes large prediction errors more than MAE.

## 3. R2 (Coefficient of Determination)
R2 measures how well the model explains the variation in actual values.

- Best possible value: 1.0
- Value of 0: equivalent to predicting the mean.
- Negative value: worse than predicting the mean.
- Higher is better.

## Comparison Rules
- Compare models using the same validation or test samples.
- Use the same target values and evaluation stride.
- Calculate metrics in original kWh, not scaled values.
- Use validation results for model selection.
- Keep the test set for final evaluation only.
