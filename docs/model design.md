# GreenTrade AI: Model & Task Design (Milestone 1.2)

**Course:** AIE231 Neural Networks | **Task type:** Time-series forecasting (multi-step regression)
**Dataset:** UCI Steel Industry Energy Consumption | **Target:** `Usage_kWh`

## 1. Task Definition

Steel plants consume large amounts of electricity, and the lack of an accurate forecast for the next day makes operational planning and energy-saving comparisons harder. The model takes the **last 24 hours** of readings and predicts the **next 24 hours**.

| Item | Decision | 
| ----- | ----- | 
| User | Plant operations planner / energy manager | 
| Target | `Usage_kWh` | 
| Task type | Direct multi-step forecasting (regression) | 
| Input | Last 96 readings (24 h) $\times$ $F$ features | 
| Output | The next 96 readings of `Usage_kWh` (24 h) | 
| Cost of errors | Under-forecasting leads to under-planned load and surprises at peak times. Over-forecasting leads to over-planning and wasted saving opportunities. Error in the original unit (kWh) is therefore what matters most. | 

## 2. Data Sequence Diagram

Readings arrive every 15 minutes, so one day = 96 readings.

```
Time  ────────────────────────────────────────────────────────────────►

   |<────── Input window: 96 steps (24 h) ──────>|<── Horizon: 96 steps (24 h) ──>|
   | x[t-95] ........................... x[t]    |  y[t+1] ............ y[t+96]  |
   |  (Usage_kWh + other features)               |  (Usage_kWh only)             |
                                                ^
                                          Forecast origin (t)
                       Everything left of t is known; everything right of t is unknown


Sliding window (stride = 1 during training):

 Sample 1 : [ 0 .. 95 ]  ──► [ 96 .. 191 ]
 Sample 2 : [ 1 .. 96 ]  ──► [ 97 .. 192 ]
 Sample 3 : [ 2 .. 97 ]  ──► [ 98 .. 193 ]
   ...

```

### Pipeline

```
Raw CSV ─► Clean ─► Chronological split ─► Fit scalers on TRAIN only ─► Windowing (inside each split) ─► Model
                    70% / 15% / 15%         (X scaler + y scaler)        X:(N,96,F)  y:(N,96)

```

## 3. Input / Output Shapes

|  | Shape | Note | 
| ----- | ----- | ----- | 
| `X` (per batch) | `(batch, 96, F)` | 96 time steps, $F$ = number of features | 
| `y` (per batch) | `(batch, 96)` | The next 96 values of `Usage_kWh` (scaled) | 
| Output layer | `Dense(96, activation="linear")` | Linear because this is regression and the range should not be constrained | 
| Loss | MSE (on the scaled target) | Per the spec: linear + MSE | 
| After prediction | Inverse-transform `y` | Metrics are computed in real kWh | 

### Proposed Features ($F = 12$, tentative)

| Group | Columns | Count | 
| ----- | ----- | ----- | 
| Numeric (continuous) | `Usage_kWh`, `Lagging_Current_Reactive.Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `CO2(tCO2)`, `Lagging_Current_Power_Factor`, `Leading_Current_Power_Factor` | 6 | 
| Time of day (from `NSM`) | $\sin(2\pi \cdot \text{NSM}/86400)$, $\cos(2\pi \cdot \text{NSM}/86400)$ | 2 | 
| Calendar | `is_weekend` (binary, from `WeekStatus`) | 1 | 
| Categorical | `Load_Type` (one-hot: Light / Medium / Maximum) | 3 | 
| **Total (Proposed)** |  | **12** | 

> **Note:** These 12 features are **Proposed Features** based on the UCI dataset schema. The final feature count ($F$) and exact column names will be locked only after EDA and the preprocessing pipeline are executed and verified by the team. Proposed ablations: without `CO2(tCO2)` and without `Load_Type` (see Section 9).

## 4. Why Window = 96 and Horizon = 96?

**Horizon = 96:** This is exactly the next 24 hours, which is what the project needs (full-day planning). A one-step forecast (15 minutes) is not useful for planning.

**Window = 96:**

* It covers **one full daily cycle** (plant operation follows a clear pattern during the day), so the model sees the whole previous day.

* A length of 96 suits LSTM/GRU. SimpleRNN is expected to struggle more (vanishing gradients), which is a finding to explain in the report, not a flaw.

* It yields a large number of samples (about 24 thousand for training), so there is little risk of overfitting from lack of data.

* Training time is small, which meets the 30-minute Colab limit comfortably.

**Why direct multi-output (96 outputs at once) rather than recursive?**

* Recursive forecasting (predict one step, feed it back as input) accumulates error over 96 steps.

* Direct forecasting produces all 96 values in one forward pass, and inference is simpler and faster.

* Downside: no explicit link between output steps. Seq2Seq could be tried as future work.

**Window ablation (the required architecture factor in tuning):** 48 / 96 / 192. (Optional extra: 672 = one week, but there will be fewer samples and longer training.)

## 5. Split and Windowing (No Leakage)

The data covers about one full year ($35,040 = 365 \times 96$). The split is **chronological, never shuffled**.

| Split | Share | Readings (approx.) | Windows (stride = 1) | 
| ----- | ----- | ----- | ----- | 
| Train | 70% | 24,528 | 24,337 | 
| Validation | 15% | 5,256 | 5,065 | 
| Test | 15% | 5,256 | 5,065 | 

(Windows = $\text{readings} - 192 + 1$, because each sample needs 96 input readings plus 96 target readings.)

Rules:

1. Split the series first, then do the windowing **inside each split separately**. No window crosses the boundary between splits, and no validation target appears in training.

2. **Scaler alignment:** Fit scaling only on the continuous numeric columns (currently 6 in the Proposed Features) of the training set, leaving cyclical `sin`/`cos` and binary/one-hot columns untouched. Ensure `y_scaler` uses the exact same scaling parameters as the `Usage_kWh` feature column.

3. The test set is used once, at the end. Model selection uses the validation set only.

4. For evaluation plots, use windows with stride = 96 (non-overlapping days); compute metrics over all windows.

5. In the EDA, verify that the timestamps are continuous at 15-minute spacing with no gaps, because windowing assumes this.

**Limitation to report:** The data covers a single year, so the test set is the last \~55 days (end of year) and the model has not seen that season during training. This should be stated as a seasonality limitation in the report.

## 6. Models

All recurrent models follow the same template so the comparison is fair:

```
Input (96, F)
  └─► RNN layer (SimpleRNN | LSTM | GRU), units = 64, return_sequences = False  ──► Output (64,)
        └─► Dropout(0.2)                                                         ──► Output (64,)
              └─► Dense(96, activation="linear")                                 ──► Output (96,)

```

* **Activations:** Recurrent layers use their default `tanh` activation (plus `sigmoid` gates in LSTM/GRU) because `tanh` is zero-centered and bounded in $[-1, 1]$, helping stabilize hidden-state magnitudes across 96 time steps. The output layer uses `linear` activation so regression predictions are unconstrained.

* **Reference optimizer:** Adam, $\text{lr} = 10^{-3}$, `clipnorm = 1.0` (handles exploding gradients).

* **Regularization:** `Dropout(0.2)` + `EarlyStopping` on `val_loss` (patience = 10).

* **Reference batch size:** 64.

* **Core comparison:** SimpleRNN $\rightarrow$ LSTM $\rightarrow$ GRU (Models A, B, C per the spec).

### Hand-computed parameter counts (based on $F = 12$ Proposed Features, $\text{units} = 64$)

| Layer | Formula | Calculation | Params | 
| ----- | ----- | ----- | ----- | 
| `SimpleRNN(64)` | $\text{units} \times (F + \text{units} + 1)$ | $64 \times (12 + 64 + 1) = 64 \times 77$ | 4,928 | 
| `LSTM(64)` | $4 \times \text{units} \times (F + \text{units} + 1)$ | $4 \times 4,928$ | 19,712 | 
| `GRU(64)` | $3 \times \text{units} \times (F + \text{units} + 2)$ | $3 \times 64 \times 78$ | 14,976 | 
| `Dense(96)` | $64 \times 96 + 96$ | $6,144 + 96$ | 6,240 | 

| Model | Total (for $F = 12$) | 
| ----- | ----- | 
| SimpleRNN + Dense | $4,928 + 6,240 =$ **11,168** | 
| LSTM + Dense | $19,712 + 6,240 =$ **25,952** | 
| GRU + Dense | $14,976 + 6,240 =$ **21,216** | 

> These numbers are calculated using the 12 Proposed Features and will be updated if $F$ changes after preprocessing. They must match `model.summary()` once the models are built. (Dropout has no parameters.)

The "LSTM vs GRU at equal parameter count" ablation will need different unit counts (GRU slightly larger).

## 7. Baselines (coordinated with Member 1)

| \# | Baseline | Definition | Parameters (for $F = 12$) | 
| ----- | ----- | ----- | ----- | 
| 1 | **Trivial Naive ("tomorrow = today")** | Repeat the last 96 readings of `Usage_kWh` as the forecast for the next 96: $\hat{y}[t+k] = y[t+k-96]$ | 0 | 
| 2 | **Simple Learned (Ridge Regression)** | L2-regularized linear regression on the flattened input window $(96 \times F = 1,152) \rightarrow 96$ | $1,152 \times 96 + 96 = 110,688$ | 
| Optional | Weekly naive | Repeat the same weekday from last week (needs a 672-reading history) | 0 | 

All models use the same splits and the same metrics, and every model is compared against both official baselines.

## 8. Metrics

| Metric | Use | Why | 
| ----- | ----- | ----- | 
| **MAE (kWh)** | **Primary** (model selection on validation) | Easy to interpret in the original unit and less sensitive to outliers | 
| **RMSE (kWh)** | Secondary | Penalizes large errors (peaks), which matter for planning | 
| MAE per horizon step | Plot | Shows how error grows across the 24-hour horizon | 
| Skill score vs naive | $1 - \text{MAE}_{\text{model}} / \text{MAE}_{\text{naive}}$ | Shows the model beats a non-trivial baseline | 
| Predicted vs actual | Plot | Required by the spec | 

**Why not MAPE?** Night-time consumption is close to zero, so MAPE produces huge, unstable values.

**Training loss:** MSE on the scaled target (per the spec). Selection and reporting use MAE/RMSE in kWh after the inverse transform. Huber loss can be tried as an ablation.

## 9. Leakage Review

The forecast origin is $t$. Anything known up to $t$ ($\le t$) is allowed; anything after it ($> t$) is forbidden unless strictly deterministic calendar metadata.

| Item | Verdict | Reason | 
| ----- | ----- | ----- | 
| `Usage_kWh` inside the window ($\le t$) | Allowed | Historical consumption up to forecast origin $t$ | 
| `Usage_kWh` after $t$ as input | Forbidden | That is the target itself | 
| `CO2(tCO2)` | Allowed only inside the window ($\le t$) | Nearly proportional to `Usage_kWh`, so future values would be direct leakage. Inside the window it is valid past context, and we will run an ablation without it | 
| `Load_Type` (Light/Medium/Maximum) | Allowed only inside the window ($\le t$) | Derived from the instantaneous load level, so using it after $t$ would leak the target | 
| Reactive power / power factor | Inside the window only ($\le t$) | Instantaneous sensor readings known only up to $t$; future values are unavailable at forecast time | 
| `NSM`, `WeekStatus`, `is_weekend` | Allowed (sliced $\le t$ in v1) | In our current input tensor `(batch, 96, F)`, all features are strictly sliced $\le t$. Because calendar values are deterministic, passing future `is_weekend` flags ($t+1 \dots t+96$) is leak-free and can be tested as an enhancement for weekday/weekend transitions | 
| Scalers | Only if fit on train | Prevents distribution leakage from validation/test | 
| Windowing across split boundaries | Forbidden | Leaks target readings from one split into another | 
| Shuffling before the split | Forbidden | Breaks temporal order and causes future-to-past leakage | 
