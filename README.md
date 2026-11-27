# GreenTrade AI: 24-Hour Electricity Forecasting for Steel Plants

**AIE231 Neural Networks** course project, Alamein International University.
Framework: TensorFlow/Keras. Track: sequence forecasting (SimpleRNN, LSTM, GRU).

> **Status:** Week 9-10 (problem framing, EDA, baselines). See
> [Project status](#11-project-status) for what is done and what is next.


**Contents:**
[1. Overview](#1-overview-in-plain-words) ·
[2. Problem](#2-the-problem) ·
[3. Solution](#3-the-solution) ·
[4. Scope](#4-scope-of-this-course-project) ·
[5. Task](#5-task-definition) ·
[6. Dataset](#6-dataset) ·
[7. Method](#7-method-how-the-project-works) ·
[8. Repository](#8-repository-structure) ·
[9. Run it](#9-setup-and-run) ·
[10. Results](#10-results) ·
[11. Status](#11-project-status) ·
[12. Team](#12-team-contributions) ·
[13. Limitations](#13-limitations-and-ethics) ·
[14. AI tools](#14-ai-tools-statement) ·
[15. References](#15-references)

---

## 1. Overview in plain words

Steel plants use a huge amount of electricity, and the amount changes every hour
depending on furnaces, production lines and the day of the week. Today a plant
only learns its consumption **after** it has happened, when the bill arrives.

**GreenTrade AI** is a decision-support concept that lets a plant look one day
ahead. A neural network reads the **last 24 hours** of meter readings and
predicts the **next 24 hours** of electricity use. With that forecast, managers
can plan production, avoid expensive demand peaks and test saving or solar ideas
before spending money.

The full vision also covers carbon analysis and an export (CBAM) cost estimate
for plants that sell to the European Union. **This course project builds the
forecasting core**; the other modules are described as future work.

```
 last 24 h of readings        neural network         next 24 h forecast
 (96 readings, every     -->  SimpleRNN / LSTM  -->  (96 values of Usage_kWh)
  15 minutes)                 / GRU
                                                       |
                                                       v
                                    peak hours, saving and solar scenarios,
                                    cost and carbon estimates (future work)
```

## 2. The problem

A plant can read yesterday's consumption from its meters, but it cannot see
tomorrow's. This causes three connected problems.

| Challenge | What happens | Impact |
|---|---|---|
| Operations and cost | Consumption changes hour by hour and between workdays and weekends, depending on furnace and production-line loads | Unexpected demand peaks raise the bill; saving or solar decisions are taken without being tested first |
| Environment | Grid electricity and burned fuel emit CO2 | The plant cannot link its energy plans to their carbon effect |
| Export and CBAM | The EU Carbon Border Adjustment Mechanism has been in its definitive phase since 1 January 2026. When verified emissions data are missing, default values with a mark-up are applied | Higher cost per tonne, and no early estimate to price a contract |

What a wrong forecast costs: if the forecast is too low, the plant is surprised by demand peaks and pays a higher bill; if it is too high, production or saving plans are made for consumption that never happens. For this reason the forecast is reported with its error in kWh (MAE), so managers can see how far to trust it. The cost effects are potential, not measured results.
## 3. The solution

The plant moves from **reacting** (waiting for the bill or a carbon charge) to
**planning ahead** (knowing what will happen and deciding on that basis).

**Before production (planning and contracting)**

1. The plant provides the last 24 hours of readings (96 readings, one every 15 minutes).
2. A neural network forecasts the next 24 hours (96 readings) and highlights peak hours.
3. Management tests scenarios before spending money: an energy-saving percentage, a solar share, or both.
4. A preliminary export-cost estimate supports pricing of the shipment.

**After production (actual shipment)**

1. The plant enters the actual readings.
2. The system compares them with the estimate and helps prepare the file for the accredited verifier.

The system is a decision-support tool, **not** a legal certificate.

## 4. Scope of this course project

| Module | Status in this project |
|---|---|
| Electricity forecasting (SimpleRNN, LSTM, GRU vs baselines) | **Implemented (core of the course)** |
| Saving simulator (forecast x percentage x tariff entered by the user) | Planned for the demo if time allows |
| Carbon analysis (CO2 intensity from the dataset's CO2 column) | Planned if time allows |
| Renewable-energy simulator | Future work |
| CBAM calculator and final report generator | Future work (the dataset has no production tonnage, fuel or CN-code data) |

Note: for iron and steel, CBAM counts direct emissions only, so electricity
forecasting supports cost, peak planning and the plant's overall footprint
rather than entering the CBAM charge directly.

## 5. Task definition

| Item | Definition |
|---|---|
| Task | Multi-step time-series forecasting (regression) |
| Input | `(96, F)`: last 24 h of features at 15-minute resolution (`F` = 11 features, see section 7) |
| Output | `(96,)`: next 24 h of `Usage_kWh` |
| Loss | MSE |
| **Primary metric** | **MAE in kWh (original units)** |
| Secondary metrics | RMSE, R2 |
| Split | Chronological 70 / 15 / 15, never shuffled |
| Random seed | 42 |

All values are set in one place: [`configs/config.yaml`](configs/config.yaml).

**Why MAE:** it is expressed directly in kWh, so the energy engineer reads it as
the average error in real units, and it is less dominated by rare spikes than
RMSE. RMSE is reported to expose large peak errors.

## 6. Dataset

- **Steel Industry Energy Consumption**, UCI Machine Learning Repository (id 851)
- 35,040 rows, 15-minute readings (one full year), one small steel plant (DAEWOO Steel, South Korea)
- Licence: CC BY 4.0
- Citation: V E, S., Shin, C., & Cho, Y. (2021). *Steel Industry Energy Consumption* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52G8C
- Dataset page: https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption
Main columns used:

| Column | Meaning |
|---|---|
| `date` | Timestamp of the reading (day/month/year hour:minute) |
| `Usage_kWh` | Electricity used in the 15-minute interval (**the target**) |
| `Lagging_Current_Reactive.Power_kVarh`, `Leading_Current_Reactive_Power_kVarh` | Reactive power |
| `Lagging_Current_Power_Factor`, `Leading_Current_Power_Factor` | Power factor |
| `CO2(tCO2)` | CO2 emitted in the interval |

**Why this dataset meets the course criteria**

| Criterion | How the dataset meets it |
|---|---|
| Size | Sequence data with 35,040 time steps (minimum is 10,000) |
| Task difficulty | Forecasting with a non-trivial baseline to beat: "tomorrow = today" (see Results) |
| Real-world mess | Mixed feature types (numeric plus text columns such as `WeekStatus`, `Day_of_week`, `Load_Type`); duplicates and missing values are checked in the EDA |
| Not a toy | Real industrial smart-meter data, not a tutorial dataset |
| Licence | CC BY 4.0, cited above |
| Feasibility | Baselines run in seconds; neural models are designed to train in under 30 minutes on a free Colab GPU |

The raw data are **not stored in git**. Download them with:

```bash
python data/download_data.py
```

## 7. Method: how the project works

1. **Load and clean** (`src/data.py`): parse dates, sort by time, remove duplicate timestamps.
2. **Features:** the 6 numeric columns above plus 5 time features (hour and day-of-week as sine/cosine, and a weekend flag), so `F = 11`.
3. **Chronological split:** first 70% of the year for training, next 15% for validation, last 15% for test. Nothing is shuffled, so the model is never tested on data from before its training period.
4. **Scaling:** a standard scaler is fitted on the **training split only** and then applied to validation and test (no data leakage).
5. **Windowing:** each sample is 96 past steps (input) and the next 96 values of `Usage_kWh` (target). Training windows slide by 1 step; validation and test windows do not overlap.
6. **Baselines** (`src/models.py`): simple models every neural network must beat:
   - *Seasonal naive*: tomorrow = today (repeat the last 24 h).
   - *Last value*: repeat the last reading for the next 24 h.
   - *Ridge regression*: a linear model on the flattened input window.
7. **Neural models** (SimpleRNN, LSTM, GRU, built in Keras): trained with MSE loss and compared with the baselines on MAE in kWh.
8. **Logging:** every run is appended to `experiments/experiment_log.csv`.

## 8. Repository structure

```
greentrade-ai/
├── README.md              this file
├── requirements.txt       package requirements
├── configs/
│   └── config.yaml        all experiment settings (seed, split, window sizes)
├── data/
│   └── download_data.py   downloads the dataset (no raw data in git)
├── notebooks/             01_eda.ipynb, 02_results.ipynb
├── src/
│   ├── data.py            loading, cleaning, splitting, scaling, windowing
│   ├── models.py          baselines and neural architectures
│   ├── train.py           training loop, callbacks, logging   (planned)
│   ├── evaluate.py        metrics in kWh, plots, error analysis
│   ├── predict.py         inference on raw input             (planned)
│   └── run_baselines.py   evaluates the baselines and logs them
├── experiments/           experiment log (CSV) and saved curves
├── models/                final saved model and preprocessing objects
├── app/                   demo application                   (planned)
└── report/                report PDF, slides, figures
```

## 9. Setup and run

```bash
git clone https://github.com/wafaaalkaddaf-afk/greentrade-ai.git
cd greentrade-ai
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python data/download_data.py
python -m src.run_baselines --author "YOUR NAME"
```

The last command prints the validation MAE, RMSE and R2 of the three baselines
and appends them to `experiments/experiment_log.csv`.

To explore the data, open `notebooks/01_eda.ipynb` in Jupyter or VS Code.

Commands for training, evaluation and the demo will be added as those modules
are merged.

## 10. Results

Validation results of the baselines (the test set is used once, at the end).
Values come from `experiments/experiment_log.csv`:

| Model | Val MAE (kWh) | Val RMSE | Val R2 |
|---|---|---|---|
| Seasonal naive (tomorrow = today) | 15.455 | 29.039 | 0.107 |
| Last value | 35.072 | 46.231 | -1.263 |
| Ridge regression | 13.258 | 19.191 | 0.610 |

Ridge regression is the strongest baseline so far; the neural models (next stage) must beat its MAE of 13.258 kWh.

Final test results (mean ± std over 3 seeds, to be filled when the neural models are trained):

| Model | Test MAE (kWh) | Test RMSE | Test R2 |
|---|---|---|---|
| SimpleRNN | | | |
| LSTM | | | |
| GRU | | | |

## 11. Project status

| Item | Owner | Status |
|---|---|---|
| Repository skeleton | Tech Lead | Done |
| Problem statement and config | Tech Lead | Done |
| Data pipeline (leak-free split, scaling, windows) | Data Engineer / Tech Lead | Done |
| EDA with findings table | Tech Lead | Done |
| Baselines | Tech Lead | Done |
| Models A, B, C with hand parameter counts | Model Architect | Planned |
| Tuning and ablations | Experimentation Engineer | Planned |
| Evaluation, error analysis, inference and demo | Evaluation Engineer | Planned |

## 12. Team contributions

| Member | GitHub | Work package | Main files | Experiments run |
|---|---|---|---|---|
| Wafaa | wafaaalkaddaf-afk | Tech Lead, EDA, baselines | README, configs, notebooks/01_eda.ipynb, src/data.py, src/models.py (baselines), src/evaluate.py (metrics) | B001-B003 |
| Mariam | Maryam7075m | Data Engineer | src/data.py | imbalance/augmentation ablations |
| Donia | dodoali202005-netizen | Model Architect | src/models.py (Models B, C) | architecture and regularization ablations |
| Menna Allah Ahmed| mn26-s | Experimentation Engineer | src/train.py, experiments/ | learning-rate, batch-size, optimizer studies |
| RodinaHasaan |  | Evaluation and Deployment | src/evaluate.py, src/predict.py, app/ | evaluation of all models |

## 13. Limitations and ethics

- One year of data from one plant: the test period (last ~15%) may differ from training (distribution shift), and results may not transfer to other plants.
- The data are public and contain no personal information.
- Benefits described for factory owners (lower bills, better pricing) are **potential benefits**, not measured results.
- The system is a course project and decision-support concept, not a certified emissions-verification tool.

## 14. AI tools statement

An AI assistant (Claude) was used to help draft the project code (data pipeline, baselines, metrics), the README and the EDA notebook. The Tech Lead ran all code on the real dataset, checked the numbers (for example the 11 input features and the baseline results), wrote the EDA conclusions from the notebook outputs and is responsible for the content.

## 15. References

1. V E, S., Shin, C., & Cho, Y. (2021). Steel Industry Energy Consumption. UCI ML Repository. https://doi.org/10.24432/C52G8C
2. Sathishkumar V E, Shin C., Cho Y. (2021). Efficient energy consumption prediction model for a data analytic-enabled industry building in a smart city. *Building Research & Information*, 49(1), 127-143.
3. European Commission, Carbon Border Adjustment Mechanism: https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en
