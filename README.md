# Retail Sales Forecasting — End-to-End Time Series Pipeline

A complete machine learning pipeline for daily retail sales forecasting, built on data from **Corporación Favorita** (Ecuador, Guayas region). The project covers the full journey from raw data to a deployed interactive web application: exploratory analysis, feature engineering, multi-model training, Bayesian hyperparameter tuning, MLflow experiment tracking, and a Streamlit dashboard for stakeholders.

---

## Project Overview

| | |
|---|---|
| **Domain** | Retail / Supply Chain |
| **Task** | Daily unit sales forecasting |
| **Region** | Guayas, Ecuador |
| **Date range** | January 2013 – March 2014 |
| **Forecast horizon** | 1 – 30 days ahead |
| **Best model** | ARIMA with Hyperopt tuning (MAE = 94.81) |

---

## Repository Structure

```
├── 1- Project Preparation.ipynb       # EDA on all raw datasets
├── 2- Feature_Engineering.ipynb       # Data merging + feature engineering + EDA
├── 3- Mlflow-new-chaged & hyperopt-added.ipynb  # Model training, tuning, MLflow
├── 4- Streamlit.ipynb                # Streamlit app development
├── app_prototype.py                      # Deployable Streamlit application
├── data/
│   ├── timeseries.csv                    # Daily unit sales
│   ├── oil.csv                           # Daily crude oil prices
│   ├── holidays.csv                      # Ecuador public holidays
│   └── stores.csv                        # Store locations
├── models/
│   ├── best_model.pkl                    # Saved best model (auto-generated)
│   └── best_model_name.txt               # Name of best model (auto-generated)
└── mlruns/                               # MLflow experiment tracking (auto-generated)
```

---

## Datasets

Four raw CSV files are merged into a single feature-rich dataset:

| File | Rows | Description |
|---|---|---|
| `timeseries.csv` | 452 | Daily aggregated unit sales — the prediction target |
| `oil.csv` | 1,218 | Daily WTI crude oil price (USD/barrel) — economic indicator |
| `holidays.csv` | 350 | Ecuador public holidays with locale (National / Regional / Local) |
| `stores.csv` | 54 | Store metadata including city and region |

---

## Notebooks

### 1 — Project Preparation (`1- Project Preparation.ipynb`)

Exploratory Data Analysis on each raw dataset individually before any merging.

- **Sales:** time series plot, distribution, stationarity test (ADF), seasonal decomposition (multiplicative, period=7), boxplot, autocorrelation (ACF)
- **Oil prices:** trend plot, missing value identification, linear interpolation
- **Holidays:** locale distribution, holidays per year and per month, top locale names, heatmap by month × year
- **Stores:** regional distribution

---

### 2 — Feature Engineering (`2- Feature_Engineering.ipynb`)

Merges all four datasets and engineers 24 predictive features. Produces `timeseries_with_features.csv`.

**Merging strategy:** left joins on `date` from the timeseries table as the anchor.

**Features created:**

| Group | Features |
|---|---|
| Calendar | `year`, `month`, `day`, `dayofweek`, `quarter`, `week_of_year`, `is_weekend`, `is_month_start`, `is_month_end` |
| Lag | `lag_1`, `lag_7`, `lag_14`, `lag_30` |
| Rolling window | `rolling_7d_mean`, `rolling_14d_mean`, `rolling_30d_mean`, `rolling_7d_std` |
| Oil price | `dcoilwtico`, `oil_lag_1`, `oil_rolling_7d_mean` |
| Holiday flags | `is_national_holiday`, `is_regional_holiday`, `is_local_holiday` |

**Final dataset:** 452 rows × 25 columns (1 target + 24 features)

Post-engineering EDA covers unit sales distribution by day of week, month, and year; holiday impact analysis; oil price trend and correlation with sales; lag feature scatter plots; rolling mean overlays; and a full 24-feature correlation heatmap.

---

### 3 — Modeling, Tuning & MLflow (`3- Mlflow-new-chaged & hyperopt-added.ipynb`)

Trains five model families, applies Bayesian hyperparameter tuning to each, compares all results, saves the best model, and logs everything to MLflow.

**Train / test split (temporal):**
- Training: January 2013 – December 2013 (364 days)
- Test: January 2014 – March 2014 (90 days)

**Models trained:**

| Model | Notes |
|---|---|
| ARIMA / SARIMAX | Seasonal order (1,1,1,7) for weekly seasonality |
| Holt-Winters | Additive trend and seasonality, period = 7 |
| Prophet | Weekly + yearly seasonality |
| XGBoost | Gradient boosted trees using all 23 engineered features |
| LSTM | Sequence model with 7-day look-back window |

**Hyperparameter tuning:**
All models are tuned using **Hyperopt Bayesian TPE** with a 30-day held-out validation window carved from the end of the training set (no data leakage). XGBoost is additionally tuned with `RandomizedSearchCV` + `TimeSeriesSplit`.

**Evaluation metrics:**

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error — average daily sales error |
| RMSE | Root Mean Squared Error — penalises large errors more heavily |
| MAPE | Mean Absolute Percentage Error |
| Bias | Systematic over- or under-prediction |
| R² | Proportion of variance explained |

**Final leaderboard (sorted by MAE):**

| Model | MAE | RMSE | MAPE | Bias | R² |
|---|---|---|---|---|---|
| **ARIMA (Hyperopt)** ⭐ | **94.81** | **142.67** | **21.1%** | -6.90 | 0.395 |
| Holt-Winters (Hyperopt) | 94.83 | 143.07 | 21.0% | -9.12 | 0.392 |
| XGBoost (tuned) | 96.30 | 141.45 | 22.0% | -2.19 | 0.405 |
| XGBoost (Hyperopt) | 96.42 | 143.88 | 21.9% | -1.96 | 0.385 |
| ARIMA (baseline) | 95.65 | 143.68 | 21.2% | -9.29 | 0.386 |
| LSTM (Hyperopt) | 123.13 | 172.22 | 26.5% | -10.96 | 0.118 |
| Prophet (Hyperopt) | 127.86 | 182.61 | 24.9% | -80.26 | 0.009 |

The best model is saved to `models/best_model.pkl` and logged in MLflow as `BEST__ARIMA (Hyperopt)`.

---

### 4 — Streamlit Application (`4- Streamlit.ipynb` + `app_prototype.py`)

Introduces Streamlit and builds a fully interactive forecasting dashboard for business stakeholders.

**App features:**

| Tab | Contents |
|---|---|
| Forecast | Interactive forecast chart with 90% confidence band, statistical summary (avg, min, max, total), CSV and PDF download |
| Metrics | Full model performance table pulled from MLflow, per-model metric cards |
| Backtesting | Actual vs predicted chart on the last 20% of data, MAE / RMSE / MAPE |
| Decomposition | Seasonal decomposition into trend, seasonal, and residual components |
| Anomalies | Isolation Forest anomaly detection with adjustable sensitivity slider |
| Data | Raw dataset preview with shape and date range |

**Sidebar controls:** cutoff date picker, forecast horizon slider (1–30 days), history window slider, model selector.

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib scikit-learn xgboost statsmodels prophet mlflow joblib hyperopt fpdf streamlit torch
```

### Run the notebooks in order

```
1- Project Preparation.ipynb
2- Feature_Engineering.ipynb
3- Mlflow-new-chaged & hyperopt-added.ipynb
4- Streamlit.ipynb  (optional — for app development walkthrough)
```

### Launch the Streamlit app

```bash
cd "path/to/your/project"
python -m streamlit run app_prototype.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** Run notebook `3-` first to generate `models/best_model.pkl`. The app will not load without it.

---

## Key Findings

- **Seasonal patterns are dominant:** weekly cycles (day-of-week effects) drive the majority of sales variation, making ARIMA with seasonal order `(1,1,1,7)` the strongest overall performer.
- **Hyperopt tuning improved all models:** XGBoost improved by 12.30 MAE units over its baseline; ARIMA improved by 0.84 MAE units.
- **Oil price has a weak direct correlation** with unit sales but contributes indirectly through lag and rolling window features.
- **National holidays increase average sales by ~17%** compared to non-holiday days (556 vs 476 average units).
- **Prophet and LSTM underperformed** relative to classical statistical models on this dataset size (452 days), likely due to insufficient data for their model capacity.

---

## Tech Stack

| Area | Tools |
|---|---|
| Data processing | pandas, numpy |
| Visualisation | matplotlib |
| Statistical models | statsmodels (SARIMAX, Holt-Winters) |
| ML models | XGBoost, scikit-learn |
| Deep learning | PyTorch (LSTM) |
| Probabilistic forecasting | Prophet (Meta) |
| Hyperparameter tuning | Hyperopt (Bayesian TPE), RandomizedSearchCV |
| Experiment tracking | MLflow |
| Deployment | Streamlit |
| Model persistence | joblib |