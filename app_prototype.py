# Retail Sales Forecasting App — Prototype
# Built in Day 2. Polished and deployed in Day 3.
#
# To launch:
#   Open Command Prompt
#   cd C:\\Users\\alipa\\Documents\\3- Academic\\6- Data Science, phyton\\1- class files\\9- Time Series Modeling\\Time-Series
#   python -m streamlit run app_prototype.py

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
PROJECT_DIR = r"C:\\Users\\alipa\\Documents\\3- Academic\\6- Data Science, phyton\\1- class files\\9- Time Series Modeling\\Time-Series"
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")

FEATURES = [
    "year", "month", "day", "dayofweek", "quarter", "week_of_year",
    "is_weekend", "is_month_start", "is_month_end",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_7d_mean", "rolling_14d_mean", "rolling_30d_mean", "rolling_7d_std",
    "dcoilwtico", "oil_lag_1", "oil_rolling_7d_mean",
    "is_national_holiday", "is_regional_holiday", "is_local_holiday",
]
TARGET = "unit_sales"

# ── Load data (cached — runs once) ───────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "timeseries_with_features.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date").dropna()

# ── Load model (cached — runs once) ──────────────────────────
@st.cache_resource
def load_model():
    pkl_path = os.path.join(MODELS_DIR, "best_model.pkl")
    pt_path  = os.path.join(MODELS_DIR, "best_model.pt")

    if os.path.exists(pkl_path):
        return joblib.load(pkl_path)

    elif os.path.exists(pt_path):
        import torch
        import torch.nn as nn
        class SimpleLSTM(nn.Module):
            def __init__(self, hidden=64):
                super().__init__()
                self.lstm   = nn.LSTM(1, hidden, batch_first=True)
                self.linear = nn.Linear(hidden, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.linear(out[:, -1, :])
        model = SimpleLSTM(hidden=64)
        model.load_state_dict(torch.load(pt_path))
        model.eval()
        return model

    else:
        return None

# ── Load best model name ──────────────────────────────────────
def load_model_name():
    name_path = os.path.join(MODELS_DIR, "best_model_name.txt")
    if os.path.exists(name_path):
        with open(name_path) as f:
            return f.read().strip()
    return "Unknown"

# ── Forecast function ─────────────────────────────────────────
def make_forecast(df, model, features, cutoff_date, n_days=1):
    """
    Generates a forecast for n_days starting after cutoff_date.
    Automatically handles XGBoost, ARIMA, Holt-Winters, and Prophet.

    Parameters:
        df          : full feature-engineered DataFrame
        model       : trained model object
        features    : list of feature column names (used by XGBoost only)
        cutoff_date : forecast starts the day after this date
        n_days      : number of days to forecast

    Returns:
        DataFrame with columns ['date', 'forecast']
    """
    cutoff     = pd.to_datetime(cutoff_date)
    model_type = type(model).__name__

    # ── ARIMA / SARIMAX ──────────────────────────────────────
    # Uses forecast() from the last training date — no feature row needed
    if 'SARIMAXResults' in model_type or 'ARIMAResults' in model_type:
        preds = model.forecast(steps=n_days)
        forecasts = [
            {'date': cutoff + pd.Timedelta(days=i+1),
             'forecast': round(float(max(0, p)), 2)}
            for i, p in enumerate(preds)
        ]
        return pd.DataFrame(forecasts).set_index('date')

    # ── Holt-Winters ─────────────────────────────────────────
    # Uses forecast() exactly like ARIMA
    elif 'HoltWintersResults' in model_type or 'ExponentialSmoothing' in model_type:
        preds = model.forecast(n_days)
        forecasts = [
            {'date': cutoff + pd.Timedelta(days=i+1),
             'forecast': round(float(max(0, p)), 2)}
            for i, p in enumerate(preds)
        ]
        return pd.DataFrame(forecasts).set_index('date')

    # ── Prophet ──────────────────────────────────────────────
    # Requires a future DataFrame with a 'ds' column
    elif 'Prophet' in model_type:
        future_dates = pd.date_range(
            start=cutoff + pd.Timedelta(days=1),
            periods=n_days,
            freq='D'
        )
        future_df    = pd.DataFrame({'ds': future_dates})
        forecast_out = model.predict(future_df)
        forecasts = [
            {'date': row['ds'],
             'forecast': round(float(max(0, row['yhat'])), 2)}
            for _, row in forecast_out.iterrows()
        ]
        return pd.DataFrame(forecasts).set_index('date')

    # ── XGBoost / sklearn-compatible models ──────────────────
    # Uses feature rows — one prediction per day
    else:
        history   = df.loc[df.index <= cutoff].copy()
        forecasts = []

        if len(history) == 0:
            raise ValueError(f"No data found on or before {cutoff_date}.")

        for i in range(n_days):
            next_date = cutoff + pd.Timedelta(days=i+1)

            if next_date in df.index:
                row = df.loc[[next_date], features]
            else:
                row = history.iloc[[-1]][features].copy()
                row.index = [next_date]

            pred = model.predict(row)[0]
            pred = max(0, pred)
            forecasts.append({
                'date':     next_date,
                'forecast': round(float(pred), 2)
            })

        return pd.DataFrame(forecasts).set_index('date')

# ── App layout ────────────────────────────────────────────────
st.title("Retail Sales Forecasting")
st.write("Corporacion Favorita — Guayas region")

model_name = load_model_name()
st.info(f"Active model: {model_name}")

# Sidebar
st.sidebar.header("Forecast settings")

cutoff_date = st.sidebar.date_input(
    "Cutoff date",
    value=date(2014, 1, 15),
    min_value=date(2013, 6, 1),
    max_value=date(2014, 3, 30),
    help="Forecast starts the day after this date."
)

n_days = st.sidebar.slider("Days to forecast", 1, 30, 7)
history_days = st.sidebar.slider("History days to show", 14, 120, 60)
run_button = st.sidebar.button("Run Forecast")

# Main panel
if run_button:
    model = load_model()

    if model is None:
        st.error("Model not found. Run W3-mlflow.ipynb first.")
    else:
        with st.spinner("Loading data..."):
            df = load_data()

        with st.spinner("Generating forecast..."):
            cutoff       = pd.to_datetime(cutoff_date)
            history_plot = df.loc[
                (df.index >= cutoff - pd.Timedelta(days=history_days)) &
                (df.index <= cutoff)
            ][TARGET]
            forecast_df = make_forecast(df, model, FEATURES, cutoff, n_days)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(history_plot.index, history_plot.values,
                label="Historical sales", color="steelblue", linewidth=1.5)
        ax.plot(forecast_df.index, forecast_df["forecast"].values,
                label=f"{n_days}-day forecast", color="orange",
                linestyle="--", linewidth=2, marker="o", markersize=4)
        ax.axvline(cutoff, color="red", linestyle=":", linewidth=1.5, label="Cutoff date")
        ax.set_title(f"Sales Forecast from {cutoff.date()}")
        ax.set_ylabel("Unit Sales")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Forecast values")
        st.dataframe(forecast_df.reset_index().rename(
            columns={"date": "Date", "forecast": "Predicted Sales"}
        ))

        csv = forecast_df.reset_index().to_csv(index=False)
        st.download_button(
            label="Download forecast as CSV",
            data=csv,
            file_name=f"forecast_{cutoff_date}.csv",
            mime="text/csv"
        )

        st.success("Forecast complete!")

else:
    st.info("Adjust the settings in the sidebar and click Run Forecast.")
    st.write("**Data range:** January 2013 – March 2014")
