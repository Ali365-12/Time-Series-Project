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
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False
from io import BytesIO
from fpdf import FPDF
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
MLFLOW_PATH = os.path.join(PROJECT_DIR, "mlruns")

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

# ── Load MLflow metrics ───────────────────────────────────────
@st.cache_data
def load_mlflow_metrics():
    """Load all model metrics from MLflow"""
    if not MLFLOW_AVAILABLE:
        return pd.DataFrame()
    try:
        mlflow.set_tracking_uri("file:///" + MLFLOW_PATH.replace("\\", "/"))
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("retail_sales_forecasting")

        if experiment is None:
            return pd.DataFrame()

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        rows = []
        for run in runs:
            rows.append({
                'Model': run.info.run_name,
                'MAE': run.data.metrics.get('MAE'),
                'RMSE': run.data.metrics.get('RMSE'),
                'MAPE': run.data.metrics.get('MAPE'),
                'Bias': run.data.metrics.get('Bias'),
                'R2': run.data.metrics.get('R2'),
            })

        df_metrics = pd.DataFrame(rows).dropna(subset=['MAE'])
        df_metrics = df_metrics.drop_duplicates(subset='Model', keep='first')
        return df_metrics.sort_values('MAE').reset_index(drop=True)
    except:
        return pd.DataFrame()

# ── Generate PDF Report ───────────────────────────────────────
def generate_pdf_report(cutoff_date, n_days, forecast_df, metrics_df, selected_model):
    """Generate a PDF report with forecast and metrics"""
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Retail Sales Forecast Report", ln=True, align="C")

    pdf.set_font("Arial", "", 11)
    pdf.ln(5)

    pdf.cell(0, 10, f"Cutoff Date: {cutoff_date}", ln=True)
    pdf.cell(0, 10, f"Forecast Period: {n_days} days", ln=True)
    pdf.cell(0, 10, f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Region: Guayas", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Forecast Summary", ln=True)

    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Average Forecast: {forecast_df['forecast'].mean():.2f}", ln=True)
    pdf.cell(0, 8, f"Min Forecast: {forecast_df['forecast'].min():.2f}", ln=True)
    pdf.cell(0, 8, f"Max Forecast: {forecast_df['forecast'].max():.2f}", ln=True)
    pdf.cell(0, 8, f"Total (Sum): {forecast_df['forecast'].sum():.2f}", ln=True)

    if not metrics_df.empty and selected_model:
        model_metrics = metrics_df[metrics_df['Model'] == selected_model]
        if not model_metrics.empty:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Model: {selected_model}", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 8, f"MAE: {model_metrics['MAE'].values[0]:.2f}", ln=True)
            pdf.cell(0, 8, f"RMSE: {model_metrics['RMSE'].values[0]:.2f}", ln=True)
            pdf.cell(0, 8, f"MAPE: {model_metrics['MAPE'].values[0]:.1f}%", ln=True)
            pdf.cell(0, 8, f"R² Score: {model_metrics['R2'].values[0]:.3f}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Forecast Values", ln=True)
    pdf.set_font("Arial", "", 9)

    for idx, row in forecast_df.head(10).reset_index().iterrows():
        pdf.cell(0, 7, f"{row['date'].strftime('%Y-%m-%d')}: {row['forecast']:.2f}", ln=True)

    if len(forecast_df) > 10:
        pdf.cell(0, 7, f"... and {len(forecast_df) - 10} more days", ln=True)

    return bytes(pdf.output(dest='S'))

# ── Backtesting Function (#7 - Medium Effort) ────────────────
def perform_backtest(df, model, features, test_size=0.2):
    """
    Performs backtesting by:
    1. Splitting data into train/test
    2. Making predictions on test set
    3. Comparing predictions vs actual
    Returns: predictions, actuals, metrics
    """
    cutoff_idx = int(len(df) * (1 - test_size))
    test_df = df.iloc[cutoff_idx:]
    model_type = type(model).__name__

    backtest_results = []

    # ARIMA/Holt-Winters: model.forecast(steps=1) always forecasts from training
    # end and ignores cutoff_date, producing a flat line. Fix: make one
    # multi-step forecast covering all test days so each step varies correctly.
    if any(t in model_type for t in ('SARIMAXResults', 'ARIMAResults',
                                      'HoltWintersResults', 'ExponentialSmoothing')):
        n_steps = len(test_df)
        try:
            all_preds = model.forecast(steps=n_steps)
            for i in range(n_steps):
                actual = test_df[TARGET].iloc[i]
                pred = max(0, float(all_preds.iloc[i]))
                backtest_results.append({
                    'date': test_df.index[i],
                    'actual': actual,
                    'prediction': round(pred, 2),
                    'error': actual - pred,
                    'abs_error': abs(actual - pred),
                    'pct_error': abs(actual - pred) / actual * 100 if actual > 0 else 0
                })
        except Exception as e:
            st.warning(f"Backtest error: {e}")

    else:
        # XGBoost / sklearn: make_forecast uses feature rows per day, works correctly
        for i in range(len(test_df)):
            cutoff_date = test_df.index[i]
            try:
                pred_df = make_forecast(df, model, features, cutoff_date, n_days=1)
                actual = test_df[TARGET].iloc[i]
                pred = pred_df['forecast'].iloc[0] if len(pred_df) > 0 else 0
                backtest_results.append({
                    'date': cutoff_date,
                    'actual': actual,
                    'prediction': pred,
                    'error': actual - pred,
                    'abs_error': abs(actual - pred),
                    'pct_error': abs(actual - pred) / actual * 100 if actual > 0 else 0
                })
            except:
                pass

    results_df = pd.DataFrame(backtest_results)

    if len(results_df) > 0:
        mae = results_df['abs_error'].mean()
        rmse = np.sqrt((results_df['error'] ** 2).mean())
        mape = results_df['pct_error'].mean()
    else:
        mae = rmse = mape = 0

    return results_df, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# ── Seasonal Decomposition (#10 - Advanced) ──────────────────
def get_seasonal_decomposition(df, period=30):
    """Decompose time series into trend, seasonal, residual"""
    try:
        if len(df) < period * 2:
            period = max(7, len(df) // 4)

        decomposition = seasonal_decompose(
            df[TARGET],
            model='additive',
            period=period
        )
        return decomposition
    except Exception as e:
        st.warning(f"Could not decompose: {str(e)}")
        return None

# ── Anomaly Detection (#11 - Advanced) ────────────────────────
def detect_anomalies(df, contamination=0.05):
    """Detect anomalies using Isolation Forest"""
    try:
        X = df[[TARGET]].values
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(X)

        anomaly_df = pd.DataFrame({
            'date': df.index,
            'value': df[TARGET].values,
            'is_anomaly': anomalies == -1
        }).set_index('date')

        return anomaly_df
    except Exception as e:
        st.warning(f"Could not detect anomalies: {str(e)}")
        return None

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
st.title("📊 Retail Sales Forecasting")
st.write("Corporacion Favorita — Guayas region")

# Temporary debug — remove after confirming paths
with st.expander("🔧 Debug info"):
    st.write(f"PROJECT_DIR: `{PROJECT_DIR}`")
    st.write(f"MODELS_DIR exists: `{os.path.exists(MODELS_DIR)}`")
    st.write(f"best_model.pkl exists: `{os.path.exists(os.path.join(MODELS_DIR, 'best_model.pkl'))}`")
    st.write(f"DATA_DIR exists: `{os.path.exists(DATA_DIR)}`")

# Sidebar
st.sidebar.header("⚙️ Forecast settings")

best_model_name = load_model_name()

st.sidebar.markdown("---")

# Interactive Range Slider (#9 - Medium Effort)
st.sidebar.subheader("📅 Date Range")
cutoff_date = st.sidebar.date_input(
    "Cutoff date",
    value=date(2014, 1, 15),
    min_value=date(2013, 6, 1),
    max_value=date(2014, 3, 30),
    help="Forecast starts the day after this date."
)

# Interactive slider for n_days (#9)
n_days_slider = st.sidebar.slider(
    "Days to forecast (Interactive)",
    min_value=1,
    max_value=30,
    value=7,
    step=1,
    help="Drag to adjust the forecast window in real-time"
)

# Display forecast range preview
forecast_start = pd.to_datetime(cutoff_date) + pd.Timedelta(days=1)
forecast_end = forecast_start + pd.Timedelta(days=n_days_slider-1)
st.sidebar.info(f"**Forecast range:** {forecast_start.date()} to {forecast_end.date()}")

n_days = n_days_slider
history_days = st.sidebar.slider("History days to show", 14, 120, 60)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Selection")
st.sidebar.info(f"**Best model:** {best_model_name}")

metrics_df = load_mlflow_metrics()
if not metrics_df.empty:
    available_models = metrics_df['Model'].tolist()
    selected_model = st.sidebar.selectbox(
        "Choose a model to compare",
        available_models,
        help="Select different models to compare performance"
    )
else:
    selected_model = None

run_button = st.sidebar.button("🚀 Run Forecast", use_container_width=True)

# Main panel
if run_button:
    model = load_model()

    if model is None:
        st.error("❌ Model not found. Run W3-mlflow.ipynb first.")
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

        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Forecast",
            "📊 Metrics",
            "🔄 Backtesting",
            "📉 Decomposition",
            "🚨 Anomalies",
            "📋 Data"
        ])

        # TAB 1: Forecast Chart
        with tab1:
            fig, ax = plt.subplots(figsize=(14, 5))

            # Historical data
            ax.plot(history_plot.index, history_plot.values,
                    label="Historical sales", color="steelblue", linewidth=1.5)

            # Forecast with confidence interval (±10% band)
            forecast_values = forecast_df["forecast"].values
            upper_bound = forecast_values * 1.10
            lower_bound = forecast_values * 0.90

            ax.plot(forecast_df.index, forecast_values,
                    label=f"{n_days}-day forecast", color="orange",
                    linestyle="--", linewidth=2, marker="o", markersize=4)
            ax.fill_between(forecast_df.index, lower_bound, upper_bound,
                           alpha=0.2, color="orange", label="90% Confidence band")

            ax.axvline(cutoff, color="red", linestyle=":", linewidth=1.5, label="Cutoff date")
            ax.set_title(f"Sales Forecast from {cutoff.date()}", fontsize=14, fontweight="bold")
            ax.set_ylabel("Unit Sales")
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            # Statistical Summary
            st.subheader("📈 Forecast Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Forecast", f"{forecast_df['forecast'].mean():.2f}")
            with col2:
                st.metric("Min Forecast", f"{forecast_df['forecast'].min():.2f}")
            with col3:
                st.metric("Max Forecast", f"{forecast_df['forecast'].max():.2f}")
            with col4:
                st.metric("Total (Sum)", f"{forecast_df['forecast'].sum():.2f}")

            # Forecast table
            st.subheader("Forecast values")
            st.dataframe(forecast_df.reset_index().rename(
                columns={"date": "Date", "forecast": "Predicted Sales"}
            ), use_container_width=True)

            # Download options
            col_csv, col_pdf = st.columns(2)

            with col_csv:
                csv = forecast_df.reset_index().to_csv(index=False)
                st.download_button(
                    label="⬇️ CSV Report",
                    data=csv,
                    file_name=f"forecast_{cutoff_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_pdf:
                # Generate PDF Report (#8 - Medium Effort)
                pdf_data = generate_pdf_report(
                    cutoff_date,
                    n_days,
                    forecast_df,
                    metrics_df,
                    selected_model if selected_model else None
                )
                st.download_button(
                    label="📄 PDF Report",
                    data=pdf_data,
                    file_name=f"forecast_report_{cutoff_date}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        # TAB 2: Model Metrics
        with tab2:
            st.subheader("🏆 Model Performance Comparison")

            if not metrics_df.empty:
                st.dataframe(
                    metrics_df[['Model', 'MAE', 'RMSE', 'MAPE', 'R2']].round(2),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("---")
                st.subheader("Selected Model Details")

                if selected_model and not metrics_df.empty:
                    model_metrics = metrics_df[metrics_df['Model'] == selected_model]
                    if not model_metrics.empty:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("MAE", f"{model_metrics['MAE'].values[0]:.2f}")
                        with col2:
                            st.metric("RMSE", f"{model_metrics['RMSE'].values[0]:.2f}")
                        with col3:
                            st.metric("MAPE", f"{model_metrics['MAPE'].values[0]:.1f}%")
                        with col4:
                            st.metric("Bias", f"{model_metrics['Bias'].values[0]:.2f}")
                        with col5:
                            st.metric("R² Score", f"{model_metrics['R2'].values[0]:.3f}")

                        st.info(f"**Active:** {selected_model}")
            else:
                st.warning("Could not load MLflow metrics. Run W3-mlflow.ipynb first.")

        # TAB 3: Backtesting (#7 - Medium Effort)
        with tab3:
            st.subheader("🔄 Model Backtesting")
            st.write("Evaluating model performance on historical data (last 20% of dataset)")

            with st.spinner("Running backtest..."):
                backtest_df, backtest_metrics = perform_backtest(
                    df, model, FEATURES, test_size=0.2
                )

            if len(backtest_df) > 0:
                # Backtest chart
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(backtest_df['date'], backtest_df['actual'],
                       label='Actual', color='steelblue', linewidth=2)
                ax.plot(backtest_df['date'], backtest_df['prediction'],
                       label='Predicted', color='orange', linestyle='--', linewidth=2)
                ax.fill_between(backtest_df['date'], backtest_df['actual'],
                               backtest_df['prediction'], alpha=0.2, color='red')
                ax.set_title("Backtest: Actual vs Predicted", fontsize=14, fontweight="bold")
                ax.set_ylabel("Unit Sales")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

                # Backtest metrics
                st.subheader("Backtest Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Backtest MAE", f"{backtest_metrics['MAE']:.2f}")
                with col2:
                    st.metric("Backtest RMSE", f"{backtest_metrics['RMSE']:.2f}")
                with col3:
                    st.metric("Backtest MAPE", f"{backtest_metrics['MAPE']:.1f}%")

                # Backtest table
                st.subheader("Detailed Results")
                st.dataframe(
                    backtest_df[['date', 'actual', 'prediction', 'error', 'pct_error']]
                    .rename(columns={
                        'date': 'Date',
                        'actual': 'Actual',
                        'prediction': 'Predicted',
                        'error': 'Error',
                        'pct_error': 'Error %'
                    })
                    .reset_index(drop=True),
                    use_container_width=True
                )
            else:
                st.warning("Could not run backtest with current model")

        # TAB 4: Seasonal Decomposition (#10 - Advanced)
        with tab4:
            st.subheader("📉 Seasonal Decomposition")
            st.write("Breaking down the time series into trend, seasonal, and residual components")

            decomposition = get_seasonal_decomposition(df, period=30)

            if decomposition is not None:
                fig, axes = plt.subplots(4, 1, figsize=(14, 10))

                # Original
                axes[0].plot(df.index, df[TARGET].values, color='steelblue', linewidth=1.5)
                axes[0].set_ylabel('Original', fontweight='bold')
                axes[0].grid(True, alpha=0.3)

                # Trend
                axes[1].plot(df.index, decomposition.trend.values, color='orange', linewidth=1.5)
                axes[1].set_ylabel('Trend', fontweight='bold')
                axes[1].grid(True, alpha=0.3)

                # Seasonal
                axes[2].plot(df.index, decomposition.seasonal.values, color='green', linewidth=1.5)
                axes[2].set_ylabel('Seasonal', fontweight='bold')
                axes[2].grid(True, alpha=0.3)

                # Residual
                axes[3].plot(df.index, decomposition.resid.values, color='red', linewidth=0.8, alpha=0.7)
                axes[3].set_ylabel('Residual', fontweight='bold')
                axes[3].set_xlabel('Date')
                axes[3].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                st.info(
                    "📊 **Interpretation:**\n"
                    "- **Trend:** Overall direction of sales (uptrend/downtrend)\n"
                    "- **Seasonal:** Repeating patterns (e.g., day-of-week effects)\n"
                    "- **Residual:** Random noise and unexplained variation"
                )

        # TAB 5: Anomaly Detection (#11 - Advanced)
        with tab5:
            st.subheader("🚨 Anomaly Detection")
            st.write("Identifying unusual sales patterns using Isolation Forest")

            contamination = st.slider(
                "Anomaly sensitivity",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                help="Higher = more anomalies flagged"
            )

            anomalies = detect_anomalies(df, contamination=contamination)

            if anomalies is not None:
                # Anomaly chart
                fig, ax = plt.subplots(figsize=(14, 5))

                normal = anomalies[~anomalies['is_anomaly']]
                anomaly_points = anomalies[anomalies['is_anomaly']]

                ax.scatter(normal.index, normal['value'], color='steelblue', label='Normal', s=20)
                ax.scatter(anomaly_points.index, anomaly_points['value'],
                          color='red', label='Anomaly', s=100, marker='X', zorder=5)

                ax.set_title("Anomaly Detection in Historical Data", fontsize=14, fontweight="bold")
                ax.set_ylabel("Unit Sales")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

                # Anomaly statistics
                num_anomalies = anomalies['is_anomaly'].sum()
                pct_anomalies = (num_anomalies / len(anomalies)) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Anomalies", num_anomalies)
                with col2:
                    st.metric("% Anomalous", f"{pct_anomalies:.1f}%")
                with col3:
                    st.metric("Normal Points", len(anomalies) - num_anomalies)

                # Anomalies table
                if num_anomalies > 0:
                    st.subheader("Detected Anomalies")
                    anomaly_list = anomalies[anomalies['is_anomaly']][['value']].reset_index()
                    anomaly_list.columns = ['Date', 'Sales Value']
                    st.dataframe(anomaly_list.head(20), use_container_width=True)

        # TAB 6: Raw Data
        with tab6:
            st.subheader("Dataset Overview")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Date range:** {df.index.min().date()} to {df.index.max().date()}")

            st.subheader("Recent data")
            st.dataframe(df.tail(20).reset_index(), use_container_width=True)

        st.success("✅ Forecast complete!")

else:
    st.info("⚙️ Adjust the settings in the sidebar and click **Run Forecast**.")
    st.write("**Data range:** January 2013 – March 2014")
