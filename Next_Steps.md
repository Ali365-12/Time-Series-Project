# Retail Sales Forecasting — Next Steps & Technical Roadmap

Based on a full review of the four project notebooks (`1- W1-Project Preparation`, `2- W1-Feature_Engineering`, `9- W3-mlflow-new-chaged & hyperopt-added`, `10- W3-streamlit`) and the current results, the following technical improvements are recommended.

---

## Suggested Order of Implementation

| Priority | Task | Effort | Impact |
|---|---|---|---|
| 1 | Residual diagnostics (Ljung-Box, ACF) | 1–2 hours | Validates current results |
| 2 | Walk-forward cross-validation | 2–3 hours | Methodological rigor |
| 3 | ARIMA prediction intervals in Streamlit | 1–2 hours | Completes existing feature |
| 4 | SHAP analysis for XGBoost | 1–2 hours | Feature insight |
| 5 | Hybrid ARIMA + XGBoost ensemble | 3–4 hours | Likely MAE improvement |
| 6 | Expand Hyperopt search space | 2–3 hours | Marginal MAE gains |
| 7 | Multi-step direct forecasting strategy | 3–5 hours | Better long-horizon accuracy |
| 8 | Streamlit Cloud deployment | 1–2 hours | Portfolio / shareability |

---

## Step 1 — Residual Diagnostics

**Why:** Every model in the project has negative bias (under-predicts). Before improving models, confirm whether ARIMA truly captured all signal or whether systematic structure remains in the residuals.

**What to do:**
- Run the **Ljung-Box test** on ARIMA residuals — if p-value < 0.05, autocorrelation remains and the model missed a pattern
- Plot **ACF / PACF of residuals** — should resemble white noise
- Run an **ARCH test** (Engle's LM test) for heteroscedasticity — checks whether forecast variance is changing over time
- Plot **residuals over time** and a **residual histogram** to check for normality

**Expected outcome:** Confirm whether ARIMA (Hyperopt) is well-specified or whether a higher-order seasonal term is needed.

**Libraries:** `statsmodels.stats.diagnostic.acorr_ljungbox`, `arch.unitroot.engle_granger`

---

## Step 2 — Walk-Forward (Rolling Origin) Cross-Validation

**Why:** The current evaluation uses a single 80/20 temporal split (364 train / 90 test days). This is one observation of model performance. Walk-forward CV provides confidence intervals on MAE and reveals whether the model is stable across time, not just fortunate on one window.

**What to do:**

```
Train: [Jan–Jun 2013],  Test: [Jul 2013]
Train: [Jan–Jul 2013],  Test: [Aug 2013]
Train: [Jan–Aug 2013],  Test: [Sep 2013]
...continuing through Mar 2014
```

- Use **expanding window** (recommended) or **sliding window** (fixed train size)
- Report mean ± std of MAE across all folds
- Compare fold-level stability across ARIMA, Holt-Winters, and XGBoost

**Libraries:** `sklearn.model_selection.TimeSeriesSplit`

---

## Step 3 — Prediction Intervals in Streamlit App

**Why:** Every forecast in the current app is a point estimate only. The README mentions a "90% confidence band" but it is not implemented. Stakeholders need uncertainty ranges to make inventory and supply chain decisions.

**What to do:**

| Approach | Effort | How |
|---|---|---|
| ARIMA native intervals | Low | `model.get_forecast(steps=n).conf_int(alpha=0.1)` returns 90% CI |
| Prophet native intervals | Low | `yhat_lower` / `yhat_upper` columns already in Prophet output |
| Conformal prediction (model-agnostic) | Medium | Calibrate residual quantiles on holdout set |
| XGBoost quantile regression | Medium | Set `objective='reg:quantileerror'`, train P10/P50/P90 models |

**Minimum viable:** Add ARIMA confidence bands to the Forecast tab in `app_prototype.py` using the shaded area between `lower unit_sales` and `upper unit_sales` from `get_forecast()`.

---

## Step 4 — SHAP Feature Importance for XGBoost

**Why:** 23 features were engineered but never analyzed for actual contribution. `lag_1` has r≈0.99 with the target — the XGBoost model may be mostly copying yesterday's value, with the remaining 22 features adding little signal. High collinearity between `lag_1`, `lag_7`, `rolling_7d_mean` etc. may be hurting the model.

**What to do:**
- Compute **SHAP values** for XGBoost on the test set
- Plot `shap.summary_plot()` (feature importance bar + beeswarm)
- Identify features with near-zero SHAP values and test removing them
- Check whether oil price, holiday flags, and calendar features contribute meaningfully

**Expected outcome:** A leaner feature set (potentially 8–12 features instead of 23), reduced overfitting risk, and better XGBoost MAE.

**Libraries:** `shap`

---

## Step 5 — Hybrid ARIMA + XGBoost Ensemble

**Why:** The current leaderboard shows a complementary split in model strengths:
- **ARIMA (Hyperopt):** MAE = 94.81, MAPE = 21.1% — best on average error and percentage error
- **XGBoost (tuned):** RMSE = 141.45, R² = 0.405 — best on large errors and variance explained

A hybrid model that combines both can beat either alone.

**Approach 1 — Weighted Average Ensemble:**
```python
ensemble_forecast = 0.5 * arima_forecast + 0.5 * xgb_forecast
# Optimise weights on validation set using scipy.optimize.minimize
```

**Approach 2 — Residual Hybrid (recommended):**
1. Fit ARIMA on the training series → get in-sample residuals
2. Use ARIMA residuals as an additional feature to train XGBoost
3. Final forecast = ARIMA forecast + XGBoost correction of residuals

This is a well-established technique: ARIMA captures linear autocorrelation; XGBoost corrects non-linear patterns ARIMA misses.

**Expected outcome:** MAE improvement of 3–8 units over standalone ARIMA based on typical hybrid performance on similar datasets.

---

## Step 6 — Expand Hyperopt Search Space

**Why:** Current tuning used small search spaces and few trials. Notable gaps:

| Model | What Was Tuned | What Was Missed |
|---|---|---|
| ARIMA | `(p,d,q)` only | Seasonal order `(P,D,Q,7)` was fixed — never searched |
| XGBoost | Core params | `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda` (regularization) not tuned |
| LSTM | 3 params, 10 trials | Dropout, bidirectional layers, number of layers, activation functions |
| Prophet | 3 params | `holidays_prior_scale`, `n_changepoints`, Fourier order for seasonality |

**What to do:**
- Extend ARIMA Hyperopt to search over `(P,D,Q)` seasonal terms: `hp.choice('P', [0,1,2])` etc.
- Add regularization params to XGBoost search: `reg_alpha` ∈ [0, 1], `reg_lambda` ∈ [0, 2]
- Increase LSTM trials from 10 to 30–50; add dropout rate `[0.1, 0.5]` to search space
- Increase all trial budgets by 2× (current cap of 15–30 trials is low for Bayesian search)

---

## Step 7 — Multi-Step Direct Forecasting Strategy

**Why:** The current setup forecasts 30 days ahead by **recursively chaining 1-step predictions** — each step's error feeds into the next, causing error compounding on longer horizons. The **direct strategy** trains a separate model for each horizon, which avoids this compounding.

**What to do:**
- Train 4 separate XGBoost models for horizons h = 1, 7, 14, 30
- Each model targets `unit_sales_shifted_by_h` instead of `unit_sales_next_day`
- Compare recursive vs. direct MAE at each horizon

**Expected outcome:** Noticeably better accuracy at h=14 and h=30 for XGBoost; ARIMA already handles this natively via its `forecast(steps=n)` method.

**Libraries:** `sklearn` pipelines with custom target shifts

---

## Step 8 — Deploy to Streamlit Cloud

**Why:** The app currently runs locally only. A live URL is significantly more impactful for portfolio presentation than a local demo.

**What to do:**
1. Commit `models/best_model.pkl` to the repository (or store on Hugging Face Hub / GitHub LFS)
2. Create a `requirements.txt` with all dependencies pinned
3. Push to GitHub and connect repo to [share.streamlit.io](https://share.streamlit.io)
4. Set `app_prototype.py` as the entry point

**Blockers to resolve:**
- `best_model.pkl` must be accessible at runtime (confirm file size < 100 MB for direct commit)
- `mlruns/` folder must be committed or MLflow metrics must be hardcoded as a fallback in the Metrics tab
- Confirm `torch` and `prophet` install correctly on Streamlit Cloud (known dependency issues)

---

## Additional Ideas (Lower Priority)

### Store-Level Forecasting
All current models operate on aggregated daily sales across all 54 stores. Training per-store or per-region models would dramatically increase business value but requires restructuring the data pipeline.

### External Features
Only oil price and holidays are used as external regressors. Potential additions:
- **Weather data** (temperature, precipitation in Guayas region)
- **Macro-economic indicators** (Ecuador GDP, inflation, exchange rate)
- **Promotional events** (if available in the Corporación Favorita dataset)

### Automated Retraining Pipeline
Current models are static — trained once and never updated. A scheduled retraining pipeline (e.g., weekly) with MLflow model versioning would make the system production-ready.

---

## Current Baseline to Beat

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| **ARIMA (Hyperopt)** ⭐ | **94.81** | 142.67 | 21.1% | 0.395 |
| Holt-Winters (Hyperopt) | 94.83 | 143.07 | 21.0% | 0.392 |
| XGBoost (tuned) | 96.30 | **141.45** | 22.0% | **0.405** |
| LSTM (Hyperopt) | 123.13 | 172.22 | 26.5% | 0.118 |
| Prophet (Hyperopt) | 127.86 | 182.61 | 24.9% | 0.009 |

Target: push best MAE below **90** and R² above **0.45** through ensembling and improved cross-validation.
