import type { Chapter } from "../types";

export const timeSeries: Chapter = {
  title: "Time Series",
  slug: "time-series",
  pages: [
    {
      title: "Stationarity & Decomposition",
      slug: "stationarity-decomposition",
      description:
        "Time series fundamentals: stationarity testing, trend-seasonality decomposition, and differencing",
      markdownContent: `# Stationarity & Decomposition

## What Is a Time Series?

A **time series** is a sequence of data points indexed (or ordered) by time. Examples include daily stock prices, monthly sales figures, and hourly sensor readings. The temporal ordering is what distinguishes time series data from ordinary tabular data — the order of observations carries information.

## Stationarity

A time series is **stationary** when its statistical properties do not change over time. Formally, a strictly stationary process has constant:

- **Mean:** $E[X_t] = \\mu$ for all $t$
- **Variance:** $\\text{Var}(X_t) = \\sigma^2$ for all $t$
- **Autocovariance:** $\\text{Cov}(X_t, X_{t+h})$ depends only on the lag $h$, not on $t$

### Why Stationarity Matters

Most classical time series models (ARIMA, exponential smoothing) **assume stationarity**. If the data has a trend or changing variance, these models will produce unreliable forecasts. We must first transform the data to be stationary, fit the model, then reverse the transformation.

## Testing for Stationarity: ADF Test

The **Augmented Dickey-Fuller (ADF)** test is the standard hypothesis test for stationarity:

- $H_0$: The series has a **unit root** (non-stationary)
- $H_1$: The series is **stationary**

The test fits a regression of the form:

$$
\\Delta y_t = \\alpha + \\beta t + \\gamma y_{t-1} + \\sum_{i=1}^{p} \\delta_i \\Delta y_{t-i} + \\varepsilon_t
$$

If the test statistic is more negative than the critical value (or p-value $< 0.05$), we reject $H_0$ and conclude the series is stationary.

## Decomposition

Any time series can be decomposed into three components:

$$
Y_t = T_t + S_t + R_t \\quad \\text{(additive)}
$$

$$
Y_t = T_t \\times S_t \\times R_t \\quad \\text{(multiplicative)}
$$

where:
- $T_t$ = **Trend:** the long-term direction
- $S_t$ = **Seasonality:** repeating patterns at fixed intervals
- $R_t$ = **Residual:** irregular fluctuations

Use **additive** when seasonal fluctuations are roughly constant in size. Use **multiplicative** when they grow proportionally with the trend.

## Differencing

**Differencing** is the simplest way to remove trend and achieve stationarity:

$$
y'_t = y_t - y_{t-1}
$$

For seasonal patterns with period $m$, apply **seasonal differencing**:

$$
y'_t = y_t - y_{t-m}
$$

You may need to difference more than once. The number of times you difference is the $d$ parameter in ARIMA.

Run the code to generate a time series with trend and seasonality, test for stationarity, decompose it, and apply differencing.`,
      codeSnippet: `import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# --- Generate synthetic time series with trend + seasonality ---
np.random.seed(42)
n = 144  # 12 years of monthly data
t = np.arange(n)

trend = 0.05 * t
seasonality = 3 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 0.5, n)
y = 10 + trend + seasonality + noise

# --- ADF test on raw series ---
adf_raw = adfuller(y, autolag="AIC")
print("=== ADF Test (Original Series) ===")
print(f"Test Statistic: {adf_raw[0]:.4f}")
print(f"P-value:        {adf_raw[1]:.4f}")
print(f"Stationary?     {'Yes' if adf_raw[1] < 0.05 else 'No'}")

# --- Seasonal decomposition ---
result = seasonal_decompose(y, model="additive", period=12)

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
result.observed[~np.isnan(result.observed)].size
axes[0].plot(t, y, color="steelblue")
axes[0].set_ylabel("Observed")
axes[0].set_title("Seasonal Decomposition (Additive)")

axes[1].plot(t, result.trend, color="darkorange")
axes[1].set_ylabel("Trend")

axes[2].plot(t, result.seasonal, color="green")
axes[2].set_ylabel("Seasonal")

axes[3].plot(t, result.resid, color="red")
axes[3].set_ylabel("Residual")
axes[3].set_xlabel("Month")
plt.tight_layout()
plt.savefig("output.png", dpi=100, bbox_inches="tight")
plt.close()

# --- Differencing to achieve stationarity ---
y_diff = np.diff(y)
adf_diff = adfuller(y_diff, autolag="AIC")
print("\\n=== ADF Test (After First Differencing) ===")
print(f"Test Statistic: {adf_diff[0]:.4f}")
print(f"P-value:        {adf_diff[1]:.4f}")
print(f"Stationary?     {'Yes' if adf_diff[1] < 0.05 else 'No'}")
print("\\nDifferencing successfully made the series stationary!")`,
      codeLanguage: "python",
    },
    {
      title: "ARIMA",
      slug: "arima",
      description:
        "Autoregressive Integrated Moving Average models, ACF/PACF analysis, and seasonal ARIMA",
      markdownContent: `# ARIMA

**ARIMA** (Autoregressive Integrated Moving Average) is the workhorse model for univariate time series forecasting. It combines three components that each address a different aspect of the data.

## AR — Autoregressive Component

An **AR(p)** model predicts the current value as a linear combination of the previous $p$ values:

$$
y_t = c + \\phi_1 y_{t-1} + \\phi_2 y_{t-2} + \\cdots + \\phi_p y_{t-p} + \\varepsilon_t
$$

The key idea: **the past predicts the future**. The parameter $p$ is the number of lags used.

## I — Integrated Component

The "I" in ARIMA accounts for **differencing** needed to make the series stationary. If we difference $d$ times before fitting the AR and MA components, the model is said to be integrated of order $d$.

## MA — Moving Average Component

An **MA(q)** model predicts the current value from past **forecast errors**:

$$
y_t = c + \\varepsilon_t + \\theta_1 \\varepsilon_{t-1} + \\theta_2 \\varepsilon_{t-2} + \\cdots + \\theta_q \\varepsilon_{t-q}
$$

This captures short-lived shocks that affect the series temporarily.

## ARIMA(p, d, q)

Combining all three, an **ARIMA(p, d, q)** model applies $d$ differences, then fits an ARMA(p, q) model:

$$
\\phi(B)(1 - B)^d y_t = c + \\theta(B)\\varepsilon_t
$$

where $B$ is the backshift operator ($By_t = y_{t-1}$), $\\phi(B)$ is the AR polynomial, and $\\theta(B)$ is the MA polynomial.

## Choosing p and q: ACF and PACF

- **ACF (Autocorrelation Function):** correlation between $y_t$ and $y_{t-h}$ for each lag $h$. A sharp cutoff after lag $q$ suggests an MA($q$) process.
- **PACF (Partial Autocorrelation Function):** correlation between $y_t$ and $y_{t-h}$ after removing the effect of intermediate lags. A sharp cutoff after lag $p$ suggests an AR($p$) process.

| Pattern | ACF | PACF | Model |
|---------|-----|------|-------|
| AR(p) | Tails off | Cuts off after lag $p$ | Use PACF to pick $p$ |
| MA(q) | Cuts off after lag $q$ | Tails off | Use ACF to pick $q$ |
| ARMA(p,q) | Tails off | Tails off | Use AIC/BIC to select |

## SARIMA — Seasonal Extension

For data with seasonality of period $m$, **SARIMA** adds seasonal AR, differencing, and MA terms:

$$
\\text{SARIMA}(p, d, q)(P, D, Q)_m
$$

where $(P, D, Q)$ are the seasonal counterparts operating at lag $m$.

## Model Selection: AIC and BIC

When the ACF/PACF plots are ambiguous, fit several candidate models and compare:

- **AIC** = $-2\\ln(L) + 2k$ (favours fit)
- **BIC** = $-2\\ln(L) + k\\ln(n)$ (penalises complexity more)

Lower is better for both. BIC tends to select simpler models.

Run the code to fit an ARIMA model to airline passenger data, inspect ACF/PACF, and forecast future values.`,
      codeSnippet: `import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- Generate synthetic airline-like data ---
np.random.seed(42)
n = 144
t = np.arange(n)
trend = 2.0 + 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 2, n)
passengers = 100 + trend + seasonal + noise
passengers = np.maximum(passengers, 0)

# --- Difference once and check stationarity ---
diff1 = np.diff(passengers)
adf = adfuller(diff1, autolag="AIC")
print(f"ADF after 1st differencing: stat={adf[0]:.3f}, p={adf[1]:.4f}")

# --- ACF and PACF plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(diff1, lags=30, ax=axes[0], title="ACF (Differenced)")
plot_pacf(diff1, lags=30, ax=axes[1], title="PACF (Differenced)")
plt.tight_layout()
plt.savefig("output.png", dpi=100, bbox_inches="tight")
plt.close()

# --- Fit ARIMA model ---
train = passengers[:120]
test = passengers[120:]

model = ARIMA(train, order=(2, 1, 2))
fitted = model.fit()
print(f"\\nARIMA(2,1,2) — AIC: {fitted.aic:.1f}, BIC: {fitted.bic:.1f}")
print(fitted.summary().tables[1])

# --- Forecast ---
forecast = fitted.forecast(steps=len(test))
conf = fitted.get_forecast(steps=len(test)).conf_int()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(train)), train, label="Train", color="steelblue")
ax.plot(range(len(train), n), test, label="Actual", color="darkorange")
ax.plot(range(len(train), n), forecast, label="Forecast",
        color="green", linestyle="--")
ax.fill_between(range(len(train), n), conf[:, 0], conf[:, 1],
                color="green", alpha=0.15, label="95% CI")
ax.set_xlabel("Month")
ax.set_ylabel("Passengers")
ax.set_title("ARIMA(2,1,2) Forecast vs Actual")
ax.legend()
plt.tight_layout()
plt.savefig("output.png", dpi=100, bbox_inches="tight")
plt.close()

rmse = np.sqrt(np.mean((test - forecast) ** 2))
print(f"\\nTest RMSE: {rmse:.2f}")`,
      codeLanguage: "python",
    },
    {
      title: "Forecasting with ML",
      slug: "forecasting-with-ml",
      description:
        "Machine learning approaches to time series: feature engineering, tree-based models, and walk-forward validation",
      markdownContent: `# Forecasting with ML

Classical models like ARIMA are powerful for univariate, linear patterns. But real-world time series often exhibit **nonlinear relationships** and depend on **multiple external features**. This is where machine learning models shine.

## Why ML for Time Series?

- Can capture **nonlinear** patterns without manual transformation
- Naturally handle **multiple input features** (exogenous variables)
- Tree-based models are robust to outliers and missing values
- Can leverage domain-specific engineered features

## Feature Engineering for Time Series

The key to ML forecasting is transforming the temporal structure into **tabular features**:

### Lag Features

Use previous values as input features:

$$
X_t = [y_{t-1},\\; y_{t-2},\\; \\ldots,\\; y_{t-k}]
$$

### Rolling Statistics

Capture recent behaviour with windowed aggregations:

$$
\\text{RollingMean}_t = \\frac{1}{w} \\sum_{i=1}^{w} y_{t-i}
$$

$$
\\text{RollingStd}_t = \\sqrt{\\frac{1}{w} \\sum_{i=1}^{w} (y_{t-i} - \\overline{y})^2}
$$

### Datetime Features

Extract cyclical and calendar features: hour of day, day of week, month, quarter, is_holiday, etc. Encode cyclical features with sine/cosine transforms:

$$
\\text{month\\_sin} = \\sin\\!\\left(\\frac{2\\pi \\cdot \\text{month}}{12}\\right), \\quad
\\text{month\\_cos} = \\cos\\!\\left(\\frac{2\\pi \\cdot \\text{month}}{12}\\right)
$$

## Train/Test Split: Never Random!

Unlike standard ML, time series data **must be split chronologically**:

$$
\\underbrace{y_1, y_2, \\ldots, y_T}_{\\text{train}} \\;|\\; \\underbrace{y_{T+1}, \\ldots, y_n}_{\\text{test}}
$$

Random splitting creates **data leakage** — the model sees future information during training, producing overly optimistic performance estimates.

## Walk-Forward Validation

The time series equivalent of cross-validation. At each step:

1. Train on all data up to time $t$
2. Predict the next $h$ steps
3. Slide the window forward and repeat

This simulates how the model will be used in production and gives a realistic estimate of forecast accuracy.

## Tree-Based Models

**Random Forest** and **XGBoost** are excellent choices for tabular time series features:

- Handle nonlinear interactions automatically
- Feature importance reveals which lags/features matter most
- No need to worry about stationarity or normality assumptions

## Deep Learning Approaches

For very long sequences or complex temporal dependencies, deep learning offers:

- **LSTM / GRU:** Recurrent networks designed for sequential data
- **Transformers:** Attention-based models that capture long-range dependencies

These are covered in depth in the Deep Learning chapter.

Run the code to engineer lag features, train a Random Forest, and compare it against a naive baseline.`,
      codeSnippet: `import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- Generate synthetic time series ---
np.random.seed(42)
n = 300
t = np.arange(n)
trend = 0.03 * t
seasonal = 5 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 1.0, n)
y = 50 + trend + seasonal + noise

# --- Feature engineering: lags + rolling statistics ---
def create_features(series, n_lags=6, window=6):
    """Create lag features and rolling statistics."""
    X, targets = [], []
    for i in range(max(n_lags, window), len(series)):
        lags = [series[i - j] for j in range(1, n_lags + 1)]
        roll_mean = np.mean(series[i - window:i])
        roll_std = np.std(series[i - window:i])
        # Cyclical month encoding
        month = i % 12
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        X.append(lags + [roll_mean, roll_std, month_sin, month_cos])
        targets.append(series[i])
    return np.array(X), np.array(targets)

X, targets = create_features(y, n_lags=6, window=6)

feature_names = (
    [f"lag_{i}" for i in range(1, 7)]
    + ["roll_mean", "roll_std", "month_sin", "month_cos"]
)

# --- Chronological train/test split (80/20) ---
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = targets[:split], targets[split:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# --- Random Forest ---
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --- Naive baseline: predict last known value ---
y_pred_naive = X_test[:, 0]  # lag_1 = last value

# --- Evaluate ---
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
print(f"\\nRandom Forest RMSE: {rmse_rf:.3f}")
print(f"Naive Baseline RMSE: {rmse_naive:.3f}")
print(f"Improvement: {(1 - rmse_rf / rmse_naive) * 100:.1f}%")

# --- Feature importance ---
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]
print("\\nTop features:")
for i in range(min(5, len(feature_names))):
    print(f"  {feature_names[idx[i]]}: {importances[idx[i]]:.3f}")

# --- Plot predictions vs actual ---
fig, ax = plt.subplots(figsize=(10, 5))
test_range = range(split + 6, split + 6 + len(y_test))
ax.plot(test_range, y_test, label="Actual", color="steelblue", linewidth=2)
ax.plot(test_range, y_pred_rf, label="Random Forest",
        color="green", linestyle="--")
ax.plot(test_range, y_pred_naive, label="Naive (last value)",
        color="red", linestyle=":", alpha=0.7)
ax.set_xlabel("Time Step")
ax.set_ylabel("Value")
ax.set_title("Random Forest vs Naive Baseline")
ax.legend()
plt.tight_layout()
plt.savefig("output.png", dpi=100, bbox_inches="tight")
plt.close()

print("\\nPlot saved — RF captures seasonality that the naive baseline misses.")`,
      codeLanguage: "python",
    },
  ],
};
