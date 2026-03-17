import type { Chapter } from "../types";

export const eda: Chapter = {
  title: "Exploratory Data Analysis",
  slug: "eda",
  pages: [
    {
      title: "Descriptive Statistics",
      slug: "descriptive-statistics",
      description:
        "Mean, median, mode, variance, standard deviation, percentiles, skewness, and kurtosis",
      markdownContent: `# Descriptive Statistics

**Descriptive statistics** summarize a dataset with a handful of numbers, giving you an immediate feel for its centre, spread, and shape before any modelling begins.

## Measures of Central Tendency

- **Mean** — the arithmetic average: $\\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n} x_i$
- **Median** — the middle value when data is sorted. Robust to outliers.
- **Mode** — the most frequently occurring value, especially useful for categorical data.

## Measures of Spread

**Variance** quantifies how far values deviate from the mean:

$$
\\sigma^2 = \\frac{1}{n}\\sum_{i=1}^{n}(x_i - \\bar{x})^2
$$

The sample variance uses $n - 1$ (Bessel's correction) to produce an unbiased estimate:

$$
s^2 = \\frac{1}{n-1}\\sum_{i=1}^{n}(x_i - \\bar{x})^2
$$

**Standard deviation** $\\sigma = \\sqrt{\\sigma^2}$ is in the same units as the data, making it easier to interpret.

**Percentiles** divide sorted data into 100 equal parts. The 25th, 50th, and 75th percentiles form the **interquartile range (IQR)**, which captures the middle 50 % of the distribution.

## Shape of a Distribution

**Skewness** measures asymmetry:

$$
\\text{Skewness} = \\frac{1}{n}\\sum_{i=1}^{n}\\left(\\frac{x_i - \\bar{x}}{s}\\right)^3
$$

- Skewness $= 0$: symmetric (e.g., normal distribution)
- Skewness $> 0$: right-skewed (long right tail)
- Skewness $< 0$: left-skewed (long left tail)

**Kurtosis** measures the heaviness of tails relative to a normal distribution:

$$
\\text{Kurtosis} = \\frac{1}{n}\\sum_{i=1}^{n}\\left(\\frac{x_i - \\bar{x}}{s}\\right)^4
$$

Excess kurtosis (kurtosis $- 3$) equals zero for a normal distribution. Positive excess kurtosis means heavier tails; negative means lighter tails.

## Why This Matters

Descriptive statistics are the first thing you compute on any new dataset. They reveal data quality issues (unexpected ranges, missing values), inform feature engineering decisions, and help you choose appropriate models and preprocessing steps.

Run the code to load the Iris dataset and compute all descriptive statistics using both pandas shortcuts and manual calculations.`,
      codeSnippet: `import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris

# Load the Iris dataset into a DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# --- pandas .describe() shortcut ---
print("=== pandas .describe() ===")
print(df.describe().round(4))

# --- Manual calculations for one feature ---
feature = "sepal length (cm)"
x = df[feature]

print(f"\\n=== Manual Stats for '{feature}' ===")
print(f"Mean:               {x.mean():.4f}")
print(f"Median:             {x.median():.4f}")
print(f"Mode:               {x.mode().values[0]:.4f}")
print(f"Variance (sample):  {x.var():.4f}")
print(f"Std Dev (sample):   {x.std():.4f}")
print(f"25th percentile:    {x.quantile(0.25):.4f}")
print(f"75th percentile:    {x.quantile(0.75):.4f}")
print(f"IQR:                {x.quantile(0.75) - x.quantile(0.25):.4f}")
print(f"Skewness:           {x.skew():.4f}")
print(f"Kurtosis (excess):  {x.kurtosis():.4f}")

# --- Verify with scipy ---
print(f"\\n=== scipy.stats verification ===")
print(f"Skewness (scipy):   {stats.skew(x):.4f}")
print(f"Kurtosis (scipy):   {stats.kurtosis(x):.4f}")`,
      codeLanguage: "python",
    },
    {
      title: "Distribution Analysis",
      slug: "distribution-analysis",
      description:
        "Histograms, KDE plots, box plots, violin plots, and normality checking with QQ plots",
      markdownContent: `# Distribution Analysis

Understanding how your features are distributed is crucial for choosing the right models and transformations. **Distribution analysis** uses visual and statistical tools to characterise the shape, centre, and spread of your data.

## Histograms

A histogram divides the range of a variable into equal-width **bins** and counts how many observations fall into each bin. The choice of bin count matters — too few bins hide structure, too many create noise.

## Kernel Density Estimation (KDE)

KDE produces a smooth, continuous estimate of the probability density function. It places a small **kernel** (typically Gaussian) at each data point and sums them:

$$
\\hat{f}(x) = \\frac{1}{nh}\\sum_{i=1}^{n} K\\!\\left(\\frac{x - x_i}{h}\\right)
$$

where $h$ is the **bandwidth** controlling smoothness. KDE is often overlaid on histograms for a cleaner picture.

## Box Plots and Violin Plots

A **box plot** shows the median, IQR (box), and whiskers extending to $1.5 \\times \\text{IQR}$. Points beyond the whiskers are marked as potential outliers.

A **violin plot** combines a box plot with a mirrored KDE, showing the full shape of the distribution — useful for spotting bimodality or skew that a box plot alone would miss.

## Normal Distribution

Many statistical methods assume data follows a **normal** (Gaussian) distribution:

$$
f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\!\\left(-\\frac{(x - \\mu)^2}{2\\sigma^2}\\right)
$$

In practice, real data is often skewed or heavy-tailed.

## QQ Plots

A **quantile-quantile (QQ) plot** compares observed quantiles against theoretical quantiles from a reference distribution (usually normal). If the data follows the reference distribution, points fall along a straight diagonal line. Systematic deviations reveal skew, heavy tails, or other departures from normality.

## Why This Matters

Distribution analysis drives decisions about:
- Whether to apply log or power transforms to reduce skew
- Which models are appropriate (e.g., linear models assume roughly normal residuals)
- Whether outlier treatment is needed

Run the code to generate data from different distributions and visualise them with histograms, KDE, box plots, and QQ plots.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Generate data from three distributions
normal_data = np.random.normal(loc=50, scale=10, size=500)
skewed_data = np.random.exponential(scale=10, size=500)
bimodal_data = np.concatenate([
    np.random.normal(30, 5, 250),
    np.random.normal(60, 5, 250)
])

datasets = {
    "Normal": normal_data,
    "Right-Skewed (Exp)": skewed_data,
    "Bimodal": bimodal_data,
}

# --- Histograms + KDE ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, data) in zip(axes, datasets.items()):
    ax.hist(data, bins=30, density=True, alpha=0.6, color="steelblue",
            edgecolor="white")
    xs = np.linspace(data.min() - 5, data.max() + 5, 300)
    kde = stats.gaussian_kde(data)
    ax.plot(xs, kde(xs), color="coral", linewidth=2)
    ax.set_title(name)
    ax.set_ylabel("Density")
fig.suptitle("Histograms with KDE Overlay", fontsize=14)
plt.tight_layout()
plt.show()

# --- Box plots ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot(list(datasets.values()), labels=list(datasets.keys()),
           patch_artist=True,
           boxprops=dict(facecolor="steelblue", alpha=0.6))
ax.set_title("Box Plots")
ax.set_ylabel("Value")
plt.tight_layout()
plt.show()

# --- QQ plot for normal data vs skewed data ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, data) in zip(axes, [("Normal", normal_data),
                                     ("Right-Skewed", skewed_data)]):
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f"QQ Plot — {name}")
plt.tight_layout()
plt.show()

# Print summary
for name, data in datasets.items():
    print(f"{name:>20s}  mean={data.mean():.2f}  std={data.std():.2f}"
          f"  skew={stats.skew(data):.2f}  kurtosis={stats.kurtosis(data):.2f}")`,
      codeLanguage: "python",
    },
    {
      title: "Outlier Detection",
      slug: "outlier-detection",
      description:
        "IQR method, Z-score method, isolation forest, and impact of outliers on models",
      markdownContent: `# Outlier Detection

**Outliers** are data points that differ significantly from the rest of the dataset. They can be legitimate extreme values or errors. Identifying them is critical because outliers can distort summary statistics, violate model assumptions, and degrade prediction performance.

## IQR Method

The **interquartile range** method flags any point outside the "fences":

$$
\\text{Lower fence} = Q_1 - 1.5 \\times \\text{IQR}
$$
$$
\\text{Upper fence} = Q_3 + 1.5 \\times \\text{IQR}
$$

where $\\text{IQR} = Q_3 - Q_1$. This method is robust because it relies on percentiles rather than the mean.

## Z-Score Method

The **Z-score** standardises each value by subtracting the mean and dividing by the standard deviation:

$$
z_i = \\frac{x_i - \\bar{x}}{s}
$$

Points with $|z_i| > 3$ (or another chosen threshold) are flagged as outliers. This method is simple but sensitive to the outliers themselves, since they inflate $\\bar{x}$ and $s$.

## Isolation Forest

**Isolation Forest** is a tree-based anomaly detection algorithm. The intuition: outliers are few and different, so they are easier to isolate. The algorithm builds random trees that partition the data with random splits. Outliers require fewer splits to be isolated, producing shorter average path lengths.

The anomaly score for a point $x$ is:

$$
s(x, n) = 2^{-\\frac{E[h(x)]}{c(n)}}
$$

where $E[h(x)]$ is the average path length and $c(n)$ is a normalisation factor. Scores close to 1 indicate anomalies; scores near 0.5 indicate normal points.

## Impact on Models

Outliers affect:
- **Linear regression** — a single extreme point can tilt the regression line, since MSE penalises large errors quadratically.
- **Mean and variance** — both are pulled towards outliers, distorting descriptive statistics.
- **Distance-based models** (k-NN, k-means) — outliers warp distance calculations and cluster assignments.

Robust alternatives (median, MAD, Huber loss) reduce outlier influence without removing data.

## Why This Matters

Blindly removing outliers discards information. The goal is to understand *why* outliers exist, then decide whether to keep, transform, or remove them based on domain knowledge.

Run the code to inject outliers into synthetic data and detect them with all three methods.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest

np.random.seed(42)

# Generate clean data + inject outliers
clean = np.random.normal(loc=50, scale=5, size=200)
outliers = np.array([10, 12, 90, 95, 100])
data = np.concatenate([clean, outliers])

print(f"Dataset: {len(data)} points ({len(clean)} normal + "
      f"{len(outliers)} injected outliers)")

# --- Method 1: IQR ---
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
iqr_mask = (data < lower) | (data > upper)
print(f"\\n=== IQR Method ===")
print(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
print(f"Fences: [{lower:.2f}, {upper:.2f}]")
print(f"Outliers detected: {iqr_mask.sum()}")

# --- Method 2: Z-Score ---
z_scores = np.abs(stats.zscore(data))
z_threshold = 3
z_mask = z_scores > z_threshold
print(f"\\n=== Z-Score Method (threshold={z_threshold}) ===")
print(f"Outliers detected: {z_mask.sum()}")

# --- Method 3: Isolation Forest ---
iso = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso.fit_predict(data.reshape(-1, 1))
iso_mask = iso_labels == -1
print(f"\\n=== Isolation Forest ===")
print(f"Outliers detected: {iso_mask.sum()}")

# --- Visualise ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
methods = [("IQR", iqr_mask), ("Z-Score", z_mask),
           ("Isolation Forest", iso_mask)]
for ax, (name, mask) in zip(axes, methods):
    ax.scatter(range(len(data)), data, c=mask.astype(int),
               cmap="coolwarm", alpha=0.7, edgecolors="k", s=30)
    ax.set_title(f"{name} ({mask.sum()} outliers)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    if name == "IQR":
        ax.axhline(lower, ls="--", color="gray", label="Fences")
        ax.axhline(upper, ls="--", color="gray")
        ax.legend()
fig.suptitle("Outlier Detection Comparison", fontsize=14)
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Correlation Analysis",
      slug: "correlation-analysis",
      description:
        "Pearson, Spearman, and Kendall correlation, heatmaps, and the correlation-causation distinction",
      markdownContent: `# Correlation Analysis

**Correlation** measures the strength and direction of the relationship between two variables. Understanding correlations helps you select features, spot redundancy, and avoid multicollinearity in models.

## Pearson Correlation

The **Pearson coefficient** $r$ measures the *linear* relationship between two variables:

$$
r = \\frac{\\sum_{i=1}^{n}(x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_{i=1}^{n}(x_i - \\bar{x})^2 \\sum_{i=1}^{n}(y_i - \\bar{y})^2}}
$$

$r = 1$: perfect positive linear relationship; $r = -1$: perfect negative; $r = 0$: no linear relationship. Pearson is sensitive to outliers and only captures linear associations.

## Spearman Correlation

**Spearman's** $\\rho$ converts values to ranks before computing the Pearson coefficient on those ranks. It captures any *monotonic* relationship (linear or not) and is robust to outliers.

$$
\\rho = 1 - \\frac{6 \\sum d_i^2}{n(n^2 - 1)}
$$

where $d_i$ is the difference between ranks of corresponding values. If a relationship is monotonically increasing but non-linear, Spearman will detect it while Pearson may underestimate it.

## Kendall Correlation

**Kendall's** $\\tau$ counts concordant and discordant pairs:

$$
\\tau = \\frac{(\\text{concordant pairs}) - (\\text{discordant pairs})}{\\binom{n}{2}}
$$

It is more robust than Spearman for small samples and handles ties well, but is computationally more expensive.

## Correlation Heatmaps

For datasets with many features, a **heatmap** of the correlation matrix provides a quick visual summary. Bright spots reveal strong correlations that may indicate:
- **Redundant features** — highly correlated features carry similar information.
- **Multicollinearity** — correlated predictors inflate variance in linear models.
- **Potential leakage** — a feature that is suspiciously correlated with the target.

## Correlation $\\neq$ Causation

A high correlation between $X$ and $Y$ does **not** mean $X$ causes $Y$. Common pitfalls:
- **Confounding variable** — a hidden variable $Z$ drives both $X$ and $Y$.
- **Reverse causation** — $Y$ might cause $X$.
- **Spurious correlation** — coincidence, especially with many variables or time-series data.

Establishing causation requires controlled experiments or causal inference techniques (e.g., instrumental variables, do-calculus).

## Why This Matters

Correlation analysis is a fast, essential step in EDA. It guides feature selection, flags data issues, and informs modelling decisions — but only when interpreted carefully.

Run the code to compute Pearson, Spearman, and Kendall correlations on a multi-feature dataset and visualise the results as a heatmap.`,
      codeSnippet: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# --- Compute correlation matrices ---
pearson = df.corr(method="pearson")
spearman = df.corr(method="spearman")
kendall = df.corr(method="kendall")

print("=== Pearson Correlation ===")
print(pearson.round(4))
print("\\n=== Spearman Correlation ===")
print(spearman.round(4))
print("\\n=== Kendall Correlation ===")
print(kendall.round(4))

# --- Heatmaps ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
matrices = [("Pearson", pearson), ("Spearman", spearman),
            ("Kendall", kendall)]

for ax, (name, corr) in zip(axes, matrices):
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels([c.replace(" (cm)", "") for c in corr.columns],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([c.replace(" (cm)", "") for c in corr.columns],
                       fontsize=9)
    # Annotate cells
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(corr.values[i, j]) > 0.6
                    else "black")
    ax.set_title(name)

fig.colorbar(im, ax=axes, shrink=0.8, label="Correlation")
fig.suptitle("Correlation Heatmaps — Iris Dataset", fontsize=14)
plt.tight_layout()
plt.show()

# --- Highlight strong correlations ---
print("\\n=== Strong Pearson Correlations (|r| > 0.8) ===")
for i in range(len(pearson)):
    for j in range(i + 1, len(pearson)):
        r = pearson.iloc[i, j]
        if abs(r) > 0.8:
            print(f"  {pearson.index[i]} <-> {pearson.columns[j]}: "
                  f"r = {r:.4f}")`,
      codeLanguage: "python",
    },
  ],
};
