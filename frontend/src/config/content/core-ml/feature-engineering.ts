import type { Chapter } from "../types";

export const featureEngineering: Chapter = {
  title: "Feature Engineering",
  slug: "feature-engineering",
  pages: [
    {
      title: "Missing Value Handling",
      slug: "missing-value-handling",
      description:
        "Types of missingness, imputation strategies, and practical tools for handling missing data",
      markdownContent: `# Missing Value Handling

Real-world datasets almost always contain **missing values**. How you handle them has a direct impact on model performance. Before choosing a strategy, it helps to understand *why* values are missing.

## Types of Missingness

**MCAR (Missing Completely At Random)** — the probability of a value being missing is the same for all observations. Example: a sensor randomly fails regardless of the reading.

**MAR (Missing At Random)** — missingness depends on *observed* data but not the missing value itself. Example: younger respondents skip an income question more often, but missingness doesn't depend on their actual income.

**MNAR (Missing Not At Random)** — missingness depends on the *unobserved* value. Example: high earners refuse to report income. MNAR is the hardest to handle correctly and may require domain-specific modelling.

## Imputation Strategies

| Strategy | When to Use | Limitation |
|---|---|---|
| **Drop rows** | Very few missing values, MCAR | Loses data, biases sample if not MCAR |
| **Mean / Median** | Numerical features, quick baseline | Ignores feature relationships, reduces variance |
| **Mode** | Categorical features | Same issues as mean imputation |
| **KNN Imputation** | Features are correlated | Slower, sensitive to scale and $k$ |

### Mean / Median Imputation

Replace missing values with the column mean (or median for skewed data):

$$
x_{\\text{imputed}} = \\bar{x} = \\frac{1}{n_{\\text{obs}}} \\sum_{i \\in \\text{obs}} x_i
$$

This is fast but assumes values are MCAR and reduces the natural variance of the feature.

### KNN Imputation

For each missing entry, find the $k$ nearest neighbours (using observed features) and impute with their average:

$$
x_{\\text{imputed}} = \\frac{1}{k} \\sum_{j=1}^{k} x_j^{(\\text{neighbour})}
$$

KNN imputation preserves feature correlations better than simple mean imputation, but requires scaled features and is more computationally expensive.

## Best Practices

- Always examine the **pattern** of missingness before choosing a strategy.
- Create a binary **missingness indicator** column — the fact that a value is missing can itself be predictive.
- Use **median** over mean for skewed distributions to avoid the influence of outliers.
- For production pipelines, fit the imputer on training data only to avoid data leakage.

Run the code to see SimpleImputer and KNNImputer in action on a dataset with missing values.`,
      codeSnippet: `import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

# Create sample data with missing values
np.random.seed(42)
data = pd.DataFrame({
    "age": [25, 30, np.nan, 45, 50, np.nan, 35, 60, 28, 40],
    "income": [40000, np.nan, 55000, 70000, np.nan, 65000, 48000, 80000, np.nan, 60000],
    "score": [85, 90, 78, np.nan, 92, 88, np.nan, 95, 80, np.nan],
})

print("=== Original Data ===")
print(data.to_string())
print(f"\\nMissing values per column:\\n{data.isnull().sum()}")

# Strategy 1: Mean imputation
mean_imp = SimpleImputer(strategy="mean")
data_mean = pd.DataFrame(mean_imp.fit_transform(data), columns=data.columns)

print("\\n=== After Mean Imputation ===")
print(data_mean.to_string())

# Strategy 2: Median imputation
median_imp = SimpleImputer(strategy="median")
data_median = pd.DataFrame(median_imp.fit_transform(data), columns=data.columns)

print("\\n=== After Median Imputation ===")
print(data_median.to_string())

# Strategy 3: KNN imputation (k=3)
knn_imp = KNNImputer(n_neighbors=3)
data_knn = pd.DataFrame(knn_imp.fit_transform(data), columns=data.columns)

print("\\n=== After KNN Imputation (k=3) ===")
print(data_knn.to_string())

# Compare imputed values for 'income' (originally missing at rows 1, 4, 8)
print("\\n=== Comparison of Imputed 'income' Values ===")
missing_idx = data["income"].isnull()
comparison = pd.DataFrame({
    "Mean": data_mean.loc[missing_idx, "income"].values,
    "Median": data_median.loc[missing_idx, "income"].values,
    "KNN": data_knn.loc[missing_idx, "income"].values,
}, index=data.index[missing_idx])
print(comparison.to_string())`,
      codeLanguage: "python",
    },
    {
      title: "Categorical Encoding",
      slug: "categorical-encoding",
      description:
        "One-hot, ordinal, and target encoding strategies for converting categories to numbers",
      markdownContent: `# Categorical Encoding

Most ML algorithms require numerical input, so categorical features must be **encoded** into numbers. The right encoding depends on the nature of the category and the model you plan to use.

## One-Hot Encoding

Creates a binary column for each category. A sample gets a 1 in the column matching its category and 0 everywhere else.

$$
\\text{color} = \\{\\text{red, green, blue}\\} \\rightarrow [x_{\\text{red}},\\; x_{\\text{green}},\\; x_{\\text{blue}}]
$$

**When to use:** Nominal (unordered) categories with **low cardinality** (< ~15 values). Works well with linear models that would otherwise impose a false ordering.

**Watch out:** High-cardinality features create many sparse columns — this increases memory use and can hurt tree-based models.

## Ordinal Encoding

Maps each category to a single integer: $\\{\\text{low}=0,\\; \\text{medium}=1,\\; \\text{high}=2\\}$.

**When to use:** Categories with a natural **order** (e.g., education level, size). Tree-based models (Random Forest, XGBoost) handle ordinal encoding well even for nominal features because they split on thresholds.

**Watch out:** Linear models will treat the integers as continuous, implying equal spacing and order — don't use this for nominal categories with linear models.

## Target Encoding

Replaces each category with the **mean of the target variable** for that category:

$$
x_{\\text{encoded}} = \\frac{1}{n_c} \\sum_{i \\in \\text{category } c} y_i
$$

**When to use:** High-cardinality features where one-hot is impractical (e.g., zip codes, product IDs).

**Watch out:** Target encoding leaks information from the target into the features. Always use **cross-validated** target encoding or compute it only on training folds to avoid overfitting.

## Quick Reference

| Encoding | Best For | Model Compatibility |
|---|---|---|
| One-Hot | Nominal, low cardinality | All models |
| Ordinal | Ordinal, any cardinality | Trees (safe), linear (only if truly ordinal) |
| Target | High cardinality | All models (with regularisation) |

Run the code to apply all three encodings to a sample dataset and compare their outputs.`,
      codeSnippet: `import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Sample dataset
df = pd.DataFrame({
    "color": ["red", "blue", "green", "blue", "red", "green", "red", "blue"],
    "size": ["S", "M", "L", "XL", "M", "S", "L", "XL"],
    "city": ["NYC", "LA", "NYC", "SF", "LA", "SF", "NYC", "LA"],
    "target": [10, 25, 15, 30, 20, 35, 12, 28],
})

print("=== Original Data ===")
print(df.to_string())

# 1. One-Hot Encoding (color)
ohe = OneHotEncoder(sparse_output=False)
color_ohe = ohe.fit_transform(df[["color"]])
color_df = pd.DataFrame(color_ohe, columns=ohe.get_feature_names_out())
print("\\n=== One-Hot Encoding (color) ===")
print(color_df.to_string())

# 2. Ordinal Encoding (size — has natural order)
oe = OrdinalEncoder(categories=[["S", "M", "L", "XL"]])
df["size_ordinal"] = oe.fit_transform(df[["size"]])
print("\\n=== Ordinal Encoding (size) ===")
print(df[["size", "size_ordinal"]].to_string())

# 3. Target Encoding (city — manual cross-validated style)
# Simple demonstration: global mean smoothed target encoding
global_mean = df["target"].mean()
city_means = df.groupby("city")["target"].mean()
smoothing = 3  # regularisation parameter
city_counts = df.groupby("city")["target"].count()

# Smoothed target encoding: blend category mean with global mean
smooth_enc = (city_counts * city_means + smoothing * global_mean) / (city_counts + smoothing)
df["city_target_enc"] = df["city"].map(smooth_enc)

print("\\n=== Target Encoding (city) ===")
print(f"Global mean: {global_mean:.2f}")
print(f"\\nCategory means and smoothed encodings:")
enc_summary = pd.DataFrame({
    "count": city_counts,
    "raw_mean": city_means.round(2),
    "smoothed": smooth_enc.round(2),
})
print(enc_summary.to_string())
print(f"\\n{df[['city', 'target', 'city_target_enc']].to_string()}")`,
      codeLanguage: "python",
    },
    {
      title: "Feature Scaling",
      slug: "feature-scaling",
      description:
        "StandardScaler, MinMaxScaler, and RobustScaler — when scaling matters and how each works",
      markdownContent: `# Feature Scaling

Features often live on very different scales. A person's age (20–80) and income (20,000–200,000) differ by orders of magnitude. Many algorithms are **sensitive to scale**, so transforming features to a common range is critical.

## When Scaling Matters

| Needs Scaling | Does NOT Need Scaling |
|---|---|
| Linear / Logistic Regression | Decision Trees |
| SVM | Random Forest |
| KNN | Gradient Boosted Trees |
| Neural Networks | Naive Bayes |
| PCA / clustering | |

Distance-based and gradient-based methods are affected by feature magnitude. Tree-based models split on thresholds and are scale-invariant.

## StandardScaler (Z-score)

Centres each feature at zero with unit variance:

$$
z = \\frac{x - \\mu}{\\sigma}
$$

where $\\mu$ is the mean and $\\sigma$ the standard deviation. After scaling, most values lie in $[-3, 3]$. **Best when features are roughly Gaussian.**

## MinMaxScaler

Rescales features to a fixed range $[0, 1]$:

$$
x' = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}
$$

Preserves the shape of the distribution but is **sensitive to outliers** — a single extreme value compresses all other values into a narrow band.

## RobustScaler

Uses the **median** and **interquartile range (IQR)** instead:

$$
x' = \\frac{x - \\text{median}}{\\text{IQR}}
$$

where $\\text{IQR} = Q_3 - Q_1$. This makes it **robust to outliers** — extreme values do not heavily influence the scaling.

## Best Practices

- Always **fit the scaler on training data only**, then transform both train and test sets.
- Use **RobustScaler** when outliers are present but shouldn't dominate.
- For algorithms that assume normality (e.g., PCA), prefer **StandardScaler**.
- For bounded outputs (e.g., pixel values for neural nets), prefer **MinMaxScaler**.

Run the code to compare all three scalers on skewed data with outliers.`,
      codeSnippet: `import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

# Generate skewed data with outliers
np.random.seed(42)
income = np.concatenate([
    np.random.exponential(50000, 200),
    np.array([500000, 750000, 1000000])  # outliers
])
age = np.concatenate([
    np.random.normal(40, 12, 200),
    np.array([95, 98, 100])  # outliers
])
data = pd.DataFrame({"income": income, "age": age})

print("=== Original Data Statistics ===")
print(data.describe().round(2))

# Apply all three scalers
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Original distributions
axes[0, 0].hist(data["income"], bins=30, color="steelblue", alpha=0.7)
axes[0, 0].set_title("Original (income)")
axes[0, 0].set_xlabel("Value")

for idx, (name, scaler) in enumerate(scalers.items()):
    scaled = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled, columns=data.columns)

    row, col = divmod(idx + 1, 2)
    axes[row, col].hist(scaled_df["income"], bins=30, color="coral", alpha=0.7)
    axes[row, col].set_title(f"{name} (income)")
    axes[row, col].set_xlabel("Scaled value")

    print(f"\\n=== {name} ===")
    print(scaled_df.describe().round(4))

plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Feature Selection",
      slug: "feature-selection",
      description:
        "Filter, wrapper, and embedded methods to identify the most informative features",
      markdownContent: `# Feature Selection

Not all features are useful. Irrelevant or redundant features add noise, increase training time, and can hurt generalisation. **Feature selection** identifies the subset of features that contribute most to prediction.

## Filter Methods

Evaluate each feature independently using a statistical measure, **before** training a model.

**Correlation** — for regression tasks, rank features by their Pearson correlation $|r|$ with the target. Highly correlated features with each other can also be removed to reduce redundancy.

**Mutual Information** — measures how much knowing $X$ reduces uncertainty about $Y$:

$$
I(X; Y) = \\sum_{x,y} p(x,y) \\log \\frac{p(x,y)}{p(x)\\,p(y)}
$$

Unlike correlation, mutual information captures **non-linear** relationships. scikit-learn provides \`mutual_info_classif\` and \`mutual_info_regression\`.

## Wrapper Methods

Train a model repeatedly with different feature subsets and pick the best-performing set.

**Recursive Feature Elimination (RFE)** starts with all features, trains a model, removes the least important feature, and repeats until the desired number remains. Computationally expensive but often gives strong results.

## Embedded Methods

Feature selection happens **during** model training.

**L1 Regularisation (Lasso)** adds a penalty $\\lambda \\sum |\\theta_j|$ to the loss function. This drives some coefficients to exactly zero, effectively removing those features:

$$
J(\\theta) = \\text{Loss}(\\theta) + \\lambda \\sum_{j=1}^{n} |\\theta_j|
$$

**Tree Feature Importance** — after training a tree-based model, feature importance is computed from the total impurity reduction each feature provides across all splits.

## Comparison

| Method | Speed | Captures Non-linearity | Model-Dependent |
|---|---|---|---|
| Filter (MI, correlation) | Fast | MI: Yes, Corr: No | No |
| Wrapper (RFE) | Slow | Yes (via model) | Yes |
| Embedded (L1, tree importance) | Medium | Yes (via model) | Yes |

## Best Practices

- Start with **filter methods** for a quick baseline, then refine with wrapper or embedded methods.
- Use **mutual information** over correlation when relationships may be non-linear.
- Combine approaches: use filter to discard obviously useless features, then RFE for fine-tuning.

Run the code to see SelectKBest and RFE in action on a classification dataset.`,
      codeSnippet: `import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate dataset with informative and noisy features
X, y = make_classification(
    n_samples=500, n_features=12, n_informative=5,
    n_redundant=3, n_repeated=0, random_state=42
)
feature_names = [f"f{i}" for i in range(X.shape[1])]

# --- Filter Method: SelectKBest with Mutual Information ---
selector = SelectKBest(score_func=mutual_info_classif, k=5)
selector.fit(X, y)
mi_scores = selector.scores_

print("=== Filter Method: Mutual Information Scores ===")
mi_ranking = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
for feat, score in mi_ranking.items():
    marker = " <-- selected" if feat in np.array(feature_names)[selector.get_support()] else ""
    print(f"  {feat}: {score:.4f}{marker}")

# --- Wrapper Method: Recursive Feature Elimination ---
estimator = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator, n_features_to_select=5, step=1)
rfe.fit(X, y)

print("\\n=== Wrapper Method: RFE Rankings ===")
rfe_ranking = pd.Series(rfe.ranking_, index=feature_names).sort_values()
for feat, rank in rfe_ranking.items():
    marker = " <-- selected" if rank == 1 else ""
    print(f"  {feat}: rank {rank}{marker}")

# --- Embedded Method: Tree Feature Importance ---
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X, y)
importances = forest.feature_importances_

print("\\n=== Embedded Method: Random Forest Importance ===")
imp_ranking = pd.Series(importances, index=feature_names).sort_values(ascending=False)
for feat, imp in imp_ranking.items():
    print(f"  {feat}: {imp:.4f}")

# Visualise all three methods side by side
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].barh(feature_names, mi_scores, color="steelblue")
axes[0].set_title("Mutual Information")
axes[0].set_xlabel("Score")

axes[1].barh(feature_names, rfe.ranking_, color="coral")
axes[1].set_title("RFE Ranking (lower = better)")
axes[1].set_xlabel("Rank")

axes[2].barh(feature_names, importances, color="seagreen")
axes[2].set_title("Random Forest Importance")
axes[2].set_xlabel("Importance")

plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Handling Imbalanced Data",
      slug: "handling-imbalanced-data",
      description:
        "Why accuracy fails on skewed classes and how to fix it with SMOTE, class weights, and threshold tuning",
      markdownContent: `# Handling Imbalanced Data

In many real-world problems — fraud detection, disease diagnosis, churn prediction — the class of interest is **rare**. A dataset with 99% negatives and 1% positives is **imbalanced**, and standard approaches can fail silently.

## Why Accuracy Is Misleading

A model that always predicts the majority class achieves 99% accuracy on a 99/1 split — yet it catches zero fraud cases. Accuracy treats all errors equally, but in imbalanced settings, **false negatives are far more costly** than false positives.

Better metrics for imbalanced data:

- **Precision** — of all predicted positives, how many are correct? $\\frac{TP}{TP + FP}$
- **Recall** — of all actual positives, how many did we find? $\\frac{TP}{TP + FN}$
- **F1 Score** — harmonic mean of precision and recall: $F_1 = 2 \\cdot \\frac{P \\cdot R}{P + R}$
- **AUROC** — area under the ROC curve, measures ranking quality across all thresholds.

## SMOTE (Synthetic Minority Over-sampling)

**SMOTE** generates synthetic minority samples by interpolating between existing minority examples:

1. Pick a minority sample $x_i$.
2. Find its $k$ nearest minority neighbours.
3. Create a new sample along the line between $x_i$ and a randomly chosen neighbour:

$$
x_{\\text{new}} = x_i + \\lambda \\cdot (x_{\\text{neighbour}} - x_i), \\quad \\lambda \\sim U(0, 1)
$$

SMOTE should only be applied to the **training set** — never to validation or test data.

## Class Weights

Most classifiers accept a \`class_weight\` parameter. Setting \`class_weight="balanced"\` adjusts weights inversely proportional to class frequency:

$$
w_c = \\frac{n}{k \\cdot n_c}
$$

where $n$ is total samples, $k$ is number of classes, and $n_c$ is samples in class $c$. This makes the loss function penalise minority misclassifications more heavily.

## Threshold Adjustment

By default, classifiers use 0.5 as the decision threshold. Lowering the threshold increases recall at the cost of precision. Use the **precision-recall curve** to find the optimal trade-off for your use case.

## Best Practices

- Always evaluate with **precision, recall, F1** — not just accuracy.
- Apply SMOTE only to training data (inside a cross-validation loop).
- Combine approaches: class weights + threshold tuning is often effective.
- Consider **stratified splits** to maintain class proportions in train/test sets.

Run the code to compare a baseline classifier with SMOTE-augmented training on imbalanced data.`,
      codeSnippet: `import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Create imbalanced dataset (95% class 0, 5% class 1)
X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=5,
    weights=[0.95, 0.05], flip_y=0, random_state=42
)

print(f"=== Class Distribution ===")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# --- Baseline: no handling ---
baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

print("\\n=== Baseline (no imbalance handling) ===")
print(classification_report(y_test, y_pred_base, digits=4))

# --- With class_weight='balanced' ---
weighted = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
weighted.fit(X_train, y_train)
y_pred_weighted = weighted.predict(X_test)

print("=== With class_weight='balanced' ===")
print(classification_report(y_test, y_pred_weighted, digits=4))

# --- With SMOTE ---
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"=== SMOTE Resampled Training Set ===")
unique_sm, counts_sm = np.unique(y_train_sm, return_counts=True)
for cls, cnt in zip(unique_sm, counts_sm):
    print(f"  Class {cls}: {cnt}")

smote_model = LogisticRegression(max_iter=1000, random_state=42)
smote_model.fit(X_train_sm, y_train_sm)
y_pred_smote = smote_model.predict(X_test)

print("\\n=== With SMOTE ===")
print(classification_report(y_test, y_pred_smote, digits=4))

# Compare F1 scores for minority class
f1_scores = {
    "Baseline": f1_score(y_test, y_pred_base),
    "Class Weight": f1_score(y_test, y_pred_weighted),
    "SMOTE": f1_score(y_test, y_pred_smote),
}

plt.figure(figsize=(8, 5))
plt.bar(f1_scores.keys(), f1_scores.values(), color=["steelblue", "coral", "seagreen"])
plt.ylabel("F1 Score (minority class)")
plt.title("Imbalanced Data: F1 Score Comparison")
plt.ylim(0, 1)
for i, (name, score) in enumerate(f1_scores.items()):
    plt.text(i, score + 0.02, f"{score:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Datetime Feature Extraction",
      slug: "datetime-feature-extraction",
      description:
        "Extract temporal features from datetime columns and apply cyclical encoding for periodic patterns",
      markdownContent: `# Datetime Feature Extraction

Raw datetime columns (e.g., \`2024-03-15 14:30:00\`) are not directly usable by ML models. We need to **extract meaningful features** that capture temporal patterns.

## Basic Extractions

From a single datetime column you can derive many features:

| Feature | Example | Captures |
|---|---|---|
| \`year\` | 2024 | Long-term trends |
| \`month\` | 3 | Seasonality |
| \`day_of_week\` | 4 (Friday) | Weekly patterns |
| \`hour\` | 14 | Intra-day patterns |
| \`is_weekend\` | 0 or 1 | Work vs leisure |
| \`quarter\` | 1 | Quarterly cycles |

These simple extractions are powerful — day-of-week alone can capture that taxi demand spikes on Friday nights, or that retail sales peak on weekends.

## The Problem with Raw Integers

If we encode month as $1, 2, \\ldots, 12$, the model sees December (12) and January (1) as **maximally distant**. But in reality they are adjacent! The same applies to hours: 23:00 and 00:00 are one hour apart, not 23.

## Cyclical Encoding

We project periodic features onto a **circle** using sine and cosine:

$$
x_{\\sin} = \\sin\\!\\left(\\frac{2\\pi \\cdot v}{T}\\right), \\quad x_{\\cos} = \\cos\\!\\left(\\frac{2\\pi \\cdot v}{T}\\right)
$$

where $v$ is the raw value and $T$ is the period (12 for months, 24 for hours, 7 for weekdays).

This encoding maps the feature to a point on the unit circle. Now December and January are close together (as they should be), and the model can learn smooth periodic patterns.

## Why Two Components?

Using only $\\sin$ would make some distinct time points indistinguishable (e.g., month 3 and month 9 both have $\\sin = 0$). The $\\cos$ component resolves this ambiguity — together, $(\\sin, \\cos)$ uniquely identify every point on the cycle.

## Best Practices

- Use **cyclical encoding** for features with known periods (hour, weekday, month).
- Use **raw integers** or one-hot for features without cyclical nature (year, is_weekend).
- Combine basic and cyclical features — they capture different aspects of the signal.
- Consider **lag features** (value from $t-1$, $t-7$, etc.) for time-series problems.

Run the code to extract datetime features and visualise cyclical encoding.`,
      codeSnippet: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create sample datetime data
dates = pd.date_range("2024-01-01", periods=365 * 2, freq="h")
np.random.seed(42)
df = pd.DataFrame({
    "timestamp": dates[:500],
    "value": np.random.randn(500),
})

print("=== Sample Timestamps ===")
print(df.head(10).to_string())

# Basic feature extraction
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["hour"] = df["timestamp"].dt.hour
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["quarter"] = df["timestamp"].dt.quarter

print("\\n=== Extracted Features ===")
print(df[["timestamp", "year", "month", "day_of_week", "hour", "is_weekend", "quarter"]].head(10).to_string())

# Cyclical encoding
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

print("\\n=== Cyclical Encoding (first 10 rows) ===")
print(df[["hour", "hour_sin", "hour_cos", "month", "month_sin", "month_cos"]].head(10).to_string())

# Visualise cyclical encoding
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Hour cycle
hours = df.drop_duplicates("hour").sort_values("hour")
axes[0].scatter(hours["hour_sin"], hours["hour_cos"], c=hours["hour"],
               cmap="twilight", s=80, edgecolors="k")
axes[0].set_title("Hour (cyclical)")
axes[0].set_xlabel("sin(hour)")
axes[0].set_ylabel("cos(hour)")
axes[0].set_aspect("equal")

# Month cycle
months = df.drop_duplicates("month").sort_values("month")
axes[1].scatter(months["month_sin"], months["month_cos"], c=months["month"],
               cmap="hsv", s=80, edgecolors="k")
for _, row in months.iterrows():
    axes[1].annotate(f'{int(row["month"])}', (row["month_sin"], row["month_cos"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
axes[1].set_title("Month (cyclical)")
axes[1].set_xlabel("sin(month)")
axes[1].set_ylabel("cos(month)")
axes[1].set_aspect("equal")

# Day of week cycle
dows = df.drop_duplicates("day_of_week").sort_values("day_of_week")
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
axes[2].scatter(dows["dow_sin"], dows["dow_cos"], c=dows["day_of_week"],
               cmap="tab10", s=80, edgecolors="k")
for _, row in dows.iterrows():
    axes[2].annotate(day_names[int(row["day_of_week"])],
                    (row["dow_sin"], row["dow_cos"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
axes[2].set_title("Day of Week (cyclical)")
axes[2].set_xlabel("sin(dow)")
axes[2].set_ylabel("cos(dow)")
axes[2].set_aspect("equal")

plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
  ],
};
