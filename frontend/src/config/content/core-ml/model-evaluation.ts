import type { Chapter } from "../types";

export const modelEvaluation: Chapter = {
  title: "Model Evaluation",
  slug: "model-evaluation",
  pages: [
    {
      title: "Metrics & Scoring",
      slug: "metrics-scoring",
      description:
        "Accuracy, precision, recall, F1-score, confusion matrix, and ROC curve",
      markdownContent: `# Metrics & Scoring

Choosing the right metric determines whether your model is actually solving the problem. **Accuracy** — the fraction of correct predictions — is a starting point, but it can be misleading when classes are imbalanced.

## Precision, Recall & F1-Score

For binary classification, predictions fall into four categories: true positives ($TP$), true negatives ($TN$), false positives ($FP$), and false negatives ($FN$).

**Precision** measures how many predicted positives are truly positive:

$$
P = \\frac{TP}{TP + FP}
$$

**Recall** (sensitivity) measures how many actual positives are captured:

$$
R = \\frac{TP}{TP + FN}
$$

These two metrics trade off against each other. The **F1-score** provides their harmonic mean, balancing both concerns into a single number:

$$
F_1 = 2 \\cdot \\frac{P \\cdot R}{P + R}
$$

When $P = R$, the F1-score equals both. When they diverge, F1 is pulled toward the lower value — it penalizes models that sacrifice one metric for the other.

## Confusion Matrix

A **confusion matrix** is a table that visualizes all four outcome categories at once. Each row represents the actual class and each column represents the predicted class. It reveals patterns that a single number cannot — for example, which classes are commonly confused with each other.

## ROC Curve & AUC

The **ROC curve** plots the true positive rate against the false positive rate at every classification threshold. A model with no skill traces the diagonal (AUC = 0.5), while a perfect model hugs the top-left corner (AUC = 1.0). The **area under the curve** (AUC) summarizes threshold-independent performance in a single scalar.

Run the code below to train a classifier, inspect the confusion matrix, and plot the ROC curve.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)

# Synthetic binary classification dataset
X, y = make_classification(n_samples=800, n_features=10, n_informative=5,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)

model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Classification report
print(classification_report(y_test, y_pred, digits=3))

# Confusion matrix & ROC curve side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax1)
ax1.set_title("Confusion Matrix")

fpr, tpr, _ = roc_curve(y_test, y_prob)
ax2.plot(fpr, tpr, color="steelblue", lw=2,
         label=f"ROC curve (AUC = {auc(fpr, tpr):.3f})")
ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
ax2.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
        title="ROC Curve")
ax2.legend(loc="lower right")
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Cross-Validation",
      slug: "cross-validation",
      description:
        "K-fold cross-validation, bias-variance tradeoff, and stratified splitting",
      markdownContent: `# Cross-Validation

A single train/test split gives one estimate of model performance — and that estimate depends heavily on which samples ended up in each set. **K-fold cross-validation** provides a more robust picture by using every data point for both training and validation.

## K-Fold Procedure

The dataset is partitioned into $K$ equally sized folds. The model trains $K$ times, each time holding out a different fold for validation and training on the remaining $K - 1$ folds. The final score is the mean across all folds:

$$
\\bar{S} = \\frac{1}{K} \\sum_{i=1}^{K} S_i
$$

where $S_i$ is the validation score on fold $i$. Common choices are $K = 5$ or $K = 10$.

## Bias-Variance Tradeoff in K Selection

The choice of $K$ creates a tradeoff. Small $K$ (e.g., 2) uses less training data per fold, introducing **pessimistic bias** — the model underperforms because it sees fewer samples. Large $K$ (e.g., $n$, leave-one-out) reduces bias but increases **variance** because the $K$ training sets overlap heavily, making fold scores highly correlated.

In practice, $K = 5$ or $K = 10$ strikes a good balance. The standard deviation across folds gives a rough confidence interval: $\\bar{S} \\pm \\sigma_S$.

## Stratified K-Fold

When classes are imbalanced — say 95% negative and 5% positive — a random split might leave some folds with no positive samples at all. **Stratified K-fold** preserves the class distribution in every fold, ensuring each fold is a representative mini-version of the full dataset. This is the default behavior in scikit-learn's \`cross_val_score\` for classifiers.

Run the code below to compare different values of $K$ and visualize how the validation score stabilizes.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Imbalanced binary dataset (30% positive class)
X, y = make_classification(n_samples=600, n_features=10, n_informative=5,
                           weights=[0.7, 0.3], random_state=42)

model = LogisticRegression(max_iter=200, random_state=42)

# Evaluate across different K values
k_values = [2, 3, 5, 7, 10, 15, 20]
means, stds = [], []

for k in k_values:
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    means.append(scores.mean())
    stds.append(scores.std())
    print(f"K={k:>2d}:  F1 = {scores.mean():.4f} ± {scores.std():.4f}")

# Plot K vs validation score
means, stds = np.array(means), np.array(stds)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_values, means, "o-", color="steelblue", lw=2, label="Mean F1")
ax.fill_between(k_values, means - stds, means + stds,
                alpha=0.2, color="steelblue", label="± 1 std dev")
ax.set(xlabel="Number of Folds (K)", ylabel="F1 Score",
       title="K-Fold Cross-Validation: K vs F1 Score")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Bias-Variance Tradeoff",
      slug: "bias-variance-tradeoff",
      description:
        "Understanding underfitting, overfitting, and the tradeoff between model bias and variance",
      markdownContent: `# Bias-Variance Tradeoff

Every predictive model's total error can be decomposed into three components: **bias**, **variance**, and **irreducible noise**. Understanding this decomposition is essential for diagnosing model performance and choosing the right level of complexity.

## What Is Bias?

**Bias** is the error introduced by approximating a complex real-world problem with a simplified model. A model with high bias makes strong assumptions about the data — for example, fitting a straight line to a curved relationship.

High bias leads to **underfitting**: the model is too simple to capture the underlying pattern, so it performs poorly on both training and test data.

## What Is Variance?

**Variance** is the error introduced by the model's sensitivity to fluctuations in the training data. A model with high variance fits the training data very closely — including its noise — so small changes in the training set produce very different models.

High variance leads to **overfitting**: the model captures noise as if it were signal, performing well on training data but poorly on unseen data.

## The Tradeoff

The expected prediction error at a point $x$ decomposes as:

$$
\\text{Error}(x) = \\text{Bias}^2(x) + \\text{Variance}(x) + \\sigma^2_{\\text{noise}}
$$

where $\\sigma^2_{\\text{noise}}$ is the **irreducible error** — the inherent noise in the data that no model can eliminate.

As model complexity increases:
- **Bias decreases** — the model can represent more complex relationships
- **Variance increases** — the model becomes more sensitive to the particular training data

This creates a characteristic **U-shaped curve** of test error vs model complexity. The optimal model sits at the bottom of this curve, balancing bias and variance.

## Diagnosing with Learning Curves

**Learning curves** plot training error and validation error as a function of training set size:
- **High bias**: both training and validation errors are high and converge — more data won't help, you need a more complex model
- **High variance**: training error is low but validation error is high with a gap — more data or regularisation can help

## Practical Implications

| Scenario | Bias | Variance | Fix |
|----------|------|----------|-----|
| Linear model on non-linear data | High | Low | Use a more flexible model |
| Deep neural network on small data | Low | High | Regularise, get more data, or simplify |
| Well-tuned ensemble | Moderate | Moderate | The sweet spot |

Run the code below to see underfitting, good fit, and overfitting in action with polynomial regression of different degrees.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve

# Generate noisy data from a known curve
np.random.seed(42)
n = 60
X = np.sort(np.random.uniform(0, 1, n))
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.3, n)
X = X.reshape(-1, 1)

# Three polynomial degrees: underfitting, good fit, overfitting
degrees = [1, 4, 15]
labels = ["Degree 1 (Underfit)", "Degree 4 (Good Fit)", "Degree 15 (Overfit)"]
colors = ["#e74c3c", "#27ae60", "#8e44ad"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

X_plot = np.linspace(0, 1, 200).reshape(-1, 1)

for i, (deg, label, color) in enumerate(zip(degrees, labels, colors)):
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X, y)
    y_plot = model.predict(X_plot)

    # Top row: fitted curves
    ax = axes[0, i]
    ax.scatter(X, y, s=15, alpha=0.6, color="gray", label="Data")
    ax.plot(X_plot, y_plot, color=color, lw=2, label=label)
    ax.plot(X_plot, np.sin(2 * np.pi * X_plot), "k--", lw=1, alpha=0.4,
            label="True function")
    ax.set(title=label, xlabel="x", ylabel="y", ylim=(-2, 2))
    ax.legend(fontsize=7)

    # Bottom row: learning curves
    ax2 = axes[1, i]
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 8), random_state=42
    )
    train_mse = -train_scores.mean(axis=1)
    val_mse = -val_scores.mean(axis=1)
    ax2.plot(train_sizes, train_mse, "o-", color=color, label="Training error")
    ax2.plot(train_sizes, val_mse, "s--", color=color, alpha=0.7,
             label="Validation error")
    ax2.set(title=f"Learning Curve ({label.split(' ')[0]} {label.split(' ')[1]})",
            xlabel="Training set size", ylabel="MSE")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

plt.suptitle("Bias-Variance Tradeoff: Polynomial Regression", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Regularisation (L1/L2)",
      slug: "regularisation",
      description:
        "Ridge, Lasso, and Elastic Net regularisation to prevent overfitting",
      markdownContent: `# Regularisation (L1/L2)

When a model has many features or high polynomial degree, it can fit training data perfectly by assigning large coefficients — but this leads to overfitting. **Regularisation** prevents this by adding a penalty term to the loss function that discourages large weights.

## Why Regularise?

Without regularisation, the model minimises only the training loss:

$$
\\hat{w} = \\arg\\min_w \\sum_{i=1}^{n} \\mathcal{L}(y_i, f(x_i; w))
$$

With regularisation, we add a penalty $\\Omega(w)$ controlled by a hyperparameter $\\lambda$:

$$
\\hat{w} = \\arg\\min_w \\sum_{i=1}^{n} \\mathcal{L}(y_i, f(x_i; w)) + \\lambda \\, \\Omega(w)
$$

Larger $\\lambda$ means stronger regularisation (simpler model, higher bias, lower variance).

## L2 Regularisation (Ridge)

**Ridge regression** uses the squared L2 norm as the penalty:

$$
\\Omega_{L2}(w) = \\sum_{j=1}^{p} w_j^2
$$

Ridge shrinks all coefficients **toward zero** but never exactly to zero. It's effective when many features each contribute a small amount to the prediction.

## L1 Regularisation (Lasso)

**Lasso regression** uses the L1 norm:

$$
\\Omega_{L1}(w) = \\sum_{j=1}^{p} |w_j|
$$

The key property of L1 is **sparsity**: it drives some coefficients to **exactly zero**, performing automatic feature selection. This is valuable when you suspect only a few features are truly relevant.

## Elastic Net

**Elastic Net** combines both penalties:

$$
\\Omega_{EN}(w) = \\alpha \\sum |w_j| + (1 - \\alpha) \\sum w_j^2
$$

where $\\alpha \\in [0, 1]$ controls the mix. Elastic Net inherits L1's sparsity while gaining L2's stability when features are correlated.

## Geometric Interpretation

The penalties define constraint regions in weight space:
- **L2 (Ridge)**: a circular (spherical) constraint — the optimum touches the circle smoothly, shrinking all weights
- **L1 (Lasso)**: a diamond-shaped constraint — the corners lie on the axes, so the optimum is more likely to land at a corner where some weights are exactly zero

## Choosing $\\lambda$

The regularisation strength $\\lambda$ is a hyperparameter chosen by **cross-validation**. A **regularisation path** plots coefficients as $\\lambda$ varies, revealing which features are most important (they persist at higher $\\lambda$ values).

Run the code below to compare Ridge, Lasso, and Elastic Net, and see how coefficients change as regularisation increases.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Create dataset with many features, only a few informative
X, y, true_coefs = make_regression(n_samples=200, n_features=20,
                                    n_informative=5, noise=10,
                                    coef=True, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Range of regularisation strengths
alphas = np.logspace(-2, 3, 50)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
models = [
    ("Ridge (L2)", Ridge, "#2980b9"),
    ("Lasso (L1)", Lasso, "#e74c3c"),
    ("Elastic Net", ElasticNet, "#27ae60"),
]

for ax, (name, Model, color) in zip(axes, models):
    coef_paths = []
    for a in alphas:
        kwargs = {"alpha": a, "max_iter": 10000}
        if Model == ElasticNet:
            kwargs["l1_ratio"] = 0.5
        m = Model(**kwargs)
        m.fit(X, y)
        coef_paths.append(m.coef_)
    coef_paths = np.array(coef_paths)

    for j in range(coef_paths.shape[1]):
        ax.plot(alphas, coef_paths[:, j], lw=1, alpha=0.7)
    ax.set(xscale="log", xlabel=r"$\\lambda$ (alpha)", ylabel="Coefficient value",
           title=f"{name} — Coefficient Paths")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare models at a single alpha with cross-validation
print("\\n5-Fold CV (R² score) at alpha=1.0:")
for name, Model, _ in models:
    kwargs = {"alpha": 1.0, "max_iter": 10000}
    if Model == ElasticNet:
        kwargs["l1_ratio"] = 0.5
    m = Model(**kwargs)
    scores = cross_val_score(m, X, y, cv=5, scoring="r2")
    print(f"  {name:20s}: R² = {scores.mean():.4f} ± {scores.std():.4f}")

# Show which features Lasso zeros out
lasso = Lasso(alpha=1.0, max_iter=10000)
lasso.fit(X, y)
n_zero = np.sum(lasso.coef_ == 0)
print(f"\\nLasso (alpha=1.0) zeroed out {n_zero} of {len(lasso.coef_)} features")
print(f"Non-zero features: {np.where(lasso.coef_ != 0)[0].tolist()}")`,
      codeLanguage: "python",
    },
    {
      title: "Hyperparameter Tuning",
      slug: "hyperparameter-tuning",
      description:
        "Grid search, random search, and Bayesian optimisation for finding optimal hyperparameters",
      markdownContent: `# Hyperparameter Tuning

**Hyperparameters** are settings chosen *before* training begins — they control the learning process itself rather than being learned from data.

| | Parameters | Hyperparameters |
|---|---|---|
| **Set by** | Learning algorithm | Practitioner |
| **Examples** | Weights, biases, split points | Learning rate, tree depth, $\\lambda$ |
| **Optimised via** | Gradient descent, etc. | Search + cross-validation |

Choosing good hyperparameters can make the difference between a mediocre and a state-of-the-art model.

## Grid Search

**Grid search** evaluates every combination in a predefined parameter grid. For example, with 5 values of \`max_depth\` and 4 values of \`n_estimators\`, grid search trains $5 \\times 4 = 20$ models.

**Pros**: exhaustive — guaranteed to find the best combination in the grid.
**Cons**: scales exponentially with the number of hyperparameters. With $d$ hyperparameters each taking $k$ values, the cost is $O(k^d)$.

## Random Search

**Random search** samples hyperparameter combinations randomly from specified distributions. Bergstra & Bengio (2012) showed that random search is often more efficient than grid search because:

1. Not all hyperparameters are equally important
2. Grid search wastes evaluations by exhaustively varying unimportant parameters
3. Random search explores more distinct values of each important parameter for the same budget

## Bayesian Optimisation

**Bayesian optimisation** treats hyperparameter tuning as a black-box optimisation problem:

1. Build a **surrogate model** (e.g., Gaussian Process) of the objective function from evaluated points
2. Use an **acquisition function** (e.g., Expected Improvement) to decide which point to evaluate next — balancing exploration vs exploitation
3. Evaluate the objective, update the surrogate, and repeat

This is the most **sample-efficient** method — it typically finds good hyperparameters with far fewer evaluations than grid or random search.

## Nested Cross-Validation

When tuning hyperparameters, there's a risk of **overfitting the validation set** — the chosen hyperparameters might be tuned to the specific validation fold rather than generalising well.

**Nested CV** addresses this with two loops:
- **Outer loop**: $K_1$ folds for estimating generalisation performance
- **Inner loop**: $K_2$ folds within each outer training set for hyperparameter selection

This gives an unbiased estimate of how well the *entire tuning procedure* generalises.

## Practical Tips

1. **Start broad**: use random search with wide ranges to identify promising regions
2. **Then refine**: zoom into the best region with a finer grid or Bayesian optimisation
3. **Use log scales** for parameters that span orders of magnitude (learning rate, regularisation strength)
4. **Set a budget**: decide how many evaluations you can afford before starting

Run the code below to compare Grid Search and Random Search on a Random Forest, and visualise the search results.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                      StratifiedKFold)
from scipy.stats import randint

# Synthetic classification dataset
X, y = make_classification(n_samples=500, n_features=15, n_informative=8,
                           n_redundant=2, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_model = RandomForestClassifier(random_state=42)

# --- Grid Search ---
param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring="f1",
                           n_jobs=-1, return_train_score=True)
grid_search.fit(X, y)
n_grid = len(grid_search.cv_results_["mean_test_score"])

# --- Random Search (same budget as grid) ---
param_dist = {
    "n_estimators": randint(50, 250),
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_split": randint(2, 20),
}

random_search = RandomizedSearchCV(base_model, param_dist, n_iter=n_grid, cv=cv,
                                    scoring="f1", n_jobs=-1, random_state=42,
                                    return_train_score=True)
random_search.fit(X, y)

print(f"Grid Search:   best F1 = {grid_search.best_score_:.4f}  "
      f"({n_grid} evaluations)")
print(f"  Best params: {grid_search.best_params_}")
print(f"\\nRandom Search: best F1 = {random_search.best_score_:.4f}  "
      f"({n_grid} evaluations)")
print(f"  Best params: {random_search.best_params_}")

# --- Heatmap: Grid Search scores for max_depth vs n_estimators ---
results = grid_search.cv_results_
depths = sorted([d for d in param_grid["max_depth"] if d is not None])
n_ests = param_grid["n_estimators"]

# Average over min_samples_split for the heatmap
heatmap = np.zeros((len(depths), len(n_ests)))
for idx in range(len(results["mean_test_score"])):
    d = results["params"][idx]["max_depth"]
    n = results["params"][idx]["n_estimators"]
    if d is not None:
        i = depths.index(d)
        j = n_ests.index(n)
        heatmap[i, j] += results["mean_test_score"][idx]

# Average over min_samples_split values
heatmap /= len(param_grid["min_samples_split"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

im = ax1.imshow(heatmap, cmap="YlOrRd", aspect="auto")
ax1.set(xticks=range(len(n_ests)), xticklabels=n_ests,
        yticks=range(len(depths)), yticklabels=depths,
        xlabel="n_estimators", ylabel="max_depth",
        title="Grid Search: Mean F1 Score")
for i in range(len(depths)):
    for j in range(len(n_ests)):
        ax1.text(j, i, f"{heatmap[i, j]:.3f}", ha="center", va="center",
                 fontsize=8, color="black")
plt.colorbar(im, ax=ax1, label="F1 Score")

# Comparison bar chart
grid_scores = grid_search.cv_results_["mean_test_score"]
rand_scores = random_search.cv_results_["mean_test_score"]
ax2.hist(grid_scores, bins=15, alpha=0.6, color="#2980b9", label="Grid Search")
ax2.hist(rand_scores, bins=15, alpha=0.6, color="#e74c3c", label="Random Search")
ax2.axvline(grid_search.best_score_, color="#2980b9", ls="--", lw=2)
ax2.axvline(random_search.best_score_, color="#e74c3c", ls="--", lw=2)
ax2.set(xlabel="F1 Score", ylabel="Count",
        title="Distribution of Evaluated Scores")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "PR-AUC, Calibration & Lift Charts",
      slug: "pr-auc-calibration-lift",
      description:
        "Precision-recall AUC, calibration curves, and lift/gain charts for imbalanced datasets",
      markdownContent: `# PR-AUC, Calibration & Lift Charts

Standard ROC-AUC can paint an overly optimistic picture when classes are highly imbalanced. This page covers three complementary evaluation tools that are essential for credit scoring, fraud detection, and similar domains where the positive class is rare.

---

## 1. Precision-Recall AUC (PR-AUC)

### Why ROC-AUC Can Be Misleading

ROC-AUC measures the tradeoff between the **true positive rate** and the **false positive rate**. When negatives vastly outnumber positives (e.g., 95% vs 5%), a model can achieve a low FPR simply because the denominator ($FP + TN$) is enormous. A large number of false positives can still look like a tiny FPR, inflating the ROC-AUC.

**Precision-recall curves** focus exclusively on the positive class, making them far more informative for imbalanced problems.

### The PR Curve

At each classification threshold $t$, we compute:

$$
\\text{Precision}(t) = \\frac{TP(t)}{TP(t) + FP(t)}, \\qquad \\text{Recall}(t) = \\frac{TP(t)}{TP(t) + FN(t)}
$$

Plotting precision (y-axis) against recall (x-axis) traces the **PR curve**. A perfect classifier hugs the top-right corner (precision = recall = 1). A no-skill classifier sits at $\\text{Precision} = \\pi$, where $\\pi$ is the prevalence of the positive class.

### Average Precision (AP)

The area under the PR curve is summarised by **Average Precision**:

$$
\\text{AP} = \\sum_{k} (R_k - R_{k-1}) \\cdot P_k
$$

This is equivalent to the weighted mean of precisions at each threshold, weighted by the increase in recall. Unlike ROC-AUC, AP is sensitive to improvements in the rare positive class.

---

## 2. Calibration Curves

### What Is Calibration?

A model is **well-calibrated** if its predicted probabilities match observed frequencies. When the model says "30% chance of fraud", roughly 30% of those cases should actually be fraud.

### Reliability Diagrams

A **reliability diagram** (calibration curve) bins predictions by predicted probability, then plots the **mean predicted probability** (x-axis) against the **observed fraction of positives** (y-axis) in each bin. A perfectly calibrated model follows the diagonal $y = x$.

### Why Calibration Matters

In risk scoring (credit default, fraud, insurance), decisions are based on **predicted probabilities**, not just rankings. An uncalibrated model that outputs 0.8 when the true risk is 0.3 leads to overly aggressive risk controls.

### Post-hoc Calibration Methods

- **Platt Scaling**: fits a logistic regression on the model's raw scores — learns a sigmoid mapping $P(y=1 \\mid s) = \\frac{1}{1 + e^{-(as + b)}}$
- **Isotonic Regression**: fits a non-parametric, monotonically increasing function — more flexible but needs more data

### Brier Score

The **Brier score** measures calibration and refinement together:

$$
\\text{BS} = \\frac{1}{n} \\sum_{i=1}^{n} (\\hat{p}_i - y_i)^2
$$

Lower is better. A perfectly calibrated model that also separates classes well achieves the minimum Brier score.

---

## 3. Lift and Gain Charts

### Cumulative Gains Curve

Sort all observations by predicted probability (descending). The **cumulative gains curve** plots the percentage of the population examined (x-axis) against the percentage of all positives captured (y-axis).

A useful model rises steeply above the diagonal baseline. For example, "examining the top 20% of scored customers captures 60% of all fraudsters."

### Lift Curve

**Lift** at a given percentile is the ratio of the model's gain to the baseline (random) gain:

$$
\\text{Lift}(p) = \\frac{\\text{\\% positives captured at } p}{p}
$$

A lift of 3.0 at the 10th percentile means the model is three times better than random selection at that cutoff. Lift always starts high (if the model is useful) and converges to 1.0 as the entire population is included.

### Decile Analysis

In banking and marketing, analysts often group customers into **deciles** by predicted score and report the concentration of positives in each decile. This provides an actionable summary: "Decile 1 contains 45% of all defaults."

---

Run the code below to train two models on a synthetic imbalanced dataset, then compare their PR curves, calibration plots, and lift/gain charts side by side.`,
      codeSnippet: `import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             brier_score_loss, roc_auc_score)
import os, base64, io

mode = os.environ.get("ML_CATALOGUE_MODE", "quick")
n_samples = 2000 if mode == "quick" else 10000

# --- Synthetic imbalanced dataset (95% negative, 5% positive) ---
X, y = make_classification(n_samples=n_samples, n_features=15,
                           n_informative=8, n_redundant=2,
                           weights=[0.95, 0.05], flip_y=0.01,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Model A: Logistic Regression (well-calibrated)
model_a = LogisticRegression(max_iter=300, random_state=42)
model_a.fit(X_train, y_train)
prob_a = model_a.predict_proba(X_test)[:, 1]

# Model B: Decision Tree (poorly calibrated)
model_b = DecisionTreeClassifier(max_depth=5, random_state=42)
model_b.fit(X_train, y_train)
prob_b = model_b.predict_proba(X_test)[:, 1]

# --- Helper: cumulative gains and lift ---
def cumulative_gains_lift(y_true, y_prob):
    order = np.argsort(-y_prob)
    y_sorted = np.array(y_true)[order]
    n = len(y_sorted)
    total_pos = y_sorted.sum()
    cum_pos = np.cumsum(y_sorted)
    pct_population = np.arange(1, n + 1) / n
    pct_captured = cum_pos / total_pos
    lift = pct_captured / pct_population
    return pct_population, pct_captured, lift

# --- Compute metrics ---
prec_a, rec_a, _ = precision_recall_curve(y_test, prob_a)
prec_b, rec_b, _ = precision_recall_curve(y_test, prob_b)
ap_a = average_precision_score(y_test, prob_a)
ap_b = average_precision_score(y_test, prob_b)

frac_pos_a, mean_pred_a = calibration_curve(y_test, prob_a, n_bins=10,
                                             strategy="uniform")
frac_pos_b, mean_pred_b = calibration_curve(y_test, prob_b, n_bins=10,
                                             strategy="uniform")
brier_a = brier_score_loss(y_test, prob_a)
brier_b = brier_score_loss(y_test, prob_b)

pop_a, gain_a, lift_a = cumulative_gains_lift(y_test, prob_a)
pop_b, gain_b, lift_b = cumulative_gains_lift(y_test, prob_b)

roc_a = roc_auc_score(y_test, prob_a)
roc_b = roc_auc_score(y_test, prob_b)

# --- Print summary ---
print("Model Comparison on Imbalanced Data (95/5 split)")
print("=" * 52)
print(f"{'Metric':<22} {'Logistic Reg':>14} {'Decision Tree':>14}")
print("-" * 52)
print(f"{'ROC-AUC':<22} {roc_a:>14.4f} {roc_b:>14.4f}")
print(f"{'PR-AUC (Avg Prec)':<22} {ap_a:>14.4f} {ap_b:>14.4f}")
print(f"{'Brier Score':<22} {brier_a:>14.4f} {brier_b:>14.4f}")
prevalence = y_test.mean()
print(f"\\nPositive prevalence: {prevalence:.2%}")

# Lift at top 10% and 20%
n_test = len(y_test)
for pct in [0.10, 0.20]:
    idx = int(pct * n_test)
    print(f"\\nTop {pct:.0%} of population:")
    print(f"  Logistic Reg captures {gain_a[idx]:.1%} of positives "
          f"(lift = {lift_a[idx]:.1f}x)")
    print(f"  Decision Tree captures {gain_b[idx]:.1%} of positives "
          f"(lift = {lift_b[idx]:.1f}x)")

# --- Plot 2x2 chart grid ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. PR Curve
ax = axes[0, 0]
ax.plot(rec_a, prec_a, color="steelblue", lw=2,
        label=f"Logistic Reg (AP={ap_a:.3f})")
ax.plot(rec_b, prec_b, color="#e74c3c", lw=2,
        label=f"Decision Tree (AP={ap_b:.3f})")
ax.axhline(prevalence, color="gray", ls="--", lw=1, label=f"No skill ({prevalence:.3f})")
ax.set(xlabel="Recall", ylabel="Precision",
       title="Precision-Recall Curve")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Calibration Curve
ax = axes[0, 1]
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.plot(mean_pred_a, frac_pos_a, "o-", color="steelblue", lw=2,
        label=f"Logistic Reg (Brier={brier_a:.4f})")
ax.plot(mean_pred_b, frac_pos_b, "s-", color="#e74c3c", lw=2,
        label=f"Decision Tree (Brier={brier_b:.4f})")
ax.set(xlabel="Mean Predicted Probability", ylabel="Observed Fraction of Positives",
       title="Calibration Curve (Reliability Diagram)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Cumulative Gains Chart
ax = axes[1, 0]
ax.plot(pop_a, gain_a, color="steelblue", lw=2, label="Logistic Reg")
ax.plot(pop_b, gain_b, color="#e74c3c", lw=2, label="Decision Tree")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
ax.set(xlabel="Fraction of Population (sorted by score)",
       ylabel="Fraction of Positives Captured",
       title="Cumulative Gains Chart")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Lift Chart
ax = axes[1, 1]
step = max(1, n_test // 200)
ax.plot(pop_a[::step], lift_a[::step], color="steelblue", lw=2,
        label="Logistic Reg")
ax.plot(pop_b[::step], lift_b[::step], color="#e74c3c", lw=2,
        label="Decision Tree")
ax.axhline(1.0, color="gray", ls="--", lw=1, label="No lift (random)")
ax.set(xlabel="Fraction of Population (sorted by score)",
       ylabel="Lift", title="Lift Chart")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle("Imbalanced Classification Evaluation (95/5 class split)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
  ],
};
