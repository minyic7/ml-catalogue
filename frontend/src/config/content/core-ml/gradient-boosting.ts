import type { Chapter } from "../types";

export const gradientBoosting: Chapter = {
  title: "Gradient Boosting",
  slug: "gradient-boosting",
  pages: [
    {
      title: "Gradient Boosting Fundamentals",
      slug: "gradient-boosting-fundamentals",
      description:
        "Boosting vs bagging, gradient descent in function space, loss functions, and building a simple GBM from scratch",
      markdownContent: `# Gradient Boosting Fundamentals

**Gradient boosting** builds an ensemble of weak learners (typically decision trees) **sequentially**, where each new tree corrects the errors of the combined ensemble so far. Unlike bagging (e.g. Random Forest) which reduces variance by averaging independent models, boosting reduces **bias** by focusing on hard-to-predict examples.

## Boosting vs Bagging

| | Bagging | Boosting |
|---|---|---|
| Training | Parallel, independent models | Sequential, each model depends on previous |
| Sampling | Bootstrap samples | Weighted samples or residual fitting |
| Reduces | Variance | Bias (and variance with regularization) |
| Risk | Underfitting if base learner is too weak | Overfitting if too many rounds |

## Gradient Descent in Function Space

Instead of optimizing parameters, gradient boosting performs gradient descent in **function space**. We start with an initial prediction $F_0(x)$ and iteratively add functions:

$$
F_m(x) = F_{m-1}(x) + \\eta \\cdot h_m(x)
$$

where $\\eta$ is the learning rate and $h_m$ is a weak learner fit to the **negative gradient** (pseudo-residuals) of the loss:

$$
r_{im} = -\\frac{\\partial L(y_i, F(x_i))}{\\partial F(x_i)} \\Bigg|_{F = F_{m-1}}
$$

## Common Loss Functions

For **regression** (squared error):

$$
L(y, F) = \\frac{1}{2}(y - F)^2 \\quad \\Rightarrow \\quad r_i = y_i - F_{m-1}(x_i)
$$

For **binary classification** (log loss / deviance):

$$
L(y, F) = -\\bigl[y \\ln \\sigma(F) + (1-y)\\ln(1-\\sigma(F))\\bigr]
$$

where $\\sigma(F) = \\frac{1}{1+e^{-F}}$ is the sigmoid function.

## Bias-Variance Tradeoff

Boosting primarily reduces bias, but can overfit with too many rounds. Key regularization controls:

- **Learning rate** ($\\eta$): smaller values require more trees but generalize better
- **Tree depth**: shallow trees (stumps or depth 3–5) act as weak learners
- **Subsampling**: using a fraction of data per round (stochastic gradient boosting)
- **Early stopping**: halt training when validation loss stops improving

## The GBM Algorithm

1. Initialize $F_0(x) = \\arg\\min_c \\sum L(y_i, c)$
2. For $m = 1, \\dots, M$:
   - Compute pseudo-residuals $r_{im}$
   - Fit tree $h_m$ to residuals
   - Update $F_m(x) = F_{m-1}(x) + \\eta \\cdot h_m(x)$
3. Output $F_M(x)$

Run the code to see a simple GBM built from scratch with numpy, then compared against scikit-learn's implementation on a synthetic fraud detection dataset.`,
      codeSnippet: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from ml_catalogue_runtime import MODE

# --- Dataset: synthetic fraud detection ---
mode = MODE
n_samples = 1000 if mode == "quick" else 10000
n_rounds = 30 if mode == "quick" else 100

X, y = make_classification(
    n_samples=n_samples, n_features=20, n_informative=12,
    n_redundant=4, weights=[0.95, 0.05],  # 5% fraud rate
    flip_y=0.01, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print(f"=== Gradient Boosting from Scratch (mode={mode}) ===")
print(f"Dataset: {n_samples} samples, {y.mean():.1%} fraud rate\\n")

# --- Simple GBM from scratch ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

learning_rate = 0.1
F_train = np.zeros(len(X_train))
F_test = np.zeros(len(X_test))
train_losses = []

for m in range(n_rounds):
    # Pseudo-residuals for log loss
    p_train = sigmoid(F_train)
    residuals = y_train - p_train

    # Fit a shallow tree to residuals
    tree = DecisionTreeRegressor(max_depth=3, random_state=m)
    tree.fit(X_train, residuals)

    # Update predictions
    F_train += learning_rate * tree.predict(X_train)
    F_test += learning_rate * tree.predict(X_test)

    # Track log loss
    p = sigmoid(F_train)
    eps = 1e-15
    loss = -np.mean(y_train * np.log(p + eps) + (1 - y_train) * np.log(1 - p + eps))
    train_losses.append(loss)

# Scratch model results
scratch_preds = (sigmoid(F_test) > 0.5).astype(int)
scratch_proba = sigmoid(F_test)
print("From-Scratch GBM:")
print(f"  Accuracy:  {accuracy_score(y_test, scratch_preds):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, scratch_proba):.4f}")

# --- Scikit-learn GBM for comparison ---
sklearn_gbm = GradientBoostingClassifier(
    n_estimators=n_rounds, learning_rate=0.1, max_depth=3, random_state=42)
sklearn_gbm.fit(X_train, y_train)
sk_proba = sklearn_gbm.predict_proba(X_test)[:, 1]
sk_preds = sklearn_gbm.predict(X_test)
print(f"\\nScikit-learn GBM:")
print(f"  Accuracy:  {accuracy_score(y_test, sk_preds):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, sk_proba):.4f}")

# --- Visualizations ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Training loss curve
axes[0].plot(range(1, n_rounds + 1), train_losses, color="steelblue", linewidth=2)
axes[0].set_xlabel("Boosting Round")
axes[0].set_ylabel("Log Loss")
axes[0].set_title("Training Loss (From-Scratch GBM)")
axes[0].grid(True, alpha=0.3)

# Feature importance (sklearn)
importances = sklearn_gbm.feature_importances_
top_k = 10
top_idx = np.argsort(importances)[-top_k:]
axes[1].barh(range(top_k), importances[top_idx], color="steelblue")
axes[1].set_yticks(range(top_k))
axes[1].set_yticklabels([f"Feature {i}" for i in top_idx])
axes[1].set_xlabel("Importance")
axes[1].set_title("Top Feature Importances (sklearn GBM)")
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "XGBoost",
      slug: "xgboost",
      description:
        "Regularized objective, tree pruning, column subsampling, and XGBoost for fraud detection",
      markdownContent: `# XGBoost

**XGBoost** (eXtreme Gradient Boosting) is an optimized, scalable implementation of gradient boosting that introduced several key innovations making it the dominant algorithm for structured/tabular data.

## Regularized Objective

XGBoost adds explicit regularization to the objective function:

$$
\\mathcal{L} = \\sum_{i=1}^{n} L(y_i, \\hat{y}_i) + \\sum_{k=1}^{K} \\Omega(f_k)
$$

where the regularization term penalizes model complexity:

$$
\\Omega(f) = \\gamma T + \\frac{1}{2}\\lambda \\sum_{j=1}^{T} w_j^2
$$

Here $T$ is the number of leaves, $w_j$ are leaf weights, $\\gamma$ controls the minimum loss reduction for a split, and $\\lambda$ is L2 regularization on leaf weights.

## Second-Order Approximation

XGBoost uses a second-order Taylor expansion of the loss:

$$
\\mathcal{L}^{(t)} \\approx \\sum_{i=1}^{n}\\bigl[g_i f_t(x_i) + \\frac{1}{2} h_i f_t^2(x_i)\\bigr] + \\Omega(f_t)
$$

where $g_i = \\partial_{\\hat{y}} L(y_i, \\hat{y}^{(t-1)})$ and $h_i = \\partial^2_{\\hat{y}} L(y_i, \\hat{y}^{(t-1)})$ are the first and second derivatives of the loss. This enables a closed-form solution for optimal leaf weights:

$$
w_j^* = -\\frac{\\sum_{i \\in I_j} g_i}{\\sum_{i \\in I_j} h_i + \\lambda}
$$

## Split Finding & Tree Pruning

The gain for a candidate split is:

$$
\\text{Gain} = \\frac{1}{2}\\left[\\frac{G_L^2}{H_L + \\lambda} + \\frac{G_R^2}{H_R + \\lambda} - \\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}\\right] - \\gamma
$$

XGBoost prunes trees **post-hoc**: it grows to max depth, then removes splits with negative gain. This is more effective than greedy pre-pruning.

## Key Features for Fraud Detection

- **\`scale_pos_weight\`**: handles class imbalance by weighting the positive (fraud) class — set to ratio of negative/positive samples
- **Column subsampling**: randomly samples features at tree or split level, reducing overfitting and correlation between trees
- **Early stopping**: monitors validation metric to halt training before overfitting
- **Built-in cross-validation**: \`xgb.cv()\` for hyperparameter tuning

## Sparsity-Aware Split Finding

XGBoost natively handles missing values by learning a default direction at each split — crucial for real-world transaction data where features may be incomplete.

Run the code to train XGBoost on a synthetic fraud dataset with class imbalance handling, early stopping, and feature importance visualization.`,
      codeSnippet: `import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report, confusion_matrix)
import matplotlib.pyplot as plt
from ml_catalogue_runtime import MODE

# --- Synthetic fraud dataset ---
mode = MODE
n_samples = 1000 if mode == "quick" else 15000
n_rounds = 50 if mode == "quick" else 300

X, y = make_classification(
    n_samples=n_samples, n_features=20, n_informative=12,
    n_redundant=4, weights=[0.95, 0.05],
    flip_y=0.01, random_state=42
)

feature_names = [f"txn_feat_{i}" for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Handle class imbalance
neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale_weight = neg_count / pos_count

print(f"=== XGBoost Fraud Detection (mode={mode}) ===")
print(f"Training: {len(y_train)} samples ({pos_count} fraud, {neg_count} legit)")
print(f"scale_pos_weight: {scale_weight:.1f}\\n")

# --- Train XGBoost with early stopping ---
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "learning_rate": 0.1,
    "scale_pos_weight": scale_weight,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "lambda": 1.0,
    "gamma": 0.1,
    "seed": 42,
}

evals_result = {}
model = xgb.train(
    params, dtrain,
    num_boost_round=n_rounds,
    evals=[(dtrain, "train"), (dtest, "eval")],
    evals_result=evals_result,
    early_stopping_rounds=15,
    verbose_eval=False
)

# --- Results ---
y_proba = model.predict(dtest)
y_pred = (y_proba > 0.5).astype(int)

print(f"Best iteration: {model.best_iteration}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\\n")
print("Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Legit", "Fraud"], zero_division=0))

# --- Visualizations ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# AUC over rounds
axes[0].plot(evals_result["train"]["auc"], label="Train", linewidth=2)
axes[0].plot(evals_result["eval"]["auc"], label="Eval", linewidth=2)
axes[0].axvline(model.best_iteration, color="red", linestyle="--",
                alpha=0.7, label=f"Best ({model.best_iteration})")
axes[0].set_xlabel("Boosting Round")
axes[0].set_ylabel("AUC")
axes[0].set_title("XGBoost Training Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Feature importance
importance = model.get_score(importance_type="gain")
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
names, values = zip(*sorted_imp) if sorted_imp else ([], [])
axes[1].barh(range(len(names)), values, color="darkorange")
axes[1].set_yticks(range(len(names)))
axes[1].set_yticklabels(names)
axes[1].set_xlabel("Gain")
axes[1].set_title("XGBoost Feature Importance (Top 10)")
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis="x")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[2].imshow(cm, cmap="Blues")
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(["Legit", "Fraud"])
axes[2].set_yticklabels(["Legit", "Fraud"])
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")
axes[2].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=16, color="white" if cm[i, j] > cm.max()/2 else "black")
plt.colorbar(im, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "LightGBM",
      slug: "lightgbm",
      description:
        "Histogram-based splitting, leaf-wise growth, GOSS, EFB, and LightGBM for fraud detection with speed comparison",
      markdownContent: `# LightGBM

**LightGBM** (Light Gradient Boosting Machine) is a gradient boosting framework developed by Microsoft, optimized for speed and memory efficiency on large-scale datasets. It introduces several algorithmic innovations that make it significantly faster than traditional GBM implementations.

## Histogram-Based Splitting

Instead of sorting features to find split points, LightGBM **bins** continuous values into discrete histograms:

$$
\\text{bin}(x_j) = \\lfloor k \\cdot \\frac{x_j - \\min(x_j)}{\\max(x_j) - \\min(x_j)} \\rfloor
$$

This reduces split-finding from $O(n \\cdot \\text{features})$ to $O(k \\cdot \\text{features})$ where $k$ (default 255) is the number of bins. It also reduces memory usage and improves cache efficiency.

## Leaf-Wise Growth vs Level-Wise

Most GBM implementations grow trees **level-wise** (breadth-first), splitting all leaves at the current depth. LightGBM grows **leaf-wise** (best-first), always splitting the leaf with the highest gain:

- **Level-wise**: balanced trees, safer against overfitting, but wastes splits on low-gain leaves
- **Leaf-wise**: deeper, asymmetric trees that converge faster with fewer leaves, but can overfit on small data — controlled by \`num_leaves\` and \`max_depth\`

## GOSS (Gradient-Based One-Side Sampling)

GOSS keeps all instances with large gradients (hard examples) and randomly samples from instances with small gradients (easy examples):

$$
\\tilde{G}_j = \\sum_{x_i \\in A} g_i + \\frac{1-a}{b} \\sum_{x_i \\in B} g_i
$$

where $A$ is the top-$a$ fraction by gradient magnitude and $B$ is a random $b$ fraction from the rest. This approximates the full gradient while reducing computation.

## EFB (Exclusive Feature Bundling)

Many features in real-world data are sparse and rarely nonzero simultaneously. EFB bundles such **mutually exclusive features** together, reducing effective dimensionality without loss of information.

## Categorical Feature Handling

LightGBM natively handles categorical features using an optimal split algorithm rather than requiring one-hot encoding. It finds the best partition of category values by sorting on cumulative gradient statistics.

## LightGBM vs XGBoost for Fraud Detection

| Aspect | XGBoost | LightGBM |
|---|---|---|
| Split finding | Pre-sorted or histogram | Histogram only (faster) |
| Tree growth | Level-wise | Leaf-wise (more accurate per tree) |
| Speed | Fast | Faster (2–10x typical) |
| Memory | Higher | Lower (histograms) |
| Categoricals | Requires encoding | Native support |

Run the code to train LightGBM on the same synthetic fraud dataset and compare speed and accuracy against XGBoost.`,
      codeSnippet: `import numpy as np
import time
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report, roc_curve)
import matplotlib.pyplot as plt
from ml_catalogue_runtime import MODE

# --- Synthetic fraud dataset ---
mode = MODE
n_samples = 1000 if mode == "quick" else 15000
n_rounds = 50 if mode == "quick" else 300

X, y = make_classification(
    n_samples=n_samples, n_features=20, n_informative=12,
    n_redundant=4, weights=[0.95, 0.05],
    flip_y=0.01, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale_weight = neg_count / pos_count

print(f"=== LightGBM vs XGBoost (mode={mode}) ===")
print(f"Dataset: {n_samples} samples, {y.mean():.1%} fraud rate\\n")

# --- XGBoost ---
start = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=n_rounds, max_depth=5, learning_rate=0.1,
    scale_pos_weight=scale_weight, colsample_bytree=0.8,
    subsample=0.8, reg_lambda=1.0,
    eval_metric="auc", random_state=42, verbosity=0
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              verbose=False)
xgb_time = time.time() - start
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

# --- LightGBM ---
start = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=n_rounds, max_depth=-1, num_leaves=31,
    learning_rate=0.1, scale_pos_weight=scale_weight,
    colsample_bytree=0.8, subsample=0.8, reg_lambda=1.0,
    random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.log_evaluation(period=0)])
lgb_time = time.time() - start
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
lgb_auc = roc_auc_score(y_test, lgb_proba)
lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test))

# --- Print comparison ---
print(f"{'Metric':<20} {'XGBoost':>10} {'LightGBM':>10}")
print("-" * 42)
print(f"{'ROC-AUC':<20} {xgb_auc:>10.4f} {lgb_auc:>10.4f}")
print(f"{'Accuracy':<20} {xgb_acc:>10.4f} {lgb_acc:>10.4f}")
print(f"{'Training time (s)':<20} {xgb_time:>10.3f} {lgb_time:>10.3f}")
print(f"{'Speedup':<20} {'':>10} {xgb_time/lgb_time:>9.1f}x")

print(f"\\nLightGBM Classification Report:")
print(classification_report(y_test, lgb_model.predict(X_test),
      target_names=["Legit", "Fraud"], zero_division=0))

# --- Visualizations ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC curves
for name, proba, color in [("XGBoost", xgb_proba, "darkorange"),
                             ("LightGBM", lgb_proba, "steelblue")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    axes[0].plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{name} (AUC={auc_val:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Speed comparison
bars = axes[1].bar(["XGBoost", "LightGBM"], [xgb_time, lgb_time],
                    color=["darkorange", "steelblue"])
axes[1].set_ylabel("Training Time (seconds)")
axes[1].set_title("Training Speed Comparison")
axes[1].grid(True, alpha=0.3, axis="y")
for bar, t in zip(bars, [xgb_time, lgb_time]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{t:.3f}s", ha="center", va="bottom", fontweight="bold")

# LightGBM feature importance
importances = lgb_model.feature_importances_
top_k = 10
top_idx = np.argsort(importances)[-top_k:]
axes[2].barh(range(top_k), importances[top_idx], color="steelblue")
axes[2].set_yticks(range(top_k))
axes[2].set_yticklabels([f"Feature {i}" for i in top_idx])
axes[2].set_xlabel("Split Count")
axes[2].set_title("LightGBM Feature Importance (Top 10)")
axes[2].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "CatBoost",
      slug: "catboost",
      description:
        "Ordered boosting, symmetric trees, native categorical encoding, and CatBoost on mixed-feature transaction data with SHAP",
      markdownContent: `# CatBoost

**CatBoost** (Categorical Boosting), developed by Yandex, is a gradient boosting library designed to handle **categorical features natively** and reduce prediction shift through a novel ordered boosting technique. It excels on datasets with mixed numerical and categorical features — common in transaction and fraud modelling.

## Ordered Boosting

Standard gradient boosting suffers from **prediction shift**: the model's predictions on training data are biased because the same data was used to compute gradients. CatBoost addresses this with **ordered boosting**:

For each example $x_i$, residuals are computed using a model trained only on examples that appear **before** $x_i$ in a random permutation:

$$
r_i^{(\\sigma)} = y_{\\sigma(i)} - M_{\\sigma(i)-1}(x_{\\sigma(i)})
$$

where $\\sigma$ is a random permutation and $M_{\\sigma(i)-1}$ is the model trained on examples $\\{\\sigma(1), \\dots, \\sigma(i-1)\\}$. This eliminates target leakage and reduces overfitting, especially on small datasets.

## Symmetric (Oblivious) Decision Trees

CatBoost uses **oblivious decision trees** where the same splitting condition is used across an entire level:

- All nodes at depth $d$ use the same feature and threshold
- Results in balanced trees with exactly $2^d$ leaves
- Very fast inference (can be compiled to simple if-else chains)
- Acts as strong regularization against overfitting

## Native Categorical Encoding

Instead of one-hot encoding, CatBoost uses **ordered target statistics**:

$$
\\hat{x}_i^k = \\frac{\\sum_{j: \\sigma(j) < \\sigma(i)} [x_j^k = x_i^k] \\cdot y_j + a \\cdot p}{\\sum_{j: \\sigma(j) < \\sigma(i)} [x_j^k = x_i^k] + a}
$$

where $a$ is a smoothing parameter and $p$ is the prior (global target mean). The ordering prevents target leakage that plagues naive target encoding.

## CatBoost for Transaction/Fraud Data

Transaction datasets typically contain:
- **Categorical**: merchant category, payment method, card type, country
- **Numerical**: amount, time since last transaction, distance from home
- **High cardinality**: merchant ID, user ID

CatBoost handles all of these natively without manual feature engineering.

## Feature Importance with SHAP

CatBoost integrates with **SHAP** (SHapley Additive exPlanations) to explain individual predictions — critical in fraud detection where decisions must be interpretable:

$$
\\phi_j = \\sum_{S \\subseteq N \\setminus \\{j\\}} \\frac{|S|!(|N|-|S|-1)!}{|N|!}\\bigl[f(S \\cup \\{j\\}) - f(S)\\bigr]
$$

Run the code to train CatBoost on a synthetic transaction dataset with mixed categorical and numerical features, and visualize SHAP-based feature importance.`,
      codeSnippet: `import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report, confusion_matrix)
import matplotlib.pyplot as plt
from ml_catalogue_runtime import MODE

# --- Synthetic transaction dataset with categorical features ---
mode = MODE
n_samples = 1000 if mode == "quick" else 15000
n_rounds = 50 if mode == "quick" else 300

np.random.seed(42)

# Numerical features
amount = np.random.exponential(scale=150, size=n_samples)
hour = np.random.randint(0, 24, size=n_samples)
days_since_last = np.random.exponential(scale=3, size=n_samples)
distance_from_home = np.random.exponential(scale=50, size=n_samples)
num_txn_24h = np.random.poisson(lam=3, size=n_samples)

# Categorical features
categories = {
    "merchant_type": ["retail", "online", "restaurant", "travel",
                      "grocery", "entertainment", "gas_station"],
    "payment_method": ["chip", "swipe", "contactless", "online", "manual"],
    "card_type": ["visa", "mastercard", "amex", "discover"],
    "country": ["US", "UK", "CA", "DE", "FR", "BR", "IN", "other"],
}

cat_data = {}
for col, vals in categories.items():
    cat_data[col] = np.random.choice(vals, size=n_samples)

# Generate fraud labels with realistic patterns
fraud_score = (
    0.3 * (amount > 500).astype(float) +
    0.2 * ((hour >= 1) & (hour <= 5)).astype(float) +
    0.15 * (distance_from_home > 100).astype(float) +
    0.15 * (num_txn_24h > 5).astype(float) +
    0.1 * np.isin(cat_data["payment_method"], ["online", "manual"]).astype(float) +
    0.1 * np.isin(cat_data["country"], ["BR", "IN", "other"]).astype(float) +
    np.random.normal(0, 0.15, n_samples)
)
threshold = np.percentile(fraud_score, 95)  # ~5% fraud rate
y = (fraud_score > threshold).astype(int)

# Build DataFrame
df = pd.DataFrame({
    "amount": amount, "hour": hour, "days_since_last": days_since_last,
    "distance_from_home": distance_from_home, "num_txn_24h": num_txn_24h,
    **cat_data
})

cat_features = list(categories.keys())
feature_names = list(df.columns)

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.3, random_state=42, stratify=y)

print(f"=== CatBoost Transaction Fraud Detection (mode={mode}) ===")
print(f"Dataset: {n_samples} samples, {y.mean():.1%} fraud rate")
print(f"Features: {len(feature_names)} ({len(cat_features)} categorical)\\n")

# --- Train CatBoost ---
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=n_rounds,
    depth=6,
    learning_rate=0.1,
    auto_class_weights="Balanced",
    eval_metric="AUC",
    random_seed=42,
    verbose=0,
    early_stopping_rounds=15,
)
model.fit(train_pool, eval_set=test_pool, verbose=0)

# --- Results ---
y_proba = model.predict_proba(test_pool)[:, 1]
y_pred = model.predict(test_pool)

print(f"Best iteration: {model.get_best_iteration()}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\\n")
print("Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Legit", "Fraud"], zero_division=0))

# --- SHAP-based feature importance ---
shap_values = model.get_feature_importance(test_pool, type="ShapValues")
# Last column is base value; remove it
shap_importance = np.abs(shap_values[:, :-1]).mean(axis=0)

# --- Visualizations ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Feature importance (SHAP)
sorted_idx = np.argsort(shap_importance)
axes[0].barh(range(len(feature_names)), shap_importance[sorted_idx],
             color=["coral" if feature_names[i] in cat_features
                    else "steelblue" for i in sorted_idx])
axes[0].set_yticks(range(len(feature_names)))
axes[0].set_yticklabels([feature_names[i] for i in sorted_idx])
axes[0].set_xlabel("Mean |SHAP Value|")
axes[0].set_title("CatBoost Feature Importance (SHAP)")
axes[0].grid(True, alpha=0.3, axis="x")
# Legend
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color="steelblue", label="Numerical"),
                         Patch(color="coral", label="Categorical")],
               loc="lower right")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap="Oranges")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["Legit", "Fraud"])
axes[1].set_yticklabels(["Legit", "Fraud"])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=16, color="white" if cm[i, j] > cm.max()/2 else "black")
plt.colorbar(im, ax=axes[1], fraction=0.046)

# SHAP summary for top features (bee swarm approximation)
top_n = 8
top_feat_idx = np.argsort(shap_importance)[-top_n:]
for rank, feat_i in enumerate(top_feat_idx):
    vals = shap_values[:, feat_i]
    feat_vals = X_test.iloc[:, feat_i]
    if feature_names[feat_i] in cat_features:
        color = "coral"
    else:
        # Color by feature value (normalized)
        fv = pd.to_numeric(feat_vals, errors="coerce").values
        fv_norm = (fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv) + 1e-10)
        color = plt.cm.coolwarm(fv_norm)
    axes[2].scatter(vals, np.full_like(vals, rank) + np.random.normal(0, 0.1, len(vals)),
                    c=color, alpha=0.3, s=8, rasterized=True)
axes[2].set_yticks(range(top_n))
axes[2].set_yticklabels([feature_names[i] for i in top_feat_idx])
axes[2].set_xlabel("SHAP Value")
axes[2].set_title("SHAP Summary (Top Features)")
axes[2].axvline(0, color="black", linewidth=0.5)
axes[2].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
  ],
};
