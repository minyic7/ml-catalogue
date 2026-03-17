import type { Chapter } from "../types";

export const supervisedLearning: Chapter = {
  title: "Supervised Learning",
  slug: "supervised-learning",
  pages: [
    {
      title: "Linear Regression",
      slug: "linear-regression",
      description:
        "Ordinary least squares, cost functions, and gradient descent for continuous prediction",
      markdownContent: `# Linear Regression

**Linear regression** models the relationship between input features and a continuous target by fitting a straight line (or hyperplane) through the data. It is the starting point for most supervised learning.

## The Hypothesis

For a single feature $x$, the model predicts:

$$
h_\\theta(x) = \\theta_0 + \\theta_1 x
$$

In the general case with $n$ features, $h_\\theta(x) = \\theta^T x$ where $\\theta$ is the parameter vector.

## Cost Function (MSE)

We measure how well the line fits the training data using the **mean squared error** cost function:

$$
J(\\theta) = \\frac{1}{2m}\\sum_{i=1}^{m}\\bigl(h_\\theta(x^{(i)}) - y^{(i)}\\bigr)^2
$$

Here $m$ is the number of training examples. The factor $\\frac{1}{2}$ is a convenience that simplifies the derivative.

## Gradient Descent

To minimize $J(\\theta)$ we repeatedly update each parameter in the direction that reduces the cost:

$$
\\theta_j := \\theta_j - \\alpha \\frac{\\partial}{\\partial \\theta_j} J(\\theta)
$$

The learning rate $\\alpha$ controls step size. Too large and the algorithm overshoots; too small and convergence is slow.

## Ordinary Least Squares

Instead of iterating, we can solve for the optimal $\\theta$ directly with the **normal equation**: $\\theta = (X^T X)^{-1} X^T y$. This is efficient for small datasets but becomes expensive when $n$ is large.

## Why This Matters

Linear regression is used for trend analysis, forecasting, and as a building block inside more complex models. Understanding its cost function and optimization is essential — the same gradient descent idea powers neural networks.

Run the code to fit a linear regression on synthetic data and visualize the result.`,
      codeSnippet: `import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1,
                       noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print results
print("=== Linear Regression Results ===")
print(f"Coefficient (slope):  {model.coef_[0]:.4f}")
print(f"Intercept:            {model.intercept_:.4f}")
print(f"R² score (test):      {r2_score(y_test, y_pred):.4f}")
print(f"MSE (test):           {mean_squared_error(y_test, y_pred):.4f}")

# Plot data and fit line
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color="steelblue", label="Test data")
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(x_line, model.predict(x_line), color="coral", linewidth=2,
         label="Fit line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Classification",
      slug: "classification",
      description:
        "Logistic regression, sigmoid function, and decision boundaries for categorical prediction",
      markdownContent: `# Classification

In **classification** the goal is to predict a discrete label rather than a continuous value. The simplest approach — logistic regression — adapts linear regression by passing its output through a non-linear function.

## The Sigmoid Function

Logistic regression maps any real number to a probability between 0 and 1 using the **sigmoid** (logistic) function:

$$
\\sigma(z) = \\frac{1}{1 + e^{-z}}
$$

Given features $x$, the model computes $z = \\theta^T x$ and predicts $P(y=1 \\mid x) = \\sigma(\\theta^T x)$. The output is interpreted as the probability the example belongs to the positive class.

## Cost Function

We cannot use MSE for classification because the sigmoid makes $J(\\theta)$ non-convex. Instead we use **binary cross-entropy**:

$$
J(\\theta) = -\\frac{1}{m}\\sum_{i=1}^{m}\\bigl[y^{(i)}\\log(h_\\theta(x^{(i)})) + (1-y^{(i)})\\log(1-h_\\theta(x^{(i)}))\\bigr]
$$

This loss is convex, so gradient descent is guaranteed to find the global minimum.

## Decision Boundary

The model predicts class 1 when $\\sigma(\\theta^T x) \\geq 0.5$, which is equivalent to $\\theta^T x \\geq 0$. This defines a linear **decision boundary** in feature space — a line (2D), plane (3D), or hyperplane in higher dimensions.

## Binary vs. Multiclass

Logistic regression naturally handles two classes. For $K > 2$ classes, common strategies include:

- **One-vs-Rest (OvR)** — train $K$ binary classifiers, each separating one class from the rest.
- **Softmax regression** — generalize the sigmoid to $K$ classes: $P(y=k) = \\frac{e^{\\theta_k^T x}}{\\sum_j e^{\\theta_j^T x}}$.

scikit-learn handles this automatically when you pass multiclass labels.

## Why This Matters

Logistic regression is fast, interpretable, and often surprisingly competitive. It serves as the baseline for almost every classification task, and the sigmoid/softmax functions appear throughout deep learning.

Run the code to train a logistic classifier on a 2-class dataset and visualize the decision boundary.`,
      codeSnippet: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate 2-class, 2-feature dataset
X, y = make_classification(n_samples=200, n_features=2,
                           n_redundant=0, n_clusters_per_class=1,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Train logistic regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== Logistic Regression Results ===")
print(f"Accuracy:     {accuracy_score(y_test, y_pred):.4f}")
print(f"Coefficients: {model.coef_[0].round(4)}")
print(f"Intercept:    {model.intercept_[0]:.4f}")

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm",
            edgecolors="k", s=40)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Decision Boundary")
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Decision Trees",
      slug: "decision-trees",
      description:
        "Tree-based models, information gain, Gini impurity, and managing overfitting",
      markdownContent: `# Decision Trees

A **decision tree** makes predictions by learning a sequence of if-else rules from the data. At each internal node the tree asks a yes/no question about a feature, splitting the data into purer subsets until it reaches a leaf that outputs a prediction.

## Splitting Criteria

To decide which feature and threshold to split on, the algorithm measures how much a split reduces **impurity**. Two common metrics:

**Gini impurity** — for a node with $K$ classes:

$$
G = 1 - \\sum_{k=1}^{K} p_k^2
$$

where $p_k$ is the fraction of samples belonging to class $k$. A pure node (all one class) has $G = 0$.

**Information gain** uses **entropy** $H = -\\sum_{k=1}^{K} p_k \\log_2 p_k$ and selects the split that maximizes:

$$
\\text{IG} = H(\\text{parent}) - \\sum_{j} \\frac{n_j}{n} H(\\text{child}_j)
$$

The algorithm greedily picks the split with the highest information gain (or largest Gini decrease) at every node.

## Overfitting

An unrestricted tree will keep splitting until every leaf is pure, memorizing the training data perfectly. This leads to **overfitting** — high training accuracy but poor generalization. Common remedies:

- **Max depth** — limit how deep the tree can grow.
- **Min samples per leaf** — require a minimum number of samples to create a leaf.
- **Pruning** — grow a full tree, then remove branches that don't improve validation performance.

## Feature Importance

After training, a tree provides a natural measure of **feature importance**: the total reduction in impurity each feature contributes across all splits. Features never used in any split get zero importance.

## Why This Matters

Decision trees are interpretable, require no feature scaling, and handle both numerical and categorical data. They are also the foundation of powerful ensemble methods — **Random Forests** and **Gradient Boosted Trees** — which dominate tabular ML competitions.

Run the code to train a decision tree and explore how tree depth affects overfitting.`,
      codeSnippet: `import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_classification(n_samples=300, n_features=6,
                           n_informative=4, n_redundant=1,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Train a default tree and show feature importances
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
print("=== Decision Tree (no depth limit) ===")
print(f"Train accuracy: {accuracy_score(y_train, tree.predict(X_train)):.4f}")
print(f"Test accuracy:  {accuracy_score(y_test, tree.predict(X_test)):.4f}")
print(f"Tree depth:     {tree.get_depth()}")
print(f"\\nFeature importances:")
for i, imp in enumerate(tree.feature_importances_):
    print(f"  Feature {i}: {imp:.4f}")

# Depth vs accuracy tradeoff
depths = range(1, 16)
train_acc, test_acc = [], []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_acc, "o-", label="Train", color="coral")
plt.plot(depths, test_acc, "s-", label="Test", color="steelblue")
plt.xlabel("Max Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Depth vs Accuracy")
plt.legend()
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "SVM (Support Vector Machine)",
      slug: "svm",
      description:
        "Maximum-margin classifiers, support vectors, kernel trick, and soft margins",
      markdownContent: `# Support Vector Machine (SVM)

A **Support Vector Machine** finds the hyperplane that separates classes with the **maximum margin** — the largest possible gap between the decision boundary and the nearest data points from each class.

## Core Idea

Given labelled training data, there are infinitely many hyperplanes that can separate two classes. SVM picks the one that maximizes the distance to the closest points from either class. These closest points are called **support vectors** — they alone determine the position and orientation of the boundary. Removing any non-support-vector point does not change the model at all.

## The Math

The decision boundary is defined by $w \\cdot x + b = 0$. The margin (distance between the two class boundaries) is:

$$
\\text{margin} = \\frac{2}{\\|w\\|}
$$

Maximizing the margin is equivalent to minimizing $\\|w\\|$, giving the **primal optimization problem**:

$$
\\min_{w, b} \\frac{1}{2}\\|w\\|^2 \\quad \\text{s.t.} \\quad y_i(w \\cdot x_i + b) \\geq 1 \\; \\forall i
$$

Each constraint ensures that every training point is on the correct side of the margin.

## Soft Margin

Real data is rarely perfectly separable. The **soft margin** formulation introduces slack variables $\\xi_i \\geq 0$ that allow some points to violate the margin:

$$
\\min_{w, b} \\frac{1}{2}\\|w\\|^2 + C \\sum_{i=1}^{m} \\xi_i
$$

The regularisation parameter **C** controls the trade-off:
- **Large C** — penalise misclassifications heavily → narrow margin, risk of overfitting
- **Small C** — allow more misclassifications → wider margin, better generalisation

## The Kernel Trick

When classes are not linearly separable, SVM can map features to a higher-dimensional space where a linear separator exists. The **kernel trick** computes inner products in this space without explicitly performing the transformation:

| Kernel | Formula | Use Case |
|--------|---------|----------|
| Linear | $K(x_i, x_j) = x_i \\cdot x_j$ | Linearly separable data, high-dimensional text |
| RBF (Gaussian) | $K(x_i, x_j) = e^{-\\gamma \\|x_i - x_j\\|^2}$ | General-purpose, non-linear boundaries |
| Polynomial | $K(x_i, x_j) = (x_i \\cdot x_j + r)^d$ | Interaction features, moderate non-linearity |

## When to Use SVM

- **SVM vs Logistic Regression** — SVM works better when classes are well-separated; logistic regression provides calibrated probabilities and scales better to large datasets.
- **SVM vs Tree-based models** — Trees handle mixed feature types and missing values naturally; SVMs need feature scaling and struggle with very large datasets but excel in high-dimensional spaces (e.g. text classification).

Run the code to train SVMs with linear and RBF kernels and compare their decision boundaries.`,
      codeSnippet: `import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate a non-linear 2D dataset
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Train two SVMs: linear and RBF kernels
svm_linear = SVC(kernel="linear", C=1.0, random_state=42)
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

print("=== SVM Results ===")
print(f"Linear kernel — Test accuracy: "
      f"{accuracy_score(y_test, svm_linear.predict(X_test)):.4f}")
print(f"  Support vectors per class: {svm_linear.n_support_}")
print(f"RBF kernel    — Test accuracy: "
      f"{accuracy_score(y_test, svm_rbf.predict(X_test)):.4f}")
print(f"  Support vectors per class: {svm_rbf.n_support_}")

# Plot decision boundaries side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
                      np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

for ax, model, title in zip(axes, [svm_linear, svm_rbf],
                             ["Linear Kernel", "RBF Kernel"]):
    Z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm",
               edgecolors="k", s=40)
    # Highlight support vectors
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=100, facecolors="none",
               edgecolors="gold", linewidths=1.5, label="Support vectors")
    ax.set_title(f"SVM — {title}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "KNN (K-Nearest Neighbours)",
      slug: "knn",
      description:
        "Instance-based learning, distance metrics, choosing K, and the curse of dimensionality",
      markdownContent: `# K-Nearest Neighbours (KNN)

**KNN** is one of the simplest machine learning algorithms: to classify a new point, find the $K$ closest training examples and let them vote. Despite its simplicity, KNN can model complex decision boundaries and is a useful baseline for many tasks.

## Algorithm

1. Store the entire training set (no explicit training phase — KNN is a **lazy learner**).
2. For a new query point $x_q$:
   - Compute the distance from $x_q$ to every training point.
   - Select the $K$ nearest neighbours.
   - **Classification**: return the majority class among the $K$ neighbours.
   - **Regression**: return the mean (or weighted mean) of the neighbours' targets.

## Distance Metrics

The choice of distance metric affects which points count as "nearest":

| Metric | Formula | Best For |
|--------|---------|----------|
| Euclidean | $d(a,b) = \\sqrt{\\sum_{i=1}^{n}(a_i - b_i)^2}$ | Continuous features, isotropic data |
| Manhattan | $d(a,b) = \\sum_{i=1}^{n}\\lvert a_i - b_i \\rvert$ | High-dimensional data, grid-like features |
| Cosine | $d(a,b) = 1 - \\frac{a \\cdot b}{\\|a\\| \\|b\\|}$ | Text/NLP, when magnitude doesn't matter |

Feature scaling is critical — without it, features with large ranges dominate the distance calculation.

## Choosing K

The value of $K$ controls the bias-variance trade-off:

- **Small K** (e.g. 1–3) — low bias, high variance. The boundary is jagged and sensitive to noise; the model overfits.
- **Large K** (e.g. 50+) — high bias, low variance. The boundary is smooth but may miss local patterns; the model underfits.

A common strategy is to evaluate odd values of $K$ (to avoid ties in binary classification) via cross-validation and pick the one with the best validation accuracy.

## Curse of Dimensionality

In high-dimensional spaces, distances between points become increasingly uniform — all points look roughly equally far apart. This means:

- The concept of "nearest" loses meaning.
- KNN needs exponentially more data as dimensions grow.
- Dimensionality reduction (PCA, feature selection) helps mitigate this.

As a rule of thumb, KNN works best with fewer than ~20 informative features after preprocessing.

## Computational Cost

Since KNN stores all training data and computes distances at prediction time, it has:
- **Training**: $O(1)$ — just store the data.
- **Prediction**: $O(mn)$ for $m$ training points and $n$ features. KD-trees or ball trees can speed this up, but they too degrade in high dimensions.

Run the code to train KNN classifiers with different K values and visualize how K affects the decision boundary.`,
      codeSnippet: `import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Iris dataset (use first 2 features for visualisation)
iris = load_iris()
X, y = iris.data[:, :2], iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Evaluate accuracy for different K values
k_values = range(1, 26)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print("=== KNN Results ===")
print(f"Best K (by 5-fold CV): {best_k}  "
      f"(CV accuracy: {max(cv_scores):.4f})")

# Train final model with best K
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
print(f"Test accuracy (K={best_k}): "
      f"{accuracy_score(y_test, knn_best.predict(X_test)):.4f}")

# Plot: decision boundaries for K=1, best K, and K=25
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

for ax, k in zip(axes, [1, best_k, 25]):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    Z = knn.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="viridis",
               edgecolors="k", s=40)
    acc = accuracy_score(y_test, knn.predict(X_test))
    ax.set_title(f"K = {k}  (acc: {acc:.2f})")
    ax.set_xlabel("Sepal Length (scaled)")
    ax.set_ylabel("Sepal Width (scaled)")

plt.suptitle("KNN Decision Boundaries — Iris Dataset", y=1.02)
plt.tight_layout()
plt.show()

# Accuracy vs K curve
plt.figure(figsize=(8, 4))
plt.plot(k_values, cv_scores, "o-", color="steelblue")
plt.axvline(best_k, color="coral", linestyle="--",
            label=f"Best K = {best_k}")
plt.xlabel("K (number of neighbours)")
plt.ylabel("5-Fold CV Accuracy")
plt.title("KNN: Accuracy vs K")
plt.legend()
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Ensemble Methods",
      slug: "ensemble-methods",
      description:
        "Bagging, boosting, and stacking — combining multiple models for stronger predictions",
      markdownContent: `# Ensemble Methods

**Ensemble methods** combine multiple models to produce predictions that are more accurate and robust than any single model alone. The three main strategies are **bagging**, **boosting**, and **stacking**.

## Bagging (Bootstrap Aggregating)

Bagging reduces **variance** by training multiple models on different random subsets of the training data (bootstrap samples) and aggregating their predictions.

1. Draw $B$ bootstrap samples (sample with replacement) from the training set.
2. Train an independent model on each sample.
3. **Classification**: take the majority vote. **Regression**: take the average.

Because each model sees slightly different data, their errors are partially uncorrelated, and averaging smooths out the noise.

### Random Forest

A **Random Forest** extends bagging by adding **random feature selection**: at each split, the tree considers only a random subset of $\\sqrt{n}$ features (classification) or $n/3$ features (regression). This further decorrelates the trees and improves performance.

$$
\\hat{y} = \\text{mode}\\bigl\\{T_1(x),\\; T_2(x),\\; \\dots,\\; T_B(x)\\bigr\\}
$$

where each $T_b$ is a decision tree trained on a different bootstrap sample with random feature subsets.

## Boosting

Boosting reduces **bias** by training models **sequentially**, where each new model focuses on correcting the mistakes of the previous ensemble.

### Gradient Boosting

Gradient boosting builds an additive model by fitting each new tree to the **negative gradient** (pseudo-residuals) of the loss function:

$$
F_m(x) = F_{m-1}(x) + \\eta \\cdot h_m(x)
$$

where $\\eta$ is the learning rate and $h_m$ is a shallow tree fitted to the residuals. Popular implementations include **XGBoost**, **LightGBM**, and **CatBoost**. scikit-learn provides \`GradientBoostingClassifier\` which is sufficient for many tasks.

### AdaBoost

AdaBoost adjusts sample **weights** after each round — misclassified samples get higher weight so the next weak learner focuses on them. It is effective but more sensitive to noisy data than gradient boosting.

## Stacking

**Stacking** uses a **meta-learner** (often logistic regression or a linear model) to learn the optimal way to combine predictions from multiple diverse base models. The base models' predictions become features for the meta-learner, which is trained on held-out (cross-validated) predictions to avoid overfitting.

## Comparison

| Aspect | Bagging | Boosting | Stacking |
|--------|---------|----------|----------|
| Reduces | Variance | Bias | Both |
| Training | Parallel (independent models) | Sequential (each depends on previous) | Two stages |
| Overfitting risk | Low | Higher (can overfit with too many rounds) | Moderate |
| Key hyperparameters | Number of estimators | Learning rate, number of rounds, tree depth | Choice of base + meta learners |
| Example | Random Forest | XGBoost, LightGBM | Blending diverse model types |

## When to Use What

- **Random Forest** — great default for tabular data; robust, parallelisable, few hyperparameters to tune.
- **Gradient Boosting** — when you need maximum accuracy and are willing to tune hyperparameters; dominates Kaggle competitions on tabular data.
- **Stacking** — when you have diverse model types (trees, linear, KNN) and want to squeeze out the last bit of performance.

Run the code to compare a single Decision Tree, Random Forest, and Gradient Boosting on a classification task.`,
      codeSnippet: `import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=12, n_redundant=4,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        random_state=42),
}

# Train, evaluate, and time each model
results = {}
print("=== Ensemble Methods Comparison ===\\n")
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    results[name] = {"train": train_acc, "test": test_acc,
                     "cv": cv, "time": train_time}
    print(f"{name}:")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  5-Fold CV:      {cv:.4f}")
    print(f"  Training time:  {train_time:.3f}s\\n")

# Bar chart: accuracy comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
names = list(results.keys())
test_accs = [results[n]["test"] for n in names]
cv_accs = [results[n]["cv"] for n in names]

x = np.arange(len(names))
axes[0].bar(x - 0.15, test_accs, 0.3, label="Test", color="steelblue")
axes[0].bar(x + 0.15, cv_accs, 0.3, label="5-Fold CV", color="coral")
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, rotation=15)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Model Accuracy Comparison")
axes[0].legend()
axes[0].set_ylim(0.8, 1.0)

# Feature importance comparison (RF vs GB)
rf_imp = models["Random Forest"].feature_importances_
gb_imp = models["Gradient Boosting"].feature_importances_
top_n = 10
top_idx = np.argsort(rf_imp)[-top_n:]

axes[1].barh(np.arange(top_n) - 0.15, rf_imp[top_idx], 0.3,
             label="Random Forest", color="steelblue")
axes[1].barh(np.arange(top_n) + 0.15, gb_imp[top_idx], 0.3,
             label="Gradient Boosting", color="coral")
axes[1].set_yticks(np.arange(top_n))
axes[1].set_yticklabels([f"Feature {i}" for i in top_idx])
axes[1].set_xlabel("Importance")
axes[1].set_title(f"Top {top_n} Feature Importances")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
  ],
};
