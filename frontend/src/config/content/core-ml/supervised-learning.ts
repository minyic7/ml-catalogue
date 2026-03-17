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
  ],
};
