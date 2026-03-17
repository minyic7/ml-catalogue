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
  ],
};
