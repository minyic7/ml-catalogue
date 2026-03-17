import type { Chapter } from "../types";

const shapLimeMarkdown = `
# SHAP & LIME

As machine learning models grow more complex, understanding **why** a model made a particular prediction becomes critical. Interpretability is not just a nice-to-have — it drives trust, aids debugging, and satisfies regulatory requirements such as GDPR's "right to explanation."

## Why Interpretability Matters

- **Trust** — stakeholders (doctors, loan officers, judges) need to understand predictions before acting on them
- **Debugging** — interpretability reveals when a model relies on spurious correlations (e.g., predicting pneumonia risk from hospital metadata rather than clinical features)
- **Compliance** — regulations like GDPR Article 22 and the EU AI Act require that individuals affected by automated decisions can obtain meaningful explanations

## Global vs Local Interpretability

**Global interpretability** answers: "How does the model behave overall?" It summarizes which features matter most across the entire dataset.

**Local interpretability** answers: "Why did the model make this specific prediction?" It explains a single data point.

Both perspectives are essential. A feature can be globally important yet irrelevant for a particular prediction, and vice versa.

## SHAP (SHapley Additive exPlanations)

SHAP is grounded in **Shapley values** from cooperative game theory. The idea: treat each feature as a "player" in a coalition, and fairly distribute the "payout" (the prediction) among all players.

For a model $f$ and input $x$ with $M$ features, the Shapley value for feature $j$ is:

$$\\phi_j = \\sum_{S \\subseteq \\{1,\\ldots,M\\} \\setminus \\{j\\}} \\frac{|S|!\\;(M - |S| - 1)!}{M!} \\Big[ f(S \\cup \\{j\\}) - f(S) \\Big]$$

This averages the marginal contribution of feature $j$ across all possible subsets $S$ of the other features.

### Key Properties

- **Local accuracy**: the Shapley values sum to the difference between the prediction and the expected value: $f(x) = \\mathbb{E}[f] + \\sum_{j=1}^{M} \\phi_j$
- **Missingness**: features that are absent contribute zero
- **Consistency**: if a feature's marginal contribution never decreases in a new model, its Shapley value does not decrease

### Visualisations

- **Summary plot**: shows feature importance globally — each point is a SHAP value for one feature on one data point
- **Force plot**: explains a single prediction — arrows push the prediction up or down from the base value
- **Dependence plot**: shows how one feature's SHAP value varies with its actual value, revealing nonlinear effects

## LIME (Local Interpretable Model-agnostic Explanations)

LIME explains a single prediction by:

1. **Perturbing** the input: generate neighbours around the data point $x$
2. **Querying** the black-box model on each neighbour
3. **Fitting** a simple, interpretable model (typically linear regression or a decision tree) on the perturbed data, weighted by proximity to $x$

Formally, LIME solves:

$$\\xi(x) = \\arg\\min_{g \\in G} \\; \\mathcal{L}(f, g, \\pi_x) + \\Omega(g)$$

where $\\pi_x$ is a proximity kernel centred on $x$, $\\mathcal{L}$ measures how well $g$ approximates $f$ locally, and $\\Omega(g)$ penalises complexity.

### Strengths and Limitations

| | Strengths | Limitations |
|---|---|---|
| LIME | Model-agnostic, intuitive | Unstable — different runs can give different explanations |
| | Simple to implement | Sensitive to the choice of perturbation method and kernel width |

## SHAP vs LIME

| Aspect | SHAP | LIME |
|---|---|---|
| Theoretical basis | Shapley values (axiomatic) | Local linear approximation |
| Guarantees | Local accuracy, consistency, missingness | No formal guarantees |
| Scope | Global + local | Local only |
| Speed | Can be slow for exact computation; fast with TreeSHAP | Generally fast |
| Stability | Deterministic for exact SHAP | Stochastic — results vary across runs |

In practice, SHAP is preferred when theoretical rigour matters. LIME remains useful for quick, intuitive explanations and for models where SHAP approximations are unavailable.

Run the code to see a manual Shapley value computation on a small example and permutation-based feature importance as a practical SHAP-like approach.
`;

const shapLimeCode = `import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.inspection import permutation_importance
from itertools import combinations
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Load data and train model ──
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
print("Model accuracy:", f"{clf.score(X, y):.4f}")

# ── Manual Shapley Value Computation (tiny example) ──
# Demonstrate on a single prediction using a subset of features
# to keep computation tractable (Shapley values are exponential)
print("\\n" + "="*55)
print("MANUAL SHAPLEY VALUE COMPUTATION")
print("="*55)

x_explain = X[0:1]  # explain the first sample
pred_proba = clf.predict_proba(x_explain)[0]
predicted_class = np.argmax(pred_proba)
print(f"\\nExplaining prediction for sample 0:")
print(f"  Features: {dict(zip(feature_names, x_explain[0]))}")
print(f"  Predicted class: {iris.target_names[predicted_class]}")
print(f"  Predicted probability: {pred_proba[predicted_class]:.4f}")

# Use 2 features for tractable exact Shapley computation
feat_indices = [2, 3]  # petal length, petal width
feat_subset_names = [feature_names[i] for i in feat_indices]
M = len(feat_indices)

# Baseline: mean prediction when features are marginalized
baseline_pred = clf.predict_proba(X)[:, predicted_class].mean()
print(f"\\nBaseline (expected) probability for class "
      f"'{iris.target_names[predicted_class]}': {baseline_pred:.4f}")

def model_pred_with_subset(x_ref, x_bg, subset_indices, feat_indices, cls):
    """Predict using features in subset from x_ref, rest marginalized over x_bg."""
    X_masked = np.tile(x_bg, (1, 1))  # copy background
    for idx in subset_indices:
        col = feat_indices[idx]
        X_masked[:, col] = x_ref[0, col]
    return X_masked

# Compute exact Shapley values for the 2-feature subset
print(f"\\nComputing Shapley values for: {feat_subset_names}")
shapley_values = {}

for j in range(M):
    phi_j = 0.0
    other_indices = [i for i in range(M) if i != j]

    for size in range(M):  # |S| from 0 to M-1
        for S in combinations(other_indices, size):
            S = list(S)
            # f(S ∪ {j})
            X_with = model_pred_with_subset(
                x_explain, X, S + [j], feat_indices, predicted_class)
            v_with = clf.predict_proba(X_with)[:, predicted_class].mean()

            # f(S)
            X_without = model_pred_with_subset(
                x_explain, X, S, feat_indices, predicted_class)
            v_without = clf.predict_proba(X_without)[:, predicted_class].mean()

            # Shapley weight
            import math
            weight = (math.factorial(size) * math.factorial(M - size - 1)
                      / math.factorial(M))
            phi_j += weight * (v_with - v_without)

    shapley_values[feat_subset_names[j]] = phi_j
    print(f"  φ({feat_subset_names[j]}) = {phi_j:+.4f}")

print(f"\\n  Sum of Shapley values: {sum(shapley_values.values()):+.4f}")

# ── Permutation Importance (global, SHAP-like) ──
print("\\n" + "="*55)
print("PERMUTATION IMPORTANCE (sklearn)")
print("="*55)

result = permutation_importance(clf, X, y, n_repeats=30, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Shapley values for the explained sample
ax1 = axes[0]
names = list(shapley_values.keys())
vals = list(shapley_values.values())
colors = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
ax1.barh(names, vals, color=colors)
ax1.set_xlabel('Shapley Value (contribution to prediction)')
ax1.set_title(f'Local Explanation (Sample 0)\\nPredicted: {iris.target_names[predicted_class]}')
ax1.axvline(x=0, color='black', linewidth=0.8)

# Right: Permutation importance (global)
ax2 = axes[1]
sorted_idx = result.importances_mean.argsort()
ax2.barh(
    [feature_names[i] for i in sorted_idx],
    result.importances_mean[sorted_idx],
    xerr=result.importances_std[sorted_idx],
    color='#2ecc71'
)
ax2.set_xlabel('Mean Accuracy Decrease')
ax2.set_title('Global Feature Importance\\n(Permutation-based)')

plt.tight_layout()
plt.savefig('output.png', dpi=100, bbox_inches='tight')
plt.show()
print("\\nPlots saved — left: local Shapley values, right: global permutation importance")
`;

const featureImportanceMarkdown = `
# Feature Importance

Feature importance answers a deceptively simple question: **which features matter most?** The answer depends entirely on the method used, and different methods can — and frequently do — disagree.

## Impurity-Based Importance

Tree-based models (Random Forest, Gradient Boosting) split nodes by maximising an impurity criterion — Gini impurity for classification, variance reduction for regression.

For Gini impurity at a node $t$ with $K$ classes:

$$G(t) = 1 - \\sum_{k=1}^{K} p_k^2$$

where $p_k$ is the proportion of class $k$ samples at node $t$. Each split reduces impurity by:

$$\\Delta G = G(\\text{parent}) - \\frac{n_{\\text{left}}}{n} G(\\text{left}) - \\frac{n_{\\text{right}}}{n} G(\\text{right})$$

**Impurity-based importance** for feature $j$ sums the impurity reduction across all nodes that split on feature $j$, weighted by the number of samples reaching each node:

$$\\text{Importance}(j) = \\sum_{t \\in \\text{nodes splitting on } j} \\frac{n_t}{N} \\Delta G(t)$$

### Bias Warning

Impurity-based importance is **biased toward high-cardinality features** (features with many unique values). A random ID column can appear highly "important" simply because it offers many possible split points. This is a well-known deficiency.

## Permutation Importance

Permutation importance measures how much the model's performance degrades when a single feature is shuffled:

1. Compute baseline score $s$ on validation data
2. For feature $j$: randomly shuffle column $j$, compute new score $s_j$
3. Importance of feature $j$: $\\Delta s_j = s - s_j$

$$\\text{PermImp}(j) = \\frac{1}{R} \\sum_{r=1}^{R} \\big(s - s_j^{(r)}\\big)$$

where $R$ is the number of shuffle repetitions (typically 10–30).

### Advantages

- **Model-agnostic** — works with any model, not just trees
- **Less biased** — does not favour high-cardinality features
- **Uses validation data** — reflects actual predictive power, not training artefacts

### Limitations

- **Correlated features** dilute importance — if features $j$ and $k$ are correlated, shuffling $j$ alone barely hurts because $k$ compensates

## Drop-Column Importance

The most rigorous (and expensive) approach: retrain the model without each feature and measure the performance change.

$$\\text{DropImp}(j) = s_{\\text{full}} - s_{\\text{without } j}$$

This avoids the correlation problem — removing a feature forces the model to truly learn without it. However, it requires $M + 1$ full training runs, making it impractical for large models or many features.

## When Importances Disagree

Consider two highly correlated features $A$ and $B$ ($\\rho \\approx 0.95$):

| Method | Feature A | Feature B | Explanation |
|---|---|---|---|
| Impurity-based | High | High | Both offer good splits |
| Permutation | Low | Low | Shuffling one barely hurts because the other compensates |
| Drop-column | High | High | Removing one forces the model to lose that signal entirely |

**Key insight**: correlated features create a "substitution effect." Permutation importance underestimates each feature individually, while impurity importance can overcount their shared contribution.

### Practical Recommendations

1. **Start with permutation importance** — it is model-agnostic and less biased
2. **Use impurity importance as a quick sanity check**, but do not rely on it as the sole measure
3. **Check for feature correlations** — if features are highly correlated, results from any single method should be interpreted cautiously
4. **Use drop-column importance for critical decisions** (e.g., regulatory submissions), when the computational cost is justified

Run the code to train a Gradient Boosting model, compare impurity-based and permutation importance, and see where they disagree.
`;

const featureImportanceCode = `import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Create dataset with correlated features ──
# Generate base features, then add a correlated copy to show disagreement
X_base, y = make_classification(
    n_samples=1000, n_features=6, n_informative=4,
    n_redundant=0, n_clusters_per_class=2, random_state=42
)

# Add a feature highly correlated with feature 0
correlated_feat = X_base[:, 0] + np.random.normal(0, 0.1, size=1000)
# Add a high-cardinality random feature (should be unimportant)
random_id = np.random.uniform(0, 1000, size=1000)

X = np.column_stack([X_base, correlated_feat, random_id])
feature_names = [f'feat_{i}' for i in range(6)] + ['corr_of_0', 'random_id']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ── Train Gradient Boosting model ──
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
)
gb.fit(X_train, y_train)

train_acc = gb.score(X_train, y_train)
test_acc = gb.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# ── Method 1: Impurity-based importance ──
impurity_imp = gb.feature_importances_
print("\\n" + "="*55)
print("IMPURITY-BASED IMPORTANCE")
print("="*55)
for name, imp in sorted(zip(feature_names, impurity_imp),
                         key=lambda x: -x[1]):
    print(f"  {name:12s}: {imp:.4f}")

# ── Method 2: Permutation importance ──
perm_result = permutation_importance(
    gb, X_test, y_test, n_repeats=30, random_state=42
)
perm_imp = perm_result.importances_mean
perm_std = perm_result.importances_std

print("\\n" + "="*55)
print("PERMUTATION IMPORTANCE")
print("="*55)
for name, imp, std in sorted(zip(feature_names, perm_imp, perm_std),
                              key=lambda x: -x[1]):
    print(f"  {name:12s}: {imp:.4f} ± {std:.4f}")

# ── Highlight disagreements ──
print("\\n" + "="*55)
print("DISAGREEMENTS")
print("="*55)

# Rank features by each method
impurity_rank = np.argsort(-impurity_imp)
perm_rank = np.argsort(-perm_imp)

impurity_ranks = {i: rank for rank, i in enumerate(impurity_rank)}
perm_ranks = {i: rank for rank, i in enumerate(perm_rank)}

print(f"  {'Feature':12s} | {'Impurity Rank':>14s} | {'Permutation Rank':>16s} | {'Δ Rank':>6s}")
print(f"  {'-'*12} | {'-'*14} | {'-'*16} | {'-'*6}")
for i, name in enumerate(feature_names):
    ir = impurity_ranks[i] + 1
    pr = perm_ranks[i] + 1
    delta = abs(ir - pr)
    flag = " ◀ DISAGREE" if delta >= 2 else ""
    print(f"  {name:12s} | {ir:14d} | {pr:16d} | {delta:6d}{flag}")

# ── Side-by-side bar charts ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Impurity-based
sorted_idx_imp = np.argsort(impurity_imp)
colors_imp = ['#e74c3c' if feature_names[i] in ('random_id', 'corr_of_0')
              else '#3498db' for i in sorted_idx_imp]
axes[0].barh(
    [feature_names[i] for i in sorted_idx_imp],
    impurity_imp[sorted_idx_imp],
    color=colors_imp
)
axes[0].set_xlabel('Importance')
axes[0].set_title('Impurity-Based Importance\\n(biased toward high-cardinality)')

# Permutation-based
sorted_idx_perm = np.argsort(perm_imp)
colors_perm = ['#e74c3c' if feature_names[i] in ('random_id', 'corr_of_0')
               else '#2ecc71' for i in sorted_idx_perm]
axes[1].barh(
    [feature_names[i] for i in sorted_idx_perm],
    perm_imp[sorted_idx_perm],
    xerr=perm_std[sorted_idx_perm],
    color=colors_perm
)
axes[1].set_xlabel('Mean Accuracy Decrease')
axes[1].set_title('Permutation Importance\\n(model-agnostic, less biased)')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='Potentially misleading features'),
    Patch(facecolor='#3498db', label='Impurity importance'),
    Patch(facecolor='#2ecc71', label='Permutation importance'),
]
fig.legend(handles=legend_elements, loc='lower center',
           ncol=3, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig('output.png', dpi=100, bbox_inches='tight')
plt.show()

print("\\nNotice: 'random_id' may rank higher in impurity-based importance")
print("(due to high cardinality) but near zero in permutation importance.")
print("'corr_of_0' shows how correlated features behave differently across methods.")
`;

export const modelInterpretability: Chapter = {
  title: "Model Interpretability",
  slug: "model-interpretability",
  pages: [
    {
      title: "SHAP & LIME",
      slug: "shap-lime",
      description:
        "Understand SHAP (Shapley values) and LIME for explaining individual predictions and global model behavior",
      markdownContent: shapLimeMarkdown,
      codeSnippet: shapLimeCode,
      codeLanguage: "python",
    },
    {
      title: "Feature Importance",
      slug: "feature-importance",
      description:
        "Compare impurity-based, permutation, and drop-column feature importance methods",
      markdownContent: featureImportanceMarkdown,
      codeSnippet: featureImportanceCode,
      codeLanguage: "python",
    },
  ],
};
