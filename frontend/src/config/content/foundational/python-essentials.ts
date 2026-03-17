import type { Chapter } from "../types";

export const pythonEssentials: Chapter = {
  title: "Python Essentials",
  slug: "python-essentials",
  pages: [
    {
      title: "NumPy Fundamentals",
      slug: "numpy-fundamentals",
      description: "Array creation, indexing, broadcasting, and vectorized operations",
      markdownContent: `# NumPy Fundamentals

NumPy is the backbone of numerical computing in Python. Its \`ndarray\` type provides fast, memory-efficient arrays with **vectorized operations** that avoid slow Python loops.

## Array Creation & Indexing

Create arrays from lists, or use built-in generators like \`np.zeros\`, \`np.arange\`, and \`np.linspace\`. NumPy arrays support powerful indexing — slices, boolean masks, and fancy indexing all return views or copies efficiently.

## Broadcasting

Broadcasting lets NumPy operate on arrays of different shapes without copying data. When two arrays have different dimensions, NumPy automatically expands the smaller one. The rule is simple: dimensions are compared from the right, and they must either match or one of them must be 1.

For arrays with shapes $(m, n)$ and $(1, n)$:

$$
C_{ij} = A_{ij} + B_{1j} \\quad \\text{for all } i = 1, \\ldots, m
$$

The row vector $B$ is "broadcast" across every row of $A$.

## Vectorized Operations vs Loops

A vectorized operation applies a compiled C function across an entire array in one call, avoiding Python's per-element interpreter overhead. For an element-wise computation $y_i = f(x_i)$, vectorization turns an $O(n)$ Python loop into a single low-level call:

$$
\\mathbf{y} = f(\\mathbf{x}) \\quad \\text{(vectorized, executes in C)}
$$

The code below demonstrates array creation, broadcasting, and a timing comparison between loops and vectorized computation.`,
      codeSnippet: `import numpy as np
import time

# --- Array Creation ---
a = np.array([1, 2, 3, 4, 5])
b = np.linspace(0, 1, 5)       # 5 evenly spaced values in [0, 1]
print("a         :", a)
print("linspace  :", b)

# --- Indexing & Slicing ---
matrix = np.arange(12).reshape(3, 4)
print("\\nMatrix:\\n", matrix)
print("Row 1      :", matrix[1])
print("Col 2      :", matrix[:, 2])
print("Bool mask  :", matrix[matrix > 6])

# --- Broadcasting ---
row_vec = np.array([[10, 20, 30, 40]])   # shape (1, 4)
result = matrix + row_vec                 # (3,4) + (1,4) -> (3,4)
print("\\nBroadcast add:\\n", result)

# --- Vectorized vs Loop ---
size = 500_000
x = np.random.randn(size)

start = time.perf_counter()
loop_result = np.empty(size)
for i in range(size):
    loop_result[i] = x[i] ** 2 + 2 * x[i] + 1
loop_time = time.perf_counter() - start

start = time.perf_counter()
vec_result = x ** 2 + 2 * x + 1
vec_time = time.perf_counter() - start

print(f"\\nLoop time       : {loop_time:.4f}s")
print(f"Vectorized time : {vec_time:.4f}s")
print(f"Speedup         : {loop_time / vec_time:.0f}x")`,
      codeLanguage: "python",
    },
    {
      title: "Pandas Basics",
      slug: "pandas-basics",
      description: "Series, DataFrames, filtering, groupby, and data pipelines",
      markdownContent: `# Pandas Basics

Pandas provides the **DataFrame** — a labeled, column-oriented table — and the **Series** — a single labeled column. Together they make tabular data manipulation concise and expressive.

## Series and DataFrame

A Series is a one-dimensional array with an index. A DataFrame is a collection of Series sharing the same index (rows). You can create them from dictionaries, lists, NumPy arrays, or CSV files.

## Indexing and Filtering

Pandas supports label-based indexing (\`.loc\`), position-based indexing (\`.iloc\`), and boolean filtering. Boolean filters are the most common way to select rows in data analysis:

\`\`\`
df[df["column"] > threshold]
\`\`\`

## GroupBy Aggregation

The **split-apply-combine** pattern is central to data analysis. Given groups defined by a column, Pandas splits the data, applies an aggregation, and combines the results. For a column $x$ grouped by category $g$:

$$
\\bar{x}_g = \\frac{1}{n_g} \\sum_{i \\in g} x_i
$$

This lets you compute per-group statistics in a single readable expression.

The code below builds a small dataset, filters it, groups by category, and computes aggregated statistics.`,
      codeSnippet: `import numpy as np
import pandas as pd

# --- Create a DataFrame ---
np.random.seed(42)
n = 20
df = pd.DataFrame({
    "student": [f"S{i+1:02d}" for i in range(n)],
    "subject": np.random.choice(["Math", "Science", "English"], n),
    "score": np.random.randint(55, 100, n),
    "hours_studied": np.round(np.random.uniform(1, 10, n), 1),
})
print("First 5 rows:")
print(df.head(), "\\n")

# --- Filtering ---
high_scores = df[df["score"] >= 85]
print(f"Students scoring >= 85: {len(high_scores)} of {len(df)}")
print(high_scores[["student", "subject", "score"]].to_string(index=False), "\\n")

# --- GroupBy Aggregation ---
summary = (
    df.groupby("subject")
    .agg(
        count=("score", "size"),
        mean_score=("score", "mean"),
        max_score=("score", "max"),
        avg_hours=("hours_studied", "mean"),
    )
    .round(1)
)
print("Per-subject summary:")
print(summary, "\\n")

# --- Chained Pipeline ---
top_per_subject = (
    df.sort_values("score", ascending=False)
    .groupby("subject")
    .head(2)
    .sort_values(["subject", "score"], ascending=[True, False])
)
print("Top 2 students per subject:")
print(top_per_subject[["subject", "student", "score"]].to_string(index=False))`,
      codeLanguage: "python",
    },
    {
      title: "Data Visualization",
      slug: "data-visualization",
      description: "Matplotlib fundamentals for ML data exploration and presentation",
      markdownContent: `# Data Visualization

Effective visualization is essential for understanding data, diagnosing models, and communicating results. **Matplotlib** is Python's foundational plotting library — nearly every other visualization tool builds on it.

## Core Plot Types

- **Line plots** — track trends over time or training epochs
- **Scatter plots** — reveal relationships between two variables
- **Histograms** — show the distribution of a single variable

## Customization

Good charts need clear labels, titles, and legends. Matplotlib gives you full control:

\`\`\`python
ax.set_xlabel("x label")
ax.set_title("My Plot")
ax.legend()
\`\`\`

## A Key Formula: Correlation

When exploring scatter plots, the Pearson correlation coefficient tells you how linearly related two variables are:

$$
r = \\frac{\\sum_{i=1}^{n}(x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_{i=1}^{n}(x_i - \\bar{x})^2 \\;\\sum_{i=1}^{n}(y_i - \\bar{y})^2}}
$$

Values near $+1$ or $-1$ indicate strong linear relationships; $r \\approx 0$ suggests no linear trend.

The code below creates a multi-panel figure demonstrating the three core plot types, styled for clarity.`,
      codeSnippet: `import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(7)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# --- Panel 1: Line Plot (training curves) ---
epochs = np.arange(1, 51)
train_loss = 2.5 * np.exp(-0.08 * epochs) + 0.1 + np.random.normal(0, 0.03, 50)
val_loss = 2.5 * np.exp(-0.06 * epochs) + 0.25 + np.random.normal(0, 0.05, 50)
axes[0].plot(epochs, train_loss, label="Train Loss", color="#2563eb")
axes[0].plot(epochs, val_loss, label="Val Loss", color="#dc2626", linestyle="--")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Curves")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Panel 2: Scatter Plot (feature correlation) ---
x = np.random.randn(100)
y = 0.7 * x + np.random.randn(100) * 0.5
r = np.corrcoef(x, y)[0, 1]
axes[1].scatter(x, y, alpha=0.6, c="#7c3aed", edgecolors="white", s=40)
axes[1].set_xlabel("Feature A")
axes[1].set_ylabel("Feature B")
axes[1].set_title(f"Scatter (r = {r:.2f})")
axes[1].grid(True, alpha=0.3)

# --- Panel 3: Histogram (score distribution) ---
scores = np.concatenate([np.random.normal(65, 10, 200), np.random.normal(85, 5, 100)])
axes[2].hist(scores, bins=25, color="#059669", edgecolor="white", alpha=0.8)
axes[2].set_xlabel("Score")
axes[2].set_ylabel("Count")
axes[2].set_title("Score Distribution")
axes[2].axvline(scores.mean(), color="#dc2626", linestyle="--", label=f"Mean={scores.mean():.1f}")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ml_visualizations.png", dpi=100)
print("Figure saved: ml_visualizations.png")
print(f"Correlation coefficient: r = {r:.4f}")
print(f"Score stats: mean={scores.mean():.1f}, std={scores.std():.1f}, n={len(scores)}")`,
      codeLanguage: "python",
    },
  ],
};
