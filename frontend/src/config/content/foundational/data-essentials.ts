import type { Chapter } from "../types";

export const dataEssentials: Chapter = {
  title: "Data Essentials",
  slug: "data-essentials",
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
    {
      title: "SQL Basics",
      slug: "sql-basics",
      description: "Core SQL for data extraction: SELECT, JOINs, aggregations, and window functions",
      markdownContent: `# SQL Basics

**SQL** (Structured Query Language) is a declarative query language for relational databases. Unlike Python, where you describe *how* to compute something step by step, SQL describes *what* data you want — the database engine figures out the execution plan.

For data scientists, SQL is often the first step in any analysis: extracting and shaping raw data from warehouses (Postgres, BigQuery, Snowflake, Redshift) before it ever reaches a Pandas DataFrame.

## Core Statements

Every SQL query is built from a handful of clauses, evaluated in this logical order:

1. **FROM** — choose the table(s)
2. **WHERE** — filter individual rows
3. **GROUP BY** — collapse rows into groups
4. **HAVING** — filter groups (after aggregation)
5. **SELECT** — pick columns and compute expressions
6. **ORDER BY** — sort the result
7. **LIMIT** — cap the number of rows returned

\`\`\`sql
SELECT department, AVG(salary) AS avg_salary
FROM employees
WHERE hire_date >= '2020-01-01'
GROUP BY department
HAVING AVG(salary) > 60000
ORDER BY avg_salary DESC
LIMIT 10;
\`\`\`

## JOINs

JOINs combine rows from two tables based on a related column. The four main types:

- **INNER JOIN** — returns only rows with matches in *both* tables (intersection)
- **LEFT JOIN** — all rows from the left table, matched rows from the right (nulls where no match)
- **RIGHT JOIN** — all rows from the right table, matched rows from the left
- **FULL OUTER JOIN** — all rows from both tables (union), nulls where either side has no match

Visually, think of two overlapping circles (a Venn diagram):

\`\`\`
  Table A       Table B
  ┌─────┐     ┌─────┐
  │  A  ├──┬──┤  B  │
  │only │  │AB│ only│
  └─────┘  └──┘─────┘
  INNER  = AB only
  LEFT   = A + AB
  RIGHT  = AB + B
  FULL   = A + AB + B
\`\`\`

## Aggregations

Aggregate functions collapse many rows into a single summary value:

| Function | Description |
|----------|-------------|
| \`COUNT(*)\` | Number of rows |
| \`SUM(col)\` | Total of a numeric column |
| \`AVG(col)\` | Arithmetic mean |
| \`MIN(col)\` | Smallest value |
| \`MAX(col)\` | Largest value |

These are used with \`GROUP BY\` to compute per-group statistics — the SQL equivalent of Pandas \`groupby().agg()\`.

## Subqueries and CTEs

A **subquery** is a query nested inside another query. A **CTE** (Common Table Expression), introduced with the \`WITH\` clause, is a named subquery that makes complex queries more readable:

\`\`\`sql
WITH dept_stats AS (
    SELECT department, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY department
)
SELECT e.name, e.salary, d.avg_sal
FROM employees e
JOIN dept_stats d ON e.department = d.department
WHERE e.salary > d.avg_sal;
\`\`\`

CTEs are essential for breaking down multi-step transformations — think of each CTE as an intermediate DataFrame.

## Window Functions

Window functions compute values across a set of rows *related to the current row*, without collapsing the result. They use the \`OVER()\` clause:

| Function | Description |
|----------|-------------|
| \`ROW_NUMBER()\` | Sequential integer per partition |
| \`RANK()\` | Rank with gaps for ties |
| \`LAG(col, n)\` | Value from \`n\` rows before |
| \`LEAD(col, n)\` | Value from \`n\` rows after |
| \`SUM(col) OVER(...)\` | Running total |

\`\`\`sql
SELECT name, department, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
FROM employees;
\`\`\`

## Why Data Scientists Need SQL

Most real-world data lives in relational databases and data warehouses — not CSV files. Before any Python analysis begins, you need to:

1. **Extract** the right subset of data (often billions of rows — filtering in SQL is far more efficient than pulling everything into memory)
2. **Join** across multiple tables (users, orders, products)
3. **Aggregate** at the right granularity (daily totals, per-user metrics)

SQL handles all three at the database level, where computation is optimized with indexes, query planners, and distributed execution.

The code below uses Python's built-in \`sqlite3\` to create an in-memory database, populate it with sample data, and demonstrate SELECT with JOINs, GROUP BY, and window functions.`,
      codeSnippet: `import sqlite3

# --- Create in-memory database and populate tables ---
conn = sqlite3.connect(":memory:")
cur = conn.cursor()

cur.executescript("""
    CREATE TABLE departments (
        id   INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    );
    INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Marketing'), (3, 'Sales');

    CREATE TABLE employees (
        id         INTEGER PRIMARY KEY,
        name       TEXT NOT NULL,
        dept_id    INTEGER REFERENCES departments(id),
        salary     INTEGER NOT NULL,
        hire_date  TEXT NOT NULL
    );
    INSERT INTO employees VALUES
        (1, 'Alice',   1, 95000,  '2019-03-15'),
        (2, 'Bob',     1, 105000, '2020-07-01'),
        (3, 'Charlie', 2, 72000,  '2021-01-10'),
        (4, 'Diana',   2, 68000,  '2020-11-20'),
        (5, 'Eve',     3, 78000,  '2018-06-01'),
        (6, 'Frank',   3, 82000,  '2022-02-14'),
        (7, 'Grace',   1, 110000, '2017-09-01'),
        (8, 'Hank',    NULL, 60000, '2023-01-05');

    CREATE TABLE projects (
        id       INTEGER PRIMARY KEY,
        title    TEXT NOT NULL,
        lead_id  INTEGER REFERENCES employees(id)
    );
    INSERT INTO projects VALUES
        (1, 'ML Pipeline',    2),
        (2, 'Brand Refresh',  3),
        (3, 'Data Warehouse', 7),
        (4, 'Sales Dashboard', 5);
""")

def run_query(title, sql):
    """Execute a query and print results as a formatted table."""
    cur.execute(sql)
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(f"\\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))

# --- 1. INNER JOIN: employees with their department names ---
run_query("INNER JOIN — Employees with Departments", """
    SELECT e.name, d.name AS department, e.salary
    FROM employees e
    INNER JOIN departments d ON e.dept_id = d.id
    ORDER BY e.salary DESC
""")

# --- 2. LEFT JOIN: all employees, including those without a department ---
run_query("LEFT JOIN — All Employees (incl. no department)", """
    SELECT e.name, COALESCE(d.name, 'Unassigned') AS department, e.salary
    FROM employees e
    LEFT JOIN departments d ON e.dept_id = d.id
    ORDER BY e.name
""")

# --- 3. GROUP BY with aggregations ---
run_query("GROUP BY — Department Statistics", """
    SELECT d.name AS department,
           COUNT(*)     AS headcount,
           MIN(e.salary) AS min_salary,
           MAX(e.salary) AS max_salary,
           ROUND(AVG(e.salary)) AS avg_salary,
           SUM(e.salary) AS total_salary
    FROM employees e
    JOIN departments d ON e.dept_id = d.id
    GROUP BY d.name
    ORDER BY avg_salary DESC
""")

# --- 4. CTE + JOIN: employees earning above their department average ---
run_query("CTE — Employees Above Department Average", """
    WITH dept_avg AS (
        SELECT dept_id, ROUND(AVG(salary)) AS avg_sal
        FROM employees
        WHERE dept_id IS NOT NULL
        GROUP BY dept_id
    )
    SELECT e.name, d.name AS department, e.salary,
           da.avg_sal AS dept_avg,
           e.salary - da.avg_sal AS above_avg
    FROM employees e
    JOIN departments d ON e.dept_id = d.id
    JOIN dept_avg da ON e.dept_id = da.dept_id
    WHERE e.salary > da.avg_sal
    ORDER BY above_avg DESC
""")

# --- 5. Window functions: rank within department + running total ---
run_query("Window Functions — Rank & Running Total", """
    SELECT e.name, d.name AS department, e.salary,
           RANK() OVER (PARTITION BY d.name ORDER BY e.salary DESC) AS dept_rank,
           SUM(e.salary) OVER (ORDER BY e.salary DESC) AS running_total
    FROM employees e
    JOIN departments d ON e.dept_id = d.id
""")

conn.close()
print("\\nAll queries executed successfully.")`,
      codeLanguage: "python",
    },
  ],
};
