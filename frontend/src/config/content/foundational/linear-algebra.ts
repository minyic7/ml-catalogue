import type { Chapter } from "../types";

export const linearAlgebra: Chapter = {
  title: "Linear Algebra",
  slug: "linear-algebra",
  pages: [
    {
      title: "Vectors",
      slug: "vectors",
      description: "Vector operations and spaces",
      markdownContent: `# Vectors

A **vector** is an ordered list of numbers that represents a point or direction in space. In machine learning, vectors are the fundamental data structure — every input sample, weight, and gradient is a vector.

A vector $\\mathbf{v}$ in $\\mathbb{R}^n$ has $n$ components: $\\mathbf{v} = (v_1, v_2, \\ldots, v_n)$.

## Dot Product

The dot product of two vectors $\\mathbf{a}$ and $\\mathbf{b}$ measures their similarity:

$$
\\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i=1}^{n} a_i b_i = \\|\\mathbf{a}\\| \\|\\mathbf{b}\\| \\cos\\theta
$$

When $\\mathbf{a} \\cdot \\mathbf{b} = 0$, the vectors are **orthogonal** (perpendicular). This concept is central to projections, least-squares fitting, and many ML algorithms.

## Vector Norm

The Euclidean norm (or $L^2$ norm) gives the length of a vector:

$$
\\|\\mathbf{v}\\| = \\sqrt{\\sum_{i=1}^{n} v_i^2}
$$

Try the code below to see these operations in action with NumPy.`,
      codeSnippet: `import numpy as np

# Define two vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Basic operations
print("a + b =", a + b)
print("a * 3 =", a * 3)

# Dot product
dot = np.dot(a, b)
print("a · b =", dot)

# Euclidean norm
norm_a = np.linalg.norm(a)
print("||a|| =", round(norm_a, 4))

# Unit vector (normalization)
unit_a = a / norm_a
print("â =", np.round(unit_a, 4))`,
      codeLanguage: "python",
    },
    {
      title: "Matrices",
      slug: "matrices",
      description: "Matrix operations and transformations",
      markdownContent: `# Matrices

A **matrix** is a rectangular array of numbers arranged in rows and columns. In machine learning, matrices represent datasets (rows = samples, columns = features), weight layers in neural networks, and linear transformations.

A matrix $A \\in \\mathbb{R}^{m \\times n}$ has $m$ rows and $n$ columns.

## Basic Operations

**Transpose:** flipping a matrix over its diagonal swaps rows and columns. If $A$ is $m \\times n$, then $A^\\top$ is $n \\times m$, with $(A^\\top)_{ij} = A_{ji}$.

**Addition:** two matrices of the same shape are added element-wise: $(A + B)_{ij} = A_{ij} + B_{ij}$.

## Matrix Multiplication

Matrix multiplication is the workhorse of ML computation. For $A \\in \\mathbb{R}^{m \\times p}$ and $B \\in \\mathbb{R}^{p \\times n}$, the product $C = AB$ is an $m \\times n$ matrix where each entry is a dot product of a row of $A$ with a column of $B$:

$$
C_{ij} = \\sum_{k=1}^{p} A_{ik} B_{kj}
$$

Note that matrix multiplication is **not commutative** — in general $AB \\neq BA$. However, it is **associative**: $(AB)C = A(BC)$.

## Matrix-Vector Multiplication

Multiplying a matrix $A$ by a vector $\\mathbf{x}$ produces a new vector $\\mathbf{y} = A\\mathbf{x}$. This represents a **linear transformation** — it can rotate, scale, shear, or project the input vector. Every layer in a neural network performs exactly this operation (plus a bias and activation).

Run the snippet below to explore these operations with NumPy.`,
      codeSnippet: `import numpy as np

# Define matrices
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[7, 8], [9, 10]])

print("A (3x2):")
print(A)
print("\\nB (2x2):")
print(B)

# Transpose
print("\\nA transposed (2x3):")
print(A.T)

# Matrix multiplication: (3x2) @ (2x2) -> (3x2)
C = A @ B
print("\\nC = A @ B (3x2):")
print(C)

# Verify one entry: C[0,0] = 1*7 + 2*9 = 25
print("\\nC[0,0] =", C[0, 0], "(1*7 + 2*9 = 25)")

# Matrix-vector multiplication
x = np.array([1, 2])
y = A @ x
print("\\nA @ [1, 2] =", y)`,
      codeLanguage: "python",
    },
    {
      title: "Eigenvalues & Eigenvectors",
      slug: "eigenvalues-eigenvectors",
      description: "Eigendecomposition and its role in ML",
      markdownContent: `# Eigenvalues & Eigenvectors

An **eigenvector** of a square matrix $A$ is a non-zero vector $\\mathbf{v}$ that, when multiplied by $A$, only gets scaled — it doesn't change direction. The scaling factor $\\lambda$ is called the **eigenvalue**:

$$
A\\mathbf{v} = \\lambda\\mathbf{v}
$$

In other words, the matrix $A$ acts on $\\mathbf{v}$ like simple scalar multiplication. This makes eigenvectors the "natural axes" of the transformation that $A$ represents.

## Finding Eigenvalues

Eigenvalues are the roots of the **characteristic polynomial** $\\det(A - \\lambda I) = 0$. For a $2 \\times 2$ matrix $A = \\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}$, this gives:

$$
\\lambda^2 - (a + d)\\lambda + (ad - bc) = 0
$$

The sum of eigenvalues equals the **trace** (diagonal sum) and their product equals the **determinant**.

## Why Eigenvalues Matter in ML

Eigendecomposition is the mathematical engine behind several key ML techniques:

- **PCA (Principal Component Analysis):** the eigenvectors of the covariance matrix point in the directions of maximum variance. Projecting data onto the top-$k$ eigenvectors gives the best $k$-dimensional summary.
- **Spectral clustering:** uses eigenvectors of a graph Laplacian to find clusters.
- **Stability analysis:** eigenvalues of the Hessian reveal whether a critical point is a minimum, maximum, or saddle point — important for understanding optimizer behavior.

Run the snippet below to compute eigenvalues and see PCA in action.`,
      codeSnippet: `import numpy as np

# Symmetric matrix (e.g. a covariance matrix)
A = np.array([[4, 2],
              [2, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print("\\nEigenvalues:", eigenvalues)
print("\\nEigenvectors (columns):")
print(eigenvectors)

# Verify A @ v = lambda * v for the first eigenpair
v1 = eigenvectors[:, 0]
lam1 = eigenvalues[0]
print("\\n--- Verification ---")
print("A @ v1       =", A @ v1)
print("lambda1 * v1 =", lam1 * v1)

# Mini PCA: project random data onto principal axes
np.random.seed(42)
data = np.random.randn(100, 2) @ np.array([[2, 1], [1, 1.5]])
cov = np.cov(data, rowvar=False)
vals, vecs = np.linalg.eig(cov)
print("\\n--- Mini PCA ---")
print("Covariance eigenvalues:", np.round(vals, 3))
print("Variance explained:", np.round(vals / vals.sum() * 100, 1), "%")`,
      codeLanguage: "python",
    },
  ],
};
