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
    { title: "Matrices", slug: "matrices", description: "Matrix operations and transformations" },
    { title: "Eigenvalues & Eigenvectors", slug: "eigenvalues-eigenvectors" },
  ],
};
