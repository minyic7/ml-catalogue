import type { Chapter } from "../types";

export const unsupervisedLearning: Chapter = {
  title: "Unsupervised Learning",
  slug: "unsupervised-learning",
  pages: [
    {
      title: "Clustering",
      slug: "clustering",
      description: "K-Means clustering and the elbow method",
      markdownContent: `# Clustering

Clustering is the task of grouping data points so that points within the same group (or **cluster**) are more similar to each other than to those in other groups — all without any labeled examples to guide the process.

## K-Means Algorithm

K-Means is the most widely used clustering algorithm. It partitions $m$ data points into $K$ clusters by iterating two steps:

1. **Assign** each point $x^{(i)}$ to the nearest centroid $\\mu_k$.
2. **Update** each centroid to the mean of its assigned points.

The algorithm minimizes the **within-cluster sum of squares** (inertia):

$$
J = \\sum_{i=1}^{m} \\|x^{(i)} - \\mu_{c^{(i)}}\\|^2
$$

where $c^{(i)}$ is the cluster index assigned to point $x^{(i)}$ and $\\mu_{c^{(i)}}$ is the centroid of that cluster.

K-Means converges when the assignments no longer change. Because it uses random initialization, the result can vary between runs — a common mitigation is to run the algorithm multiple times and keep the solution with the lowest $J$.

## Choosing K — The Elbow Method

Picking the right number of clusters is crucial. The **elbow method** plots inertia against $K$ and looks for the "elbow" — the value of $K$ where adding another cluster yields diminishing returns:

$$
\\text{Elbow} = \\arg\\min_K \\left\\{ K \\;\\middle|\\; \\frac{J_{K} - J_{K+1}}{J_{K-1} - J_{K}} < \\tau \\right\\}
$$

In practice, you simply look for the bend in the curve rather than computing a formal threshold.

## Key Considerations

- **Scaling matters**: K-Means uses Euclidean distance, so features on different scales can dominate. Always standardize your data first.
- **Initialization**: The \`k-means++\` initialization (default in scikit-learn) spreads initial centroids apart, producing more reliable results.
- **Limitations**: K-Means assumes spherical, equally-sized clusters. For non-convex shapes, consider DBSCAN or Gaussian Mixture Models.

Run the code below to see K-Means in action on synthetic blob data.`,
      codeSnippet: `import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate synthetic blob data with 4 centers
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# Fit K-Means with K=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)

print("Cluster centers:")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"  Cluster {i}: ({center[0]:.2f}, {center[1]:.2f})")
print(f"\\nInertia (J): {kmeans.inertia_:.2f}")

# Elbow method: try K = 1..8
inertias = []
K_range = range(1, 9)
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append(model.inertia_)

# Plot clusters and elbow curve side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis", s=20, alpha=0.7)
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="X", s=200, edgecolors="black", label="Centroids")
ax1.set_title("K-Means Clustering (K=4)")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.legend()

ax2.plot(list(K_range), inertias, "bo-")
ax2.set_title("Elbow Method")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Inertia")
plt.tight_layout()
plt.savefig("chart.png", dpi=100)
plt.show()
print("Chart saved.")`,
      codeLanguage: "python",
    },
    {
      title: "Dimensionality Reduction",
      slug: "dimensionality-reduction",
      description: "PCA and explained variance for reducing feature dimensions",
      markdownContent: `# Dimensionality Reduction

High-dimensional data is common in machine learning — images, text embeddings, and genomic data can have thousands of features. **Dimensionality reduction** projects data into a lower-dimensional space while preserving as much information as possible, aiding visualization, speeding up training, and reducing noise.

## Principal Component Analysis (PCA)

PCA finds the directions (principal components) along which the data varies the most. Given a centered data matrix $X$ with $m$ samples and $n$ features, PCA works by eigendecomposing the **covariance matrix**:

$$
\\Sigma = \\frac{1}{m} X^T X
$$

The eigenvectors of $\\Sigma$ are the principal component directions, and the eigenvalues $\\lambda_1 \\geq \\lambda_2 \\geq \\cdots \\geq \\lambda_n$ tell us how much variance each component captures.

## Explained Variance Ratio

To decide how many components to keep, we look at the **explained variance ratio** for each component $k$:

$$
\\text{EVR}_k = \\frac{\\lambda_k}{\\sum_{j=1}^{n} \\lambda_j}
$$

The cumulative explained variance tells us what fraction of total information is retained. A common rule of thumb is to keep enough components to explain 90–95% of the variance.

## How PCA Works Step by Step

1. **Center** the data by subtracting the mean of each feature.
2. **Compute** the covariance matrix $\\Sigma$.
3. **Find** the eigenvalues and eigenvectors of $\\Sigma$.
4. **Sort** eigenvectors by decreasing eigenvalue.
5. **Project** the data onto the top $k$ eigenvectors to get a $k$-dimensional representation.

## Why It Matters for ML

- **Curse of dimensionality**: Many algorithms degrade with too many features. PCA reduces features while keeping the signal.
- **Visualization**: Projecting to 2D or 3D lets you visually inspect cluster structure and separability.
- **Noise reduction**: Dropping low-variance components removes noise that can hurt generalization.

Run the code below to apply PCA to a 4D dataset and visualize the 2D projection.`,
      codeSnippet: `import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate a 4-feature dataset with 3 informative features
X, y = make_classification(n_samples=250, n_features=4, n_informative=3,
                           n_redundant=1, n_classes=3, n_clusters_per_class=1,
                           random_state=42)

# Standardize features before PCA
X_scaled = StandardScaler().fit_transform(X)

# Fit PCA keeping all components to inspect variance
pca_full = PCA().fit(X_scaled)
evr = pca_full.explained_variance_ratio_

print("Explained Variance Ratios:")
for i, ratio in enumerate(evr):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.1f}%)")
print(f"\\nCumulative (first 2): {sum(evr[:2])*100:.1f}%")

# Project onto 2 components
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

# Plot 2D projection and cumulative variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="Set1", s=25, alpha=0.8)
ax1.set_title("PCA — 2D Projection")
ax1.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
ax1.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
ax1.legend(*scatter.legend_elements(), title="Class")

cumulative = np.cumsum(evr)
ax2.bar(range(1, len(evr)+1), evr, alpha=0.6, label="Individual")
ax2.step(range(1, len(evr)+1), cumulative, where="mid", color="red", label="Cumulative")
ax2.axhline(y=0.95, color="gray", linestyle="--", label="95% threshold")
ax2.set_title("Explained Variance")
ax2.set_xlabel("Principal Component")
ax2.set_ylabel("Variance Ratio")
ax2.set_xticks(range(1, len(evr)+1))
ax2.legend()
plt.tight_layout()
plt.savefig("chart.png", dpi=100)
plt.show()
print("Chart saved.")`,
      codeLanguage: "python",
    },
  ],
};
