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
    {
      title: "Anomaly Detection",
      slug: "anomaly-detection",
      description:
        "Isolation Forest, Local Outlier Factor, and One-Class SVM for identifying rare events",
      markdownContent: `# Anomaly Detection

Anomaly detection (also called **outlier detection**) is the task of identifying rare items, events, or observations that differ significantly from the majority of the data. Unlike supervised classification, anomaly detection typically works with very few (or zero) labeled anomalies — making it a natural fit for unsupervised learning.

## Use Cases

- **Fraud detection**: flagging unusual credit-card transactions
- **Network intrusion**: detecting abnormal traffic patterns
- **Manufacturing**: identifying defective products on an assembly line
- **Medical diagnosis**: spotting rare conditions in patient data

## Methods

### Statistical Approaches

Simple statistical tests such as **Z-score** and the **interquartile range (IQR)** flag points that fall far from the centre of the distribution. These are quick baselines, but assume unimodal or roughly Gaussian data.

For a feature $x$ with mean $\\mu$ and standard deviation $\\sigma$:

$$
z = \\frac{x - \\mu}{\\sigma}
$$

A point is typically flagged when $|z| > 3$. For the IQR method, a point is an outlier if it falls below $Q_1 - 1.5 \\cdot \\text{IQR}$ or above $Q_3 + 1.5 \\cdot \\text{IQR}$.

### Isolation Forest

Isolation Forest exploits the fact that anomalies are **few and different**. The algorithm builds an ensemble of random trees (isolation trees) that recursively split features at random values. Because anomalies sit in sparse regions, they are isolated in fewer splits:

$$
s(x, n) = 2^{-\\frac{E[h(x)]}{c(n)}}
$$

where $h(x)$ is the path length for point $x$, $E[h(x)]$ is the average over all trees, and $c(n)$ is the average path length in an unsuccessful BST search of $n$ items. Scores close to 1 indicate anomalies; scores near 0.5 indicate normal points.

The key hyperparameter is **contamination** — the expected proportion of anomalies in the dataset.

### Local Outlier Factor (LOF)

LOF is a **density-based** method. It compares the local density around a point with the densities around its $k$ nearest neighbours. The LOF score for a point $p$ is:

$$
\\text{LOF}_k(p) = \\frac{1}{|N_k(p)|} \\sum_{o \\in N_k(p)} \\frac{\\text{lrd}_k(o)}{\\text{lrd}_k(p)}
$$

where $\\text{lrd}_k(p)$ is the **local reachability density** of $p$. A LOF score $\\gg 1$ means the point is in a much sparser region than its neighbours — a likely anomaly.

### One-Class SVM

One-Class SVM learns a decision boundary that encloses the **normal** data in a high-dimensional feature space. It maximizes the margin between the origin and the mapped data points:

$$
\\min_{w, \\rho, \\xi} \\frac{1}{2} \\|w\\|^2 + \\frac{1}{\\nu m} \\sum_{i=1}^{m} \\xi_i - \\rho
$$

subject to $w \\cdot \\Phi(x^{(i)}) \\geq \\rho - \\xi_i$ and $\\xi_i \\geq 0$. The parameter $\\nu$ approximates the fraction of outliers.

## Evaluation

Because anomalies are rare, accuracy is misleading. Focus on:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | $\\frac{TP}{TP + FP}$ | Of predicted anomalies, how many are real? |
| **Recall** | $\\frac{TP}{TP + FN}$ | Of real anomalies, how many were caught? |
| **F1-Score** | $2 \\cdot \\frac{P \\cdot R}{P + R}$ | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the ROC curve | Ranking quality across all thresholds |

Run the code below to compare Isolation Forest and LOF on a synthetic dataset with injected anomalies.`,
      codeSnippet: `import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate normal data (2D Gaussian blobs)
n_normal = 300
X_normal = np.vstack([
    np.random.randn(n_normal // 2, 2) * 0.8 + [2, 2],
    np.random.randn(n_normal // 2, 2) * 0.6 + [-2, -2],
])

# Inject anomalies uniformly across feature space
n_anomalies = 30
X_anomalies = np.random.uniform(-6, 6, size=(n_anomalies, 2))

X = np.vstack([X_normal, X_anomalies])
y_true = np.array([1] * n_normal + [-1] * n_anomalies)  # 1=normal, -1=anomaly

# Isolation Forest
iso = IsolationForest(contamination=0.1, random_state=42)
y_iso = iso.fit_predict(X)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_lof = lof.fit_predict(X)

# Evaluation (anomaly = positive class -> map -1 to 1, 1 to 0)
y_true_bin = (y_true == -1).astype(int)
for name, y_pred in [("Isolation Forest", y_iso), ("LOF", y_lof)]:
    y_pred_bin = (y_pred == -1).astype(int)
    p = precision_score(y_true_bin, y_pred_bin)
    r = recall_score(y_true_bin, y_pred_bin)
    f = f1_score(y_true_bin, y_pred_bin)
    print(f"{name:18s}  Precision={p:.3f}  Recall={r:.3f}  F1={f:.3f}")

# Visualise
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for ax, y_pred, title in [(ax1, y_iso, "Isolation Forest"), (ax2, y_lof, "LOF")]:
    normal = y_pred == 1
    anomaly = y_pred == -1
    ax.scatter(X[normal, 0], X[normal, 1], s=15, alpha=0.6, label="Normal")
    ax.scatter(X[anomaly, 0], X[anomaly, 1], c="red", s=30, marker="x",
               label="Detected Anomaly")
    # Mark true anomalies with a ring
    ax.scatter(X_anomalies[:, 0], X_anomalies[:, 1], facecolors="none",
               edgecolors="green", s=80, linewidths=1.5, label="True Anomaly")
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("chart.png", dpi=100)
plt.show()
print("Chart saved.")`,
      codeLanguage: "python",
    },
    {
      title: "Recommender Systems",
      slug: "recommender-systems",
      description:
        "Collaborative filtering, content-based methods, and matrix factorisation for recommendations",
      markdownContent: `# Recommender Systems

Recommender systems predict a user's preference for items they haven't yet interacted with. They power product suggestions, movie recommendations, news feeds, and more. Although they can leverage supervised signals (ratings), the core problem — uncovering latent structure in a sparse user-item matrix — is fundamentally unsupervised.

## Collaborative Filtering

Collaborative filtering (CF) relies solely on **user-item interactions** (e.g., ratings, clicks) without needing any feature information about users or items.

### User-Based CF

Find users similar to the target user, then recommend items those similar users liked:

$$
\\hat{r}_{ui} = \\bar{r}_u + \\frac{\\sum_{v \\in N(u)} \\text{sim}(u, v) \\cdot (r_{vi} - \\bar{r}_v)}{\\sum_{v \\in N(u)} |\\text{sim}(u, v)|}
$$

where $N(u)$ is the neighbourhood of users most similar to $u$, and $\\text{sim}(u, v)$ is typically cosine similarity or Pearson correlation.

### Item-Based CF

Instead of finding similar users, find items similar to those the user already rated highly. This tends to be more scalable because item-item similarity is more stable over time than user-user similarity.

The intuition: *"Users who liked X also liked Y."*

## Content-Based Filtering

Content-based methods recommend items similar to what the user liked before, using **item features** (genre, description, price). A user profile is built from features of items they've rated, and new items are scored by similarity to that profile.

**Pros**: No cold-start for new items (features are known). **Cons**: Limited discovery — tends to recommend more of the same.

## Matrix Factorisation

The most powerful unsupervised approach decomposes the sparse **user-item rating matrix** $R \\in \\mathbb{R}^{m \\times n}$ into two low-rank matrices:

$$
R \\approx U V^T
$$

where $U \\in \\mathbb{R}^{m \\times k}$ captures $k$ latent user factors and $V \\in \\mathbb{R}^{n \\times k}$ captures $k$ latent item factors. Each predicted rating is:

$$
\\hat{r}_{ui} = \\mathbf{u}_u^T \\mathbf{v}_i = \\sum_{f=1}^{k} u_{uf} \\cdot v_{if}
$$

This can be solved via **Singular Value Decomposition (SVD)**:

$$
R = U \\Sigma V^T
$$

By keeping only the top $k$ singular values, we get the best rank-$k$ approximation (Eckart–Young theorem). The Netflix Prize (2006–2009) famously demonstrated that matrix factorisation methods outperform traditional CF.

## The Cold-Start Problem

| Scenario | Issue | Mitigation |
|----------|-------|------------|
| **New user** | No interaction history | Ask for initial preferences; use demographics |
| **New item** | No ratings received | Use content-based features; promote to random users |

## Hybrid Approaches

Modern systems combine collaborative and content-based signals. For example, Netflix uses matrix factorisation for its core engine but incorporates content features (genre, actors) and contextual signals (time of day, device) to handle cold-start and improve diversity.

## Evaluation Metrics

| Metric | Formula / Description |
|--------|-----------------------|
| **RMSE** | $\\sqrt{\\frac{1}{N}\\sum(r_{ui} - \\hat{r}_{ui})^2}$ — prediction accuracy |
| **Precision@K** | Fraction of top-$K$ recommendations that are relevant |
| **Recall@K** | Fraction of relevant items that appear in top-$K$ |
| **NDCG** | Normalized Discounted Cumulative Gain — rewards relevant items ranked higher |

Run the code below to build a simple collaborative filtering recommender using SVD on a small movie ratings dataset.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Small movie ratings dataset (0 = unrated)
movies = ["Inception", "Titanic", "Toy Story", "Matrix", "Frozen", "Godfather"]
users = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]

R = np.array([
    [5, 3, 4, 5, 1, 4],
    [4, 0, 5, 4, 0, 3],
    [0, 4, 0, 2, 5, 0],
    [5, 0, 4, 5, 2, 5],
    [1, 5, 2, 0, 5, 1],
    [4, 2, 5, 4, 0, 0],
    [0, 5, 1, 0, 4, 2],
    [5, 1, 4, 5, 1, 5],
], dtype=float)

# Store which entries are observed
mask = R > 0

# Fill missing values with row (user) means for SVD
user_means = np.where(mask.sum(axis=1, keepdims=True) > 0,
                      (R * mask).sum(axis=1, keepdims=True) / mask.sum(axis=1, keepdims=True),
                      0)
R_filled = np.where(mask, R, user_means)

# Center ratings by user mean
R_centered = R_filled - user_means

# SVD factorisation
U, sigma, Vt = np.linalg.svd(R_centered, full_matrices=False)

# Keep top k=3 latent factors
k = 3
U_k = U[:, :k]
S_k = np.diag(sigma[:k])
Vt_k = Vt[:k, :]

# Reconstruct predicted ratings
R_pred = user_means + U_k @ S_k @ Vt_k

# Compute RMSE on observed entries only
observed_errors = (R[mask] - R_pred[mask])
rmse = np.sqrt(np.mean(observed_errors ** 2))
print(f"RMSE on observed ratings (k={k}): {rmse:.4f}\\n")

# Show predictions for users with missing ratings
print("Predicted ratings for unrated movies:")
for i in range(len(users)):
    for j in range(len(movies)):
        if not mask[i, j]:
            print(f"  {users[i]:6s} -> {movies[j]:10s}: {R_pred[i,j]:.2f}")

# Visualise actual vs predicted for all observed ratings
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

observed_actual = R[mask]
observed_pred = R_pred[mask]
ax1.scatter(observed_actual, observed_pred, alpha=0.7, edgecolors="black", s=50)
ax1.plot([0, 5.5], [0, 5.5], "r--", label="Perfect prediction")
ax1.set_xlabel("Actual Rating")
ax1.set_ylabel("Predicted Rating")
ax1.set_title(f"Actual vs Predicted (RMSE={rmse:.3f})")
ax1.legend()
ax1.set_xlim(0, 5.5)
ax1.set_ylim(0, 5.5)

# Heatmap of full predicted rating matrix
im = ax2.imshow(R_pred, cmap="YlOrRd", aspect="auto", vmin=0, vmax=5)
ax2.set_xticks(range(len(movies)))
ax2.set_xticklabels(movies, rotation=45, ha="right")
ax2.set_yticks(range(len(users)))
ax2.set_yticklabels(users)
ax2.set_title("Predicted Rating Matrix (SVD k=3)")
# Mark unrated cells with a dot
for i in range(len(users)):
    for j in range(len(movies)):
        if not mask[i, j]:
            ax2.plot(j, i, "ko", markersize=6)
        ax2.text(j, i, f"{R_pred[i,j]:.1f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax2, shrink=0.8)

plt.tight_layout()
plt.savefig("chart.png", dpi=100)
plt.show()
print("\\nChart saved. Black dots = originally unrated (predicted by SVD).")`,
      codeLanguage: "python",
    },
  ],
};
