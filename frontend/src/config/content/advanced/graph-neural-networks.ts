import type { Chapter } from "../types";

export const graphNeuralNetworks: Chapter = {
  title: "Graph Neural Networks",
  slug: "graph-neural-networks",
  pages: [
    {
      title: "Graph Basics & Representations",
      slug: "graph-basics-representations",
      description:
        "Adjacency matrices, node/edge features, graph types, and graph visualization",
      isDeepLearning: true,
      markdownContent: `# Graph Basics & Representations

Many real-world datasets are naturally structured as **graphs**: social networks, molecules, citation networks, and knowledge bases. Unlike images (grids) or text (sequences), graphs have irregular topology — each node can have a different number of neighbours.

## Graphs Formally

A graph $G = (V, E)$ consists of a set of **nodes** $V$ (also called vertices) and a set of **edges** $E \\subseteq V \\times V$. The number of nodes is $n = |V|$ and the number of edges is $m = |E|$.

## Adjacency Matrix

The structure of a graph is captured by its **adjacency matrix** $A \\in \\{0,1\\}^{n \\times n}$, where:

$$
A_{ij} = \\begin{cases} 1 & \\text{if } (i, j) \\in E \\\\ 0 & \\text{otherwise} \\end{cases}
$$

For **undirected** graphs, $A$ is symmetric: $A_{ij} = A_{ji}$. For **weighted** graphs, entries can take real values representing edge weights.

## Degree Matrix

The **degree matrix** $D$ is a diagonal matrix where each entry counts the number of connections for a node:

$$
D_{ii} = \\sum_{j=1}^{n} A_{ij}
$$

The degree matrix plays a key role in normalising graph convolutions, as we will see in later pages.

## Node Feature Matrix

Each node can carry a feature vector. Stacking all node features gives the **node feature matrix** $X \\in \\mathbb{R}^{n \\times d}$, where $d$ is the feature dimension. For example, in a citation network each node (paper) might have a bag-of-words feature vector.

## Graph Laplacian

The **graph Laplacian** is defined as:

$$
L = D - A
$$

The **normalised Laplacian** is:

$$
L_{\\text{norm}} = I - D^{-1/2} A D^{-1/2}
$$

Its eigenvalues lie in $[0, 2]$ and its eigenvectors form a basis for signals on the graph — the foundation of **spectral graph theory**.

## Common Graph Types

| Type | Description |
|------|-------------|
| **Undirected** | Edges have no direction; $A$ is symmetric |
| **Directed** | Edges have direction; $A$ may be asymmetric |
| **Weighted** | Edges carry real-valued weights |
| **Bipartite** | Nodes split into two disjoint sets; edges only between sets |

Run the code to build the Zachary Karate Club graph, visualise its adjacency matrix as a heatmap, and display the graph using a spring layout.`,
      codeSnippet: `from ml_catalogue_runtime import MODE
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Zachary Karate Club adjacency list ---
# 34 nodes (members), edges represent social ties
edges = [
    (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),
    (0,12),(0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),
    (1,17),(1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),
    (2,28),(2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),
    (6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),
    (15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),
    (23,25),(23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),
    (25,31),(26,29),(26,33),(27,33),(28,31),(28,33),(29,32),(29,33),
    (30,32),(30,33),(31,32),(31,33),(32,33),
]

n_nodes = 34
A = np.zeros((n_nodes, n_nodes))
for i, j in edges:
    A[i, j] = 1.0
    A[j, i] = 1.0

# Ground-truth community labels (two factions)
labels = np.array([
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1
])

print("=== Zachary Karate Club Graph ===")
print(f"Nodes: {n_nodes}")
print(f"Edges: {len(edges)}")
D = np.diag(A.sum(axis=1))
print(f"Degree range: {int(A.sum(axis=1).min())} - {int(A.sum(axis=1).max())}")

# --- Compute graph Laplacian ---
L = D - A
eigenvalues = np.linalg.eigvalsh(L)
print(f"\\nGraph Laplacian smallest eigenvalues: {eigenvalues[:4].round(4)}")
print(f"(Second-smallest eigenvalue = algebraic connectivity = {eigenvalues[1]:.4f})")

# --- Spring layout (Fruchterman-Reingold) ---
np.random.seed(42)
pos = np.random.randn(n_nodes, 2) * 0.5
n_iter = 200 if MODE == "full" else 80
k = 1.0  # optimal distance
temp = 1.0

for iteration in range(n_iter):
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 2)
    dist = np.sqrt((delta ** 2).sum(axis=2) + 1e-6)

    # Repulsive forces (all pairs)
    rep_force = (k ** 2 / dist)[:, :, np.newaxis] * (delta / dist[:, :, np.newaxis])
    np.fill_diagonal(rep_force[:, :, 0], 0)
    np.fill_diagonal(rep_force[:, :, 1], 0)
    displacement = rep_force.sum(axis=1)

    # Attractive forces (edges only)
    for i, j in edges:
        d = pos[i] - pos[j]
        d_norm = np.sqrt((d ** 2).sum() + 1e-6)
        attract = d * d_norm / k
        displacement[i] -= attract
        displacement[j] += attract

    # Limit displacement by temperature
    disp_norm = np.sqrt((displacement ** 2).sum(axis=1, keepdims=True) + 1e-6)
    pos += displacement / disp_norm * min(temp, disp_norm).clip(max=temp)
    temp *= 0.95

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Adjacency matrix heatmap
ax = axes[0]
im = ax.imshow(A, cmap="Blues", interpolation="nearest")
ax.set_title("Adjacency Matrix Heatmap")
ax.set_xlabel("Node")
ax.set_ylabel("Node")
fig.colorbar(im, ax=ax, shrink=0.8)

# Graph visualisation
ax = axes[1]
colors = ["#4A90D9" if l == 0 else "#E74C3C" for l in labels]
for i, j in edges:
    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], "gray", alpha=0.3, linewidth=0.7)
ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=120, edgecolors="k", linewidths=0.8, zorder=5)
for idx in range(n_nodes):
    ax.annotate(str(idx), pos[idx], fontsize=6, ha="center", va="center", zorder=6)
ax.set_title("Karate Club Graph (Spring Layout)")
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
plt.savefig("output.png", dpi=100)
plt.show()
print("Plot saved to output.png")`,
      codeLanguage: "python",
    },
    {
      title: "Graph Convolutional Networks (GCN)",
      slug: "graph-convolutional-networks",
      description:
        "Spectral vs spatial convolutions, GCN layer math, and node classification on the Karate Club graph",
      isDeepLearning: true,
      markdownContent: `# Graph Convolutional Networks (GCN)

**Graph Convolutional Networks** extend the idea of convolution from regular grids (images) to irregular graph structures. The key insight is that a node's representation should be informed by its neighbours — just as a pixel in a CNN is influenced by its local patch.

## Spectral vs Spatial Convolutions

There are two families of graph convolutions:

**Spectral approaches** define convolution via the graph Fourier transform. Given the normalised Laplacian $L_{\\text{norm}} = U \\Lambda U^T$ (eigendecomposition), a spectral filter $g_\\theta$ acts on a signal $x$ as:

$$
g_\\theta \\star x = U\\, g_\\theta(\\Lambda)\\, U^T x
$$

This is computationally expensive ($O(n^2)$ for the eigenvector multiply). **ChebNet** approximates the filter using Chebyshev polynomials of $\\Lambda$, bringing the cost down to $O(m)$.

**Spatial approaches** define convolution directly in the node domain by aggregating features from each node's neighbourhood. This is more intuitive and scales better.

## The GCN Layer (Kipf & Welling, 2017)

The most widely used graph convolution is a first-order approximation of ChebNet. A single GCN layer computes:

$$
H^{(l+1)} = \\sigma\\!\\left(\\tilde{D}^{-1/2}\\, \\tilde{A}\\, \\tilde{D}^{-1/2}\\, H^{(l)}\\, W^{(l)}\\right)
$$

where:
- $\\tilde{A} = A + I_n$ is the adjacency matrix with **self-loops** added
- $\\tilde{D}_{ii} = \\sum_j \\tilde{A}_{ij}$ is the degree matrix of $\\tilde{A}$
- $H^{(l)} \\in \\mathbb{R}^{n \\times d_l}$ is the node feature matrix at layer $l$ (with $H^{(0)} = X$)
- $W^{(l)} \\in \\mathbb{R}^{d_l \\times d_{l+1}}$ is the learnable weight matrix
- $\\sigma$ is a nonlinear activation (typically ReLU)

## Intuition: Normalised Neighbourhood Aggregation

Expanding the formula for a single node $i$:

$$
h_i^{(l+1)} = \\sigma\\!\\left(\\sum_{j \\in \\mathcal{N}(i) \\cup \\{i\\}} \\frac{1}{\\sqrt{\\tilde{d}_i\\, \\tilde{d}_j}}\\, h_j^{(l)}\\, W^{(l)}\\right)
$$

Each neighbour $j$'s features are weighted by $1/\\sqrt{\\tilde{d}_i \\tilde{d}_j}$ — this symmetric normalisation prevents nodes with many connections from dominating. The self-loop ($j = i$) ensures a node retains its own features.

## Stacking Layers

Each GCN layer lets information flow one hop. With $k$ layers, each node's representation incorporates information from its **$k$-hop neighbourhood**. In practice, 2–3 layers work best; deeper GCNs suffer from **over-smoothing**, where all node representations converge.

## Why GCN Works

The symmetric normalisation $\\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2}$ is equivalent to averaging neighbour features weighted by inverse degree — a form of **Laplacian smoothing**. This is a low-pass filter on the graph that encourages connected nodes to have similar representations, which is exactly what we want for tasks like node classification.

Run the code to train a 2-layer GCN from scratch on the Karate Club dataset and visualise the learned node embeddings.`,
      codeSnippet: `from ml_catalogue_runtime import MODE, DEVICE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device(DEVICE)

# --- Build Karate Club graph ---
edges = [
    (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),
    (0,12),(0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),
    (1,17),(1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),
    (2,28),(2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),
    (6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),
    (15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),
    (23,25),(23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),
    (25,31),(26,29),(26,33),(27,33),(28,31),(28,33),(29,32),(29,33),
    (30,32),(30,33),(31,32),(31,33),(32,33),
]
n_nodes = 34
labels = torch.tensor([
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1
], dtype=torch.long).to(device)

# Build adjacency matrix with self-loops: A_tilde = A + I
A = np.zeros((n_nodes, n_nodes))
for i, j in edges:
    A[i, j] = 1.0
    A[j, i] = 1.0
A_tilde = A + np.eye(n_nodes)

# Symmetric normalisation: D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1)))
A_hat = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
A_hat = torch.tensor(A_hat, dtype=torch.float32).to(device)

# Node features: identity matrix (one-hot encoding of node id)
X = torch.eye(n_nodes, dtype=torch.float32).to(device)


class GCNLayer(nn.Module):
    """A single Graph Convolutional layer."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, A_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # H' = A_hat @ H @ W + b
        return A_hat @ H @ self.weight + self.bias


class GCN(nn.Module):
    """2-layer GCN for node classification."""
    def __init__(self, n_input: int, n_hidden: int, n_classes: int):
        super().__init__()
        self.gcn1 = GCNLayer(n_input, n_hidden)
        self.gcn2 = GCNLayer(n_hidden, n_classes)

    def forward(self, A_hat: torch.Tensor, X: torch.Tensor):
        H = F.relu(self.gcn1(A_hat, X))
        H = self.gcn2(A_hat, H)
        return H  # raw logits


model = GCN(n_input=n_nodes, n_hidden=16, n_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Semi-supervised: only a few labelled nodes (node 0 and node 33 are leaders)
train_mask = torch.zeros(n_nodes, dtype=torch.bool)
train_mask[0] = True   # instructor
train_mask[33] = True   # admin
train_mask[1] = True
train_mask[32] = True

n_epochs = 300 if MODE == "full" else 100
print("=== Training 2-Layer GCN on Karate Club ===")
print(f"Mode: {MODE} | Epochs: {n_epochs} | Train nodes: {train_mask.sum().item()}")

for epoch in range(n_epochs):
    model.train()
    logits = model(A_hat, X)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % (50 if MODE == "full" else 20) == 0:
        model.eval()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean()
        print(f"Epoch {epoch+1:>3d}  Loss: {loss.item():.4f}  Accuracy: {acc.item():.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    logits = model(A_hat, X)
    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean()
    embeddings = F.relu(model.gcn1(A_hat, X)).cpu().numpy()

print(f"\\nFinal accuracy: {acc.item():.4f}")
print(f"Predictions: {pred.cpu().tolist()}")
print(f"Ground truth: {labels.cpu().tolist()}")

# --- Visualise node embeddings with t-SNE ---
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=8)
emb_2d = tsne.fit_transform(embeddings)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predicted labels
ax = axes[0]
colors_pred = ["#4A90D9" if p == 0 else "#E74C3C" for p in pred.cpu().numpy()]
ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors_pred, s=100, edgecolors="k", linewidths=0.8)
for idx in range(n_nodes):
    ax.annotate(str(idx), emb_2d[idx], fontsize=7, ha="center", va="center")
ax.set_title("GCN Embeddings (Predicted Labels)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")

# Ground-truth labels
ax = axes[1]
colors_gt = ["#4A90D9" if l == 0 else "#E74C3C" for l in labels.cpu().numpy()]
ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors_gt, s=100, edgecolors="k", linewidths=0.8)
for idx in range(n_nodes):
    ax.annotate(str(idx), emb_2d[idx], fontsize=7, ha="center", va="center")
ax.set_title("GCN Embeddings (Ground Truth)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")

plt.tight_layout()
plt.savefig("output.png", dpi=100)
plt.show()
print("Plot saved to output.png")`,
      codeLanguage: "python",
    },
    {
      title: "GraphSAGE",
      slug: "graphsage",
      description:
        "Sampling and aggregating neighbours, mean/LSTM/pool aggregators, and inductive learning",
      isDeepLearning: true,
      markdownContent: `# GraphSAGE

**GraphSAGE** (Graph SAmple and aggreGatE) addresses a key limitation of GCN: the need to operate on the **entire graph** at once. By **sampling** a fixed number of neighbours and **aggregating** their features, GraphSAGE can generate embeddings for unseen nodes — making it the first major **inductive** graph learning method.

## Transductive vs Inductive Learning

Standard GCN is **transductive**: it trains on a fixed graph and cannot generalise to new nodes without retraining. GraphSAGE is **inductive**: it learns a function that *generates* embeddings by sampling and aggregating from a node's local neighbourhood. This function transfers to new nodes (or entirely new graphs) at inference time.

## The GraphSAGE Algorithm

For each node $v$, at each layer $k$:

1. **Sample** a fixed-size set of neighbours $\\mathcal{N}_S(v) \\subseteq \\mathcal{N}(v)$
2. **Aggregate** their features: $h_{\\mathcal{N}(v)}^{(k)} = \\text{AGGREGATE}_k\\!\\left(\\{h_u^{(k-1)} : u \\in \\mathcal{N}_S(v)\\}\\right)$
3. **Combine** with the node's own features:

$$
h_v^{(k)} = \\sigma\\!\\left(W^{(k)} \\cdot \\text{CONCAT}\\!\\left(h_v^{(k-1)},\\; h_{\\mathcal{N}(v)}^{(k)}\\right)\\right)
$$

4. **Normalise**: $h_v^{(k)} = \\frac{h_v^{(k)}}{\\|h_v^{(k)}\\|_2}$

## Aggregator Functions

The choice of aggregator defines how neighbour information is combined:

### Mean Aggregator

$$
h_{\\mathcal{N}(v)}^{(k)} = \\frac{1}{|\\mathcal{N}_S(v)|} \\sum_{u \\in \\mathcal{N}_S(v)} h_u^{(k-1)}
$$

Simple element-wise mean of neighbour features. This is equivalent to the GCN propagation rule when you also average in the node's own features.

### LSTM Aggregator

Applies an LSTM to a random permutation of neighbour features:

$$
h_{\\mathcal{N}(v)}^{(k)} = \\text{LSTM}\\!\\left(\\text{PERMUTE}\\!\\left(\\{h_u^{(k-1)} : u \\in \\mathcal{N}_S(v)\\}\\right)\\right)
$$

The LSTM can capture more complex patterns than a simple mean, but introduces ordering sensitivity (addressed by random permutation).

### Pool Aggregator

Each neighbour's features are transformed by a fully-connected layer, then max-pooled:

$$
h_{\\mathcal{N}(v)}^{(k)} = \\max\\!\\left(\\{\\sigma(W_{\\text{pool}}\\, h_u^{(k-1)} + b) : u \\in \\mathcal{N}_S(v)\\}\\right)
$$

Max-pooling captures the most salient features across the neighbourhood.

## Why Sampling Matters

Sampling a fixed number of neighbours ($S$ per layer) bounds the computation. With $K$ layers and sample size $S$, each node's receptive field is at most $S^K$ nodes — regardless of the actual graph size. This makes GraphSAGE scalable to graphs with millions of nodes.

## Training Objective

GraphSAGE can be trained with:
- **Supervised loss** (e.g., cross-entropy for classification)
- **Unsupervised loss** encouraging nearby nodes to have similar embeddings:

$$
J(z_v) = -\\log\\!\\left(\\sigma(z_v^T z_u)\\right) - Q \\cdot \\mathbb{E}_{v_n \\sim P_n} \\log\\!\\left(\\sigma(-z_v^T z_{v_n})\\right)
$$

where $u$ is a neighbour, $v_n$ is a negative sample, and $Q$ is the number of negative samples.

Run the code to implement GraphSAGE with mean and pool aggregators on a synthetic community graph and compare their node classification performance.`,
      codeSnippet: `from ml_catalogue_runtime import MODE, DEVICE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device(DEVICE)

# --- Generate synthetic community graph ---
np.random.seed(42)
torch.manual_seed(42)

n_communities = 3
nodes_per_community = 30
n_nodes = n_communities * nodes_per_community
labels = np.repeat(np.arange(n_communities), nodes_per_community)

# Stochastic block model: high intra-community, low inter-community edges
p_intra = 0.3
p_inter = 0.02
A = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        p = p_intra if labels[i] == labels[j] else p_inter
        if np.random.rand() < p:
            A[i, j] = 1.0
            A[j, i] = 1.0

# Build adjacency list for fast neighbour lookup
adj_list = {i: list(np.where(A[i] > 0)[0]) for i in range(n_nodes)}
# Ensure every node has at least a self-connection for sampling
for i in range(n_nodes):
    if len(adj_list[i]) == 0:
        adj_list[i] = [i]

# Node features: random features + community signal
X_np = np.random.randn(n_nodes, 8).astype(np.float32)
for c in range(n_communities):
    mask = labels == c
    X_np[mask, :3] += np.array([c * 2, -c, c * 0.5])

X = torch.tensor(X_np).to(device)
y = torch.tensor(labels, dtype=torch.long).to(device)

print(f"=== Synthetic Community Graph ===")
print(f"Nodes: {n_nodes} | Communities: {n_communities}")
print(f"Edges: {int(A.sum()) // 2} | Avg degree: {A.sum(axis=1).mean():.1f}")


def sample_neighbours(adj_list, nodes, n_samples):
    """Sample fixed number of neighbours for a batch of nodes."""
    sampled = []
    for node in nodes:
        neighbours = adj_list[node]
        if len(neighbours) >= n_samples:
            idx = np.random.choice(len(neighbours), n_samples, replace=False)
        else:
            idx = np.random.choice(len(neighbours), n_samples, replace=True)
        sampled.append([neighbours[i] for i in idx])
    return sampled


class MeanAggregator(nn.Module):
    """Mean aggregator: average neighbour features."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, self_feats, neigh_feats):
        # neigh_feats: (batch, n_samples, feat_dim) -> mean over samples
        neigh_mean = neigh_feats.mean(dim=1)
        combined = torch.cat([self_feats, neigh_mean], dim=1)
        return F.relu(self.linear(combined))


class PoolAggregator(nn.Module):
    """Pool aggregator: transform + max-pool neighbour features."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.neigh_linear = nn.Linear(in_features, in_features)
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, self_feats, neigh_feats):
        # Transform neighbours then max-pool
        neigh_transformed = F.relu(self.neigh_linear(neigh_feats))
        neigh_pool = neigh_transformed.max(dim=1).values
        combined = torch.cat([self_feats, neigh_pool], dim=1)
        return F.relu(self.linear(combined))


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE model."""
    def __init__(self, in_features, hidden_dim, n_classes, aggregator_type="mean"):
        super().__init__()
        AggClass = MeanAggregator if aggregator_type == "mean" else PoolAggregator
        self.agg1 = AggClass(in_features, hidden_dim)
        self.agg2 = AggClass(hidden_dim, n_classes)

    def forward(self, X, nodes, adj_list, n_samples=10):
        # Layer 1: aggregate from 2-hop neighbours to get 1-hop representations
        neigh_1hop = sample_neighbours(adj_list, nodes, n_samples)
        all_1hop = list(set(n for ns in neigh_1hop for n in ns))
        neigh_2hop = sample_neighbours(adj_list, all_1hop, n_samples)

        # Compute 1-hop node features via aggregation from 2-hop
        h_1hop = {}
        for idx, node in enumerate(all_1hop):
            n_ids = neigh_2hop[idx]
            neigh_feats = X[n_ids].unsqueeze(0)
            self_feat = X[node].unsqueeze(0)
            h_1hop[node] = self.agg1(self_feat, neigh_feats).squeeze(0)

        # Layer 2: aggregate from 1-hop representations
        out = []
        for idx, node in enumerate(nodes):
            n_ids = neigh_1hop[idx]
            neigh_feats = torch.stack([h_1hop[n] for n in n_ids]).unsqueeze(0)
            self_feat = h_1hop.get(node, self.agg1(X[node].unsqueeze(0), X[[node]].unsqueeze(0)).squeeze(0)).unsqueeze(0)
            h = self.agg2(self_feat, neigh_feats)
            out.append(h.squeeze(0))

        return torch.stack(out)


# --- Train and compare aggregators ---
n_epochs = 200 if MODE == "full" else 80
n_samples = 10

# Train/test split
perm = np.random.permutation(n_nodes)
n_train = int(0.6 * n_nodes)
train_nodes = perm[:n_train].tolist()
test_nodes = perm[n_train:].tolist()

results = {}
all_nodes = list(range(n_nodes))

for agg_name in ["mean", "pool"]:
    print(f"\\n--- GraphSAGE with {agg_name.upper()} aggregator ---")
    model = GraphSAGE(8, 16, n_communities, aggregator_type=agg_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        model.train()
        logits = model(X, train_nodes, adj_list, n_samples)
        loss = F.cross_entropy(logits, y[train_nodes])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % (40 if MODE == "full" else 20) == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(X, test_nodes, adj_list, n_samples)
                test_pred = test_logits.argmax(dim=1)
                test_acc = (test_pred == y[test_nodes]).float().mean()
            print(f"  Epoch {epoch+1:>3d}  Loss: {loss.item():.4f}  Test Acc: {test_acc.item():.4f}")

    # Final evaluation on all nodes
    model.eval()
    with torch.no_grad():
        all_logits = model(X, all_nodes, adj_list, n_samples)
        all_pred = all_logits.argmax(dim=1)
        acc = (all_pred == y).float().mean()
    results[agg_name] = {"acc": acc.item(), "pred": all_pred.cpu().numpy(), "emb": all_logits.cpu().numpy()}
    print(f"  Final accuracy (all nodes): {acc.item():.4f}")

# --- Visualise embeddings ---
from sklearn.manifold import TSNE

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cmap = ["#4A90D9", "#E74C3C", "#2ECC71"]

for idx, agg_name in enumerate(["mean", "pool"]):
    ax = axes[idx]
    emb = results[agg_name]["emb"]
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    emb_2d = tsne.fit_transform(emb)
    for c in range(n_communities):
        mask = labels == c
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=cmap[c], s=40,
                   edgecolors="k", linewidths=0.5, label=f"Community {c}")
    ax.set_title(f"GraphSAGE ({agg_name.upper()}) — Acc: {results[agg_name]['acc']:.3f}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("output.png", dpi=100)
plt.show()
print("\\nPlot saved to output.png")`,
      codeLanguage: "python",
    },
    {
      title: "Graph Attention Networks (GAT)",
      slug: "graph-attention-networks",
      description:
        "Attention mechanisms on graphs, multi-head attention, and GAT layer implementation with attention weight visualization",
      isDeepLearning: true,
      markdownContent: `# Graph Attention Networks (GAT)

**Graph Attention Networks** (Veličković et al., 2018) replace the fixed normalisation weights in GCN with **learned attention coefficients**. Instead of weighting all neighbours equally (or by degree), GAT learns *which* neighbours are most important for each node — bringing the power of attention mechanisms to graph-structured data.

## Motivation

In GCN, the weight $1/\\sqrt{\\tilde{d}_i \\tilde{d}_j}$ is determined entirely by graph structure. But not all neighbours are equally informative: in a citation network, some cited papers are more relevant than others. GAT lets the model learn these importance weights.

## Attention Mechanism

Given node features $h_i$ and $h_j$ for a node $i$ and its neighbour $j$, the attention coefficient is:

$$
e_{ij} = \\text{LeakyReLU}\\!\\left(a^T \\cdot \\left[W h_i \\,\\|\\, W h_j\\right]\\right)
$$

where $W \\in \\mathbb{R}^{d' \\times d}$ is a shared linear transformation, $a \\in \\mathbb{R}^{2d'}$ is the attention vector, and $\\|$ denotes concatenation.

The coefficients are normalised across neighbours using softmax:

$$
\\alpha_{ij} = \\text{softmax}_j(e_{ij}) = \\frac{\\exp(e_{ij})}{\\sum_{k \\in \\mathcal{N}(i)} \\exp(e_{ik})}
$$

The output feature for node $i$ is:

$$
h_i' = \\sigma\\!\\left(\\sum_{j \\in \\mathcal{N}(i)} \\alpha_{ij}\\, W h_j\\right)
$$

## Multi-Head Attention

To stabilise learning and increase expressiveness, GAT uses **$K$ independent attention heads**, each with its own parameters:

$$
h_i' = \\Big\\|_{k=1}^{K} \\sigma\\!\\left(\\sum_{j \\in \\mathcal{N}(i)} \\alpha_{ij}^{(k)}\\, W^{(k)} h_j\\right)
$$

where $\\|$ denotes concatenation across heads. For the final layer, averaging is used instead:

$$
h_i' = \\sigma\\!\\left(\\frac{1}{K} \\sum_{k=1}^{K} \\sum_{j \\in \\mathcal{N}(i)} \\alpha_{ij}^{(k)}\\, W^{(k)} h_j\\right)
$$

## GAT vs GCN

| Aspect | GCN | GAT |
|--------|-----|-----|
| **Neighbour weighting** | Fixed by degree | Learned via attention |
| **Expressiveness** | Same weight for all neighbours | Different weights per neighbour |
| **Computation** | $O(n \\cdot d \\cdot d')$ | $O(n \\cdot d \\cdot d' \\cdot K)$ |
| **Interpretability** | Limited | Attention weights are interpretable |

## Why Attention Helps

Attention allows the model to perform **anisotropic** message passing — each neighbour can contribute differently based on feature similarity. This is especially valuable when graph structure is noisy or when edge importance varies significantly.

Run the code to implement a multi-head GAT from scratch, train it on the Karate Club graph, and visualise the learned attention weights as an edge heatmap.`,
      codeSnippet: `from ml_catalogue_runtime import MODE, DEVICE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device(DEVICE)

# --- Build Karate Club graph ---
edges = [
    (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),
    (0,12),(0,13),(0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),
    (1,17),(1,19),(1,21),(1,30),(2,3),(2,7),(2,8),(2,9),(2,13),(2,27),
    (2,28),(2,32),(3,7),(3,12),(3,13),(4,6),(4,10),(5,6),(5,10),(5,16),
    (6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),(14,33),(15,32),
    (15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),
    (23,25),(23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),
    (25,31),(26,29),(26,33),(27,33),(28,31),(28,33),(29,32),(29,33),
    (30,32),(30,33),(31,32),(31,33),(32,33),
]
n_nodes = 34
labels = torch.tensor([
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1
], dtype=torch.long).to(device)

# Build adjacency with self-loops
A = np.zeros((n_nodes, n_nodes))
for i, j in edges:
    A[i, j] = 1.0
    A[j, i] = 1.0
np.fill_diagonal(A, 1.0)  # self-loops
adj = torch.tensor(A, dtype=torch.float32).to(device)

# Node features: identity
X = torch.eye(n_nodes, dtype=torch.float32).to(device)


class GATHead(nn.Module):
    """Single attention head."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(out_features, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, H, adj):
        Wh = self.W(H)  # (n, out)
        # Compute attention logits
        e_src = Wh @ self.a_src  # (n, 1)
        e_dst = Wh @ self.a_dst  # (n, 1)
        e = e_src + e_dst.T      # (n, n) broadcast

        # LeakyReLU
        e = F.leaky_relu(e, negative_slope=0.2)

        # Mask non-edges with large negative value
        e = e.masked_fill(adj == 0, float("-inf"))

        # Softmax over neighbours
        alpha = F.softmax(e, dim=1)  # (n, n)

        # Weighted aggregation
        h_prime = alpha @ Wh  # (n, out)
        return h_prime, alpha


class MultiHeadGAT(nn.Module):
    """Multi-head GAT layer."""
    def __init__(self, in_features, out_features, n_heads, concat=True):
        super().__init__()
        self.heads = nn.ModuleList([
            GATHead(in_features, out_features) for _ in range(n_heads)
        ])
        self.concat = concat

    def forward(self, H, adj):
        head_outputs = []
        head_attentions = []
        for head in self.heads:
            h, alpha = head(H, adj)
            head_outputs.append(h)
            head_attentions.append(alpha)

        if self.concat:
            return torch.cat(head_outputs, dim=1), head_attentions
        else:
            return torch.stack(head_outputs).mean(dim=0), head_attentions


class GAT(nn.Module):
    """2-layer GAT for node classification."""
    def __init__(self, in_features, hidden_dim, n_classes, n_heads=4):
        super().__init__()
        self.layer1 = MultiHeadGAT(in_features, hidden_dim, n_heads, concat=True)
        self.layer2 = MultiHeadGAT(hidden_dim * n_heads, n_classes, 1, concat=False)

    def forward(self, X, adj):
        H, attn1 = self.layer1(X, adj)
        H = F.elu(H)
        H, attn2 = self.layer2(H, adj)
        return H, attn1, attn2


n_heads = 4
model = GAT(in_features=n_nodes, hidden_dim=8, n_classes=2, n_heads=n_heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Semi-supervised labels
train_mask = torch.zeros(n_nodes, dtype=torch.bool)
train_mask[0] = True
train_mask[33] = True
train_mask[1] = True
train_mask[32] = True
train_mask[2] = True
train_mask[31] = True

n_epochs = 400 if MODE == "full" else 150
print("=== Training Multi-Head GAT on Karate Club ===")
print(f"Mode: {MODE} | Epochs: {n_epochs} | Heads: {n_heads} | Train nodes: {train_mask.sum().item()}")

for epoch in range(n_epochs):
    model.train()
    logits, _, _ = model(X, adj)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % (80 if MODE == "full" else 30) == 0:
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(X, adj)
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean()
        print(f"Epoch {epoch+1:>3d}  Loss: {loss.item():.4f}  Accuracy: {acc.item():.4f}")

# --- Final evaluation ---
model.eval()
with torch.no_grad():
    logits, attn_layer1, attn_layer2 = model(X, adj)
    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean()

print(f"\\nFinal accuracy: {acc.item():.4f}")
print(f"Predictions: {pred.cpu().tolist()}")

# --- Visualise attention weights ---
# Average attention across heads for layer 1
avg_attn = torch.stack(attn_layer1).mean(dim=0).cpu().numpy()  # (n, n)

# Spring layout for graph
np.random.seed(42)
pos = np.random.randn(n_nodes, 2) * 0.5
for _ in range(100):
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((delta ** 2).sum(axis=2) + 1e-6)
    rep = (1.0 / dist)[:, :, np.newaxis] * (delta / dist[:, :, np.newaxis])
    np.fill_diagonal(rep[:, :, 0], 0)
    np.fill_diagonal(rep[:, :, 1], 0)
    disp = rep.sum(axis=1)
    for i, j in edges:
        d = pos[i] - pos[j]
        d_norm = np.sqrt((d ** 2).sum() + 1e-6)
        attract = d * d_norm
        disp[i] -= attract
        disp[j] += attract
    d_norm = np.sqrt((disp ** 2).sum(axis=1, keepdims=True) + 1e-6)
    pos += disp / d_norm * np.minimum(0.1, d_norm)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Attention heatmap (subset of interesting nodes)
ax = axes[0]
key_nodes = [0, 1, 2, 3, 32, 33, 8, 13, 23, 31]
attn_sub = avg_attn[np.ix_(key_nodes, key_nodes)]
im = ax.imshow(attn_sub, cmap="YlOrRd", interpolation="nearest")
ax.set_xticks(range(len(key_nodes)))
ax.set_xticklabels(key_nodes, fontsize=8)
ax.set_yticks(range(len(key_nodes)))
ax.set_yticklabels(key_nodes, fontsize=8)
ax.set_title("GAT Attention Weights (Key Nodes)")
ax.set_xlabel("Target Node")
ax.set_ylabel("Source Node")
fig.colorbar(im, ax=ax, shrink=0.8)

# Graph with attention-weighted edges
ax = axes[1]
colors = ["#4A90D9" if p == 0 else "#E74C3C" for p in pred.cpu().numpy()]

# Draw edges with width proportional to attention
for i, j in edges:
    w = (avg_attn[i, j] + avg_attn[j, i]) / 2
    alpha = float(np.clip(w * 5, 0.1, 1.0))
    lw = float(np.clip(w * 8, 0.3, 3.0))
    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
            color="gray", alpha=alpha, linewidth=lw)

ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=120, edgecolors="k", linewidths=0.8, zorder=5)
for idx in range(n_nodes):
    ax.annotate(str(idx), pos[idx], fontsize=6, ha="center", va="center", zorder=6)
ax.set_title(f"GAT Predictions (Acc: {acc.item():.3f}) — Edge Width = Attention")
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
plt.savefig("output.png", dpi=100)
plt.show()
print("Plot saved to output.png")`,
      codeLanguage: "python",
    },
  ],
};
