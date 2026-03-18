import type { Chapter } from "../types";

export const deepLearning: Chapter = {
  title: "Deep Learning",
  slug: "deep-learning",
  pages: [
    {
      title: "Neural Network Basics",
      slug: "neural-network-basics",
      description:
        "Perceptrons, multi-layer networks, forward pass, and common activation functions",
      isDeepLearning: true,
      markdownContent: `# Neural Network Basics

A **neural network** is a function approximator built from layers of interconnected nodes. Even a small network can learn surprisingly complex decision boundaries that linear models cannot represent.

## The Perceptron

The simplest neural unit is the **perceptron**, which computes a weighted sum of its inputs and applies an activation function:

$$
y = \\sigma\\!\\left(\\sum_{i=1}^{n} w_i x_i + b\\right)
$$

Here $w_i$ are learnable weights, $b$ is a bias term, and $\\sigma$ is a nonlinear activation function. A single perceptron can only learn linearly separable patterns — to go further we need multiple layers.

## Multi-Layer Networks

By stacking layers we create a **multi-layer perceptron (MLP)**. A two-layer network computes:

$$
\\hat{y} = \\sigma\\bigl(W_2 \\cdot \\sigma(W_1 \\cdot x + b_1) + b_2\\bigr)
$$

The first layer maps the input $x$ into a hidden representation, and the second layer maps that representation to the output. The nonlinear activations between layers are essential — without them the entire network would collapse into a single linear transformation.

## Activation Functions

The choice of activation function shapes how a network learns:

- **ReLU:** $f(x) = \\max(0, x)$ — simple, fast, and the default choice for hidden layers.
- **Sigmoid:** $\\sigma(x) = \\frac{1}{1+e^{-x}}$ — squashes outputs to $(0, 1)$, common in output layers for binary classification.
- **Tanh:** $\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$ — zero-centred alternative to sigmoid.

## Why This Matters

Neural networks are the foundation of deep learning. Understanding how data flows through layers, how weights transform inputs, and how activations introduce nonlinearity is essential before tackling more advanced architectures.

Run the code to train a 2-layer neural network on spiral data and watch the training loss decrease.`,
      codeSnippet: `from ml_catalogue_runtime import DEVICE
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device(DEVICE)

# Generate spiral dataset (2 classes)
np.random.seed(42)
N = 200  # points per class
t = np.linspace(0, 4 * np.pi, N)
X = np.vstack([
    np.column_stack([t * np.cos(t) + np.random.randn(N) * 0.3,
                     t * np.sin(t) + np.random.randn(N) * 0.3]),
    np.column_stack([t * np.cos(t + np.pi) + np.random.randn(N) * 0.3,
                     t * np.sin(t + np.pi) + np.random.randn(N) * 0.3])
])
y = np.array([0] * N + [1] * N, dtype=np.float32)

X_t = torch.tensor(X, dtype=torch.float32).to(device)
y_t = torch.tensor(y).unsqueeze(1).to(device)

# 2-layer neural network
model = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1), nn.Sigmoid()
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

print("=== Training 2-Layer Neural Network on Spiral Data ===")
for epoch in range(301):
    pred = model(X_t)
    loss = criterion(pred, y_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        acc = ((pred > 0.5).float() == y_t).float().mean()
        print(f"Epoch {epoch:3d}  Loss: {loss.item():.4f}  Accuracy: {acc.item():.4f}")

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                      np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
with torch.no_grad():
    zz = model(grid).cpu().numpy().reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, zz, levels=50, cmap="RdYlBu", alpha=0.8)
plt.scatter(X[:N, 0], X[:N, 1], c="royalblue", edgecolors="k", s=15, label="Class 0")
plt.scatter(X[N:, 0], X[N:, 1], c="crimson", edgecolors="k", s=15, label="Class 1")
plt.title("Neural Network Decision Boundary (Spiral Data)")
plt.legend()
plt.tight_layout()
plt.savefig("output.png", dpi=100)
plt.show()
print("Plot saved to output.png")`,
      codeLanguage: "python",
    },
    {
      title: "Backpropagation & Activation Functions",
      slug: "backpropagation-activation-functions",
      description:
        "Chain rule, gradient flow, gradient descent, and the role of activation functions in training",
      isDeepLearning: true,
      markdownContent: `# Backpropagation & Activation Functions

**Backpropagation** is the algorithm that makes training deep networks practical. It efficiently computes how much each weight contributed to the error, enabling gradient descent to update every parameter in the network. **Activation functions** are central to this process — they introduce the nonlinearity that makes deep networks powerful, and their derivatives directly shape gradient flow during backpropagation.

## The Chain Rule

Consider a network where input $x$ passes through a linear layer to produce $z = wx + b$, then through an activation to give $\\hat{y} = \\sigma(z)$, and finally into a loss function $L(\\hat{y}, y)$. The gradient of the loss with respect to weight $w$ is:

$$
\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial \\hat{y}} \\cdot \\frac{\\partial \\hat{y}}{\\partial z} \\cdot \\frac{\\partial z}{\\partial w}
$$

Each factor is a local derivative that can be computed independently. Backpropagation chains these local gradients from the output layer back to the input — hence the name.

## Gradient Descent Update

Once we have the gradient $\\frac{\\partial L}{\\partial w}$ for every weight, we update each parameter to reduce the loss:

$$
w \\leftarrow w - \\alpha \\frac{\\partial L}{\\partial w}
$$

The learning rate $\\alpha$ controls the step size. Modern optimizers like **Adam** adapt $\\alpha$ per-parameter using running averages of the gradient and its square.

## Computational Graph Perspective

Frameworks like PyTorch build a **computational graph** during the forward pass. Each operation records its inputs and the local gradient formula. Calling \`.backward()\` traverses this graph in reverse, accumulating gradients via the chain rule — this is **automatic differentiation**.

## Activation Functions in the Forward & Backward Pass

Activation functions introduce nonlinearity into neural networks. Without them, a multi-layer network would be equivalent to a single linear transformation, no matter how many layers it has. Their derivatives are a critical part of the backward pass.

### ReLU (Rectified Linear Unit)

The most widely used activation in modern networks. It is fast to compute and avoids the vanishing gradient problem for positive inputs:

$$
\\text{ReLU}(x) = \\max(0, x), \\quad \\text{ReLU}'(x) = \\begin{cases} 1 & x > 0 \\\\ 0 & x \\leq 0 \\end{cases}
$$

**Drawback:** neurons can "die" — once a ReLU unit's input is always negative, its gradient is permanently zero.

### LeakyReLU

A fix for dying neurons that allows a small gradient $\\alpha$ (typically 0.01) when $x < 0$:

$$
\\text{LeakyReLU}(x) = \\begin{cases} x & x > 0 \\\\ \\alpha x & x \\leq 0 \\end{cases}
$$

### Sigmoid

Maps inputs to the range $(0, 1)$, making it natural for probability outputs:

$$
\\sigma(x) = \\frac{1}{1 + e^{-x}}, \\quad \\sigma'(x) = \\sigma(x)(1 - \\sigma(x))
$$

The maximum derivative is only $0.25$ (at $x = 0$), which causes **vanishing gradients** in deep networks.

### Tanh

Zero-centred version of sigmoid, mapping to $(-1, 1)$:

$$
\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}, \\quad \\tanh'(x) = 1 - \\tanh^2(x)
$$

### Softmax

Used in the output layer for multi-class classification. For a vector $z$ of logits, the $i$-th output is $\\text{softmax}(z)_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$, producing a probability distribution that sums to 1.

## The Vanishing Gradient Problem

When many sigmoid or tanh layers are stacked, the gradients at each layer are multiplied together. Since each derivative is less than 1, the product shrinks exponentially — early layers learn extremely slowly. This is a key reason ReLU and its variants dominate hidden layers in deep networks.

## Common Pitfalls

- **Vanishing gradients:** In deep networks with sigmoid activations, gradients shrink exponentially as they propagate backwards, stalling learning in early layers.
- **Exploding gradients:** Conversely, large weight values can cause gradients to grow exponentially, destabilising training. Gradient clipping mitigates this.

## Why This Matters

Every deep learning training loop relies on backpropagation. Understanding how gradients flow through a network — and how activation function choice shapes that flow — helps you debug training issues, choose architectures, and reason about why certain designs work better than others.

Run the code to compute backpropagation manually on a tiny network, verify the results against PyTorch autograd, and visualise how different activation functions and their derivatives affect gradient flow.`,
      codeSnippet: `from ml_catalogue_runtime import DEVICE
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device(DEVICE)

# --- Manual Backpropagation on a Tiny Network ---
# Network: input x -> z1 = w1*x + b1 -> a1 = relu(z1)
#                    -> z2 = w2*a1 + b2 -> y_hat = sigmoid(z2)
# Loss: L = (y_hat - y)^2

np.random.seed(42)
x_val, y_val = 1.5, 1.0
w1_val, b1_val = 0.8, 0.1
w2_val, b2_val = -0.6, 0.3

print("=== Manual Backpropagation ===")
print(f"Input x={x_val}, Target y={y_val}")
print(f"Weights: w1={w1_val}, b1={b1_val}, w2={w2_val}, b2={b2_val}\\n")

# Forward pass (manual)
z1 = w1_val * x_val + b1_val
a1 = max(0.0, z1)  # ReLU
z2 = w2_val * a1 + b2_val
y_hat = 1.0 / (1.0 + np.exp(-z2))  # Sigmoid
loss = (y_hat - y_val) ** 2

print("--- Forward Pass ---")
print(f"z1 = {z1:.4f}, a1 = ReLU(z1) = {a1:.4f}")
print(f"z2 = {z2:.4f}, y_hat = sigmoid(z2) = {y_hat:.4f}")
print(f"Loss = (y_hat - y)^2 = {loss:.4f}\\n")

# Backward pass (manual chain rule)
dL_dyhat = 2 * (y_hat - y_val)
dyhat_dz2 = y_hat * (1 - y_hat)  # sigmoid derivative
dz2_dw2 = a1
dz2_db2 = 1.0
dz2_da1 = w2_val
da1_dz1 = 1.0 if z1 > 0 else 0.0  # ReLU derivative
dz1_dw1 = x_val
dz1_db1 = 1.0

dL_dw2 = dL_dyhat * dyhat_dz2 * dz2_dw2
dL_db2 = dL_dyhat * dyhat_dz2 * dz2_db2
dL_dw1 = dL_dyhat * dyhat_dz2 * dz2_da1 * da1_dz1 * dz1_dw1
dL_db1 = dL_dyhat * dyhat_dz2 * dz2_da1 * da1_dz1 * dz1_db1

print("--- Manual Gradients (Chain Rule) ---")
print(f"dL/dw2 = {dL_dw2:.6f}")
print(f"dL/db2 = {dL_db2:.6f}")
print(f"dL/dw1 = {dL_dw1:.6f}")
print(f"dL/db1 = {dL_db1:.6f}\\n")

# Verify with PyTorch autograd
w1 = torch.tensor(w1_val, requires_grad=True, device=device)
b1 = torch.tensor(b1_val, requires_grad=True, device=device)
w2 = torch.tensor(w2_val, requires_grad=True, device=device)
b2 = torch.tensor(b2_val, requires_grad=True, device=device)
x_t = torch.tensor(x_val, device=device)
y_t = torch.tensor(y_val, device=device)

a1_t = torch.relu(w1 * x_t + b1)
y_hat_t = torch.sigmoid(w2 * a1_t + b2)
loss_t = (y_hat_t - y_t) ** 2
loss_t.backward()

print("--- PyTorch Autograd Gradients ---")
print(f"dL/dw2 = {w2.grad.item():.6f}")
print(f"dL/db2 = {b2.grad.item():.6f}")
print(f"dL/dw1 = {w1.grad.item():.6f}")
print(f"dL/db1 = {b1.grad.item():.6f}\\n")

print("--- Verification ---")
for name, manual, auto in [("w1", dL_dw1, w1.grad.item()),
                            ("b1", dL_db1, b1.grad.item()),
                            ("w2", dL_dw2, w2.grad.item()),
                            ("b2", dL_db2, b2.grad.item())]:
    match = "PASS" if abs(manual - auto) < 1e-6 else "FAIL"
    print(f"  d{name}: manual={manual:.6f}  autograd={auto:.6f}  [{match}]")

# --- Activation Function Comparison ---
print("\\n=== Activation Function Properties ===")
x_arr = np.linspace(-5, 5, 500)

def relu(x): return np.maximum(0, x)
def relu_d(x): return (x > 0).astype(float)
def leaky_relu(x, a=0.01): return np.where(x > 0, x, a * x)
def leaky_relu_d(x, a=0.01): return np.where(x > 0, 1.0, a)
def sigmoid_fn(x): return 1 / (1 + np.exp(-x))
def sigmoid_d(x): s = sigmoid_fn(x); return s * (1 - s)
def tanh_fn(x): return np.tanh(x)
def tanh_d(x): return 1 - np.tanh(x) ** 2

activations = [
    ("ReLU", relu, relu_d),
    ("LeakyReLU (a=0.01)", leaky_relu, leaky_relu_d),
    ("Sigmoid", sigmoid_fn, sigmoid_d),
    ("Tanh", tanh_fn, tanh_d),
]

print(f"{'Function':<22} {'Range':<16} {'Max Derivative':<16} {'Zero-centred'}")
print("-" * 66)
for name, fn, dfn in activations:
    vals = fn(x_arr)
    derivs = dfn(x_arr)
    rng = f"({vals.min():.2f}, {vals.max():.2f})"
    print(f"{name:<22} {rng:<16} {derivs.max():<16.4f} {'Yes' if vals.min() < 0 else 'No'}")
print(f"{'Softmax':<22} {'(0, 1)':<16} {'—':<16} {'No'}")

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i, (name, fn, dfn) in enumerate(activations):
    axes[0, i].plot(x_arr, fn(x_arr), color="steelblue", linewidth=2)
    axes[0, i].set_title(name)
    axes[0, i].axhline(0, color="gray", linewidth=0.5)
    axes[0, i].axvline(0, color="gray", linewidth=0.5)
    axes[0, i].set_ylabel("f(x)" if i == 0 else "")
    axes[0, i].grid(alpha=0.3)

    axes[1, i].plot(x_arr, dfn(x_arr), color="coral", linewidth=2)
    axes[1, i].set_title(f"{name} derivative")
    axes[1, i].axhline(0, color="gray", linewidth=0.5)
    axes[1, i].axvline(0, color="gray", linewidth=0.5)
    axes[1, i].set_ylabel("f'(x)" if i == 0 else "")
    axes[1, i].set_xlabel("x")
    axes[1, i].grid(alpha=0.3)

fig.suptitle("Activation Functions and Their Derivatives", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("output.png", dpi=100, bbox_inches="tight")
plt.show()
print("\\nPlot saved to output.png")`,
      codeLanguage: "python",
    },
    {
      title: "CNNs",
      slug: "cnns",
      description:
        "Convolutional neural networks: convolution, pooling, feature maps, and image classification",
      isDeepLearning: true,
      markdownContent: `# Convolutional Neural Networks (CNNs)

**Convolutional Neural Networks** are specialised architectures for processing grid-structured data such as images. Instead of connecting every input to every neuron, CNNs use small learnable **filters** that slide across the input, exploiting spatial locality and translation invariance.

## The Convolution Operation

A 2D convolution slides a filter (kernel) of size $k \\times k$ across an input feature map. At each position, it computes the element-wise product and sums the result to produce one output value:

$$
(I * K)[i, j] = \\sum_{m=0}^{k-1} \\sum_{n=0}^{k-1} I[i+m,\\, j+n] \\cdot K[m, n]
$$

Key parameters that control the output size:

- **Stride ($s$):** how many pixels the filter moves per step. Stride 2 halves the spatial dimensions.
- **Padding ($p$):** zeros added around the input border. "Same" padding preserves spatial size.
- **Output size:** for input size $W$, filter size $k$, stride $s$, and padding $p$: $\\lfloor (W - k + 2p) / s \\rfloor + 1$.

## Feature Maps and Filters

Each convolutional layer applies multiple filters to produce multiple **feature maps** (channels). Early layers typically learn low-level features like edges and textures, while deeper layers capture increasingly abstract patterns like shapes and object parts.

## Pooling Layers

**Pooling** reduces spatial dimensions while retaining the most important information:

- **Max pooling:** takes the maximum value in each pooling window — preserves the strongest activation.
- **Average pooling:** takes the mean — useful for reducing noise.

A typical pooling window is $2 \\times 2$ with stride 2, halving both height and width.

## Standard CNN Architecture

The classic CNN pattern stacks convolutional blocks followed by a classifier:

$$
\\text{Input} \\rightarrow [\\text{Conv} \\rightarrow \\text{ReLU} \\rightarrow \\text{Pool}] \\times N \\rightarrow \\text{Flatten} \\rightarrow \\text{Dense} \\rightarrow \\text{Output}
$$

Each convolutional block extracts features at increasing levels of abstraction. The flatten operation converts the 2D feature maps into a 1D vector for the fully connected (dense) classification layers.

## Why CNNs Work

1. **Parameter sharing:** the same filter is applied everywhere, drastically reducing the number of parameters compared to fully connected layers.
2. **Translation invariance:** a learned feature detector works regardless of where the pattern appears in the image.
3. **Hierarchical features:** stacking layers lets the network compose simple features into complex ones.

## Why This Matters

CNNs are the backbone of modern computer vision — from image classification and object detection to medical imaging and autonomous driving. Understanding convolutions, pooling, and feature hierarchies is essential for working with any visual data.

Run the code to build and train a CNN on handwritten digit data and inspect the learned feature maps.`,
      codeSnippet: `from ml_catalogue_runtime import DEVICE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = torch.device(DEVICE)

# Generate synthetic digit-like data (8x8 images, 4 classes)
np.random.seed(42)
n_samples = 2000
n_classes = 4
img_size = 8

def make_pattern(cls, n):
    imgs = np.random.randn(n, 1, img_size, img_size).astype(np.float32) * 0.1
    for i in range(n):
        if cls == 0:  # horizontal line
            imgs[i, 0, 3:5, 1:7] += 1.0
        elif cls == 1:  # vertical line
            imgs[i, 0, 1:7, 3:5] += 1.0
        elif cls == 2:  # diagonal
            for j in range(img_size):
                if 0 <= j < img_size:
                    imgs[i, 0, j, j] += 1.0
        elif cls == 3:  # cross
            imgs[i, 0, 3:5, 1:7] += 1.0
            imgs[i, 0, 1:7, 3:5] += 1.0
    return imgs

X_all, y_all = [], []
for c in range(n_classes):
    X_all.append(make_pattern(c, n_samples // n_classes))
    y_all.append(np.full(n_samples // n_classes, c, dtype=np.int64))
X_all = np.concatenate(X_all)
y_all = np.concatenate(y_all)

# Shuffle and split
idx = np.random.permutation(n_samples)
X_all, y_all = X_all[idx], y_all[idx]
split = int(0.8 * n_samples)
X_train, X_test = torch.tensor(X_all[:split]).to(device), torch.tensor(X_all[split:]).to(device)
y_train, y_test = torch.tensor(y_all[:split]).to(device), torch.tensor(y_all[split:]).to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# Define CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 8x8 -> 4x4
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 4x4 -> 4x4
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 4x4 -> 2x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),           # 32 * 2 * 2 = 128
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Architecture summary
print("=== CNN Architecture ===")
print("Conv2d(1→16, 3x3) → ReLU → MaxPool(2x2)")
print("Conv2d(16→32, 3x3) → ReLU → MaxPool(2x2)")
print("Flatten → Dense(128→64) → ReLU → Dense(64→4)")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}\\n")

# Train
print("=== Training ===")
for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / len(X_train)
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test).argmax(dim=1)
            acc = (test_pred == y_test).float().mean().item()
        print(f"Epoch {epoch+1:2d}  Loss: {avg_loss:.4f}  Test Acc: {acc:.4f}")

# Visualise first-layer filters and feature maps
model.eval()
filters = model.features[0].weight.data.cpu().numpy()
sample = X_test[:1]
with torch.no_grad():
    feat_maps = model.features[:2](sample).cpu().numpy()[0]  # after conv1 + relu

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle("Conv Layer 1: Filters (top) and Feature Maps (bottom)", fontsize=13)
for i in range(8):
    axes[0, i].imshow(filters[i, 0], cmap="coolwarm", interpolation="nearest")
    axes[0, i].set_title(f"Filter {i}")
    axes[0, i].axis("off")
    axes[1, i].imshow(feat_maps[i], cmap="viridis", interpolation="nearest")
    axes[1, i].set_title(f"FMap {i}")
    axes[1, i].axis("off")
plt.tight_layout()
plt.savefig("output.png", dpi=100, bbox_inches="tight")
plt.show()
print("\\nPlot saved to output.png")`,
      codeLanguage: "python",
    },
    {
      title: "RNNs & LSTMs",
      slug: "rnns-lstms",
      description:
        "Recurrent neural networks, vanishing gradients, LSTM gates, and GRU for sequential data",
      isDeepLearning: true,
      markdownContent: `# Recurrent Neural Networks & LSTMs

**Recurrent Neural Networks (RNNs)** are designed for sequential data — text, time series, audio — where the order of inputs matters. Unlike feedforward networks, RNNs maintain a **hidden state** that carries information from previous time steps.

## Vanilla RNN

At each time step $t$, a vanilla RNN updates its hidden state $h_t$ by combining the current input $x_t$ with the previous hidden state $h_{t-1}$:

$$
h_t = \\tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

$$
y_t = W_{hy} h_t + b_y
$$

The same weight matrices $W_{xh}$, $W_{hh}$, and $W_{hy}$ are shared across all time steps — this is what gives RNNs their ability to handle variable-length sequences.

## The Vanishing Gradient Problem

When training vanilla RNNs with backpropagation through time (BPTT), gradients are multiplied by $W_{hh}$ at each step. Over many steps:

$$
\\frac{\\partial h_T}{\\partial h_1} = \\prod_{t=2}^{T} \\frac{\\partial h_t}{\\partial h_{t-1}}
$$

If the spectral radius of $W_{hh}$ is less than 1, gradients **vanish** exponentially, making it impossible to learn long-range dependencies. If greater than 1, they **explode**.

## Long Short-Term Memory (LSTM)

LSTMs solve the vanishing gradient problem by introducing a **cell state** $c_t$ — a highway that carries information across time steps with minimal interference — and three **gates** that control information flow:

**Forget gate** — decides what to discard from the cell state:
$$
f_t = \\sigma(W_f [h_{t-1}, x_t] + b_f)
$$

**Input gate** — decides what new information to store:
$$
i_t = \\sigma(W_i [h_{t-1}, x_t] + b_i), \\quad \\tilde{c}_t = \\tanh(W_c [h_{t-1}, x_t] + b_c)
$$

**Cell state update:**
$$
c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t
$$

**Output gate** — decides what to output:
$$
o_t = \\sigma(W_o [h_{t-1}, x_t] + b_o), \\quad h_t = o_t \\odot \\tanh(c_t)
$$

The cell state $c_t$ can carry gradients across many time steps because the forget gate allows the gradient to flow through with a multiplicative factor close to 1.

## Gated Recurrent Unit (GRU)

The **GRU** is a simplified variant of the LSTM that merges the cell state and hidden state into a single $h_t$, and uses two gates instead of three:

$$
z_t = \\sigma(W_z [h_{t-1}, x_t]) \\quad \\text{(update gate)}
$$
$$
r_t = \\sigma(W_r [h_{t-1}, x_t]) \\quad \\text{(reset gate)}
$$
$$
h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tanh(W_h [r_t \\odot h_{t-1}, x_t])
$$

GRUs have fewer parameters than LSTMs and often perform comparably on many tasks.

## Why This Matters

RNNs and LSTMs are foundational for understanding sequential modelling. While transformers have largely replaced them for NLP tasks, LSTMs remain widely used for time series forecasting, speech processing, and real-time sequential data where their incremental hidden state is advantageous.

Run the code to build an LSTM that learns a sine wave pattern and predicts future values.`,
      codeSnippet: `from ml_catalogue_runtime import DEVICE, MODE
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device(DEVICE)
QUICK = MODE == "quick"

# Generate sine wave data
np.random.seed(42)
n_points = 500 if QUICK else 2000
t = np.linspace(0, 10 * np.pi, n_points)
data = np.sin(t) + 0.1 * np.random.randn(len(t))
data = data.astype(np.float32)

# Create sequences: use seq_len points to predict the next one
seq_len = 30
X, y = [], []
for i in range(len(data) - seq_len):
    X.append(data[i:i + seq_len])
    y.append(data[i + seq_len])
X = torch.tensor(np.array(X)).unsqueeze(-1).to(device)  # (N, seq_len, 1)
y = torch.tensor(np.array(y)).unsqueeze(-1).to(device)   # (N, 1)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM model
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take last time step
        return self.fc(last_hidden)

model = LSTMPredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

print("=== LSTM Sequence Predictor ===")
print(f"Architecture: LSTM(input=1, hidden=32, layers=1) → Dense(32→1)")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\\n")

# Train
print("=== Training ===")
n_epochs = 30 if QUICK else 50
batch_size = 128
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        xb = X_train[i:i + batch_size]
        yb = y_train[i:i + batch_size]
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    if (epoch + 1) % (10 if not QUICK else 10) == 0:
        avg_loss = total_loss / len(X_train)
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test).item()
        print(f"Epoch {epoch+1:2d}  Train MSE: {avg_loss:.6f}  Test MSE: {test_loss:.6f}")

# Predict and plot
model.eval()
with torch.no_grad():
    pred_test = model(X_test).cpu().numpy().flatten()
actual_test = y_test.cpu().numpy().flatten()

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

# Full prediction vs actual
axes[0].plot(actual_test, label="Actual", color="steelblue", linewidth=1.5)
axes[0].plot(pred_test, label="LSTM Prediction", color="coral", linewidth=1.5, alpha=0.8)
axes[0].set_title("LSTM Sine Wave Prediction (Test Set)")
axes[0].set_xlabel("Time Step")
axes[0].set_ylabel("Value")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Zoomed-in view
zoom = min(100, len(actual_test))
axes[1].plot(range(zoom), actual_test[:zoom], "o-", label="Actual", color="steelblue", markersize=3)
axes[1].plot(range(zoom), pred_test[:zoom], "s-", label="Predicted", color="coral", markersize=3, alpha=0.8)
axes[1].set_title(f"Zoomed View (First {zoom} Steps)")
axes[1].set_xlabel("Time Step")
axes[1].set_ylabel("Value")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("output.png", dpi=100, bbox_inches="tight")
plt.show()
print("\\nPlot saved to output.png")`,
      codeLanguage: "python",
    },
    {
      title: "Transformers",
      slug: "transformers",
      description:
        "Self-attention, multi-head attention, positional encoding, and the encoder-decoder architecture",
      isDeepLearning: true,
      markdownContent: `# Transformers

The **Transformer** architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), replaced recurrence with **self-attention** as the primary mechanism for modelling sequences. This design enables massive parallelisation and captures long-range dependencies far more effectively than RNNs.

## Self-Attention Mechanism

Self-attention computes how much each element in a sequence should attend to every other element. Given an input sequence, we project it into three representations:

- **Query ($Q$):** what am I looking for?
- **Key ($K$):** what do I contain?
- **Value ($V$):** what information do I provide?

The attention output is a weighted sum of values, where each weight reflects the relevance between a query and a key:

$$
\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V
$$

The scaling factor $\\sqrt{d_k}$ prevents the dot products from becoming too large, which would push the softmax into regions with vanishingly small gradients.

## Multi-Head Attention

Instead of computing a single attention function, transformers run $h$ attention heads in parallel, each with its own learned projections:

$$
\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W^O
$$

$$
\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

Different heads can specialise — one might capture syntactic relationships while another captures semantic similarity.

## Positional Encoding

Since self-attention treats the input as a **set** (order-agnostic), the model needs explicit position information. The original transformer uses sinusoidal positional encodings:

$$
PE_{(pos, 2i)} = \\sin\\!\\left(\\frac{pos}{10000^{2i/d}}\\right), \\quad PE_{(pos, 2i+1)} = \\cos\\!\\left(\\frac{pos}{10000^{2i/d}}\\right)
$$

These are added to the input embeddings, giving the model a sense of token order. The sinusoidal pattern allows the model to generalise to sequence lengths not seen during training.

## Encoder-Decoder Architecture

The full transformer has two halves:

- **Encoder:** processes the input sequence. Each layer has multi-head self-attention followed by a feedforward network, with layer normalisation and residual connections.
- **Decoder:** generates the output sequence. Each layer has masked self-attention (preventing the model from attending to future tokens), cross-attention over the encoder output, and a feedforward network.

For each sub-layer:
$$
\\text{output} = \\text{LayerNorm}(x + \\text{SubLayer}(x))
$$

## Why Transformers Dominate

1. **Parallelisation:** unlike RNNs that process tokens sequentially, self-attention computes all pairwise interactions simultaneously — enabling efficient training on GPUs.
2. **Long-range dependencies:** any two tokens interact directly regardless of distance, avoiding the vanishing gradient bottleneck of RNNs.
3. **Scalability:** transformer performance scales predictably with model size and data ("scaling laws").

## BERT vs GPT: Encoder-Only vs Decoder-Only

The transformer architecture has been adapted into two dominant paradigms:

- **BERT (encoder-only):** uses bidirectional self-attention — each token attends to all other tokens. Trained with masked language modelling (predict masked tokens). Excels at understanding tasks: classification, NER, question answering.
- **GPT (decoder-only):** uses causal (left-to-right) self-attention — each token only attends to preceding tokens. Trained with next-token prediction. Excels at generation tasks: text completion, dialogue, code generation.

## Why This Matters

Transformers are the foundation of modern AI — from large language models (GPT, Claude) to vision transformers (ViT) and protein structure prediction (AlphaFold). Understanding self-attention, positional encoding, and the encoder-decoder structure is essential for working with any state-of-the-art model.

Run the code to implement self-attention from scratch, visualise attention weights, and see how the mechanism captures token relationships.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# === Self-Attention from Scratch ===

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def self_attention(X, W_Q, W_K, W_V):
    """
    Compute scaled dot-product self-attention.
    X: (seq_len, d_model) - input embeddings
    Returns: attention output and attention weights
    """
    Q = X @ W_Q  # (seq_len, d_k)
    K = X @ W_K
    V = X @ W_V

    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)  # (seq_len, seq_len)
    weights = softmax(scores)           # attention weights
    output = weights @ V                # (seq_len, d_v)
    return output, weights, Q, K, V

def multi_head_attention(X, n_heads, d_model, d_k):
    """Multi-head attention with random projections."""
    heads_output = []
    heads_weights = []
    for _ in range(n_heads):
        W_Q = np.random.randn(d_model, d_k) * 0.1
        W_K = np.random.randn(d_model, d_k) * 0.1
        W_V = np.random.randn(d_model, d_k) * 0.1
        out, w, _, _, _ = self_attention(X, W_Q, W_K, W_V)
        heads_output.append(out)
        heads_weights.append(w)
    concat = np.concatenate(heads_output, axis=-1)  # (seq_len, n_heads * d_k)
    W_O = np.random.randn(n_heads * d_k, d_model) * 0.1
    return concat @ W_O, heads_weights

def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding."""
    PE = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, np.newaxis]
    div = 10000 ** (2 * np.arange(d_model // 2)[np.newaxis, :] / d_model)
    PE[:, 0::2] = np.sin(pos / div)
    PE[:, 1::2] = np.cos(pos / div)
    return PE

# --- Demo: Attention on a simple sentence ---
tokens = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(tokens)
d_model = 32
d_k = 16

# Create mock embeddings (in practice these come from a learned embedding layer)
embeddings = np.random.randn(seq_len, d_model) * 0.5

# Add positional encoding
PE = positional_encoding(seq_len, d_model)
X = embeddings + PE

print("=== Self-Attention from Scratch ===")
print(f"Tokens: {tokens}")
print(f"Embedding dim: {d_model}, Key/Query dim: {d_k}\\n")

# Single-head attention
W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_k) * 0.1

output, attn_weights, Q, K, V = self_attention(X, W_Q, W_K, W_V)

print("--- Single-Head Attention Weights ---")
print("Each row shows how much a token attends to every other token:")
header = "         " + "  ".join(f"{t:>5}" for t in tokens)
print(header)
for i, tok in enumerate(tokens):
    row = "  ".join(f"{attn_weights[i, j]:5.3f}" for j in range(seq_len))
    print(f"{tok:>6}   {row}")

# Multi-head attention
n_heads = 4
mh_output, mh_weights = multi_head_attention(X, n_heads, d_model, d_k)
print(f"\\n--- Multi-Head Attention ({n_heads} heads) ---")
print(f"Input shape:  {X.shape}")
print(f"Output shape: {mh_output.shape}")

# --- Visualisation ---
fig, axes = plt.subplots(1, n_heads + 1, figsize=(20, 4))

# Single head
im = axes[0].imshow(attn_weights, cmap="Blues", vmin=0, vmax=1)
axes[0].set_xticks(range(seq_len))
axes[0].set_yticks(range(seq_len))
axes[0].set_xticklabels(tokens, rotation=45, ha="right")
axes[0].set_yticklabels(tokens)
axes[0].set_title("Single Head")
axes[0].set_xlabel("Key")
axes[0].set_ylabel("Query")

# Multiple heads
for h in range(n_heads):
    im = axes[h + 1].imshow(mh_weights[h], cmap="Blues", vmin=0, vmax=1)
    axes[h + 1].set_xticks(range(seq_len))
    axes[h + 1].set_yticks(range(seq_len))
    axes[h + 1].set_xticklabels(tokens, rotation=45, ha="right")
    axes[h + 1].set_yticklabels(tokens)
    axes[h + 1].set_title(f"Head {h + 1}")
    axes[h + 1].set_xlabel("Key")

fig.suptitle("Self-Attention Weights: How Tokens Attend to Each Other", fontsize=14, y=1.02)
plt.colorbar(im, ax=axes, shrink=0.8, label="Attention Weight")
plt.tight_layout()
plt.savefig("output.png", dpi=100, bbox_inches="tight")
plt.show()

# Positional encoding visualisation
print("\\n--- Positional Encoding Pattern ---")
PE_vis = positional_encoding(20, d_model)
fig2, ax2 = plt.subplots(figsize=(10, 4))
im2 = ax2.imshow(PE_vis.T, cmap="RdBu", aspect="auto", interpolation="nearest")
ax2.set_xlabel("Position")
ax2.set_ylabel("Embedding Dimension")
ax2.set_title("Sinusoidal Positional Encoding")
plt.colorbar(im2, label="Value")
plt.tight_layout()
plt.savefig("output.png", dpi=100, bbox_inches="tight")
plt.show()
print("\\nPlot saved to output.png")`,
      codeLanguage: "python",
    },
  ],
};
