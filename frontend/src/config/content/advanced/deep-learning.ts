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
      codeSnippet: `import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device(os.environ.get("ML_CATALOGUE_DEVICE", "cpu"))

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
      title: "Backpropagation",
      slug: "backpropagation",
      description:
        "Chain rule, gradient flow, and gradient descent for training neural networks",
      isDeepLearning: true,
      markdownContent: `# Backpropagation

**Backpropagation** is the algorithm that makes training deep networks practical. It efficiently computes how much each weight contributed to the error, enabling gradient descent to update every parameter in the network.

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

## Common Pitfalls

- **Vanishing gradients:** In deep networks with sigmoid activations, gradients shrink exponentially as they propagate backwards, stalling learning in early layers.
- **Exploding gradients:** Conversely, large weight values can cause gradients to grow exponentially, destabilising training. Gradient clipping mitigates this.

## Why This Matters

Every deep learning training loop relies on backpropagation. Understanding how gradients flow through a network helps you debug training issues, choose architectures, and reason about why certain designs work better than others.

Run the code to compute backpropagation manually on a tiny network and verify the results against PyTorch autograd.`,
      codeSnippet: `import os
import torch
import numpy as np

device = torch.device(os.environ.get("ML_CATALOGUE_DEVICE", "cpu"))

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
    print(f"  d{name}: manual={manual:.6f}  autograd={auto:.6f}  [{match}]")`,
      codeLanguage: "python",
    },
    {
      title: "Activation Functions",
      slug: "activation-functions",
      description:
        "ReLU, LeakyReLU, sigmoid, tanh, and softmax — comparison and gradient behaviour",
      markdownContent: `# Activation Functions

**Activation functions** introduce nonlinearity into neural networks. Without them, a multi-layer network would be equivalent to a single linear transformation, no matter how many layers it has.

## Common Activations

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

## Why This Matters

Choosing the right activation function affects training speed, gradient health, and model accuracy. ReLU is the safe default for hidden layers, while sigmoid and softmax serve specific roles in output layers.

Run the code to plot all five activation functions and their derivatives side by side.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)

# Activation functions and derivatives
def relu(x): return np.maximum(0, x)
def relu_d(x): return (x > 0).astype(float)

def leaky_relu(x, a=0.01): return np.where(x > 0, x, a * x)
def leaky_relu_d(x, a=0.01): return np.where(x > 0, 1.0, a)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_d(x): s = sigmoid(x); return s * (1 - s)

def tanh(x): return np.tanh(x)
def tanh_d(x): return 1 - np.tanh(x) ** 2

def softmax(x): e = np.exp(x - x.max()); return e / e.sum()

activations = [
    ("ReLU", relu, relu_d),
    ("LeakyReLU (a=0.01)", leaky_relu, leaky_relu_d),
    ("Sigmoid", sigmoid, sigmoid_d),
    ("Tanh", tanh, tanh_d),
]

print("=== Activation Function Properties ===")
print(f"{'Function':<22} {'Range':<16} {'Max Derivative':<16} {'Zero-centred'}")
print("-" * 66)
for name, fn, dfn in activations:
    vals = fn(x)
    derivs = dfn(x)
    rng = f"({vals.min():.2f}, {vals.max():.2f})"
    print(f"{name:<22} {rng:<16} {derivs.max():<16.4f} {'Yes' if vals.min() < 0 else 'No'}")
print(f"{'Softmax':<22} {'(0, 1)':<16} {'—':<16} {'No'}")

# Plot activation functions and their derivatives
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i, (name, fn, dfn) in enumerate(activations):
    axes[0, i].plot(x, fn(x), color="steelblue", linewidth=2)
    axes[0, i].set_title(name)
    axes[0, i].axhline(0, color="gray", linewidth=0.5)
    axes[0, i].axvline(0, color="gray", linewidth=0.5)
    axes[0, i].set_ylabel("f(x)" if i == 0 else "")
    axes[0, i].grid(alpha=0.3)

    axes[1, i].plot(x, dfn(x), color="coral", linewidth=2)
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
  ],
};
