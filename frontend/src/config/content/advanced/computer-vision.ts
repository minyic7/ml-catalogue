import type { Chapter } from "../types";

export const computerVision: Chapter = {
  title: "Computer Vision",
  slug: "computer-vision",
  pages: [
    {
      title: "Image Representation & Convolutions",
      slug: "image-representation-convolutions",
      description:
        "Images as tensors, convolution operations, and common kernels for edge detection and blurring",
      markdownContent: `# Image Representation & Convolutions

Digital images are stored as multi-dimensional arrays — or **tensors**. A grayscale image is a 2D matrix of shape $H \\times W$, where each element is a pixel intensity (typically 0–255). A colour image adds a channel dimension, giving shape $H \\times W \\times C$ where $C = 3$ for RGB.

## The Convolution Operation

A **convolution** slides a small matrix called a **kernel** (or filter) across the image, computing a weighted sum at every position. For a 2D input image $I$ and kernel $F$ of size $m \\times n$, the output (feature map) $G$ is:

$$
G(i,j) = \\sum_{m} \\sum_{n} F(m,n) \\cdot I(i - m, j - n)
$$

The kernel acts as a local pattern detector. Different kernels extract different features from the image.

## Common Kernels

**Edge detection** kernels (like the Sobel filter) highlight regions where pixel intensity changes rapidly — these correspond to edges in the image. A horizontal Sobel kernel is:

$$
S_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix}
$$

**Blurring** kernels (like a box filter) average nearby pixels, smoothing out noise. **Sharpening** kernels enhance edges by subtracting a blurred version from the original, amplifying high-frequency detail.

The output dimensions depend on the kernel size, stride, and padding — concepts we explore on the next page.

Run the code below to apply convolution kernels to a synthetic image and see how each kernel transforms the input.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic 64x64 grayscale image with edges and gradients
image = np.zeros((64, 64), dtype=np.float64)
image[16:48, 16:48] = 1.0         # bright square
image[24:40, 24:40] = 0.5         # inner square
image[30:34, 10:54] = 0.8         # horizontal bar
print(f"Image shape: {image.shape} (H x W grayscale)")

# Define convolution kernels
kernels = {
    "Edge (Sobel-X)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64),
    "Blur (Box 3x3)": np.ones((3, 3), dtype=np.float64) / 9.0,
    "Sharpen":        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64),
}

def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return out

# Apply each kernel and display results
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
axes[0].imshow(image, cmap="gray"); axes[0].set_title("Original")
for idx, (name, kernel) in enumerate(kernels.items()):
    result = convolve2d(image, kernel)
    print(f"{name}: kernel {kernel.shape}, output {result.shape}, "
          f"range [{result.min():.2f}, {result.max():.2f}]")
    axes[idx+1].imshow(result, cmap="gray"); axes[idx+1].set_title(name)
for ax in axes: ax.axis("off")
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Feature Detection & Pooling",
      slug: "feature-detection-pooling",
      description:
        "Max pooling, average pooling, stride and padding effects, and hierarchical feature maps",
      markdownContent: `# Feature Detection & Pooling

After convolution produces a feature map, **pooling** reduces its spatial dimensions while retaining the most important information. This makes the representation more compact and invariant to small translations.

## Max Pooling & Average Pooling

**Max pooling** slides a window across the feature map and keeps only the maximum value in each region. It preserves the strongest activations — if an edge was detected anywhere inside the window, the output retains that signal.

**Average pooling** takes the mean of all values in each window. It produces a smoother, more blended summary of the region. Max pooling is more common in modern architectures because it better preserves sharp features.

Both operations use a window of size $K \\times K$ and move it with stride $S$.

## Output Dimensions

The spatial size of the output after convolution or pooling depends on the input width $W$, kernel size $K$, padding $P$, and stride $S$:

$$
O = \\left\\lfloor \\frac{W - K + 2P}{S} \\right\\rfloor + 1
$$

For example, a $32 \\times 32$ feature map with a $2 \\times 2$ pooling window and stride 2 (no padding) gives $O = \\lfloor \\frac{32 - 2}{2} \\rfloor + 1 = 16$. Each pooling layer halves the spatial resolution.

## Hierarchical Feature Learning

In a CNN, stacking convolution + pooling layers builds a **feature hierarchy**. Early layers detect simple patterns like edges and corners with small receptive fields. Deeper layers combine these into complex features — textures, parts, and eventually whole objects.

Each neuron in a deeper layer "sees" a larger region of the original image (its **receptive field** grows), enabling it to recognize increasingly abstract patterns.

Run the code below to see how max and average pooling transform a feature map and how different stride/padding settings change the output dimensions.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt

# Create a sample 8x8 feature map with interesting patterns
np.random.seed(42)
feature_map = np.zeros((8, 8))
feature_map[1:3, 1:3] = 0.9   # top-left patch
feature_map[5:7, 5:7] = 0.7   # bottom-right patch
feature_map[3:5, :] = 0.4     # horizontal band
feature_map += np.random.rand(8, 8) * 0.15
print(f"Input feature map shape: {feature_map.shape}")

def pool2d(x, size=2, stride=2, mode="max"):
    h, w = x.shape
    oh = (h - size) // stride + 1
    ow = (w - size) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            region = x[i*stride:i*stride+size, j*stride:j*stride+size]
            out[i, j] = region.max() if mode == "max" else region.mean()
    return out

max_pooled = pool2d(feature_map, size=2, stride=2, mode="max")
avg_pooled = pool2d(feature_map, size=2, stride=2, mode="avg")
print(f"After 2x2 max pool (stride 2): {max_pooled.shape}")
print(f"After 2x2 avg pool (stride 2): {avg_pooled.shape}")

# Show dimension formula for different configs
for W, K, P, S in [(32, 3, 1, 1), (32, 3, 0, 2), (28, 5, 2, 1), (16, 2, 0, 2)]:
    O = (W - K + 2*P) // S + 1
    print(f"W={W}, K={K}, P={P}, S={S}  =>  Output: {O}")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
for ax, data, title in zip(axes,
    [feature_map, max_pooled, avg_pooled],
    ["Original (8x8)", "Max Pool (4x4)", "Avg Pool (4x4)"]):
    ax.imshow(data, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title); ax.axis("off")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if data[i,j] < 0.5 else "black")
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Simple CNN Demo",
      slug: "simple-cnn-demo",
      description:
        "Build and train a small convolutional neural network on image classification with PyTorch",
      isDeepLearning: true,
      markdownContent: `# Simple CNN Demo

Now we put convolutions, activations, and pooling together into a complete **Convolutional Neural Network** (CNN) and train it on an image classification task.

## CNN Architecture

A typical CNN follows the pattern: **Conv → ReLU → Pool → Fully Connected**. Each convolutional layer learns $F$ filters, producing $F$ feature maps. ReLU introduces non-linearity, and pooling reduces spatial dimensions. Finally, the feature maps are flattened into a vector and passed through fully connected (dense) layers for classification.

For an input image of size $H \\times W \\times C_{in}$ and a convolutional layer with $F$ filters of size $K \\times K$:

$$
\\text{Output shape} = \\left\\lfloor \\frac{H - K + 2P}{S} \\right\\rfloor + 1 \\;\\times\\; \\left\\lfloor \\frac{W - K + 2P}{S} \\right\\rfloor + 1 \\;\\times\\; F
$$

## Training with Cross-Entropy Loss

For multi-class classification, we minimize the **cross-entropy loss** between predicted probabilities $\\hat{y}$ and true labels $y$:

$$
\\mathcal{L} = -\\sum_{c=1}^{C} y_c \\log(\\hat{y}_c)
$$

The network learns filter weights through backpropagation — gradients flow backward through the pooling, ReLU, and convolution layers, updating each kernel to detect features that improve classification accuracy.

## Building the Model

Below we build a small CNN with two convolutional layers and train it on synthetic image data. The architecture is:
- **Conv1**: 1 → 8 filters (3×3) → ReLU → MaxPool(2×2)
- **Conv2**: 8 → 16 filters (3×3) → ReLU → MaxPool(2×2)
- **FC**: Flatten → 64 units → ReLU → 4 classes

Run the code to train the CNN and watch the loss decrease over epochs.`,
      codeSnippet: `import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device(os.environ.get("ML_CATALOGUE_DEVICE", "cpu"))
print(f"Using device: {device}")
torch.manual_seed(42); np.random.seed(42)

# Synthetic dataset: 4 classes of 16x16 images with distinct patterns
n_per_class = 150
imgs, labels = [], []
for c in range(4):
    for _ in range(n_per_class):
        img = np.random.rand(16, 16).astype(np.float32) * 0.2
        if c == 0: img[2:6, 2:14] += 0.8     # top bar
        elif c == 1: img[10:14, 2:14] += 0.8  # bottom bar
        elif c == 2: img[2:14, 2:6] += 0.8    # left bar
        else: img[2:14, 10:14] += 0.8         # right bar
        imgs.append(img); labels.append(c)
X = torch.tensor(np.array(imgs)).unsqueeze(1).to(device)  # (N,1,16,16)
y = torch.tensor(labels).to(device)
perm = torch.randperm(len(y)); X, y = X[perm], y[perm]
split = int(0.8 * len(y))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, 4))
    def forward(self, x): return self.classifier(self.features(x))

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()
losses = []
for epoch in range(30):
    model.train(); optimizer.zero_grad()
    loss = criterion(model(X_train), y_train); loss.backward(); optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0: print(f"Epoch {epoch+1:>2d}: loss = {loss.item():.4f}")

model.eval()
with torch.no_grad():
    acc = (model(X_test).argmax(1) == y_test).float().mean().item()
print(f"Test accuracy: {acc:.1%}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(losses, color="steelblue", lw=2)
ax.set(xlabel="Epoch", ylabel="Training Loss", title="CNN Training Loss Curve")
ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()`,
      codeLanguage: "python",
    },
  ],
};
