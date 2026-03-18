import type { Chapter } from "../types";

const vaeMarkdown = `
# Variational Autoencoders (VAE)

A **Variational Autoencoder** (Kingma & Welling, 2014) is a generative model that learns a low-dimensional **latent representation** of data and can generate new samples by decoding points from that latent space.

## Autoencoder Recap

A standard autoencoder has two parts:
- **Encoder** $f_\\phi: x \\to z$ — compresses input $x$ into a latent code $z$
- **Decoder** $g_\\theta: z \\to \\hat{x}$ — reconstructs the input from $z$

The problem: the latent space of a regular autoencoder has no structure — nearby points in $z$-space may decode to very different outputs, making it useless for generation.

## The VAE Idea

A VAE imposes **probabilistic structure** on the latent space. Instead of encoding $x$ to a single point $z$, the encoder outputs the parameters of a distribution:

$$q_\\phi(z \\mid x) = \\mathcal{N}(z; \\mu_\\phi(x), \\sigma^2_\\phi(x) I)$$

We then **sample** $z$ from this distribution and decode it. The prior over $z$ is a standard normal:

$$p(z) = \\mathcal{N}(0, I)$$

## The Reparameterization Trick

We cannot backpropagate through a random sampling operation. The **reparameterization trick** solves this by expressing the sample as a deterministic function of the parameters plus noise:

$$z = \\mu_\\phi(x) + \\sigma_\\phi(x) \\odot \\varepsilon, \\quad \\varepsilon \\sim \\mathcal{N}(0, I)$$

Now gradients flow through $\\mu$ and $\\sigma$ while $\\varepsilon$ is treated as a constant input.

## VAE Loss Function (ELBO)

The VAE is trained by maximising the **Evidence Lower Bound (ELBO)**:

$$\\mathcal{L}(\\theta, \\phi; x) = \\underbrace{\\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)]}_{\\text{Reconstruction}} - \\underbrace{D_{\\text{KL}}(q_\\phi(z|x) \\| p(z))}_{\\text{KL Divergence}}$$

- **Reconstruction term**: measures how well the decoder reconstructs $x$ from $z$ (typically BCE or MSE loss)
- **KL divergence term**: regularises the encoder to keep $q_\\phi(z|x)$ close to the prior $p(z) = \\mathcal{N}(0, I)$

For Gaussian $q$ and $p$, the KL divergence has a closed-form solution:

$$D_{\\text{KL}} = -\\frac{1}{2} \\sum_{j=1}^{d} \\left(1 + \\log \\sigma_j^2 - \\mu_j^2 - \\sigma_j^2\\right)$$

## Why This Matters

VAEs give us a smooth, continuous latent space where interpolation is meaningful — walking between two points in latent space produces smooth transitions in data space. Run the code to train a VAE on MNIST and visualise reconstructions and latent space interpolation.
`;

const vaeCode = `from ml_catalogue_runtime import MODE, DEVICE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device(DEVICE)
QUICK = MODE == "quick"

# ============================================================
# Load MNIST data (subset for speed)
# ============================================================
def load_mnist_subset(n_train=5000, n_test=500):
    """Generate synthetic MNIST-like data for the sandbox demo."""
    torch.manual_seed(42)
    # Create digit-like patterns: each "digit" class gets a distinct template
    x_train = torch.zeros(n_train, 28, 28)
    x_test = torch.zeros(n_test, 28, 28)
    y_train = torch.randint(0, 10, (n_train,))
    y_test = torch.randint(0, 10, (n_test,))
    for x_data, y_data in [(x_train, y_train), (x_test, y_test)]:
        for i in range(len(x_data)):
            digit = y_data[i].item()
            # Each digit class gets a unique spatial pattern
            row_start = (digit % 5) * 4 + 2
            col_start = (digit // 5) * 10 + 4
            x_data[i, row_start:row_start+6, col_start:col_start+10] = 0.8
            x_data[i] += torch.rand(28, 28) * 0.2  # add noise
            x_data[i].clamp_(0, 1)
    return x_train, y_train, x_test, y_test

n_train = 2000 if QUICK else 10000
n_test = 200 if QUICK else 1000
x_train, y_train, x_test, y_test = load_mnist_subset(n_train, n_test)
x_train_flat = x_train.view(-1, 784).to(device)
x_test_flat = x_test.view(-1, 784).to(device)

print(f"=== Variational Autoencoder on MNIST ===")
print(f"Training samples: {n_train}, Test samples: {n_test}\\n")

# ============================================================
# VAE Model
# ============================================================
LATENT_DIM = 2  # 2D for visualization

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=2):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld

# ============================================================
# Training
# ============================================================
torch.manual_seed(42)
model = VAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

batch_size = 128
n_epochs = 10 if QUICK else 30
train_loader = DataLoader(TensorDataset(x_train_flat), batch_size=batch_size, shuffle=True)

train_losses = []
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for (batch_x,) in train_loader:
        recon, mu, logvar = model(batch_x)
        loss, bce, kld = vae_loss(recon, batch_x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(x_train_flat)
    train_losses.append(avg_loss)
    if (epoch + 1) % max(1, n_epochs // 5) == 0:
        print(f"  Epoch {epoch+1:>3}/{n_epochs}  loss={avg_loss:.2f}")

# ============================================================
# Visualization
# ============================================================
model.eval()
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1) Reconstructions
ax = axes[0, 0]
with torch.no_grad():
    test_sample = x_test_flat[:8]
    recon, _, _ = model(test_sample)
n_show = 8
combined = torch.zeros(2 * n_show, 784)
for i in range(n_show):
    combined[2 * i] = test_sample[i].cpu()
    combined[2 * i + 1] = recon[i].cpu()
grid = combined.view(2 * n_show, 28, 28).numpy()
mosaic = np.zeros((28 * 2, 28 * n_show))
for i in range(n_show):
    mosaic[0:28, i * 28:(i + 1) * 28] = grid[2 * i]
    mosaic[28:56, i * 28:(i + 1) * 28] = grid[2 * i + 1]
ax.imshow(mosaic, cmap='gray')
ax.set_title('Reconstructions (top: original, bottom: reconstructed)', fontsize=11)
ax.axis('off')

# 2) Latent space colored by digit
ax = axes[0, 1]
with torch.no_grad():
    mu_all, _ = model.encode(x_test_flat)
    mu_np = mu_all.cpu().numpy()
    labels_np = y_test.numpy()
scatter = ax.scatter(mu_np[:, 0], mu_np[:, 1], c=labels_np, cmap='tab10', s=10, alpha=0.7)
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_title('Latent Space (colored by digit)', fontsize=11)
plt.colorbar(scatter, ax=ax, ticks=range(10))
ax.grid(True, alpha=0.3)

# 3) Latent space interpolation (grid of decoded samples)
ax = axes[1, 0]
n_grid = 15
z1 = np.linspace(-3, 3, n_grid)
z2 = np.linspace(-3, 3, n_grid)
canvas = np.zeros((28 * n_grid, 28 * n_grid))
with torch.no_grad():
    for i, z2_val in enumerate(reversed(z2)):
        for j, z1_val in enumerate(z1):
            z = torch.tensor([[z1_val, z2_val]], dtype=torch.float32).to(device)
            decoded = model.decode(z).cpu().view(28, 28).numpy()
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = decoded
ax.imshow(canvas, cmap='gray')
ax.set_title('Latent Space Grid (decoded samples)', fontsize=11)
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_xticks(np.linspace(0, 28 * n_grid, 5))
ax.set_xticklabels([f'{v:.1f}' for v in np.linspace(-3, 3, 5)])
ax.set_yticks(np.linspace(0, 28 * n_grid, 5))
ax.set_yticklabels([f'{v:.1f}' for v in np.linspace(3, -3, 5)])

# 4) Training loss curve
ax = axes[1, 1]
ax.plot(train_losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Average Loss (BCE + KLD)')
ax.set_title('VAE Training Loss', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output.png', dpi=100)
print("\\nVAE visualisation saved to output.png")
`;

const ganMarkdown = `
# Generative Adversarial Networks (GAN)

A **Generative Adversarial Network** (Goodfellow et al., 2014) trains two neural networks in competition: a **generator** that creates fake data and a **discriminator** that tries to distinguish real data from fakes.

## Architecture

- **Generator** $G_\\theta: z \\to x_{\\text{fake}}$ — takes random noise $z \\sim p(z)$ (typically $\\mathcal{N}(0, I)$) and produces a synthetic sample
- **Discriminator** $D_\\phi: x \\to [0, 1]$ — receives either a real sample or a fake sample and outputs the probability that the input is real

## The Minimax Game

The two networks play a **minimax game**:

$$\\min_G \\max_D \\; V(D, G) = \\mathbb{E}_{x \\sim p_{\\text{data}}}[\\log D(x)] + \\mathbb{E}_{z \\sim p(z)}[\\log(1 - D(G(z)))]$$

- The discriminator $D$ wants to **maximise** $V$: assign high probability to real data and low probability to fakes
- The generator $G$ wants to **minimise** $V$: fool the discriminator into assigning high probability to its fakes

At the Nash equilibrium, $G$ produces samples indistinguishable from real data and $D$ outputs $\\frac{1}{2}$ for everything.

## Training Procedure

In practice, we alternate between two gradient steps:

**1. Update Discriminator** (maximize $V$ w.r.t. $\\phi$):
$$\\nabla_\\phi \\frac{1}{m} \\sum_{i=1}^{m} \\left[\\log D_\\phi(x^{(i)}) + \\log(1 - D_\\phi(G_\\theta(z^{(i)})))\\right]$$

**2. Update Generator** (minimize $V$ w.r.t. $\\theta$):

In practice, instead of minimizing $\\log(1 - D(G(z)))$ which has vanishing gradients early in training, we maximize:
$$\\nabla_\\theta \\frac{1}{m} \\sum_{i=1}^{m} \\log D_\\phi(G_\\theta(z^{(i)}))$$

## Mode Collapse

A common failure mode is **mode collapse**: the generator learns to produce only a small set of outputs that fool the discriminator, ignoring the diversity of the real data distribution.

For example, if trained on MNIST, the generator might only produce "1"s because those are easiest to fake, rather than learning all 10 digits.

Mitigation strategies include:
- **Mini-batch discrimination** — let $D$ look at batches to detect lack of diversity
- **Feature matching** — train $G$ to match statistics of intermediate $D$ features
- **Wasserstein loss** — replace the JS divergence objective with Earth Mover distance

## Why This Matters

GANs can produce remarkably realistic synthetic data and are the basis for style transfer, image super-resolution, and data augmentation. Run the code to train a simple GAN on MNIST and watch the quality of generated digits improve across training.
`;

const ganCode = `from ml_catalogue_runtime import MODE, DEVICE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device(DEVICE)
QUICK = MODE == "quick"

# ============================================================
# Load MNIST data (subset)
# ============================================================
def load_mnist_subset(n_train=5000):
    """Generate synthetic MNIST-like data for the sandbox demo."""
    torch.manual_seed(42)
    x_train = torch.zeros(n_train, 28, 28)
    labels = torch.randint(0, 10, (n_train,))
    for i in range(n_train):
        digit = labels[i].item()
        row_start = (digit % 5) * 4 + 2
        col_start = (digit // 5) * 10 + 4
        x_train[i, row_start:row_start+6, col_start:col_start+10] = 0.8
        x_train[i] += torch.rand(28, 28) * 0.2
        x_train[i].clamp_(0, 1)
    return x_train

n_train = 3000 if QUICK else 10000
x_train = load_mnist_subset(n_train)
x_train_flat = x_train.view(-1, 784).to(device)

print(f"=== GAN on MNIST ===")
print(f"Training samples: {n_train}\\n")

# ============================================================
# Generator and Discriminator
# ============================================================
LATENT_DIM = 64

class Generator(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# Training
# ============================================================
torch.manual_seed(42)
G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

batch_size = 128
n_epochs = 30 if QUICK else 80
train_loader = DataLoader(TensorDataset(x_train_flat), batch_size=batch_size, shuffle=True)

g_losses, d_losses = [], []
# Store snapshots of generated images at different epochs
snapshot_epochs = [0, n_epochs // 4, n_epochs // 2, 3 * n_epochs // 4, n_epochs - 1]
snapshots = {}
fixed_noise = torch.randn(16, LATENT_DIM, device=device)

for epoch in range(n_epochs):
    g_loss_epoch, d_loss_epoch = 0, 0
    n_batches = 0

    for (real_batch,) in train_loader:
        bs = real_batch.size(0)
        real_label = torch.ones(bs, 1, device=device) * 0.9  # label smoothing
        fake_label = torch.zeros(bs, 1, device=device)

        # --- Train Discriminator ---
        z = torch.randn(bs, LATENT_DIM, device=device)
        fake = G(z).detach()

        d_real = D(real_batch)
        d_fake = D(fake)
        loss_d = criterion(d_real, real_label) + criterion(d_fake, fake_label)

        opt_D.zero_grad()
        loss_d.backward()
        opt_D.step()

        # --- Train Generator ---
        z = torch.randn(bs, LATENT_DIM, device=device)
        fake = G(z)
        d_fake = D(fake)
        loss_g = criterion(d_fake, torch.ones(bs, 1, device=device))

        opt_G.zero_grad()
        loss_g.backward()
        opt_G.step()

        g_loss_epoch += loss_g.item()
        d_loss_epoch += loss_d.item()
        n_batches += 1

    g_losses.append(g_loss_epoch / n_batches)
    d_losses.append(d_loss_epoch / n_batches)

    # Save snapshot
    if epoch in snapshot_epochs:
        with torch.no_grad():
            samples = G(fixed_noise).cpu().view(-1, 28, 28).numpy()
        snapshots[epoch] = samples

    if (epoch + 1) % max(1, n_epochs // 5) == 0:
        print(f"  Epoch {epoch+1:>3}/{n_epochs}  D_loss={d_losses[-1]:.3f}  G_loss={g_losses[-1]:.3f}")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Top row: generated samples at different training stages
for idx, ep in enumerate(sorted(snapshots.keys())[:3]):
    ax = axes[0, idx]
    imgs = snapshots[ep]
    grid = np.zeros((4 * 28, 4 * 28))
    for i in range(4):
        for j in range(4):
            grid[i*28:(i+1)*28, j*28:(j+1)*28] = imgs[i*4+j]
    ax.imshow(grid, cmap='gray')
    ax.set_title(f'Epoch {ep + 1}', fontsize=11)
    ax.axis('off')

# Bottom-left: later snapshots
later_keys = sorted(snapshots.keys())[3:]
for idx, ep in enumerate(later_keys[:2]):
    ax = axes[1, idx]
    imgs = snapshots[ep]
    grid = np.zeros((4 * 28, 4 * 28))
    for i in range(4):
        for j in range(4):
            grid[i*28:(i+1)*28, j*28:(j+1)*28] = imgs[i*4+j]
    ax.imshow(grid, cmap='gray')
    ax.set_title(f'Epoch {ep + 1}', fontsize=11)
    ax.axis('off')

# Bottom-right: loss curves
ax = axes[1, 2]
ax.plot(g_losses, label='Generator', linewidth=2)
ax.plot(d_losses, label='Discriminator', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('GAN Training Losses', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('GAN Training Progress on MNIST', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output.png', dpi=100)
print("\\nGAN visualisation saved to output.png")
`;

const diffusionMarkdown = `
# Diffusion Models

**Diffusion models** (Sohl-Dickstein et al., 2015; Ho et al., 2020) generate data by learning to reverse a gradual noising process. Starting from pure noise, the model iteratively denoises to produce a clean sample.

## Forward Diffusion Process

Given a data sample $x_0$, the **forward process** gradually adds Gaussian noise over $T$ steps:

$$q(x_t \\mid x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1 - \\beta_t}\\, x_{t-1},\\; \\beta_t I)$$

where $\\beta_1, \\dots, \\beta_T$ is a **noise schedule** (small positive values increasing over time).

A key property: we can sample $x_t$ directly from $x_0$ without iterating through all steps. Define $\\alpha_t = 1 - \\beta_t$ and $\\bar{\\alpha}_t = \\prod_{s=1}^{t} \\alpha_s$:

$$q(x_t \\mid x_0) = \\mathcal{N}(x_t; \\sqrt{\\bar{\\alpha}_t}\\, x_0,\\; (1 - \\bar{\\alpha}_t) I)$$

$$x_t = \\sqrt{\\bar{\\alpha}_t}\\, x_0 + \\sqrt{1 - \\bar{\\alpha}_t}\\, \\varepsilon, \\quad \\varepsilon \\sim \\mathcal{N}(0, I)$$

As $t \\to T$, $\\bar{\\alpha}_T \\approx 0$ and $x_T$ is nearly pure Gaussian noise.

## Reverse Process (Denoising)

The **reverse process** learns to undo the noise, step by step:

$$p_\\theta(x_{t-1} \\mid x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t, t),\\; \\sigma_t^2 I)$$

A neural network predicts the mean $\\mu_\\theta(x_t, t)$. In DDPM (Ho et al., 2020), the network actually predicts the noise $\\varepsilon_\\theta(x_t, t)$, and the mean is derived as:

$$\\mu_\\theta(x_t, t) = \\frac{1}{\\sqrt{\\alpha_t}} \\left(x_t - \\frac{\\beta_t}{\\sqrt{1 - \\bar{\\alpha}_t}} \\varepsilon_\\theta(x_t, t)\\right)$$

## Training Objective

The simplified DDPM loss trains the network to predict the noise added at each step:

$$L = \\mathbb{E}_{t, x_0, \\varepsilon}\\!\\left[\\|\\varepsilon - \\varepsilon_\\theta(x_t, t)\\|^2\\right]$$

where $t \\sim \\text{Uniform}(1, T)$, $x_0 \\sim q(x_0)$, and $\\varepsilon \\sim \\mathcal{N}(0, I)$.

## Sampling (Generation)

To generate a new sample:
1. Sample $x_T \\sim \\mathcal{N}(0, I)$
2. For $t = T, T-1, \\dots, 1$:
   - Predict noise: $\\hat{\\varepsilon} = \\varepsilon_\\theta(x_t, t)$
   - Compute $x_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}}(x_t - \\frac{\\beta_t}{\\sqrt{1-\\bar{\\alpha}_t}} \\hat{\\varepsilon}) + \\sigma_t z$, where $z \\sim \\mathcal{N}(0,I)$

## Why This Matters

Diffusion models are the engine behind modern image generators (DALL-E, Stable Diffusion, Midjourney). Run the code to see a simplified diffusion model learn to denoise 2D data — watch it recover structure from pure noise step by step.
`;

const diffusionCode = `from ml_catalogue_runtime import MODE, DEVICE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device(DEVICE)
QUICK = MODE == "quick"

# ============================================================
# Generate 2D dataset: concentric circles
# ============================================================
def make_circles(n_samples=2000):
    """Generate 2D concentric circles dataset."""
    np.random.seed(42)
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    # Outer circle
    theta_out = np.random.uniform(0, 2 * np.pi, n_outer)
    r_out = 2.0 + np.random.normal(0, 0.08, n_outer)
    x_out = np.stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)], axis=1)

    # Inner circle
    theta_in = np.random.uniform(0, 2 * np.pi, n_inner)
    r_in = 0.8 + np.random.normal(0, 0.08, n_inner)
    x_in = np.stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)], axis=1)

    return np.vstack([x_out, x_in]).astype(np.float32)

n_samples = 2000 if QUICK else 5000
data = make_circles(n_samples)
data_tensor = torch.tensor(data, device=device)

print(f"=== Diffusion Model on 2D Concentric Circles ===")
print(f"Data samples: {n_samples}\\n")

# ============================================================
# Diffusion schedule
# ============================================================
T = 50 if QUICK else 100  # number of diffusion steps
beta_start, beta_end = 1e-4, 0.02
betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def forward_diffusion(x0, t, noise=None):
    """Add noise to x0 at timestep t."""
    if noise is None:
        noise = torch.randn_like(x0)
    ab = alpha_bar[t].view(-1, 1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise

# ============================================================
# Denoising network: predicts noise given (x_t, t)
# ============================================================
class NoisePredictor(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128, time_emb_dim=32):
        super().__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t):
        t_norm = t.float().view(-1, 1) / T
        t_emb = self.time_emb(t_norm)
        inp = torch.cat([x, t_emb], dim=1)
        return self.net(inp)

# ============================================================
# Training
# ============================================================
torch.manual_seed(42)
model = NoisePredictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

batch_size = 256
n_epochs = 80 if QUICK else 200
losses = []

for epoch in range(n_epochs):
    perm = torch.randperm(len(data_tensor), device=device)
    epoch_loss = 0
    n_batches = 0

    for i in range(0, len(data_tensor), batch_size):
        batch = data_tensor[perm[i:i + batch_size]]
        t = torch.randint(0, T, (len(batch),), device=device)
        noise = torch.randn_like(batch)
        x_t, _ = forward_diffusion(batch, t, noise)

        pred_noise = model(x_t, t)
        loss = nn.functional.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    losses.append(epoch_loss / n_batches)
    if (epoch + 1) % max(1, n_epochs // 5) == 0:
        print(f"  Epoch {epoch+1:>3}/{n_epochs}  loss={losses[-1]:.5f}")

# ============================================================
# Sampling (reverse diffusion)
# ============================================================
@torch.no_grad()
def sample(model, n_samples=500):
    x = torch.randn(n_samples, 2, device=device)
    trajectory = [x.cpu().numpy()]

    for t_idx in reversed(range(T)):
        t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
        pred_noise = model(x, t)

        beta_t = betas[t_idx]
        alpha_t = alphas[t_idx]
        alpha_bar_t = alpha_bar[t_idx]

        # Compute mean
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        )

        if t_idx > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean

        if t_idx % max(1, T // 8) == 0 or t_idx == 0:
            trajectory.append(x.cpu().numpy())

    return trajectory

print("\\nGenerating samples via reverse diffusion...")
trajectory = sample(model, n_samples=800)

# ============================================================
# Visualization
# ============================================================
n_steps_to_show = min(len(trajectory), 6)
step_indices = np.linspace(0, len(trajectory) - 1, n_steps_to_show, dtype=int)

fig, axes = plt.subplots(2, n_steps_to_show, figsize=(3.5 * n_steps_to_show, 7))

# Top row: forward diffusion
steps_fwd = np.linspace(0, T - 1, n_steps_to_show, dtype=int)
for idx, t_val in enumerate(steps_fwd):
    ax = axes[0, idx]
    t_tensor = torch.full((len(data_tensor),), t_val, device=device, dtype=torch.long)
    x_noisy, _ = forward_diffusion(data_tensor, t_tensor)
    pts = x_noisy.cpu().numpy()
    ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.4, c='steelblue')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title(f'Forward t={t_val}', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    if idx == 0:
        ax.set_ylabel('Forward\\n(adding noise)', fontsize=10)

# Bottom row: reverse diffusion (sampling)
for idx, step_i in enumerate(step_indices):
    ax = axes[1, idx]
    pts = trajectory[step_i]
    ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.4, c='coral')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    if step_i == 0:
        ax.set_title('Noise (start)', fontsize=10)
    elif step_i == len(trajectory) - 1:
        ax.set_title('Generated (final)', fontsize=10)
    else:
        ax.set_title(f'Denoise step {step_i}', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    if idx == 0:
        ax.set_ylabel('Reverse\\n(denoising)', fontsize=10)

plt.suptitle('Diffusion Model: Forward & Reverse Process', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output.png', dpi=100)
print("Diffusion model visualisation saved to output.png")
`;

export const generativeModels: Chapter = {
  title: "Generative Models",
  slug: "generative-models",
  pages: [
    {
      title: "Variational Autoencoders (VAE)",
      slug: "variational-autoencoders",
      description:
        "Latent space, encoder/decoder architecture, reparameterization trick, KL divergence loss, and MNIST reconstruction demo",
      isDeepLearning: true,
      markdownContent: vaeMarkdown,
      codeSnippet: vaeCode,
      codeLanguage: "python",
    },
    {
      title: "Generative Adversarial Networks (GAN)",
      slug: "generative-adversarial-networks",
      description:
        "Generator/discriminator architecture, minimax game, training dynamics, mode collapse, and MNIST generation demo",
      isDeepLearning: true,
      markdownContent: ganMarkdown,
      codeSnippet: ganCode,
      codeLanguage: "python",
    },
    {
      title: "Diffusion Models",
      slug: "diffusion-models",
      description:
        "Forward/reverse diffusion process, denoising score matching, DDPM fundamentals, and 2D data denoising demo",
      isDeepLearning: true,
      markdownContent: diffusionMarkdown,
      codeSnippet: diffusionCode,
      codeLanguage: "python",
    },
  ],
};
