import type { Chapter } from "../types";

export const calculusOptimisation: Chapter = {
  title: "Calculus & Optimisation",
  slug: "calculus-optimisation",
  pages: [
    {
      title: "Derivatives & Gradients",
      slug: "derivatives-gradients",
      description: "Rates of change, partial derivatives, and gradient vectors",
      markdownContent: `# Derivatives & Gradients

The **derivative** of a function $f(x)$ at a point $x$ measures its instantaneous rate of change — geometrically, it is the slope of the tangent line to the curve at that point:

$$
f'(x) = \\lim_{h \\to 0} \\frac{f(x + h) - f(x)}{h}
$$

For example, if $f(x) = x^2$, then $f'(x) = 2x$. At $x = 3$ the slope is $6$: the function is increasing at a rate of $6$ units per unit step.

## Partial Derivatives

When a function depends on multiple variables, $f(x_1, x_2, \\ldots, x_n)$, we take a **partial derivative** with respect to one variable while holding the others constant:

$$
\\frac{\\partial f}{\\partial x_i} = \\lim_{h \\to 0} \\frac{f(x_1, \\ldots, x_i + h, \\ldots, x_n) - f(x_1, \\ldots, x_n)}{h}
$$

For example, if $f(x, y) = x^2 + 3xy + y^2$, then $\\frac{\\partial f}{\\partial x} = 2x + 3y$ and $\\frac{\\partial f}{\\partial y} = 3x + 2y$.

## The Gradient Vector

The **gradient** collects all partial derivatives into a single vector:

$$
\\nabla f = \\left( \\frac{\\partial f}{\\partial x_1}, \\frac{\\partial f}{\\partial x_2}, \\ldots, \\frac{\\partial f}{\\partial x_n} \\right)
$$

The gradient has two critical properties:
- It points in the direction of **steepest ascent** of $f$.
- Its magnitude $\\|\\nabla f\\|$ tells you how steep that ascent is.

In machine learning, we want to **minimise** a loss function, so we move in the **opposite** direction of the gradient — this is the core idea behind gradient descent.

Run the code below to visualise a function, its derivative, tangent lines, and gradient vectors on a contour plot.`,
      codeSnippet: `import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1D: Plot f(x) = x^2 and its derivative ---
x = np.linspace(-3, 3, 200)
f = x ** 2
f_prime = 2 * x

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: function with tangent lines
axes[0].plot(x, f, "b-", linewidth=2, label="$f(x) = x^2$")
for x0, color in [(-2, "red"), (0, "green"), (1.5, "purple")]:
    slope = 2 * x0
    tangent = slope * (x - x0) + x0 ** 2
    axes[0].plot(x, tangent, "--", color=color, label=f"Tangent at x={x0} (slope={slope})")
    axes[0].plot(x0, x0 ** 2, "o", color=color, markersize=8)
axes[0].set_ylim(-2, 9)
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].set_title("Function and Tangent Lines")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Right plot: gradient vectors on a 2D function f(x,y) = x^2 + y^2
xg = np.linspace(-3, 3, 30)
yg = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(xg, yg)
Z = X ** 2 + Y ** 2

# Gradient: (2x, 2y)
xq = np.linspace(-2.5, 2.5, 8)
yq = np.linspace(-2.5, 2.5, 8)
XQ, YQ = np.meshgrid(xq, yq)
U, V = 2 * XQ, 2 * YQ

axes[1].contour(X, Y, Z, levels=15, cmap="viridis")
axes[1].quiver(XQ, YQ, -U, -V, color="red", alpha=0.7)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("Negative Gradient on $f(x,y) = x^2 + y^2$")
axes[1].set_aspect("equal")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("derivatives_gradients.png", dpi=100, bbox_inches="tight")
plt.show()
print("Left: f(x)=x² with tangent lines at x=-2, 0, 1.5")
print("Right: negative gradient arrows point toward the minimum")`,
      codeLanguage: "python",
    },
    {
      title: "Chain Rule",
      slug: "chain-rule",
      description: "Composing derivatives and the backbone of backpropagation",
      markdownContent: `# Chain Rule

The **chain rule** tells us how to differentiate a **composite function** — a function built by plugging one function into another.

If $y = f(g(x))$, then:

$$
\\frac{dy}{dx} = f'(g(x)) \\cdot g'(x)
$$

Or equivalently, using Leibniz notation with $u = g(x)$:

$$
\\frac{dy}{dx} = \\frac{dy}{du} \\cdot \\frac{du}{dx}
$$

This extends to longer chains. If $y = f(g(h(x)))$, then:

$$
\\frac{dy}{dx} = f'(g(h(x))) \\cdot g'(h(x)) \\cdot h'(x)
$$

## Example

Let $y = (3x + 1)^2$. We can decompose this as $y = u^2$ where $u = 3x + 1$:

$$
\\frac{dy}{dx} = \\frac{dy}{du} \\cdot \\frac{du}{dx} = 2u \\cdot 3 = 6(3x + 1)
$$

## Why It Matters: Backpropagation

**Backpropagation** — the algorithm that trains neural networks — is simply the chain rule applied systematically through a **computational graph**. Each node in the graph computes a simple operation, and the chain rule lets us propagate the gradient of the loss backward through every node to every parameter.

Consider a simple computational graph:

$$
x \\xrightarrow{\\times w} z \\xrightarrow{\\sigma} a \\xrightarrow{L} \\text{loss}
$$

The gradient of the loss with respect to $w$ is:

$$
\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial a} \\cdot \\frac{\\partial a}{\\partial z} \\cdot \\frac{\\partial z}{\\partial w}
$$

Each factor is a simple local derivative, but chaining them together gives us the global gradient. This decomposition is what makes training deep networks computationally tractable.

Run the code below to manually apply the chain rule and verify against numerical differentiation.`,
      codeSnippet: `import numpy as np

# --- Chain Rule Example ---
# y = sin(x^2)  =>  y = f(g(x)) where f(u) = sin(u), g(x) = x^2
# dy/dx = f'(g(x)) * g'(x) = cos(x^2) * 2x

def g(x):
    return x ** 2

def f(u):
    return np.sin(u)

def y(x):
    return f(g(x))

# Analytical derivative via chain rule
def dy_dx_analytical(x):
    return np.cos(g(x)) * 2 * x  # cos(x^2) * 2x

# Numerical derivative for verification
def dy_dx_numerical(x, h=1e-7):
    return (y(x + h) - y(x - h)) / (2 * h)

x_test = 1.5
print("=== Chain Rule: y = sin(x²) ===")
print(f"At x = {x_test}:")
print(f"  Analytical dy/dx = {dy_dx_analytical(x_test):.8f}")
print(f"  Numerical  dy/dx = {dy_dx_numerical(x_test):.8f}")
print(f"  Difference       = {abs(dy_dx_analytical(x_test) - dy_dx_numerical(x_test)):.2e}")

# --- Computational Graph: Backpropagation Demo ---
print("\\n=== Computational Graph: Forward & Backward Pass ===")
print("Graph: x -> (*w) -> z -> (sigmoid) -> a -> (MSE loss vs target) -> L")

# Forward pass
x = 2.0
w = 0.5
target = 0.8

z = x * w                         # z = x * w
a = 1 / (1 + np.exp(-z))          # a = sigmoid(z)
L = (a - target) ** 2             # L = (a - target)^2

print(f"\\nForward pass:")
print(f"  x={x}, w={w}, target={target}")
print(f"  z = x*w = {z}")
print(f"  a = sigmoid(z) = {a:.6f}")
print(f"  L = (a - target)² = {L:.6f}")

# Backward pass (chain rule at each node)
dL_da = 2 * (a - target)          # dL/da
da_dz = a * (1 - a)               # sigmoid derivative
dz_dw = x                         # d(x*w)/dw = x

dL_dw = dL_da * da_dz * dz_dw     # chain rule

print(f"\\nBackward pass (chain rule):")
print(f"  dL/da = 2(a - target) = {dL_da:.6f}")
print(f"  da/dz = a(1-a)        = {da_dz:.6f}")
print(f"  dz/dw = x             = {dz_dw}")
print(f"  dL/dw = dL/da * da/dz * dz/dw = {dL_dw:.6f}")

# Numerical verification
h = 1e-7
z_h = x * (w + h)
a_h = 1 / (1 + np.exp(-z_h))
L_h = (a_h - target) ** 2
dL_dw_numerical = (L_h - L) / h
print(f"\\n  Numerical dL/dw = {dL_dw_numerical:.6f}")
print(f"  Match: {np.isclose(dL_dw, dL_dw_numerical, atol=1e-5)}")`,
      codeLanguage: "python",
    },
    {
      title: "Gradient Descent & Variants",
      slug: "gradient-descent-variants",
      description:
        "Optimisation algorithms from vanilla GD to Adam",
      markdownContent: `# Gradient Descent & Variants

**Gradient descent** is the primary optimisation algorithm in machine learning. The idea is simple: to minimise a loss function $L(\\theta)$, repeatedly take steps in the direction that decreases $L$ the fastest — the negative gradient:

$$
\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)
$$

where $\\eta$ is the **learning rate**, a hyperparameter that controls the step size.

## Learning Rate

The learning rate is arguably the most important hyperparameter:
- **Too large:** steps overshoot the minimum and the loss may diverge.
- **Too small:** convergence is painfully slow and may get stuck in shallow local minima.
- **Just right:** the loss decreases steadily toward a minimum.

## Variants

### Stochastic Gradient Descent (SGD)

Instead of computing the gradient over the entire dataset (expensive), SGD estimates it from a single random sample. This is noisy but much faster per step and can help escape local minima.

### Mini-batch Gradient Descent

A compromise: compute the gradient on a small batch of $B$ samples. This reduces noise relative to SGD while remaining much cheaper than full-batch GD. Typical batch sizes: $32$, $64$, $128$, $256$.

### Momentum

Momentum accelerates SGD by accumulating a velocity vector that smooths out oscillations:

$$
v_t = \\beta v_{t-1} + \\nabla L(\\theta_t)
$$
$$
\\theta_{t+1} = \\theta_t - \\eta v_t
$$

Typical $\\beta = 0.9$. Think of a ball rolling downhill — it builds speed in consistent directions and dampens oscillations.

### Adam (Adaptive Moment Estimation)

Adam combines momentum with per-parameter adaptive learning rates. It tracks both the first moment (mean) and second moment (uncentred variance) of the gradient:

$$
m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t
$$
$$
v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2
$$

Bias-corrected estimates: $\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}$, $\\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$

$$
\\theta_{t+1} = \\theta_t - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}
$$

Default values: $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\epsilon = 10^{-8}$.

## Convergence

Gradient descent converges when:
- The gradient norm $\\|\\nabla L\\|$ becomes sufficiently small.
- The loss change between steps falls below a threshold.
- For convex functions, GD is guaranteed to converge with a small enough learning rate. Non-convex losses (common in deep learning) may converge to local minima or saddle points.

Run the code below to implement these optimisers from scratch and compare their convergence.`,
      codeSnippet: `import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Loss function: Rosenbrock-like  f(x,y) = (1-x)^2 + 10*(y-x^2)^2
def loss(params):
    x, y = params
    return (1 - x) ** 2 + 10 * (y - x ** 2) ** 2

def grad(params):
    x, y = params
    dl_dx = -2 * (1 - x) + 10 * 2 * (y - x ** 2) * (-2 * x)
    dl_dy = 10 * 2 * (y - x ** 2)
    return np.array([dl_dx, dl_dy])

# --- Optimisers ---
def vanilla_gd(lr=0.002, steps=500):
    params = np.array([-1.0, -1.0])
    path = [params.copy()]
    for _ in range(steps):
        params = params - lr * grad(params)
        path.append(params.copy())
    return np.array(path)

def momentum_gd(lr=0.002, beta=0.9, steps=500):
    params = np.array([-1.0, -1.0])
    v = np.zeros(2)
    path = [params.copy()]
    for _ in range(steps):
        v = beta * v + grad(params)
        params = params - lr * v
        path.append(params.copy())
    return np.array(path)

def adam(lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8, steps=500):
    params = np.array([-1.0, -1.0])
    m = np.zeros(2)
    v = np.zeros(2)
    path = [params.copy()]
    for t in range(1, steps + 1):
        g = grad(params)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(params.copy())
    return np.array(path)

# Run optimisers
path_gd = vanilla_gd()
path_mom = momentum_gd()
path_adam = adam()

# --- Visualisation ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: contour plot with descent paths
xc = np.linspace(-1.5, 1.5, 200)
yc = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(xc, yc)
Z = (1 - X) ** 2 + 10 * (Y - X ** 2) ** 2

axes[0].contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap="viridis", alpha=0.6)
axes[0].plot(*path_gd.T, "r.-", markersize=2, linewidth=1, label="Vanilla GD", alpha=0.8)
axes[0].plot(*path_mom.T, "b.-", markersize=2, linewidth=1, label="Momentum", alpha=0.8)
axes[0].plot(*path_adam.T, "g.-", markersize=2, linewidth=1, label="Adam", alpha=0.8)
axes[0].plot(1, 1, "k*", markersize=15, label="Minimum (1,1)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Descent Paths on Rosenbrock Function")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Right: loss vs iteration
losses_gd = [loss(p) for p in path_gd]
losses_mom = [loss(p) for p in path_mom]
losses_adam = [loss(p) for p in path_adam]

axes[1].semilogy(losses_gd, "r-", label="Vanilla GD", alpha=0.8)
axes[1].semilogy(losses_mom, "b-", label="Momentum", alpha=0.8)
axes[1].semilogy(losses_adam, "g-", label="Adam", alpha=0.8)
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Loss (log scale)")
axes[1].set_title("Convergence Comparison")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gradient_descent.png", dpi=100, bbox_inches="tight")
plt.show()

print(f"Final loss — Vanilla GD: {losses_gd[-1]:.6f}")
print(f"Final loss — Momentum:   {losses_mom[-1]:.6f}")
print(f"Final loss — Adam:       {losses_adam[-1]:.6f}")
print(f"\\nMinimum is at (1, 1) with loss = 0")`,
      codeLanguage: "python",
    },
  ],
};
