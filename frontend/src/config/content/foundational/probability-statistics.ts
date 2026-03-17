import type { Chapter } from "../types";

export const probabilityStatistics: Chapter = {
  title: "Probability & Statistics",
  slug: "probability-statistics",
  pages: [
    {
      title: "Probability Distributions",
      slug: "probability-distributions",
      description: "Normal, uniform, and binomial distributions with PDF/PMF concepts",
      markdownContent: `# Probability Distributions

A **probability distribution** describes how likely each outcome of a random variable is. Distributions are the language of uncertainty in machine learning — from modeling noise in data to defining loss functions and priors.

## Discrete vs. Continuous

A **probability mass function** (PMF) applies to discrete random variables. For a binomial distribution with $n$ trials and success probability $p$, the PMF is:

$$
P(X = k) = \\binom{n}{k} p^k (1 - p)^{n-k}
$$

A **probability density function** (PDF) applies to continuous variables. The most important PDF in ML is the **normal (Gaussian) distribution**:

$$
f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} \\, e^{-\\frac{(x - \\mu)^2}{2\\sigma^2}}
$$

Here $\\mu$ is the mean (center) and $\\sigma$ is the standard deviation (spread).

## Mean and Variance

Every distribution is summarized by its **mean** $\\mu = E[X]$ and **variance** $\\sigma^2 = E[(X - \\mu)^2]$. The variance measures how spread out the values are. A **uniform distribution** on $[a, b]$ has mean $\\frac{a+b}{2}$ and variance $\\frac{(b-a)^2}{12}$.

## Why This Matters for ML

Neural network weight initialization often uses normal or uniform distributions. Understanding variance helps you diagnose exploding or vanishing gradients, and the central limit theorem explains why Gaussian assumptions work so often in practice.

Run the code to sample from each distribution and visualize the results.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Normal distribution: mean=0, std=1
normal_samples = rng.normal(loc=0, scale=1, size=1000)

# Uniform distribution: [0, 1]
uniform_samples = rng.uniform(low=0, high=1, size=1000)

# Binomial distribution: n=10, p=0.5
binomial_samples = rng.binomial(n=10, p=0.5, size=1000)

# Print summary statistics
for name, data in [("Normal(0,1)", normal_samples),
                    ("Uniform(0,1)", uniform_samples),
                    ("Binomial(10,0.5)", binomial_samples)]:
    print(f"{name}: mean={data.mean():.4f}, var={data.var():.4f}")

# Plot histograms
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].hist(normal_samples, bins=30, color="steelblue", edgecolor="white")
axes[0].set_title("Normal(0, 1)")
axes[1].hist(uniform_samples, bins=30, color="coral", edgecolor="white")
axes[1].set_title("Uniform(0, 1)")
axes[2].hist(binomial_samples, bins=range(12), color="seagreen", edgecolor="white")
axes[2].set_title("Binomial(10, 0.5)")
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "Bayesian Thinking",
      slug: "bayesian-thinking",
      description: "Bayes' theorem, priors, likelihoods, and posterior reasoning",
      markdownContent: `# Bayesian Thinking

**Bayes' theorem** lets you update beliefs when you observe new evidence. It is the foundation of probabilistic reasoning in machine learning — from spam filters to medical diagnosis to the training of large language models.

## Bayes' Theorem

Given a hypothesis $A$ and observed evidence $B$:

$$
P(A \\mid B) = \\frac{P(B \\mid A) \\, P(A)}{P(B)}
$$

Each term has an intuitive meaning:

- **Prior** $P(A)$ — your belief about $A$ before seeing evidence.
- **Likelihood** $P(B \\mid A)$ — how probable the evidence is if $A$ is true.
- **Posterior** $P(A \\mid B)$ — your updated belief after seeing the evidence.
- **Evidence** $P(B)$ — the total probability of the observation, which acts as a normalizing constant.

## Example: Medical Testing

Suppose a disease affects 1% of the population. A test for the disease has a 95% true-positive rate and a 5% false-positive rate. If a patient tests positive, what is the probability they actually have the disease?

Intuitively you might guess 95%, but Bayes' theorem reveals a much lower number — because the disease is rare, most positives are false positives. The prior $P(\\text{disease}) = 0.01$ heavily influences the result.

## Why This Matters for ML

Bayesian reasoning explains regularization (priors over weights), Bayesian neural networks, and probabilistic classifiers like Naive Bayes. Even when we use point estimates, understanding the Bayesian perspective helps you reason about uncertainty and overfitting.

Run the code below to walk through the medical test calculation step by step.`,
      codeSnippet: `import numpy as np

# --- Medical Test Example ---
prior_disease = 0.01          # P(disease) = 1%
prior_healthy = 1 - prior_disease

sensitivity = 0.95            # P(positive | disease)
false_positive_rate = 0.05    # P(positive | healthy)

# P(positive) via law of total probability
p_positive = (sensitivity * prior_disease
              + false_positive_rate * prior_healthy)

# Posterior: P(disease | positive)
posterior = (sensitivity * prior_disease) / p_positive

print("=== Medical Test: Bayes' Theorem ===")
print(f"Prior P(disease)        = {prior_disease:.4f}")
print(f"Sensitivity P(+|disease)= {sensitivity:.4f}")
print(f"False-pos P(+|healthy)  = {false_positive_rate:.4f}")
print(f"P(positive)             = {p_positive:.4f}")
print(f"Posterior P(disease|+)  = {posterior:.4f}")
print(f"\\nDespite a 95% sensitive test, a positive result")
print(f"only means a {posterior:.1%} chance of disease.")

# --- Updating with a second independent test ---
prior_2 = posterior  # use posterior as new prior
p_pos_2 = sensitivity * prior_2 + false_positive_rate * (1 - prior_2)
posterior_2 = (sensitivity * prior_2) / p_pos_2

print(f"\\n=== After a Second Positive Test ===")
print(f"Updated prior           = {prior_2:.4f}")
print(f"New posterior            = {posterior_2:.4f}")
print(f"Two positives -> {posterior_2:.1%} chance of disease.")`,
      codeLanguage: "python",
    },
  ],
};
