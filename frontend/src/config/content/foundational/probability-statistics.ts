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
    {
      title: "Hypothesis Testing & p-values",
      slug: "hypothesis-testing",
      description: "Null/alternative hypotheses, p-values, t-tests, chi-squared tests, and the Central Limit Theorem",
      markdownContent: `# Hypothesis Testing & p-values

**Hypothesis testing** is a framework for making decisions from data. You start with a default assumption (the null hypothesis), collect data, and ask: *"Is this data surprising enough to reject the default?"*

## Null vs. Alternative Hypothesis

- **Null hypothesis** $H_0$: There is no effect or no difference (the status quo).
- **Alternative hypothesis** $H_1$: There is an effect or a difference.

For example, if you want to test whether a new drug lowers blood pressure, $H_0$: the drug has no effect, and $H_1$: the drug lowers blood pressure.

## Type I and Type II Errors

| | $H_0$ true | $H_0$ false |
|---|---|---|
| **Reject $H_0$** | Type I error (false positive) | Correct |
| **Fail to reject $H_0$** | Correct | Type II error (false negative) |

- **Type I error rate** = $\\alpha$ (significance level, typically 0.05)
- **Type II error rate** = $\\beta$; **Power** = $1 - \\beta$

## p-value

The **p-value** is the probability of observing data at least as extreme as what was actually observed, assuming $H_0$ is true:

$$
p = P(\\text{test statistic} \\geq t_{\\text{obs}} \\mid H_0)
$$

If $p < \\alpha$, we reject $H_0$. A small p-value means the data is unlikely under the null — it does **not** mean the alternative is certainly true.

## Common Tests

**t-test** — compares means of continuous data:
- *One-sample*: Is this sample mean different from a known value?
- *Two-sample*: Do two groups have different means?
- *Paired*: Do matched pairs show a difference?

For a two-sample t-test with equal variance, the test statistic is:

$$
t = \\frac{\\bar{X}_1 - \\bar{X}_2}{s_p \\sqrt{\\frac{1}{n_1} + \\frac{1}{n_2}}}
$$

where $s_p$ is the pooled standard deviation.

**Chi-squared test** — tests independence of categorical variables:

$$
\\chi^2 = \\sum \\frac{(O_i - E_i)^2}{E_i}
$$

where $O_i$ are observed counts and $E_i$ are expected counts under independence.

## Central Limit Theorem

The **Central Limit Theorem** (CLT) states that the sampling distribution of the sample mean approaches a normal distribution as $n \\to \\infty$, regardless of the population distribution:

$$
\\bar{X}_n \\xrightarrow{d} N\\!\\left(\\mu, \\frac{\\sigma^2}{n}\\right)
$$

This is why t-tests and z-tests work — they rely on the sample mean being approximately normal for large enough $n$.

## Why This Matters for DS

Hypothesis testing is the backbone of A/B testing, feature selection, and model comparison. Understanding p-values prevents common misinterpretations that lead to bad business decisions.

Run the code to perform a two-sample t-test and chi-squared test, then visualize the test statistic distribution with the rejection region shaded.`,
      codeSnippet: `import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# === Two-Sample t-test ===
# Group A: control (mean=100), Group B: treatment (mean=105)
group_a = rng.normal(loc=100, scale=15, size=50)
group_b = rng.normal(loc=105, scale=15, size=50)

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print("=== Two-Sample t-test ===")
print(f"Group A: mean={group_a.mean():.2f}, std={group_a.std():.2f}")
print(f"Group B: mean={group_b.mean():.2f}, std={group_b.std():.2f}")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value     = {p_value:.4f}")
print(f"Result: {'Reject' if p_value < 0.05 else 'Fail to reject'} H₀ at α=0.05")

# === Chi-Squared Test ===
# Observed: rows=gender, cols=preference (A, B, C)
observed = np.array([[30, 10, 10],
                      [15, 20, 15]])
chi2, p_chi, dof, expected = stats.chi2_contingency(observed)
print(f"\\n=== Chi-Squared Test ===")
print(f"Observed:\\n{observed}")
print(f"Expected:\\n{np.round(expected, 1)}")
print(f"χ² = {chi2:.4f}, dof = {dof}, p-value = {p_chi:.4f}")
print(f"Result: {'Reject' if p_chi < 0.05 else 'Fail to reject'} H₀ at α=0.05")

# === Visualize t-distribution with rejection region ===
df = len(group_a) + len(group_b) - 2
x = np.linspace(-4, 4, 300)
y = stats.t.pdf(x, df)
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha / 2, df)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, "k-", linewidth=2, label="t-distribution")
ax.fill_between(x, y, where=(x <= -t_crit), color="red", alpha=0.3, label=f"Rejection region (α={alpha})")
ax.fill_between(x, y, where=(x >= t_crit), color="red", alpha=0.3)
ax.axvline(t_stat, color="blue", linestyle="--", linewidth=2, label=f"Observed t={t_stat:.2f}")
ax.set_xlabel("t-statistic")
ax.set_ylabel("Density")
ax.set_title("Two-Sample t-test: Distribution under H₀")
ax.legend()
plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
    {
      title: "A/B Testing Basics",
      slug: "ab-testing-basics",
      description: "Designing and analyzing A/B tests with power analysis, z-tests, and confidence intervals",
      markdownContent: `# A/B Testing Basics

**A/B testing** is a randomised controlled experiment used to compare two variants and determine which performs better on a target metric. It is the gold standard for causal inference in product and business decisions.

## How A/B Testing Works

1. **Randomly split** users into a control group (A) and a treatment group (B).
2. **Expose** each group to a different variant (e.g., old vs. new checkout page).
3. **Measure** a key metric (conversion rate, CTR, revenue per user).
4. **Analyze** whether the difference is statistically significant.

Randomisation ensures that any observed difference is caused by the treatment, not confounding factors.

## Sample Size & Power Analysis

Before running a test, you need to determine **how many users** are required. This depends on:

- **Baseline conversion rate** $p_0$
- **Minimum detectable effect** (MDE) $\\delta$
- **Significance level** $\\alpha$ (typically 0.05)
- **Power** $1 - \\beta$ (typically 0.80)

For a two-proportion z-test, the approximate sample size per group is:

$$
n \\approx \\frac{(z_{1-\\alpha/2} + z_{1-\\beta})^2 \\, [p_0(1-p_0) + p_1(1-p_1)]}{(p_1 - p_0)^2}
$$

where $p_1 = p_0 + \\delta$.

## Statistical Significance vs. Practical Significance

A result can be **statistically significant** (small p-value) but **practically insignificant** (the effect is too small to matter). Always check the **confidence interval** for the effect size, not just whether $p < 0.05$.

The confidence interval for the difference in proportions is:

$$
(\\hat{p}_1 - \\hat{p}_0) \\pm z_{1-\\alpha/2} \\sqrt{\\frac{\\hat{p}_1(1-\\hat{p}_1)}{n_1} + \\frac{\\hat{p}_0(1-\\hat{p}_0)}{n_0}}
$$

## Common Pitfalls

- **Peeking**: Checking results repeatedly inflates the false positive rate. Use sequential testing methods if you must peek.
- **Multiple comparisons**: Testing many metrics without correction increases Type I errors. Apply Bonferroni or FDR corrections.
- **Simpson's paradox**: An overall trend can reverse when data is split by subgroups. Always segment your analysis.
- **Under-powering**: Running a test with too few users means you cannot detect real effects.

## Key Metrics

| Metric | Formula |
|---|---|
| Conversion rate | $\\frac{\\text{conversions}}{\\text{visitors}}$ |
| Click-through rate (CTR) | $\\frac{\\text{clicks}}{\\text{impressions}}$ |
| Revenue per user | $\\frac{\\text{total revenue}}{\\text{users}}$ |

## Why This Matters for DS

A/B testing is one of the most common interview topics and day-to-day tasks for data scientists at tech companies. Understanding the full pipeline — from power analysis to result interpretation — is essential.

Run the code to simulate an A/B test end-to-end: generate data, run a z-test for proportions, compute the confidence interval, and calculate the required sample size.`,
      codeSnippet: `import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# === Simulate A/B Test Data ===
n_control = 1000
n_treatment = 1000
p_control = 0.10     # 10% baseline conversion
p_treatment = 0.12   # 12% treatment conversion (2pp lift)

control = rng.binomial(1, p_control, n_control)
treatment = rng.binomial(1, p_treatment, n_treatment)

p_hat_c = control.mean()
p_hat_t = treatment.mean()
lift = p_hat_t - p_hat_c

print("=== A/B Test Simulation ===")
print(f"Control:   n={n_control}, conversions={control.sum()}, rate={p_hat_c:.4f}")
print(f"Treatment: n={n_treatment}, conversions={treatment.sum()}, rate={p_hat_t:.4f}")
print(f"Observed lift: {lift:.4f} ({lift/p_hat_c:.1%} relative)")

# === Z-test for Two Proportions ===
p_pool = (control.sum() + treatment.sum()) / (n_control + n_treatment)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
z_stat = lift / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\\n=== Z-test for Proportions ===")
print(f"Pooled proportion = {p_pool:.4f}")
print(f"z-statistic = {z_stat:.4f}")
print(f"p-value     = {p_value:.4f}")
print(f"Result: {'Reject' if p_value < 0.05 else 'Fail to reject'} H₀ at α=0.05")

# === Confidence Interval ===
se_ci = np.sqrt(p_hat_t*(1-p_hat_t)/n_treatment + p_hat_c*(1-p_hat_c)/n_control)
ci_low = lift - 1.96 * se_ci
ci_high = lift + 1.96 * se_ci
print(f"\\n=== 95% Confidence Interval for Lift ===")
print(f"Lift = {lift:.4f}, CI = [{ci_low:.4f}, {ci_high:.4f}]")

# === Power Analysis: Required Sample Size ===
alpha = 0.05
power = 0.80
p0 = 0.10
mde = 0.02  # minimum detectable effect
p1 = p0 + mde

z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(power)
n_required = ((z_alpha + z_beta)**2 * (p0*(1-p0) + p1*(1-p1))) / mde**2
print(f"\\n=== Power Analysis ===")
print(f"Baseline rate: {p0}, MDE: {mde}, α={alpha}, power={power}")
print(f"Required sample size per group: {int(np.ceil(n_required))}")

# === Visualize Results ===
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart of conversion rates with CI
rates = [p_hat_c, p_hat_t]
ci_err = [1.96*np.sqrt(p_hat_c*(1-p_hat_c)/n_control),
          1.96*np.sqrt(p_hat_t*(1-p_hat_t)/n_treatment)]
bars = axes[0].bar(["Control", "Treatment"], rates, yerr=ci_err,
                    color=["steelblue", "coral"], edgecolor="white", capsize=8)
axes[0].set_ylabel("Conversion Rate")
axes[0].set_title("A/B Test: Conversion Rates (95% CI)")

# Sampling distribution under H0
x = np.linspace(-4, 4, 300)
y = stats.norm.pdf(x)
z_crit = stats.norm.ppf(1 - alpha/2)
axes[1].plot(x, y, "k-", linewidth=2)
axes[1].fill_between(x, y, where=(x <= -z_crit), color="red", alpha=0.3, label="Rejection region")
axes[1].fill_between(x, y, where=(x >= z_crit), color="red", alpha=0.3)
axes[1].axvline(z_stat, color="blue", linestyle="--", linewidth=2, label=f"Observed z={z_stat:.2f}")
axes[1].set_xlabel("z-statistic")
axes[1].set_ylabel("Density")
axes[1].set_title("Z-test: Distribution under H₀")
axes[1].legend()

plt.tight_layout()
plt.show()`,
      codeLanguage: "python",
    },
  ],
};
