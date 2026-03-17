import type { Chapter } from "../types";

export const productionDeployment: Chapter = {
  title: "Production Deployment",
  slug: "production-deployment",
  pages: [
    {
      title: "Model Serving",
      slug: "model-serving",
      description:
        "REST API patterns, request/response design, batching strategies, and latency optimization for model inference",
      markdownContent: `# Model Serving

Deploying a trained model means wrapping it in a **serving layer** that accepts requests, runs inference, and returns predictions. A well-designed serving pipeline handles input validation, error reporting, and throughput optimization — all without requiring the caller to know anything about the model internals.

## Request / Response Design

Every prediction request should be validated before it reaches the model. A minimal serving contract includes:

1. **Input validation** — check shape, dtype, and value ranges.
2. **Prediction** — run the model's \`.predict()\` method.
3. **Output formatting** — return predictions with metadata (e.g., class labels, probabilities).
4. **Error handling** — surface clear messages for bad inputs or model failures.

A prediction endpoint typically returns a JSON-like response:

$$
\\text{response} = \\bigl\\{\\text{predictions}: \\hat{y},\\; \\text{status}: \\text{ok},\\; \\text{latency\\_ms}: t\\bigr\\}
$$

## Batching for Throughput

Serving models one request at a time wastes compute. **Batching** groups multiple requests into a single forward pass, amortising fixed overhead across $n$ samples:

$$
\\text{throughput} = \\frac{\\text{batch\\_size}}{\\text{latency}}
$$

With vectorised libraries like NumPy and scikit-learn, a batch of 64 predictions often takes barely longer than a single one. The trade-off is added latency for individual requests while the batch fills up, so production systems tune a **max batch size** and a **max wait time**.

## Latency Optimisation

Beyond batching, several techniques reduce per-request latency:

- **Model quantisation** — reducing weight precision (e.g., float64 → float32) shrinks memory footprint and speeds up arithmetic. The accuracy cost $\\Delta$ is often negligible: $|\\text{acc}_{64} - \\text{acc}_{32}| < 0.01$.
- **Prediction caching** — if the same inputs recur, cache outputs to skip inference entirely.
- **Feature pre-computation** — materialise expensive feature transforms once and reuse them.

## Putting It Together

The code below simulates a model serving pipeline in pure Python. It trains a scikit-learn classifier, builds a request-processing pipeline with validation, and compares single-inference vs batched-inference timing.`,
      codeSnippet: `import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# --- Train a model offline ---
np.random.seed(42)
X_train, y_train = make_classification(n_samples=500, n_features=10,
                                       n_informative=6, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print("=== Model Serving Simulation ===")
print(f"Model trained on {X_train.shape[0]} samples, {X_train.shape[1]} features\\n")

# --- Request handler with validation ---
def handle_request(payload, trained_model):
    """Validate input, run inference, return response dict."""
    try:
        features = np.array(payload["features"], dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != 10:
            return {"status": "error", "message": f"Expected 10 features, got {features.shape[1]}"}
        start = time.perf_counter()
        preds = trained_model.predict(features).tolist()
        probas = trained_model.predict_proba(features).max(axis=1).round(3).tolist()
        latency = (time.perf_counter() - start) * 1000
        return {"status": "ok", "predictions": preds,
                "confidence": probas, "latency_ms": round(latency, 2)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- Single inference ---
single_req = {"features": X_train[0].tolist()}
resp = handle_request(single_req, model)
print(f"Single request  -> pred={resp['predictions']}, "
      f"conf={resp['confidence']}, latency={resp['latency_ms']} ms")

bad_req = {"features": [1.0, 2.0]}
print(f"Bad request     -> {handle_request(bad_req, model)}\\n")

# --- Batched vs single timing ---
test_data = np.random.randn(200, 10)
start = time.perf_counter()
for row in test_data:
    handle_request({"features": row.tolist()}, model)
single_time = (time.perf_counter() - start) * 1000

start = time.perf_counter()
handle_request({"features": test_data.tolist()}, model)
batch_time = (time.perf_counter() - start) * 1000

print(f"200 single requests : {single_time:.1f} ms total")
print(f"1 batched request   : {batch_time:.1f} ms total")
print(f"Speedup             : {single_time / batch_time:.1f}x")`,
      codeLanguage: "python",
    },
    {
      title: "A/B Testing in Production",
      slug: "ab-testing-production",
      description:
        "Traffic splitting, canary deployments, shadow mode, statistical rigour, multi-armed bandits, and experiment infrastructure for ML models",
      markdownContent: `# A/B Testing in Production

A/B testing for ML models means comparing a **challenger** (new model) against the **champion** (current model) using live traffic. Offline metrics like validation accuracy don't always predict real-world performance — an A/B test measures actual **business impact** before a full rollout.

## Why A/B Test Models?

Offline evaluation has blind spots:

- **Distribution mismatch** — holdout data may not reflect current production traffic.
- **Proxy metrics** — accuracy on a test set doesn't guarantee improvement in revenue, engagement, or conversion.
- **Interaction effects** — a model change may alter user behaviour in ways that only surface online.

A/B testing closes the loop between offline development and online impact.

## Traffic Splitting

The simplest approach is **percentage-based routing**: send a fixed fraction of requests to each model variant.

$$
P(\\text{challenger}) = \\frac{n_{\\text{challenger}}}{n_{\\text{total}}} = \\alpha, \\quad P(\\text{champion}) = 1 - \\alpha
$$

A common starting split is $\\alpha = 0.1$ (90/10 champion/challenger). The key requirement is **random assignment** — each user or request must be independently routed to avoid selection bias.

### Canary Deployments

A canary deployment is traffic splitting with **gradual ramp-up**:

1. Start at 1–5 % challenger traffic.
2. Monitor key metrics for a burn-in period.
3. If metrics are healthy, increase to 10 %, 25 %, 50 %.
4. If any guardrail metric breaches a threshold, **roll back** immediately.

This limits blast radius: if the challenger is broken, only a small fraction of users are affected.

### Shadow Mode (Dark Launching)

In shadow mode, the challenger runs **in parallel** but its predictions are **never served** to users. Both models receive the same inputs; outputs are logged and compared offline.

$$
\\text{shadow\\_divergence} = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbb{1}[\\hat{y}_{\\text{champion}}^{(i)} \\neq \\hat{y}_{\\text{challenger}}^{(i)}]
$$

Shadow mode is risk-free — users always see the champion — making it ideal for validating a new model before any live exposure.

## Statistical Rigour

An A/B test is a **hypothesis test**. The null hypothesis is that the challenger is no better than the champion:

$$
H_0: \\mu_{\\text{challenger}} = \\mu_{\\text{champion}}, \\quad H_1: \\mu_{\\text{challenger}} \\neq \\mu_{\\text{champion}}
$$

Key considerations:

- **Sample size** — must be large enough to detect a meaningful effect. The required sample size depends on baseline rate, minimum detectable effect (MDE), and desired power ($1 - \\beta$, typically 0.8).
- **Duration** — run the test long enough to capture weekly/seasonal cycles (at least 1–2 weeks for most products).
- **Significance level** — typically $\\alpha = 0.05$. Reject $H_0$ when $p < \\alpha$.
- **Multiple comparisons** — testing many metrics inflates false-positive rate. Apply Bonferroni or Benjamini–Hochberg corrections.

> **Link:** For foundational concepts on hypothesis testing and significance, see the *A/B Testing Basics* chapter.

## Metrics to Track

A well-designed experiment tracks three tiers of metrics:

| Tier | Examples | Purpose |
|------|----------|---------|
| **Business metrics** | Conversion rate, revenue per user, click-through rate | Ultimate success criteria |
| **Model metrics** | Accuracy, AUC, precision, recall, latency (mean & p50) | Technical performance |
| **Guardrail metrics** | Error rate, latency p99, crash rate, timeout rate | Safety — must not degrade |

The experiment succeeds only if business metrics improve **and** guardrail metrics remain within acceptable bounds.

## Multi-Armed Bandits

Classical A/B testing allocates traffic **statically**. Multi-armed bandits (MABs) allocate traffic **adaptively** — shifting more requests to the better-performing variant over time.

### Thompson Sampling

Each arm $k$ maintains a Beta posterior over its success probability:

$$
\\theta_k \\sim \\text{Beta}(\\alpha_k, \\beta_k)
$$

At each step:
1. Sample $\\hat{\\theta}_k$ from each arm's posterior.
2. Route the request to the arm with the highest $\\hat{\\theta}_k$.
3. Observe the reward and update: success → $\\alpha_k += 1$, failure → $\\beta_k += 1$.

Thompson sampling naturally balances **exploration** (trying uncertain arms) and **exploitation** (favouring the current best).

### Upper Confidence Bound (UCB)

UCB selects the arm that maximises:

$$
\\text{UCB}_k = \\hat{\\mu}_k + \\sqrt{\\frac{2 \\ln t}{n_k}}
$$

where $\\hat{\\mu}_k$ is the observed mean reward for arm $k$, $t$ is the total number of rounds, and $n_k$ is the number of times arm $k$ has been selected. The second term is an **optimism bonus** that shrinks as the arm is pulled more often.

## Common Pitfalls

- **Peeking too early** — checking results before the required sample size inflates false-positive rates. Use sequential testing or always wait for full power.
- **Novelty effect** — users may engage more with a new experience simply because it's new. Allow a burn-in period before measuring.
- **Network effects** — in social products, one user's experience depends on others. Standard i.i.d. assumptions break down; consider cluster-randomised designs.
- **Segment differences** — a model may improve results for one segment while hurting another. Always check heterogeneous treatment effects.

## Infrastructure

Running A/B tests at scale requires dedicated tooling:

- **Feature flags** — toggle model variants per user/request without redeploying (e.g., LaunchDarkly, Unleash).
- **Traffic routers** — load balancers or API gateways that implement splitting logic (e.g., Istio, Envoy).
- **Experiment platforms** — end-to-end systems for assignment, logging, and analysis (e.g., Optimizely, internal platforms).
- **Logging and attribution** — every prediction must be tagged with its experiment variant for correct post-hoc analysis.

## Code Demo

The code below simulates:
1. An A/B test between two models with different accuracy rates, including a statistical significance test.
2. A Thompson sampling multi-armed bandit that adaptively allocates traffic between two models, plotting cumulative reward and traffic allocation over time.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ============================================================
# Part 1: Classical A/B Test Simulation
# ============================================================
print("=" * 55)
print("Part 1: Classical A/B Test — Champion vs Challenger")
print("=" * 55)

n_requests = 2000
champion_accuracy = 0.72   # true conversion/success rate
challenger_accuracy = 0.76  # challenger is genuinely better

# Simulate traffic split: 90/10
n_champion = int(n_requests * 0.9)
n_challenger = n_requests - n_champion

champion_results = np.random.binomial(1, champion_accuracy, n_champion)
challenger_results = np.random.binomial(1, challenger_accuracy, n_challenger)

champion_rate = champion_results.mean()
challenger_rate = challenger_results.mean()
lift = (challenger_rate - champion_rate) / champion_rate * 100

print(f"\\nTraffic split: {n_champion} champion / {n_challenger} challenger")
print(f"Champion conversion rate   : {champion_rate:.4f}")
print(f"Challenger conversion rate  : {challenger_rate:.4f}")
print(f"Observed lift               : {lift:+.2f}%")

# Two-proportion z-test
n1, p1 = n_champion, champion_rate
n2, p2 = n_challenger, challenger_rate
p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
z_stat = (p2 - p1) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\\nZ-statistic : {z_stat:.4f}")
print(f"p-value     : {p_value:.6f}")
print(f"Verdict     : {'SIGNIFICANT — challenger wins!' if p_value < 0.05 else 'Not significant — keep champion'}")

# ============================================================
# Part 2: Thompson Sampling Multi-Armed Bandit
# ============================================================
print("\\n" + "=" * 55)
print("Part 2: Thompson Sampling Multi-Armed Bandit")
print("=" * 55)

true_rates = [champion_accuracy, challenger_accuracy]
arm_names = ["Champion", "Challenger"]
n_rounds = 1500

# Beta distribution parameters (start with uniform prior)
alphas = [1.0, 1.0]
betas_param = [1.0, 1.0]

chosen_arms = []
rewards = []
cumulative_rewards = []
arm_counts = [0, 0]
total_reward = 0

for t in range(n_rounds):
    # Sample from each arm's posterior
    samples = [np.random.beta(alphas[k], betas_param[k]) for k in range(2)]
    chosen = int(np.argmax(samples))

    # Observe reward
    reward = np.random.binomial(1, true_rates[chosen])

    # Update posterior
    alphas[chosen] += reward
    betas_param[chosen] += 1 - reward

    chosen_arms.append(chosen)
    rewards.append(reward)
    arm_counts[chosen] += 1
    total_reward += reward
    cumulative_rewards.append(total_reward)

chosen_arms = np.array(chosen_arms)
cumulative_rewards = np.array(cumulative_rewards)

print(f"\\nTotal rounds: {n_rounds}")
for k in range(2):
    frac = arm_counts[k] / n_rounds * 100
    print(f"  {arm_names[k]:12s}: pulled {arm_counts[k]:5d} times ({frac:.1f}%)")
print(f"Total reward: {total_reward} / {n_rounds} ({total_reward/n_rounds:.3f} avg)")

# Optimal reward (always picking the best arm)
optimal_reward = np.cumsum(np.random.binomial(1, max(true_rates), n_rounds))
regret = optimal_reward - cumulative_rewards

# ============================================================
# Plots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: A/B test results bar chart
ax = axes[0, 0]
bars = ax.bar(arm_names, [champion_rate, challenger_rate],
              color=["steelblue", "coral"], edgecolor="black", linewidth=0.8)
ax.axhline(y=champion_accuracy, color="steelblue", linestyle="--", alpha=0.5, label="True champion rate")
ax.axhline(y=challenger_accuracy, color="coral", linestyle="--", alpha=0.5, label="True challenger rate")
ax.set_ylabel("Conversion Rate")
ax.set_title(f"A/B Test Results (p={p_value:.4f})")
ax.legend(fontsize=8)
ax.set_ylim(0, 1)

# Plot 2: Cumulative reward over time
ax = axes[0, 1]
ax.plot(cumulative_rewards, color="coral", label="Thompson Sampling")
ax.plot(optimal_reward, color="grey", linestyle="--", alpha=0.7, label="Optimal (always best arm)")
ax.set_xlabel("Round")
ax.set_ylabel("Cumulative Reward")
ax.set_title("Cumulative Reward Over Time")
ax.legend(fontsize=8)

# Plot 3: Traffic allocation over time (rolling window)
ax = axes[1, 0]
window = 50
challenger_frac = np.convolve(chosen_arms, np.ones(window)/window, mode="valid")
ax.plot(challenger_frac, color="coral", label="Challenger fraction")
ax.axhline(y=0.5, color="grey", linestyle=":", alpha=0.5)
ax.set_xlabel("Round")
ax.set_ylabel("Fraction of Traffic → Challenger")
ax.set_title(f"Traffic Allocation (rolling {window}-round window)")
ax.set_ylim(0, 1)
ax.legend(fontsize=8)

# Plot 4: Cumulative regret
ax = axes[1, 1]
ax.plot(regret, color="purple")
ax.set_xlabel("Round")
ax.set_ylabel("Cumulative Regret")
ax.set_title("Cumulative Regret (Thompson Sampling)")

plt.tight_layout()
plt.savefig("output.png", dpi=100)
plt.show()
print("\\nPlot saved to output.png")`,
      codeLanguage: "python",
    },
    {
      title: "Monitoring & Observability",
      slug: "monitoring-observability",
      description:
        "Data drift detection, statistical tests (KS, PSI), performance degradation tracking, and alerting strategies",
      markdownContent: `# Monitoring & Observability

A model that performs well at launch can silently degrade as the world changes. **Monitoring** catches this decay before it reaches users, by continuously comparing production behaviour against the training baseline.

## Data Drift

Data drift occurs when the distribution of incoming features shifts away from what the model was trained on. Even if the model code hasn't changed, a shift in $P(X)$ can invalidate learned decision boundaries.

Two standard statistical tests quantify drift:

### Kolmogorov–Smirnov (KS) Test

The KS statistic measures the maximum distance between two empirical CDFs:

$$
D = \\max_x \\bigl| F_{\\text{train}}(x) - F_{\\text{prod}}(x) \\bigr|
$$

A large $D$ with a small $p$-value (typically $p < 0.05$) signals that the production distribution has shifted significantly.

### Population Stability Index (PSI)

PSI is widely used in credit scoring and risk modelling. It bins both distributions and sums the divergence per bin:

$$
\\text{PSI} = \\sum_{i=1}^{n} (p_i - q_i) \\cdot \\ln\\frac{p_i}{q_i}
$$

where $p_i$ and $q_i$ are the proportions in bin $i$ for production and training data respectively. Rules of thumb:

| PSI Value | Interpretation |
|-----------|---------------|
| < 0.1     | No significant drift |
| 0.1 – 0.2 | Moderate drift — investigate |
| > 0.2     | Significant drift — retrain |

## Concept Drift and Performance Decay

Even when inputs look similar, the **relationship** between features and target can change — this is **concept drift**. For example, a fraud model trained pre-pandemic may see entirely new spending patterns. Monitoring prediction-time metrics (accuracy, precision, recall) against a labelled holdout or delayed ground truth is the most direct way to detect concept drift.

## Alerting Strategies

Effective alerting avoids both false alarms and missed detections:

- **Threshold-based alerts** — trigger when a metric crosses a fixed boundary (e.g., PSI > 0.2).
- **Rolling-window comparisons** — compare the last $k$ hours of predictions to a baseline window.
- **Rate-of-change alerts** — flag when drift *accelerates*, even if absolute values are still within bounds.

The code below simulates drift detection end-to-end: it generates training and "production" data with a deliberate distribution shift, computes both the KS statistic and PSI, and plots the distributions side by side.`,
      codeSnippet: `import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# --- Simulate training vs drifted production data ---
train_data = np.random.normal(loc=0.0, scale=1.0, size=1000)
prod_data  = np.random.normal(loc=0.4, scale=1.3, size=1000)  # shifted

# --- KS Test ---
ks_stat, ks_p = stats.ks_2samp(train_data, prod_data)
print("=== Drift Detection Report ===")
print(f"KS Statistic : {ks_stat:.4f}")
print(f"KS p-value   : {ks_p:.6f}")
print(f"KS Verdict   : {'DRIFT DETECTED' if ks_p < 0.05 else 'No significant drift'}\\n")

# --- PSI Calculation ---
def compute_psi(train, prod, bins=10):
    breakpoints = np.linspace(min(train.min(), prod.min()),
                              max(train.max(), prod.max()), bins + 1)
    train_counts = np.histogram(train, bins=breakpoints)[0] / len(train)
    prod_counts  = np.histogram(prod,  bins=breakpoints)[0] / len(prod)
    # Avoid log(0) with small epsilon
    train_counts = np.clip(train_counts, 1e-4, None)
    prod_counts  = np.clip(prod_counts,  1e-4, None)
    return np.sum((prod_counts - train_counts) * np.log(prod_counts / train_counts))

psi = compute_psi(train_data, prod_data)
psi_label = "No drift" if psi < 0.1 else ("Moderate" if psi < 0.2 else "SIGNIFICANT")
print(f"PSI Score    : {psi:.4f}")
print(f"PSI Verdict  : {psi_label} (thresholds: <0.1 ok, 0.1-0.2 moderate, >0.2 significant)")

# --- Alert Summary ---
alerts = []
if ks_p < 0.05:
    alerts.append("KS test: distribution shift detected")
if psi >= 0.1:
    alerts.append(f"PSI={psi:.3f}: {'moderate' if psi < 0.2 else 'significant'} drift")
print(f"\\nActive Alerts: {len(alerts)}")
for a in alerts:
    print(f"  ⚠ {a}")

# --- Plot distributions ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(train_data, bins=30, alpha=0.6, color="steelblue", label="Training", density=True)
axes[0].hist(prod_data,  bins=30, alpha=0.6, color="coral",     label="Production", density=True)
axes[0].set_title("Feature Distribution Comparison")
axes[0].set_xlabel("Feature Value")
axes[0].set_ylabel("Density")
axes[0].legend()

# CDF comparison
sorted_all = np.sort(np.concatenate([train_data, prod_data]))
train_cdf = np.searchsorted(np.sort(train_data), sorted_all) / len(train_data)
prod_cdf  = np.searchsorted(np.sort(prod_data),  sorted_all) / len(prod_data)
axes[1].plot(sorted_all, train_cdf, label="Training CDF", color="steelblue")
axes[1].plot(sorted_all, prod_cdf,  label="Production CDF", color="coral")
axes[1].set_title(f"Empirical CDFs (KS={ks_stat:.3f})")
axes[1].set_xlabel("Feature Value")
axes[1].legend()
plt.tight_layout()
plt.savefig("output.png", dpi=100)
plt.show()
print("\\nPlot saved to output.png")`,
      codeLanguage: "python",
    },
  ],
};
