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
