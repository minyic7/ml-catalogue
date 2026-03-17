import type { Chapter } from "../types";

const experimentTrackingMarkdown = `
# Experiment Tracking

In machine learning, training a model once is never enough. You iterate — tuning hyperparameters, swapping features, adjusting architectures. Without systematic experiment tracking, this process quickly becomes chaotic: which learning rate produced the best F1 score? What preprocessing was active in last Tuesday's run?

## Why Track Experiments?

**Reproducibility** — every result should be recreatable. If you can't reproduce a model's performance, you can't trust it in production. Tracking captures the exact parameters, code state, and data that produced each result.

**Comparison** — with dozens or hundreds of runs, you need structured records to compare performance across configurations. A well-organized experiment log lets you answer questions like "did dropout help?" in seconds, not hours.

**Collaboration** — teams need shared visibility. When multiple people train models, a central experiment log prevents duplicated work and ensures everyone builds on the best known configuration.

## Key Concepts

Every experiment tracking system organizes around these primitives:

- **Run**: A single execution of a training script with specific settings
- **Parameters**: The inputs to a run — hyperparameters like learning rate $\\alpha$, batch size $B$, number of epochs $E$
- **Metrics**: The outputs — quantities that measure performance, such as accuracy, loss, or F1 score
- **Artifacts**: Files produced by a run — saved models, plots, predictions

## Structured Logging

A minimal experiment record captures the mapping from parameters to metrics:

$$\\text{Run}: (\\alpha, B, E, \\ldots) \\longrightarrow (\\text{accuracy}, \\text{loss}, \\ldots)$$

Over many runs, this forms a table where each row is a run and columns are parameters and metrics. You can then query: which parameter combination minimized validation loss?

$$\\text{best run} = \\arg\\min_{r \\in \\text{runs}} \\; \\mathcal{L}_{\\text{val}}(r)$$

## Best Practices

- **Log everything automatically** — don't rely on manual notes
- **Use consistent naming** — standardize parameter and metric names across runs
- **Record environment details** — Python version, package versions, random seeds
- **Tag runs** — add labels like "baseline", "experiment-v2" for easy filtering

Run the code to see a pure-Python experiment tracker that logs runs to JSON and produces a comparison summary.
`;

const experimentTrackingCode = `import json
import os
import random

random.seed(42)

# --- Simple Experiment Tracker ---
class ExperimentTracker:
    def __init__(self):
        self.runs = []

    def log_run(self, name, params, metrics):
        run = {
            "run_id": len(self.runs) + 1,
            "name": name,
            "params": params,
            "metrics": metrics,
        }
        self.runs.append(run)
        return run

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.runs, f, indent=2)

    def best_run(self, metric, minimize=True):
        key = (min if minimize else max)
        return key(self.runs, key=lambda r: r["metrics"][metric])

    def summary(self):
        header = f"{'Run':<6} {'Name':<18} {'LR':<8} {'Epochs':<8} {'Acc':>6} {'Loss':>7}"
        print(header)
        print("-" * len(header))
        for r in self.runs:
            p, m = r["params"], r["metrics"]
            print(f"{r['run_id']:<6} {r['name']:<18} {p['lr']:<8} "
                  f"{p['epochs']:<8} {m['accuracy']:>5.3f} {m['loss']:>7.4f}")

# --- Simulate training runs with different hyperparameters ---
tracker = ExperimentTracker()
configs = [
    ("baseline",    {"lr": 0.01,  "epochs": 10, "batch_size": 32}),
    ("high-lr",     {"lr": 0.1,   "epochs": 10, "batch_size": 32}),
    ("long-train",  {"lr": 0.01,  "epochs": 50, "batch_size": 32}),
    ("small-batch", {"lr": 0.01,  "epochs": 10, "batch_size": 8}),
    ("tuned",       {"lr": 0.005, "epochs": 30, "batch_size": 16}),
]

print("=== Experiment Tracking Demo ===\\n")
for name, params in configs:
    # Simulate metrics (higher epochs & lower lr -> generally better)
    acc = min(0.99, 0.7 + params["epochs"] * 0.004 - params["lr"] * 0.3
              + random.uniform(-0.02, 0.02))
    loss = max(0.01, 0.5 - params["epochs"] * 0.005 + params["lr"] * 0.4
               + random.uniform(-0.03, 0.03))
    tracker.log_run(name, params, {"accuracy": round(acc, 4), "loss": round(loss, 4)})

tracker.summary()
tracker.save("experiment_log.json")

best = tracker.best_run("loss", minimize=True)
print(f"\\nBest run (lowest loss): #{best['run_id']} '{best['name']}' "
      f"-> loss={best['metrics']['loss']}, acc={best['metrics']['accuracy']}")
print(f"\\nExperiment log saved to experiment_log.json")
`;

const modelRegistryMarkdown = `
# Model Registry

A model registry is a centralized store for managing trained model versions throughout their lifecycle. Think of it as version control specifically designed for ML models — tracking not just the model artifact, but its lineage, performance, and deployment status.

## Why Version Models?

In production ML, models are retrained regularly as new data arrives. Without versioning:

- You can't roll back to a previous model when a new version underperforms
- There's no audit trail for regulatory compliance
- Teams deploy models without knowing which version is running where

### Semantic Versioning for ML

Borrowing from software, ML models can follow semantic versioning \`MAJOR.MINOR.PATCH\`:

- **MAJOR** — new architecture or significant retraining (e.g., switching from logistic regression to a neural network)
- **MINOR** — retrained on new data with the same architecture
- **PATCH** — same model with updated preprocessing or bug fixes

## Model Lifecycle Stages

Every registered model moves through well-defined stages:

$$\\text{Development} \\xrightarrow{\\text{validate}} \\text{Staging} \\xrightarrow{\\text{approve}} \\text{Production} \\xrightarrow{\\text{retire}} \\text{Archived}$$

- **Development**: Model is being trained and evaluated
- **Staging**: Model passes offline tests and is being validated in a pre-production environment
- **Production**: Model is serving live traffic
- **Archived**: Model has been retired and is kept for audit purposes

## Model Metadata

Each registry entry should capture:

- **Training data hash** — a fingerprint of the training data to ensure reproducibility. For a dataset $D$, compute $h(D)$ so you can verify the exact data used:

$$h(D) = \\text{SHA256}(\\text{serialize}(D))$$

- **Metrics** — the model's evaluation scores (accuracy, F1, AUC, etc.)
- **Dependencies** — library versions (e.g., scikit-learn 1.4, numpy 2.0)
- **Author and timestamp** — who registered the model and when

## Why This Matters

A model registry is the backbone of responsible ML deployment. It answers critical questions: what model is in production? How does it compare to the previous version? Can we reproduce it? Run the code to see a model registry simulation.
`;

const modelRegistryCode = `from dataclasses import dataclass, field
from datetime import datetime

STAGES = ["development", "staging", "production", "archived"]

@dataclass
class ModelVersion:
    name: str
    version: str
    stage: str
    accuracy: float
    data_hash: str
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

class ModelRegistry:
    def __init__(self):
        self.models: dict[str, list[ModelVersion]] = {}

    def register(self, name, version, accuracy, data_hash):
        mv = ModelVersion(name, version, "development", accuracy, data_hash)
        self.models.setdefault(name, []).append(mv)
        print(f"  Registered {name} v{version} (acc={accuracy:.3f})")
        return mv

    def promote(self, name, version, target_stage):
        for mv in self.models.get(name, []):
            if mv.version == version:
                old = mv.stage
                mv.stage = target_stage
                # Archive old production model when promoting new one
                if target_stage == "production":
                    for other in self.models[name]:
                        if other.version != version and other.stage == "production":
                            other.stage = "archived"
                print(f"  {name} v{version}: {old} -> {target_stage}")
                return
        print(f"  Model {name} v{version} not found!")

    def get_production(self, name):
        for mv in self.models.get(name, []):
            if mv.stage == "production":
                return mv
        return None

    def show(self):
        print(f"{'Model':<16} {'Version':<10} {'Stage':<14} {'Accuracy':>8} {'Data Hash':<12}")
        print("-" * 64)
        for versions in self.models.values():
            for mv in versions:
                print(f"{mv.name:<16} {mv.version:<10} {mv.stage:<14} "
                      f"{mv.accuracy:>7.3f}  {mv.data_hash:<12}")

# --- Demo ---
print("=== Model Registry Demo ===\\n")
registry = ModelRegistry()

print("-- Registering models --")
registry.register("fraud-detect", "1.0.0", 0.912, "a3f2c9e1")
registry.register("fraud-detect", "1.1.0", 0.935, "b7d4e8f2")
registry.register("fraud-detect", "2.0.0", 0.958, "c1a9b3d5")

print("\\n-- Promoting through lifecycle --")
registry.promote("fraud-detect", "1.0.0", "staging")
registry.promote("fraud-detect", "1.0.0", "production")
registry.promote("fraud-detect", "1.1.0", "staging")
registry.promote("fraud-detect", "1.1.0", "production")  # auto-archives v1.0.0
registry.promote("fraud-detect", "2.0.0", "staging")

print("\\n-- Current Registry State --")
registry.show()

prod = registry.get_production("fraud-detect")
if prod:
    print(f"\\nCurrent production model: {prod.name} v{prod.version} "
          f"(accuracy={prod.accuracy:.3f}, data={prod.data_hash})")
`;

const cicdMarkdown = `
# CI/CD for ML

Continuous Integration and Continuous Deployment (CI/CD) extends beyond traditional software to ML pipelines. An ML CI/CD pipeline automates the path from raw data to a deployed model, with validation gates at every stage.

## ML Pipeline Stages

A typical ML CI/CD pipeline has four stages, each with automated checks:

$$\\text{Data Validation} \\rightarrow \\text{Training} \\rightarrow \\text{Evaluation} \\rightarrow \\text{Deployment}$$

If any stage fails its checks, the pipeline halts and alerts the team. This prevents bad data or underperforming models from reaching production.

## Data Validation

Before training, validate the incoming data:

**Schema checks** — verify that expected columns exist with correct types. Missing or renamed features will silently break a model.

**Distribution drift detection** — compare the new data's statistical properties against a reference distribution. A common metric is the **Population Stability Index (PSI)**:

$$\\text{PSI} = \\sum_{i=1}^{k} (p_i - q_i) \\cdot \\ln\\frac{p_i}{q_i}$$

where $p_i$ and $q_i$ are the proportions in bin $i$ for the new and reference distributions. A PSI above 0.2 typically signals significant drift that warrants investigation.

## Model Testing

After training, the model must pass automated quality gates:

**Performance thresholds** — the model must meet minimum metric requirements. For example, accuracy must exceed a predefined threshold $\\tau$:

$$\\text{pass if } \\; \\text{accuracy}(M) \\geq \\tau$$

**A/B comparison with baseline** — the new model should perform at least as well as the current production model. A simple gate checks:

$$\\text{pass if } \\; \\text{metric}(M_{\\text{new}}) \\geq \\text{metric}(M_{\\text{baseline}})$$

More sophisticated pipelines use statistical tests to ensure improvements are significant, not due to random variation.

## Why This Matters

Without CI/CD, ML teams deploy models manually — a process that's slow, error-prone, and hard to audit. Automated pipelines catch data quality issues before they corrupt models, enforce performance standards, and create a repeatable deployment process. Run the code to see a simulated ML CI/CD pipeline with validation gates.
`;

const cicdCode = `import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

def check(name, passed):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    return passed

print("=== ML CI/CD Pipeline Demo ===\\n")
all_passed = True

# --- Stage 1: Data Validation ---
print("Stage 1: Data Validation")
expected_schema = {"feature_0": "float", "feature_1": "float",
                   "feature_2": "float", "label": "int"}

# Simulate incoming data
X = np.random.randn(200, 3)
y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
columns = {"feature_0": "float", "feature_1": "float",
           "feature_2": "float", "label": "int"}

schema_ok = check("Schema matches expected columns", columns == expected_schema)
null_ok = check("No null values detected", not np.any(np.isnan(X)))

# PSI drift check (compare two halves as reference vs new)
ref, new = np.histogram(X[:100, 0], bins=5)[0], np.histogram(X[100:, 0], bins=5)[0]
ref_p, new_p = ref / ref.sum() + 1e-6, new / new.sum() + 1e-6
psi = np.sum((new_p - ref_p) * np.log(new_p / ref_p))
drift_ok = check(f"Distribution drift PSI={psi:.4f} < 0.2", psi < 0.2)
all_passed &= schema_ok and null_ok and drift_ok

# --- Stage 2: Model Training ---
print("\\nStage 2: Model Training")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print(f"  Trained LogisticRegression on {len(X_train)} samples")

# --- Stage 3: Model Evaluation ---
print("\\nStage 3: Model Evaluation")
acc = accuracy_score(y_test, model.predict(X_test))
threshold = 0.75
baseline_acc = 0.78
acc_ok = check(f"Accuracy {acc:.3f} >= threshold {threshold}", acc >= threshold)
baseline_ok = check(f"Accuracy {acc:.3f} >= baseline {baseline_acc}", acc >= baseline_acc)
all_passed &= acc_ok and baseline_ok

# --- Stage 4: Deployment Decision ---
print("\\nStage 4: Deployment Decision")
if all_passed:
    print("  All gates passed -> Model APPROVED for deployment")
else:
    print("  Some gates failed -> Model REJECTED, review required")

print(f"\\nPipeline result: {'PASSED' if all_passed else 'FAILED'}")
`;

export const mlops: Chapter = {
  title: "MLOps",
  slug: "mlops",
  pages: [
    {
      title: "Experiment Tracking",
      slug: "experiment-tracking",
      description:
        "Reproducible experiment logging with parameters, metrics, and run comparison",
      markdownContent: experimentTrackingMarkdown,
      codeSnippet: experimentTrackingCode,
      codeLanguage: "python",
    },
    {
      title: "Model Registry",
      slug: "model-registry",
      description:
        "Model versioning, lifecycle stages, and metadata management",
      markdownContent: modelRegistryMarkdown,
      codeSnippet: modelRegistryCode,
      codeLanguage: "python",
    },
    {
      title: "CI/CD for ML",
      slug: "ci-cd-for-ml",
      description:
        "Automated ML pipelines with data validation, training, and deployment gates",
      markdownContent: cicdMarkdown,
      codeSnippet: cicdCode,
      codeLanguage: "python",
    },
  ],
};
