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

    def to_json(self):
        return json.dumps(self.runs, indent=2)

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
log_json = tracker.to_json()
print(f"\\nExperiment log (JSON):\\n{log_json[:200]}...")

best = tracker.best_run("loss", minimize=True)
print(f"\\nBest run (lowest loss): #{best['run_id']} '{best['name']}' "
      f"-> loss={best['metrics']['loss']}, acc={best['metrics']['accuracy']}")
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

const featureStoreMarkdown = `
# Feature Store

A feature store is a centralised repository for storing, sharing, and serving ML features. It sits between raw data and model training/serving, acting as the single source of truth for feature definitions across an organisation.

## Why Use a Feature Store?

**Consistency between training and serving** — the most dangerous bug in ML is *training/serving skew*: when the features used at training time differ from those at inference time. A feature store enforces the same transformation logic in both paths.

**Feature reuse across teams** — without a shared store, data scientists on different teams independently re-derive the same features (e.g., "user's 30-day purchase count"). A feature store lets one team define a feature and others discover and reuse it.

**Avoid training/serving skew** — when training uses one code path and serving uses another, subtle differences (different aggregation windows, timezone handling, null imputation) cause model performance to degrade silently in production.

## Core Components

A feature store has three main components:

### Offline Store (Batch)

Stores historical feature values for training. Features are computed via batch jobs over large datasets and stored in columnar formats (Parquet, Delta Lake):

$$\\text{Offline Store}: \\; f(\\text{entity}, t) \\rightarrow \\text{feature vector at time } t$$

This enables **point-in-time correct** training — you retrieve features as they existed at a specific timestamp, preventing data leakage.

### Online Store (Low-Latency Serving)

Serves the latest feature values for real-time inference. Uses low-latency key-value stores (Redis, DynamoDB):

$$\\text{Online Store}: \\; f(\\text{entity}) \\rightarrow \\text{current feature vector}$$

Latency requirements are typically under 10ms for real-time models.

### Feature Registry

A metadata catalogue that stores:
- Feature definitions and transformation logic
- Data sources and freshness requirements
- Ownership and documentation
- Feature lineage and dependencies

## Feature Engineering Pipeline

The flow from raw data to model predictions through a feature store:

$$\\text{Raw Data} \\xrightarrow{\\text{transform}} \\text{Feature Store} \\xrightarrow{\\text{retrieve}} \\begin{cases} \\text{Training (offline)} \\\\ \\text{Serving (online)} \\end{cases}$$

This architecture guarantees that the transformation $T(x)$ applied to raw data $x$ is identical whether producing training data or serving features.

## Popular Tools

| Tool | Type | Key Strength |
|------|------|-------------|
| **Feast** | Open source | Flexible, cloud-agnostic |
| **Tecton** | Managed | Enterprise-grade, real-time features |
| **Hopsworks** | Open source / managed | Integrated ML platform |

## Best Practices

- **Define features as code** — version-controlled transformation functions
- **Monitor feature freshness** — stale features degrade model quality
- **Validate feature distributions** — detect upstream data issues before they reach models
- **Use point-in-time joins** — prevent data leakage during training

Run the code to see a simple feature store implementation that demonstrates feature definition, storage, retrieval, and consistent use across training and serving.
`;

const featureStoreCode = `import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# === Simple Feature Store Implementation ===

class FeatureDefinition:
    """Defines a feature transformation that can be applied consistently."""
    def __init__(self, name, description, transform_fn):
        self.name = name
        self.description = description
        self.transform_fn = transform_fn
        self.created_at = datetime.now().strftime("%Y-%m-%d")

    def compute(self, raw_data):
        return self.transform_fn(raw_data)


class SimpleFeatureStore:
    """A minimal feature store demonstrating core concepts."""
    def __init__(self):
        self.registry = {}       # Feature definitions
        self.offline_store = {}  # Historical features (for training)
        self.online_store = {}   # Latest features (for serving)

    def register_feature(self, feature_def):
        self.registry[feature_def.name] = feature_def
        print(f"  Registered: {feature_def.name} — {feature_def.description}")

    def materialise(self, raw_data, entity_col="user_id"):
        """Compute all features and store in both offline and online stores."""
        print("\\n-- Materialising features --")
        for name, feat_def in self.registry.items():
            computed = feat_def.compute(raw_data)
            # Offline store: keep full history
            self.offline_store[name] = computed
            # Online store: keep only latest per entity
            self.online_store[name] = computed.groupby(entity_col).last()
            print(f"  {name}: {len(computed)} rows (offline), "
                  f"{len(self.online_store[name])} entities (online)")

    def get_training_features(self, feature_names, entity_col="user_id"):
        """Retrieve historical features for training (offline store)."""
        frames = [self.offline_store[f] for f in feature_names]
        return pd.concat(frames, axis=1).reset_index(drop=True)

    def get_serving_features(self, entity_id, feature_names):
        """Retrieve latest features for a single entity (online store)."""
        result = {}
        for name in feature_names:
            if entity_id in self.online_store[name].index:
                row = self.online_store[name].loc[entity_id]
                if isinstance(row, pd.Series):
                    result.update(row.to_dict())
                else:
                    result[name] = row
        return result

    def show_registry(self):
        print(f"\\n{'Feature':<28} {'Description':<45} {'Created'}")
        print("-" * 85)
        for name, fd in self.registry.items():
            print(f"{name:<28} {fd.description:<45} {fd.created_at}")


# === Simulate raw transaction data ===
print("=== Feature Store Demo ===\\n")
print("-- Generating raw transaction data --")
n_transactions = 50
users = [f"user_{i}" for i in range(1, 6)]

raw_data = pd.DataFrame({
    "user_id": np.random.choice(users, n_transactions),
    "amount": np.round(np.random.exponential(50, n_transactions), 2),
    "timestamp": [datetime.now() - timedelta(days=np.random.randint(0, 30))
                  for _ in range(n_transactions)],
    "category": np.random.choice(["food", "transport", "shopping", "entertainment"],
                                 n_transactions),
})
print(f"  Generated {n_transactions} transactions for {len(users)} users")
print(f"  Columns: {list(raw_data.columns)}")

# === Define feature transformations ===
store = SimpleFeatureStore()
print("\\n-- Registering feature definitions --")

store.register_feature(FeatureDefinition(
    "txn_count_30d",
    "Transaction count in last 30 days",
    lambda df: df.groupby("user_id").agg(
        txn_count_30d=("amount", "count")
    )
))

store.register_feature(FeatureDefinition(
    "avg_txn_amount",
    "Average transaction amount per user",
    lambda df: df.groupby("user_id").agg(
        avg_txn_amount=("amount", "mean")
    ).round(2)
))

store.register_feature(FeatureDefinition(
    "max_txn_amount",
    "Maximum single transaction amount",
    lambda df: df.groupby("user_id").agg(
        max_txn_amount=("amount", "max")
    ).round(2)
))

store.register_feature(FeatureDefinition(
    "unique_categories",
    "Number of distinct spending categories",
    lambda df: df.groupby("user_id").agg(
        unique_categories=("category", "nunique")
    )
))

# === Materialise features into stores ===
store.materialise(raw_data)
store.show_registry()

# === Training: retrieve from offline store ===
print("\\n-- Training: Retrieve from OFFLINE store --")
feature_names = ["txn_count_30d", "avg_txn_amount", "max_txn_amount", "unique_categories"]
training_df = store.get_training_features(feature_names)
print(training_df.to_string(index=False))

# === Serving: retrieve from online store ===
print("\\n-- Serving: Retrieve from ONLINE store --")
for uid in ["user_1", "user_3"]:
    features = store.get_serving_features(uid, feature_names)
    print(f"  {uid}: {features}")

# === Key point: same definitions used in both paths ===
print("\\n-- Consistency Check --")
print("  Training and serving use the SAME feature definitions.")
print("  This eliminates training/serving skew!")
print(f"  Registered features: {list(store.registry.keys())}")
`;

const modelDriftMonitoringMarkdown = `
# Model Drift & Monitoring

Deploying a model is not the finish line — it is the start of a new challenge. Real-world data changes over time, and a model trained on yesterday's patterns may fail on today's inputs. **Model monitoring** detects these changes so you can respond before business impact occurs.

## Types of Drift

### Data Drift (Covariate Shift)

The input feature distribution changes while the true relationship between inputs and outputs remains the same:

$$P_{\\text{train}}(X) \\neq P_{\\text{prod}}(X), \\quad \\text{but} \\quad P(Y|X) \\text{ unchanged}$$

**Example**: A fraud model trained on transaction amounts averaging \\$50 starts receiving data where the average is \\$200 (due to inflation, market changes, or a new customer segment).

### Concept Drift

The relationship between inputs and outputs changes — the "concept" the model learned is no longer valid:

$$P_{\\text{train}}(Y|X) \\neq P_{\\text{prod}}(Y|X)$$

**Example**: A spam classifier trained before a new type of phishing attack emerges. The input distribution may look similar, but what constitutes "spam" has changed.

## Detection Methods

### Kolmogorov–Smirnov (KS) Test

The KS test measures the maximum distance between two cumulative distribution functions (CDFs):

$$D_{KS} = \\sup_x |F_{\\text{ref}}(x) - F_{\\text{new}}(x)|$$

A large $D_{KS}$ (with a small p-value, typically $p < 0.05$) indicates the distributions are significantly different.

### Population Stability Index (PSI)

PSI quantifies how much a distribution has shifted relative to a baseline. It divides the feature range into $k$ bins and compares bin proportions:

$$\\text{PSI} = \\sum_{i=1}^{k} (p_i - q_i) \\cdot \\ln\\frac{p_i}{q_i}$$

where $p_i$ is the proportion in bin $i$ for the new data and $q_i$ for the reference. Interpretation:

| PSI Value | Interpretation |
|-----------|---------------|
| < 0.1 | No significant drift |
| 0.1 – 0.2 | Moderate drift — investigate |
| > 0.2 | Significant drift — action required |

## Response Strategies

When drift is detected, teams typically follow this escalation path:

1. **Alert** — notify the ML team when drift exceeds a threshold
2. **Investigate** — determine if drift is meaningful (seasonal changes vs. data pipeline bugs)
3. **Retrain** — update the model on recent data
4. **Automated retraining** — set up pipelines that trigger retraining when drift exceeds thresholds:

$$\\text{if } \\text{PSI} > \\tau \\text{ then trigger retraining pipeline}$$

## Monitoring Dashboards

Production ML models should track:

- **Prediction distribution** — are model outputs shifting? A sudden change in predicted probabilities indicates something has changed
- **Feature distributions** — monitor each input feature for drift
- **Model latency** — inference time affects user experience; track p50, p95, p99
- **Error rates** — if ground truth is available (even delayed), track accuracy over time

Run the code to see a simulation that generates drifting data, detects drift using KS test and PSI, and visualises drift over time windows with alerting thresholds.
`;

const modelDriftMonitoringCode = `import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# === Drift Detection Utilities ===

def compute_psi(reference, current, bins=10):
    """Compute Population Stability Index between two distributions."""
    # Create bins from reference distribution
    breakpoints = np.linspace(min(reference.min(), current.min()),
                              max(reference.max(), current.max()), bins + 1)
    ref_counts = np.histogram(reference, bins=breakpoints)[0] + 1  # smoothing
    cur_counts = np.histogram(current, bins=breakpoints)[0] + 1
    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi

def interpret_psi(psi_value):
    if psi_value < 0.1:
        return "No significant drift"
    elif psi_value < 0.2:
        return "Moderate drift"
    else:
        return "Significant drift"

# === Simulate Data with Gradual Drift ===
print("=== Model Drift & Monitoring Demo ===\\n")

n_samples = 500
n_windows = 8
reference_mean = 0.0
reference_std = 1.0
drift_rate = 0.3  # mean shifts by this amount each window

print("-- Simulating data with gradual drift --")
print(f"   Reference: N({reference_mean}, {reference_std}^2)")
print(f"   Drift rate: +{drift_rate} per window\\n")

reference_data = np.random.normal(reference_mean, reference_std, n_samples)

windows = []
ks_stats_list = []
ks_pvals = []
psi_values = []

for i in range(n_windows):
    shifted_mean = reference_mean + drift_rate * i
    window_data = np.random.normal(shifted_mean, reference_std, n_samples)
    windows.append(window_data)

    # KS test
    ks_stat, ks_pval = stats.ks_2samp(reference_data, window_data)
    ks_stats_list.append(ks_stat)
    ks_pvals.append(ks_pval)

    # PSI
    psi = compute_psi(reference_data, window_data)
    psi_values.append(psi)

# === Print Detection Results ===
print(f"{'Window':<8} {'Mean':>6} {'KS Stat':>9} {'KS p-val':>10} "
      f"{'PSI':>7} {'Status':<24} {'Alert'}")
print("-" * 82)

alert_threshold = 0.2
for i in range(n_windows):
    shifted_mean = reference_mean + drift_rate * i
    alert = "!! ALERT" if psi_values[i] > alert_threshold else ""
    sig = "*" if ks_pvals[i] < 0.05 else " "
    print(f"  W{i:<5} {shifted_mean:>6.2f} {ks_stats_list[i]:>9.4f} "
          f"{ks_pvals[i]:>10.4f}{sig} {psi_values[i]:>7.4f} "
          f"{interpret_psi(psi_values[i]):<24} {alert}")

print(f"\\n  * = statistically significant (p < 0.05)")
print(f"  Alert threshold: PSI > {alert_threshold}")

# === Plot Drift Over Time ===
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Feature distributions shifting over time
ax1 = axes[0, 0]
for i in [0, 2, 4, 7]:
    ax1.hist(windows[i], bins=30, alpha=0.4, density=True,
             label=f"W{i} (mean={reference_mean + drift_rate*i:.1f})")
ax1.hist(reference_data, bins=30, alpha=0.3, density=True,
         color="black", linestyle="--", label="Reference")
ax1.set_title("Feature Distribution Over Time")
ax1.set_xlabel("Feature Value")
ax1.set_ylabel("Density")
ax1.legend(fontsize=8)

# Plot 2: KS statistic over windows
ax2 = axes[0, 1]
ax2.plot(range(n_windows), ks_stats_list, "bo-", linewidth=2, markersize=6)
ax2.axhline(y=0.05, color="orange", linestyle="--", label="Warning (0.05)")
ax2.axhline(y=0.1, color="red", linestyle="--", label="Critical (0.1)")
ax2.set_title("KS Statistic Over Time")
ax2.set_xlabel("Time Window")
ax2.set_ylabel("KS Statistic")
ax2.legend(fontsize=8)
ax2.set_xticks(range(n_windows))
ax2.set_xticklabels([f"W{i}" for i in range(n_windows)])

# Plot 3: PSI over windows with threshold
ax3 = axes[1, 0]
colors = ["green" if p < 0.1 else "orange" if p < 0.2 else "red"
          for p in psi_values]
ax3.bar(range(n_windows), psi_values, color=colors, edgecolor="black", alpha=0.8)
ax3.axhline(y=0.1, color="orange", linestyle="--", label="Moderate (0.1)")
ax3.axhline(y=0.2, color="red", linestyle="--", label="Alert threshold (0.2)")
ax3.set_title("PSI Over Time Windows")
ax3.set_xlabel("Time Window")
ax3.set_ylabel("PSI")
ax3.legend(fontsize=8)
ax3.set_xticks(range(n_windows))
ax3.set_xticklabels([f"W{i}" for i in range(n_windows)])

# Plot 4: KS p-value over time (log scale)
ax4 = axes[1, 1]
ax4.plot(range(n_windows), ks_pvals, "rs-", linewidth=2, markersize=6)
ax4.axhline(y=0.05, color="red", linestyle="--", label="Significance (p=0.05)")
ax4.set_yscale("log")
ax4.set_title("KS Test p-value Over Time")
ax4.set_xlabel("Time Window")
ax4.set_ylabel("p-value (log scale)")
ax4.legend(fontsize=8)
ax4.set_xticks(range(n_windows))
ax4.set_xticklabels([f"W{i}" for i in range(n_windows)])

plt.tight_layout()
plt.savefig("drift_monitoring.png", dpi=100, bbox_inches="tight")
plt.show()
print("\\nDrift monitoring plot saved to drift_monitoring.png")

# === Summary ===
first_alert = next((i for i, p in enumerate(psi_values) if p > alert_threshold), None)
if first_alert is not None:
    print(f"\\nFirst alert triggered at Window {first_alert} "
          f"(mean shifted by {drift_rate * first_alert:.1f})")
    print("Action: Trigger automated retraining pipeline!")
else:
    print("\\nNo alerts triggered in this simulation.")
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
    {
      title: "Feature Store",
      slug: "feature-store",
      description:
        "Centralised feature repository for consistent training and serving with feature reuse",
      markdownContent: featureStoreMarkdown,
      codeSnippet: featureStoreCode,
      codeLanguage: "python",
    },
    {
      title: "Model Drift & Monitoring",
      slug: "model-drift-monitoring",
      description:
        "Detect data drift and concept drift with KS test, PSI, and automated alerting",
      markdownContent: modelDriftMonitoringMarkdown,
      codeSnippet: modelDriftMonitoringCode,
      codeLanguage: "python",
    },
  ],
};
