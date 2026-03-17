import type { Chapter } from "../types";

const creditRiskMarkdown = `
# Credit Risk Modelling

Credit risk modelling is one of the oldest and most regulated applications of machine learning in banking. At its core, a bank needs to answer: **how likely is this borrower to default, and how much will we lose if they do?**

The Basel regulatory framework formalises this into three key parameters:

## The Basel Risk Parameters

### Probability of Default (PD)

PD estimates the likelihood that a borrower will default within a given time horizon (typically 12 months):

$$\\text{PD} = P(\\text{default within } T \\mid \\text{borrower characteristics } X)$$

PD models are usually built with **logistic regression** for regulatory transparency:

$$\\text{PD}(X) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1 x_1 + \\cdots + \\beta_p x_p)}}$$

### Loss Given Default (LGD)

LGD measures the fraction of exposure that is lost if default occurs:

$$\\text{LGD} = \\frac{\\text{Loss}}{\\text{EAD}} = 1 - \\text{Recovery Rate}$$

LGD typically ranges from 10% (well-collateralised mortgages) to 60%+ (unsecured consumer credit). Models often use **beta regression** since LGD is bounded in $[0, 1]$.

### Exposure at Default (EAD)

EAD estimates the total exposure at the moment of default. For revolving credit (credit cards, lines of credit), this requires estimating how much of the undrawn limit the borrower will use before defaulting:

$$\\text{EAD} = \\text{Drawn Amount} + \\text{CCF} \\times \\text{Undrawn Amount}$$

where CCF (Credit Conversion Factor) is the fraction of undrawn credit expected to be drawn down.

### Expected Loss

The three parameters combine to produce **Expected Loss**:

$$\\text{EL} = \\text{PD} \\times \\text{LGD} \\times \\text{EAD}$$

This drives provisioning (IFRS 9 / CECL) and regulatory capital calculations (Basel III).

## Credit Scorecards

Traditional credit scorecards convert logistic regression coefficients into an additive **points system** that is transparent and auditable.

### Weight of Evidence (WoE)

WoE transforms categorical or binned continuous features into a metric that measures the predictive power of each bin:

$$\\text{WoE}_i = \\ln\\left(\\frac{\\text{Distribution of Events}_i}{\\text{Distribution of Non-Events}_i}\\right) = \\ln\\left(\\frac{p_i^{\\text{bad}} / P^{\\text{bad}}}{p_i^{\\text{good}} / P^{\\text{good}}}\\right)$$

Positive WoE indicates the bin has more "goods" (non-defaults) relative to its share, and vice versa.

### Information Value (IV)

IV measures the overall predictive power of a feature:

$$\\text{IV} = \\sum_{i=1}^{k} \\left(\\text{Distr. Events}_i - \\text{Distr. Non-Events}_i\\right) \\times \\text{WoE}_i$$

| IV Range | Predictive Power |
|----------|-----------------|
| < 0.02 | Not useful |
| 0.02 -- 0.1 | Weak |
| 0.1 -- 0.3 | Medium |
| 0.3 -- 0.5 | Strong |
| > 0.5 | Suspicious (possible overfit) |

### Scorecard Scaling

After fitting logistic regression on WoE-transformed features, coefficients are converted to scorecard points:

$$\\text{Score} = \\text{Offset} + \\text{Factor} \\times \\ln(\\text{Odds})$$

where Offset and Factor are chosen so that a target score (e.g., 600) corresponds to a target odds ratio (e.g., 50:1).

## Application vs Behavioural Scoring

- **Application scoring**: used at the point of credit application. Features are from the application form and credit bureau (income, employment length, existing debt).
- **Behavioural scoring**: used for existing customers. Features include account history (payment patterns, utilisation trends, days past due).

Run the code to build a credit scorecard on synthetic loan data: compute WoE/IV for feature selection, train logistic regression, and output scorecard points.
`;

const creditRiskCode = `import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)
MODE = os.environ.get("ML_CATALOGUE_MODE", "quick")

# === Generate Synthetic Loan Dataset ===
n = 2000 if MODE == "full" else 800
print("=== Credit Risk Scorecard Demo ===")
print(f"Mode: {MODE} | Samples: {n}\\n")

income = np.random.lognormal(mean=10.5, sigma=0.5, size=n)
debt_ratio = np.clip(np.random.beta(2, 5, size=n), 0.01, 0.99)
employment_length = np.random.poisson(lam=5, size=n).clip(0, 30)
age = np.random.normal(40, 12, size=n).clip(21, 75).astype(int)
num_credit_lines = np.random.poisson(lam=3, size=n).clip(0, 15)

# Default probability driven by features
log_odds = (-3.0
    + 1.5 * (debt_ratio - 0.3)
    - 0.8 * np.log(income / 30000)
    - 0.05 * employment_length
    + 0.1 * (num_credit_lines - 3)
    + np.random.normal(0, 0.3, n))
prob_default = 1 / (1 + np.exp(-log_odds))
default = (np.random.uniform(size=n) < prob_default).astype(int)

df = pd.DataFrame({
    "income": np.round(income, 0),
    "debt_ratio": np.round(debt_ratio, 3),
    "employment_length": employment_length,
    "age": age,
    "num_credit_lines": num_credit_lines,
    "default": default,
})
print(f"Default rate: {default.mean():.1%}")

# === WoE / IV Computation ===
def compute_woe_iv(df, feature, target, n_bins=5):
    """Compute WoE and IV for a feature."""
    df_temp = df[[feature, target]].copy()
    # Bin continuous features
    df_temp["bin"] = pd.qcut(df_temp[feature], q=n_bins, duplicates="drop")

    grouped = df_temp.groupby("bin", observed=True)[target].agg(["sum", "count"])
    grouped.columns = ["events", "total"]
    grouped["non_events"] = grouped["total"] - grouped["events"]

    # Avoid division by zero
    grouped["events"] = grouped["events"].clip(lower=0.5)
    grouped["non_events"] = grouped["non_events"].clip(lower=0.5)

    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()

    grouped["dist_events"] = grouped["events"] / total_events
    grouped["dist_non_events"] = grouped["non_events"] / total_non_events
    grouped["woe"] = np.log(grouped["dist_non_events"] / grouped["dist_events"])
    grouped["iv_component"] = (grouped["dist_non_events"] - grouped["dist_events"]) * grouped["woe"]

    iv = grouped["iv_component"].sum()
    return grouped, iv

print("\\n-- Feature Selection via Information Value --")
features = ["income", "debt_ratio", "employment_length", "age", "num_credit_lines"]
iv_results = {}
woe_maps = {}

for feat in features:
    grouped, iv = compute_woe_iv(df, feat, "default")
    iv_results[feat] = iv
    woe_maps[feat] = grouped
    strength = ("Strong" if iv > 0.3 else "Medium" if iv > 0.1
                else "Weak" if iv > 0.02 else "Not useful")
    print(f"  {feat:<22} IV = {iv:.4f}  ({strength})")

# Select features with IV > 0.02
selected = [f for f in features if iv_results[f] > 0.02]
print(f"\\nSelected features: {selected}")

# === WoE Transformation ===
print("\\n-- WoE Transformation --")
def woe_transform(df, feature, target, n_bins=5):
    df_temp = df[[feature, target]].copy()
    df_temp["bin"] = pd.qcut(df_temp[feature], q=n_bins, duplicates="drop")
    grouped = df_temp.groupby("bin", observed=True)[target].agg(["sum", "count"])
    grouped.columns = ["events", "total"]
    grouped["non_events"] = grouped["total"] - grouped["events"]
    grouped["events"] = grouped["events"].clip(lower=0.5)
    grouped["non_events"] = grouped["non_events"].clip(lower=0.5)
    total_e = grouped["events"].sum()
    total_ne = grouped["non_events"].sum()
    grouped["woe"] = np.log((grouped["non_events"] / total_ne) / (grouped["events"] / total_e))

    bin_edges = pd.qcut(df[feature], q=n_bins, duplicates="drop")
    woe_map = dict(zip(grouped.index, grouped["woe"]))
    return bin_edges.map(woe_map)

X_woe = pd.DataFrame()
for feat in selected:
    X_woe[feat + "_woe"] = woe_transform(df, feat, "default")

print(f"  Transformed {len(selected)} features to WoE values")
print(f"  Shape: {X_woe.shape}")

# === Logistic Regression on WoE Features ===
X_train, X_test, y_train, y_test = train_test_split(
    X_woe, df["default"], test_size=0.3, random_state=42
)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"\\n-- Model Performance --")
print(f"  AUC-ROC: {auc:.4f}")

# === Scorecard Points ===
print("\\n-- Scorecard Points --")
target_score = 600
target_odds = 50
pdo = 20  # points to double the odds

factor = pdo / np.log(2)
offset = target_score - factor * np.log(target_odds)
print(f"  Offset = {offset:.1f}, Factor = {factor:.1f}")
print(f"  (Score {target_score} = odds {target_odds}:1, {pdo} pts to double odds)\\n")

coefs = model.coef_[0]
intercept = model.intercept_[0]
n_feats = len(selected)

base_points = offset - factor * intercept
print(f"  {'Feature':<22} {'Coefficient':>12} {'Base Points':>12}")
print(f"  {'-'*48}")
print(f"  {'(intercept)':<22} {intercept:>12.4f} {base_points / n_feats:>12.1f}")
for feat_name, coef in zip(X_woe.columns, coefs):
    pts = -factor * coef
    print(f"  {feat_name:<22} {coef:>12.4f} {pts:>12.1f}")

# === Score a Sample Applicant ===
print("\\n-- Sample Scoring --")
sample = X_test.iloc[0:1]
log_odds_pred = model.decision_function(sample)[0]
score = offset - factor * log_odds_pred
prob = model.predict_proba(sample)[0, 1]
print(f"  Log-odds: {log_odds_pred:.4f}")
print(f"  Credit Score: {score:.0f}")
print(f"  PD: {prob:.4f} ({prob:.1%})")

# === Expected Loss Calculation ===
print("\\n-- Expected Loss Example --")
ead = 25000
lgd = 0.45
el = prob * lgd * ead
print(f"  PD = {prob:.4f}, LGD = {lgd}, EAD = ${ead:,.0f}")
print(f"  Expected Loss = PD x LGD x EAD = ${el:,.2f}")
`;

const fraudDetectionMarkdown = `
# Fraud Detection Pipeline

Fraud detection is a high-stakes ML application where milliseconds matter and the cost of both false positives (blocking legitimate transactions) and false negatives (missing fraud) is significant. Modern systems use a **hybrid architecture** combining rule engines with ML models.

## Rule Engine + ML Hybrid Architecture

### Why Not Pure ML?

- **Regulatory requirements**: some fraud patterns must be caught deterministically (e.g., sanctioned entities)
- **Explainability**: rules provide clear audit trails; regulators can inspect exactly why a transaction was flagged
- **Speed**: simple rules execute in microseconds; ML inference adds latency
- **Cold start**: rules work from day one; ML models need training data

### The Hybrid Flow

$$\\text{Transaction} \\xrightarrow{\\text{rules}} \\begin{cases} \\text{Block (hard rules)} \\\\ \\text{Pass to ML} \\end{cases} \\xrightarrow{\\text{ML score}} \\begin{cases} \\text{Approve} \\\\ \\text{Review queue} \\\\ \\text{Block} \\end{cases}$$

Hard rules catch obvious fraud (e.g., transaction from sanctioned country, card reported stolen). The ML model handles the grey area — transactions that look somewhat suspicious but aren't caught by deterministic rules.

## Feature Engineering for Transaction Data

Feature engineering is the most impactful part of a fraud detection pipeline. Raw transaction fields (amount, merchant, timestamp) are transformed into **behavioural signals**.

### Velocity Features

Count or sum of transactions in rolling time windows:

$$v_{\\text{count}}(t, w) = \\sum_{i : t - w < t_i \\leq t} 1 \\qquad v_{\\text{sum}}(t, w) = \\sum_{i : t - w < t_i \\leq t} a_i$$

where $w$ is the window size (1 hour, 24 hours, 7 days) and $a_i$ is the transaction amount.

### Time-Since-Last Features

The time elapsed since the previous transaction:

$$\\Delta t = t_{\\text{current}} - t_{\\text{previous}}$$

Fraudsters often make rapid sequences of transactions. A sudden drop in $\\Delta t$ is a strong signal.

### Aggregation Windows

For each customer, compute rolling statistics over multiple time horizons:

| Feature | 1 hour | 24 hours | 7 days |
|---------|--------|----------|--------|
| Transaction count | $n_{1h}$ | $n_{24h}$ | $n_{7d}$ |
| Total amount | $\\sum_{1h}$ | $\\sum_{24h}$ | $\\sum_{7d}$ |
| Average amount | $\\mu_{1h}$ | $\\mu_{24h}$ | $\\mu_{7d}$ |
| Max single amount | $\\max_{1h}$ | $\\max_{24h}$ | $\\max_{7d}$ |

### Deviation Features

Compare the current transaction against historical behaviour:

$$z_{\\text{amount}} = \\frac{a_{\\text{current}} - \\mu_{\\text{historical}}}{\\sigma_{\\text{historical}}}$$

A transaction 3+ standard deviations above the customer's norm is suspicious.

## Real-Time vs Batch Scoring

| Aspect | Real-Time | Batch |
|--------|-----------|-------|
| Latency | < 100ms | Hours |
| Use case | Transaction authorisation | Retrospective analysis |
| Features | Pre-computed aggregates | Full history available |
| Model | Optimised for speed | Can use complex ensembles |

## Precision-Recall Tradeoff

In fraud detection, the classes are heavily imbalanced (fraud rate is often < 0.1%). Accuracy is meaningless — a model that always predicts "not fraud" achieves 99.9%+ accuracy.

The key metrics are **precision** and **recall**:

$$\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}} \\qquad \\text{Recall} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$$

The threshold $\\tau$ on the model's predicted probability controls this tradeoff:

$$\\hat{y} = \\begin{cases} \\text{fraud} & \\text{if } P(\\text{fraud} | X) \\geq \\tau \\\\ \\text{legit} & \\text{otherwise} \\end{cases}$$

Lowering $\\tau$ catches more fraud (higher recall) but increases false positives (lower precision). The optimal $\\tau$ depends on the cost ratio of missed fraud vs. false alerts.

Run the code to build a fraud detection pipeline on synthetic transaction data: feature engineering, XGBoost model training, and threshold tuning with precision-recall analysis.
`;

const fraudDetectionCode = `import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_auc_score, classification_report)
from xgboost import XGBClassifier

np.random.seed(42)
MODE = os.environ.get("ML_CATALOGUE_MODE", "quick")

# === Generate Synthetic Transaction Data ===
n_txn = 5000 if MODE == "full" else 2000
fraud_rate = 0.03
print("=== Fraud Detection Pipeline Demo ===")
print(f"Mode: {MODE} | Transactions: {n_txn} | Fraud rate: {fraud_rate:.0%}\\n")

n_customers = n_txn // 10
customer_ids = np.random.randint(0, n_customers, size=n_txn)
timestamps = np.sort(np.random.uniform(0, 30 * 24 * 3600, size=n_txn))  # 30 days in seconds
amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n_txn).clip(1, 10000)
categories = np.random.choice(["grocery", "gas", "online", "restaurant", "atm", "travel"], size=n_txn)

# Inject fraud pattern: rapid high-value transactions
is_fraud = np.zeros(n_txn, dtype=int)
n_fraud = int(n_txn * fraud_rate)
fraud_idx = np.random.choice(n_txn, size=n_fraud, replace=False)
is_fraud[fraud_idx] = 1
amounts[fraud_idx] *= np.random.uniform(3, 8, size=n_fraud)  # higher amounts

df = pd.DataFrame({
    "customer_id": customer_ids,
    "timestamp": timestamps,
    "amount": np.round(amounts, 2),
    "category": categories,
    "is_fraud": is_fraud,
})

print(f"Fraud transactions: {is_fraud.sum()} / {n_txn} ({is_fraud.mean():.1%})")

# === Feature Engineering ===
print("\\n-- Feature Engineering --")

# Sort by customer and time for rolling features
df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

# Time-since-last transaction per customer
df["time_since_last"] = df.groupby("customer_id")["timestamp"].diff().fillna(0)

# Per-customer rolling statistics (using expanding window for simplicity)
cust_stats = df.groupby("customer_id")["amount"].agg(["mean", "std", "count"]).reset_index()
cust_stats.columns = ["customer_id", "cust_avg_amount", "cust_std_amount", "cust_txn_count"]
cust_stats["cust_std_amount"] = cust_stats["cust_std_amount"].fillna(1.0)
df = df.merge(cust_stats, on="customer_id", how="left")

# Amount deviation (z-score relative to customer history)
df["amount_zscore"] = (df["amount"] - df["cust_avg_amount"]) / df["cust_std_amount"].clip(lower=0.01)

# Amount relative to customer average
df["amount_ratio"] = df["amount"] / df["cust_avg_amount"].clip(lower=0.01)

# Hour of day (cyclic)
hour = (df["timestamp"] % 86400) / 3600
df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

# Category encoding
df["is_online"] = (df["category"] == "online").astype(int)
df["is_atm"] = (df["category"] == "atm").astype(int)
df["is_travel"] = (df["category"] == "travel").astype(int)

# Velocity features: count txns per customer in last N seconds window (approx)
df["log_amount"] = np.log1p(df["amount"])
df["time_since_last_log"] = np.log1p(df["time_since_last"])

feature_cols = [
    "amount", "log_amount", "time_since_last", "time_since_last_log",
    "cust_avg_amount", "cust_std_amount", "cust_txn_count",
    "amount_zscore", "amount_ratio",
    "hour_sin", "hour_cos",
    "is_online", "is_atm", "is_travel",
]
print(f"  Engineered {len(feature_cols)} features:")
for f in feature_cols:
    print(f"    - {f}")

X = df[feature_cols].values
y = df["is_fraud"].values

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === XGBoost Model ===
print("\\n-- Training XGBoost --")
scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
n_estimators = 200 if MODE == "full" else 80

model = XGBClassifier(
    n_estimators=n_estimators,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos,
    random_state=42,
    eval_metric="aucpr",
    verbosity=0,
)
model.fit(X_train, y_train)
print(f"  Trained with {n_estimators} trees, scale_pos_weight={scale_pos:.1f}")

y_prob = model.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_prob)
auc_pr = average_precision_score(y_test, y_prob)
print(f"  AUC-ROC: {auc_roc:.4f}")
print(f"  AUC-PR:  {auc_pr:.4f}")

# === Feature Importance ===
print("\\n-- Top Features (by gain) --")
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for i in range(min(8, len(feature_cols))):
    idx = sorted_idx[i]
    print(f"  {feature_cols[idx]:<24} {importances[idx]:.4f}")

# === Threshold Tuning ===
print("\\n-- Threshold Tuning (Precision-Recall) --")
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find thresholds at key precision levels
print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
print(f"  {'-'*42}")
for target_prec in [0.3, 0.5, 0.7, 0.9]:
    mask = precisions >= target_prec
    if mask.any():
        idx = np.where(mask)[0][0]
        p, r = precisions[idx], recalls[idx]
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        t = thresholds[min(idx, len(thresholds) - 1)]
        print(f"  {t:>10.4f} {p:>10.3f} {r:>10.3f} {f1:>8.3f}")

# === Operational Decision ===
# Pick threshold that maximises F1
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores[:-1])  # last element is sentinel
best_thresh = thresholds[best_idx]
print(f"\\n  Optimal threshold (max F1): {best_thresh:.4f}")
print(f"  Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}")

y_pred_opt = (y_prob >= best_thresh).astype(int)
print(f"\\n-- Classification Report (threshold={best_thresh:.4f}) --")
print(classification_report(y_test, y_pred_opt, target_names=["Legit", "Fraud"]))

# === Rule Engine + ML Summary ===
print("-- Pipeline Summary --")
n_test = len(y_test)
n_flagged = y_pred_opt.sum()
print(f"  Transactions scored: {n_test}")
print(f"  ML-flagged for review: {n_flagged} ({n_flagged/n_test:.1%})")
print(f"  True fraud caught: {((y_pred_opt == 1) & (y_test == 1)).sum()} / {y_test.sum()}")
print(f"  False alerts: {((y_pred_opt == 1) & (y_test == 0)).sum()}")
`;

const amlMarkdown = `
# Anti-Money Laundering (AML)

Anti-Money Laundering (AML) compliance is a legal requirement for financial institutions worldwide. ML-based transaction monitoring systems help banks detect suspicious activity that may indicate money laundering, terrorist financing, or sanctions evasion.

## Transaction Monitoring Fundamentals

Traditional AML systems are **rule-based**: they flag transactions exceeding thresholds (e.g., cash deposits over $10,000) or matching predefined patterns (e.g., structuring — breaking large amounts into smaller ones to avoid reporting thresholds).

The problem with pure rule-based systems:
- **High false positive rates**: typically 95-99% of alerts are false positives
- **Static rules**: criminals adapt faster than rules are updated
- **No pattern generalisation**: rules catch exact patterns but miss novel variations

ML augments rules by learning complex patterns from historical data:

$$\\text{Alert} = \\begin{cases} 1 & \\text{if } \\text{Rule}(x) = 1 \\text{ OR } f_{\\text{ML}}(x) > \\tau \\\\ 0 & \\text{otherwise} \\end{cases}$$

## Anomaly Detection for AML

Since labelled money laundering cases are rare, AML often relies on **anomaly detection** — identifying transactions that deviate from expected behaviour.

### Statistical Approaches

For each customer, establish a baseline profile and flag deviations:

$$z_i = \\frac{x_i - \\mu_{\\text{customer}}}{\\sigma_{\\text{customer}}}$$

A transaction with $|z_i| > 3$ is unusual relative to the customer's history.

### Isolation Forest

Isolation Forest detects anomalies by measuring how easy it is to isolate a data point. Anomalies require fewer random splits to isolate:

$$s(x, n) = 2^{-\\frac{E[h(x)]}{c(n)}}$$

where $h(x)$ is the path length to isolate point $x$ and $c(n)$ is the average path length in a binary search tree of $n$ samples. Scores close to 1 indicate anomalies.

### Clustering-Based Detection

Cluster transactions by behaviour and flag outliers — points that don't belong to any cluster or belong to small, unusual clusters:

$$\\text{suspicious}(x) = \\begin{cases} 1 & \\text{if } \\min_j \\|x - c_j\\| > \\delta \\\\ 1 & \\text{if } |C_{\\text{nearest}}| < n_{\\min} \\\\ 0 & \\text{otherwise} \\end{cases}$$

## Network/Graph-Based AML

Money laundering typically involves **networks** of accounts moving funds in complex patterns (layering). Graph analysis reveals structures invisible to transaction-level monitoring.

### Key Graph Patterns

- **Circular flows**: $A \\to B \\to C \\to A$ — money returns to the originator through intermediaries
- **Fan-out / fan-in**: one account distributes to many, then a different account collects from many
- **Rapid pass-through**: funds enter and leave an account within a short time window:

$$\\text{pass-through}(v) = \\frac{\\min(\\text{inflow}(v, w), \\text{outflow}(v, w))}{\\max(\\text{inflow}(v, w), \\text{outflow}(v, w))}$$

A ratio close to 1 with a short window $w$ suggests the account is merely a conduit.

### Graph Metrics for Suspicion

- **Betweenness centrality**: accounts that sit on many shortest paths between other accounts may be facilitating layering
- **PageRank on risk**: propagate suspicion scores through the network — accounts connected to known suspicious entities inherit higher risk
- **Community detection**: identify tightly connected clusters of accounts that transact primarily with each other

## Regulatory Context

AML regulations (BSA/USA PATRIOT Act, EU AMLD6, FATF recommendations) require financial institutions to:
1. Monitor customer transactions for suspicious activity
2. File Suspicious Activity Reports (SARs) for flagged transactions
3. Maintain records of Customer Due Diligence (CDD) and Enhanced Due Diligence (EDD)
4. Screen against sanctions lists (OFAC, EU, UN)

Run the code to build a simple AML transaction monitoring system: anomaly detection on synthetic wire transfers with cluster-based suspicious activity identification.
`;

const amlCode = `import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

np.random.seed(42)
MODE = os.environ.get("ML_CATALOGUE_MODE", "quick")

# === Generate Synthetic Wire Transfer Data ===
n_txn = 3000 if MODE == "full" else 1200
n_accounts = 60 if MODE == "full" else 30
print("=== AML Transaction Monitoring Demo ===")
print(f"Mode: {MODE} | Transactions: {n_txn} | Accounts: {n_accounts}\\n")

senders = np.random.randint(0, n_accounts, size=n_txn)
receivers = np.random.randint(0, n_accounts, size=n_txn)
# Avoid self-transfers
mask = senders == receivers
receivers[mask] = (receivers[mask] + 1) % n_accounts

amounts = np.random.lognormal(mean=7, sigma=1.5, size=n_txn).clip(100, 500000)
timestamps = np.sort(np.random.uniform(0, 30 * 86400, size=n_txn))
countries = np.random.choice(
    ["US", "UK", "DE", "SG", "HK", "CH", "KY", "PA", "BZ"],
    size=n_txn,
    p=[0.30, 0.15, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05, 0.05]
)

# Inject suspicious patterns
n_suspicious = int(n_txn * 0.04)
sus_idx = np.random.choice(n_txn, size=n_suspicious, replace=False)

# Pattern 1: structuring (amounts just below reporting threshold)
n_struct = n_suspicious // 3
amounts[sus_idx[:n_struct]] = np.random.uniform(8000, 9900, size=n_struct)

# Pattern 2: rapid pass-through (high amount, offshore)
n_passthru = n_suspicious // 3
amounts[sus_idx[n_struct:n_struct + n_passthru]] *= 5
countries[sus_idx[n_struct:n_struct + n_passthru]] = np.random.choice(["KY", "PA", "BZ"], size=n_passthru)

# Pattern 3: round-trip (circular flows)
n_roundtrip = n_suspicious - n_struct - n_passthru
for i in range(n_roundtrip):
    idx = sus_idx[n_struct + n_passthru + i]
    senders[idx] = idx % n_accounts
    receivers[idx] = (idx + 1) % n_accounts

df = pd.DataFrame({
    "sender": [f"ACC_{s:03d}" for s in senders],
    "receiver": [f"ACC_{r:03d}" for r in receivers],
    "amount": np.round(amounts, 2),
    "timestamp": timestamps,
    "country": countries,
})

print(f"Generated {n_txn} wire transfers")
print(f"Injected ~{n_suspicious} suspicious transactions")

# === Feature Engineering ===
print("\\n-- Feature Engineering --")

# Account-level features (for senders)
acct_features = df.groupby("sender").agg(
    txn_count=("amount", "count"),
    total_amount=("amount", "sum"),
    avg_amount=("amount", "mean"),
    std_amount=("amount", "std"),
    max_amount=("amount", "max"),
    n_unique_receivers=("receiver", "nunique"),
    n_unique_countries=("country", "nunique"),
).reset_index()
acct_features["std_amount"] = acct_features["std_amount"].fillna(0)

# Structuring indicator: count of transactions near reporting threshold (8k-10k)
near_threshold = df[(df["amount"] >= 8000) & (df["amount"] <= 10000)]
struct_counts = near_threshold.groupby("sender").size().reset_index(name="near_threshold_count")
acct_features = acct_features.merge(struct_counts, on="sender", how="left")
acct_features["near_threshold_count"] = acct_features["near_threshold_count"].fillna(0)

# High-risk country ratio
high_risk = {"KY", "PA", "BZ"}
df["is_high_risk_country"] = df["country"].isin(high_risk).astype(int)
hr_ratio = df.groupby("sender")["is_high_risk_country"].mean().reset_index(name="high_risk_ratio")
acct_features = acct_features.merge(hr_ratio, on="sender", how="left")

feature_cols = [
    "txn_count", "total_amount", "avg_amount", "std_amount", "max_amount",
    "n_unique_receivers", "n_unique_countries", "near_threshold_count", "high_risk_ratio"
]
print(f"  Computed {len(feature_cols)} account-level features")

X = acct_features[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Anomaly Detection: Isolation Forest ===
print("\\n-- Anomaly Detection (Isolation Forest) --")
iso_forest = IsolationForest(
    n_estimators=100 if MODE == "full" else 50,
    contamination=0.1,
    random_state=42,
)
acct_features["anomaly_score"] = iso_forest.fit_predict(X_scaled)
acct_features["anomaly_label"] = (acct_features["anomaly_score"] == -1).astype(int)

n_anomalies = acct_features["anomaly_label"].sum()
print(f"  Flagged {n_anomalies} / {len(acct_features)} accounts as anomalous")

# Show flagged accounts
print(f"\\n  {'Account':<12} {'Txns':>5} {'Total Amt':>12} {'Avg Amt':>10} "
      f"{'Near Thresh':>12} {'HR Ratio':>9}")
print(f"  {'-'*65}")
flagged = acct_features[acct_features["anomaly_label"] == 1].sort_values("total_amount", ascending=False)
for _, row in flagged.head(8).iterrows():
    print(f"  {row['sender']:<12} {row['txn_count']:>5.0f} "
          f"{row['total_amount']:>12,.0f} {row['avg_amount']:>10,.0f} "
          f"{row['near_threshold_count']:>12.0f} {row['high_risk_ratio']:>9.2f}")

# === Clustering: DBSCAN ===
print("\\n-- Suspicious Cluster Detection (DBSCAN) --")
clustering = DBSCAN(eps=1.2, min_samples=2)
acct_features["cluster"] = clustering.fit_predict(X_scaled)

n_clusters = acct_features["cluster"].max() + 1
n_noise = (acct_features["cluster"] == -1).sum()
print(f"  Found {n_clusters} clusters, {n_noise} noise points (potential outliers)")

# Identify small suspicious clusters
cluster_sizes = acct_features[acct_features["cluster"] >= 0]["cluster"].value_counts()
small_clusters = cluster_sizes[cluster_sizes <= 3].index.tolist()

if small_clusters:
    print(f"\\n  Small clusters (size <= 3) — potentially suspicious:")
    for cl in small_clusters[:5]:
        members = acct_features[acct_features["cluster"] == cl]
        avg_amt = members["avg_amount"].mean()
        hr = members["high_risk_ratio"].mean()
        accounts = ", ".join(members["sender"].tolist())
        print(f"    Cluster {cl}: [{accounts}] avg_amt={avg_amt:,.0f} hr_ratio={hr:.2f}")

# === Network Analysis (Simple) ===
print("\\n-- Network Analysis (Transaction Graph) --")
# Build edge weights (total flow between account pairs)
edges = df.groupby(["sender", "receiver"])["amount"].agg(["sum", "count"]).reset_index()
edges.columns = ["sender", "receiver", "total_flow", "n_transfers"]

# Find accounts with high fan-out or fan-in
fan_out = edges.groupby("sender")["receiver"].nunique().reset_index(name="fan_out")
fan_in = edges.groupby("receiver")["sender"].nunique().reset_index(name="fan_in")
fan_out.columns = ["account", "fan_out"]
fan_in.columns = ["account", "fan_in"]

network_stats = fan_out.merge(fan_in, on="account", how="outer").fillna(0)
network_stats["fan_total"] = network_stats["fan_out"] + network_stats["fan_in"]

# Flag high connectivity accounts
high_conn = network_stats.nlargest(5, "fan_total")
print(f"  Top connected accounts (potential hubs):")
print(f"  {'Account':<12} {'Fan-out':>8} {'Fan-in':>8} {'Total':>8}")
print(f"  {'-'*40}")
for _, row in high_conn.iterrows():
    print(f"  {row['account']:<12} {row['fan_out']:>8.0f} {row['fan_in']:>8.0f} {row['fan_total']:>8.0f}")

# === Combined Risk Score ===
print("\\n-- Combined Risk Scoring --")
# Merge anomaly and network info
risk = acct_features[["sender", "anomaly_label", "near_threshold_count",
                       "high_risk_ratio", "total_amount"]].copy()
risk.columns = ["account", "is_anomaly", "structuring_signals",
                "high_risk_ratio", "total_amount"]

# Simple composite score
risk["risk_score"] = (
    risk["is_anomaly"] * 40
    + (risk["structuring_signals"] > 0).astype(int) * 25
    + (risk["high_risk_ratio"] > 0.3).astype(int) * 20
    + (risk["total_amount"] > risk["total_amount"].quantile(0.9)).astype(int) * 15
)

high_risk = risk[risk["risk_score"] >= 40].sort_values("risk_score", ascending=False)
print(f"  High-risk accounts (score >= 40): {len(high_risk)}")
print(f"\\n  {'Account':<12} {'Risk Score':>11} {'Anomaly':>8} {'Structuring':>12} {'HR Ratio':>9}")
print(f"  {'-'*56}")
for _, row in high_risk.head(8).iterrows():
    print(f"  {row['account']:<12} {row['risk_score']:>11.0f} "
          f"{'Yes' if row['is_anomaly'] else 'No':>8} "
          f"{row['structuring_signals']:>12.0f} {row['high_risk_ratio']:>9.2f}")

print(f"\\n  These accounts would be escalated for SAR review.")
`;

const modelRiskMarkdown = `
# Model Risk Management

Model Risk Management (MRM) is the framework by which financial institutions govern the development, validation, and ongoing use of quantitative models. It exists because **models can be wrong** — and in banking, wrong models can lead to mispriced risk, regulatory penalties, and systemic failures.

## Regulatory Frameworks

### SR 11-7 (Federal Reserve / OCC, USA)

The foundational guidance on model risk in US banking, issued in 2011. Key principles:

1. **Model risk** is defined as the potential for adverse consequences from decisions based on incorrect or misused model outputs
2. Banks must maintain a **model inventory** — a comprehensive catalogue of all models in use
3. Every model requires **independent validation** before deployment
4. Senior management and boards bear **accountability** for model risk

SR 11-7 defines a model broadly: *"A quantitative method, system, or approach that applies statistical, economic, financial, or mathematical theories, techniques, and assumptions to process input data into quantitative estimates."*

### SS1/23 (PRA / Bank of England, UK)

The PRA's 2023 supervisory statement updates model risk expectations for UK banks:

- Requires a **Model Risk Management Framework (MRMF)** with clear governance
- Introduces **model tiering** — models are classified by materiality:
  - **Tier 1**: High-impact models (regulatory capital, IFRS 9 provisioning)
  - **Tier 2**: Medium-impact models (pricing, stress testing)
  - **Tier 3**: Lower-impact models (operational, management reporting)
- Mandates a **Chief Model Risk Officer** or equivalent role
- Requires annual **model risk appetite statements** approved by the board

### Key Differences

| Aspect | SR 11-7 (US) | SS1/23 (UK) |
|--------|-------------|-------------|
| Scope | All models | All models, with tiering |
| Governance | Senior management oversight | Board-level accountability, CMRO |
| Validation | Independent validation | Tiered validation intensity |
| Inventory | Required | Required with materiality assessment |
| Reporting | To senior management | To board and PRA |

## Model Validation Lifecycle

### Phase 1: Model Development

The development team builds and tests the model. Documentation must capture:

- **Conceptual soundness**: Is the theoretical approach appropriate? Does the model specification match the underlying economics?

$$\\text{Model}: Y = f(X; \\theta) + \\varepsilon$$

- **Data quality**: Are the training data representative? Are there selection biases?
- **Methodology**: Full specification of the algorithm, assumptions, and limitations
- **Performance**: In-sample and out-of-sample metrics on development data

### Phase 2: Independent Validation

A team **independent of the development team** reviews the model. Validation activities include:

**Conceptual review** — evaluate whether the model's theoretical foundations are sound:

- Are the assumptions reasonable?
- Is the functional form appropriate?
- Are there known limitations of this approach?

**Replication testing** — independently reproduce the model's results:

$$\\text{Replication test}: |\\hat{Y}_{\\text{validator}} - \\hat{Y}_{\\text{developer}}| < \\epsilon$$

**Benchmarking** — compare against alternative approaches:

$$\\text{Benchmark}: \\text{metric}(f_{\\text{proposed}}) \\geq \\text{metric}(f_{\\text{benchmark}})$$

**Sensitivity analysis** — test how model outputs change with input perturbations:

$$\\text{Sensitivity}_{x_j} = \\frac{\\partial f}{\\partial x_j} \\approx \\frac{f(x + \\Delta e_j) - f(x - \\Delta e_j)}{2\\Delta}$$

**Outcomes analysis** — compare model predictions against actual outcomes (backtesting):

$$\\text{Backtest}: \\text{Compare } \\hat{Y}_{t} \\text{ vs } Y_{t} \\text{ over historical periods}$$

### Phase 3: Ongoing Monitoring

After deployment, models must be continuously monitored:

- **Performance monitoring**: track key metrics (KS, Gini, AUC) over time with alert thresholds
- **Stability monitoring**: use PSI/CSI to detect input or output drift
- **Trigger-based reviews**: material changes in portfolio, regulation, or economic conditions trigger re-validation
- **Annual review**: full re-validation at least annually for Tier 1 models

## Champion-Challenger Framework

When replacing a production model, banks use a **champion-challenger** approach:

$$\\text{Champion}: f_{\\text{current}}(X) \\quad \\text{vs} \\quad \\text{Challenger}: f_{\\text{new}}(X)$$

The challenger must demonstrate **statistically significant improvement** over the champion across multiple dimensions:

1. **Discriminatory power**: does the challenger rank-order risk better?

$$\\text{Gini}_{\\text{challenger}} > \\text{Gini}_{\\text{champion}}$$

2. **Calibration**: are predicted probabilities accurate?

$$\\text{For each rating grade } g: \\; |\\text{PD}_{\\text{predicted}}(g) - \\text{PD}_{\\text{observed}}(g)| < \\delta$$

3. **Stability**: is the challenger robust across time periods and segments?

4. **Explainability**: can the challenger's predictions be explained to stakeholders and regulators?

The transition period typically involves **parallel running** — both models score all accounts simultaneously for 3-6 months, and results are compared before switching.

## Model Documentation Requirements

Regulatory-compliant model documentation typically includes:

### Model Development Document (MDD)
- Purpose and scope
- Data sources and preparation
- Methodology and assumptions
- Variable selection and rationale
- Performance metrics and benchmarks
- Known limitations

### Model Validation Report (MVR)
- Scope of validation activities
- Findings and severity ratings (High / Medium / Low)
- Replication results
- Benchmark comparisons
- Recommendations and conditions of approval

### Ongoing Monitoring Report (OMR)
- Quarterly/monthly performance metrics
- Drift analysis (PSI, CSI)
- Trigger event assessment
- Remediation actions taken

## Key Metrics for Model Governance

| Metric | Purpose | Threshold (typical) |
|--------|---------|-------------------|
| Gini / AUC | Discriminatory power | Gini > 0.40 for PD models |
| KS Statistic | Separation of good/bad | KS > 0.30 for scorecards |
| PSI | Population stability | PSI < 0.20 |
| Hosmer-Lemeshow | Calibration goodness-of-fit | p-value > 0.05 |
| Binomial test | Observed vs predicted default rate | Within confidence interval |
`;

const bankingMetricsMarkdown = `
# Banking-Specific Metrics

Banking model validation relies on specialised metrics that go beyond standard ML evaluation. These metrics are deeply embedded in regulatory requirements and are computed routinely during model development, validation, and ongoing monitoring.

## KS Statistic (Kolmogorov-Smirnov)

The KS statistic measures the **maximum separation** between the cumulative distribution functions (CDFs) of the positive and negative classes. In credit risk, it quantifies how well a model separates defaulters from non-defaulters.

$$\\text{KS} = \\max_s \\left| F_{\\text{bad}}(s) - F_{\\text{good}}(s) \\right|$$

where $F_{\\text{bad}}(s)$ is the CDF of model scores for defaulters and $F_{\\text{good}}(s)$ is the CDF for non-defaulters, both evaluated at score threshold $s$.

### Interpretation

| KS Value | Interpretation |
|----------|---------------|
| < 0.20 | Poor separation |
| 0.20 -- 0.40 | Acceptable |
| 0.40 -- 0.60 | Good |
| 0.60 -- 0.75 | Very good |
| > 0.75 | Excellent (check for overfit) |

The KS statistic is popular in banking because:
- It is **threshold-independent** — it evaluates the model across all possible cut-offs
- It has a direct **visual interpretation** — the KS value is the widest gap between the two CDFs
- It is required by many regulators for scorecard validation

## Gini Coefficient (from ROC)

The Gini coefficient in credit risk is derived from the ROC curve and measures overall discriminatory power:

$$\\text{Gini} = 2 \\times \\text{AUC} - 1$$

where AUC is the Area Under the ROC Curve. Equivalently, Gini equals the area between the ROC curve and the diagonal:

$$\\text{Gini} = \\frac{\\text{Area between ROC and diagonal}}{\\text{Area of triangle above diagonal}} = 2 \\times \\text{AUC} - 1$$

### Relationship to AUC

| AUC | Gini | Interpretation |
|-----|------|---------------|
| 0.50 | 0.00 | Random model |
| 0.70 | 0.40 | Acceptable |
| 0.80 | 0.60 | Good |
| 0.90 | 0.80 | Excellent |
| 1.00 | 1.00 | Perfect |

The Gini coefficient is the standard metric for comparing credit models in European banking (especially under ECB/PRA supervision). US banks more commonly report AUC directly, but the information content is identical.

## PSI (Population Stability Index)

PSI measures **how much a distribution has shifted** between two time periods (typically development sample vs. recent production data). It is the primary metric for detecting population drift.

$$\\text{PSI} = \\sum_{i=1}^{k} (A_i - E_i) \\times \\ln\\left(\\frac{A_i}{E_i}\\right)$$

where:
- $A_i$ = proportion of the **actual** (current) population in bin $i$
- $E_i$ = proportion of the **expected** (development) population in bin $i$
- $k$ = number of bins (typically 10 decile bins)

### Properties

- PSI is always $\\geq 0$ (it is a symmetrised KL divergence)
- PSI = 0 means the distributions are identical
- PSI is not symmetric in general, but is approximately symmetric for small shifts

### Interpretation

| PSI | Interpretation | Action |
|-----|---------------|--------|
| < 0.10 | No significant shift | Continue monitoring |
| 0.10 -- 0.25 | Moderate shift | Investigate root cause |
| > 0.25 | Significant shift | Model review / retrain |

PSI is computed on the **model score distribution** (output PSI) and on **individual features** (input PSI / CSI).

## CSI (Characteristic Stability Index)

CSI is **PSI applied to individual input features** rather than the model score. It uses the same formula but is computed feature-by-feature to identify **which inputs are drifting**:

$$\\text{CSI}_j = \\sum_{i=1}^{k} (A_{ij} - E_{ij}) \\times \\ln\\left(\\frac{A_{ij}}{E_{ij}}\\right)$$

where the subscript $j$ indexes the feature.

### Why CSI Matters

When PSI flags overall score drift, CSI pinpoints the **source of the drift**. This is crucial for remediation:

- If one feature has high CSI but others are stable, the issue may be an upstream data pipeline change
- If multiple features drift simultaneously, it may indicate a genuine population shift (e.g., economic recession changing borrower profiles)
- CSI helps distinguish **data quality issues** from **real-world changes**

### CSI Thresholds

The same thresholds as PSI apply:

| CSI | Interpretation |
|-----|---------------|
| < 0.10 | Stable |
| 0.10 -- 0.25 | Moderate drift — investigate |
| > 0.25 | Significant drift — action needed |

## Using These Metrics Together

In practice, a model monitoring dashboard tracks all four metrics:

1. **KS and Gini/AUC** — has the model's discriminatory power degraded?
2. **PSI (score)** — has the overall score distribution shifted?
3. **CSI (features)** — which specific inputs are driving the shift?

A typical monitoring workflow:

$$\\text{PSI} > 0.25 \\implies \\text{check CSI for each feature} \\implies \\text{identify drift source} \\implies \\text{retrain or recalibrate}$$

Run the code to compute all four metrics (KS, Gini, PSI, CSI) on a synthetic credit scoring dataset with simulated population drift, and visualise the KS curve, ROC/Gini, and PSI distribution comparison.
`;

const bankingMetricsCode = `import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

np.random.seed(42)
MODE = os.environ.get("ML_CATALOGUE_MODE", "quick")

n_dev = 3000 if MODE == "full" else 1000
n_prod = 3000 if MODE == "full" else 1000
print("=== Banking-Specific Metrics Demo ===")
print(f"Mode: {MODE} | Dev samples: {n_dev} | Prod samples: {n_prod}\\n")

# === Generate Synthetic Credit Scoring Data ===
def generate_credit_data(n, income_mean=10.5, debt_shift=0.0, seed=42):
    rng = np.random.RandomState(seed)
    income = rng.lognormal(mean=income_mean, sigma=0.4, size=n)
    debt_ratio = np.clip(rng.beta(2, 5, size=n) + debt_shift, 0.01, 0.99)
    emp_length = rng.poisson(lam=5, size=n).clip(0, 25)
    age = rng.normal(38, 10, size=n).clip(21, 70).astype(int)

    log_odds = (-2.5
        + 1.8 * (debt_ratio - 0.3)
        - 0.6 * np.log(income / 30000)
        - 0.04 * emp_length
        + rng.normal(0, 0.3, n))
    prob = 1 / (1 + np.exp(-log_odds))
    default = (rng.uniform(size=n) < prob).astype(int)
    return pd.DataFrame({
        "income": income, "debt_ratio": debt_ratio,
        "emp_length": emp_length, "age": age, "default": default
    })

# Development data (baseline)
dev_data = generate_credit_data(n_dev, seed=42)
# Production data (with drift: higher debt ratios, slightly lower income)
prod_data = generate_credit_data(n_prod, income_mean=10.3, debt_shift=0.05, seed=99)

print(f"Dev default rate:  {dev_data['default'].mean():.1%}")
print(f"Prod default rate: {prod_data['default'].mean():.1%}")

# === Train Model on Dev Data ===
features = ["income", "debt_ratio", "emp_length", "age"]
X_dev = dev_data[features].values
y_dev = dev_data["default"].values
X_prod = prod_data[features].values
y_prod = prod_data["default"].values

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_dev, y_dev)

scores_dev = model.predict_proba(X_dev)[:, 1]
scores_prod = model.predict_proba(X_prod)[:, 1]

# ============================================================
# 1. KS STATISTIC
# ============================================================
print("\\n" + "=" * 50)
print("1. KS STATISTIC")
print("=" * 50)

def compute_ks(y_true, y_scores):
    """Compute KS statistic and the score at which it occurs."""
    thresholds = np.sort(np.unique(y_scores))
    ks_max = 0
    ks_threshold = 0
    cum_bad = []
    cum_good = []

    bad_scores = np.sort(y_scores[y_true == 1])
    good_scores = np.sort(y_scores[y_true == 0])

    all_thresholds = np.linspace(y_scores.min(), y_scores.max(), 200)
    for t in all_thresholds:
        cdf_bad = np.mean(bad_scores <= t)
        cdf_good = np.mean(good_scores <= t)
        cum_bad.append(cdf_bad)
        cum_good.append(cdf_good)
        ks = abs(cdf_bad - cdf_good)
        if ks > ks_max:
            ks_max = ks
            ks_threshold = t

    return ks_max, ks_threshold, all_thresholds, np.array(cum_bad), np.array(cum_good)

ks_dev, ks_thresh_dev, thresholds_dev, cdf_bad_dev, cdf_good_dev = compute_ks(y_dev, scores_dev)
ks_prod, ks_thresh_prod, thresholds_prod, cdf_bad_prod, cdf_good_prod = compute_ks(y_prod, scores_prod)

print(f"  Dev KS:  {ks_dev:.4f} (at score {ks_thresh_dev:.4f})")
print(f"  Prod KS: {ks_prod:.4f} (at score {ks_thresh_prod:.4f})")
quality_dev = ("Good" if ks_dev > 0.4 else "Acceptable" if ks_dev > 0.2 else "Poor")
quality_prod = ("Good" if ks_prod > 0.4 else "Acceptable" if ks_prod > 0.2 else "Poor")
print(f"  Dev quality:  {quality_dev}")
print(f"  Prod quality: {quality_prod}")

# ============================================================
# 2. GINI COEFFICIENT
# ============================================================
print("\\n" + "=" * 50)
print("2. GINI COEFFICIENT (from ROC)")
print("=" * 50)

auc_dev = roc_auc_score(y_dev, scores_dev)
auc_prod = roc_auc_score(y_prod, scores_prod)
gini_dev = 2 * auc_dev - 1
gini_prod = 2 * auc_prod - 1

print(f"  Dev  AUC: {auc_dev:.4f}  ->  Gini: {gini_dev:.4f}")
print(f"  Prod AUC: {auc_prod:.4f}  ->  Gini: {gini_prod:.4f}")
print(f"  Gini degradation: {gini_dev - gini_prod:.4f}")

# ============================================================
# 3. PSI (Population Stability Index)
# ============================================================
print("\\n" + "=" * 50)
print("3. PSI (Population Stability Index)")
print("=" * 50)

def compute_psi(expected, actual, bins=10):
    """Compute PSI between two distributions."""
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )
    exp_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    act_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)

    # Avoid zero bins
    exp_counts = np.maximum(exp_counts, 0.5)
    act_counts = np.maximum(act_counts, 0.5)

    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return psi, exp_pct, act_pct, breakpoints

score_psi, dev_pct, prod_pct, psi_bins = compute_psi(scores_dev, scores_prod)
interpretation = ("No significant shift" if score_psi < 0.1
                  else "Moderate shift" if score_psi < 0.25
                  else "Significant shift")
print(f"  Score PSI: {score_psi:.4f} ({interpretation})")

print(f"\\n  {'Bin':>4} {'Dev %':>8} {'Prod %':>8} {'Diff':>8} {'Contribution':>13}")
print(f"  {'-'*45}")
for i in range(len(dev_pct)):
    diff = prod_pct[i] - dev_pct[i]
    contrib = (prod_pct[i] - dev_pct[i]) * np.log(prod_pct[i] / dev_pct[i])
    print(f"  {i+1:>4} {dev_pct[i]:>8.3f} {prod_pct[i]:>8.3f} {diff:>+8.3f} {contrib:>13.5f}")

# ============================================================
# 4. CSI (Characteristic Stability Index)
# ============================================================
print("\\n" + "=" * 50)
print("4. CSI (Characteristic Stability Index)")
print("=" * 50)

print(f"  {'Feature':<16} {'CSI':>8} {'Status':<22}")
print(f"  {'-'*48}")
csi_values = {}
for feat in features:
    csi, _, _, _ = compute_psi(dev_data[feat].values, prod_data[feat].values)
    csi_values[feat] = csi
    status = ("Stable" if csi < 0.1
              else "Moderate drift" if csi < 0.25
              else "Significant drift")
    print(f"  {feat:<16} {csi:>8.4f} {status:<22}")

# Identify drift source
max_csi_feat = max(csi_values, key=csi_values.get)
print(f"\\n  Primary drift source: '{max_csi_feat}' (CSI = {csi_values[max_csi_feat]:.4f})")
if score_psi > 0.1:
    print(f"  Recommendation: Investigate '{max_csi_feat}' for upstream data changes")

# ============================================================
# VISUALISATION
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: KS Curve (Dev)
ax1 = axes[0, 0]
ax1.plot(thresholds_dev, cdf_bad_dev, "r-", linewidth=2, label="CDF Bad (defaulters)")
ax1.plot(thresholds_dev, cdf_good_dev, "b-", linewidth=2, label="CDF Good (non-defaulters)")
ks_idx = np.argmax(np.abs(cdf_bad_dev - cdf_good_dev))
ax1.vlines(thresholds_dev[ks_idx], cdf_good_dev[ks_idx], cdf_bad_dev[ks_idx],
           colors="green", linewidth=2, linestyles="--", label=f"KS = {ks_dev:.3f}")
ax1.set_title(f"KS Curve (Dev) — KS = {ks_dev:.3f}")
ax1.set_xlabel("Model Score")
ax1.set_ylabel("Cumulative Proportion")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: ROC Curve / Gini
ax2 = axes[0, 1]
fpr_dev, tpr_dev, _ = roc_curve(y_dev, scores_dev)
fpr_prod, tpr_prod, _ = roc_curve(y_prod, scores_prod)
ax2.plot(fpr_dev, tpr_dev, "b-", linewidth=2, label=f"Dev (Gini={gini_dev:.3f})")
ax2.plot(fpr_prod, tpr_prod, "r--", linewidth=2, label=f"Prod (Gini={gini_prod:.3f})")
ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
ax2.fill_between(fpr_dev, fpr_dev, tpr_dev, alpha=0.1, color="blue")
ax2.set_title("ROC Curve — Gini Comparison")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: PSI Distribution Comparison
ax3 = axes[1, 0]
bin_centers = (psi_bins[:-1] + psi_bins[1:]) / 2
width = (psi_bins[1] - psi_bins[0]) * 0.35
ax3.bar(bin_centers - width/2, dev_pct, width=width, alpha=0.7, color="blue", label="Dev (Expected)")
ax3.bar(bin_centers + width/2, prod_pct, width=width, alpha=0.7, color="red", label="Prod (Actual)")
ax3.set_title(f"PSI Score Distribution — PSI = {score_psi:.4f}")
ax3.set_xlabel("Model Score")
ax3.set_ylabel("Proportion")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: CSI by Feature
ax4 = axes[1, 1]
feat_names = list(csi_values.keys())
csi_vals = list(csi_values.values())
colors = ["green" if v < 0.1 else "orange" if v < 0.25 else "red" for v in csi_vals]
bars = ax4.barh(feat_names, csi_vals, color=colors, edgecolor="black", alpha=0.8)
ax4.axvline(x=0.1, color="orange", linestyle="--", linewidth=1.5, label="Moderate (0.10)")
ax4.axvline(x=0.25, color="red", linestyle="--", linewidth=1.5, label="Significant (0.25)")
ax4.set_title("CSI by Feature")
ax4.set_xlabel("CSI Value")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("banking_metrics.png", dpi=100, bbox_inches="tight")
plt.show()
print("\\nVisualisations saved to banking_metrics.png")

# === Summary ===
print("\\n-- Summary --")
print(f"  KS (Dev/Prod):   {ks_dev:.4f} / {ks_prod:.4f}")
print(f"  Gini (Dev/Prod): {gini_dev:.4f} / {gini_prod:.4f}")
print(f"  Score PSI:       {score_psi:.4f}")
print(f"  Drifting features: {[f for f, v in csi_values.items() if v > 0.1]}")
`;

export const financialMl: Chapter = {
  title: "Financial ML",
  slug: "financial-ml",
  pages: [
    {
      title: "Credit Risk Modelling",
      slug: "credit-risk-modelling",
      description:
        "PD/LGD/EAD estimation, credit scorecards with WoE/IV feature selection, and scorecard point assignment",
      markdownContent: creditRiskMarkdown,
      codeSnippet: creditRiskCode,
      codeLanguage: "python",
    },
    {
      title: "Fraud Detection Pipeline",
      slug: "fraud-detection-pipeline",
      description:
        "Rule engine and ML hybrid architecture with feature engineering, XGBoost, and precision-recall threshold tuning",
      markdownContent: fraudDetectionMarkdown,
      codeSnippet: fraudDetectionCode,
      codeLanguage: "python",
    },
    {
      title: "Anti-Money Laundering",
      slug: "anti-money-laundering",
      description:
        "Transaction monitoring with anomaly detection, clustering, and network analysis for suspicious activity",
      markdownContent: amlMarkdown,
      codeSnippet: amlCode,
      codeLanguage: "python",
    },
    {
      title: "Model Risk Management",
      slug: "model-risk-management",
      description:
        "SR 11-7 and SS1/23 regulatory frameworks, model validation lifecycle, and champion-challenger governance",
      markdownContent: modelRiskMarkdown,
      codeLanguage: "python",
    },
    {
      title: "Banking-Specific Metrics",
      slug: "banking-specific-metrics",
      description:
        "KS statistic, Gini coefficient, PSI, and CSI for credit model validation and drift monitoring",
      markdownContent: bankingMetricsMarkdown,
      codeSnippet: bankingMetricsCode,
      codeLanguage: "python",
    },
  ],
};
