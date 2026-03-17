import type { Chapter } from "../types";

export const largeModels: Chapter = {
  title: "Large Models",
  slug: "large-models",
  pages: [
    {
      title: "Fine-Tuning",
      slug: "fine-tuning",
      description:
        "Transfer learning, full fine-tuning vs parameter-efficient methods, and LoRA",
      markdownContent: `# Fine-Tuning

Large pre-trained models learn general-purpose representations from massive datasets. **Fine-tuning** adapts these representations to a specific downstream task — classification, summarization, code generation — by continuing training on a smaller, task-specific dataset. This is the core idea behind **transfer learning**: reuse expensive pre-training and specialize cheaply.

## Why Fine-Tune?

Training a large model from scratch requires enormous compute and data. A pre-trained model already encodes rich linguistic and semantic structure. Fine-tuning leverages that foundation, typically needing only a fraction of the data and compute to reach strong task performance.

## Full Fine-Tuning vs Parameter-Efficient Methods

**Full fine-tuning** updates every parameter in the model. For a model with $N$ parameters, this means storing $N$ gradients and optimizer states — memory and compute scale linearly with model size.

**Parameter-efficient fine-tuning (PEFT)** methods freeze most of the model and only train a small number of additional or modified parameters. Popular approaches include:

- **Adapters** — small trainable bottleneck layers inserted between frozen layers
- **LoRA (Low-Rank Adaptation)** — decomposes weight updates into low-rank matrices

## LoRA: Low-Rank Adaptation

LoRA keeps the original weight matrix $W \\in \\mathbb{R}^{d \\times k}$ frozen and learns a low-rank update:

$$
W' = W + BA
$$

where $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times k}$, with rank $r \\ll \\min(d, k)$.

The number of trainable parameters drops from $d \\times k$ to $r \\times (d + k)$. For a $4096 \\times 4096$ weight matrix with $r = 8$, that is a reduction from **16.8M** to just **65.5K** parameters — a **256× saving**.

## When to Fine-Tune vs When to Prompt

| Scenario | Approach |
|---|---|
| Task is well-served by general knowledge | Prompting |
| Need domain-specific behavior or terminology | Fine-tuning |
| Very little labeled data available | Few-shot prompting |
| Consistent, repeatable output format required | Fine-tuning |
| Rapid iteration and experimentation | Prompting |

Run the code below to see LoRA's low-rank decomposition in action and compare parameter counts.`,
      codeSnippet: `import numpy as np

# Simulate a pre-trained weight matrix
d, k = 4096, 4096
W = np.random.randn(d, k) * 0.02  # frozen pre-trained weights

# LoRA: low-rank decomposition with rank r
r = 8
B = np.random.randn(d, r) * 0.01  # trainable
A = np.random.randn(r, k) * 0.01  # trainable

# The adapted weight: W' = W + B @ A
delta_W = B @ A
W_prime = W + delta_W

# Parameter comparison
full_params = d * k
lora_params = r * (d + k)
ratio = full_params / lora_params

print("=== LoRA Parameter Efficiency ===")
print(f"Weight matrix shape: ({d}, {k})")
print(f"LoRA rank: {r}")
print(f"Full fine-tuning params:  {full_params:>12,}")
print(f"LoRA trainable params:   {lora_params:>12,}")
print(f"Reduction factor:         {ratio:>11.0f}x")
print()

# Verify the update is truly low-rank
rank_of_delta = np.linalg.matrix_rank(delta_W)
print(f"Rank of delta_W (B @ A): {rank_of_delta}")
print(f"Frobenius norm of W:        {np.linalg.norm(W):.2f}")
print(f"Frobenius norm of delta_W:  {np.linalg.norm(delta_W):.2f}")
print()

# Show scaling across different ranks
print("=== Savings by Rank ===")
print(f"{'Rank':>6} {'LoRA Params':>14} {'Reduction':>10}")
for rank in [1, 4, 8, 16, 64]:
    lp = rank * (d + k)
    print(f"{rank:>6} {lp:>14,} {full_params / lp:>9.0f}x")`,
      codeLanguage: "python",
    },
    {
      title: "Prompt Engineering",
      slug: "prompt-engineering",
      description:
        "Prompt design patterns, templates, few-shot prompting, and evaluation",
      markdownContent: `# Prompt Engineering

Prompt engineering is the practice of designing inputs to large language models to elicit desired outputs. Unlike fine-tuning, it requires no model modification — you shape behavior entirely through the input text.

## Prompt Design Patterns

### Zero-Shot Prompting

Provide only the task instruction with no examples. The model relies entirely on its pre-trained knowledge:

> *Classify the sentiment of this review as positive or negative: "The battery lasts forever."*

### Few-Shot Prompting

Include $n$ labeled examples before the query. The model learns the input-output mapping in context:

$$
P(y \\mid x, \\{(x_1, y_1), \\dots, (x_n, y_n)\\})
$$

Few-shot prompting is especially effective when $n$ is small (2–5 examples) and examples are representative of the task distribution.

### Chain-of-Thought (CoT)

Append reasoning steps before the final answer. This improves performance on tasks requiring multi-step logic:

> *Q: If a store has 3 boxes with 12 apples each and sells 15, how many remain?*
> *A: 3 boxes × 12 apples = 36 total. 36 − 15 = 21 apples remain.*

## Prompt Templates and Variable Injection

A **prompt template** separates structure from content. Variables like \`{text}\` or \`{examples}\` are injected at runtime:

\`\`\`
Given these examples:
{examples}

Now classify: {input}
\`\`\`

This makes prompts reusable and testable across different inputs.

## Evaluating Prompt Quality

Prompt quality can be measured by comparing model outputs against expected answers. Common metrics include:

- **Exact match** — does the output match the expected answer exactly?
- **Contains match** — does the output contain the expected answer?
- **Scoring function** — a weighted combination: $\\text{score} = w_1 \\cdot \\text{exact} + w_2 \\cdot \\text{contains}$

Systematic evaluation across a test suite reveals which prompt patterns perform best for a given task.

Run the code below to build prompt templates, simulate few-shot prompting, and evaluate outputs.`,
      codeSnippet: `# Prompt engineering simulation — no LLM needed

# --- Prompt Templates ---
ZERO_SHOT = "Classify the sentiment as positive or negative: \\"{text}\\""

FEW_SHOT = """Examples:
{examples}
Now classify: "{text}"
Answer:"""

COT = """Classify sentiment. Think step by step.
Text: "{text}"
Step 1: Identify key words.
Step 2: Determine overall tone.
Answer:"""

# --- Few-shot example bank ---
examples = [
    ("The camera quality is amazing", "positive"),
    ("Broke after two days", "negative"),
    ("Best purchase I ever made", "positive"),
    ("Terrible customer service", "negative"),
]

# --- Build a few-shot prompt ---
def build_few_shot(text, shots):
    ex_str = "\\n".join(f'  "{t}" -> {l}' for t, l in shots)
    return FEW_SHOT.format(examples=ex_str, text=text)

test_input = "Battery life is incredible"
print("=== Few-Shot Prompt ===")
print(build_few_shot(test_input, examples[:3]))
print()

# --- Simulated model outputs (stand-in for real LLM) ---
test_suite = [
    {"text": "Absolutely love it",    "expected": "positive", "output": "positive"},
    {"text": "Total waste of money",   "expected": "negative", "output": "negative"},
    {"text": "It works fine I guess",  "expected": "positive", "output": "negative"},
    {"text": "Screen is gorgeous",     "expected": "positive", "output": "Positive"},
    {"text": "Worst product ever",     "expected": "negative", "output": "negative"},
]

# --- Evaluation ---
def evaluate(suite):
    exact, contains, n = 0, 0, len(suite)
    for case in suite:
        exp, out = case["expected"].lower(), case["output"].lower()
        if out == exp:
            exact += 1
        if exp in out:
            contains += 1
    return {"exact_match": exact / n, "contains_match": contains / n}

results = evaluate(test_suite)
print("=== Prompt Evaluation ===")
for metric, val in results.items():
    print(f"  {metric}: {val:.0%}")
print(f"  tested: {len(test_suite)} cases")`,
      codeLanguage: "python",
    },
  ],
};

