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

# Simulate a pre-trained weight matrix (scaled down for sandbox)
d, k = 1024, 1024
W = np.random.randn(d, k).astype(np.float32) * 0.02  # frozen pre-trained weights

# LoRA: low-rank decomposition with rank r
r = 8
B = np.random.randn(d, r).astype(np.float32) * 0.01  # trainable
A = np.random.randn(r, k).astype(np.float32) * 0.01  # trainable

# The adapted weight: W' = W + B @ A
delta_W = B @ A
W_prime = W + delta_W

# Parameter comparison (show real-world 4096x4096 numbers)
d_real, k_real = 4096, 4096
full_params = d_real * k_real
lora_params = r * (d_real + k_real)
ratio = full_params / lora_params

print("=== LoRA Parameter Efficiency ===")
print(f"Real-world weight matrix: ({d_real}, {k_real})")
print(f"Demo weight matrix:       ({d}, {k})")
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

# Show scaling across different ranks (real-world 4096x4096)
print("=== Savings by Rank (4096x4096 layer) ===")
print(f"{'Rank':>6} {'LoRA Params':>14} {'Reduction':>10}")
for rank in [1, 4, 8, 16, 64]:
    lp = rank * (d_real + k_real)
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
    {
      title: "RAG (Retrieval-Augmented Generation)",
      slug: "rag",
      description:
        "Retrieval-augmented generation: architecture, chunking, embeddings, vector search, and evaluation",
      markdownContent: `# RAG (Retrieval-Augmented Generation)

## What is RAG?

**Retrieval-Augmented Generation (RAG)** combines a **retrieval system** with a **generative model** to ground responses in external knowledge. Instead of relying solely on what a language model memorized during training, RAG fetches relevant documents at inference time and injects them into the prompt as context.

## Why RAG?

Large language models have fundamental limitations that RAG addresses:

- **Hallucination** — LLMs can generate plausible-sounding but factually incorrect answers
- **Knowledge cutoff** — pre-trained models cannot access information created after their training data was collected
- **No access to private data** — enterprise documents, internal wikis, and proprietary databases are invisible to public models

RAG mitigates all three by grounding generation in retrieved, up-to-date, authoritative documents.

## Architecture

The RAG pipeline follows a straightforward flow:

$$
\\text{Query} \\xrightarrow{\\text{encode}} \\text{Retriever} \\xrightarrow{\\text{top-}k \\text{ docs}} \\text{Augmented Prompt} \\xrightarrow{} \\text{Generator (LLM)} \\xrightarrow{} \\text{Response}
$$

1. **Query** — the user's question or input
2. **Retriever** — searches a knowledge base for the $k$ most relevant documents
3. **Augmented Prompt** — combines the original query with retrieved context
4. **Generator** — an LLM produces an answer conditioned on both query and context

## The Retriever: Embeddings and Vector Search

The retriever converts both documents and queries into dense vector representations (embeddings) and finds the closest matches.

**Cosine similarity** is the standard metric for comparing embedding vectors:

$$
\\text{sim}(\\mathbf{q}, \\mathbf{d}) = \\frac{\\mathbf{q} \\cdot \\mathbf{d}}{\\|\\mathbf{q}\\| \\, \\|\\mathbf{d}\\|}
$$

where $\\mathbf{q}$ is the query embedding and $\\mathbf{d}$ is a document embedding.

For large-scale retrieval, approximate nearest neighbor (ANN) libraries like **FAISS** enable sub-linear search over millions of vectors.

## Chunking Strategies

Before embedding, documents must be split into chunks. The choice of strategy affects retrieval quality:

| Strategy | Description | Tradeoff |
|---|---|---|
| **Fixed-size** | Split every $n$ tokens/characters | Simple but may cut mid-sentence |
| **Sentence-based** | Split on sentence boundaries | Preserves meaning, variable sizes |
| **Semantic** | Group sentences by topic similarity | Best quality, most compute |

Chunk size is a critical hyperparameter: too small loses context, too large dilutes relevance.

## Embedding Models

Common choices for generating text embeddings:

- **Sentence Transformers** — open-source models (e.g., all-MiniLM-L6-v2) producing 384-dimensional embeddings
- **OpenAI Embeddings** — API-based models (e.g., text-embedding-3-small) with 1536 dimensions
- **TF-IDF** — a classical sparse embedding based on term frequency and inverse document frequency, useful as a baseline

TF-IDF computes the weight of term $t$ in document $d$ from corpus $D$ as:

$$
\\text{tfidf}(t, d, D) = \\text{tf}(t, d) \\times \\log\\frac{|D|}{\\text{df}(t, D)}
$$

## Vector Databases

Vector databases are purpose-built for storing and querying embeddings at scale:

- **Pinecone** — fully managed, serverless vector search
- **Weaviate** — open-source with hybrid (vector + keyword) search
- **Chroma** — lightweight, developer-friendly, embeddable
- **pgvector** — PostgreSQL extension for vector similarity search

All support CRUD operations on vectors plus filtered nearest-neighbor queries.

## Evaluation

RAG systems are evaluated across three dimensions:

| Metric | What it measures |
|---|---|
| **Context relevance** | Are the retrieved documents actually relevant to the query? |
| **Faithfulness** | Does the generated answer stay faithful to the retrieved context? |
| **Answer correctness** | Is the final answer factually correct? |

A system can retrieve the right documents but still hallucinate, or generate a faithful answer from irrelevant context — all three metrics matter.

## RAG vs Fine-Tuning

| Dimension | RAG | Fine-Tuning |
|---|---|---|
| Knowledge source | External, dynamic | Baked into weights |
| Update frequency | Real-time (swap docs) | Requires retraining |
| Cost | Retrieval infra + LLM calls | GPU compute for training |
| Best for | Factual Q&A, search, support | Style, format, domain adaptation |
| Hallucination control | Grounds in evidence | Learns patterns, can still hallucinate |

In practice, RAG and fine-tuning are complementary — fine-tune for style and format, use RAG for factual grounding.

## Challenges

- **Chunk size tradeoffs** — small chunks improve precision but lose context; large chunks retain context but reduce retrieval specificity
- **Retrieval quality** — if the retriever returns irrelevant documents, the generator inherits that noise ("garbage in, garbage out")
- **Latency** — the retrieval step adds latency on top of LLM inference; vector search must be fast enough for interactive use

Run the code below to build a minimal RAG pipeline from scratch using TF-IDF embeddings and cosine similarity — no external vector DB or LLM needed.`,
      codeSnippet: `import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Knowledge base: small set of text paragraphs ---
knowledge_base = [
    "The mitochondria are the powerhouses of the cell. They generate "
    "most of the cell's supply of adenosine triphosphate (ATP), used as "
    "a source of chemical energy.",

    "Photosynthesis is the process by which green plants convert sunlight "
    "into chemical energy. It takes place primarily in the chloroplasts "
    "using chlorophyll pigments.",

    "DNA replication is the biological process of producing two identical "
    "copies of DNA from one original DNA molecule. It occurs during the "
    "S phase of the cell cycle.",

    "The Theory of General Relativity, published by Einstein in 1915, "
    "describes gravity as the curvature of spacetime caused by mass and "
    "energy.",

    "Neural networks are computing systems inspired by biological neural "
    "networks. They consist of layers of interconnected nodes that learn "
    "patterns from data through backpropagation.",

    "The water cycle describes the continuous movement of water within "
    "the Earth and atmosphere through evaporation, condensation, and "
    "precipitation.",
]

print(f"Knowledge base: {len(knowledge_base)} documents")
print()

# --- 2. Generate TF-IDF embeddings for each chunk ---
vectorizer = TfidfVectorizer(stop_words="english")
doc_embeddings = vectorizer.fit_transform(knowledge_base)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Embedding shape: {doc_embeddings.shape}")
print()

# --- 3. Retrieval: find most similar chunks for a query ---
def retrieve(query, top_k=2):
    """Retrieve top-k most relevant documents for a query."""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_embeddings).flatten()
    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    return [(i, similarities[i], knowledge_base[i]) for i in ranked_indices]

query = "How do cells produce energy?"
print(f"Query: {query!r}")
print()

results = retrieve(query, top_k=3)
print("=== Retrieved Documents (ranked by similarity) ===")
for rank, (idx, score, text) in enumerate(results, 1):
    print(f"  [{rank}] (score={score:.3f}) Doc {idx}: {text[:80]}...")
print()

# --- 4. Augment the prompt with retrieved context ---
def build_augmented_prompt(query, retrieved_docs):
    """Combine retrieved context with the query into an augmented prompt."""
    context_block = "\\n\\n".join(
        f"[Document {i+1}]: {doc}" for i, (_, _, doc) in enumerate(retrieved_docs)
    )
    return (
        f"Use the following context to answer the question.\\n\\n"
        f"Context:\\n{context_block}\\n\\n"
        f"Question: {query}\\n"
        f"Answer:"
    )

augmented_prompt = build_augmented_prompt(query, results)
print("=== Augmented Prompt (sent to LLM) ===")
print(augmented_prompt)
print()

# --- 5. Compare retrieval for different queries ---
queries = [
    "What is photosynthesis?",
    "Explain how neural networks learn",
    "Tell me about Einstein's theory",
    "How does rain form?",
]

print("=== Multi-Query Retrieval ===")
for q in queries:
    top = retrieve(q, top_k=1)[0]
    print(f"  Q: {q}")
    print(f"     -> Doc {top[0]} (score={top[1]:.3f}): {top[2][:60]}...")
    print()`,
      codeLanguage: "python",
    },
  ],
};

