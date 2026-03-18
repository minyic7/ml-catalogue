import type { Chapter } from "../types";

const textPreprocessingMarkdown = `
# Text Preprocessing & Tokenization

Text preprocessing transforms raw text into a clean, structured format that machine learning models can work with. It's the essential first step in any NLP pipeline.

## Tokenization

Tokenization splits text into smaller units called **tokens**. The choice of tokenization strategy directly impacts model performance.

### Word-Level Tokenization

The simplest approach splits text on whitespace and punctuation:

$$\\text{tokens} = \\text{split}(\\text{text}, \\text{delimiters})$$

This is intuitive but struggles with large vocabularies and out-of-vocabulary (OOV) words.

### Subword Tokenization (BPE)

**Byte Pair Encoding** iteratively merges the most frequent character pairs. Starting from individual characters, BPE builds a vocabulary of common subwords. For a pair $(a, b)$ with frequency $f(a,b)$, the merge priority is:

$$\\text{priority}(a, b) = f(a, b)$$

This balances vocabulary size with the ability to represent any word — even unseen ones — as a sequence of subword tokens.

### Character-Level Tokenization

Every character becomes a token. The vocabulary is tiny (typically under 256 tokens), but sequences become very long and lose word-level semantics.

## Text Cleaning

Before tokenization, we typically apply several cleaning steps:

- **Lowercasing**: Reduces vocabulary by treating "The" and "the" as the same token
- **Stopword removal**: Filters out common words like "the", "is", "at" that carry little meaning
- **Stemming**: Reduces words to their root form (e.g., "running" → "run") using rule-based truncation
- **Lemmatization**: Similar to stemming but uses vocabulary and morphological analysis to return valid words (e.g., "better" → "good")

The number of unique tokens after preprocessing defines the **vocabulary size** $|V|$, which directly affects model complexity.

## Why This Matters

Every NLP model — from simple bag-of-words to large transformers — depends on tokenization.

## Bag-of-Words (BoW)

Once text is tokenized, we need a way to represent documents as numerical vectors that models can process. The **bag-of-words** model represents each document as a vector of word counts, ignoring grammar and word order. For a vocabulary of size $|V|$, each document becomes a vector in $\\mathbb{R}^{|V|}$.

$$\\text{BoW}(d) = [\\text{count}(t_1, d),\\; \\text{count}(t_2, d),\\; \\dots,\\; \\text{count}(t_{|V|}, d)]$$

The simplicity of BoW is both its strength and weakness — it captures word presence but treats "good" and "great" as completely unrelated dimensions.

## TF-IDF: Term Frequency–Inverse Document Frequency

Raw word counts overweight common words. **TF-IDF** solves this by scaling each term's frequency by how rare it is across the corpus:

$$\\text{tfidf}(t, d) = \\text{tf}(t, d) \\cdot \\log\\frac{N}{\\text{df}(t)}$$

where:
- $\\text{tf}(t, d)$ is the frequency of term $t$ in document $d$
- $N$ is the total number of documents
- $\\text{df}(t)$ is the number of documents containing term $t$

Words that appear in every document get a low IDF score, while distinctive words get a high score.

## Document Similarity

With TF-IDF vectors, we can measure how similar two documents are using **cosine similarity**:

$$\\cos(\\theta) = \\frac{A \\cdot B}{\\|A\\|\\;\\|B\\|}$$

A cosine similarity of 1 means identical direction (same topic), while 0 means completely unrelated. This metric is preferred over Euclidean distance because it's robust to document length differences.

## Why This Matters

Run the code to see how tokenization, cleaning, BoW, and TF-IDF work together to transform raw text into numerical representations that reveal document similarity.
`;

const textPreprocessingCode = `import re
import numpy as np
from collections import Counter

# Sample corpus
documents = [
    "Machine learning algorithms learn patterns from data",
    "Deep learning neural networks require large datasets",
    "Natural language processing analyzes text and speech",
    "Text classification assigns labels to documents",
    "Neural networks are a subset of machine learning models",
    "Speech recognition converts spoken language to text"
]

# --- Text Cleaning ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)  # remove punctuation
    return text.strip()

stopwords = {'the', 'a', 'an', 'is', 'in', 'to', 'or', 'and', 'of', 'over',
             'any', 'are', 'from'}

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopwords]

# --- Simple Stemmer ---
def simple_stem(word):
    for suffix in ['ing', 'tion', 'ly', 'es', 's']:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)]
    return word

# --- Preprocessing Pipeline ---
print("=== Text Preprocessing & Tokenisation ===\\n")
processed_docs = []
for i, doc in enumerate(documents):
    cleaned = clean_text(doc)
    tokens = cleaned.split()
    filtered = remove_stopwords(tokens)
    stemmed = [simple_stem(t) for t in filtered]
    processed_docs.append(stemmed)
    print(f"Doc {i+1}: {doc}")
    print(f"  Tokens:  {filtered}")
    print(f"  Stemmed: {stemmed}\\n")

# --- Build vocabulary ---
vocab = sorted(set(t for doc in processed_docs for t in doc))
word_to_idx = {w: i for i, w in enumerate(vocab)}
print(f"Vocabulary size |V|: {len(vocab)}")
print(f"Vocabulary: {vocab}\\n")

# --- Bag-of-Words ---
print("=== Bag-of-Words Representation ===\\n")
bow_matrix = np.zeros((len(processed_docs), len(vocab)))
for i, doc in enumerate(processed_docs):
    for token in doc:
        bow_matrix[i, word_to_idx[token]] += 1

print(f"BoW matrix shape: {bow_matrix.shape}")
for i in range(len(processed_docs)):
    nonzero = [(vocab[j], int(bow_matrix[i, j]))
               for j in range(len(vocab)) if bow_matrix[i, j] > 0]
    print(f"  Doc {i+1}: {nonzero}")

# --- TF-IDF ---
print("\\n=== TF-IDF Representation ===\\n")
tf = bow_matrix / (bow_matrix.sum(axis=1, keepdims=True) + 1e-10)
df = (bow_matrix > 0).sum(axis=0)
N = len(processed_docs)
idf = np.log(N / (df + 1e-10))
tfidf_matrix = tf * idf

# Show top TF-IDF terms per document
for i, doc in enumerate(documents):
    row = tfidf_matrix[i]
    top_indices = row.argsort()[-3:][::-1]
    top_terms = [(vocab[j], round(row[j], 3)) for j in top_indices]
    print(f'Doc {i+1}: "{doc}"')
    print(f"  Top TF-IDF terms: {top_terms}\\n")

# --- Document Similarity ---
print("=== Document Cosine Similarity ===\\n")
norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True) + 1e-10
normed = tfidf_matrix / norms
sim_matrix = normed @ normed.T

header = "      " + "  ".join(f"D{i+1:>4}" for i in range(N))
print(header)
for i in range(N):
    row_str = "  ".join(f"{sim_matrix[i][j]:5.2f}" for j in range(N))
    print(f"D{i+1}  {row_str}")

np.fill_diagonal(sim_matrix, 0)
max_idx = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
print(f"\\nMost similar pair: Doc {max_idx[0]+1} & Doc {max_idx[1]+1} "
      f"(cosine similarity = {sim_matrix[max_idx]:.3f})")
`;

const wordEmbeddingsMarkdown = `
# Word Embeddings (Word2Vec, GloVe)

Sparse representations like Bag-of-Words and TF-IDF treat every word as an independent dimension. This has two major limitations:

- **No semantic meaning**: "happy" and "joyful" are as unrelated as "happy" and "volcano"
- **Curse of dimensionality**: vectors have $|V|$ dimensions (often 10,000+), mostly zeros

**Word embeddings** solve this by mapping each word to a **dense, low-dimensional vector** (typically 50–300 dimensions) where semantically similar words are close together.

## Word2Vec

Word2Vec (Mikolov et al., 2013) learns embeddings by predicting words from their context. It has two architectures:

### Skip-gram

Given a target word, predict the surrounding context words. For a target word $w_t$ and context window size $c$, the model maximises:

$$J = \\frac{1}{T} \\sum_{t=1}^{T} \\sum_{-c \\leq j \\leq c,\\; j \\neq 0} \\log P(w_{t+j} \\mid w_t)$$

where $P(w_O \\mid w_I) = \\frac{\\exp(\\mathbf{v}'_{w_O} \\cdot \\mathbf{v}_{w_I})}{\\sum_{w=1}^{|V|} \\exp(\\mathbf{v}'_w \\cdot \\mathbf{v}_{w_I})}$

Skip-gram works well with small datasets and captures rare words effectively.

### CBOW (Continuous Bag of Words)

The reverse: predict the target word from its context. CBOW averages the context word vectors and predicts the centre word. It's faster to train but less effective for rare words.

### Word Analogies

The famous result: vector arithmetic captures semantic relationships:

$$\\mathbf{v}_{\\text{king}} - \\mathbf{v}_{\\text{man}} + \\mathbf{v}_{\\text{woman}} \\approx \\mathbf{v}_{\\text{queen}}$$

## GloVe: Global Vectors

**GloVe** (Pennington et al., 2014) takes a different approach — it factorises the global **word co-occurrence matrix**. For words $i$ and $j$ with co-occurrence count $X_{ij}$, the objective minimises:

$$J = \\sum_{i,j=1}^{|V|} f(X_{ij}) \\left( \\mathbf{w}_i^T \\tilde{\\mathbf{w}}_j + b_i + \\tilde{b}_j - \\log X_{ij} \\right)^2$$

where $f(x)$ is a weighting function that caps the influence of very frequent co-occurrences. GloVe combines the advantages of global matrix factorisation with local context window methods.

## FastText: Subword Embeddings

**FastText** (Bojanowski et al., 2017) extends Word2Vec by representing each word as a bag of character n-grams. The word "learning" with $n=3$ includes the n-grams: \`<le\`, \`lea\`, \`ear\`, \`arn\`, \`rni\`, \`nin\`, \`ing\`, \`ng>\`.

The word vector is the sum of its n-gram vectors:

$$\\mathbf{v}_{\\text{word}} = \\sum_{g \\in G(\\text{word})} \\mathbf{z}_g$$

This means FastText can generate embeddings for **out-of-vocabulary words** by composing their subword pieces — a major advantage over Word2Vec and GloVe.

## Cosine Similarity

We measure the similarity between word vectors using cosine similarity:

$$\\text{sim}(\\mathbf{a}, \\mathbf{b}) = \\frac{\\mathbf{a} \\cdot \\mathbf{b}}{\\|\\mathbf{a}\\|\\;\\|\\mathbf{b}\\|}$$

Values range from $-1$ (opposite) to $1$ (identical direction), with $0$ indicating no relationship.

## Pre-trained Embeddings

Training embeddings from scratch requires massive corpora. In practice, we use **pre-trained embeddings** (GloVe trained on Wikipedia, Word2Vec on Google News) and fine-tune them for specific tasks — an early form of **transfer learning** for NLP.

## Why This Matters

Word embeddings are the foundation of modern NLP. They power everything from search engines to chatbots. Run the code to build simple word embeddings from co-occurrence data, explore word analogies, and visualise word clusters.
`;

const wordEmbeddingsCode = `import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Build word embeddings from co-occurrence ---
# A small corpus about animals, royalty, and technology
sentences = [
    "the king rules the kingdom with power",
    "the queen rules the kingdom with grace",
    "the man works in the city",
    "the woman works in the city",
    "the prince is the son of the king",
    "the princess is the daughter of the queen",
    "the cat chases the mouse quickly",
    "the dog chases the cat outside",
    "the kitten is a young cat",
    "the puppy is a young dog",
    "the computer processes data quickly",
    "the algorithm learns from data",
]

# Tokenise and build vocabulary
tokens_list = [s.split() for s in sentences]
vocab = sorted(set(w for s in tokens_list for w in s))
word_to_idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f"=== Word Embeddings Demo ===\\n")
print(f"Vocabulary size: {V}")

# Build co-occurrence matrix (window size = 2)
window = 2
cooccurrence = np.zeros((V, V))
for tokens in tokens_list:
    for i, word in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i != j:
                cooccurrence[word_to_idx[word], word_to_idx[tokens[j]]] += 1

# Apply PPMI (Positive Pointwise Mutual Information)
total = cooccurrence.sum()
row_sum = cooccurrence.sum(axis=1, keepdims=True) + 1e-10
col_sum = cooccurrence.sum(axis=0, keepdims=True) + 1e-10
pmi = np.log2((cooccurrence * total) / (row_sum * col_sum) + 1e-10)
ppmi = np.maximum(pmi, 0)

# Reduce dimensionality with SVD to get dense embeddings
embedding_dim = 20
U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
embeddings = U[:, :embedding_dim] * np.sqrt(S[:embedding_dim])
print(f"Embedding dimensions: {embedding_dim}")
print(f"Embedding matrix shape: {embeddings.shape}\\n")

# --- Cosine similarity ---
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def get_vec(word):
    return embeddings[word_to_idx[word]]

def most_similar(vec, top_n=5, exclude=None):
    exclude = exclude or []
    sims = []
    for w in vocab:
        if w in exclude:
            continue
        sims.append((w, cosine_sim(vec, get_vec(w))))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]

# Show word similarities
print("--- Word Similarities ---")
test_pairs = [("king", "queen"), ("cat", "dog"), ("king", "cat"),
              ("man", "woman"), ("computer", "algorithm")]
for w1, w2 in test_pairs:
    sim = cosine_sim(get_vec(w1), get_vec(w2))
    print(f"  sim({w1}, {w2}) = {sim:.3f}")

# --- Word Analogies ---
print("\\n--- Word Analogies (a - b + c = ?) ---")
analogies = [
    ("king", "man", "woman"),     # expect: queen
    ("cat", "kitten", "puppy"),   # expect: dog
]
for a, b, c in analogies:
    vec = get_vec(a) - get_vec(b) + get_vec(c)
    results = most_similar(vec, top_n=3, exclude=[a, b, c])
    result_str = ", ".join(f"{w} ({s:.3f})" for w, s in results)
    print(f"  {a} - {b} + {c} = {result_str}")

# --- Most similar words ---
print("\\n--- Nearest Neighbours ---")
for word in ["king", "cat", "computer"]:
    neighbours = most_similar(get_vec(word), top_n=4, exclude=[word])
    nb_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbours)
    print(f"  {word}: {nb_str}")

# --- Visualise with PCA ---
from numpy.linalg import svd as pca_svd

# Select words to visualise
vis_words = ["king", "queen", "man", "woman", "prince", "princess",
             "cat", "dog", "kitten", "puppy",
             "computer", "algorithm", "data"]
vis_idx = [word_to_idx[w] for w in vis_words]
vis_vecs = embeddings[vis_idx]

# PCA to 2D
vis_centered = vis_vecs - vis_vecs.mean(axis=0)
_, _, Vt_pca = pca_svd(vis_centered, full_matrices=False)
coords = vis_centered @ Vt_pca[:2].T

# Plot
fig, ax = plt.subplots(figsize=(10, 7))
colors = (['#e74c3c'] * 6 + ['#2ecc71'] * 4 + ['#3498db'] * 3)
ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=120, zorder=5)
for i, word in enumerate(vis_words):
    ax.annotate(word, (coords[i, 0], coords[i, 1]),
                fontsize=12, fontweight='bold',
                xytext=(8, 8), textcoords='offset points')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Royalty/People'),
                   Patch(facecolor='#2ecc71', label='Animals'),
                   Patch(facecolor='#3498db', label='Technology')]
ax.legend(handles=legend_elements, loc='best', fontsize=11)
ax.set_title('Word Embeddings — PCA Visualisation', fontsize=14)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output.png', dpi=100)
print("\\nWord embedding cluster plot saved to output.png")
`;

const sentimentMarkdown = `
# Sentiment Analysis Demo

Sentiment analysis classifies text as expressing positive or negative opinions. It's one of the most common real-world NLP tasks, used in product reviews, social media monitoring, and customer feedback analysis.

## The Text Classification Pipeline

A complete sentiment analysis system follows these steps:

1. **Data collection** — gather labeled text (e.g., movie reviews with ratings)
2. **Feature extraction** — convert text to numerical features using TF-IDF or similar
3. **Model training** — fit a classifier on the feature vectors
4. **Prediction** — classify new, unseen text

## Feature Extraction with TF-IDF

We represent each review as a TF-IDF vector. For a review $d$ with $n$ terms, the feature vector is:

$$\\mathbf{x}_d = [\\text{tfidf}(t_1, d),\\; \\text{tfidf}(t_2, d),\\; \\dots,\\; \\text{tfidf}(t_{|V|}, d)]$$

## Naive Bayes Classifier

**Multinomial Naive Bayes** is a natural choice for text classification. It applies Bayes' theorem with the "naive" assumption that features are conditionally independent given the class:

$$P(y \\mid \\mathbf{x}) = \\frac{P(\\mathbf{x} \\mid y)\\, P(y)}{P(\\mathbf{x})}$$

The predicted class is the one with the highest posterior probability:

$$\\hat{y} = \\arg\\max_y \\; P(y) \\prod_{i=1}^{|V|} P(x_i \\mid y)$$

Despite its simplicity, Naive Bayes performs surprisingly well on text data because the independence assumption, while technically violated, still produces good decision boundaries.

## Evaluating the Classifier

We measure performance with **accuracy** and a **confusion matrix**, which shows how predictions align with true labels across all classes.

## Why This Matters

Sentiment analysis is a gateway to more complex NLP tasks like aspect-based analysis, emotion detection, and stance classification. Run the code to train a classifier on movie review snippets and see how well it distinguishes positive from negative sentiment.
`;

const sentimentCode = `import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Sample movie review dataset
reviews = [
    "This movie was absolutely fantastic and thrilling",
    "Terrible film with awful acting and bad plot",
    "I loved every minute of this wonderful movie",
    "Worst movie I have ever seen completely boring",
    "An excellent masterpiece with brilliant performances",
    "Dull and predictable storyline wasted my time",
    "Amazing visuals and a heartwarming story",
    "Horrible dialogue and painfully slow pacing",
    "A delightful and entertaining experience overall",
    "Disappointing and overrated with no real substance",
    "Beautifully crafted with superb acting and direction",
    "A complete disaster from start to finish",
    "Gripping and emotionally powerful storytelling",
    "Uninspired and forgettable waste of talent",
    "Outstanding performances and a captivating plot",
    "Laughably bad script with zero originality",
    "A truly moving and uplifting cinematic gem",
    "Tedious and unimaginative with flat characters",
    "Riveting from beginning to end highly recommended",
    "Stale and cliched with nothing new to offer",
]
# Labels: 1 = positive, 0 = negative (alternating)
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                   1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.3, random_state=42)

# TF-IDF feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=== Sentiment Analysis Demo ===\\n")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
print(f"\\nAccuracy: {acc:.1%}\\n")
print("Confusion Matrix:")
print(f"  Predicted:  Neg  Pos")
print(f"  Actual Neg: {cm[0][0]:>3}  {cm[0][1]:>3}")
print(f"  Actual Pos: {cm[1][0]:>3}  {cm[1][1]:>3}\\n")

# Show individual predictions
print("--- Test Set Predictions ---")
label_map = {1: "Positive", 0: "Negative"}
for review, true, pred in zip(X_test, y_test, y_pred):
    status = "ok" if true == pred else "MISS"
    print(f"  [{status:>4}] {label_map[pred]:>8} | {review}")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_yticklabels(['Negative', 'Positive'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Sentiment Analysis - Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                color='white' if cm[i][j] > cm.max()/2 else 'black', fontsize=18)
plt.colorbar(im); plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
print("\\nConfusion matrix plot saved to confusion_matrix.png")
`;

export const nlp: Chapter = {
  title: "Natural Language Processing",
  slug: "nlp",
  pages: [
    {
      title: "Text Preprocessing & Tokenization",
      slug: "text-preprocessing",
      description:
        "Tokenization, text cleaning, stemming, Bag-of-Words, and TF-IDF representation",
      markdownContent: textPreprocessingMarkdown,
      codeSnippet: textPreprocessingCode,
      codeLanguage: "python",
    },
    {
      title: "Word Embeddings (Word2Vec, GloVe)",
      slug: "word-embeddings",
      description:
        "Dense word vectors, Word2Vec, GloVe, FastText, and cosine similarity",
      markdownContent: wordEmbeddingsMarkdown,
      codeSnippet: wordEmbeddingsCode,
      codeLanguage: "python",
    },
    {
      title: "Sentiment Analysis Demo",
      slug: "sentiment-analysis",
      description:
        "End-to-end text classification with Naive Bayes and TF-IDF features",
      markdownContent: sentimentMarkdown,
      codeSnippet: sentimentCode,
      codeLanguage: "python",
    },
  ],
};
