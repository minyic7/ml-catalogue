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

Every NLP model — from simple bag-of-words to large transformers — depends on tokenization. Run the code to see how different strategies produce different token counts and vocabularies from the same text.
`;

const textPreprocessingCode = `import re
from collections import Counter

# Sample corpus
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models process textual data efficiently.",
    "Natural language processing enables computers to understand text.",
    "Text preprocessing is the first step in any NLP pipeline.",
    "Tokenization splits sentences into individual words or subwords."
]

# --- Text Cleaning ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)  # remove punctuation
    return text.strip()

stopwords = {'the', 'a', 'an', 'is', 'in', 'to', 'or', 'and', 'of', 'over', 'any'}

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopwords]

# --- Simple Stemmer ---
def simple_stem(word):
    for suffix in ['ing', 'tion', 'ly', 'es', 's']:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[:-len(suffix)]
    return word

# --- Process all documents ---
print("=== Text Preprocessing & Tokenization Demo ===\\n")
all_tokens = []
for i, doc in enumerate(documents):
    cleaned = clean_text(doc)
    word_tokens = cleaned.split()
    filtered = remove_stopwords(word_tokens)
    stemmed = [simple_stem(t) for t in filtered]
    all_tokens.extend(stemmed)
    print(f"Doc {i+1}: {doc}")
    print(f"  Cleaned tokens:  {word_tokens}")
    print(f"  After stopwords: {filtered}")
    print(f"  After stemming:  {stemmed}\\n")

# --- Vocabulary Statistics ---
vocab = set(all_tokens)
freq = Counter(all_tokens)
print(f"Total tokens: {len(all_tokens)}")
print(f"Vocabulary size |V|: {len(vocab)}")
print(f"\\nTop 10 most common tokens:")
for token, count in freq.most_common(10):
    print(f"  '{token}': {count}")
`;

const tfidfMarkdown = `
# Bag-of-Words & TF-IDF

Once text is tokenized, we need a way to represent documents as numerical vectors that models can process. **Bag-of-Words** and **TF-IDF** are foundational techniques for this.

## Bag-of-Words (BoW)

The bag-of-words model represents each document as a vector of word counts, ignoring grammar and word order. For a vocabulary of size $|V|$, each document becomes a vector in $\\mathbb{R}^{|V|}$.

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

TF-IDF is used in search engines, document clustering, and as a baseline for text classification. Run the code to build TF-IDF vectors from a small corpus and discover which documents are most similar.
`;

const tfidfCode = `import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Small corpus of documents
corpus = [
    "Machine learning algorithms learn patterns from data",
    "Deep learning neural networks require large datasets",
    "Natural language processing analyzes text and speech",
    "Text classification assigns labels to documents",
    "Neural networks are a subset of machine learning models",
    "Speech recognition converts spoken language to text"
]

# Build TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

print("=== Bag-of-Words & TF-IDF Demo ===\\n")
print(f"Corpus size: {len(corpus)} documents")
print(f"Vocabulary size: {len(feature_names)}")
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}\\n")

# Show top TF-IDF terms per document
print("--- Top 3 TF-IDF terms per document ---")
for i, doc in enumerate(corpus):
    row = tfidf_matrix[i].toarray().flatten()
    top_indices = row.argsort()[-3:][::-1]
    top_terms = [(feature_names[j], round(row[j], 3)) for j in top_indices]
    print(f"Doc {i+1}: \\"{doc}\\"")
    print(f"  Top terms: {top_terms}\\n")

# Compute pairwise cosine similarity
sim_matrix = cosine_similarity(tfidf_matrix)

print("--- Document Similarity Matrix ---")
header = "      " + "  ".join(f"D{i+1:>4}" for i in range(len(corpus)))
print(header)
for i in range(len(corpus)):
    row_str = "  ".join(f"{sim_matrix[i][j]:5.2f}" for j in range(len(corpus)))
    print(f"D{i+1}  {row_str}")

# Find most similar pair
np.fill_diagonal(sim_matrix, 0)
max_idx = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
print(f"\\nMost similar pair: Doc {max_idx[0]+1} & Doc {max_idx[1]+1} "
      f"(cosine similarity = {sim_matrix[max_idx]:.3f})")
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
        "Tokenization strategies, text cleaning, stemming, and lemmatization",
      markdownContent: textPreprocessingMarkdown,
      codeSnippet: textPreprocessingCode,
      codeLanguage: "python",
    },
    {
      title: "Bag-of-Words & TF-IDF",
      slug: "bag-of-words-tfidf",
      description:
        "Document vectorization with TF-IDF and cosine similarity",
      markdownContent: tfidfMarkdown,
      codeSnippet: tfidfCode,
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
