/**
 * Content structure configuration — single source of truth for the
 * documentation knowledge hierarchy consumed by sidebar navigation
 * and page routing.
 */

export interface Page {
  title: string;
  slug: string;
  description?: string;
  metadata?: Record<string, unknown>;
}

export interface Chapter {
  title: string;
  slug: string;
  pages: Page[];
  metadata?: Record<string, unknown>;
}

export interface Level {
  title: string;
  slug: string;
  icon?: string;
  chapters: Chapter[];
  metadata?: Record<string, unknown>;
}

export const CONTENT_STRUCTURE: Level[] = [
  {
    title: "Foundational",
    slug: "foundational",
    icon: "book",
    chapters: [
      {
        title: "Linear Algebra",
        slug: "linear-algebra",
        pages: [
          { title: "Vectors", slug: "vectors", description: "Vector operations and spaces" },
          { title: "Matrices", slug: "matrices", description: "Matrix operations and transformations" },
          { title: "Eigenvalues & Eigenvectors", slug: "eigenvalues-eigenvectors" },
        ],
      },
      {
        title: "Probability & Statistics",
        slug: "probability-statistics",
        pages: [
          { title: "Probability Distributions", slug: "probability-distributions" },
          { title: "Bayesian Thinking", slug: "bayesian-thinking" },
        ],
      },
      {
        title: "Python Essentials",
        slug: "python-essentials",
        pages: [
          { title: "NumPy Fundamentals", slug: "numpy-fundamentals" },
          { title: "Pandas Basics", slug: "pandas-basics" },
          { title: "Data Visualization", slug: "data-visualization" },
        ],
      },
    ],
  },
  {
    title: "Core ML",
    slug: "core-ml",
    icon: "cpu",
    chapters: [
      {
        title: "Supervised Learning",
        slug: "supervised-learning",
        pages: [
          { title: "Linear Regression", slug: "linear-regression" },
          { title: "Classification", slug: "classification" },
          { title: "Decision Trees", slug: "decision-trees" },
        ],
      },
      {
        title: "Unsupervised Learning",
        slug: "unsupervised-learning",
        pages: [
          { title: "Clustering", slug: "clustering" },
          { title: "Dimensionality Reduction", slug: "dimensionality-reduction" },
        ],
      },
      {
        title: "Model Evaluation",
        slug: "model-evaluation",
        pages: [
          { title: "Metrics & Scoring", slug: "metrics-scoring" },
          { title: "Cross-Validation", slug: "cross-validation" },
        ],
      },
    ],
  },
  {
    title: "Advanced",
    slug: "advanced",
    icon: "brain",
    chapters: [
      {
        title: "Deep Learning",
        slug: "deep-learning",
        pages: [
          { title: "Neural Network Basics", slug: "neural-network-basics" },
          { title: "Convolutional Networks", slug: "convolutional-networks" },
          { title: "Recurrent Networks", slug: "recurrent-networks" },
        ],
      },
      {
        title: "Natural Language Processing",
        slug: "nlp",
        pages: [
          { title: "Text Preprocessing", slug: "text-preprocessing" },
          { title: "Transformers", slug: "transformers" },
        ],
      },
      {
        title: "Computer Vision",
        slug: "computer-vision",
        pages: [
          { title: "Image Classification", slug: "image-classification" },
          { title: "Object Detection", slug: "object-detection" },
        ],
      },
    ],
  },
  {
    title: "Professional",
    slug: "professional",
    icon: "briefcase",
    chapters: [
      {
        title: "MLOps",
        slug: "mlops",
        pages: [
          { title: "Experiment Tracking", slug: "experiment-tracking" },
          { title: "Model Registry", slug: "model-registry" },
          { title: "CI/CD for ML", slug: "ci-cd-for-ml" },
        ],
      },
      {
        title: "Large Models",
        slug: "large-models",
        pages: [
          { title: "Fine-Tuning", slug: "fine-tuning" },
          { title: "Prompt Engineering", slug: "prompt-engineering" },
        ],
      },
      {
        title: "Production Deployment",
        slug: "production-deployment",
        pages: [
          { title: "Model Serving", slug: "model-serving" },
          { title: "Monitoring & Observability", slug: "monitoring-observability" },
        ],
      },
    ],
  },
] as const satisfies Level[];
