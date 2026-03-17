# ml-catalog — Vision

## Goal

A documentation-style ML knowledge catalog covering the full spectrum from foundational math to professional MLOps. Each concept page pairs clear written explanation with a runnable code snippet and live output. Users learn by reading and running — no setup required. Deployed on a personal Mac Mini server, updated on every push to main.

## Scope

### In Scope
- Four knowledge levels: Foundational → Core ML → Advanced → Professional
- Documentation UI — left sidebar navigation with chapters and subpages, tabbed sections within pages
- Each knowledge page: concept explanation (Markdown + KaTeX for math) + key code snippet + [▶ Run] button + output area
- Two dataset modes per runnable snippet: [⚡ Quick] (small sample) and [🔬 Full] (complete dataset)
- Apple Silicon (MPS) vs CPU toggle for relevant deep learning demos
- Global search across all content (client-side, FlexSearch)
- Backend code execution: FastAPI receives snippet + mode, runs in isolated Python environment, returns stdout + charts (base64)
- Light / dark mode
- GHA CI/CD: push to main → build → deploy to Mac Mini via SSH
- Docker containerized deployment: multi-stage Dockerfile (frontend build + FastAPI runtime), GHCR image registry
- docker-compose with production overlay (docker-compose.prod.yml)
- Tailscale from GitHub Actions to Mac Mini for SSH deploy
- Cloudflare Quick Tunnel (cloudflared) for temporary public URL — no domain required
- FastAPI serves frontend static files (StaticFiles mount) — single container for frontend + backend
- LLM Q&A assistant — interactive learning aid powered by Anthropic Claude API (with Vision):
  - Floating draggable toolbox (default bottom-right, collapsible) with Ask, Screenshot, and Highlight tools
  - Ask mode: open dialog directly to ask a free-form question — no highlight or screenshot required; current page content is sent as context automatically
  - Highlight mode: select text on any page → click Ask → popover dialog with the selected text as context
  - Screenshot mode: click capture → drag to select a screen region (html2canvas) → popover dialog with the image as context
  - Dialog supports multi-turn conversation and paste-image input
  - Current page content is included as context alongside the user's question
  - Session identity: frontend generates a UUID stored in localStorage, sent with every API call to isolate conversations per browser
  - Backend conversation store: in-memory dict keyed by session ID, 7-day TTL with auto-eviction on inactivity
  - Context management: memory usage indicator showing how much context is consumed, manual Compact button to summarise and compress history, auto-compact when context approaches the limit
  - Backend: FastAPI proxy endpoint to Anthropic Claude API (text + vision)
- Extended content chapters:
  - Gradient Boosting chapter (Core ML level): Gradient Boosting Fundamentals, XGBoost, LightGBM, CatBoost — with fraud/transaction modelling examples
  - Reinforcement Learning chapter (Advanced level): MDP & Bellman Equations, Q-Learning, Policy Gradient, Deep RL (DQN)
  - Graph Neural Networks chapter (Advanced level): Graph Basics & Representations, GCN, GraphSAGE, Graph Attention Networks
  - Generative Models chapter (Advanced level): Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), Diffusion Models
  - Financial ML chapter (Professional level) — Banking & Financial Institution ML applications:
    - Credit Risk Modelling: PD/LGD/EAD estimation, credit scorecards (logistic regression + WoE/IV), application vs behavioural scoring
    - Fraud Detection Pipeline: rule engine → ML hybrid, real-time vs batch scoring, feature engineering for transactions
    - Anti-Money Laundering (AML): transaction monitoring, suspicious activity detection, network/graph-based AML
    - Model Risk Management: regulatory context (SR 11-7 / SS1/23), model validation, champion-challenger framework, model documentation
    - Banking-specific metrics: KS statistic, Gini coefficient, PSI (Population Stability Index), CSI (Characteristic Stability Index)
  - Model Evaluation additions (Core ML level): PR-AUC (Precision-Recall AUC) for imbalanced datasets, calibration curves, lift/gain charts — especially relevant for credit/fraud use cases
- Learning progress tracker:
  - Per-page read status toggle (mark as read / unread), persisted in localStorage
  - Sidebar progress indicators showing completion per chapter (e.g., "3/5 pages read")
  - Overall progress bar on the home page showing total completion percentage

### Out of Scope
- User accounts or authentication
- Editable code (snippets are read-only)
- Jupyter notebook interface
- GPU execution (Mac Mini M4 uses Neural Engine / MPS, not CUDA)
- Mobile-specific optimisation (desktop-first)
- Comments or community features
- Chinese translation (deferred)

## Milestones

- [ ] Project scaffold: React 19 + TypeScript + Vite + Tailwind v4 + shadcn/ui + FastAPI backend
- [ ] Documentation layout: sidebar navigation, page routing, chapter/subpage structure
- [ ] Content engine: Markdown + KaTeX rendering, code block component, Run button, output area
- [ ] Backend execution: FastAPI endpoint, Python sandbox, Quick/Full mode, chart return
- [ ] Foundational content: math basics, Python/NumPy/Pandas essentials
- [ ] Core ML content: supervised learning, unsupervised learning, model evaluation
- [ ] Global search (FlexSearch, indexes all page content)
- [ ] Advanced content: deep learning, NLP, CV
- [ ] Professional content: MLOps, large models, production deployment
- [ ] GHA CI/CD pipeline: test → build → SSH deploy to Mac Mini
- [ ] Light/dark mode, polish, error handling

---

<!-- KANBAN_CC_RULES:
  - Goal and Scope are the source of truth for all ticket planning.
  - In writable mode, CC may APPEND to Goal and Scope sections only.
  - CC must NEVER modify or remove existing content in any section.
  - CC must NEVER touch Milestones — only the user manages milestones.
  - In readonly mode, CC proposes amendments in the Step 10 report.
-->
