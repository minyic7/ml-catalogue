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
