# ML Catalogue

A machine learning model catalogue application.

## Live Demo

> **https://loads-optimum-crest-addressing.trycloudflare.com**
>
> This is a temporary Cloudflare Quick Tunnel URL for testing purposes. It may become unavailable at any time (e.g., on redeploy or restart). To find the current URL, run on the Mac Mini:
>
> ```bash
> docker compose logs cloudflared
> ```

## Tech Stack

- **Frontend:** React 19, TypeScript, Vite
- **Backend:** Python 3.12, FastAPI, uv
- **Tooling:** ESLint, Prettier, Ruff, pnpm workspaces

## Prerequisites

- [Node.js](https://nodejs.org/) (LTS)
- [pnpm](https://pnpm.io/)
- [Python 3.12](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/)

## Quick Start

### Install dependencies

```bash
# Frontend dependencies
pnpm install

# Backend dependencies
cd backend && uv sync
```

### Run dev servers

```bash
# Start both frontend and backend concurrently
pnpm dev

# Or start individually
pnpm dev:frontend   # Vite dev server
pnpm dev:backend    # FastAPI dev server
```

### Linting & Formatting

```bash
pnpm lint       # Run ESLint (frontend) + Ruff (backend)
pnpm format     # Run Prettier (frontend) + Ruff format (backend)
```

## Deployment

The app runs as a single Docker container on a Mac Mini (M4, ARM64), with a Cloudflare Quick Tunnel for public access.

### Production deploy

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

This starts the app container and a `cloudflared` container that creates a Quick Tunnel. The tunnel URL is logged in the cloudflared container stdout:

```bash
docker compose logs cloudflared
```

### CI/CD

Pushing to `main` triggers a GitHub Actions workflow that:

1. Builds a Docker image and pushes it to GHCR
2. Deploys to the Mac Mini over Tailscale SSH
3. The cloudflared container automatically provisions a new Quick Tunnel URL

## Project Structure

```
ml-catalogue/
├── frontend/           # React + TypeScript + Vite
│   ├── src/            # Application source code
│   └── public/         # Static assets
├── backend/            # Python + FastAPI
│   └── app/            # Application source code
├── Dockerfile          # Multi-stage build (frontend + backend)
├── docker-compose.yml  # Base Compose config
├── docker-compose.prod.yml  # Production overrides
├── .dockerignore       # Docker build exclusions
├── package.json        # Root monorepo scripts
└── .editorconfig       # Shared editor settings
```
