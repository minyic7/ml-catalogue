# ML Catalogue

A machine learning model catalogue application.

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

## Project Structure

```
ml-catalogue/
├── frontend/       # React + TypeScript + Vite
│   ├── src/        # Application source code
│   └── public/     # Static assets
├── backend/        # Python + FastAPI
│   └── app/        # Application source code
├── package.json    # Root monorepo scripts
└── .editorconfig   # Shared editor settings
```
