# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Docu-AI-Search is an AI-powered local document search engine. The backend is Python/FastAPI with FAISS vector search, BM25 hybrid search, and LangChain-based LLM integration. The frontend is React 19 + Vite with Tailwind CSS and a glassmorphism design system.

## Development Commands

### Setup
```bash
pip install -r requirements.txt
npm run install-all   # installs root + frontend npm deps
```

### Running the App
```bash
npm run start                   # recommended: starts backend (8000) + frontend concurrently
```
Or manually in two terminals:
```bash
python -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
cd frontend && npm run dev      # Vite dev server with HMR and /api proxy to backend
```

### Testing
```bash
npm run test                    # quick backend unit tests (~30s), uses scripts/run_tests.py --quick
npm run test:full               # all backend tests including slow model tests (5-10 min)
npm run test:frontend           # Vitest frontend tests
npm run test:all                # backend + frontend
npm run test:stress             # performance stress tests
python scripts/run_tests.py --quick           # run a specific backend test suite directly
python scripts/run_tests.py --coverage        # requires pip install pytest-cov
cd frontend && npx vitest run src/test/SearchBar.test.jsx   # run a single frontend test file
```

### Other
```bash
npm run validate        # check project structure compliance
npm run benchmark       # run model performance benchmarks
```

## Architecture

### Backend (`backend/`)
- **`api.py`** — FastAPI entry point (~2000 lines). Defines all REST and WebSocket routes, CORS middleware, SlowAPI rate limiting, and lazy-loads heavy modules (embeddings, search, indexing) to keep startup fast.
- **`settings.py`** — Runtime configuration: reads `config.ini`, routes to the correct LLM provider.
- **`llm_integration.py`** — Provider abstraction layer over OpenAI, Gemini, Anthropic, Grok, HuggingFace, and local GGUF models.
- **`providers.py`** — Multi-provider support helpers.
- **`indexing.py`** — FAISS index creation from document chunks.
- **`search.py`** — Hybrid search combining FAISS (dense) + BM25 (sparse), with cross-encoder reranking via `rag_optimizers.py`.
- **`file_processing.py`** — Extracts text from PDF, DOCX, XLSX, PPTX, TXT.
- **`database.py`** — SQLite metadata storage (`data/metadata.db`) with B-Tree indices on `faiss_idx` and `filename`.
- **`model_manager.py`** — Downloads and manages 25+ GGUF model binaries stored in `models/`.
- **`agent.py`** — Agentic researcher mode; uses `tools.py` for tool calling.
- **`auth.py`**, **`background.py`**, **`websocket_manager.py`** — Auth, background tasks, real-time WebSocket updates.

### Frontend (`frontend/src/`)
- **`App.jsx`** — Root component; holds global state, fetches config/models/prompts from the backend.
- **`components/`** — All UI components. Largest: `SettingsModal.jsx` (34 KB), `ModelManager.jsx` (18 KB), `ModelComparison.jsx` (14 KB), `AgentChat.jsx` (agentic UI).
- **`lib/`** — `logger.js`, `utils.js`.
- **`test/`** — 19 Vitest test files (`.test.jsx`); setup in `test/setup.js`.
- Vite proxies `/api` → `http://localhost:8000` and `/ws` → `ws://localhost:8000`, so frontend never hard-codes the backend URL.

### Data & Runtime Artifacts
All runtime-generated files live in `data/`:
- `metadata.db` — SQLite (source of truth for document metadata)
- `index.faiss` / `index_docs.pkl` / `index_tags.pkl` — FAISS index and supporting data
- `benchmark_results.json`, `app.log`

GGUF model binaries live in `models/` (downloaded on demand).

### CI/CD (`.github/workflows/ci.yml`)
Runs on ubuntu-latest (Python 3.10, Node 18). Three jobs: backend quick tests, frontend Vitest, security scan (Bandit + pip-audit). Triggers on push/PR to main.

## Key Conventions

- **File placement:** application code → `backend/`, utility/test scripts → `scripts/`, generated data → `data/`, model binaries → `models/`.
- **Backend tests** use `unittest` as the primary runner (discovered by `scripts/run_tests.py`); `conftest.py` provides a session-scoped temp SQLite DB for pytest compatibility.
- **Frontend tests** use **Vitest** (not Jest) with `happy-dom` as the DOM environment.
- **LLM provider selection** is driven by `config.ini` / `.env`. API keys go in `.env` (see `.env.example`): `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `GROK_API_KEY`.
- After making changes, update the Change Log in `AGENTS.md`.
