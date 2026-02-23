# Copilot Usage Guide

## What This Repo Is
- AI-powered local document search: FastAPI + FAISS + LangChain + LlamaCpp backend with React/Vite frontend. Supports PDF/DOCX/XLSX/PPTX/TXT ingestion, semantic search, AI summaries, model downloads, caching, and folder history.
- Main runtimes: Python 3.10 (CI uses 3.10; code targets 3.8+), Node 18 (CI) / works on 16+. Data lives in `data/`, models in `models/`.
- CI pipeline: `.github/workflows/ci.yml` runs `python scripts/run_tests.py --quick` (backend) and `cd frontend && npm run test` (frontend) on ubuntu-latest with Python 3.10 + Node 18.

## Bootstrap / Install (always do these first)
1) Python deps: `pip install -r requirements.txt` (needed for FastAPI, psutil, etc.). Skipping this causes import errors in quick tests (e.g., missing `fastapi`, `psutil`).
2) Node deps:
   - Root: `npm install`
   - Frontend: `cd frontend && npm install`
   - Or `npm run install-all` (runs both npm installs but does NOT pip install).
3) Optional venv: create/activate before pip install (e.g., `python -m venv venv && source venv/bin/activate`).

## Run / Dev
- One command: `npm run start` (kills ports 8000/5173 if busy, starts backend via `scripts/start_all.js`, then frontend, opens browser).
- Manual:
  - Backend: `uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000` (ensure `PYTHONPATH` includes repo root; data files under `data/`).
  - Frontend: `cd frontend && npm run dev -- --host 127.0.0.1 --port 5173`.
- Config: `config.ini` at repo root; runtime-generated data in `data/`; GGUF models in `models/`.

## Tests / Validation
- Quick backend suite (CI equivalent): `npm run test` → calls `python scripts/run_tests.py --quick`. Requires pip deps installed; from a clean env without `pip install -r requirements.txt`, expect import errors (`fastapi`, `psutil`) and many loader failures.
- Full backend: `npm run test:full` (slow, includes model-related tests).
- Frontend: `cd frontend && npm run test` (Vitest).
- Stress/model: `npm run test:stress`, `npm run test:model-stress`.
- Coverage: `python scripts/run_tests.py --coverage` (needs pytest-cov installed separately).
- Structure check: `npm run validate`.
- If tests manipulate DB, they autouse SQLite in `data/metadata.db` and clean via `database.cleanup_test_data()` on success.
- Reproduce CI locally: run pip install + `npm run test` (root) and `cd frontend && npm run test`.

## Project Layout (where to edit)
- Backend code: `backend/` (`api.py`, `database.py`, `indexing.py`, `search.py`, `file_processing.py`, `llm_integration.py`, `model_manager.py`, `background.py`, tests in `backend/tests/`).
- Frontend code: `frontend/src/` (`App.jsx`, components/, test/ for Vitest).
- Scripts: `scripts/start_all.js`, `scripts/run_tests.py`, `scripts/benchmark_models.py`, etc.
- Data/Models: generated artifacts in `data/`; GGUF models in `models/`.
- Repo root scripts/package: `package.json` (scripts above), `requirements.txt`, `config.ini`, `AGENTS.md` (workspace rules + Change Log).

## Pitfalls / Tips
- Always install Python deps before running any tests; otherwise quick tests emit numerous import errors (psutil/fastapi).
- `npm run install-all` skips pip; run pip install separately.
- Ports 8000/5173 must be free; startup script will try to clear them.
- Update the Change Log section in `AGENTS.md` after any code/document change.
- Use Vitest (not Jest) for frontend tests; mock LLM/network calls in backend tests.
- Data files (`data/*.db`, `*.faiss`, etc.) are generated; don’t commit large artifacts.

## Trust These Instructions
Follow the sequences above; only search the repo if something here appears incomplete or incorrect.
