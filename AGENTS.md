
# Docu AI Search - Workspace Instructions

> AI-powered local document search with semantic search, LLM integration, and modern React frontend.
> **Always read and update the Change Log section after making any changes.**

## Quick Start
```bash
npm run install-all    # Install all dependencies
npm run start          # Start backend (8000) + frontend (5173)
```

## Tech Stack

**Backend (Python 3.10+):**
- FastAPI - REST API with async support
- FAISS - Vector similarity search
- LangChain - LLM integration framework
- LlamaCpp - Local GGUF model inference
- SQLite - Metadata storage (`metadata.db`)

**Frontend (Node.js 16+):**
- React 19 + Vite - UI framework with HMR
- TailwindCSS - Utility-first styling
- Framer Motion - Animations
- Vitest - Testing (not Jest!)
- Axios - HTTP client

## Project Structure
```
├── backend/                # Python backend code
│   ├── __init__.py         # Package marker
│   ├── api.py              # FastAPI endpoints (main entry)
│   ├── database.py         # SQLite CRUD operations
│   ├── indexing.py         # FAISS index creation
│   ├── search.py           # Semantic search logic
│   ├── file_processing.py  # Text extraction (PDF, DOCX, XLSX, PPTX, TXT)
│   ├── llm_integration.py  # LLM provider abstraction
│   ├── model_manager.py    # GGUF model downloads
│   ├── background.py       # Background task utilities
│   └── tests/              # pytest tests
│       ├── test_api.py, test_database.py, test_search.py
│       ├── test_indexing.py, test_file_processing.py
│       └── test_llm_integration_full.py
├── frontend/               # React frontend
│   ├── src/
│   │   ├── App.jsx         # Main component and routing
│   │   ├── components/     # UI components
│   │   │   ├── SearchView.jsx, ResultCard.jsx
│   │   │   ├── LibraryView.jsx, AgentView.jsx
│   │   │   ├── SettingsModal.jsx, ModelManager.jsx
│   │   │   ├── Sidebar.jsx, Toast.jsx, Logo.jsx
│   │   │   └── BenchmarkView.jsx, IndexingBanner.jsx
│   │   ├── lib/
│   │   │   ├── api.js      # Centralized API client (all backend calls)
│   │   │   ├── logger.js   # Structured logging
│   │   │   └── format.js   # Formatting utilities
│   │   └── test/           # Vitest tests
│   └── vite.config.js      # Vite + proxy config
├── scripts/                # Utility scripts
│   └── run_tests.py        # Test runner
├── data/                   # Runtime data (gitignored)
│   ├── metadata.db         # SQLite database
│   ├── index.faiss         # FAISS vector index
│   └── app.log             # Application log
├── models/                 # GGUF model binaries (gitignored)
├── CLAUDE.md               # Claude Code guidance
├── AGENTS.md               # This file
└── config.ini              # Runtime configuration
```

## Key Rules for Claude Agents

1. **All frontend API calls** must go through `frontend/src/lib/api.js` — never import axios directly in components
2. **Backend tests** use `unittest` (discovered by `scripts/run_tests.py`); frontend tests use **Vitest** (not Jest)
3. **Mock all LLM calls** in tests to avoid API costs
4. **Never commit** `.env`, `data/`, or `models/` directories
5. **Frontend test URLs** must use relative `/api` paths, not hardcoded `http://localhost:8000`
6. **Run `npm run validate`** after structural changes to verify compliance
7. **Update Change Log** in this file after every change

## Development Commands

```bash
# Setup
python -m venv venv_new
pip install -r requirements.txt
npm run install-all

# Run
npm run start              # Backend (8000) + Frontend (5173)

# Test
npm run test               # Backend quick tests (~30s)
npm run test:full          # All backend tests (5-10 min)
npm run test:frontend      # Frontend Vitest
npm run test:all           # All tests
npm run test:stress        # Performance stress tests

# Other
npm run validate           # Project structure check
npm run benchmark          # Model performance benchmarks
```

## Environment Variables (`.env`)

```
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
GROK_API_KEY=
```

## Configuration (`config.ini`)

```ini
[LLM]
provider = openai          # openai | gemini | anthropic | grok | huggingface | local
model_name =               # Optional: override default model for the chosen provider

[Search]
top_k = 10
rerank = true
```

---

## Change Log

> **CRITICAL: Add entry here after EVERY change with date, description, and files.**

### 2026-06-06 (Daily Fix Pass — Phase A PR Merges + Phase B Issue Fixes)

**Phase A — PR Processing:**
- **merged** PR #309 (`fix/issue-306-bm25-bare-except`): narrowed bare `except: pass` in BM25 loader to `(OSError, pickle.UnpicklingError, EOFError, ValueError)` with `logger.warning`. Closes #306.
- **merged** PR #310 (previous fix pass): squash-merged after stale CI checks resolved via `update_pull_request_branch`.
- **merged** PR #311 (`fix/issue-308-dockerfile-nonroot`): added `appuser` non-root user to `Dockerfile`, `USER appuser` before CMD, restricted docker-compose port to `127.0.0.1:8000:8000`. Closes #308.
- **closed** issue #141 as completed (duplicate of issue #308 — same Dockerfile USER directive fix already in main via PR #311).

**Phase B — New Fix PRs:**
- **fix** PR #319 (`fix/issue-317-settings-modal-config-catch`): added `.catch(() => ({ data: {} }))` to `api.getConfig()` in `SettingsModal.jsx` `loadAll()`. Prevents full settings-panel blank-out on transient `/api/config` failure. Auto-merge pending CI. Closes #317. — `frontend/src/components/SettingsModal.jsx`
- **fix** PR #320 (`fix/issue-316-llm-retry-logic`): added `client.with_retry(stop_after_attempt=3)` in cloud path of `generate_ai_answer()` and `stream_ai_answer()`. Transient 429/5xx errors now retried up to 3× before surfacing. Awaits human review. Closes #316. — `backend/llm_integration.py`
- **fix** PR #321 (`fix/p3-batch-2026-06-06`): added `_read_llm_model_override()` reading `[LLM] model_name` from `config.ini`; updated stale provider defaults (`gemini-1.5-flash`→`gemini-2.0-flash`, `claude-3-haiku-20240307`→`claude-haiku-4-5-20251001`, `grok-beta`→`grok-2-1212`). Auto-merge pending CI. Closes #318. — `backend/llm_integration.py`
- **Files**: `backend/indexing.py`, `backend/llm_integration.py` (×2 branches), `frontend/src/components/SettingsModal.jsx`, `Dockerfile`, `docker-compose.yml`, `AGENTS.md`

### 2026-05-28 (Daily Audit Log Consolidation)
- **feat**: Consolidated 28 unique daily automated code audit log entries spanning early May to late May 2026 into a single, unified, reverse-chronologically sorted `internal_audit_log.md` file.
- **verification**: Validated layout and verified project compliance via structure check.
- **Files**: `internal_audit_log.md`, `AGENTS.md`

### 2026-05-28 (Git Branch Consolidation & Test Suite Stabilization)
- **feat**: Consolidated, reviewed, and successfully merged all active development, bugfix, and configuration-preservation branches (`fix/issue-81`, `fix/slowapi-middleware-compat`, `fix/api-files-pagination`, `claude/sleepy-shaw-33a61c`, `claude/interesting-mcclintock-4e358d`, `claude/xenodochial-noether-9e832f`) into `main`.
- **clean**: Cleaned up the repository by deleting all 6 merged local branches, 23 individual issue branches (`fix/issue-55` to `fix/issue-80`), 8 helper `claude/` branches, discarding unmerged sandbox branch `claude/determined-hoover-341185`, and pruning 12 stale git worktrees.
- **fix**: Stabilized the premium sidebar settings UI (`SettingsModal.jsx`) and resolved 9 failing frontend tests (`SettingsModal.test.jsx`) by:
  - Designing a robust axios client factory mock to intercept custom axios instances created via `axios.create()`.
  - Correcting the save changes button text to use premium `'Save Changes'` title capitalization.
  - Adding asynchronous wait logic (`waitFor`) to prevent race conditions during modal configuration load before triggering save clicks.
  - Wrapping all Vitest test renders in a mock `<ToastProvider>` to correctly supply toast context.
  - Aligning tab selectors, aria-labels, and input assertions to the updated sidebar settings panels instead of the obsolete accordion layout.
- **verification**: Ran all backend pytest tests and frontend Vitest tests, achieving 100% pass rates. Verified project structure compliance with `npm run validate`.
- **Files**: `frontend/src/components/SettingsModal.jsx`, `frontend/src/test/SettingsModal.test.jsx`, `task.md`, `AGENTS.md`

### 2026-05-14 (Indexing Failure Hardening)
- **fix**: `create_index` now aborts with a clear log and the empty 8-tuple when (a) every embedding batch fails or (b) the assembled embedding count doesn't match `chunk_strings`. Previously path (a) crashed at `chunk_emb_np.shape[1]` (IndexError on a 1-D empty array) and path (b) silently built a FAISS index whose vectors no longer aligned with the chunk indices stored in `cluster_map`, routing later searches to the wrong chunks. The checkpoint is cleared on abort so the next run re-extracts cleanly.
- **test**: Added `TestIndexingEmbeddingBatchFailures` (all-fail, partial-fail, checkpoint-cleared-on-abort) and `TestIndexingNonexistentFolder` (mixed valid/missing folder list) to `backend/tests/test_indexing.py`.
- **Files**: `backend/indexing.py`, `backend/tests/test_indexing.py`, `AGENTS.md`

### 2026-05-06 (Fix SettingsModal test selectors for dual mobile/desktop nav)
- **Fixed 9 failing Vitest tests in SettingsModal caused by dual navigation rendering**
  - **fix**: `openModal()` helper was using `findByText('Library')` which matched both mobile and desktop nav buttons simultaneously, causing 8 s retries and Vitest 5 s default timeout kills. Changed to `findByText('System Configuration')` (unique modal title).
  - **fix**: All nav-tab `getByText(label)` calls changed to `getAllByText(label)[0]` to handle both mobile (`md:hidden`) and desktop (`hidden md:flex`) nav bars being present in happy-dom (CSS media queries not applied in tests).
  - **Files**: `frontend/src/test/SettingsModal.test.jsx`, `AGENTS.md`

### 2026-05-06 (CLAUDE.md & Frontend Test Fixes)
- **Added CLAUDE.md with codebase guidance for Claude Code sessions**
  - **docs**: Added `CLAUDE.md` with setup commands, architecture overview, testing commands, and project conventions.
  - **fix**: Corrected hardcoded `http://localhost:8000` URLs in `SearchResults.test.jsx` (4 tests) and `SearchHistory.test.jsx` (2 tests) to use relative `/api` paths matching actual component behavior via Vite proxy.
  - **Files**: `CLAUDE.md`, `frontend/src/test/SearchResults.test.jsx`, `frontend/src/test/SearchHistory.test.jsx`, `AGENTS.md`

### 2026-04-30 (Backend CI Stabilization & API Fixes)
- **Resolved critical backend regressions for 100% test pass rate**
  - **fix**: Added `BackgroundTasks` to `/api/search` for offloading search history logging, resolving `test_background_history.py` failure.
  - **fix**: Corrected `health_check` endpoint to use `database.execute()` on pooled connections, fixing 503 errors under load.
  - **fix**: Refactored `stream_answer_endpoint` to use validated `SearchRequest` model instead of raw `request.query`, fixing `AttributeError`.
  - **fix**: Updated `cached_smart_summary` call signature to align with `llm_integration.py` keyword arguments.
  - **fix**: Increased `SearchRequest.query` `max_length` to 5000 characters for long-form semantic search support.
  - **fix**: Updated `test_api.py` mocks to use `get_active_embedding_client` instead of legacy `get_embeddings`.
  - **clean**: Removed transient debug logs from `backend/llm_integration.py`.
  - **Files**: `backend/api.py`, `backend/llm_integration.py`, `backend/tests/test_api.py`, `frontend/src/test/SettingsModal.test.jsx`, `AGENTS.md`

  - **perf**: Implemented robust async test patterns (high timeouts, explicit `findByText` retries, and manual `window.confirm` mocks) to eliminate flakiness in the CI environment.
  - **verification**: All 10 frontend test suites (53 tests) and all backend tests are passing 100%. Verified project structure compliance via `npm run validate`.
  - **Files**: `frontend/src/components/SettingsModal.jsx`, `frontend/src/test/SettingsModal.test.jsx`, `frontend/src/test/ModelManager.test.jsx`, `AGENTS.md`

### 2026-04-29 (Dependency Baseline & Security Hardening)
- **Resolved all critical and high severity vulnerabilities**
  - **fix**: Updated `Pillow` from 10.3.0 to 10.4.0 to patch CVE-2024-28219 (buffer overflow in `_imagingcms`).
  - **fix**: Pinned `torch` to `>=2.4.0` (was `>=2.0.0`) to pull in GHSA-pg7h-5qx3-wjr3 fix.
  - **fix**: Replaced `python-jose` (unmaintained, CVE-2024-33664) with `PyJWT` for all JWT operations in `auth.py`.
  - **fix**: Added `Content-Security-Policy`, `X-Frame-Options`, `X-Content-Type-Options`, and `Strict-Transport-Security` headers in `api.py` middleware.
  - **fix**: Sanitized `filename` parameters in `file_processing.py` to prevent path traversal.
  - **verification**: `pip-audit` reports 0 known vulnerabilities; `bandit` reports 0 high-severity issues.
  - **Files**: `requirements.txt`, `backend/auth.py`, `backend/api.py`, `backend/file_processing.py`, `AGENTS.md`

### 2026-04-28 (Frontend Rebuild & Component Architecture)
- **Rebuilt entire frontend with modern React 19 + Vite architecture**
  - **feat**: Created `Sidebar.jsx` with indigo primary design system and lucide-react icons, replacing old navigation.
  - **feat**: Built `SearchView.jsx` with `ResultCard.jsx` and `HistoryDrawer.jsx` — split from monolithic App.jsx.
  - **feat**: Built `LibraryView.jsx`, `AgentView.jsx`, `BenchmarkView.jsx` as dedicated view components.
  - **feat**: Created `SettingsModal.jsx` (two-pane: General + Model) replacing old inline settings.
  - **feat**: Added `ModelManager.jsx` for GGUF download/manage, `IndexingBanner.jsx` for progress, `Toast.jsx` for notifications.
  - **feat**: Created `frontend/src/lib/api.js` as the single axios client — all components use this, no direct axios imports.
  - **feat**: Created `frontend/src/lib/logger.js` with structured logging and `format.js` for display helpers.
  - **test**: Added `logger.test.js` as the first Vitest test; added `setup.js` for happy-dom environment.
  - **clean**: Removed all old component files, test files targeting removed components, and direct axios imports from components.
  - **Files**: All files in `frontend/src/`, `AGENTS.md`
