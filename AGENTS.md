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
│   │   ├── App.jsx         # Main component (holds global state)
│   │   ├── components/     # UI components
│   │   │   ├── Header.jsx, SearchBar.jsx, SearchResults.jsx
│   │   │   ├── SettingsModal.jsx, ModelManager.jsx
│   │   │   └── SearchHistory.jsx, FileList.jsx, BenchmarkResults.jsx
│   │   └── test/           # Vitest tests
│   └── package.json
├── scripts/                # Utility scripts
│   ├── start_all.js        # Unified startup script
│   ├── run_tests.py        # Test runner
│   └── benchmark_models.py # Model benchmarking
├── data/                   # Generated/runtime files
│   ├── index.faiss         # Vector embeddings
│   ├── index_docs.pkl      # Document chunks
│   ├── index_tags.pkl      # Tag data
│   ├── metadata.db         # SQLite database
│   └── benchmark_results.json
├── models/                 # Downloaded GGUF models
├── .agent/                 # Agent configuration and skills
├── config.ini              # User configuration
├── requirements.txt        # Python dependencies
└── package.json            # Node.js scripts
```

### File Placement Protocol
- **Backend Code**: ALL Python source files MUST go in `backend/` or `backend/tests/`.
  - Exception: `api.py` is the entry point (in `backend/`).
  - NO `.py` files in the project root.
- **Scripts**: Maintenance, build, and benchmark scripts go in `scripts/`.
- **Data**: All generated files (`.db`, `.faiss`, `.log`, `.json` results) MUST go in `data/`.
  - Use `DATA_DIR` constant in code.
- **Models**: Large model binaries go in `models/`
- **Tests**: Python tests in `backend/tests/`; frontend tests in `frontend/src/test/`.
- **Config Files**: `.env`, `config.ini`, `.coderabbit.yaml` in project root.

## API Architecture

```
GET  /                    → Health check
GET  /api/status          → System status
POST /api/search          → Search documents {query, top_k, search_type}
POST /api/answer          → LLM answer {query, context}
GET  /api/stream-answer   → SSE streaming answer
POST /api/index           → Start indexing {folder_path}
GET  /api/index/status    → Indexing status
GET  /api/files           → List indexed files
POST /api/upload          → Upload file
DELETE /api/files/{id}   → Delete file
GET  /api/models          → List GGUF models
POST /api/models/download → Download GGUF model
WS   /ws/indexing         → WebSocket for real-time indexing progress
```

## Critical Rules

### Testing Requirements
- **ALWAYS run tests before committing**: `npm run test` (quick) or `npm run test:all`
- **Frontend tests**: Use Vitest, NOT Jest. Run with `npm run test:frontend`
- **Mock ALL LLM calls** in tests to avoid API costs.
- **No skipping tests** unless explicitly allowed.

### Code Standards
- Use `DATA_DIR` constant for all data file paths.
- All API calls from frontend go through `frontend/src/lib/api.js`.
- No hardcoded localhost URLs in frontend components.
- Backend tests use `unittest` (discovered by `scripts/run_tests.py`).

### Branch & PR Rules  
- Feature branches: `feature/description`
- Fix branches: `fix/description`
- Never commit directly to `main`.
- PR titles should be descriptive and reference issue numbers.

## Common Issues & Solutions

### Import Errors
```bash
# If you see 'ModuleNotFoundError: No module named backend'
export PYTHONPATH=/path/to/project  # Set project root in PYTHONPATH
```

### Frontend Not Connecting to Backend
```bash
# Check Vite proxy config in frontend/vite.config.js
# Should proxy /api to http://localhost:8000
```

### FAISS Index Issues
```bash
# Delete and rebuild if corrupted:
rm data/index.faiss data/index_docs.pkl data/index_tags.pkl
# Then re-index through the UI or API
```

### Database Locked
```bash
# Kill any running backend processes:
pkill -f uvicorn
# Then restart
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:
```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
GROK_API_KEY=...
```

## Change Log

> **CRITICAL: Add entry here after EVERY change with date, description, and files.**

### 2026-06-05 (P3 Batch Fix Pass — Daily Automated Fix)
- **fix**: Added toast error notification to AI stream failure path in `SearchView.jsx` — silent `console.error` catch block now shows `toast.error('AI answer stream failed. Search results are still available.')` (closes #299)
- **fix**: Hardened `scripts/e2e_verify.py` — replaced bare `except:` with `except requests.RequestException:` to avoid swallowing `KeyboardInterrupt`; replaced unbounded `while True:` loop in `trigger_indexing()` with a 300-iteration (10-min) hard cap with timeout fallback (closes #300)
- **Files**: `frontend/src/components/SearchView.jsx`, `scripts/e2e_verify.py`, `internal_fix_log.md`, `AGENTS.md`

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
  - **fix**: Resolved 5 failing Vitest tests (`SearchHistory`, `FileList`, `BenchmarkResults`, `Header`, `SearchBar`) caused by missing `vi.mock('axios')` in test setup — added `frontend/src/test/setup.js` as global Vitest setup file.
  - **fix**: Fixed logger.js import warning by using explicit `.js` extension in logger.test.js.
  - **Files**: `CLAUDE.md`, `frontend/src/test/setup.js`, `frontend/src/test/logger.test.js`, `AGENTS.md`

### 2026-05-05 (Frontend Rebuild)
- **Rebuilt frontend with new component architecture (post-2026-05 rebuild)**
  - **feat**: Rebuilt all frontend components with new indigo-primary design system:
    - `Sidebar.jsx` — left nav (desktop + mobile drawer).
    - `SearchView.jsx` — primary search with `ResultCard.jsx` and `HistoryDrawer.jsx`.
    - `LibraryView.jsx` — indexed-folder browser.
    - `AgentView.jsx` — agentic researcher chat.
    - `BenchmarkView.jsx` — benchmark runner.
    - `SettingsModal.jsx` — two-pane settings.
    - `ModelManager.jsx` — GGUF model downloads.
    - `IndexingBanner.jsx` — global indexing-progress banner.
    - `Toast.jsx`, `Logo.jsx`, `ErrorBoundary.jsx` — utilities.
  - **feat**: Centralized all API calls through `frontend/src/lib/api.js`.
  - **feat**: Added `frontend/src/lib/logger.js`, `format.js`, `utils.js`.
  - **test**: Added `frontend/src/test/logger.test.js` and `frontend/src/test/setup.js`.
  - **Files**: All files in `frontend/src/`

### 2026-04-20 (Security Fixes: Command Injection Prevention)
- **Fixed critical command injection vulnerability in file processing endpoint**
  - **fix**: Replaced `subprocess.call(f'process {filename}', shell=True)` with `subprocess.run(['process', filename], shell=False)` in `backend/api.py`.
  - **test**: Added `TestSecurityCommandInjection` to `backend/tests/test_security_command_injection.py` to verify the fix.
  - **Files**: `backend/api.py`, `backend/tests/test_security_command_injection.py`
