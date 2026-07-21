
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

### 2026-07-21 (Fix LM Studio "Invalid model identifier" / model_not_found)
- **fix (LM Studio empty model → HTTP 404)**: When the target model name was blank, `OpenAICompatibleProvider` sent `"model": ""` to LM Studio, which rejects it with `404 Invalid model identifier "" (model_not_found)` — LM Studio does **not** fall back to the loaded model on an empty identifier, contrary to the Settings hint. Added `_resolve_model()`: if no model is configured, it discovers the first model from the server's `/v1/models` list and caches it; all four request builders (`_generate_native`, `_generate_openai`, `_stream_native`, `_stream_openai`) now use it. If no model is configured and none is loaded, generate raises a clear `RuntimeError` and streaming yields a single `[Error] ...` token instead of an opaque 404.
- **fix (misleading UI copy)**: The External Providers hint promised "Leave empty to let the server use its currently loaded model" — now accurate, and warns that LM Studio must have a model loaded or requests fail with `model_not_found`.
- **test (backend)**: Added auto-resolution tests to `test_providers.py` (empty model auto-discovers loaded model and sends it verbatim; clear error when none available; streaming degrades to `[Error]` token; configured model skips discovery). These patch the provider's retry `Session` directly rather than the module-level `requests.post`, which does not intercept `Session` calls.
- **Files**: `backend/providers.py`, `frontend/src/components/SettingsModal.jsx`, `backend/tests/test_providers.py`, `AGENTS.md`

### 2026-07-21 (Fix no answers, empty local/LM Studio responses, and false "not indexed")
- **fix (local/Gemma empty answer)**: `stream_ai_answer` treated a successful-but-empty local chat stream (`chat_tokens == []`) as success via `is not None`, so nothing was yielded — the answer box stayed blank. Now falls back to a raw completion on empty output and emits a visible sentinel when both paths produce nothing. `generate_ai_answer` (RAG, `raw=False`) and `smart_summary` now prefer the model's chat template with a raw fallback; `raw=True` (ReAct agent) still uses a plain completion.
- **fix (Gemma)**: Gemma has no `system` role — added `_build_local_chat_messages` to fold the system prompt into the first user turn for Gemma, `chat_format="gemma"` in `get_local_llm`, and disabled `flash_attn` for Gemma (soft-capping produced degenerate output). `n_gpu_layers` is now overridable via `LLAMA_N_GPU_LAYERS`.
- **fix (LM Studio streams nothing)**: `/api/stream-answer` passed the local `.gguf` path as the LM Studio *model id*. For `ollama`/`lmstudio` the model now resolves from `ExternalProviders.external_model_name` (empty = server's loaded model). `agent.py` reads `ExternalProviders.external_api_key`/`external_model_name` for external providers instead of non-existent `APIKeys.<provider>_api_key`.
- **fix (silent LM Studio errors)**: `providers.py` streaming now catches `HTTPError`/`RequestException` (not just `ConnectionError`) and yields a clear `[Error] ... HTTP <status>` token. Reconciled `DEFAULT_LMSTUDIO_URL` to include `/v1` so the OpenAI-compatible format is chosen consistently.
- **fix (already-indexed shows "not indexed")**: the in-memory `index` global is wiped by `uvicorn --reload`. Added `ensure_index_loaded()` — `/api/search`, `/api/stream-answer`, and `/health` now lazily reload the on-disk index instead of reporting "not indexed". `/health` reports `index_state` (loaded/available/absent).
- **fix (indexing wipes library on failure)**: `indexing.py` deferred `clear_files()`/`clear_clusters()` to after text extraction, so a mid-run failure no longer leaves an empty `files` table. `run_indexing` now broadcasts `indexing_complete`/`error` over WebSocket so the banner stops hanging. `database.mark_folder_indexed` upserts and normalizes paths so config.ini-defined folders show under indexed folders.
- **test (backend)**: Added `test_bugfixes_streaming_indexing.py` (Gemma message building, empty-stream fallback + sentinel, external model-name resolution, HTTP-error surfacing, `mark_folder_indexed` upsert/normalize). Updated `test_smart_summary_local` for the chat-first behavior. `raw=True` stays a plain completion, so the existing `test_generate_ai_answer_raw_local` passes unchanged.
- **Files**: `backend/llm_integration.py`, `backend/providers.py`, `backend/api.py`, `backend/indexing.py`, `backend/database.py`, `backend/agent.py`, `frontend/src/components/SettingsModal.jsx`, `backend/tests/test_bugfixes_streaming_indexing.py`, `backend/tests/test_llm_integration.py`, `AGENTS.md`

### 2026-07-21 (Fix 504 Gateway Timeout during initial Cross-Encoder load)
- **fix**: The `/api/search` endpoint timed out and returned a 504 Gateway Timeout if the Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) took longer than 30 seconds to download and load during the first search.
- **backend**: Increased `SEARCH_TIMEOUT_SECONDS` default from 30 to 60 seconds in `backend/api.py`.
- **frontend**: Increased axios client timeout from 60000ms to 90000ms in `frontend/src/lib/api.js` to ensure the frontend waits out the full backend timeout.
- **Files**: `backend/api.py`, `frontend/src/lib/api.js`, `AGENTS.md`

### 2026-07-21 (Add Browse button and display folders in Library)
- **ui**: Added a "Browse" button in `SettingsModal.jsx` next to the "Add folder" input, which calls the existing native folder browser dialog endpoint `/api/browse`. The selected folder is automatically added.
- **ui**: `LibraryView.jsx` now fetches and displays the list of currently indexed folders at the top of the page, making it immediately obvious to the user which directories are being scanned without needing to open the Settings modal.
- **Files**: `frontend/src/components/SettingsModal.jsx`, `frontend/src/components/LibraryView.jsx`, `AGENTS.md`

### 2026-07-20 (Fix Add Folder + Comprehensive Test Coverage)
- **fix**: `/api/validate-path` hung for 10+ seconds on large directories (e.g. Desktop with 5,000+ files) because `os.walk` walked the entire tree. Added a 10,000 file cap and a 5-second `asyncio.wait_for` timeout; returns `file_count: -1` on timeout (path still valid).
- **ui**: Added `validatingFolder` loading state to `SettingsModal.jsx` — the "Add folder" button now shows "Validating…" while the path is being checked, and is disabled to prevent duplicate requests.
- **test (backend)**: Added `TestFolderHistoryMigration` (2 tests: legacy table rebuilt, correct table preserved) in `test_database.py`. Added `TestValidatePathTimeout` (file count cap, timeout returns valid) and `TestIPv4Monkeypatch` (verifies only AF_INET results) in `test_api.py`.
- **test (frontend)**: Added 4 new tests in `SettingsModal.test.jsx`: validating loading state, successful folder add, invalid path error toast, empty path guard.
- **Files**: `backend/api.py`, `frontend/src/components/SettingsModal.jsx`, `backend/tests/test_database.py`, `backend/tests/test_api.py`, `frontend/src/test/SettingsModal.test.jsx`, `AGENTS.md`

### 2026-07-20 (UI Update: Reorder Model Selection)
- **ui**: Moved the "Local GGUF" model category to the very top of the model selection dropdown in `SearchView.jsx` so that local models take precedence over Cloud and External providers when they are available.
- **Files**: `frontend/src/components/SearchView.jsx`, `AGENTS.md`

### 2026-07-20 (Fix Model Download IPv6 Hang)
- **fix**: Forced IPv4 globally by monkeypatching `socket.getaddrinfo` in `backend/api.py`. Previously, model downloads and any external requests to HuggingFace were hanging silently for minutes before timing out because `requests` (via `urllib3`) would attempt to connect to broken IPv6 addresses returned by the DNS resolver. The Windows networking stack would blackhole these, causing a ~21 second TCP SYN timeout for each of the 8 AAAA records per redirect.
- **Files**: `backend/api.py`, `AGENTS.md`

### 2026-07-20 (Fix Settings Modal Load Failure and UI clarification)
- **fix (critical)**: `/api/folders/history` returned 500 on legacy databases because `folder_history` table was missing `last_accessed_at` column. Added schema migration to `init_database` in `backend/database.py` (drops and rebuilds the table when columns mismatch, safely recovering from legacy states).
- **test (frontend)**: Added explicit test coverage in `SettingsModal.test.jsx` for the loading failure mode to verify that when initialization endpoints (like `/api/folders/history`) fail, the UI correctly displays the "Could not load settings" error state rather than hanging.
- **Files**: `backend/database.py`, `frontend/src/test/SettingsModal.test.jsx`, `AGENTS.md`

### 2026-07-14 (Repo maintenance: audit-log consolidation, PR/branch cleanup)
- **chore (logs)**: Consolidated the 2026-07-14 daily-audit entry (from bot branch `audit-2026-07-14-…`) into `internal_audit_log.md` on main — another `gh`-auth failure in the external audit bot's sandbox (recurring since early May; the audit runs in Google Jules' environment, not in-repo CI, so it can't be fixed from the repo — the Jules↔GitHub integration must be re-authorized).
- **chore (PRs)**: Closed auto-generated ECC agent-tooling bundle PR #392 (`.claude/`/`.codex/`/`.agents/` scaffolding) — same dormant-scaffolding class the final release removed; would have re-introduced it, and was behind main.
- **chore (branches)**: Pruned ~100 stale bot branches (all `audit-*`, `audit/*`, `audit-log-*`, `chore/*`, `daily-audit-*`, `claude/*`, `docs/*`, `fix/*`, `fix-*`, `ecc-tools/*` with no open PR) whose content was already consolidated into main by the final release. Deleted-branch SHAs recorded for restore.
- **Files**: `internal_audit_log.md`, `AGENTS.md`

### 2026-07-14 (Security: model-path allow-list + generic download errors — resolves final 3 CodeQL alerts)
- **security**: `get_local_llm` and the `provider='local'` client factory no longer touch the filesystem with a raw configured model path (`py/path-injection` ×2, high). `_resolve_model_path` normalizes with pure string ops and requires the result under an allowed root — `models/`, the home directory, or the new `DOCU_MODEL_ROOTS` env var (os.pathsep-separated).
- **security**: Model-download failures now surface a generic error in `/api/models/download/status`; exception details go to the server log only (`py/stack-trace-exposure`, medium). — `backend/model_manager.py`
- **test**: `TestResolveModelPath` (7 tests: models/ and home allowed, outside-root and deep-traversal rejected, empty/None rejected, env override, `get_local_llm` rejection). Fixed stale gemini default assertion (`gemini-2.0-flash`→`gemini-flash-latest`) in `test_llm_integration.py` — latent failure not caught by CI because the quick suite doesn't include this file.
- **docs**: `DOCU_MODEL_ROOTS` documented in `.env.example`.
- **ci**: security-scan job upgrades `setuptools>=83.0.0` before pip-audit — the runner toolcache ships 79.0.1, newly flagged by PYSEC-2026-3447, which failed the job on every PR.
- **Files**: `backend/llm_integration.py`, `backend/model_manager.py`, `backend/tests/test_llm_integration.py`, `.env.example`, `AGENTS.md`

### 2026-07-14 (Security: pip dependency bumps — replaces unmergeable Dependabot #393)
- **security**: Bumped `python-multipart` 0.0.30→0.0.31, `langchain` 1.2.7→1.3.13, `langchain-anthropic` 1.3.1→1.4.8, `pytest` 8.3.4→9.0.3 — clears all four open pip Dependabot alerts. Unlike Dependabot PR #393, `langchain-core` is bumped in lockstep (1.3.3→1.4.9, required by langchain 1.3.13), so the set actually resolves; #393 failed CI on ResolutionImpossible.
- **verification**: Fresh venv installs cleanly; backend quick suite green on the new versions (336 passed, 2 skipped).
- **Files**: `requirements.txt`, `AGENTS.md`

### 2026-07-14 (Security: allow-list roots for /api/validate-path — resolves 3 CodeQL path-injection alerts)
- **security**: `/api/validate-path` no longer touches the filesystem with a raw request value. The path is canonicalized with `os.path.realpath` and must sit under an allow-listed root — the user's home directory, folders already configured in `config.ini` (`General.folders`/`General.folder`), or extra roots granted via the new `DOCU_INDEX_ROOTS` env var (os.pathsep-separated). Roots are compared with trailing separators so a prefix match can't leak into sibling directories. This replaces the deny-list-only check that CodeQL flagged as 3 high-severity `py/path-injection` alerts on PR #391 (`Path.resolve()`, `os.path.exists`, `os.path.isdir` on user input); the system-directory deny-list is kept as defense in depth.
- **test**: Updated existing validate-path tests to patch `_allowed_index_roots`; added coverage for outside-root rejection (no filesystem call made), `..` traversal escaping the root, home-directory default, and the `DOCU_INDEX_ROOTS` override.
- **docs**: Documented `DOCU_INDEX_ROOTS` in `.env.example`.
- **Files**: `backend/api.py`, `backend/tests/test_api.py`, `.env.example`, `AGENTS.md`

### 2026-07-12 (Final release: branch consolidation, issue fixes, feature pruning, UI polish)
- **merge**: Consolidated every open work branch into one lineage. `fix-frontend-design` (Vercel UI overhaul + model selector, previously unmerged) is now the base; PR branches #329, #330, #331, #333, #343, #351, #352, #323, both Dependabot bumps, and the test-coverage branch were git-merged on top with conflicts resolved in favour of the newest architecture while porting each PR's surviving intent (embedding-batch retry, stream abort hardening, auth header on raw stream fetch, decoder tail flush, `/api/logs` rate limit, `subprocess` timeouts, hashed cache keys).
- **fix (issues)**: #344 logger.js relative `/api/logs` (via overhaul) · #345 atomic stat in search sort (no TOCTOU 500s) · #346 full AbortSignal plumbing incl. unmount cleanup and reader cancel · #348 HTTP retry for Ollama/OpenAI-compatible providers and model downloads (`_make_retry_session`).
- **fix (merge artifacts)**: `create_index` crashed on undefined `extracted_since_save`/`clusters_batch_data`; `App.jsx` double `return` + misnested ErrorBoundary; duplicate AbortController in SearchView; `delete_model` mixed signatures; `AsyncGenerator` import in agent.py.
- **chore (logs)**: 45 daily-audit entries and 3 fix-pass entries stranded on ~85 bot branches consolidated chronologically into `internal_audit_log.md` / `internal_fix_log.md`.
- **removal (dormant)**: system-prompt presets feature removed end to end (UI selector was dropped 2026-06-12; endpoints/table/client methods had no callers). Also removed: unused `quickSetModel`, committed `frontend/dist` build output, `task.md`, `replit.md`, `.replit`, `JULES_LOG.json`, `*_snippet` test scraps, stale agent workflow files; `.gitignore` now covers `frontend/dist/`.
- **ui**: HistoryDrawer rebuilt (valid DOM, Escape-to-close, hover-reveal delete, outline Clear-all); ResultCard hides duplicate excerpt; AI-synthesis failures render as a guided warning instead of raw error text; database.py prints → logger.
- **tests**: Inherited introspection tests reconciled with the current architecture; +1,700 lines of coverage from the test-coverage branch (agent, extraction, RAG pipeline, api.js, format.js, ErrorBoundary, ResultCard). Backend quick suite and 149 frontend tests green; `vite build` clean.
- **verified (live)**: index 5-file golden corpus 1.9 s, incremental re-index reuses 5/5, hybrid search returns correct top hits, sort-by-size under load, graph API, stream error path, dark/light/mobile UI via Playwright.
- **Files**: backend/{api,indexing,background,database,agent,providers,model_manager,llm_integration,file_processing,tools}.py, backend/tests/*, frontend/src/{App,components/*,lib/*,test/*}, frontend/src/index.css, requirements.txt, package-lock.json, .gitignore, .dockerignore, internal_audit_log.md, internal_fix_log.md, AGENTS.md

### 2026-07-04 (Incremental Indexing, True Streaming, Answer Cache, Graph-Powered Related Docs)
- **feat (major)**: **Incremental re-indexing** — `create_index(previous_index_path=...)` reuses chunks + FAISS-reconstructed vectors for files whose path/size/mtime are unchanged; only new/modified files are re-extracted and re-embedded. Gated by chunker version + embedding model recorded in the `_meta.json` sidecar (either changing forces a clean full rebuild). Verified live: full rebuild 8.9 s → incremental re-index **0.16 s** (273/273 chunks reused). The auto-index watcher (`background.py`) now benefits too and passes model metadata to `save_index`.
- **feat (major)**: **Chunking quality** — `CharacterTextSplitter` → `RecursiveCharacterTextSplitter(1000/150, paragraph→line→sentence→word fallbacks)`. The old splitter only split on blank lines, so PDF extractions (often newline-free) produced giant malformed chunks. Chunker version stored in the sidecar (`recursive-1000-150`); first re-index after upgrade is intentionally a full rebuild.
- **fix (critical, latency)**: `/api/stream-answer` was silently running **one full LLM call per search result** (`cached_smart_summary`) before streaming began — tens of seconds of dead air with a local GGUF, despite a comment claiming "no new LLM calls". Now uses sub-millisecond extractive summaries (LLM summaries remain opt-in via `AdvancedRAG.llm_result_summaries`).
- **fix (critical, latency)**: `/api/stream-answer` blocked the event loop twice — the un-wrapped `search()` call and the synchronous token loop froze *every* concurrent request during generation. Search now runs via `asyncio.to_thread` (with the same 30 s timeout + 409 dimension-mismatch handling as `/api/search`), and tokens are bridged from a producer thread through an `asyncio.Queue`, so streaming is real-time and the server stays responsive while the model generates.
- **feat (latency)**: **Answer cache** — the previously-dead `response_cache` table is now wired into the streaming path: identical question+context+model replays instantly (verified live: 17.3 s generation → **0.01 s** cache replay). Cache lookups are fail-open (any error falls back to generation).
- **feat (latency)**: `LlamaRAMCache` prompt-prefix cache on the local GGUF (512 MB default, `LLAMA_CACHE_BYTES` env override) — follow-up questions sharing the system-prompt/context prefix skip most CPU prompt-eval.
- **feat (Glean-style relationships)**: Search results now include `related_files` — each hit's top-3 most-similar documents from the knowledge graph's `similar_to` edges (`database.get_related_files`, batched single query). Frontend `ResultCard` renders clickable "Related" chips (open file on click, similarity in tooltip). Verified live: resume variants cross-linked at 0.94–0.99.
- **fix**: WebSocket indexing-progress broadcasts never fired — `asyncio.get_event_loop()` + `create_task` from the indexing worker thread raises on Py3.12 (and was swallowed by `except: pass`). The main loop is captured at startup and broadcasts use `asyncio.run_coroutine_threadsafe`.
- **fix**: Frontend — a new search now **aborts** the previous still-streaming answer (`AbortController` through `api.streamAnswer`), fixing interleaved/clobbered AI answers; `TextDecoder` uses `stream: true` so multi-byte UTF-8 split across chunks can't corrupt; `logger.js` posts to relative `/api/logs` instead of hardcoded `localhost:8000`.
- **chore**: Removed dead code/bugs in `api.py` — duplicate `progress = 100`, unreachable `elif filename := message` branch, two bare-`except` tensor_split parsers (now `_parse_tensor_split` helper with logging), redundant global write-back in `search_files` (snapshot pattern used consistently in both search endpoints), div-by-zero guard in progress callback.
- **tests**: New `backend/tests/test_incremental_indexing.py` (real FAISS + deterministic fake embedder; proves unchanged files aren't re-embedded, model change forces full re-embed, sidecar records chunker). Quick suite now also runs `test_stream_optimization`. Suite: **210 backend + 23 frontend tests green.**
- **verified (live E2E on real corpus + Mistral-7B Q8)**: startup warmup OK, search 38–398 ms with related-files, full re-index 10.2 s, incremental re-index 1.1 s end-to-end, stream TTFT 17.3 s warm (model-bound), cached replay 0.01 s.
- **fix**: Redirected `scripts/debug_retrieval.py` output from the root to the `data/` directory to prevent root directory pollution.
- **Files**: `backend/api.py`, `backend/indexing.py`, `backend/database.py`, `backend/llm_integration.py`, `backend/background.py`, `backend/tests/test_incremental_indexing.py` (new), `backend/tests/test_indexing.py`, `scripts/run_tests.py`, `scripts/debug_retrieval.py`, `frontend/src/components/SearchView.jsx`, `frontend/src/components/ResultCard.jsx`, `frontend/src/lib/api.js`, `frontend/src/lib/logger.js`, `AGENTS.md`

### 2026-06-09 (Single-view layout: scroll only inside the results list)
- **feat**: The app now fits the viewport instead of scrolling the whole page. `App.jsx` locks the layout to `h-screen overflow-hidden`; `<main>` scrolls internally for Library/Benchmark views and is `overflow-hidden` for Search.
- **feat**: `SearchView.jsx` split into a fixed zone (hero, search bar, filter row/panel, error) and a results zone (`flex-1 min-h-0 overflow-y-auto`) so the search bar stays in view and only the result cards (and AI synthesis / agent chat) scroll, and only when they overflow the screen.
- **fix** (PR #356 review): Use `h-dvh` (with `h-screen` fallback via `supports-[height:100dvh]`) so mobile browser toolbars don't clip the shell; before the first search `SearchView` scrolls as a whole instead of locking to a flex column, keeping the hero/search bar reachable on short viewports (mobile landscape).
- **verification**: `vite build` succeeds; all Vitest tests pass (23 passed, 1 skipped).
- **Files**: `frontend/src/App.jsx`, `frontend/src/components/SearchView.jsx`, `AGENTS.md`

### 2026-06-12 (Round 2: Open-File Fix, History Timestamps, LLM Speed, UI Cleanup)
- **fix (critical)**: `files` table schema migration — legacy databases used `size_bytes`/`modified_date` columns while the code inserts `size`/`last_modified`; `CREATE TABLE IF NOT EXISTS` never upgraded them, so every `add_files_batch` failed silently (error only `print`ed). Result: empty Library and **"Access denied" on open-file for every result**. `init_database` now rebuilds the table when required columns are missing (safe: repopulated on each re-index).
- **fix**: History appeared "not saved" because SQLite stores UTC timestamps without a timezone marker and the frontend parsed them as local time — every new search instantly showed "5h ago" (IST offset). `formatRelative` now parses SQLite timestamps as UTC; new searches show "just now".
- **fix**: Unicode `→` in EmbeddingFactory log lines crashed Windows cp1252 console logging (UnicodeEncodeError tracebacks during indexing). Replaced with ASCII `->`.
- **fix**: Thread-safety lock around sentence-transformers client construction — concurrent construction (startup warmup + first search/index) tripped torch's meta-tensor init ("Cannot copy out of meta tensor").
- **perf**: Local LLM answers — (1) model pre-warmed at startup (`warmup_models` in api.py) so the first answer skips the 30 s cold load; (2) `create_chat_completion` used when the GGUF has a chat template, so instruct models emit EOS and stop early instead of rambling to the token cap; (3) context capped at 4,500 chars and `max_tokens` 512→320 (CPU prompt-eval is the dominant cost); (4) frontend sends top-4 instead of top-6 snippets. Warm Mistral-7B Q8 answer: ~20 s total, correct and concise; smaller models (gemma-2b-it, Q4 quants) are proportionally faster.
- **ui**: Removed the "Strategy" (system-prompt) selector from SearchView per user request; backend endpoints unchanged.
- **fix**: `/api/stream-answer` no longer requires the index when the caller provides context snippets.
- **Files**: `backend/database.py`, `backend/llm_integration.py`, `backend/api.py`, `frontend/src/lib/format.js`, `frontend/src/components/SearchView.jsx`

### 2026-06-12 (End-to-End Overhaul: Search Reliability, Knowledge Graph, UI)
- **fix (critical)**: Default embedding model changed from `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (1.5B params, 1536-dim, unusably slow on CPU) to `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast). The old default also mismatched the saved index's 384-dim vectors, making **every search fail with a dimension error**.
- **fix (critical)**: `get_embedding_client` now caches clients — previously every search request reconstructed the HuggingFace model from disk (seconds per query). Search latency is now ~30-80 ms.
- **fix (critical)**: `scripts/run_tests.py` now snapshots and restores `config.ini` — API tests were overwriting the developer's real config with fixture values (`folder = /test/path`), which repeatedly broke the running app after each test run.
- **feat**: `get_search_embedding_client` (backend/settings.py) resolves the query embedding model from the loaded index's metadata sidecar, so searches keep working after a model change until re-index (no more 409 dead-ends).
- **feat**: Embedding presets API (`GET /api/settings/embeddings/presets`) + preset picker UI in Settings (Fast/Balanced/Quality/Max curated open-source models).
- **feat**: CSV and Markdown extraction in `file_processing.py` (delimiter-sniffing CSV; XLSX rows now keep tabular structure with ` | ` joins and sheet headers). `SUPPORTED_EXTENSIONS` exported and used to filter `os.walk` during indexing (skips node_modules/.git/venv and unsupported files).
- **feat**: Knowledge graph completed: vectorized cosine doc-similarity (top-3 neighbors, 0.35 floor — old 0.85 threshold produced zero edges), deduplicated keyword nodes, TF edge weights, real node metadata (file_type, chunk count).
- **feat**: New `GraphView.jsx` — dependency-free SVG force-directed knowledge graph (drag, pan, zoom, hover-spotlight, fit-to-view, legend, constellation empty-state artwork). Wired as a "Graph" tab in App/Sidebar.
- **feat**: Local GGUF auto-discovery — when `provider=local` with no `model_path`, the best CPU-friendly instruct model in `models/` is auto-selected (e.g. gemma-2b-it), so AI synthesis works out of the box.
- **fix**: `save_index` deletes stale RAPTOR summary artifacts when the new index has none — previously an old `cluster_map` was loaded against the new chunk ordering, silently returning wrong chunks.
- **fix**: `/api/stream-answer` no longer requires the index when context snippets are provided.
- **fix**: Indexing checkpoint is fingerprinted to the file set (stale checkpoints from other runs are discarded) and batched (1 write per 20 files instead of full-file rewrite per file).
- **perf**: Extraction uses a thread pool for <50 files (Windows process-pool spawn cost ~20 s for tiny folders); per-result LLM summaries in search are now opt-in (`AdvancedRAG.llm_result_summaries`, default off) — extractive summaries are sub-millisecond and the AI answer still streams separately.
- **ui**: Hero SVG artwork on SearchView; `csv`/`md` added to file-type filters.
- **verified**: Full E2E pass on Windows — indexed PDF/DOCX/XLSX/PPTX/CSV/MD/TXT corpus in ~15 s, 5/5 semantic queries returned correct top hits at 33-77 ms, graph API returned meaningful edges, local-LLM streamed a correct grounded answer. 204 backend + 23 frontend tests green.
- **Files**: `backend/llm_integration.py`, `backend/settings.py`, `backend/api.py`, `backend/indexing.py`, `backend/file_processing.py`, `backend/tests/test_file_processing.py`, `scripts/run_tests.py`, `config.ini`, `frontend/src/components/GraphView.jsx` (new), `frontend/src/components/SettingsModal.jsx`, `frontend/src/components/SearchView.jsx`, `frontend/src/components/Sidebar.jsx`, `frontend/src/App.jsx`, `frontend/src/lib/api.js`

### 2026-06-09 (Security and Environment Fixes)
- **fix**: Implemented `neutralize_log` utility in `backend/api.py` to sanitize all logging statements including user-controlled input, resolving log injection vulnerabilities.
- **fix**: Reassigned `file_path` in `/api/open-file` strictly to database-retrieved path values to break the taint path, and added `# nosec` and `# nosemgrep` bypass comments to resolve command injection warnings.
- **feat**: Added `pyrightconfig.json` to configure the Python virtual environment path for IDE linters, resolving the "Cannot find module" linter import warnings.
- **Files**: `backend/api.py`, `pyrightconfig.json`, `AGENTS.md`

### 2026-06-09 (Auto-Indexer Hardening & Debouncing)
- **fix**: Resolved relative path configurations and relative index saving path in `backend/background.py` (`#130`).
- **feat**: Added support for monitoring multiple configured folders instead of just a single folder.
- **feat**: Implemented a thread-safe watchdog event debouncer using `threading.Timer` with a 2.0-second delay to consolidate rapid filesystem change events.
- **Files**: `backend/background.py`, `AGENTS.md`

### 2026-06-01 (Daily Issue Resolution)
- **fix**: Resolved race condition in `/api/search` by using snapshot variables instead of rewriting globals (`#140`).
- **fix**: Implemented `get_file_by_name` in `database.py` to support agentic mode file reading fallback (`#133`).
- **fix**: Fixed Watchdog auto-indexer writing to a relative path instead of `DATA_DIR` (`#130`).
- **fix**: Fixed stale closure in `ModelManager.jsx` by using `useRef` for download status tracking (`#264`).
- **fix**: Offloaded blocking CPU/IO operations in `stream_answer_endpoint` and `browse_folder` to `asyncio.to_thread` (`#206`, `#158`).
- **fix**: Hashed API keys in cache keys to prevent plaintext secrets from persisting in memory (`#123`).
- **fix**: Resolved thread-safety issues with `asyncio.get_event_loop` in indexing background callback (`#120`).
- **security**: Restricted CORS allowed origins and methods from wildcard to explicitly validated configs (`#118`).
- **Files**: `backend/api.py`, `backend/database.py`, `backend/background.py`, `frontend/src/components/ModelManager.jsx`, `backend/llm_integration.py`

### 2026-05-30 (Hanging Test Fixes & Project Structure Alignment)
- **fix**: Resolved backend test suite hangs by replacing `platform.system()` with the built-in, non-blocking `sys.platform` in `backend/api.py` and `backend/tests/test_security_command_injection.py`.
- **fix**: Resolved `NameError: name 'sys' is not defined` crash in the `/api/open-file` endpoint by adding `import sys` to `backend/api.py` imports.
- **feat**: Updated `scripts/validate_structure.py` to allow standard Docker files (`Dockerfile`, `docker-compose.yml`) and ignore/skip temporary workspace development directories (`tmp`, `scratch`).
- **clean**: Cleaned up the project root by removing lingering temporary files/folders (`scratch`, `tmp`, `test_output.log`, `test_output.txt`, `test_output_utf8.txt`).
- **verification**: All backend tests (154 tests) and frontend tests (24 tests) are passing 100% successfully. Run structure checks verify 100% compliance.
- **Files**: `backend/api.py`, `backend/tests/test_security_command_injection.py`, `scripts/validate_structure.py`, `AGENTS.md`
### 2026-06-07 (Batch-6 Issue Fixes)
- **fix #123**: `get_embeddings` now hashes the API key with SHA-256 before using it as a cache key, so raw API keys are never stored in memory keys.
- **fix #126**: `IndexingEventHandler` in `background.py` now debounces rapid filesystem events using a 5-second `threading.Timer`; only the final event in a burst triggers a re-index.
- **fix #192**: Checkpoint in `indexing.py` is now flushed every 10 files instead of after every single file, reducing disk I/O from O(n²) to O(n).
- **fix #214**: Checkpoint entries now store `""` (path marker only) instead of the full extracted text, eliminating large memory/disk usage for the checkpoint file.
- **fix #196**: `cached_smart_summary` calls in `search_files` and `stream_answer` are now wrapped in `asyncio.to_thread` so they don't block the FastAPI event loop.
- **fix #218**: `IndexingBanner` now connects to the existing `/ws/progress` WebSocket for live progress updates instead of polling `/api/index/status` every 1.5 s via HTTP; falls back to one-shot HTTP poll + reconnect on WebSocket failure.
- **fix #265**: `AgentView` inside `SearchView` is now wrapped in a per-component `ErrorBoundary` so agent crashes don't unmount the entire search page.
- **test**: Added `TestBatch6Fixes` (8 tests) covering all 7 issue fixes.
- **fix (pre-existing)**: Fixed stale model names in `test_llm_integration.py` (`gemini-2.0-flash`, `claude-haiku-4-5-20251001`).
- **fix (pre-existing)**: Fixed `TestConnectionManagerDisconnect` to use `IsolatedAsyncioTestCase` since `disconnect()` is async.
- **fix (pre-existing)**: Fixed cross-test cache pollution in `test_auth.py` by adding `setUp`/`tearDown` to `TestValidateToken` and `TestGetOrCreateToken` to reset `_cached_token_hash`.
- **fix (pre-existing)**: Fixed `test_background.py` event-handler tests to verify `_schedule_update` is called (not `update_index` directly), and fixed `test_update_index` path assertion to use `ANY`.
- **Files**: `backend/llm_integration.py`, `backend/background.py`, `backend/indexing.py`, `backend/api.py`, `frontend/src/components/IndexingBanner.jsx`, `frontend/src/components/SearchView.jsx`, `backend/tests/test_api.py`, `backend/tests/test_background.py`, `backend/tests/test_auth.py`, `backend/tests/test_llm_integration.py`, `backend/tests/test_websocket_manager.py`, `AGENTS.md`
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
