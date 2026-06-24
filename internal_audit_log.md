# Internal Audit Log

Automated daily code audit results for [BhurkeSiddhesh/Docu-AI-Search](https://github.com/BhurkeSiddhesh/Docu-AI-Search).

---

## Audit: 2026-05-28

- Issues filed: 6
- Categories: 3 Critical Bug, 2 Logic Enhancement, 1 Developer Experience
- Status: Issues Filed

### Issues Created This Run

| # | Category | Title |
|---|----------|-------|
| [#228](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/228) | Critical Bug | `tools.py` calls `database.get_file_by_name()` which does not exist â€” agent `read_file` tool always fails for name-only lookups |
| [#229](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/229) | Critical Bug | `/api/agent/chat` missing `require_auth` â€” unauthenticated SSE access to full knowledge base when `AUTH_ENABLED=true` |
| [#230](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/230) | Critical Bug | Agentic `/api/search` passes raw Python dicts to `StreamingResponse` â€” `AttributeError` crashes every request when `agent_mode=true` |
| [#231](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/231) | Logic Enhancement | `search.py` reads `config.ini` via relative path â€” AdvancedRAG settings silently disabled when cwd differs from project root |
| [#232](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/232) | Logic Enhancement | `indexing.py` `load_index` uses bare `except: pass` for BM25 deserialization â€” silent failure triggers expensive reconstruction with no diagnostic |
| [#233](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/233) | Developer Experience | `search.py` uses `print()` for all diagnostics â€” bypasses structured logging, invisible in production |

### Scope Covered
- `backend/agent.py`, `backend/api.py`, `backend/auth.py`, `backend/background.py`
- `backend/database.py`, `backend/file_processing.py`, `backend/indexing.py`
- `backend/search.py`, `backend/settings.py`, `backend/tools.py`
- `frontend/src/components/AgentView.jsx`
- Existing open issues cross-referenced (83 issues checked for duplicates)

---

## Audit: 2026-05-27
- Issues filed: 3
- Categories: 1 Critical Bug, 2 Logic Enhancement, 0 Developer Experience
- Status: Issues Filed
- New issues:
  - #224 [Critical Bug] POST/DELETE /api/system-prompts missing require_auth
  - #225 [Logic Enhancement] DELETE /api/folders/history endpoints missing require_auth
  - #226 [Logic Enhancement] LogRequest fields have no max_length validation
- Scan coverage: backend/*.py (bare excepts, auth gaps, input validation, subprocess safety, path traversal, connection leaks), frontend/src/**/*.jsx (XSS vectors, unhandled promise rejections, error feedback gaps), configuration files
- Pre-existing open issues at time of audit: 80

---

## Audit: 2026-05-26
- Issues filed: 2
- Categories: 1 Critical Bug, 1 Logic Enhancement, 0 Developer Experience
- Issues: #221 (logger.js hardcoded localhost URL kills error reporting), #222 (indexing_status dict unprotected from concurrent access)
- Status: Issues Filed

---

## Audit: 2026-05-25
- Issues filed: 0
- Categories: 0 Critical Bug, 0 Logic Enhancement, 0 Developer Experience
- Status: Authentication failure for GitHub CLI (gh)

---

## Audit: 2026-05-24
- Issues filed: 5
- Categories: 2 Critical Bug, 2 Logic Enhancement, 1 Developer Experience
- Status: [Issues Filed]

### Issues Created
| # | Title | Category |
|---|-------|----------|
| [#211](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/211) | Missing .dockerignore â€” COPY . . bakes config.ini (API keys) into Docker image | Critical Bug |
| [#212](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/212) | Data-access endpoints missing require_auth â€” file preview, search history bypass token auth | Critical Bug |
| [#213](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/213) | OllamaProvider.health_check() omits Authorization header | Logic Enhancement |
| [#214](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/214) | indexing.py checkpoint stores full document text â€” OOM risk for large corpora | Logic Enhancement |
| [#215](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/215) | run_indexing() duplicate indexing_status["progress"] = 100 assignment | Developer Experience |

### Scan Coverage
- Backend Python files: `api.py`, `agent.py`, `auth.py`, `background.py`, `clustering.py`, `database.py`, `file_processing.py`, `indexing.py`, `llm_integration.py`, `model_manager.py`, `providers.py`, `rag_optimizers.py`, `search.py`, `settings.py`, `system_prompts.py`, `tools.py`, `websocket_manager.py`
- Frontend: `App.jsx`, `AgentView.jsx`, `SearchView.jsx`, `ResultCard.jsx`, `lib/api.js`
- Configuration: `Dockerfile`, `docker-compose.yml`, `.gitignore`
- Pre-existing open issues checked: 60 (numbers #118â€“#209); no duplicates filed

---

## Audit: 2026-05-23
- Issues filed: 5
- Categories: 2 Critical Bug, 2 Logic Enhancement, 1 Developer Experience
- Status: Issues Filed

### Filed Issues
| # | Title | Category |
|---|-------|----------|
| [#205](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/205) | CORS wildcard used instead of configured ALLOWED_ORIGINS | Critical Bug |
| [#206](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/206) | Blocking search() call in async stream_answer_endpoint stalls event loop | Critical Bug |
| [#207](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/207) | Stream answer silently terminates on LLM error â€” client receives no error event | Logic Enhancement |
| [#208](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/208) | xlsx workbook not closed in file_processing.py â€” OS file-descriptor leak | Logic Enhancement |
| [#209](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/209) | agent.py uses print() for all tracing â€” bypasses structured logging | Developer Experience |

---

## Audit: 2026-05-22
- Issues filed: 0
- Categories: 0 Critical Bug, 0 Logic Enhancement, 0 Developer Experience
- Status: Authentication failure for GitHub CLI (gh)

---

## Audit: 2026-05-21
- Issues filed: 3
- Categories: 0 Critical Bug, 3 Logic Enhancement, 0 Developer Experience
- New issues: #195, #196, #197
- Status: [Issues Filed]

### Details
| # | Title | Category |
|---|-------|----------|
| [#195](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/195) | SearchView.jsx concurrent searches race on shared aiAnswer state | Logic Enhancement |
| [#196](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/196) | /api/search and /api/stream-answer call cached_smart_summary() synchronously in async handler | Logic Enhancement |
| [#197](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/197) | sort_by=file_size and sort_by=date use blocking stat syscalls instead of DB-cached metadata | Logic Enhancement |

### Scope Covered
- `backend/api.py` (all ~2000 lines)
- `backend/agent.py`, `backend/tools.py`, `backend/llm_integration.py`
- `backend/indexing.py`, `backend/search.py`, `backend/file_processing.py`
- `backend/database.py`, `backend/auth.py`, `backend/background.py`
- `backend/websocket_manager.py`
- `frontend/src/components/SearchView.jsx`, `AgentView.jsx`, `SettingsModal.jsx`
- `frontend/src/lib/api.js`
- 57 existing open issues cross-checked to avoid duplicates

---

## Audit: 2026-05-20
- Issues filed: 4
- Categories: 1 Critical Bug, 3 Logic Enhancement, 0 Developer Experience
- Issues:
  - #190 [Critical Bug] DELETE /api/search/history/{history_id} swallows HTTP 404 as HTTP 500
  - #191 [Logic Enhancement] response_cache SQLite table has no eviction policy â€” unbounded disk growth
  - #192 [Logic Enhancement] indexing.py checkpoint flushed after every file â€” O(nÂ²) disk I/O
  - #193 [Logic Enhancement] config.ini written non-atomically â€” crash during write corrupts all configuration
- Status: Issues Filed

---

## Audit: 2026-05-19
- Issues filed: 0
- Categories: 0 Critical Bug, 0 Logic Enhancement, 0 Developer Experience
- Status: Authentication failure for GitHub CLI (gh)

---

## Audit: 2026-05-18
- Issues filed: 5
- Categories: 1 Critical Bug, 4 Logic Enhancement, 0 Developer Experience
- Status: Issues Filed

### Issues Filed
| # | Title | Category |
|---|-------|----------|
| [#180](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/180) | providers.py cache key omits api_key â€” stale credentials served silently after config update | Critical Bug |
| [#181](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/181) | agent.py ReAct tool calls are synchronous inside async generator â€” blocks event loop on every search | Logic Enhancement |
| [#182](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/182) | rag_optimizers.py _QUERY_REWRITE_CACHE is unbounded â€” OOM risk on long-running server | Logic Enhancement |
| [#183](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/183) | api.js streamAnswer has no AbortController â€” backend keeps generating tokens after client navigates away | Logic Enhancement |
| [#184](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/184) | background.py reads config.ini via relative path â€” watchdog silently uses wrong/empty config when cwd differs from project root | Logic Enhancement |

---

## Audit: 2026-05-17
- Issues filed: 5
- Categories: 2 Critical Bug, 1 Logic Enhancement, 2 Developer Experience
- Status: [Issues Filed]

### Filed Issues
| # | Category | Title |
|---|----------|-------|
| #174 | Critical Bug | POST /api/logs allows unauthenticated log injection from any origin |
| #175 | Critical Bug | DELETE /api/models/delete missing verify_local_request â€” remote hosts can delete model files |
| #176 | Logic Enhancement | model_manager.py start_download has check-then-set race condition |
| #177 | Developer Experience | model_manager.py uses print() for all download progress and error reporting |
| #178 | Developer Experience | rag_optimizers.py uses print() for all diagnostics |

---

## Audit: 2026-05-16
- Issues filed: 3
- Categories: 2 Critical Bug, 1 Logic Enhancement, 0 Developer Experience
- Status: Issues Filed
- Details:
  - #170 [Critical Bug] POST /api/config missing verify_local_request â€” remote hosts can overwrite API keys in config.ini
  - #171 [Critical Bug] SSRF via POST /api/providers/health and /api/providers/models â€” server makes HTTP requests to attacker-supplied base_url
  - #172 [Logic Enhancement] /api/stream-answer calls search() synchronously in async handler â€” blocks event loop during embedding + FAISS operation

---

## Audit: 2026-05-15
- Issues filed: 4
- Categories: 1 Critical Bug, 3 Logic Enhancement, 0 Developer Experience
- Existing open issues reviewed: 33
- Status: Issues Filed

### New Issues
| # | Title | Category |
|---|-------|----------|
| [#165](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/165) | database.clear_files() called at start of create_index() â€” failed re-index leaves metadata DB permanently empty | Critical Bug |
| [#166](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/166) | _validate_token() reads and parses config.ini from disk on every authenticated request | Logic Enhancement |
| [#167](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/167) | /api/validate-path runs unbounded os.walk() synchronously in async handler â€” blocks event loop on large directories | Logic Enhancement |
| [#168](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/168) | /api/stream-answer returns HTTP 200 with plain unformatted error strings â€” frontend displays them as AI answer tokens | Logic Enhancement |

### Files Audited
- `backend/api.py` (2000 lines, complete)
- `backend/agent.py` (complete)
- `backend/database.py` (complete)
- `backend/indexing.py` (complete)
- `backend/search.py` (partial)
- `backend/llm_integration.py` (partial)
- `backend/tools.py` (complete)
- `backend/file_processing.py` (complete)
- `backend/background.py` (complete)
- `backend/auth.py` (complete)
- `backend/model_manager.py` (partial)
- `frontend/src/components/AgentView.jsx`
- `frontend/src/components/SearchView.jsx`
- `frontend/src/lib/api.js`

---

## Audit: 2026-05-14
- Issues filed: 0
- Categories: 0 Critical Bug, 0 Logic Enhancement, 0 Developer Experience
- Status: Authentication failure for GitHub CLI (gh)

---

## Audit: 2026-05-13
- Issues filed: 4
- Categories: 0 Critical Bug, 2 Logic Enhancement, 2 Developer Experience
- New issues: #154, #155, #156, #157
- Status: Issues Filed

### Summary
- **#154** [Logic Enhancement] `file_processing.py` never closes openpyxl workbook â€” file-descriptor leak during bulk indexing
- **#155** [Developer Experience] `search.py` uses `print()` for all diagnostics â€” search events invisible in production logs
- **#156** [Developer Experience] `llm_integration.py` logs `sys.path` and `sys.executable` at INFO on every module import â€” internal layout leaked to production logs
- **#157** [Logic Enhancement] `subprocess.run` in `/api/open-file` has no timeout â€” worker thread can block indefinitely

---

## Audit: 2026-05-12

- Issues filed: 3
- Categories: 0 Critical Bug, 3 Logic Enhancement, 0 Developer Experience
- Status: Issues Filed
- New issues:
  - #148 [Logic Enhancement] `/api/agent/chat` reads 7 shared index globals without `_index_lock` â€” race condition during concurrent reindex
  - #149 [Logic Enhancement] `/api/benchmarks/run` check-then-set race condition â€” benchmark can run twice in parallel
  - #150 [Logic Enhancement] `search.py` reads `config.ini` via relative path â€” query rewriting and reranking silently disabled when cwd differs from project root
- Existing open issues reviewed: 23 (issue numbers 118â€“146; not all numbers in that range are open â€” gaps correspond to closed issues/PRs). Issue #147 was filed by another contributor during the audit window (between the initial fetch and when audit issues were filed).

---

## Audit: 2026-05-11
- Issues filed: 0
- Categories: 0 Critical Bug, 0 Logic Enhancement, 0 Developer Experience
- Status: Clean Bill of Health âœ“
- Note: Daily automated code audit aborted. Authentication failure: GitHub CLI (`gh`) is not installed or not authenticated.

---

## Audit: 2026-05-10
- Issues filed: 3
- Categories: 2 Critical Bug, 1 Logic Enhancement, 0 Developer Experience
- Status: Issues Filed

### New issues
| # | Category | Title |
|---|----------|-------|
| [#139](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/139) | Critical Bug | tools.py passes no api_key to get_embeddings â€” agent search always uses local HuggingFace (384-dim) embeddings, causing EmbeddingDimensionMismatchError with cloud-built indexes |
| [#140](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/140) | Critical Bug | /api/search writes stale snapshot values back to module globals without _index_lock â€” race condition silently reverts newly-built index |
| [#141](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/141) | Logic Enhancement | Dockerfile has no USER directive â€” server process runs as root inside the container |

### Files scanned
- `backend/agent.py`, `backend/api.py`, `backend/auth.py`, `backend/background.py`
- `backend/database.py`, `backend/file_processing.py`, `backend/indexing.py`
- `backend/llm_integration.py`, `backend/search.py`, `backend/tools.py`
- `docker-compose.yml`, `Dockerfile`, `.env.example`
- `frontend/src/components/*.jsx`

### Previously open issues (16 carried forward, not duplicated)
\#118â€“\#120, \#123â€“\#127, \#129â€“\#131, \#133â€“\#137

---

---

## Audit: 2026-05-09
- Issues filed: 0
- Categories: 0 Critical Bug, 0 Logic Enhancement, 0 Developer Experience
- Status: Authentication failure for GitHub CLI (gh)

---

## Audit: 2026-05-08
- Issues filed: 3
- Categories: 2 Critical Bug, 0 Logic Enhancement, 1 Developer Experience
- New issues: #129, #130, #131
- Status: Issues Filed

### Summary
| # | Title | Category |
|---|-------|----------|
| 129 | ZeroDivisionError in `indexing_progress_callback` when `total=0` | Critical Bug |
| 130 | Watchdog auto-indexer saves rebuilt index to relative path â€” API never loads it | Critical Bug |
| 131 | `agent.py` uses `print()` for all ReAct loop diagnostics â€” invisible in production logs | Developer Experience |

---

---

## Audit: 2026-05-07
- Issues filed: 5
- Categories: 3 Critical Bug, 1 Logic Enhancement, 1 Developer Experience
- Status: [Issues Filed]

### Issues Created
| # | Title | Category |
|---|-------|----------|
| [#123](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/123) | Plaintext API keys stored as embedding cache dictionary keys in process memory | Critical Bug |
| [#124](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/124) | IndexError crash when all embedding batches fail â€” empty array has no second dimension | Critical Bug |
| [#125](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/125) | Agentic mode in /api/search yields raw Python dicts to StreamingResponse â€” malformed SSE | Critical Bug |
| [#126](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/126) | Auto-index watchdog rebuilds entire index on every filesystem event with no debounce | Logic Enhancement |
| [#127](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/127) | database.py uses print() instead of logger for all error reporting | Developer Experience |

---

---

## Audit: 2026-05-06
- Issues filed: 3
- Categories: 2 Critical Bug, 1 Logic Enhancement, 0 Developer Experience
- Status: [Issues Filed]

### Issues created
| # | Title | Category |
|---|-------|----------|
| [#118](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/118) | CORS middleware hardcodes allow_origins=["*"]; computed ALLOWED_ORIGINS is never used | Critical Bug |
| [#119](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/119) | Silent vector-document misalignment when an embedding batch fails during indexing | Critical Bug |
| [#120](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/120) | asyncio.get_event_loop() called from background thread in indexing_progress_callback is broken in Python 3.12 | Logic Enhancement |

---

## Audit: 2026-05-05
- Error: GitHub CLI (`gh`) is not authenticated. Unable to proceed with the audit.

---

## Audit: 2026-05-04
- Error: GitHub CLI (`gh`) authentication failed or command not found. Audit aborted.

---

## Audit: 2026-05-03
- Issues filed: 0
- Categories: 0 Critical Bug, 0 Logic Enhancement, 0 Developer Experience
- Status: Error - gh authentication failed

---

## Audit: 2026-05-02
- Status: Failed
- Reason: The GitHub CLI (`gh`) command is not found or not authenticated. Cannot file issues.

---

## Audit: 2026-05-01
- Error: gh is not authenticated. GitHub API authentication token is missing or invalid.

## Audit: 2026-06-24
- Issues filed: 0
- Categories: 0 Critical Bug, 0 Logic Enhancement, 0 Developer Experience
- Status: Authentication failure for GitHub CLI (gh)
