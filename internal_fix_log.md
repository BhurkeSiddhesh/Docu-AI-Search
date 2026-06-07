# Internal Fix Log

Automated fix pass results from the daily issue resolver.

## Fix Pass: 2026-06-01

| Issue | Label | Description | PR | Disposition |
|-------|-------|-------------|----|-----------|
| #123 | Critical Bug | Plaintext API keys stored as embedding cache dictionary keys in process memory | PR #273 | Awaiting human review |
| #120 | Logic Enhancement | asyncio.get_event_loop() called from background thread is broken in Python 3.12 | PR #274 | Awaiting human review |
| #119 | Critical Bug | Silent vector-document misalignment when embedding batch fails during indexing | — | Auto-closed — already mitigated in main (abort assertions added in 2026-05-14) |
| #118 | Critical Bug | CORS middleware hardcodes allow_origins=["*"]; computed ALLOWED_ORIGINS never used | — | Skipped — PR #256 already open |

**Summary:** 2 PRs opened · 0 auto-merged · 2 awaiting human review · 0 escalated · 1 auto-closed · 1 skipped (existing PR)

---

## Fix Pass: 2026-06-05

### Phase A — PR Triage

Scanned 29 open `fix/` and `p3-batch/` PRs. One PR met auto-merge criteria:

| PR | Branch | Issues | Action |
|----|--------|--------|--------|
| #302 | fix/issue-299-300-dx-fixes | #299, #300 | **Merged** (squash) after branch update — all CI checks passed |

All remaining open fix-branch PRs carry P1 disposition (no auto-merge ever).

### Phase B — Issue Fixes

| Issue | Priority | Description | PR | Reviewer Feedback Applied | Disposition |
|-------|----------|-------------|-----|---------------------------|-------------|
| #306 | P3 | `indexing.py` `load_index`: bare `except: pass` swallowed MemoryError/KeyboardInterrupt/SystemExit | #309 | Gemini: broadened to `except Exception as e` + warning log | Auto-merge pending CI |
| #307 | P3 | `api.py` `run_indexing`: completion status set outside `_index_lock`; WebSocket broadcast bypassed | #310 | Gemini: call `indexing_progress_callback(100, 100, "Complete")` inside lock before setting `running=False` | Auto-merge pending CI |
| #308 | P3 | `Dockerfile`: runs as root; `chown -R` after `COPY` bloats image layers; port binds to 0.0.0.0 | #311 | Gemini: create `appuser` before COPY, use `COPY --chown=appuser:appuser`, `docker-compose.yml` binds to 127.0.0.1 | Auto-merge pending CI |
| #125 | P1 Critical Bug | `api.py` agentic search: raw dict generator passed to `StreamingResponse` — TypeError on every agentic query | #312 | Gemini: "No review comments" — CodeQL finding at line 1028 is pre-existing, noted on PR | **Awaiting human review** (P1 — never auto-merged) |

### Reviewer Notes

- **PR #309** (BM25 except): Initial fix used `except (OSError, pickle.UnpicklingError, EOFError, ValueError)` — Gemini reviewer correctly identified it missed `AttributeError`/`ImportError`. Broadened to `except Exception as e` with typed warning log.
- **PR #310** (indexing lock): Initial fix moved status updates inside `_index_lock` but bypassed WebSocket broadcast (direct dict mutation). Gemini reviewer caught this. Fixed by calling `indexing_progress_callback(100, 100, "Complete")` inside the lock, which fires `ws_manager.broadcast()` via `asyncio.run_coroutine_threadsafe`.
- **PR #311** (Dockerfile): Initial fix placed `adduser`/`chown -R` after `COPY . .`, causing layer bloat and cache invalidation on every source change. Gemini reviewer caught this. Restructured: create user → install deps → `COPY --chown=appuser:appuser . .` → `USER appuser`.
- **PR #309 CI**: First run failed with `test_stream_answer_rerun_search` ERROR (HuggingFace HTTP call not mocked — pre-existing flaky test, unrelated to BM25 fix). Branch updated; re-run in progress.
- **CodeRabbit**: Rate-limited on all four PRs throughout the run — treated as neutral per protocol.

**Summary:** 1 PR merged (Phase A) · 3 PRs opened auto-merge-pending-CI · 1 PR opened awaiting human review (P1) · 0 escalated · 0 skipped

---

## Fix Pass: 2026-06-07

### Phase A — All 30 Open Issues (batch fix)

| Issue | Label | File(s) | Fix |
|-------|-------|---------|-----|
| #221 | Critical Bug | `frontend/src/lib/logger.js` | Replace hardcoded `http://localhost:8000/api/logs` with relative `/api/logs` |
| #222 | Logic Enhancement | `backend/api.py` | Protect `indexing_status` reads/writes under `_index_lock` |
| #224 | Critical Bug | `backend/api.py` | Add `require_auth` to `POST/DELETE /api/system-prompts` |
| #225 | Logic Enhancement | `backend/api.py` | Add `require_auth` to `DELETE /api/folders/history` endpoints |
| #226 | Logic Enhancement | `backend/api.py` | Add Pydantic `Field` max_length constraints to `LogRequest` + level whitelist |
| #228 | Critical Bug | `backend/database.py` | Add `get_file_by_name()` function |
| #229 | Critical Bug | `backend/api.py` | Add `require_auth` to `GET /api/agent/chat` |
| #230 | Critical Bug | `backend/api.py` | Wrap agentic search dicts in SSE `event_generator()` |
| #231 | Logic Enhancement | `backend/search.py` | Use absolute path for `config.ini` read |
| #232 | Logic Enhancement | `backend/indexing.py` | Already fixed in prior pass (verified) |
| #233 | Developer Experience | `backend/search.py` | Replace all `print()` calls with `logger.*` |
| #235 | Critical Bug | `backend/api.py`, `frontend/src/components/SettingsModal.jsx` | Mask `external_api_key` as boolean `external_api_key_set` |
| #236 | Developer Experience | `.github/workflows/ci.yml` | Remove `\|\| true` from security scan steps |
| #237 | Critical Bug | `backend/background.py` | Read `folders` key with `folder` legacy fallback |
| #245 | Logic Enhancement | `backend/model_manager.py` | Add `threading.Lock` around all `download_status` mutations |
| #246 | Logic Enhancement | `backend/api.py` | Add 300s idle timeout to `/ws/progress` WebSocket |
| #247 | Developer Experience | `backend/clustering.py`, `backend/background.py` | Replace `print()` with `logger.*` |
| #255 | Critical Bug | `backend/indexing.py` | Add `try/except` around RAPTOR cluster futures |
| #263 | Critical Bug | `frontend/src/components/BenchmarkView.jsx` | Fix stale closure via `prevRunningRef` |
| #264 | Critical Bug | `frontend/src/components/ModelManager.jsx` | Fix stale closure via `prevDownloadingRef` |
| #265 | Logic Enhancement | `frontend/src/App.jsx` | Wrap each view in individual `<ErrorBoundary>` |
| #266 | Developer Experience | `frontend/src/components/AgentView.jsx` | Log SSE parse errors via `console.warn` |
| #298 | Critical Bug | `frontend/src/lib/api.js`, `frontend/src/App.jsx` | Add auth interceptor + token bootstrap on startup |
| #304 | Critical Bug | `backend/websocket_manager.py` | Add `asyncio.Lock` + snapshot list before broadcast iteration |
| #305 | Critical Bug | `backend/agent.py` | Wrap LLM call in `asyncio.wait_for(timeout=60)` |
| #316 | Logic Enhancement | `backend/llm_integration.py` | Add retry with exponential backoff to `invoke()` and `stream()` |
| #318 | Developer Experience | `backend/llm_integration.py` | Read `[LLM] model` from `config.ini`; update default model IDs |
| #326 | Logic Enhancement | `frontend/src/components/SettingsModal.jsx` | Rollback optimistic folder state on `persistFolders` failure |
| #327 | Logic Enhancement | `backend/indexing.py` | Add `_embed_with_retry` with 3-attempt exponential backoff |
| #219 | Developer Experience | `backend/api.py` | Replace `detail=str(e)` with generic message in 5 endpoints |
| #247 (agent.py) | Developer Experience | `backend/agent.py` | Remove `print()` calls |

**Summary:** 30 issues fixed in a single batch commit on branch `claude/open-issues-resolution-5vEA6`
