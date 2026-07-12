# Internal Fix Log

Automated fix pass results from the daily issue resolver.

## Fix Pass: 2026-05-29

Issue #237 — [Critical Bug] background.py reads singular `folder` key with no fallback — PR opened — PR #238
Issue #229 — [Critical Bug] /api/agent/chat missing require_auth — unauthenticated SSE access — PR opened — PR #239
Issue #228 — [Critical Bug] tools.py calls database.get_file_by_name() which does not exist — PR opened — PR #240
Issue #221 — [Critical Bug] logger.js hardcodes http://localhost:8000/api/logs — PR opened — PR #241
Issue #129 — [Critical Bug] ZeroDivisionError in indexing_progress_callback when total=0 — PR opened — PR #242

Summary: 5 PRs opened · 0 escalated · 0 auto-closed · 0 skipped (already has PR)

---

## Fix Pass: 2026-05-30

Issue #235 — [Critical Bug] external_api_key exposed in plaintext in GET /api/config — PR opened — PR #248
Issue #230 — [Critical Bug] agent.stream_chat yields dicts to StreamingResponse — AttributeError on every agentic search — PR opened — PR #249
Issue #224 — [Critical Bug] POST/DELETE /api/system-prompts missing require_auth — PR opened — PR #250
Issue #212 — [Critical Bug] Six data-access endpoints missing require_auth — PR opened — PR #251
Issue #211 — [Critical Bug] No .dockerignore — secrets baked into Docker image — PR opened — PR #252

Summary: 5 PRs opened · 0 escalated · 0 auto-closed · 0 skipped (already has PR)

---

---

## Fix Pass: 2026-05-31

Issue #118 — [Critical Bug] CORS middleware hardcodes allow_origins=["*"]; ALLOWED_ORIGINS never used — PR opened — PR #256 (also closes #205)
Issue #205 — [Critical Bug] CORS wildcard used instead of configured ALLOWED_ORIGINS — Covered by PR #256 (duplicate of #118)
Issue #190 — [Critical Bug] DELETE /api/search/history swallows HTTP 404 as HTTP 500 — PR opened — PR #257
Issue #186 — [Critical Bug] MAX_INDICES=900 generates 1800 SQL bind-params, exceeds SQLite limit of 999 — PR opened — PR #258
Issue #180 — [Critical Bug] providers.py cache key omits api_key — stale credentials served after config update — PR opened — PR #259
Issue #255 — [Critical Bug] RAPTOR ThreadPoolExecutor swallows no exceptions — single LLM failure aborts entire indexing job — PR opened — PR #260

Summary: 5 PRs opened · 0 escalated · 0 auto-closed · 0 skipped (already has PR)

### Review activity addressed
- PR #257: Applied Gemini suggestion — moved 404/success logic outside try block for cleaner error handling
- PR #258: Applied Gemini suggestion — chunked queries instead of raising ValueError; updated 2 tests that expected the old ValueError behaviour
- PR #259: Applied Gemini suggestion — use `"api_key" in config` check instead of truthiness to allow clearing credentials
- PR #260: Applied Gemini suggestion — eliminated duplicated progress/counter logic via `cid, summary = None, None` initialisation

---

## Fix Pass: 2026-06-01

| Issue | Label | Description | PR | Disposition |
|-------|-------|-------------|----|-----------|
| #123 | Critical Bug | Plaintext API keys stored as embedding cache dictionary keys in process memory | PR #273 | Awaiting human review |
| #120 | Logic Enhancement | asyncio.get_event_loop() called from background thread is broken in Python 3.12 | PR #274 | Awaiting human review |
| #119 | Critical Bug | Silent vector-document misalignment when embedding batch fails during indexing | — | Auto-closed — already mitigated in main (abort assertions added in 2026-05-14) |
| #118 | Critical Bug | CORS middleware hardcodes allow_origins=["*"]; computed ALLOWED_ORIGINS never used | — | Skipped — PR #256 already open |

**Summary:** 2 PRs opened · 0 auto-merged · 2 awaiting human review · 0 escalated · 1 auto-closed · 1 skipped (existing PR)

### Phase B — New PRs
Issue #123 — [Critical Bug] Plaintext API keys stored as embedding cache dictionary keys in process memory — PR #282 opened — awaits human review
Issue #120 — [Logic Enhancement] asyncio.get_event_loop() thread-safety — PR #283 opened — awaits human review
Issue #118 — [Critical Bug] Restrict CORS allowed origins — PR #284 opened — awaits human review
Issue #124 — [Critical Bug] Index builder fails completely if a single chunk embedding fails — Auto-closed (already fixed)
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

---

## Fix Pass: 2026-06-08

### Phase A — Existing PRs

| PR | Branch | Label | Description | Action |
|----|--------|-------|-------------|--------|
| #329 | fix/issue-205-cors-wildcard | Critical Bug | CORS wildcard instead of configured ALLOWED_ORIGINS | Left for human review (P1 — never auto-merged) |
| #330 | fix/issue-327-embedding-retry | Logic Enhancement | Retry failed embedding batches before aborting index job | Left for human review (awaits human review disposition) |
| #331 | fix/issue-326-optimistic-rollback | Logic Enhancement | Rollback optimistic folder state on persistFolders failure | Left for human review (awaits human review disposition) |

All other open PRs skipped: branch names do not start with `fix/` or `p3-batch/`.

### Phase B — New PRs

| Issue | Priority | Description | PR | Disposition |
|-------|----------|-------------|-----|-------------|
| #338 | P2 Logic Enhancement | ModelManager.jsx load() and pollStatus() silently swallow API errors | #340 | Auto-merge pending CI |
| #335 | P2 Logic Enhancement | Bare except: in api.py tensor_split parsing catches BaseException | #341 | Auto-merge pending CI; CodeRabbit rate-limited (bypassed) |
| #337 | P3 Developer Experience | agent.py ReAct loop uses print() — invisible in production logs | #342 | Auto-merge pending CI (P3 batch); CodeRabbit rate-limited (bypassed) |
| #336 | P3 Developer Experience | database.py has no logger — all DB errors silently printed to stdout | #342 | Auto-merge pending CI (P3 batch); CodeRabbit rate-limited (bypassed) |

**Skipped issues (existing open PRs cover them):**
- All P1 Critical Bugs: covered by PR #323 or PR #333 or PR #329
- P2/P3 issues covered by PR #333: #221, #222, #226, #231, #236, #245, #246, #316, #326, #327

**CodeRabbit:** PR #340 reviewed (in progress at log time) · PRs #341 and #342 rate-limited (bypassed per protocol)

### Summary
Phase A: 0 merged · 3 left for human review · 0 CI failing
Phase B: 3 PRs opened · 0 auto-merged (CI pending) · 0 awaiting review · 0 escalated · 0 auto-closed
CodeRabbit: 2 rate-limit bypasses · 0 review blocks

---

## Fix Pass: 2026-06-09

### Phase A — Existing PR Triage

Scanned open `fix/` and `p3-batch/` PRs:

| PR | Branch | Issues | CI | CodeRabbit | Disposition | Action |
|----|--------|--------|----|------------|-------------|--------|
| #329 | fix/* | — | — | — | P1 Critical Bug — never auto-merge | Left for human review |
| #330 | fix/* | — | — | — | awaits human review | Left open |
| #331 | fix/* | — | — | — | awaits human review | Left open |
| #343 | fix/* | — | — | — | No Merge Disposition line | Skipped |
| #350 | fix/issue-347-agentview-sse-error | #347 | ✅ all green | Rate-limited → bypassed | auto-merge | **Merged ✓** (squash) |
| #351 | fix/issue-346-stream-abort-signal | #346 | ✅ all green | Rate-limited → bypassed | awaits human review | Left open |

### Phase B — New Issue Fixes

| Issue | Priority | Description | PR | Reviewer Feedback Applied | Disposition |
|-------|----------|-------------|-----|---------------------------|-------------|
| #347 | P2 | `AgentView.jsx` SSE `onmessage` catch block silently dropped malformed frames, leaving UI frozen | #350 | Gemini: `catch (err)` + `console.error`; CodeRabbit: add error event in `onerror` handler — all applied | **Auto-merged ✓** |
| #346 | P2 | `streamAnswer()` had no cancellation mechanism; stale streams updated state after new query started | #351 | Gemini: race condition in `finally`, unmount `useEffect` cleanup, `decoder.decode({stream:true})` — all applied | Awaiting human review |
| #348 | P2 | `OllamaProvider` and `OpenAICompatibleProvider` used bare `requests.get/post` with no retry on transient errors | #352 | — | Awaiting human review |
| #349 | P3 | `test_database.py` `tearDownModule`: bare `except:` swallowed `KeyboardInterrupt`/`SystemExit` | #353 | — | **Auto-merged ✓** |

### Reviewer Notes

- **PR #350**: CodeRabbit completed full review before rate-limiting — LGTM on catch block, nitpick on `onerror` (applied). Gemini `catch (err)` + `console.error` applied. All reviewer feedback incorporated before merge.
- **PR #351**: CodeRabbit rate-limited throughout. Gemini found 3 valid bugs (HIGH race condition in `finally`, MEDIUM unmount cleanup, MEDIUM `{stream: true}` decoder) — all applied. PR left open per P2/MEDIUM disposition.
- **PR #352**: `_make_retry_session()` uses `urllib3.util.retry.Retry` + `requests.adapters.HTTPAdapter` — no new pip dependency. Retries 3× with 0.5 s backoff on 502/503/504.
- **PR #353**: P3 batch (issue #349 only this run). `except OSError:` replaces bare `except:` in `tearDownModule`. All CI green, CodeRabbit rate-limited → bypassed → auto-merged.
- **CodeRabbit**: Rate-limited on PRs #350 (after initial review), #351, #352, #353. Initial review on #350 was substantive (applied); all subsequent rate-limits treated as bypass per protocol.

### Summary

Phase A: 1 merged · 5 left for human review or skipped
Phase B: 4 PRs opened · 2 auto-merged · 2 awaiting human review
CodeRabbit: 4+ rate-limit bypasses · 0 review blocks

