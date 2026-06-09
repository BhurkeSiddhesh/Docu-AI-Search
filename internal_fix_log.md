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
