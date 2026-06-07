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

### Phase A — Existing PRs

| PR | Label | Description | Disposition |
|----|-------|-------------|-------------|
| #325 | — | chore: daily audit log (branch: `audit-log-…`) | Skipped — branch not fix/ or p3-batch/ |
| #324 | — | dependabot npm_and_yarn bump (branch: `dependabot/…`) | Skipped — branch not fix/ or p3-batch/ |
| #323 | — | Consolidate 26 pending fix PRs into main (branch: `claude/pr-review-merge-CarEO`) | Skipped — branch not fix/ or p3-batch/ |

### Phase B — New PRs

| Issue | Label | Description | PR | Disposition |
|-------|-------|-------------|----|----|
| #205 | Critical Bug | CORS wildcard used instead of configured ALLOWED_ORIGINS | PR #329 | Opened — awaits human review (P1) |
| #327 | Logic Enhancement | Single failed embedding batch aborts entire re-index — no per-batch retry | PR #330 | Opened — awaits human review (P2 MEDIUM) |
| #326 | Logic Enhancement | Optimistic folder add/remove not rolled back on persistence failure | PR #331 | Opened — awaits human review (P2 MEDIUM) |
| #328 | Developer Experience | ErrorBoundary crashes only logged to console — not forwarded via logger.js | PR #332 | **Merged** — auto-merged after CI passed (P3 HIGH) |

#### Review Activity Applied

| PR | Bot | Finding | Action |
|----|-----|---------|--------|
| #329 | Gemini | Strip trailing slashes from ALLOWED_ORIGINS for strict CORSMiddleware matching | Applied |
| #329 | Gemini | Restrict allow_methods/allow_headers from wildcard to explicit whitelist | Applied |
| #329 | CodeRabbit | Full review completed — 5/5 pre-merge checks passed | No action needed |
| #330 | Gemini | Skip retries on non-transient errors; guard None model; log warnings | Applied |
| #330 | CodeRabbit | Rate-limited — bypassed | — |
| #331 | Gemini | Restore pathInput on addFolder failure | Applied |
| #331 | CodeRabbit | Rate-limited — bypassed | — |
| #332 | Gemini | Safe error property access in componentDidCatch (high priority) | Applied |
| #332 | CodeRabbit | Rate-limited — bypassed | — |

### Summary
Phase A: 0 merged · 3 skipped (wrong branch prefix)
Phase B: 4 PRs opened · 1 auto-merged · 3 awaiting human review · 0 escalated · 0 auto-closed
CodeRabbit: 4 rate-limit bypasses · 1 full review (all checks passed, no blocking issues)
