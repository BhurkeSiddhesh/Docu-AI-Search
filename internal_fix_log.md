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
