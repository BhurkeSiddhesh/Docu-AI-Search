# Internal Fix Log

Automated fix pass results from the daily issue resolver.

## Fix Pass: 2026-06-01

| Issue | Label | Description | PR | Disposition |
|-------|-------|-------------|----|----|
| #123 | Critical Bug | Plaintext API keys stored as embedding cache dictionary keys in process memory | PR #273 | Awaiting human review |
| #120 | Logic Enhancement | asyncio.get_event_loop() called from background thread is broken in Python 3.12 | PR #274 | Awaiting human review |
| #119 | Critical Bug | Silent vector-document misalignment when embedding batch fails during indexing | — | Auto-closed — already mitigated in main (abort assertions added in 2026-05-14) |
| #118 | Critical Bug | CORS middleware hardcodes allow_origins=["*"]; computed ALLOWED_ORIGINS never used | — | Skipped — PR #256 already open |

**Summary:** 2 PRs opened · 0 auto-merged · 2 awaiting human review · 0 escalated · 1 auto-closed · 1 skipped (existing PR)

## Fix Pass: 2026-06-03

### Phase A — PR Triage

| PR | Branch | Merge Disposition | Result |
|----|--------|-------------------|--------|
| #273 | fix/issue-123-… | Awaiting human review | Skipped — no auto-merge disposition |
| #274 | fix/issue-120-… | Awaiting human review | Skipped — no auto-merge disposition |

No PRs were auto-merged in Phase A.

### Phase B — Issue Fixes

| Issue | Priority | Label | Description | PR | Disposition |
|-------|----------|-------|-------------|----|-------------|
| #291 | P2 | Logic Bug | `cleanup_test_data()` search-history deletion used broad `LIKE '%test%'` wildcards that could wipe real user queries | PR #294 | Auto-merged ✅ |
| #292 | P2 | Logic Enhancement | `SystemPromptRequest` had no field-length limits — could exhaust SQLite storage | PR #295 | Auto-merged ✅ |
| #293 | P3 | DX / Cleanup | Default RAG system prompt contained hardcoded personal name (`'Siddhesh'`) — development testing artefact | PR #296 | Auto-merged ✅ (pending at log time) |

**Summary:** 3 PRs opened · 3 auto-merged · 0 awaiting human review · 0 escalated · 0 auto-closed · 2 skipped (Phase A: #273 and #274 had no auto-merge disposition)
