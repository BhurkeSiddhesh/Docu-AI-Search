# Internal Fix Log

Automated fix pass results from the daily issue resolver.

## Fix Pass: 2026-06-01

| Issue | Label | Description | PR | Disposition |
|-------|-------|-------------|----|----||
| #123 | Critical Bug | Plaintext API keys stored as embedding cache dictionary keys in process memory | PR #273 | Awaiting human review |
| #120 | Logic Enhancement | asyncio.get_event_loop() called from background thread is broken in Python 3.12 | PR #274 | Awaiting human review |
| #119 | Critical Bug | Silent vector-document misalignment when embedding batch fails during indexing | — | Auto-closed — already mitigated in main (abort assertions added in 2026-05-14) |
| #118 | Critical Bug | CORS middleware hardcodes allow_origins=["*"]; computed ALLOWED_ORIGINS never used | — | Skipped — PR #256 already open |

**Summary:** 2 PRs opened · 0 auto-merged · 2 awaiting human review · 0 escalated · 1 auto-closed · 1 skipped (existing PR)
## Fix Pass: 2026-06-05

| Issue | Label | Description | PR | Disposition |
|-------|-------|-------------|----|----||
| #299 | Developer Experience | Stream error catch block only logged to console, never showed user-facing toast | PR #302 | Auto-merged |
| #300 | Developer Experience | Bare `except:` swallows `KeyboardInterrupt`; unbounded `while True:` in `trigger_indexing()` has no timeout | PR #302 | Auto-merged |

**Summary:** 1 batch PR opened · 1 auto-merged · 0 awaiting human review · 2 issues closed · Phase A: 1 PR merged (#294), 5 PRs flagged for CI failures
