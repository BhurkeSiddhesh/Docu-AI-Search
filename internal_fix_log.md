# Internal Fix Log

## Fix Pass: 2026-06-01
Issue #263 — [Critical Bug] BenchmarkView.jsx pollStatus stale closure — PR opened — PR #267
Issue #264 — [Critical Bug] ModelManager.jsx pollStatus stale closure — PR opened — PR #268
Issue #170 — [Critical Bug] POST /api/config missing verify_local_request — PR opened — PR #269
Issue #175 — [Critical Bug] DELETE /api/models/delete missing verify_local_request — PR opened — PR #270
Issue #174 — [Critical Bug] POST /api/logs allows unauthenticated log injection — PR opened — PR #271

Summary: 5 PRs opened · 0 escalated · 0 auto-closed · 0 skipped (already has PR)

### Notes
- PR #267 and #268 received valid review feedback (stale ref not seeded on start); fixup commits pushed addressing render-body sync pattern.
- PR #270 received CORS-bypass concern from Gemini; acknowledged in review reply — CORS wildcard is separately tracked as issues #118/#205 with open PR #256.
- PR #271 received valid feedback on sanitizer regex truncating stack traces; fixup commit pushed to escape newlines instead.
- CodeQL check on PR #270 failed in 2s (pre-existing infrastructure issue); all actual Analyze checks passed.
