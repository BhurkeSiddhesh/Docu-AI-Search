# Internal Fix Log

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

## Fix Pass: 2026-05-30

Issue #235 — [Critical Bug] external_api_key exposed in plaintext in GET /api/config — PR opened — PR #248
Issue #230 — [Critical Bug] agent.stream_chat yields dicts to StreamingResponse — AttributeError on every agentic search — PR opened — PR #249
Issue #224 — [Critical Bug] POST/DELETE /api/system-prompts missing require_auth — PR opened — PR #250
Issue #212 — [Critical Bug] Six data-access endpoints missing require_auth — PR opened — PR #251
Issue #211 — [Critical Bug] No .dockerignore — secrets baked into Docker image — PR opened — PR #252

Summary: 5 PRs opened · 0 escalated · 0 auto-closed · 0 skipped (already has PR)

## Fix Pass: 2026-05-29

Issue #237 — [Critical Bug] background.py reads 'folder' key but config writes 'folders' — PR opened — PR #238
Issue #229 — [Critical Bug] /api/agent/chat missing require_auth — PR opened — PR #239
Issue #228 — [Critical Bug] tools.py calls database.get_file_by_name() which does not exist — PR opened — PR #240
Issue #221 — [Critical Bug] logger.js uses absolute URL, bypassing Vite proxy — PR opened — PR #241
Issue #129 — [Critical Bug] ZeroDivisionError in indexing_progress_callback when total=0 — PR opened — PR #242

Summary: 5 PRs opened · 0 escalated · 0 auto-closed · 0 skipped (already has PR)
