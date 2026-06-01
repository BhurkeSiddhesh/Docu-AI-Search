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

---

## Fix Pass: 2026-06-01 (Run 2)

### Phase A — Pre-existing `fix/` PRs processed

| PR | Issue | Description | Files | Result |
|----|-------|-------------|-------|--------|
| #280 | #130 | Use absolute paths for background index saving (watchdog CWD mismatch) | `backend/background.py` | Auto-merged |
| #279 | #133 | Implement missing `get_file_by_name` in `database.py` (agentic tool crash) | `backend/database.py` | Auto-merged |
| #278 | #140 | Remove global write-back of index snapshots in `/api/search` (race condition) | `backend/api.py` | Auto-merged |
| #277 | #158 | Add `verify_local_request` + `asyncio.to_thread` to `/api/browse` (blocking + auth) | `backend/api.py` | Auto-merged |
| #276 | #206 | Wrap `search()` in `asyncio.wait_for` in stream endpoint (event loop blocking) | `backend/api.py` | Auto-merged |

**CodeRabbit note:** All rate-limited ("Insufficient review credits") on all five PRs — bypassed per protocol. All non-CodeRabbit CI checks (test-backend, test-frontend, security-scan, CodeQL, GitGuardian, submit-pypi, Analyze×3) passed green before merge.

### Phase B — New P1 fixes opened

| Issue | PR | Priority | Description | Files | Disposition |
|-------|----|----------|-------------|-------|-------------|
| #139 | #286 | P1 | Pass `api_key` to `get_embeddings` in `tool_search_knowledge_base` — missing key caused dimension mismatch on every agentic search | `backend/tools.py` | Awaiting human review |
| #171 | #287 | P1 | Add `verify_local_request` + localhost URL validation to `/api/providers/health` and `/api/providers/models` — SSRF via arbitrary `base_url` | `backend/api.py` | Awaiting human review |
| #165 | #288 | P1 | Defer `clear_files()`/`clear_clusters()` until after indexing succeeds; wrap in SQLite transaction for atomicity | `backend/indexing.py` | Awaiting human review |
| #125 | — | P2 | SSE disconnects in agentic mode | — | Skipped — duplicate of open PR #230 |
| #205 | — | P2 | CORS wildcard `allow_origins=["*"]` | — | Skipped — duplicate of open PR #256 |

**Gemini review responses:**
- PR #286: addressed defensive config access (`global_state.get('config')`, no fallback to openai key for unmapped providers)
- PR #287: addressed scheme validation (`parsed.scheme not in ("http","https")` check added before hostname check)
- PR #288: addressed atomicity (replaced separate clear/insert calls with single `BEGIN`/`COMMIT` transaction)
- PR #276: addressed missing `try/except` around `asyncio.wait_for` — now returns 504 on timeout, 409 on embedding dimension mismatch

**Summary:** 3 PRs opened (Phase B) · 5 auto-merged (PRs #280 #279 #278 #277 #276) · 3 awaiting human review (PRs #286, #287, #288 — all P1) · 2 skipped (existing PRs)
