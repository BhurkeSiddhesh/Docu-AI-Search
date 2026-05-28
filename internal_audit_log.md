# Internal Audit Log

Automated daily code audit results for [BhurkeSiddhesh/Docu-AI-Search](https://github.com/BhurkeSiddhesh/Docu-AI-Search).

---

## Audit: 2026-05-28

- Issues filed: 6
- Categories: 3 Critical Bug, 2 Logic Enhancement, 1 Developer Experience
- Status: Issues Filed

### Issues Created This Run

| # | Category | Title |
|---|----------|-------|
| [#228](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/228) | Critical Bug | `tools.py` calls `database.get_file_by_name()` which does not exist — agent `read_file` tool always fails for name-only lookups |
| [#229](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/229) | Critical Bug | `/api/agent/chat` missing `require_auth` — unauthenticated SSE access to full knowledge base when `AUTH_ENABLED=true` |
| [#230](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/230) | Critical Bug | Agentic `/api/search` passes raw Python dicts to `StreamingResponse` — `AttributeError` crashes every request when `agent_mode=true` |
| [#231](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/231) | Logic Enhancement | `search.py` reads `config.ini` via relative path — AdvancedRAG settings silently disabled when cwd differs from project root |
| [#232](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/232) | Logic Enhancement | `indexing.py` `load_index` uses bare `except: pass` for BM25 deserialization — silent failure triggers expensive reconstruction with no diagnostic |
| [#233](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/233) | Developer Experience | `search.py` uses `print()` for all diagnostics — bypasses structured logging, invisible in production |

### Scope Covered
- `backend/agent.py`, `backend/api.py`, `backend/auth.py`, `backend/background.py`
- `backend/database.py`, `backend/file_processing.py`, `backend/indexing.py`
- `backend/search.py`, `backend/settings.py`, `backend/tools.py`
- `frontend/src/components/AgentView.jsx`
- Existing open issues cross-referenced (83 issues checked for duplicates)
