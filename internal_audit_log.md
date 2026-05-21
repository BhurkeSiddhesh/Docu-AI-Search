# Internal Audit Log

Automated daily code audit results for the Docu-AI-Search repository.

---

## Audit: 2026-05-21
- Issues filed: 3
- Categories: 0 Critical Bug, 3 Logic Enhancement, 0 Developer Experience
- New issues: #195, #196, #197
- Status: [Issues Filed]

### Details
| # | Title | Category |
|---|-------|----------|
| [#195](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/195) | SearchView.jsx concurrent searches race on shared aiAnswer state | Logic Enhancement |
| [#196](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/196) | /api/search and /api/stream-answer call cached_smart_summary() synchronously in async handler | Logic Enhancement |
| [#197](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/197) | sort_by=file_size and sort_by=date use blocking stat syscalls instead of DB-cached metadata | Logic Enhancement |

### Scope Covered
- `backend/api.py` (all ~2000 lines)
- `backend/agent.py`, `backend/tools.py`, `backend/llm_integration.py`
- `backend/indexing.py`, `backend/search.py`, `backend/file_processing.py`
- `backend/database.py`, `backend/auth.py`, `backend/background.py`
- `backend/websocket_manager.py`
- `frontend/src/components/SearchView.jsx`, `AgentView.jsx`, `SettingsModal.jsx`
- `frontend/src/lib/api.js`
- 57 existing open issues cross-checked to avoid duplicates
