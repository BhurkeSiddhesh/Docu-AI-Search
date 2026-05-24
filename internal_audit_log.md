# Internal Audit Log

## Audit: 2026-05-24
- Issues filed: 5
- Categories: 2 Critical Bug, 2 Logic Enhancement, 1 Developer Experience
- Status: [Issues Filed]

### Issues Created
| # | Title | Category |
|---|-------|----------|
| [#211](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/211) | Missing .dockerignore — COPY . . bakes config.ini (API keys) into Docker image | Critical Bug |
| [#212](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/212) | Data-access endpoints missing require_auth — file preview, search history bypass token auth | Critical Bug |
| [#213](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/213) | OllamaProvider.health_check() omits Authorization header | Logic Enhancement |
| [#214](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/214) | indexing.py checkpoint stores full document text — OOM risk for large corpora | Logic Enhancement |
| [#215](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/215) | run_indexing() duplicate indexing_status["progress"] = 100 assignment | Developer Experience |

### Scan Coverage
- Backend Python files: `api.py`, `agent.py`, `auth.py`, `background.py`, `clustering.py`, `database.py`, `file_processing.py`, `indexing.py`, `llm_integration.py`, `model_manager.py`, `providers.py`, `rag_optimizers.py`, `search.py`, `settings.py`, `system_prompts.py`, `tools.py`, `websocket_manager.py`
- Frontend: `App.jsx`, `AgentView.jsx`, `SearchView.jsx`, `ResultCard.jsx`, `lib/api.js`
- Configuration: `Dockerfile`, `docker-compose.yml`, `.gitignore`
- Pre-existing open issues checked: 60 (numbers #118–#209); no duplicates filed
