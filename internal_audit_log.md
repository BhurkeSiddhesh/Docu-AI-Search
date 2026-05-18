# Internal Audit Log

Automated daily code-quality audit for the Docu-AI-Search repository.
Each entry lists new GitHub issues filed and their categories.

---

## Audit: 2026-05-18
- Issues filed: 5
- Categories: 1 Critical Bug, 4 Logic Enhancement, 0 Developer Experience
- Status: Issues Filed

### Issues Filed
| # | Title | Category |
|---|-------|----------|
| [#180](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/180) | providers.py cache key omits api_key — stale credentials served silently after config update | Critical Bug |
| [#181](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/181) | agent.py ReAct tool calls are synchronous inside async generator — blocks event loop on every search | Logic Enhancement |
| [#182](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/182) | rag_optimizers.py _QUERY_REWRITE_CACHE is unbounded — OOM risk on long-running server | Logic Enhancement |
| [#183](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/183) | api.js streamAnswer has no AbortController — backend keeps generating tokens after client navigates away | Logic Enhancement |
| [#184](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/184) | background.py reads config.ini via relative path — watchdog silently uses wrong/empty config when cwd differs from project root | Logic Enhancement |
