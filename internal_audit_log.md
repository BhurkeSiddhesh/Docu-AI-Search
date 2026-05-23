# Internal Audit Log

This file records the output of each daily automated code audit run.

---

## Audit: 2026-05-23
- Issues filed: 5
- Categories: 2 Critical Bug, 2 Logic Enhancement, 1 Developer Experience
- Status: Issues Filed

### Filed Issues
| # | Title | Category |
|---|-------|----------|
| [#205](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/205) | CORS wildcard used instead of configured ALLOWED_ORIGINS | Critical Bug |
| [#206](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/206) | Blocking search() call in async stream_answer_endpoint stalls event loop | Critical Bug |
| [#207](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/207) | Stream answer silently terminates on LLM error — client receives no error event | Logic Enhancement |
| [#208](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/208) | xlsx workbook not closed in file_processing.py — OS file-descriptor leak | Logic Enhancement |
| [#209](https://github.com/BhurkeSiddhesh/Docu-AI-Search/issues/209) | agent.py uses print() for all tracing — bypasses structured logging | Developer Experience |
