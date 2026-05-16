# Internal Audit Log

## Audit: 2026-05-16
- Issues filed: 3
- Categories: 2 Critical Bug, 1 Logic Enhancement, 0 Developer Experience
- Status: Issues Filed
- Details:
  - #170 [Critical Bug] POST /api/config missing verify_local_request — remote hosts can overwrite API keys in config.ini
  - #171 [Critical Bug] SSRF via POST /api/providers/health and /api/providers/models — server makes HTTP requests to attacker-supplied base_url
  - #172 [Logic Enhancement] /api/stream-answer calls search() synchronously in async handler — blocks event loop during embedding + FAISS operation
