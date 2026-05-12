# Internal Audit Log — Docu-AI-Search

## Audit: 2026-05-12

- Issues filed: 3
- Categories: 0 Critical Bug, 3 Logic Enhancement, 0 Developer Experience
- Status: Issues Filed
- New issues:
  - #148 [Logic Enhancement] `/api/agent/chat` reads 7 shared index globals without `_index_lock` — race condition during concurrent reindex
  - #149 [Logic Enhancement] `/api/benchmarks/run` check-then-set race condition — benchmark can run twice in parallel
  - #150 [Logic Enhancement] `search.py` reads `config.ini` via relative path — query rewriting and reranking silently disabled when cwd differs from project root
- Existing open issues reviewed: 23 (issue numbers 118–146; not all numbers in that range are open — gaps correspond to closed issues/PRs). Issue #147 was filed by another contributor during the audit window (between the initial fetch and when audit issues were filed).
