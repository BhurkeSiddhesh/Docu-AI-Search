# Code Review Protocol: Docu-AI-Search

## Standards
- **Performance:** No N+1 queries. All DB lookups in loops must be batched.
- **Security:** Strict log redaction for PII/Queries. No insecure deserialization (no pickle).
- **Stability:** Preservation of order in search results (`dict.fromkeys` over `set`).
- **Error Handling:** Graceful fallbacks for limit-breakers (e.g., `MAX_INDICES`).

## Workflow
1.  **Analyze:** Run `gh pr diff` and identify architectural risks.
2.  **Verify:** Run existing test suite (`unittest` or `pytest`).
3.  **Remediate:** Apply fixes for unhandled exceptions or edge cases.
4.  **Ship:** Merge PR and delete the branch only if all tests pass.

---
*Created: 2026-02-22*