---
name: backend-api-endpoint-update
description: Workflow command scaffold for backend-api-endpoint-update in Docu-AI-Search.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /backend-api-endpoint-update

Use this workflow when working on **backend-api-endpoint-update** in `Docu-AI-Search`.

## Goal

Implements or modifies backend API endpoints, often including input validation, pagination, filters, authentication, or error handling.

## Common Files

- `backend/api.py`
- `backend/database.py`
- `backend/indexing.py`
- `backend/auth.py`
- `backend/tests/test_api.py`
- `.env.example`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit backend/api.py to implement or modify endpoint logic.
- Optionally update related backend modules (e.g., database.py, indexing.py, auth.py) if endpoint depends on them.
- If response shape or logic changes, update backend/tests/test_api.py to cover new/changed behavior.
- If config or environment variables are involved, update .env.example and/or requirements.txt.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.