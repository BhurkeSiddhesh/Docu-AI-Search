---
name: feature-or-bugfix-with-api-frontend-test-and-doc-update
description: Workflow command scaffold for feature-or-bugfix-with-api-frontend-test-and-doc-update in Docu-AI-Search.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-or-bugfix-with-api-frontend-test-and-doc-update

Use this workflow when working on **feature-or-bugfix-with-api-frontend-test-and-doc-update** in `Docu-AI-Search`.

## Goal

Implements or fixes a feature involving backend API changes, frontend UI updates, test coverage, and documentation.

## Common Files

- `backend/api.py`
- `backend/tests/test_api.py`
- `backend/tests/test_database.py`
- `backend/tests/test_config_and_edge_cases.py`
- `frontend/src/components/SettingsModal.jsx`
- `frontend/src/components/SearchView.jsx`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Update backend/api.py to implement or fix API logic.
- Update or add backend tests (backend/tests/test_api.py, backend/tests/test_database.py, backend/tests/test_config_and_edge_cases.py).
- Update frontend components (frontend/src/components/SettingsModal.jsx, frontend/src/components/SearchView.jsx, frontend/src/components/LibraryView.jsx).
- Update or add frontend tests (frontend/src/test/SettingsModal.test.jsx).
- Update documentation (AGENTS.md).

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.