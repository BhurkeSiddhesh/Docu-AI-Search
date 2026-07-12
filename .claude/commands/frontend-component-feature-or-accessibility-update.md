---
name: frontend-component-feature-or-accessibility-update
description: Workflow command scaffold for frontend-component-feature-or-accessibility-update in Docu-AI-Search.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /frontend-component-feature-or-accessibility-update

Use this workflow when working on **frontend-component-feature-or-accessibility-update** in `Docu-AI-Search`.

## Goal

Implements new features or accessibility improvements in frontend React components, often with corresponding test updates.

## Common Files

- `frontend/src/components/*.jsx`
- `frontend/src/App.jsx`
- `frontend/src/test/*.test.jsx`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit one or more frontend/src/components/*.jsx files to implement the feature or fix.
- Update frontend/src/App.jsx if global state or routing is affected.
- Update or add tests in frontend/src/test/*.test.jsx to cover the new/changed behavior.
- If accessibility is improved, add ARIA attributes or roles as needed.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.