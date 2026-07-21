```markdown
# Docu-AI-Search Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you the core development patterns, coding conventions, and workflows used in the **Docu-AI-Search** repository. The project is primarily written in Python (with frontend components in JavaScript/TypeScript), and focuses on building and maintaining a document search system with both backend API and frontend UI components. The repository emphasizes clear commit conventions, structured file organization, and comprehensive testing and documentation updates as part of its workflows.

## Coding Conventions

### File Naming

- **PascalCase** is used for file names.
  - Example: `SettingsModal.jsx`, `SearchView.jsx`, `LibraryView.jsx`

### Import Style

- **Relative imports** are preferred in Python files.
  - Example:
    ```python
    from .database import get_documents
    ```

### Export Style

- **Mixed**: Both default and named exports are used in frontend (JS/TS) code.
  - Example (JSX):
    ```jsx
    export default function SettingsModal() { ... }
    export { SettingsModal }
    ```

### Commit Patterns

- **Conventional commits** are used, with prefixes like `fix`.
- Commit messages are descriptive, averaging 91 characters.
  - Example:
    ```
    fix: resolve issue with document indexing in api.py when new files are uploaded
    ```

## Workflows

### Feature or Bugfix with API, Frontend, Test, and Doc Update

**Trigger:** When you need to add or fix a feature that affects both backend and frontend, and requires updates to tests and documentation.

**Command:** `/feature-api-ui-docs`

**Step-by-step:**

1. **Update backend API logic**
   - Edit `backend/api.py` to implement or fix the required API functionality.
   - Example:
     ```python
     # backend/api.py
     def search_documents(query):
         # Updated search logic
         ...
     ```

2. **Update or add backend tests**
   - Modify or add tests in:
     - `backend/tests/test_api.py`
     - `backend/tests/test_database.py`
     - `backend/tests/test_config_and_edge_cases.py`
   - Example:
     ```python
     # backend/tests/test_api.py
     def test_search_documents():
         ...
     ```

3. **Update frontend components**
   - Edit relevant UI components:
     - `frontend/src/components/SettingsModal.jsx`
     - `frontend/src/components/SearchView.jsx`
     - `frontend/src/components/LibraryView.jsx`
   - Example:
     ```jsx
     // frontend/src/components/SearchView.jsx
     export default function SearchView() {
         // Updated UI logic
     }
     ```

4. **Update or add frontend tests**
   - Add or modify tests in:
     - `frontend/src/test/SettingsModal.test.jsx`
   - Example:
     ```jsx
     // frontend/src/test/SettingsModal.test.jsx
     import { render } from '@testing-library/react';
     import SettingsModal from '../components/SettingsModal';

     test('renders settings modal', () => {
         render(<SettingsModal />);
         // assertions
     });
     ```

5. **Update documentation**
   - Edit `AGENTS.md` to reflect any changes in features, APIs, or usage.

**Files Involved:**
- `backend/api.py`
- `backend/tests/test_api.py`
- `backend/tests/test_database.py`
- `backend/tests/test_config_and_edge_cases.py`
- `frontend/src/components/SettingsModal.jsx`
- `frontend/src/components/SearchView.jsx`
- `frontend/src/components/LibraryView.jsx`
- `frontend/src/test/SettingsModal.test.jsx`
- `AGENTS.md`

**Frequency:** ~2-3 times per month

## Testing Patterns

- **Frontend:** Uses `vitest` for testing, with test files matching the pattern `*.test.ts` or `*.test.jsx`.
  - Example:
    ```jsx
    // frontend/src/test/SettingsModal.test.jsx
    import { render } from '@testing-library/react';
    import SettingsModal from '../components/SettingsModal';

    test('opens modal on click', () => {
        // test logic
    });
    ```
- **Backend:** Python tests are organized by feature and edge cases, located in `backend/tests/`.

## Commands

| Command              | Purpose                                                                                 |
|----------------------|-----------------------------------------------------------------------------------------|
| /feature-api-ui-docs | Implements or fixes a feature involving backend API, frontend UI, tests, and docs update|
```
