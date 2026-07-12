```markdown
# Docu-AI-Search Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill provides a comprehensive guide to the development patterns, coding conventions, and common workflows in the **Docu-AI-Search** repository. The project is primarily Python-based (backend), with a React frontend, and features robust API development, accessibility-focused UI, and thorough testing. It uses conventional commits and emphasizes clear, maintainable code and collaborative workflows.

## Coding Conventions

### File Naming
- **Python files:** Use `snake_case` (e.g., `api.py`, `database.py`)
- **Frontend components:** Use `PascalCase.jsx` or `snake_case.js` as appropriate

### Import Style
- **Python:** Prefer relative imports  
  ```python
  from .database import get_connection
  ```
- **Frontend (JSX):** Standard ES6 imports  
  ```javascript
  import MyComponent from './MyComponent';
  ```

### Export Style
- **Python:** Mixed (functions, classes, or variables exported as needed)
- **Frontend:** Named or default exports  
  ```javascript
  export default MyComponent;
  ```

### Commit Messages
- **Conventional commit prefixes:** `fix`, `feat`, `merge`, `chore`, `refactor`
- **Example:**  
  ```
  feat(api): add pagination to document search endpoint
  ```

## Workflows

### Backend API Endpoint Update
**Trigger:** When you want to add, update, or secure a backend API endpoint  
**Command:** `/update-api-endpoint`

1. Edit `backend/api.py` to implement or modify endpoint logic.
2. Update related backend modules (`database.py`, `indexing.py`, `auth.py`) if necessary.
3. Update `backend/tests/test_api.py` to cover new or changed behavior.
4. If configs or environment variables are affected, update `.env.example` and/or `requirements.txt`.

**Example:**
```python
# backend/api.py
from .database import get_documents

@app.route('/api/search')
def search():
    # ...implement pagination, filters, etc.
```

### Frontend Component Feature or Accessibility Update
**Trigger:** When adding a new frontend feature, improving accessibility, or fixing UI bugs  
**Command:** `/update-frontend-component`

1. Edit one or more `frontend/src/components/*.jsx` files.
2. Update `frontend/src/App.jsx` if global state or routing is affected.
3. Update or add tests in `frontend/src/test/*.test.jsx`.
4. Add ARIA attributes or roles for accessibility improvements.

**Example:**
```jsx
// frontend/src/components/SearchBar.jsx
<input aria-label="Search documents" ... />
```

### Frontend Test Stabilization or Fix
**Trigger:** When fixing flaky frontend tests or adapting tests to component changes  
**Command:** `/fix-frontend-test`

1. Edit `frontend/src/test/*.test.jsx` to fix timing, selectors, or assertions.
2. Optionally update `frontend/src/components/*.jsx` if test failures reveal accessibility or DOM issues.
3. Rerun tests to confirm stability.

**Example:**
```jsx
// frontend/src/test/SearchBar.test.jsx
test('calls onSearch when Enter is pressed', async () => {
  // ...test implementation
});
```

### Backend Exception or Error Handler Update
**Trigger:** When improving or fixing backend error handling  
**Command:** `/update-backend-error-handling`

1. Edit `backend/api.py` to update exception handler logic.
2. Update `backend/tests/test_api.py` to cover new error cases.
3. If needed, update middleware usage or configuration.

**Example:**
```python
# backend/api.py
@app.errorhandler(ValidationError)
def handle_validation_error(e):
    return jsonify({'error': str(e)}), 400
```

### Merge Conflict Resolution
**Trigger:** When resolving merge conflicts after concurrent changes  
**Command:** `/resolve-merge-conflict`

1. Manually resolve conflicts in files listed by git.
2. Integrate both branches' changes in affected files.
3. Commit the merge with a message referencing the branches/issues.

**Example:**
```bash
# After resolving conflicts
git add .
git commit -m "merge(main, feature/api-pagination): resolve conflict in api.py"
```

## Testing Patterns

- **Frontend:** Uses `vitest` for testing React components.
- **Test file pattern:** `*.test.js` or `*.test.jsx`
- **Typical test structure:**
  ```javascript
  // frontend/src/test/SearchBar.test.jsx
  import { render, screen } from '@testing-library/react';
  import SearchBar from '../components/SearchBar';

  test('renders search input', () => {
    render(<SearchBar />);
    expect(screen.getByLabelText(/search/i)).toBeInTheDocument();
  });
  ```

- **Backend:** Python tests are located in `backend/tests/`, e.g., `test_api.py`.

## Commands

| Command                       | Purpose                                                        |
|-------------------------------|----------------------------------------------------------------|
| /update-api-endpoint          | Add, update, or secure a backend API endpoint                  |
| /update-frontend-component    | Add or update a frontend feature or accessibility improvement  |
| /fix-frontend-test            | Fix or stabilize frontend tests                                |
| /update-backend-error-handling| Update backend exception or error handling                     |
| /resolve-merge-conflict       | Resolve merge conflicts between branches                       |
```