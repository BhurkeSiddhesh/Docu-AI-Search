# Docu AI Search - Copilot Instructions

This file provides context for GitHub Copilot when working with this codebase.

## Project Overview

Docu AI Search is an intelligent, semantic search engine for local documents powered by AI embeddings and large language models. It searches across PDFs, Word documents, Excel spreadsheets, PowerPoint presentations, and text files using natural language queries.

## Tech Stack

### Backend (Python 3.8+)
- **FastAPI** - REST API with async support (entry point: `backend/api.py`)
- **FAISS** - Vector similarity search
- **LangChain** - LLM integration framework
- **LlamaCpp** - Local GGUF model inference
- **SQLite** - Metadata storage (`data/metadata.db`)
- **Testing** - Python's `unittest` framework (NOT pytest)

### Frontend (Node.js 16+)
- **React 19** + **Vite** - UI framework with HMR
- **TailwindCSS** - Utility-first styling
- **Framer Motion** - Animations
- **Vitest** - Testing (NOT Jest!)
- **Axios** - HTTP client

## Project Structure

```
backend/                # All Python source code
├── api.py              # FastAPI endpoints (main entry)
├── database.py         # SQLite CRUD operations
├── indexing.py         # FAISS index creation
├── search.py           # Semantic search logic
├── file_processing.py  # Text extraction (PDF, DOCX, XLSX, PPTX, TXT)
├── llm_integration.py  # LLM provider abstraction
├── model_manager.py    # GGUF model downloads
└── tests/              # unittest tests

frontend/               # React application
├── src/
│   ├── App.jsx         # Main component (global state)
│   ├── components/     # UI components
│   └── test/           # Vitest tests
└── package.json

scripts/                # Utility scripts
├── start_all.js        # Unified startup script
├── run_tests.py        # Test runner
└── benchmark_models.py # Model benchmarking

data/                   # Generated/runtime files
├── index.faiss         # Vector embeddings
├── metadata.db         # SQLite database
└── *.pkl, *.json       # Serialized data

models/                 # Downloaded GGUF models
```

## File Placement Rules

⚠️ **CRITICAL**: Follow these rules strictly:

1. **Backend Code**: ALL Python source files go in `backend/` or `backend/tests/`
   - NO `.py` files in project root (except config scripts)
2. **Scripts**: Build, maintenance scripts go in `scripts/`
3. **Data**: Generated files (`.db`, `.faiss`, `.log`, `.json`) go in `data/`
4. **Tests**:
   - Backend: `backend/tests/`
   - Frontend: `frontend/src/test/`

## Code Patterns

### Database Operations (`database.py`)
```python
def db_operation(params):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SQL", (params,))
    result = cursor.fetchone()  # or fetchall()
    conn.commit()  # For INSERT/UPDATE/DELETE
    conn.close()
    return result
```

### LLM Provider Pattern (`llm_integration.py`)
```python
# Get embeddings - works with any provider
embeddings = get_embeddings(provider='openai', api_key='sk-...')
embeddings = get_embeddings(provider='local')  # Uses HuggingFace

# Get LLM client for generation
client = get_llm_client(provider='gemini', api_key='...')
response = client.invoke("Your prompt")

# Caches: _embeddings_cache, _llm_cache (avoid reloading)
```

### FastAPI Endpoints (`api.py`)
```python
@app.post("/api/endpoint")
def endpoint(request: RequestModel, background_tasks: BackgroundTasks):
    # For long operations, use background_tasks.add_task(func, args)
    return {"status": "success", "data": result}
```

### React Components
```jsx
export default function Component({ prop, onAction }) {
    const [state, setState] = useState(null)
    const [loading, setLoading] = useState(false)
    
    useEffect(() => {
        fetchData()
    }, [])
    
    return <div className="bg-gray-900 p-4 rounded-lg">...</div>
}
```

### Frontend API Calls
```jsx
const API = 'http://localhost:8000'
try {
    const response = await axios.post(`${API}/api/search`, { query })
    setResults(response.data.results)
} catch (error) {
    console.error('Error:', error)
}
```

## Testing Practices

### Backend Tests (unittest)
```python
class TestFeature(unittest.TestCase):
    def test_behavior(self):
        result = function()
        self.assertEqual(result, expected)
    
    # ALWAYS mock LLM calls to avoid API costs
    @patch('backend.llm_integration.get_llm_client')
    def test_with_llm(self, mock_client):
        mock_client.return_value.invoke.return_value = MagicMock(content="AI text")
        result = function_using_llm()
        self.assertIn("AI", result)
```

**Important Test Patterns**:
- Database tests redirect `backend.database.DATABASE_PATH` to temp file in `setUpModule/tearDownModule`
- Tests creating temp dirs with `tempfile.mkdtemp()` MUST clean up in `tearDown()` using `shutil.rmtree()`
- Use `self.skipTest()` instead of conditional assertions when preconditions aren't met
- Place `if __name__ == '__main__': unittest.main()` at the very end of test files

### Frontend Tests (Vitest)
```jsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

describe('Component', () => {
    it('should work', () => {
        const mockFn = vi.fn()
        mockFn('arg')
        expect(mockFn).toHaveBeenCalledWith('arg')
    })
})
```

## Common Commands

```bash
# Installation
npm run install-all    # Install all dependencies

# Start
npm run start          # Start backend (8000) + frontend (5173)

# Testing
npm run test              # Quick backend tests (~12s)
npm run test:full         # With models (~10min)
cd frontend && npm run test  # Frontend tests

# Backend only
python -m uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000

# Frontend only
cd frontend && npm run dev
```

## Python Version Compatibility

⚠️ **Python 3.8+ Required**: This project must maintain Python 3.8+ compatibility.
- **DO NOT** use parenthesized context managers `with (...)` - requires Python 3.10+
- Use nested `with` blocks or comma-separated contexts instead

## Security

- Path traversal protection is implemented in `model_manager.py` using `is_safe_path()`
- Always validate file paths before operations
- Mock LLM calls in tests to prevent accidental API charges

## Additional Resources

For detailed workspace instructions, code patterns, and change logs, see `AGENTS.md` in the project root.
