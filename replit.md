# Docu-AI-Search

An AI-powered local document search engine with semantic search, LLM integration, and vector embeddings.

## Architecture

- **Frontend**: React + Vite (port 5000), Tailwind CSS, Framer Motion
- **Backend**: FastAPI (port 8000), SQLite, FAISS vector index
- **Proxy**: Vite dev server proxies `/api` and `/ws` to the backend

## Project Structure

```
├── backend/          # Python FastAPI backend
│   ├── api.py        # Main FastAPI application (port 8000)
│   ├── database.py   # SQLite database layer
│   ├── indexing.py   # FAISS vector index management
│   ├── search.py     # Semantic + BM25 hybrid search
│   ├── llm_integration.py  # LLM provider integrations
│   ├── model_manager.py    # Local model download/management
│   ├── settings.py         # Embedding config settings
│   ├── auth.py             # Optional API key auth
│   └── ...
├── frontend/         # React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/     # UI components
│   │   └── lib/            # Utility libraries
│   └── vite.config.js      # Vite config (port 5000, 0.0.0.0, allowedHosts: true)
├── data/             # SQLite DB, FAISS index, logs
└── scripts/          # Utility scripts
```

## Running the App

Two workflows are configured:

1. **Backend** — `python -m uvicorn backend.api:app --reload --host localhost --port 8000`
2. **Start application** — `cd frontend && npm run dev` (serves on port 5000)

## Key Features

- Semantic document search using FAISS + sentence-transformers
- Hybrid BM25 + vector search
- Multi-LLM support: OpenAI, Anthropic, Google Gemini, local GGUF models
- Document indexing: PDF, DOCX, PPTX, XLSX, TXT
- AI chat/agent mode
- Model manager for downloading local LLMs
- WebSocket support for real-time updates

## Environment Variables

Set optional API keys (not required for local embedding):
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROK_API_KEY`

## Database

SQLite at `data/metadata.db` — auto-initialized on startup.
FAISS index saved to `data/index.faiss`.

## Notes

- Frontend Vite config uses `allowedHosts: true` and `host: '0.0.0.0'` for Replit proxy compatibility
- Backend CORS allows all origins for development
- `llama-cpp-python` is installed for local GGUF model inference
- Auth is disabled by default; enable with `AUTH_ENABLED=true`
