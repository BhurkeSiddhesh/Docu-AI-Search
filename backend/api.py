from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Security, WebSocket, WebSocketDisconnect
from backend.auth import require_auth, _get_or_create_token, AUTH_ENABLED
from backend.websocket_manager import manager as ws_manager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIASGIMiddleware
from fastapi.responses import StreamingResponse
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import uvicorn
import os
import time
import configparser
from dotenv import load_dotenv
load_dotenv()

# Path configuration for new folder structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.ini')
INDEX_PATH = os.path.join(DATA_DIR, 'index.faiss')
BENCHMARK_PATH = os.path.join(DATA_DIR, 'benchmark_results.json')
LOG_PATH = os.path.join(DATA_DIR, 'app.log')

# Configure logging FIRST
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("--- SERVER RESTARTING (Logging Configured Early) ---")

def neutralize_log(val: Any) -> str:
    """Neutralize carriage returns and line feeds to prevent log injection."""
    if val is None:
        return ""
    s = str(val)
    return s.replace('\r', '\\r').replace('\n', '\\n')

# --- Lazy Loading Helpers ---
# These functions allow us to defer heavy imports while keeping tests patchable at the module level

def get_embeddings(*args, **kwargs):
    """
    Lazy wrapper for getting embeddings from llm_integration.
    
    Args:
        *args: Variable length argument list passed to get_embeddings.
        **kwargs: Arbitrary keyword arguments passed to get_embeddings.
        
    Returns:
        The result of backend.llm_integration.get_embeddings.
    """
    from backend.llm_integration import get_embeddings as _get_embeddings
    return _get_embeddings(*args, **kwargs)

def summarize(*args, **kwargs):
    """
    Lazy wrapper for the summarization function.
    
    Args:
        *args: Variable length argument list passed to summarize.
        **kwargs: Arbitrary keyword arguments passed to summarize.
        
    Returns:
        The result of backend.llm_integration.summarize.
    """
    from backend.llm_integration import summarize as _summarize
    return _summarize(*args, **kwargs)

def search(*args, **kwargs):
    """
    Lazy wrapper for the semantic search function.
    
    Args:
        *args: Variable length argument list passed to search.
        **kwargs: Arbitrary keyword arguments passed to search.
        
    Returns:
        The result of backend.search.search.
    """
    from backend.search import search as _search
    return _search(*args, **kwargs)

def create_index(*args, **kwargs):
    """
    Lazy wrapper for creating a new FAISS index.
    
    Args:
        *args: Variable length argument list passed to create_index.
        **kwargs: Arbitrary keyword arguments passed to create_index.
        
    Returns:
        The result of backend.indexing.create_index.
    """
    from backend.indexing import create_index as _create_index
    return _create_index(*args, **kwargs)

def save_index(*args, **kwargs):
    """
    Lazy wrapper for saving the FAISS index to disk.
    
    Args:
        *args: Variable length argument list passed to save_index.
        **kwargs: Arbitrary keyword arguments passed to save_index.
        
    Returns:
        The result of backend.indexing.save_index.
    """
    from backend.indexing import save_index as _save_index
    return _save_index(*args, **kwargs)

def load_index(*args, **kwargs):
    """
    Lazy wrapper for loading the FAISS index from disk.
    
    Args:
        *args: Variable length argument list passed to load_index.
        **kwargs: Arbitrary keyword arguments passed to load_index.
        
    Returns:
        The result of backend.indexing.load_index.
    """
    from backend.indexing import load_index as _load_index
    return _load_index(*args, **kwargs)

def smart_summary(*args, **kwargs):
    """
    Lazy wrapper for generating a smart summary using an LLM.
    
    Args:
        *args: Variable length argument list passed to smart_summary.
        **kwargs: Arbitrary keyword arguments passed to smart_summary.
        
    Returns:
        The result of backend.llm_integration.smart_summary.
    """
    from backend.llm_integration import smart_summary as _smart_summary
    return _smart_summary(*args, **kwargs)

def cached_smart_summary(*args, **kwargs):
    """
    Lazy wrapper for getting a cached smart summary.
    
    Args:
        *args: Variable length argument list passed to cached_smart_summary.
        **kwargs: Arbitrary keyword arguments passed to cached_smart_summary.
        
    Returns:
        The result of backend.llm_integration.cached_smart_summary.
    """
    from backend.llm_integration import cached_smart_summary as _cached_smart_summary
    return _cached_smart_summary(*args, **kwargs)

def cached_generate_ai_answer(*args, **kwargs):
    """
    Lazy wrapper for generating a cached AI answer for a query.
    
    Args:
        *args: Variable length argument list passed to cached_generate_ai_answer.
        **kwargs: Arbitrary keyword arguments passed to cached_generate_ai_answer.
        
    Returns:
        The result of backend.llm_integration.cached_generate_ai_answer.
    """
    from backend.llm_integration import cached_generate_ai_answer as _cached_generate_ai_answer
    return _cached_generate_ai_answer(*args, **kwargs)

def stream_ai_answer(*args, **kwargs):
    """
    Lazy wrapper for streaming an AI answer from an LLM.
    
    Args:
        *args: Variable length argument list passed to stream_ai_answer.
        **kwargs: Arbitrary keyword arguments passed to stream_ai_answer.
        
    Returns:
        An iterable or generator yielding response tokens.
    """
    from backend.llm_integration import stream_ai_answer as _stream_ai_answer
    return _stream_ai_answer(*args, **kwargs)

def get_available_models(*args, **kwargs):
    """
    Lazy wrapper for listing all models available for download.
    
    Args:
        *args: Variable length argument list passed to get_available_models.
        **kwargs: Arbitrary keyword arguments passed to get_available_models.
        
    Returns:
        The result of backend.model_manager.get_available_models.
    """
    from backend.model_manager import get_available_models as _get_available_models
    return _get_available_models(*args, **kwargs)

def get_local_models(*args, **kwargs):
    """
    Lazy wrapper for listing already downloaded local models.
    
    Args:
        *args: Variable length argument list passed to get_local_models.
        **kwargs: Arbitrary keyword arguments passed to get_local_models.
        
    Returns:
        The result of backend.model_manager.get_local_models.
    """
    from backend.model_manager import get_local_models as _get_local_models
    return _get_local_models(*args, **kwargs)

def start_download(*args, **kwargs):
    """
    Lazy wrapper for starting a model download task.
    
    Args:
        *args: Variable length argument list passed to start_download.
        **kwargs: Arbitrary keyword arguments passed to start_download.
        
    Returns:
        The result of backend.model_manager.start_download.
    """
    from backend.model_manager import start_download as _start_download
    return _start_download(*args, **kwargs)

def get_download_status(*args, **kwargs):
    """
    Lazy wrapper for checking current model download progress.
    
    Args:
        *args: Variable length argument list passed to get_download_status.
        **kwargs: Arbitrary keyword arguments passed to get_download_status.
        
    Returns:
        The result of backend.model_manager.get_download_status.
    """
    from backend.model_manager import get_download_status as _get_download_status
    return _get_download_status(*args, **kwargs)

def get_active_embedding_client(*args, **kwargs):
    """
    Lazy wrapper for getting the currently active embedding client from settings.
    
    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        
    Returns:
        An instance of an embedding client.
    """
    from backend.settings import get_active_embedding_client as _get_active
    return _get_active(*args, **kwargs)

def get_search_embedding_client(*args, **kwargs):
    """
    Lazy wrapper for the search-time embedding client resolver.

    Unlike get_active_embedding_client, this honours the loaded index's
    metadata so query vectors always match the index dimensions.

    Returns:
        An instance of an embedding client compatible with the loaded index.
    """
    from backend.settings import get_search_embedding_client as _get_search
    return _get_search(*args, **kwargs)

# -----------------------------
from backend import database

def verify_local_request(request: Request):
    """
    Security middleware to ensure sensitive operations can only be triggered from localhost.

    Args:
        request (Request): The incoming FastAPI request object.

    Raises:
        HTTPException: 403 Forbidden if the request is not from a local source.
    """
    if not request.client:
        return # Allow for test client if needed
    client_host = request.client.host
    # Support both IPv4 and IPv6 localhost, and FastAPI TestClient
    if client_host in ("127.0.0.1", "::1", "localhost", "testserver", "testclient"):
        return
    
    logger.warning(f"Security: Blocked remote request to sensitive endpoint from {client_host}")
    raise HTTPException(status_code=403, detail="Access denied: Only local connections allowed")

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.csv', '.md'}



limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app = FastAPI(
    title="Docu-AI-Search API",
    description=(
        "Semantic document search powered by FAISS vector embeddings and local/cloud LLMs. "
        "Interactive docs available at /docs (Swagger UI) and /redoc (ReDoc)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.state.limiter = limiter
app.add_middleware(SlowAPIASGIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return structured JSON for unhandled exceptions and log the traceback."""
    import traceback
    from fastapi.exceptions import RequestValidationError
    from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
    from fastapi.responses import JSONResponse
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    if isinstance(exc, RequestValidationError):
        return await request_validation_exception_handler(request, exc)
    logger.error(
        "Unhandled exception on %s %s: %s\n%s",
        request.method,
        request.url.path,
        exc,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "An unexpected error occurred. Please try again.",
            "error_code": "INTERNAL_SERVER_ERROR",
        },
    )


# Register embedding settings router
from backend.settings import router as embedding_router
app.include_router(embedding_router)

# Enable CORS for frontend
_default_origins = "http://localhost:5173,http://localhost:3000,http://localhost:5175,http://localhost:5174,http://localhost:5000"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
import threading
index = None
docs = []
tags = []
index_summaries = None
cluster_summaries = []
cluster_map = None
bm25 = None
_index_lock = threading.Lock()

@app.get("/")
async def root(request: Request):
    """
    Root endpoint to verify the API is online.

    Args:
        request (Request): The incoming request.

    Returns:
        dict: A dictionary containing the status and a welcome message.
    """
    return {"status": "online", "message": "Docu AI Search API is running"}

@app.get("/api/health")
async def health_check(request: Request):
    """
    Health check endpoint for service monitoring.

    Args:
        request (Request): The incoming request.

    Returns:
        dict: Status of the API, database connectivity, and index readiness.
    """
    from fastapi.responses import JSONResponse
    try:
        conn = database.get_connection()
        conn.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        logger.error("[Health] Database connectivity check failed: %s", e)
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "database": "error", "error": "Database connection failed"},
        )
    with _index_lock:
        idx_loaded = index is not None
    return {"status": "ok", "database": db_status, "index_loaded": idx_loaded}


@app.get("/api/auth/token")
async def get_auth_token(req: Request, _=Depends(verify_local_request)):
    """Return the API auth token (localhost-only). Only available on first retrieval."""
    if not AUTH_ENABLED:
        return {"auth_enabled": False, "message": "Set AUTH_ENABLED=true to enable authentication"}
    token = _get_or_create_token()
    if not token:
        return {"auth_enabled": True, "message": "Token already retrieved. Check config.ini or reset it manually."}
    return {"auth_enabled": True, "token": token}


def load_config():
    """
    Load the application configuration from config.ini.

    Returns:
        configparser.ConfigParser: The parsed configuration object.
    """
    import configparser
    if not os.path.exists(CONFIG_PATH):
        config = configparser.ConfigParser()
        config['General'] = {'folder': '', 'auto_index': 'False'}
        config['APIKeys'] = {'openai_api_key': ''}
        config['LocalLLM'] = {'model_path': '', 'provider': 'openai'}
        config['AdvancedRAG'] = {'query_rewriting': 'False', 'cross_encoder_reranking': 'False', 'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2'}
        with open(CONFIG_PATH, 'w') as configfile:
            config.write(configfile)
    
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    # Override API keys from environment variables if set
    env_key_map = {
        'OPENAI_API_KEY': ('APIKeys', 'openai_api_key'),
        'GEMINI_API_KEY': ('APIKeys', 'gemini_api_key'),
        'ANTHROPIC_API_KEY': ('APIKeys', 'anthropic_api_key'),
        'GROK_API_KEY': ('APIKeys', 'grok_api_key'),
    }
    for env_var, (section, key) in env_key_map.items():
        val = os.environ.get(env_var)
        if val:
            if not config.has_section(section):
                config.add_section(section)
            config.set(section, key, val)
    return config

def save_config_file(config):
    """
    Save the provided configuration to config.ini.

    Args:
        config (configparser.ConfigParser): The configuration object to save.
    """
    with open(CONFIG_PATH, 'w') as configfile:
        config.write(configfile)

def _parse_tensor_split(config) -> Optional[List[float]]:
    """Parse the LocalLLM tensor_split config value into a list of floats."""
    raw = config.get('LocalLLM', 'tensor_split', fallback=None)
    if not raw:
        return None
    try:
        return [float(x) for x in raw.split(',') if x.strip()]
    except (ValueError, AttributeError):
        logger.warning("Invalid tensor_split value in config.ini; ignoring: %s", neutralize_log(raw))
        return None

# Captured at startup so worker threads (indexing) can schedule WebSocket
# broadcasts onto the running event loop via run_coroutine_threadsafe.
_main_event_loop = None

# Initialize index on startup if available
@app.on_event("startup")
async def startup_event():
    """
    Handle application startup events.
    
    Initializes the database, seeds the embedding configuration, and
    triggers the background loading of the search index.
    """
    logger.info("Application starting up...")
    global _main_event_loop
    _main_event_loop = asyncio.get_running_loop()
    database.init_database()
    # Seed embedding config cache from config.ini
    from backend.settings import seed_app_state
    seed_app_state(app)
    # Seed default system prompts if table is empty
    try:
        from backend.system_prompts import seed_default_prompts
        seed_default_prompts()
    except Exception as e:
        logger.warning("Failed to seed system prompts: %s", e)
    # Load index in background to not block health checks
    asyncio.create_task(load_initial_index())
    # Pre-warm the local LLM and embedding model in the background so the
    # first search/answer doesn't pay the multi-second model load.
    asyncio.create_task(warmup_models())

async def warmup_models():
    """Load the embedding model and local GGUF into memory off the hot path."""
    try:
        config = load_config()
        provider = config.get('LocalLLM', 'provider', fallback='local')

        def _warm():
            try:
                from backend.settings import get_active_embedding_client
                get_active_embedding_client(app)
                logger.info("[Warmup] Embedding model ready.")
            except Exception as e:
                logger.warning("[Warmup] Embedding warmup failed: %s", e)
            if provider == 'local':
                try:
                    from backend.llm_integration import warmup_local_model, _discover_local_gguf
                    model_path = config.get('LocalLLM', 'model_path', fallback='') or ''
                    if not model_path or not os.path.exists(model_path):
                        model_path = _discover_local_gguf()
                    if model_path:
                        warmup_local_model(model_path)
                        logger.info("[Warmup] Local LLM ready: %s", os.path.basename(model_path))
                except Exception as e:
                    logger.warning("[Warmup] Local LLM warmup failed: %s", e)

        await asyncio.to_thread(_warm)
    except Exception as e:
        logger.warning("[Warmup] Skipped: %s", e)

async def load_initial_index():
    """
    Load the FAISS index and associated metadata from disk asynchronously.
    
    Updates the global variables for search, including index, docs, tags,
    summaries, and BM25 index.
    """
    global index, docs, tags, index_summaries, cluster_summaries, cluster_map, bm25
    if os.path.exists(INDEX_PATH):
        try:
            # load_index is a blocking call, run it in a thread
            logger.info("Loading existing index in background...")
            res = await asyncio.to_thread(load_index, INDEX_PATH)
            # Unpack 8-tuple (7 data items + meta dict added in latest indexing.py)
            _index_meta = res[7] if len(res) > 7 else {}
            with _index_lock:
                index, docs, tags, index_summaries, cluster_summaries, cluster_map, bm25 = res[:7]
            # Expose the index's model/dim so search can embed queries with
            # the matching model (see settings.get_search_embedding_client).
            app.state.index_meta = _index_meta
            logger.info(
                "Loaded existing index successfully. "
                "Model: %s, dim: %s",
                _index_meta.get('model_name', 'unknown'),
                _index_meta.get('embedding_dim', '?'),
            )
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            with _index_lock:
                index = None
                docs = []
                tags = []
                index_summaries = None
                cluster_summaries = []
                cluster_map = None
                bm25 = None

@app.get("/api/browse")
async def browse_folder(request: Request):
    """
    Open a folder browser dialog and return the selected path.

    Args:
        request (Request): The incoming request.

    Returns:
        dict: A dictionary containing the selected 'folder' path or None.

    Raises:
        HTTPException: 500 if the folder dialog fails to open.
    """
    import tkinter as tk
    from tkinter import filedialog
    try:
        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # Bring dialog to front
        
        folder_path = filedialog.askdirectory(title="Select Folder to Index")
        root.destroy()
        
        if folder_path:
            return {"folder": folder_path}
        else:
            return {"folder": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open folder dialog: {str(e)}")

@app.get("/api/models/available")
async def list_available_models(request: Request):
    """
    List models available for download from the cloud.

    Args:
        request (Request): The incoming request.

    Returns:
        List[dict]: A list of available model metadata.
    """
    return get_available_models()

@app.get("/api/models/local")
async def list_local_models(request: Request):
    """
    List models already downloaded to the local directory.

    Args:
        request (Request): The incoming request.

    Returns:
        List[dict]: A list of local model metadata.
    """
    return get_local_models()

@app.post("/api/models/download/{model_id}")
@limiter.limit("3/minute")
async def download_model_endpoint(model_id: str, request: Request):
    """
    Trigger a background task to download a specific model.

    Args:
        model_id (str): The identifier of the model to download.
        request (Request): The incoming request.

    Returns:
        dict: Success status and message.

    Raises:
        HTTPException: 400 if the download fails to start.
    """
    success, message = start_download(model_id)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "success", "message": message}

@app.get("/api/models/status")
async def download_status_endpoint(request: Request):
    """
    Get the current status and progress of model downloads.

    Args:
        request (Request): The incoming request.

    Returns:
        dict: Progress and status information.
    """
    return get_download_status()

@app.delete("/api/models/delete")
async def delete_model(request: dict, req: Request):
    """
    Delete a downloaded model file from disk.

    Args:
        request (dict): Body containing 'path' of the model to delete.
        req (Request): The incoming request.

    Returns:
        dict: Success status and message.

    Raises:
        HTTPException: 400 if path is missing, 404 if file missing, 500 on error.
    """
    model_path = request.get('path', '')
    if not model_path:
        raise HTTPException(status_code=400, detail="Model path required")
    
    try:
        from backend.model_manager import delete_model
        if delete_model(model_path):
            return {"status": "success", "message": "Model deleted"}
        else:
            raise HTTPException(status_code=404, detail="Model file not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[API] Failed to delete model: %s", e)
        raise HTTPException(status_code=500, detail="Failed to delete model")

@app.get("/api/cache/stats")
def cache_stats_endpoint():
    """
    Get statistics about the AI response cache.

    Returns:
        dict: Cache statistics including hit count and entry count.
    """
    return database.get_cache_stats()

@app.post("/api/cache/clear")
def clear_cache_endpoint():
    """
    Clear all entries from the AI response cache.

    Returns:
        dict: Success status and the number of entries cleared.
    """
    count = database.clear_response_cache()
    return {"status": "success", "cleared_entries": count}

# Benchmark state
benchmark_status = {
    "running": False,
    "progress": 0,
    "current_model": None,
    "error": None
}
benchmark_results = None

# Indexing state
indexing_status = {
    "running": False,
    "progress": 0,
    "current_file": None,
    "total_files": 0,
    "processed_files": 0,
    "error": None
}

def run_benchmark_task():
    """
    Background task to run benchmarks across all local models.
    
    This function:
    1. Updates global benchmark_status to 'running'.
    2. Imports and executes model benchmarking scripts.
    3. Triggers save_results to persist metrics to benchark_results.json.
    4. Updates global benchmark_results in memory for immediate API access.
    """
    global benchmark_status, benchmark_results
    import json
    
    try:
        benchmark_status["running"] = True
        benchmark_status["progress"] = 0
        benchmark_status["error"] = None
        
        # Import and run benchmarks
        from scripts.benchmark_models import run_all_benchmarks, save_results
        
        results = run_all_benchmarks(verbose=False)
        if results:
            save_results(results)
            benchmark_results = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": [r.to_dict() for r in results]
            }
        
        benchmark_status["running"] = False
        benchmark_status["progress"] = 100
        
    except Exception as e:
        benchmark_status["running"] = False
        benchmark_status["error"] = str(e)

@app.post("/api/benchmarks/run")
async def run_benchmarks(background_tasks: BackgroundTasks, request: Request):
    """
    Start the benchmark suite in the background.

    Args:
        background_tasks (BackgroundTasks): FastAPI background tasks manager.
        request (Request): The incoming request.

    Returns:
        dict: Success status and message.

    Raises:
        HTTPException: 400 if a benchmark is already running.
    """
    global benchmark_status
    
    if benchmark_status["running"]:
        raise HTTPException(status_code=400, detail="Benchmark already running")
    
    background_tasks.add_task(run_benchmark_task)
    return {"status": "started", "message": "Benchmark started in background"}

@app.get("/api/benchmarks/status")
async def get_benchmark_status(request: Request):
    """
    Get the current status of the background benchmark task.

    Args:
        request (Request): The incoming request.

    Returns:
        dict: The benchmark status (running, progress, current_model, error).
    """
    return benchmark_status

@app.get("/api/benchmarks/results")
async def get_benchmark_results(request: Request):
    """
    Get the latest benchmark results from memory or disk.

    Args:
        request (Request): The incoming request.

    Returns:
        dict: The benchmark results and timestamp.
    """
    global benchmark_results
    
    # Try to load from file if not in memory
    if benchmark_results is None:
        if os.path.exists(BENCHMARK_PATH):
            import json
            with open(BENCHMARK_PATH, 'r') as f:
                benchmark_results = json.load(f)
    
    if benchmark_results is None:
        return {"results": [], "message": "No benchmark results available. Run a benchmark first."}
    
    return benchmark_results




class SearchRequest(BaseModel):
    """
    Data model for a semantic search request.

    Attributes:
        query (str): The search query text.
        context (Optional[List[str]]): Optional context strings to refine search.
    """
    model_config = {'protected_namespaces': ()}

    query: str = Field(..., min_length=1, max_length=5000)
    context: Optional[List[str]] = None
    system_prompt_id: Optional[int] = None
    provider_override: Optional[str] = None
    model_override: Optional[str] = None
    api_key_override: Optional[str] = None
    base_url_override: Optional[str] = None
    # Search filters
    file_types: Optional[List[str]] = None  # e.g. ["pdf", "docx"]
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sort_by: Optional[str] = Field(default=None, pattern="^(relevance|date|filename|file_size)$")

class SearchResult(BaseModel):
    """
    Data model for a single search result item.

    Attributes:
        document (str): The snippet or title of the document.
        summary (Optional[str]): A summary of the document content.
        tags (List[str]): Extracted tags for the document.
        file_path (Optional[str]): Absolute system path to the file.
        file_name (Optional[str]): The filename.
        faiss_idx (Optional[int]): The internal FAISS index for the document chunk.
    """
    document: str
    summary: Optional[str] = None
    tags: List[str] = []
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    faiss_idx: Optional[int] = None
    # Knowledge-graph neighbours: [{path, filename, similarity}, ...]
    related_files: List[Dict[str, Any]] = []

class SearchResponse(BaseModel):
    """
    Data model for the complete search response.

    Attributes:
        results (List[SearchResult]): List of individual search results.
        ai_answer (Optional[str]): AI-generated natural language answer.
        active_model (Optional[str]): The name of the model used for response.
    """
    results: List[SearchResult]
    ai_answer: Optional[str] = ""
    active_model: Optional[str] = ""

class ConfigModel(BaseModel):
    """
    Data model for application configuration.

    Attributes:
        folders (List[str]): List of folders to be indexed.
        auto_index (bool): Whether to index folders automatically on change.
        openai_api_key (Optional[str]): API key for OpenAI.
        gemini_api_key (Optional[str]): API key for Gemini.
        anthropic_api_key (Optional[str]): API key for Anthropic.
        grok_api_key (Optional[str]): API key for Grok.
        local_model_path (Optional[str]): Path to the local GGUF model file.
        provider (str): The selected LLM provider.
        tensor_split (Optional[str]): GPU split configuration for LlamaCpp.
        query_rewriting (bool): Whether to enable AI query rewriting.
        cross_encoder_reranking (bool): Whether to enable Cross-Encoder reranking.
        reranker_model (str): The model used for reranking.
    """
    folders: List[str] = []
    auto_index: bool = False
    openai_api_key: Optional[str] = ""
    gemini_api_key: Optional[str] = ""
    anthropic_api_key: Optional[str] = ""
    grok_api_key: Optional[str] = ""
    local_model_path: Optional[str] = ""
    provider: str = "openai"
    tensor_split: Optional[str] = None
    query_rewriting: bool = False
    cross_encoder_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # External providers (Ollama, LM Studio)
    ollama_base_url: Optional[str] = "http://localhost:11434"
    lmstudio_base_url: Optional[str] = "http://localhost:1234/v1"
    external_model_name: Optional[str] = ""
    external_api_key: Optional[str] = ""

@app.get("/api/config")
async def get_config(request: Request):
    """
    Retrieve the current application configuration.

    Args:
        request (Request): The incoming request.

    Returns:
        dict: The configuration mapping (keys, folders, provider settings).
    """
    config = load_config()
    # Handle both old 'folder' and new 'folders' format
    folder = config.get('General', 'folder', fallback='')
    folders_str = config.get('General', 'folders', fallback='')
    
    folders = [f.strip() for f in folders_str.split(',') if f.strip()]
    if not folders and folder:
        folders = [folder]
        
    return {
        "folders": folders,
        "auto_index": config.getboolean('General', 'auto_index', fallback=False),
        "openai_api_key_set": bool(config.get('APIKeys', 'openai_api_key', fallback='')),
        "gemini_api_key_set": bool(config.get('APIKeys', 'gemini_api_key', fallback='')),
        "anthropic_api_key_set": bool(config.get('APIKeys', 'anthropic_api_key', fallback='')),
        "grok_api_key_set": bool(config.get('APIKeys', 'grok_api_key', fallback='')),
        "local_model_path": config.get('LocalLLM', 'model_path', fallback=''),
        "provider": config.get('LocalLLM', 'provider', fallback='openai'),
        "tensor_split": config.get('LocalLLM', 'tensor_split', fallback=None),
        "query_rewriting": config.getboolean('AdvancedRAG', 'query_rewriting', fallback=False),
        "cross_encoder_reranking": config.getboolean('AdvancedRAG', 'cross_encoder_reranking', fallback=False),
        "reranker_model": config.get('AdvancedRAG', 'reranker_model', fallback='cross-encoder/ms-marco-MiniLM-L-6-v2'),
        # External providers
        "ollama_base_url": config.get('ExternalProviders', 'ollama_base_url', fallback='http://localhost:11434'),
        "lmstudio_base_url": config.get('ExternalProviders', 'lmstudio_base_url', fallback='http://localhost:1234/v1'),
        "external_model_name": config.get('ExternalProviders', 'external_model_name', fallback=''),
        "external_api_key": config.get('ExternalProviders', 'external_api_key', fallback=''),
    }

@app.post("/api/config")
@limiter.limit("10/minute")
async def update_config(config_data: ConfigModel, request: Request):
    """
    Update the application configuration.

    Args:
        config_data (ConfigModel): The new configuration data.
        request (Request): The incoming request.

    Returns:
        dict: Success status and message.
    """
    config = load_config()
    config['General'] = {
        'folders': ','.join(config_data.folders),
        'auto_index': str(config_data.auto_index)
    }
    # Preserve existing key if client submits empty string (i.e. user left it blank)
    def _key(new_val, section, option):
        if new_val:
            return new_val
        return config.get(section, option, fallback='')
    config['APIKeys'] = {
        'openai_api_key': _key(config_data.openai_api_key, 'APIKeys', 'openai_api_key'),
        'gemini_api_key': _key(config_data.gemini_api_key, 'APIKeys', 'gemini_api_key'),
        'anthropic_api_key': _key(config_data.anthropic_api_key, 'APIKeys', 'anthropic_api_key'),
        'grok_api_key': _key(config_data.grok_api_key, 'APIKeys', 'grok_api_key'),
    }
    config['LocalLLM'] = {
        'model_path': config_data.local_model_path or '', 
        'provider': config_data.provider,
        'tensor_split': config_data.tensor_split or ''
    }
    config['AdvancedRAG'] = {
        'query_rewriting': str(config_data.query_rewriting),
        'cross_encoder_reranking': str(config_data.cross_encoder_reranking),
        'reranker_model': config_data.reranker_model or 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    }
    config['ExternalProviders'] = {
        'ollama_base_url': config_data.ollama_base_url or 'http://localhost:11434',
        'lmstudio_base_url': config_data.lmstudio_base_url or 'http://localhost:1234/v1',
        'external_model_name': config_data.external_model_name or '',
        'external_api_key': config_data.external_api_key or '',
    }
    save_config_file(config)
    
    # Save folders to history
    try:
        if config_data.folders:
            for folder in config_data.folders:
                database.add_folder_to_history(folder)
    except Exception as e:
        logger.info(f"Failed to update folder history: {e}")
        
    return {"status": "success", "message": "Configuration saved"}

@app.post("/api/search")
@limiter.limit("30/minute")
async def search_files(search_data: SearchRequest, request: Request, background_tasks: BackgroundTasks, _auth=Depends(require_auth)):
    """
    Perform a semantic search on the indexed documents.

    This endpoint supports:
    1. Vanilla semantic search (FAISS + BM25 hybrid).
    2. Agentic mode (if enabled in config).
    3. Advanced RAG (query rewriting, reranking).
    4. SSE Streaming for agentic responses.

    Args:
        search_data (SearchRequest): The search query and optional context.
        request (Request): The incoming request object.

    Returns:
        SearchResponse or StreamingResponse: The search results or an AI-generated answer.

    Raises:
        HTTPException: 400 if index not loaded, 409 if embedding dimension mismatch.
    """
    with _index_lock:
        index_snap, docs_snap, tags_snap = index, docs, tags
        isumm_snap, csumm_snap, cmap_snap, bm25_snap = index_summaries, cluster_summaries, cluster_map, bm25

    if not index_snap:
        raise HTTPException(status_code=400, detail="Index not loaded. Please configure and index a folder first.")

    logger.info(f"\n[API] POST /api/search - Query: <redacted>")

    try:
        start_time = time.time()
        
        config = load_config()
        provider = config.get('LocalLLM', 'provider', fallback='openai')
        
        # Determine correct API key based on provider
        api_key = config.get('APIKeys', 'openai_api_key', fallback=None)
        if provider == 'gemini':
            api_key = config.get('APIKeys', 'gemini_api_key', fallback=api_key)
        elif provider == 'anthropic':
            api_key = config.get('APIKeys', 'anthropic_api_key', fallback=api_key)
        elif provider == 'grok':
            api_key = config.get('APIKeys', 'grok_api_key', fallback=api_key)

        is_agentic = config.get('General', 'agent_mode', fallback='False').lower() == 'true'
        
        if is_agentic:
            logger.info("[API] Running in AGENTIC mode.")
            from backend.agent import ReActAgent
            agent = ReActAgent({
                'index': index_snap, 'docs': docs_snap, 'tags': tags_snap, 'config': config,
                'index_summaries': isumm_snap, 'cluster_summaries': csumm_snap,
                'cluster_map': cmap_snap, 'bm25': bm25_snap
            })
            return StreamingResponse(agent.stream_chat(search_data.query), media_type="text/event-stream")

        model_path = config.get('LocalLLM', 'model_path', fallback=None)

        # Run Search — use the active embedding client from settings state
        from backend.search import EmbeddingDimensionMismatchError
        _search_timeout = int(os.getenv("SEARCH_TIMEOUT_SECONDS", "30"))
        try:
            results, _context_snippets = await asyncio.wait_for(
                asyncio.to_thread(
                    search,
                    search_data.query, index_snap, docs_snap, tags_snap,
                    get_search_embedding_client(request.app),
                    isumm_snap, csumm_snap, cmap_snap, bm25_snap
                ),
                timeout=_search_timeout,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Search timed out. The embedding service may be unavailable.")
        except EmbeddingDimensionMismatchError as dim_err:
            logger.error("[Search] Embedding dimension mismatch: %s", dim_err)
            raise HTTPException(
                status_code=409,
                detail="Embedding dimension mismatch: the index was built with a different model. Please re-index your documents.",
            )
        
        # OPTIMIZATION: Batch database lookups for missing file info
        indices_to_lookup = list(dict.fromkeys(
            result['faiss_idx']
            for result in results
            if not result.get('file_path') and result.get('faiss_idx') is not None
        ))

        file_lookup_map = {}
        if indices_to_lookup:
            try:
                file_lookup_map = database.get_files_by_faiss_indices(indices_to_lookup)
            except ValueError as ve:
                logger.warning(f"Batch lookup failed, falling back to sequential: {ve}")
                # Fallback: manually lookup one by one if batch size exceeded
                for f_idx in indices_to_lookup:
                    info = database.get_file_by_faiss_index(f_idx)
                    if info:
                        file_lookup_map[f_idx] = info

        processed_results = []
        context_snippets = []

        # Build normalised filter values once
        _file_type_filter = {ft.lower().lstrip('.') for ft in (search_data.file_types or [])}

        # Per-result LLM summaries are opt-in: with a local GGUF configured they
        # add seconds *per result* to every search. The streamed AI answer
        # (/api/stream-answer) is the intended place for LLM output.
        _llm_result_summaries = config.getboolean('AdvancedRAG', 'llm_result_summaries', fallback=False)

        for result in results:
            faiss_idx = result.get('faiss_idx')

            # Apply min_score filter
            if search_data.min_score is not None:
                score = result.get('score', 1.0)
                if score < search_data.min_score:
                    continue

            # Use file info from search result first (it comes from FAISS doc metadata)
            file_path = result.get('file_path')
            file_name = result.get('file_name')

            # If not in search result, use batched lookup map
            if not file_path and faiss_idx in file_lookup_map:
                file_info = file_lookup_map[faiss_idx]
                file_path = file_info.get('path')
                file_name = file_info.get('filename')

            # Apply file type filter
            if _file_type_filter and file_path:
                ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                if ext not in _file_type_filter:
                    continue
            
            if _llm_result_summaries:
                summary = cached_smart_summary(text=result['document'], query=search_data.query, provider=provider, api_key=api_key, model_path=model_path)
            else:
                # Fast extractive summary — no model call, sub-millisecond
                from backend.llm_integration import summarize as _fast_summarize
                summary = _fast_summarize(result['document'], question=search_data.query)
            
            # Add file context to snippets for AI answer
            file_prefix = f"[From: {file_name}] " if file_name else ""
            # Format context for AI (Nexus Insight)
            # We prefer using the smart summary if available, otherwise raw document text
            if summary and len(summary) > 20:
                context_snippets.append(f"{file_prefix}{summary}")
            else:
                context_snippets.append(f"{file_prefix}{result['document'][:500]}")

            # Convert tags from string to list if needed
            result_tags = result.get('tags', '')
            if isinstance(result_tags, str):
                result_tags = [t.strip() for t in result_tags.split(',') if t.strip()]
            
            processed_results.append(SearchResult(
                document=result['document'],
                summary=summary,
                tags=result_tags,
                faiss_idx=faiss_idx,
                file_path=file_path,
                file_name=file_name
            ))
        
        # Apply sort_by if requested (default is relevance from FAISS)
        if search_data.sort_by and search_data.sort_by != "relevance":
            if search_data.sort_by == "filename":
                processed_results.sort(key=lambda r: (r.file_name or "").lower())
            elif search_data.sort_by == "file_size":
                processed_results.sort(key=lambda r: os.path.getsize(r.file_path) if r.file_path and os.path.exists(r.file_path) else 0, reverse=True)
            elif search_data.sort_by == "date":
                processed_results.sort(key=lambda r: os.path.getmtime(r.file_path) if r.file_path and os.path.exists(r.file_path) else 0, reverse=True)

        # Attach knowledge-graph neighbours (Glean-style "related documents")
        try:
            _result_paths = [r.file_path for r in processed_results if r.file_path]
            if _result_paths:
                _related_map = await asyncio.to_thread(database.get_related_files, _result_paths)
                if isinstance(_related_map, dict):
                    for r in processed_results:
                        if r.file_path and r.file_path in _related_map:
                            r.related_files = _related_map[r.file_path]
        except Exception as _rel_err:
            logger.warning("Related-files lookup failed: %s", _rel_err)

        # Return results immediately - AI Answer will be streamed via separate endpoint
        active_model_name = provider.capitalize()
        if provider == 'local' and model_path:
            active_model_name = os.path.basename(model_path).replace(".gguf", "").replace("-", " ")
        
        # Save to search history
        execution_time_ms = int((time.time() - start_time) * 1000)
        background_tasks.add_task(database.add_search_history, search_data.query, len(processed_results), execution_time_ms)
            
        return SearchResponse(
            results=processed_results,
            ai_answer="", # Empty for immediate return
            active_model=active_model_name
        )
    except HTTPException:
        raise  # re-raise HTTP exceptions (409 dimension mismatch, etc.) with original status
    except Exception as e:
        logger.error("Search error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred processing your search request")

@app.post("/api/stream-answer")
@limiter.limit("30/minute")
async def stream_answer_endpoint(search_data: SearchRequest, request: Request, _auth=Depends(require_auth)):
    """
    Stream the AI answer for a given search query.

    If context is provided in the request, it uses that. Otherwise, it 
    re-runs the semantic search to gather relevant snippets and then 
    streams tokens from the selected LLM provider.

    Args:
        search_data (SearchRequest): The search query and optional context snippets.
        request (Request): The incoming request object.

    Returns:
        StreamingResponse: A server-sent event stream of AI response tokens.
    """
    with _index_lock:
        index_snap, docs_snap, tags_snap = index, docs, tags
        isumm_snap, csumm_snap, cmap_snap, bm25_snap = index_summaries, cluster_summaries, cluster_map, bm25

    # The index is only needed when we have to run a fresh search; when the
    # caller already supplies context snippets, stream straight from the LLM.
    if not index_snap and not search_data.context:
         return StreamingResponse(iter(["Error: Index not loaded."]), media_type="text/event-stream")

    config = load_config()
    provider = search_data.provider_override or config.get('LocalLLM', 'provider', fallback='openai')

    api_key = search_data.api_key_override
    if not api_key:
        api_key = config.get('APIKeys', 'openai_api_key', fallback=None)
        if provider == 'gemini':
            api_key = config.get('APIKeys', 'gemini_api_key', fallback=api_key)
        elif provider == 'anthropic':
            api_key = config.get('APIKeys', 'anthropic_api_key', fallback=api_key)
        elif provider == 'grok':
            api_key = config.get('APIKeys', 'grok_api_key', fallback=api_key)
        elif provider in ('ollama', 'lmstudio'):
            api_key = config.get('ExternalProviders', 'external_api_key', fallback=api_key)

    model_path = search_data.model_override or config.get('LocalLLM', 'model_path', fallback=None)
    tensor_split = _parse_tensor_split(config)

    final_context_snippets = []

    if search_data.context:
        logger.info(f"[API] Using provided context ({len(search_data.context)} snippets) for streaming answer")
        final_context_snippets = search_data.context
    else:
        # Run the search off the event loop — it embeds the query and hits
        # FAISS/BM25, which are CPU-bound and would otherwise stall all
        # concurrent requests.
        from backend.search import EmbeddingDimensionMismatchError
        _search_timeout = int(os.getenv("SEARCH_TIMEOUT_SECONDS", "30"))
        try:
            results, _ = await asyncio.wait_for(
                asyncio.to_thread(
                    search,
                    search_data.query, index_snap, docs_snap, tags_snap,
                    get_search_embedding_client(request.app),
                    isumm_snap, csumm_snap, cmap_snap, bm25_snap,
                ),
                timeout=_search_timeout,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Search timed out. The embedding service may be unavailable.")
        except EmbeddingDimensionMismatchError as dim_err:
            logger.error("[Stream] Embedding dimension mismatch: %s", dim_err)
            raise HTTPException(
                status_code=409,
                detail="Embedding dimension mismatch: the index was built with a different model. Please re-index your documents.",
            )

        # OPTIMIZATION: Batch fetch missing file info to avoid N+1 queries and improve context quality
        missing_faiss_idxs = [
            r.get('faiss_idx') for r in results
            if not r.get('file_name') and r.get('faiss_idx') is not None
        ]
        # De-duplicate indices to avoid inflating SQL query
        missing_faiss_idxs = list(dict.fromkeys(missing_faiss_idxs))

        file_info_map = {}
        if missing_faiss_idxs:
            try:
                file_info_map = database.get_files_by_faiss_indices(missing_faiss_idxs)
            except ValueError as ve:
                 logger.warning("Batch lookup failed in stream-answer, falling back: %s", ve)
                 for f_idx in missing_faiss_idxs:
                     info = database.get_file_by_faiss_index(f_idx)
                     if info:
                         file_info_map[f_idx] = info

        # Per-result summaries: extractive by default (sub-millisecond). LLM
        # summaries here previously ran one full model call PER RESULT before
        # any token streamed — tens of seconds of dead air with a local GGUF.
        try:
            _llm_result_summaries = config.getboolean('AdvancedRAG', 'llm_result_summaries', fallback=False) is True
        except Exception:
            _llm_result_summaries = False

        # Prepare context
        for result in results:
             if _llm_result_summaries:
                 summary = cached_smart_summary(text=result['document'], query=search_data.query, provider=provider, api_key=api_key, model_path=model_path)
             else:
                 summary = summarize(result['document'], question=search_data.query)

             faiss_idx = result.get('faiss_idx')
             file_name = result.get('file_name')

             # Fallback to database lookup if missing
             if not file_name and faiss_idx is not None:
                 file_info = file_info_map.get(faiss_idx)
                 if file_info:
                     file_name = file_info.get('filename')

             file_prefix = f"[From: {file_name}] " if file_name else ""
             if summary and len(summary) > 20:
                 final_context_snippets.append(f"{file_prefix}{summary}")
             else:
                 final_context_snippets.append(f"{file_prefix}{result['document'][:500]}")

    safe_snippets = [s[:600] for s in final_context_snippets[:6]]
    context_text = "\n\n".join(safe_snippets)

    if not context_text:
        return StreamingResponse(iter(["No relevant context found."]), media_type="text/event-stream")

    system_instruction = None
    if getattr(search_data, 'system_prompt_id', None) is not None:
        from backend.system_prompts import get_system_prompt_by_id
        prompt_data = get_system_prompt_by_id(search_data.system_prompt_id)
        if prompt_data:
            system_instruction = prompt_data["content"]

    # ── Answer cache ───────────────────────────────────────────────────────
    # Repeat questions over the same context replay instantly instead of
    # paying full generation time again (~20s on a 7B CPU model). The cache is
    # a pure optimization: any failure here silently falls back to generation.
    if provider == 'local' and model_path:
        _model_id = f"local:{os.path.basename(str(model_path))}"
    else:
        _model_id = str(provider)
    _q_hash = _c_hash = None
    cached_answer = None
    try:
        from backend.llm_integration import compute_cache_key
        _cache_query = f"{search_data.query}\x1f{system_instruction or ''}"
        _hashes = compute_cache_key(_cache_query, context_text, _model_id)
        if isinstance(_hashes, (tuple, list)) and len(_hashes) == 2:
            _q_hash, _c_hash = _hashes
            cached_answer = await asyncio.to_thread(
                database.get_cached_response, _q_hash, _c_hash, _model_id, "ai_answer"
            )
    except Exception as _cache_err:
        logger.debug("[Stream] Answer cache lookup skipped: %s", _cache_err)
        _q_hash = _c_hash = None
        cached_answer = None
    if isinstance(cached_answer, str) and cached_answer.strip():
        logger.info("[Stream] Answer cache hit — replaying instantly.")
        async def replay():
            _CHUNK = 512
            for i in range(0, len(cached_answer), _CHUNK):
                yield cached_answer[i:i + _CHUNK]
        return StreamingResponse(replay(), media_type="text/event-stream")
    # ───────────────────────────────────────────────────────────────────────

    async def generate():
        """Bridge the blocking LLM token generator to async without stalling
        the event loop: a worker thread owns the iterator and pushes tokens
        into an asyncio.Queue, so every token reaches the client immediately
        and other requests stay responsive during generation."""
        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        _DONE = object()
        stop_event = threading.Event()

        def _produce():
            try:
                for token in stream_ai_answer(
                    context_text, search_data.query, provider, api_key, model_path,
                    tensor_split, system_instruction, base_url=search_data.base_url_override
                ):
                    if stop_event.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(token_queue.put(token), loop).result()
                    if stop_event.is_set():
                        break
            except Exception as _stream_err:
                logger.error("[Stream] Answer generation error: %s", _stream_err)
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(token_queue.put(_DONE), loop).result(timeout=5)
                except Exception:
                    pass

        producer = threading.Thread(target=_produce, name="answer-stream-producer", daemon=True)
        producer.start()

        collected = []
        try:
            while True:
                token = await token_queue.get()
                if token is _DONE:
                    break
                collected.append(str(token))
                yield token

            answer = "".join(collected).strip()
            if _q_hash and _c_hash and answer and not answer.startswith("Error") and not answer.startswith("No relevant"):
                try:
                    await asyncio.to_thread(
                        database.cache_response, _q_hash, _c_hash, _model_id, "ai_answer", answer
                    )
                except Exception as cache_err:
                    logger.warning("[Stream] Could not cache answer: %s", cache_err)
        finally:
            # Client disconnected or stream finished: let the producer exit so
            # it releases the local-LLM lock. Drain any queued tokens so a
            # producer blocked on a full queue can observe stop_event.
            stop_event.set()
            while not token_queue.empty():
                try:
                    token_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    return StreamingResponse(generate(), media_type="text/event-stream")


# -------------------------------------------------------------------------
# External Provider endpoints (Ollama, LM Studio, OpenAI-compatible)
# -------------------------------------------------------------------------

class ProviderQueryRequest(BaseModel):
    provider_type: str  # 'ollama' or 'lmstudio'
    base_url: Optional[str] = None
    api_key: Optional[str] = ""

@app.post("/api/providers/health")
async def provider_health_check(body: ProviderQueryRequest, request: Request):
    """Check if an external LLM provider (Ollama / LM Studio) is reachable."""
    from backend.providers import get_provider
    try:
        provider = get_provider(body.provider_type, {
            "base_url": body.base_url,
            "model": "",
            "api_key": body.api_key or "",
        })
        result = provider.health_check()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/providers/models")
async def provider_list_models(body: ProviderQueryRequest, request: Request):
    """Fetch available models from an external LLM provider."""
    from backend.providers import get_provider
    try:
        provider = get_provider(body.provider_type, {
            "base_url": body.base_url,
            "model": "",
            "api_key": body.api_key or "",
        })
        models = provider.list_models()
        return {"models": models}
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/providers/list")
async def list_providers(request: Request):
    """List all supported LLM provider types."""
    return {
        "providers": [
            {"id": "local", "name": "Local Model (GGUF)", "needs_server": False},
            {"id": "ollama", "name": "Ollama", "needs_server": True, "default_url": "http://localhost:11434"},
            {"id": "lmstudio", "name": "LM Studio", "needs_server": True, "default_url": "http://localhost:1234/v1"},
            {"id": "openai", "name": "OpenAI", "needs_server": False},
            {"id": "gemini", "name": "Google Gemini", "needs_server": False},
            {"id": "anthropic", "name": "Anthropic Claude", "needs_server": False},
            {"id": "grok", "name": "xAI Grok", "needs_server": False},
        ]
    }


# -------------------------------------------------------------------------
# System Prompts endpoints
# -------------------------------------------------------------------------

class SystemPromptRequest(BaseModel):
    name: str
    content: str
    category: str = "general"

@app.get("/api/system-prompts")
async def list_system_prompts(request: Request, category: Optional[str] = None):
    """List all system prompts, optionally filtered by category."""
    from backend.system_prompts import get_system_prompts
    return get_system_prompts(category=category)

@app.post("/api/system-prompts")
async def create_system_prompt(body: SystemPromptRequest, request: Request):
    """Create a new system prompt."""
    from backend.system_prompts import add_system_prompt
    prompt_id = add_system_prompt(body.name, body.content, body.category)
    return {"status": "success", "id": prompt_id}

@app.delete("/api/system-prompts/{prompt_id}")
async def delete_system_prompt_endpoint(prompt_id: int, request: Request):
    """Delete a system prompt by ID."""
    from backend.system_prompts import delete_system_prompt
    if delete_system_prompt(prompt_id):
        return {"status": "success", "message": "Prompt deleted"}
    raise HTTPException(status_code=404, detail="System prompt not found")

@app.get("/api/search/history")
async def get_search_history(request: Request):
    """
    Retrieve recent search history from the database.

    Args:
        request (Request): The incoming request.

    Returns:
        List[dict]: A list of search history entries.

    Raises:
        HTTPException: 500 if database retrieval fails.
    """
    try:
        history = await asyncio.to_thread(database.get_search_history, limit=50)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/search/history/{history_id}")
async def delete_search_history_item(history_id: int, request: Request):
    """
    Delete a specific search history entry.

    Args:
        history_id (int): The ID of the history item to delete.
        request (Request): The incoming request.

    Returns:
        dict: Success status and message.

    Raises:
        HTTPException: 404 if item not found, 500 on database error.
    """
    try:
        success = database.delete_search_history_item(history_id)
        if success:
            return {"status": "success", "message": "History item deleted"}
        raise HTTPException(status_code=404, detail="History item not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/search/history")
async def delete_all_search_history(request: Request):
    """
    Clear all search history from the database.

    Args:
        request (Request): The incoming request.

    Returns:
        dict: Success status and the count of deleted items.

    Raises:
        HTTPException: 500 if database operation fails.
    """
    try:
        count = database.delete_all_search_history()
        return {"status": "success", "message": f"Deleted {count} history items", "deleted_count": count}
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class LogRequest(BaseModel):
    """
    Data model for receiving logs from the frontend.

    Attributes:
        level (str): Log level (info, warn, error).
        message (str): The log message text.
        source (Optional[str]): Source of the log (defaults to 'Frontend').
        stack (Optional[str]): Optional stack trace for errors.
    """
    level: str
    message: str
    source: Optional[str] = "Frontend"
    stack: Optional[str] = None

@app.post("/api/logs")
async def receive_log(log: LogRequest, request: Request):
    """
    Receive logs from the frontend and pipe them to the backend logger.

    Args:
        log (LogRequest): The log data from the frontend.
        request (Request): The incoming request.

    Returns:
        dict: Status 'logged'.
    """
    log_msg = f"[{log.source}] {log.message}"
    if log.stack:
        log_msg += f"\nStack: {log.stack}"
    
    if log.level.lower() == 'error':
        logger.error(log_msg)
    elif log.level.lower() == 'warn' or log.level.lower() == 'warning':
        logger.warning(log_msg)
    else:
        logger.info(log_msg)
    return {"status": "logged"}

@app.post("/api/open-file")
async def open_file(body: dict, request: Request, _=Depends(verify_local_request)):
    """
    Open a file using the system's default application.

    Security Measures:
    - Only allowed from localhost (via Depends).
    - Prevents argument injection (leading dashes).
    - Only allows opening files verified to be in the index.
    - Whitelists file extensions (ALLOWED_EXTENSIONS).

    Args:
        body (dict): Body containing the 'path' of the file to open.
        request (Request): The incoming request.
        _ (Depends): Security dependency for local request verification.

    Returns:
        dict: Success status and message.

    Raises:
        HTTPException: 400 if path missing/invalid, 403 if access denied, 404 if file missing, 500 on system error.
    """
    logger.info("[DEBUG-OPEN] Entered open_file")
    file_path = body.get('path', '')
    if not file_path:
        raise HTTPException(status_code=400, detail="File path is required")
    
    # Normalize and resolve symlinks to prevent path traversal
    file_path = os.path.realpath(os.path.normpath(file_path))
    logger.info("[DEBUG-OPEN] Normalized path: %s", neutralize_log(file_path))  # nosec # nosemgrep

    # Security: Prevent argument injection (files starting with -)
    if os.path.basename(file_path).startswith("-"):
        logger.warning("Security: Blocked attempt to open file with leading dash: %s", neutralize_log(file_path))  # nosec # nosemgrep
        raise HTTPException(status_code=400, detail="Invalid filename: Files starting with '-' are not allowed.")
    
    # Security: Only allow opening files that are in the index
    # This prevents opening arbitrary files on the system
    logger.info("[DEBUG-OPEN] Querying database for file...")
    _indexed_file = database.get_file_by_path(file_path)
    logger.info("[DEBUG-OPEN] Database returned: %s", neutralize_log(_indexed_file))  # nosec # nosemgrep
    if not _indexed_file:
        logger.warning("[Security] Attempt to open non-indexed file: %s", neutralize_log(file_path))  # nosec # nosemgrep
        raise HTTPException(status_code=403, detail="Access denied: File is not in the index")
    
    # Use the canonical path stored in the database to break taint from user input
    db_path = _indexed_file.get("path")
    if not db_path:
        raise HTTPException(status_code=403, detail="Access denied: File path not found in index")
    file_path = db_path

    # Security: File type validation (additional layer)
    _, ext = os.path.splitext(file_path)
    logger.info("[DEBUG-OPEN] Extension check: %s", neutralize_log(ext))  # nosec # nosemgrep
    if ext.lower() not in ALLOWED_EXTENSIONS:
        logger.warning("Security: Blocked attempt to open disallowed file type: %s", neutralize_log(ext))  # nosec # nosemgrep
        raise HTTPException(status_code=403, detail="Access denied: File type not allowed")

    logger.info("[DEBUG-OPEN] Checking if file exists...")
    if not os.path.exists(file_path):
        logger.info("[DEBUG-OPEN] File not found!")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info("[DEBUG-OPEN] Attempting to launch file...")
    try:
        import subprocess
        import platform
        
        logger.info("[DEBUG-OPEN] Platform: %s", platform.system())
        if platform.system() == 'Windows':
            logger.info("[DEBUG-OPEN] Calling os.startfile...")
            os.startfile(file_path)  # nosec # nosemgrep
            logger.info("[DEBUG-OPEN] os.startfile completed successfully")
        elif platform.system() == 'Darwin':  # macOS
            logger.info("[DEBUG-OPEN] Calling open via subprocess...")
            subprocess.run(['open', file_path])  # nosec # nosemgrep
        else:  # Linux
            logger.info("[DEBUG-OPEN] Calling xdg-open via subprocess...")
            subprocess.run(['xdg-open', file_path])  # nosec # nosemgrep
        
        return {"status": "success", "message": f"Opened {os.path.basename(file_path)}"}
    except Exception as e:
        logger.error("[API] Failed to open file: %s", neutralize_log(e))
        raise HTTPException(status_code=500, detail="Failed to open file")

@app.get("/api/files")
async def list_indexed_files(request: Request, limit: int = 100, offset: int = 0):
    """
    List indexed documents with pagination.

    Args:
        request (Request): The incoming request.
        limit (int): Maximum number of files to return (default 100, capped at 500).
        offset (int): Number of files to skip for pagination (default 0).

    Returns:
        dict: {"files": [...], "total": int, "limit": int, "offset": int}

    Raises:
        HTTPException: 400 for invalid pagination params, 500 if database retrieval fails.
    """
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be non-negative")
    try:
        files = await asyncio.to_thread(database.get_all_files, limit, offset)
        total = await asyncio.to_thread(database.count_files)
        return {"files": files, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/preview")
async def preview_file(path: str, request: Request, chars: int = 2000):
    """
    Return a text preview of an indexed file (path traversal protected).

    Query params:
        path (str): The indexed file path.
        chars (int): Maximum characters to return (default 2000, max 10000).
    """
    if not path:
        raise HTTPException(status_code=400, detail="path is required")

    # Normalize and verify the file is indexed
    real_path = os.path.realpath(os.path.normpath(path))
    file_info = database.get_file_by_path(real_path)
    if not file_info:
        raise HTTPException(status_code=403, detail="Access denied: file is not in the index")
    # Use the canonical path stored in the database to break taint from user input
    real_path = file_info.get("path") or real_path

    if not os.path.exists(real_path):
        raise HTTPException(status_code=404, detail="File not found")

    chars = min(max(chars, 1), 10000)

    try:
        from backend.file_processing import extract_text
        text = await asyncio.to_thread(extract_text, real_path)
        if not text:
            raise HTTPException(status_code=422, detail="Could not extract text from this file")
        return {
            "path": real_path,
            "filename": file_info.get("filename"),
            "file_type": file_info.get("file_type"),
            "preview": text[:chars],
            "total_chars": len(text),
            "truncated": len(text) > chars,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[API] File preview error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to read file preview")


@app.get("/api/folders/history")
async def get_folder_history(request: Request):
    """
    Retrieve the history of successfully indexed folders.

    Args:
        request (Request): The incoming request.

    Returns:
        List[str]: A list of folder paths.

    Raises:
        HTTPException: 500 if database operation fails.
    """
    try:
        # User requested: ONLY show 100% indexed folders in history
        history = database.get_folder_history(indexed_only=True)
        return [item['path'] for item in history]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/folders/history")
async def clear_folder_history(request: Request):
    """
    Remove all folder entries from the indexing history.

    Args:
        request (Request): The incoming request.

    Returns:
        dict: Success status and count of deleted items.

    Raises:
        HTTPException: 500 if database operation fails.
    """
    try:
        count = database.clear_folder_history()
        return {"status": "success", "message": f"Cleared {count} folder history items", "deleted_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/folders/history/item")
async def delete_folder_history_item(request: dict, req: Request):
    """
    Remove a single folder from the indexing history.

    Args:
        request (dict): Body containing the 'path' to remove.
        req (Request): The incoming request.

    Returns:
        dict: Success status and message.

    Raises:
        HTTPException: 400 if path missing, 404 if path missing in DB, 500 on error.
    """
    path = request.get('path', '')
    if not path:
        raise HTTPException(status_code=400, detail="Path is required")
    
    try:
        success = database.delete_folder_history_item(path)
        if success:
            return {"status": "success", "message": "Folder removed from history"}
        raise HTTPException(status_code=404, detail="Folder not found in history")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validate-path")
async def validate_path(body: dict, request: Request):
    """
    Validate a system path and count supported file types for indexing.

    Args:
        body (dict): Body containing the 'path' to validate.
        request (Request): The incoming request.

    Returns:
        dict: A dictionary with 'valid' status and 'file_count'.
    """
    path = body.get('path', '')
    if not path:
        return {"valid": False, "error": "Path is required"}

    # Normalize to prevent path traversal tricks
    import pathlib
    try:
        normalized = str(pathlib.Path(path).resolve())
    except (ValueError, OSError):
        return {"valid": False, "error": "Invalid path"}

    # Reject system directories
    _FORBIDDEN_PREFIXES = [
        "/etc", "/sys", "/proc", "/dev", "/boot", "/run",
        "C:\\Windows", "C:\\System32", "C:\\SysWOW64",
    ]
    if any(normalized.startswith(p) for p in _FORBIDDEN_PREFIXES):
        return {"valid": False, "error": "Cannot index system directories"}

    if not os.path.exists(normalized):
        return {"valid": False, "error": "Path does not exist"}

    if not os.path.isdir(normalized):
        return {"valid": False, "error": "Path is not a directory"}

    path = normalized
    
    # Count supported files
    supported_extensions = {'.txt', '.pdf', '.docx', '.xlsx', '.pptx'}
    file_count = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_extensions:
                file_count += 1
    
    return {"valid": True, "file_count": file_count}

@app.get("/api/index/status")
async def get_indexing_status(request: Request):
    """
    Get the current progress of the background indexing task.

    Args:
        request (Request): The incoming request.

    Returns:
        dict: Indexing status including progress percentage and current file.
    """
    return indexing_status

@app.post("/api/index")
@limiter.limit("5/minute")
async def trigger_indexing(background_tasks: BackgroundTasks, request: Request):
    """
    Manually trigger the background indexing process for configured folders.

    Args:
        background_tasks (BackgroundTasks): FastAPI background task manager.
        request (Request): The incoming request.

    Returns:
        dict: Status 'accepted' and a start message.

    Raises:
        HTTPException: 400 if indexing is already running or no folders are configured.
    """
    global indexing_status
    if indexing_status["running"]:
        raise HTTPException(status_code=400, detail="Indexing already in progress")
        
    config = load_config()
    folders_str = config.get('General', 'folders', fallback='')
    folders = [f.strip() for f in folders_str.split(',') if f.strip()]
    
    if not folders:
        # Fallback to old single folder
        folder = config.get('General', 'folder', fallback='')
        if folder:
            folders = [folder]
            
    if not folders:
        raise HTTPException(status_code=400, detail="No folders configured for indexing")
    
    # Initialize status
    indexing_status = {
        "running": True,
        "progress": 0,
        "current_file": "Scanning folders...",
        "total_files": 0,
        "processed_files": 0,
        "error": None
    }
    
    background_tasks.add_task(run_indexing, config, folders)
    return {"status": "accepted", "message": "Indexing started in background"}

def indexing_progress_callback(current, total, message=None):
    """
    Callback function to update the global indexing progress state.

    Args:
        current (int): Number of files processed or current progress value.
        total (int): Total number of files or maximum progress value.
        message (Optional[str]): Status message about the current task.
    """
    global indexing_status
    if message:
        indexing_status["current_file"] = message # Reuse current_file field for generic status message

    indexing_status["processed_files"] = current
    indexing_status["total_files"] = total
    # If 0-100 scale is passed directly as current, respect it
    if total == 100 and current > 1:
        indexing_status["progress"] = current
    else:
        indexing_status["progress"] = int((current / total) * 100) if total else 0

    # Debug log
    if indexing_status["progress"] % 10 == 0 or message:
        logger.info(f"Indexing Progress: {indexing_status['progress']}% - {indexing_status['current_file']}")

    # Broadcast to WebSocket clients. This callback runs in a worker thread,
    # so schedule the coroutine onto the main loop captured at startup
    # (asyncio.get_event_loop()/create_task are invalid from other threads).
    if _main_event_loop is not None and not _main_event_loop.is_closed():
        try:
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast({
                "type": "indexing_progress",
                "percent": indexing_status["progress"],
                "current_file": indexing_status.get("current_file", ""),
                "total": total,
            }), _main_event_loop)
        except RuntimeError:
            pass  # loop shutting down

@app.get("/api/agent/chat")
async def agent_chat(query: str, request: Request):
    """
    Stream AI agent's internal thoughts and final grounded answer.

    This endpoint uses a ReAct agent loop to iteratively search the 
    knowledge base until it finds enough information to answer the query.

    Args:
        query (str): The user's question.
        request (Request): The incoming request object.

    Returns:
        StreamingResponse: A stream of JSON events (type: 'thought' or 'answer').
    """
    global index, docs, tags, index_summaries, cluster_summaries, cluster_map
    
    # Check index
    if not index:
        # We can't use HTTPException in streaming response easily, yield error
        async def yield_error():
             """Inner helper to yield an error event."""
             yield f"data: {json.dumps({'type': 'error', 'content': 'Index not loaded'})}\n\n"
        return StreamingResponse(yield_error(), media_type="text/event-stream")

    # Construct global state for the agent
    config = load_config()
    global_state = {
        'index': index,
        'docs': docs,
        'tags': tags,
        'index_summaries': index_summaries,
        'cluster_summaries': cluster_summaries,
        'cluster_map': cluster_map,
        'bm25': bm25,
        'config': config
    }
    
    from backend.agent import ReActAgent
    agent = ReActAgent(global_state)
    
    async def event_generator():
        """Inner helper to generate and yield agent events."""
        try:
            async for event in agent.stream_chat(query):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error("[Agent] Stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': 'An error occurred processing your request'})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

def run_indexing(config, folders):
    """
    Background worker function that orchestrates the document indexing pipeline.

    Steps:
    1. Extracts text from supported files in folders.
    2. Generates embeddings and builds FAISS index.
    3. Persists metadata and index to disk.
    4. Updates in-memory search objects and status.

    Args:
        config (configparser.ConfigParser): The application configuration.
        folders (List[str]): List of folder paths to index.
    """
    global index, docs, tags, index_summaries, cluster_summaries, cluster_map, bm25, indexing_status
    
    provider = config.get('LocalLLM', 'provider', fallback='openai')
    api_key = config.get('APIKeys', 'openai_api_key', fallback=None)
    model_path = config.get('LocalLLM', 'model_path', fallback=None)
    
    try:
        logger.info(f"Starting indexing for folders: {folders}")

        # Resolve the active embedding client via settings (falls back to legacy if not configured)
        from backend.settings import get_active_embedding_client
        embedding_client = get_active_embedding_client(app)
        _model_name = getattr(embedding_client, 'model_name', None) or getattr(embedding_client, 'model', 'unknown')

        # Unpack first 7 values
        res = create_index(
            folders, provider, api_key, model_path,
            progress_callback=indexing_progress_callback,
            embedding_client=embedding_client,
            previous_index_path=INDEX_PATH,
        )
        new_index, new_docs, new_tags, new_summ_index, new_summ_docs, new_cluster_map, new_bm25 = res[:7]
        if new_index:
            _embedding_dim = int(new_index.d)
            save_index(
                new_index, new_docs, new_tags, INDEX_PATH,
                new_summ_index, new_summ_docs, new_cluster_map, new_bm25,
                model_name=_model_name, embedding_dim=_embedding_dim,
            )
            with _index_lock:
                index, docs, tags = new_index, new_docs, new_tags
                index_summaries, cluster_summaries, cluster_map = new_summ_index, new_summ_docs, new_cluster_map
                bm25 = new_bm25
            # Refresh index meta so search immediately uses the new model/dim
            app.state.index_meta = res[7] if len(res) > 7 else {
                'model_name': _model_name, 'embedding_dim': _embedding_dim,
            }

            logger.info("Indexing completed successfully.")
            indexing_status["running"] = False
            indexing_status["progress"] = 100
            indexing_status["current_file"] = "Complete"
            
            # Mark folders as successfully indexed for history
            for folder in folders:
                database.mark_folder_indexed(folder)
        else:
            logger.error("Indexing failed or no documents found.")
            indexing_status["running"] = False
            indexing_status["error"] = "No documents found or processed"
    except Exception as e:
        import traceback
        logger.error(f"Error during indexing: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        indexing_status["running"] = False
        indexing_status["error"] = str(e)

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time indexing and download progress updates."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; server pushes events via ws_manager.broadcast()
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/api/graph")
async def get_knowledge_graph(request: Request):
    """
    Retrieve the Knowledge Graph of indexed files and keywords.
    """
    try:
        graph_data = await asyncio.to_thread(database.get_graph)
        return graph_data
    except Exception as e:
        logger.error(f"Error retrieving knowledge graph: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve knowledge graph")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Duplicate verify_local_request removed.
