from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import StreamingResponse
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import uvicorn
import os
import time
import configparser

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

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx', '.txt'}



limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app = FastAPI()
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Register embedding settings router
from backend.settings import router as embedding_router
app.include_router(embedding_router)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:5175", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
index = None
docs = []
tags = []
index_summaries = None
cluster_summaries = []
cluster_map = None
bm25 = None

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
        dict: A dictionary with the status 'ok'.
    """
    return {"status": "ok"}

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
    return config

def save_config_file(config):
    """
    Save the provided configuration to config.ini.

    Args:
        config (configparser.ConfigParser): The configuration object to save.
    """
    with open(CONFIG_PATH, 'w') as configfile:
        config.write(configfile)

# Initialize index on startup if available
@app.on_event("startup")
async def startup_event():
    """
    Handle application startup events.
    
    Initializes the database, seeds the embedding configuration, and
    triggers the background loading of the search index.
    """
    logger.info("Application starting up...")
    database.init_database()
    # Seed embedding config cache from config.ini
    from backend.settings import seed_app_state
    seed_app_state(app)
    # Load index in background to not block health checks
    asyncio.create_task(load_initial_index())

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
            index, docs, tags, index_summaries, cluster_summaries, cluster_map, bm25 = res[:7]
            _index_meta = res[7] if len(res) > 7 else {}
            logger.info(
                "Loaded existing index successfully. "
                "Model: %s, dim: %s",
                _index_meta.get('model_name', 'unknown'),
                _index_meta.get('embedding_dim', '?'),
            )
        except Exception as e:
            logger.error(f"Error loading index: {e}")
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
        raise HTTPException(status_code=500, detail=str(e))

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
    query: str
    context: Optional[List[str]] = None

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
        "openai_api_key": config.get('APIKeys', 'openai_api_key', fallback=''),
        "gemini_api_key": config.get('APIKeys', 'gemini_api_key', fallback=''),
        "anthropic_api_key": config.get('APIKeys', 'anthropic_api_key', fallback=''),
        "grok_api_key": config.get('APIKeys', 'grok_api_key', fallback=''),
        "local_model_path": config.get('LocalLLM', 'model_path', fallback=''),
        "provider": config.get('LocalLLM', 'provider', fallback='openai'),
        "tensor_split": config.get('LocalLLM', 'tensor_split', fallback=None),
        "query_rewriting": config.getboolean('AdvancedRAG', 'query_rewriting', fallback=False),
        "cross_encoder_reranking": config.getboolean('AdvancedRAG', 'cross_encoder_reranking', fallback=False),
        "reranker_model": config.get('AdvancedRAG', 'reranker_model', fallback='cross-encoder/ms-marco-MiniLM-L-6-v2')
    }

@app.post("/api/config")
async def update_config(config_data: ConfigModel, request: Request):
    """
    Update the application configuration.

    Args:
        config_data (ConfigModel): The new configuration data.
        request (Request): The incoming request.

    Returns:
        dict: Success status and message.
    """
    config = configparser.ConfigParser()
    config['General'] = {
        'folders': ','.join(config_data.folders),
        'auto_index': str(config_data.auto_index)
    }
    config['APIKeys'] = {
        'openai_api_key': config_data.openai_api_key or '',
        'gemini_api_key': config_data.gemini_api_key or '',
        'anthropic_api_key': config_data.anthropic_api_key or '',
        'grok_api_key': config_data.grok_api_key or ''
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
    save_config_file(config)
    
    # Save folders to history
    try:
        if config_data.folders:
            for folder in config_data.folders:
                database.add_folder_to_history(folder)
    except Exception as e:
        print(f"Failed to update folder history: {e}")
        
    return {"status": "success", "message": "Configuration saved"}

@app.post("/api/search")
async def search_files(request: SearchRequest, req: Request):
    """
    Perform a semantic search on the indexed documents.

    This endpoint supports:
    1. Vanilla semantic search (FAISS + BM25 hybrid).
    2. Agentic mode (if enabled in config).
    3. Advanced RAG (query rewriting, reranking).
    4. SSE Streaming for agentic responses.

    Args:
        request (SearchRequest): The search query and optional context.
        req (Request): The incoming request object.

    Returns:
        SearchResponse or StreamingResponse: The search results or an AI-generated answer.

    Raises:
        HTTPException: 400 if index not loaded, 409 if embedding dimension mismatch.
    """
    global index, docs, tags, index_summaries, cluster_summaries, cluster_map, bm25
    
    if not index:
        raise HTTPException(status_code=400, detail="Index not loaded. Please configure and index a folder first.")

    print(f"\n[API] POST /api/search - Query: <redacted>")

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
            print("[API] Running in AGENTIC mode.")
            from backend.agent import ReActAgent
            agent = ReActAgent(provider, api_key, {
                'index': index, 'docs': docs, 'tags': tags, 'config': config,
                'index_summaries': index_summaries, 'cluster_summaries': cluster_summaries,
                'cluster_map': cluster_map, 'bm25': bm25
            })
            return StreamingResponse(agent.stream_chat(request.query), media_type="text/event-stream")

        model_path = config.get('LocalLLM', 'model_path', fallback=None)
        tensor_split_str = config.get('LocalLLM', 'tensor_split', fallback=None)
        tensor_split = None
        if tensor_split_str:
            try:
                tensor_split = [float(x) for x in tensor_split_str.split(',')]
            except:
                pass

        # Run Search — use the active embedding client from settings state
        from backend.search import EmbeddingDimensionMismatchError
        try:
            results, _context_snippets = search(
                request.query, index, docs, tags,
                get_active_embedding_client(req.app),
                index_summaries, cluster_summaries, cluster_map, bm25
            )
        except EmbeddingDimensionMismatchError as dim_err:
            raise HTTPException(
                status_code=409,
                detail=str(dim_err),
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
        
        for result in results:
            faiss_idx = result.get('faiss_idx')
            
            # Use file info from search result first (it comes from FAISS doc metadata)
            file_path = result.get('file_path')
            file_name = result.get('file_name')
            
            # If not in search result, use batched lookup map
            if not file_path and faiss_idx in file_lookup_map:
                file_info = file_lookup_map[faiss_idx]
                file_path = file_info.get('path')
                file_name = file_info.get('filename')
            
            # OPTIMIZATION: Use fast summary for all results to avoid blocking
            summary = summarize(result['document'], provider, api_key, model_path, question=request.query)
            
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
        
        # Return results immediately - AI Answer will be streamed via separate endpoint
        active_model_name = provider.capitalize()
        if provider == 'local' and model_path:
            active_model_name = os.path.basename(model_path).replace(".gguf", "").replace("-", " ")
        
        # Save to search history
        execution_time_ms = int((time.time() - start_time) * 1000)
        database.add_search_history(request.query, len(processed_results), execution_time_ms)
            
        return SearchResponse(
            results=processed_results,
            ai_answer="", # Empty for immediate return
            active_model=active_model_name
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stream-answer")
async def stream_answer_endpoint(request: SearchRequest, req: Request):
    """
    Stream the AI answer for a given search query.

    If context is provided in the request, it uses that. Otherwise, it 
    re-runs the semantic search to gather relevant snippets and then 
    streams tokens from the selected LLM provider.

    Args:
        request (SearchRequest): The search query and optional context snippets.
        req (Request): The incoming request object.

    Returns:
        StreamingResponse: A server-sent event stream of AI response tokens.
    """
    global index, docs, tags, index_summaries, cluster_summaries, cluster_map, bm25

    if not index:
         return StreamingResponse(iter(["Error: Index not loaded."]), media_type="text/event-stream")

    config = load_config()
    provider = config.get('LocalLLM', 'provider', fallback='openai')
    api_key = config.get('APIKeys', 'openai_api_key', fallback=None)
    if provider == 'gemini':
        api_key = config.get('APIKeys', 'gemini_api_key', fallback=api_key)
    elif provider == 'anthropic':
        api_key = config.get('APIKeys', 'anthropic_api_key', fallback=api_key)
    elif provider == 'grok':
        api_key = config.get('APIKeys', 'grok_api_key', fallback=api_key)

    model_path = config.get('LocalLLM', 'model_path', fallback=None)
    tensor_split_str = config.get('LocalLLM', 'tensor_split', fallback=None)
    tensor_split = None
    if tensor_split_str:
        try:
            tensor_split = [float(x) for x in tensor_split_str.split(',')]
        except:
            pass


    final_context_snippets = []

    if request.context:
        print(f"[API] Using provided context ({len(request.context)} snippets) for streaming answer")
        final_context_snippets = request.context
    else:
        # Re-run search to get context
        results, context_snippets = search(
            request.query, index, docs, tags,
            get_embeddings(provider, api_key, model_path),
            index_summaries, cluster_summaries, cluster_map, bm25
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

        # Prepare context
        for result in results:
             # Use fast fallback summary for streaming context (no new LLM calls)
             summary = summarize(result['document'], provider, api_key, model_path, question=request.query)

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

    async def generate():
        for token in stream_ai_answer(context_text, request.query, provider, api_key, model_path, tensor_split):
            yield token

    return StreamingResponse(generate(), media_type="text/event-stream")

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
async def open_file(request: dict, req: Request, _=Depends(verify_local_request)):
    """
    Open a file using the system's default application.

    Security Measures:
    - Only allowed from localhost (via Depends).
    - Prevents argument injection (leading dashes).
    - Only allows opening files verified to be in the index.
    - Whitelists file extensions (ALLOWED_EXTENSIONS).

    Args:
        request (dict): Body containing the 'path' of the file to open.
        req (Request): The incoming request.
        _ (Depends): Security dependency for local request verification.

    Returns:
        dict: Success status and message.

    Raises:
        HTTPException: 400 if path missing/invalid, 403 if access denied, 404 if file missing, 500 on system error.
    """
    file_path = request.get('path', '')
    if not file_path:
        raise HTTPException(status_code=400, detail="File path is required")
    
    # Normalize path - fix mixed slashes from FAISS metadata
    file_path = os.path.normpath(file_path)

    # Security: Prevent argument injection (files starting with -)
    if os.path.basename(file_path).startswith("-"):
        logger.warning(f"Security: Blocked attempt to open file with leading dash: {file_path}")
        raise HTTPException(status_code=400, detail="Invalid filename: Files starting with '-' are not allowed.")
    
    # Security: Only allow opening files that are in the index
    # This prevents opening arbitrary files on the system
    if not database.get_file_by_path(file_path):
        logger.warning(f"Security: Attempt to open non-indexed file: {file_path}")
        raise HTTPException(status_code=403, detail="Access denied: File is not in the index")

    # Security: File type validation (additional layer)
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        logger.warning(f"Security: Blocked attempt to open disallowed file type: {ext}")
        raise HTTPException(status_code=403, detail="Access denied: File type not allowed")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        import subprocess
        import platform
        
        if platform.system() == 'Windows':
            os.startfile(file_path)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', file_path])
        else:  # Linux
            subprocess.run(['xdg-open', file_path])
        
        return {"status": "success", "message": f"Opened {os.path.basename(file_path)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open file: {str(e)}")

@app.get("/api/files")
async def list_indexed_files(request: Request):
    """
    List all documents currently in the database.

    Args:
        request (Request): The incoming request.

    Returns:
        List[dict]: Metadata for all indexed files.

    Raises:
        HTTPException: 500 if database retrieval fails.
    """
    try:
        files = await asyncio.to_thread(database.get_all_files)
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
async def validate_path(request: dict, req: Request):
    """
    Validate a system path and count supported file types for indexing.

    Args:
        request (dict): Body containing the 'path' to validate.
        req (Request): The incoming request.

    Returns:
        dict: A dictionary with 'valid' status and 'file_count'.
    """
    path = request.get('path', '')
    if not path:
        return {"valid": False, "error": "Path is required"}
    
    if not os.path.exists(path):
        return {"valid": False, "error": "Path does not exist"}
    
    if not os.path.isdir(path):
        return {"valid": False, "error": "Path is not a directory"}
    
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
    elif filename := message: # Fallback if called with old signature (unlikely)
        indexing_status["current_file"] = f"Processing {filename}"
        
    indexing_status["processed_files"] = current
    indexing_status["total_files"] = total
    # If 0-100 scale is passed directly as current, respect it
    if total == 100 and current > 1: 
        indexing_status["progress"] = current
    else:
        indexing_status["progress"] = int((current / total) * 100)
    
    # Debug log
    if indexing_status["progress"] % 10 == 0 or message:
        logger.info(f"Indexing Progress: {indexing_status['progress']}% - {indexing_status['current_file']}")

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
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
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
        )
        new_index, new_docs, new_tags, new_summ_index, new_summ_docs, new_cluster_map, new_bm25 = res[:7]
        if new_index:
            _embedding_dim = int(new_index.d)
            save_index(
                new_index, new_docs, new_tags, INDEX_PATH,
                new_summ_index, new_summ_docs, new_cluster_map, new_bm25,
                model_name=_model_name, embedding_dim=_embedding_dim,
            )
            index, docs, tags = new_index, new_docs, new_tags
            index_summaries, cluster_summaries, cluster_map = new_summ_index, new_summ_docs, new_cluster_map
            bm25 = new_bm25
            
            logger.info("Indexing completed successfully.")
            indexing_status["running"] = False
            indexing_status["progress"] = 100
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
        logger.error(f"Error during indexing: {e}")
        indexing_status["running"] = False
        indexing_status["error"] = str(e)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Duplicate verify_local_request removed.
