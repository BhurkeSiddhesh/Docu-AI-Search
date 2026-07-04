import sys
import os
import logging
import hashlib
import re
import threading
import multiprocessing
from backend import database
from typing import Any, List, Dict, Optional, Tuple, Union


try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_anthropic import ChatAnthropic
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
except ImportError:
    # These will be handled individually in functions if needed,
    # but having them here as None allows tests to patch them.
    OpenAIEmbeddings = None
    ChatOpenAI = None
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None
    ChatAnthropic = None
    HuggingFaceEmbeddings = None
    HuggingFaceEndpointEmbeddings = None

_local_llm_lock = threading.Lock()


# DEBUG: Print environment info IMMEDIATELY
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info(f"LLM Integration Module Loading...")
    logger.info(f"Python Executable: {sys.executable}")
    logger.info(f"Python Path: {sys.path}")
    logger.info(f"CWD: {os.getcwd()}")
except Exception as e:
    logger.error(f"{e}")

# IMPORTS FIXED: Lazy loading to prevent startup crashes
# from langchain_huggingface import HuggingFaceEmbeddings

# Llama import moved to get_local_llm to prevent top-level blocking

# Cache for loaded models
_embeddings_cache = {}
_llm_cache = {}
_llm_client_cache = {}
# Serializes sentence-transformers model construction: two threads building
# the same model concurrently (startup warmup + first search/index) trip
# torch's meta-tensor initialization and fail.
_embedding_client_lock = threading.Lock()

def get_embeddings(provider: str, api_key: str = None, model_path: str = None) -> Any:
    """
    Returns an embeddings model instance based on the provider.
    
    Caches the model instance to avoid redundant reloads.

    Args:
        provider (str): The embedding provider (e.g., 'openai', 'gemini', 'local').
        api_key (str, optional): API key for cloud providers.
        model_path (str, optional): Path to a local model (legacy/custom).

    Returns:
        Any: An instance of the requested embeddings model (e.g., OpenAIEmbeddings).
    """
    cache_key = f"{provider}:{api_key or ''}"

    if cache_key in _embeddings_cache:
        return _embeddings_cache[cache_key]

    logger.info(f"Loading embeddings for provider: {provider}")
    # Serialize sentence-transformers construction (torch meta-tensor race)
    with _embedding_client_lock:
        if cache_key in _embeddings_cache:
            return _embeddings_cache[cache_key]
        return _load_embeddings_unlocked(provider, api_key, cache_key)

def _load_embeddings_unlocked(provider: str, api_key: str, cache_key: str) -> Any:
    """Constructs and caches the legacy embeddings client. Caller holds the lock."""


    if provider == 'openai' and api_key:
        embeddings = OpenAIEmbeddings(api_key=api_key)
    elif provider == 'gemini' and api_key:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    elif provider == 'grok' and api_key:
        # Grok typically uses OpenAI compatible API, but for embeddings they might not have a dedicated endpoint yet
        # or it might be compatible. For safety, we can fallback to local or OpenAI if specified.
        # Assuming Grok users might want local embeddings to save cost/latency if not specified otherwise.
        # But actually, let's just use HuggingFace local for everything else to keep it simple and free.
        # Unless user specifically wants cloud embeddings.
        logger.info("Using local embeddings for Grok provider.")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        # Default / Local
        logger.info("Loading local embeddings (HuggingFace)...")
        if HuggingFaceEmbeddings is None:
            logger.error("HuggingFaceEmbeddings not available")
            return None

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    
    logger.info("Embeddings loaded!")
    _embeddings_cache[cache_key] = embeddings
    return embeddings

# ---------------------------------------------------------------------------
# Embedding Factory (Factory Pattern)
# ---------------------------------------------------------------------------

# Default must stay CPU-friendly: MiniLM embeds ~1000 chunks/min on a laptop CPU,
# while larger instruct models (e.g. gte-Qwen2-1.5B) take seconds per chunk and
# multi-GB downloads. Heavier models remain available as explicit opt-in presets.
_DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Curated open-source embedding presets surfaced in Settings.
EMBEDDING_PRESETS = [
    {
        "id": "fast",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "label": "Fast (MiniLM-L6)",
        "dim": 384,
        "size": "~90 MB",
        "description": "Best speed on CPU. Great default for most folders.",
    },
    {
        "id": "balanced",
        "model_name": "BAAI/bge-small-en-v1.5",
        "label": "Balanced (BGE-small)",
        "dim": 384,
        "size": "~130 MB",
        "description": "Stronger semantics than MiniLM, still fast on CPU.",
    },
    {
        "id": "quality",
        "model_name": "BAAI/bge-base-en-v1.5",
        "label": "Quality (BGE-base)",
        "dim": 768,
        "size": "~440 MB",
        "description": "Higher accuracy; ~3x slower indexing than Fast.",
    },
    {
        "id": "max",
        "model_name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "label": "Max (GTE-Qwen2 1.5B)",
        "dim": 1536,
        "size": "~6 GB",
        "description": "Top retrieval quality but very slow without a GPU.",
    },
]

# Cache embedding clients: constructing HuggingFaceEmbeddings reloads the model
# from disk (seconds), and the search path resolves a client on every request.
# Guarded by _embedding_client_lock (defined with the caches above).
_embedding_client_cache = {}

def get_embedding_client(provider_type: str, model_name: str = None, api_key: str = None) -> Any:
    """
    Factory that returns a LangChain embedding object for the requested provider.

    Args:
        provider_type (str): One of:
            'local'           – HuggingFaceEmbeddings (runs on-device, no key needed).
            'huggingface_api' – HuggingFaceEndpointEmbeddings via the Inference API.
            'commercial_api'  – OpenAI or Google Gemini, selected by model_name.
        model_name (str, optional): The embedding model identifier.
            - 'local':           defaults to _DEFAULT_LOCAL_EMBEDDING_MODEL.
            - 'huggingface_api': the repo_id (e.g. 'BAAI/bge-large-en-v1.5').
            - 'commercial_api':  any string containing 'gpt' / 'text-embedding' →
                                 OpenAI; anything containing 'gemini' / 'embedding-001'
                                 → Google; raises ValueError otherwise.
        api_key (str, optional): Required for 'huggingface_api' and 'commercial_api'.

    Returns:
        Any: A LangChain embeddings object ready to call .embed_documents() / .embed_query().

    Raises:
        ValueError: On unsupported provider_type or unrecognised commercial model.
        ImportError: If the required LangChain package is not installed.
    """
    provider_type = provider_type.strip().lower()

    cache_key = (provider_type, model_name or '', api_key or '')
    if cache_key in _embedding_client_cache:
        return _embedding_client_cache[cache_key]

    # ------------------------------------------------------------------ local
    if provider_type == 'local':
        if HuggingFaceEmbeddings is None:
            raise ImportError(
                "langchain-huggingface is not installed. "
                "Run: pip install langchain-huggingface"
            )
        resolved_model = model_name or _DEFAULT_LOCAL_EMBEDDING_MODEL
        with _embedding_client_lock:
            if cache_key in _embedding_client_cache:
                return _embedding_client_cache[cache_key]
            logger.info(f"[EmbeddingFactory] local -> {resolved_model}")
            client = HuggingFaceEmbeddings(
                model_name=resolved_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
            )
            _embedding_client_cache[cache_key] = client
            return client

    # -------------------------------------------------------- huggingface_api
    if provider_type == 'huggingface_api':
        if HuggingFaceEndpointEmbeddings is None:
            raise ImportError(
                "langchain-huggingface is not installed. "
                "Run: pip install langchain-huggingface"
            )
        if not api_key:
            raise ValueError("'huggingface_api' requires an api_key (HuggingFace token).")
        if not model_name:
            raise ValueError("'huggingface_api' requires a model_name (repo_id).")
        logger.info(f"[EmbeddingFactory] huggingface_api -> {model_name}")
        client = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=api_key,
            repo_id=model_name,
        )
        _embedding_client_cache[cache_key] = client
        return client

    # --------------------------------------------------------- commercial_api
    if provider_type == 'commercial_api':
        if not api_key:
            raise ValueError("'commercial_api' requires an api_key.")
        if not model_name:
            raise ValueError("'commercial_api' requires a model_name.")

        model_lower = model_name.lower()

        # OpenAI: model names like 'text-embedding-3-small', 'text-embedding-ada-002', 'gpt-…'
        if any(kw in model_lower for kw in ('text-embedding', 'gpt')):
            if OpenAIEmbeddings is None:
                raise ImportError(
                    "langchain-openai is not installed. "
                    "Run: pip install langchain-openai"
                )
            logger.info(f"[EmbeddingFactory] commercial_api/OpenAI -> {model_name}")
            client = OpenAIEmbeddings(model=model_name, api_key=api_key)
            _embedding_client_cache[cache_key] = client
            return client

        # Google Gemini: model names like 'models/embedding-001', 'gemini-…'
        if any(kw in model_lower for kw in ('gemini', 'embedding-001')):
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError(
                    "langchain-google-genai is not installed. "
                    "Run: pip install langchain-google-genai"
                )
            logger.info(f"[EmbeddingFactory] commercial_api/Gemini -> {model_name}")
            client = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
            )
            _embedding_client_cache[cache_key] = client
            return client

        raise ValueError(
            f"Unrecognised commercial model_name '{model_name}'. "
            "Use a name containing 'text-embedding' / 'gpt' for OpenAI, "
            "or 'gemini' / 'embedding-001' for Google."
        )

    raise ValueError(
        f"Unknown provider_type '{provider_type}'. "
        "Choose from: 'local', 'huggingface_api', 'commercial_api'."
    )


def get_local_llm(model_path: str, tensor_split: List[float] = None) -> Any:
    """
    Load and cache the GGUF model directly with LlamaCpp.

    This function implements "blazing fast" inference settings for local models, 
    including GPU offloading and Flash Attention.

    Args:
        model_path (str): Absolute or relative path to the GGUF model file.
        tensor_split (List[float], optional): Distribution of model across multiple GPUs.

    Returns:
        Any: An initialized Llama instance, or None if loading fails.
    """
    if Llama is None:
        logger.warning("llama_cpp not installed")
        return None

        
    if not model_path or not os.path.exists(model_path):
        logger.info(f"Model not found at {model_path}")
        return None

    if model_path in _llm_cache:
        return _llm_cache[model_path]

    logger.info(f"Loading Local LLM from {model_path}...")
    try:
        # Load with reasonable defaults for CPU inference
        # Advanced performance tuning for "blazing fast" inference
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,           # Context window
            n_threads=max(multiprocessing.cpu_count() - 2, 1), # Use more cores for prompt processing
            n_threads_batch=multiprocessing.cpu_count(),       # Max threads for batch processing
            n_gpu_layers=-1,      # Full GPU offload
            n_batch=512,          # Batch size for prompt processing
            f16_kv=True,          # Use FP16 for KV cache (faster/less VRAM)
            flash_attn=True,      # Enable Flash Attention (Blazing fast sequences)
            offload_kqv=True,     # Offload KQV to GPU
            tensor_split=tensor_split,
            verbose=False         # Disable verbose logs for cleaner console unless debugging
        )
        # Prompt prefix cache: consecutive calls sharing a prompt prefix
        # (static system prompt, or a follow-up question over the same
        # document context) restore the KV state instead of re-evaluating
        # the whole prompt — the dominant latency cost on CPU.
        try:
            from llama_cpp import LlamaRAMCache
            cache_bytes = int(os.getenv("LLAMA_CACHE_BYTES", str(512 * 1024 * 1024)))
            llm.set_cache(LlamaRAMCache(capacity_bytes=cache_bytes))
            logger.info(f"Prompt prefix cache enabled ({cache_bytes // (1024*1024)} MB LlamaRAMCache).")
        except Exception as cache_err:
            logger.info(f"Prompt prefix cache unavailable: {cache_err}")
        _llm_cache[model_path] = llm
        logger.info("Local LLM loaded!")
        return llm
    except Exception as e:
        logger.error(f"Failed to load Local LLM: {e}")
        return None

def warmup_local_model(model_path: str, tensor_split: List[float] = None) -> None:
    """
    Pre-load the local model into memory on startup.

    Args:
        model_path (str): Path to the GGUF model file.
        tensor_split (List[float], optional): Multi-GPU distribution.
    """
    if model_path:
        get_local_llm(model_path, tensor_split=tensor_split)


def _discover_local_gguf() -> Optional[str]:
    """
    Find a usable GGUF model in the models/ directory when none is configured.

    Prefers the largest model that still runs comfortably on CPU (≤ 3 GB —
    e.g. phi-2 / gemma-2b class); falls back to the smallest available so a
    fresh setup with only tinyllama still works.

    Returns:
        Optional[str]: Absolute path to a .gguf file, or None if none exist.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    if not os.path.isdir(models_dir):
        return None
    ggufs = []
    for name in os.listdir(models_dir):
        if name.lower().endswith('.gguf'):
            path = os.path.join(models_dir, name)
            try:
                ggufs.append((os.path.getsize(path), path))
            except OSError:
                continue
    if not ggufs:
        return None
    ggufs.sort()
    cpu_friendly = [g for g in ggufs if g[0] <= 3 * 1024 ** 3]
    pool = cpu_friendly or ggufs[:1]
    # Instruction-tuned models follow the answer format far better than base
    # models, so prefer them even when slightly smaller.
    instruct = [g for g in pool if re.search(r'instruct|chat|-it\b|-it[.-]', os.path.basename(g[1]), re.IGNORECASE)]
    return (instruct or pool)[-1][1]


def get_llm_client(provider: str, api_key: str = None, model_path: str = None, base_url: str = None) -> Any:
    """
    Returns a LangChain-compatible Chat Model or special path for local models.

    Args:
        provider (str): 'openai', 'gemini', 'anthropic', 'grok', or 'local'.
        api_key (str, optional): API key for cloud providers.
        model_path (str, optional): Filesystem path for local GGUF models.

    Returns:
        Any: A LangChain chat client, or a "LOCAL:path" string for local models, 
             or None if configuration is invalid.
    """
    cache_key = f"{provider}:{api_key or ''}:{model_path or ''}"
    if cache_key in _llm_client_cache:
        return _llm_client_cache[cache_key]

    client = None
    try:
        if provider == 'openai':
            if not api_key:
                logger.warning("OpenAI API Key missing")
                return None
            client = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.3)

        elif provider == 'gemini':
            if not api_key:
                logger.warning("Gemini API Key missing")
                return None
            client = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash", temperature=0.3)

        elif provider == 'anthropic':
            if not api_key:
                logger.warning("Anthropic API Key missing")
                return None
            client = ChatAnthropic(api_key=api_key, model="claude-3-haiku-20240307", temperature=0.3)

        elif provider == 'grok':
            if not api_key:
                logger.warning("Grok API Key missing")
                return None
            # Grok uses OpenAI-compatible endpoint
            client = ChatOpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                model="grok-beta",
                temperature=0.3
            )

        elif provider == 'local':
            # For local, we don't return a LangChain object because we are using llama-cpp-python directly
            # for better control over the 'create_completion' call in generate_ai_answer currently.
            if not model_path or not os.path.exists(model_path):
                model_path = _discover_local_gguf()
                if not model_path:
                    logger.warning("Local model path missing and no GGUF found in models/")
                    return None
                logger.info(f"[LLM] Auto-selected local model: {os.path.basename(model_path)}")
            client = "LOCAL:" + model_path

        elif provider in ('ollama', 'lmstudio', 'openai_compatible'):
            # Delegate to the external provider abstraction
            from backend.providers import get_provider as get_ext_provider
            try:
                ext_config = _build_external_provider_config(provider, base_url, model_path)
                ext_provider = get_ext_provider(provider, ext_config)
                # Return a marker to distinguish from LangChain clients
                client = f"EXTERNAL:{provider}"
                # Stash the provider instance for reuse in generate/stream
                _llm_client_cache[f"__ext_instance__{provider}"] = ext_provider
            except Exception as e:
                logger.error(f"Error initializing external provider '{provider}': {e}")
                return None

    except Exception as e:
        logger.error(f"Error initializing LLM client for {provider}: {e}")
        return None

    if client:
        _llm_client_cache[cache_key] = client
    return client


def _build_external_provider_config(provider: str, base_url_override: str = None, model_override: str = None) -> dict:
    """Read ExternalProviders section from config.ini for the given provider."""
    import configparser
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(BASE_DIR, 'config.ini')
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    if provider == 'ollama':
        return {
            "base_url": base_url_override or cfg.get('ExternalProviders', 'ollama_base_url', fallback='http://localhost:11434'),
            "model": model_override or cfg.get('ExternalProviders', 'external_model_name', fallback=''),
            "api_key": cfg.get('ExternalProviders', 'external_api_key', fallback=''),
        }
    else:  # lmstudio / openai_compatible
        return {
            "base_url": base_url_override or cfg.get('ExternalProviders', 'lmstudio_base_url', fallback='http://localhost:1234/v1'),
            "model": model_override or cfg.get('ExternalProviders', 'external_model_name', fallback=''),
            "api_key": cfg.get('ExternalProviders', 'external_api_key', fallback='lm-studio'),
        }

def generate_ai_answer(context: str, question: str, provider: str,
                       api_key: str = None, model_path: str = None,
                       tensor_split: List[float] = None, raw: bool = False,
                       system_instruction: str = None, stop: List[str] = None,
                       max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Generate a natural language answer using the selected provider.

    Args:
        context (str): Document snippets used as grounding for the answer. 
            Ignored if raw=True.
        question (str): The user's query or the full prompt (if raw=True).
        provider (str): 'openai', 'gemini', 'anthropic', 'local', etc.
        api_key (str, optional): Key for cloud providers.
        model_path (str, optional): Filesystem path for local models.
        tensor_split (List[float], optional): Distribution for multi-GPU loading.
        raw (bool): If True, bypasses standard RAG prompt wrapping.
        system_instruction (str, optional): System prompt override.
        stop (List[str], optional): List of stop sequences for generation.
        max_tokens (int): Maximum tokens allowed in response.
        temperature (float): Sampling temperature (0.0 to 1.0).

    Returns:
        str: The generated response text, or an error message.
    """
    # Get client
    client = get_llm_client(provider, api_key, model_path)
    if not client:
        return "Error: Could not initialize AI model. Check settings and API keys."

    # Default logic for non-raw (RAG) mode
    if not raw:
        system_prompt = system_instruction or """You are a precise document search assistant.
                CRITICAL: You must distinguish between similar names. If the question asks about 'Siddhesh', do NOT provide info about 'Siddharth'.
                Only answer based on the provided documents. Quote facts and reference file names."""

        # Prepare context
        # Truncate context to fit within model's context window (roughly)
        MAX_CONTEXT_CHARS = 10000
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "... [Truncated to fit context window]"

        user_content = f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer (cite specific details from the documents):"

        # Stops for RAG
        stop_seqs = stop or ["System:", "Question:", "Context:", "Documents:"]

    else:
        # RAW mode
        # 'question' is treated as the main content/prompt
        system_prompt = system_instruction # Can be None
        user_content = question
        stop_seqs = stop # Can be None

    try:
        # Handle External Providers (Ollama, LM Studio)
        if isinstance(client, str) and client.startswith("EXTERNAL:"):
            ext_provider = _llm_client_cache.get(f"__ext_instance__{provider}")
            if not ext_provider:
                return "Error: External provider not initialised. Check settings."
            return ext_provider.generate(
                user_content,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_seqs,
            )

        # Handle Local LLM (LlamaCpp)
        elif isinstance(client, str) and client.startswith("LOCAL:"):
            real_model_path = client.split("LOCAL:")[1]
            llm = get_local_llm(real_model_path, tensor_split=tensor_split)
            if not llm:
                return "Error: Local model failed to load."

            # Construct full prompt string for completion
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{user_content}"
            else:
                full_prompt = user_content

            with _local_llm_lock:
                output = llm.create_completion(
                    full_prompt,
                    max_tokens=max_tokens,
                    stop=stop_seqs,
                    echo=False,
                    temperature=temperature,
                    repeat_penalty=1.1
                )
            return output['choices'][0]['text'].strip()

        # Handle LangChain Clients (Cloud)
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            messages.append(HumanMessage(content=user_content))

            # Use bind for stop sequences if supported, otherwise just invoke
            if stop_seqs:
                 try:
                    response = client.bind(stop=stop_seqs).invoke(messages)
                 except Exception:
                    # Fallback if bind not supported
                    response = client.invoke(messages)
            else:
                 response = client.invoke(messages)

            return response.content.strip()

    except Exception as e:
        logger.error(f"Generation error ({provider}): {e}")
        return f"Error generating answer: {str(e)}"

def stream_ai_answer(context: str, question: str, provider: str,
                     api_key: str = None, model_path: str = None,
                     tensor_split: List[float] = None,
                     system_instruction: str = None, base_url: str = None) -> Any:
    """
    Generator that yields tokens for the AI answer.

    Uses a grounded RAG prompt and handles both local and cloud providers.

    Args:
        context (str): Document snippets for context.
        question (str): The user's query.
        provider (str): 'openai', 'gemini', 'anthropic', or 'local'.
        api_key (str, optional): Key for cloud providers.
        model_path (str, optional): Path for local GGUF models.
        tensor_split (List[float], optional): Distribution for multi-GPU.

    Yields:
        str: Incremental tokens of the generated answer.
    """
    # Get client
    client = get_llm_client(provider, api_key, model_path, base_url)
    if not client:
        yield "Error: Could not initialize AI model. Check settings and API keys."
        return

    system_prompt = system_instruction or """You are a precise document search assistant.
CRITICAL: You must distinguish between similar names. If the question asks about 'Siddhesh', do NOT provide info about 'Siddharth'.
Only answer based on the provided documents. Quote facts and reference file names."""

    user_content = f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer (cite specific details from the documents):"

    try:
        # Handle External Providers (Ollama, LM Studio)
        if isinstance(client, str) and client.startswith("EXTERNAL:"):
            ext_provider = _llm_client_cache.get(f"__ext_instance__{provider}")
            if not ext_provider:
                yield "Error: External provider not initialised. Check settings."
                return
            for token in ext_provider.stream(
                user_content,
                system_prompt=system_prompt,
                max_tokens=512,
                temperature=0.2,
            ):
                yield token
            return

        # Handle Local LLM (LlamaCpp)
        if isinstance(client, str) and client.startswith("LOCAL:"):
            real_model_path = client.split("LOCAL:")[1]
            llm = get_local_llm(real_model_path, tensor_split=tensor_split)
            if not llm:
                yield "Error: Local model failed to load."
                return

            # CPU prompt evaluation is the dominant cost (~30-60 tok/s on a
            # laptop): every 1000 chars of context adds seconds before the
            # first token. Keep the prompt tight.
            MAX_CONTEXT_CHARS = 4500
            if len(context) > MAX_CONTEXT_CHARS:
                context = context[:MAX_CONTEXT_CHARS] + "... [Truncated to fit context window]"
                user_content = f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer (cite specific details from the documents):"

            with _local_llm_lock:
                # Prefer the model's own chat template: instruct models then
                # emit a proper EOS, stopping early instead of rambling to the
                # token cap (faster AND cleaner answers).
                stream = None
                try:
                    stream = llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        max_tokens=320,
                        temperature=0.2,
                        repeat_penalty=1.1,
                        stream=True,
                    )
                    for output in stream:
                        delta = output['choices'][0].get('delta', {})
                        token = delta.get('content')
                        if token:
                            yield token
                    return
                except Exception as chat_err:
                    logger.info(f"[LLM] Chat template unavailable ({chat_err}); falling back to raw completion.")

                full_prompt = f"{system_prompt}\n\n{user_content}"
                stream = llm.create_completion(
                    full_prompt,
                    max_tokens=320,
                    stop=["System:", "Question:", "Context:", "Documents:", "\n\n\n"],
                    echo=False,
                    temperature=0.2,
                    repeat_penalty=1.1,
                    stream=True  # ENABLE STREAMING
                )

                for output in stream:
                    token = output['choices'][0]['text']
                    yield token

        # Handle LangChain Clients (Cloud)
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=user_content))

            for chunk in client.stream(messages):
                 yield chunk.content

    except Exception as e:
        logger.error(f"Streaming error ({provider}): {e}")
        yield f"Error generating answer: {str(e)}"

# -----------------------------------------------------------------------------
# Caching Wrappers
# -----------------------------------------------------------------------------

def compute_cache_key(query: str, context: str, model_id: str) -> tuple:
    """
    Returns SHA256 hashes for query and context.

    Used to uniquely identify the input to an AI model for caching purposes.

    Args:
        query (str): The search query or prompt.
        context (str): The document context provided to the model.
        model_id (str): Identifier for the specific model used.

    Returns:
        tuple: (query_hash, context_hash) as hex strings.
    """
    query_hash = hashlib.sha256(query.strip().lower().encode('utf-8')).hexdigest()
    # Normalize context by removing whitespace to ignore formatting changes
    context_norm = re.sub(r'\s+', ' ', context.strip())
    context_hash = hashlib.sha256(context_norm.encode('utf-8')).hexdigest()
    return query_hash, context_hash

def cached_generate_ai_answer(context: str, question: str, provider: str, 
                               api_key: str = None, model_path: str = None, 
                               tensor_split: List[float] = None) -> str:
    """
    Wrapper around generate_ai_answer that checks persistent cache first.

    This reduces API costs and improves response time for repeated queries 
    over the same context.

    Args:
        context (str): Document context for the prompt.
        question (str): User's query.
        provider (str): 'openai', 'gemini', 'anthropic', or 'local'.
        api_key (str, optional): Key for cloud providers.
        model_path (str, optional): Path for local GGUF models.
        tensor_split (List[float], optional): Distribution for multi-GPU loading.

    Returns:
        str: The AI's answer, either from cache or freshly generated.
    """
    # specific model ID for cache key
    if provider == 'local':
        model_id = f"local:{os.path.basename(model_path)}" if model_path else "local:unknown"
    else:
        model_id = provider

    query_hash, context_hash = compute_cache_key(question, context, model_id)
    
    # 1. Check Cache
    cached_text = database.get_cached_response(query_hash, context_hash, model_id, "ai_answer")
    if cached_text:
        logger.info(f"[CACHE] Hit for AI answer on query: '{query_hash[:8]}...'")
        return cached_text
    
    logger.info(f"[CACHE] Miss for AI answer. Generating with {model_id}...")
    answer = generate_ai_answer(context, question, provider, api_key, model_path, tensor_split=tensor_split)
    
    # 3. Store in Cache (if valid response)
    if answer and not answer.startswith("Error"):
        database.cache_response(query_hash, context_hash, model_id, "ai_answer", answer)
        
    return answer

def cached_smart_summary(text: str, query: str, provider: str, 
                         api_key: str = None, model_path: str = None, 
                         file_name: str = None) -> str:
    """
    Wrapper around smart_summary that checks persistent cache first.

    Caches contextual summaries to avoid redundant LLM calls during search 
    results rendering.

    Args:
        text (str): Document text to summarize.
        query (str): The keyword/question the summary should be tailored to.
        provider (str): 'openai', 'gemini', 'anthropic', or 'local'.
        api_key (str, optional): Key for cloud providers.
        model_path (str, optional): Path for local models.
        file_name (str, optional): Original name of the source file.

    Returns:
        str: A tailored summary.
    """
    if not text:
        return ""

    if provider == 'local':
        model_id = f"local:{os.path.basename(model_path)}" if model_path else "local:unknown"
    else:
        model_id = provider

    # Truncate text for cache key matching (must match logic in smart_summary)
    truncated_text = text[:3000]
    query_hash, context_hash = compute_cache_key(query, truncated_text, model_id)

    # 1. Check Cache
    cached_text = database.get_cached_response(query_hash, context_hash, model_id, "smart_summary")
    if cached_text:
        logger.info(f"[CACHE] Hit for smart_summary in {file_name or 'unknown'}")
        return cached_text
    
    logger.info(f"[CACHE] Miss for smart_summary in {file_name or 'unknown'}. Generating...")

    # 2. Generate
    summary = smart_summary(text, query, provider, api_key, model_path, file_name)

    # 3. Store
    # Logic to avoid caching fallbacks (regex summary usually just returns sentences)
    # We only cache if it looks like a genuine model output or a high quality extraction
    if summary and len(summary) > 10: 
        database.cache_response(query_hash, context_hash, model_id, "smart_summary", summary)
        
    return summary

def smart_summary(text: str, query: str, provider: str, 
                  api_key: str = None, model_path: str = None, 
                  file_name: str = None) -> str:
    """
    Generates a contextual summary of the text specifically related to the query.

    Uses an LLM to extract key findings from a document excerpt. Falls back 
    to a basic summarized version if model calls fail or are unavailable.

    Args:
        text (str): Full text of the document chunk.
        query (str): The search query to focus the summary on.
        provider (str): AI provider name.
        api_key (str, optional): API key.
        model_path (str, optional): Path to local GGUF.
        file_name (str, optional): Name of the file for prompting context.

    Returns:
        str: A natural language extract or summary.
    """
    if not text:
        return ""

    client = get_llm_client(provider, api_key, model_path)
    if not client:
        logger.error(f"[AI] Error: get_llm_client returned None for provider {provider}")
        # Fallback to regex summary if no model available
        return summarize(text, provider, api_key, model_path, query)
    
    logger.info(f"[AI] Smart Summary: Analyzing '{file_name or 'unnamed document'}' for query '{query[:3]}***'")

    # Truncate text to avoid token limits (approx 3000 chars ~ 750 tokens)
    truncated_text = text[:3000]
    
    file_context = f" from '{file_name}'" if file_name else ""

    prompt_text = f"""Analyze this document excerpt{file_context} for the query: "{query}".

Extract and quote specific facts, data, or content that answers the query. Be specific - include numbers, names, dates, or key details from the document.

If irrelevant, say "No relevant info".

Document:
{truncated_text}

Key findings:"""

    try:
        # Handle Local LLM
        if isinstance(client, str) and client.startswith("LOCAL:"):
            real_model_path = client.split("LOCAL:")[1]
            llm = get_local_llm(real_model_path)
            if not llm:
                return summarize(text, provider, api_key, model_path, query)

            with _local_llm_lock:
                output = llm.create_completion(
                    prompt_text,
                    max_tokens=128,
                    stop=["Document Excerpt:", "Summary:"],
                    echo=False,
                    temperature=0.1
                )
            result = output['choices'][0]['text'].strip()

        # Handle LangChain Clients
        else:
            from langchain_core.messages import HumanMessage
            response = client.invoke([HumanMessage(content=prompt_text)])
            result = response.content.strip()

        if "No relevant info" in result or len(result) < 5:
            return summarize(text, provider, api_key, model_path, query) # Fallback

        return result

    except Exception as e:
        logger.error(f"Smart summary error: {e}")
        return summarize(text, provider, api_key, model_path, query) # Fallback

def extract_answer(text: str, question: str) -> str:
    """
    Fast sentence-level matching fallback for answer extraction.

    Scores sentences based on the count of keywords from the question 
    and returns the highest scoring pair.

    Args:
        text (str): Document text to search.
        question (str): The search query.

    Returns:
        str: Up to two most relevant sentences.
    """
    if not text or not question:
        return ""
    
    question_lower = question.lower()
    cleaned_q = re.sub(r'\b(what|where|when|who|why|how|which|did|does|is|are|was|were)\b', '', question_lower)
    question_terms = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', cleaned_q) if w not in {'the', 'and', 'for', 'that', 'this'}]
    
    if not question_terms:
        return ""
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    scored_sentences = []
    for sent in sentences:
        sent_lower = sent.lower()
        score = sum(1 for term in question_terms if term in sent_lower)
        if score > 0:
            scored_sentences.append((score, sent))
    
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    
    if scored_sentences:
        best_sentences = [s[1].strip() for s in scored_sentences[:2]]
        answer = ' '.join(best_sentences)
        if len(answer) > 300:
            answer = answer[:297] + "..."
        return answer
    
    return ""

def summarize(text: str, provider: str = None, api_key: str = None, 
              model_path: str = None, question: str = None) -> str:
    """
    Fast regex-based summary generator.

    Args:
        text (str): Input text.
        provider (str, optional): AI provider (unused, interface consistency).
        api_key (str, optional): API key (unused).
        model_path (str, optional): Model path (unused).
        question (str, optional): If provided, focuses summary on the question.

    Returns:
        str: A short summary (first 2 significant sentences or extracted answer).
    """
    try:
        if question:
            answer = extract_answer(text, question)
            if answer:
                return answer
        
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        summary_sentences = []
        for sent in sentences:
            if len(sent) > 30:
                summary_sentences.append(sent)
                if len(summary_sentences) >= 2:
                    break
        
        if summary_sentences:
            return ' '.join(summary_sentences)
        return text[:150] + "..." if len(text) > 150 else text
    except Exception as e:
        logger.error(f"Error: {e}")
        return ""

def get_tags(text: str, provider: str, api_key: str = None, 
             model_path: str = None) -> str:
    """
    Extracts top keywords from text for tagging.

    Args:
        text (str): Input text.
        provider (str): AI provider (unused, interface consistency).
        api_key (str, optional): API key (unused).
        model_path (str, optional): Model path (unused).

    Returns:
        str: A comma-separated string of top 5 keywords.
    """
    try:
        words = re.findall(r'\b[a-zA-Z]{4,15}\b', text.lower())
        stop_words = {
            'this', 'that', 'with', 'from', 'have', 'will', 'been', 'would',
            'could', 'should', 'their', 'there', 'about', 'which', 'these',
            'other', 'more', 'some', 'such', 'only', 'than', 'into', 'over'
        }
        words = [w for w in words if w not in stop_words]
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return ', '.join([w[0] for w in top_words])
    except Exception as e:
        return ""
