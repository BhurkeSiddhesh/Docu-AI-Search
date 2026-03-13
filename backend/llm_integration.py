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
    print(f"DEBUG ERROR: {e}")

# IMPORTS FIXED: Lazy loading to prevent startup crashes
# from langchain_huggingface import HuggingFaceEmbeddings

# Llama import moved to get_local_llm to prevent top-level blocking

# Cache for loaded models
_embeddings_cache = {}
_llm_cache = {}
_llm_client_cache = {}

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
    
    print(f"Loading embeddings for provider: {provider}")

    # DEBUG: Log environment when actually requesting embeddings
    try:
        import sys
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"--- get_embeddings called ---")
        logger.info(f"Python Executable: {sys.executable}")
        logger.info(f"Python Path: {sys.path}")
        logger.info(f"CWD: {os.getcwd()}")
    except:
        pass

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
        print("Using local embeddings for Grok provider.")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        # Default / Local
        print("Loading local embeddings (HuggingFace)...")
        if HuggingFaceEmbeddings is None:
            import logging
            logger = logging.getLogger(__name__)
            logger.error("HuggingFaceEmbeddings not available")
            return None

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    
    print("Embeddings loaded!")
    _embeddings_cache[cache_key] = embeddings
    return embeddings

# ---------------------------------------------------------------------------
# Embedding Factory (Factory Pattern)
# ---------------------------------------------------------------------------

_DEFAULT_LOCAL_EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

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

    # ------------------------------------------------------------------ local
    if provider_type == 'local':
        if HuggingFaceEmbeddings is None:
            raise ImportError(
                "langchain-huggingface is not installed. "
                "Run: pip install langchain-huggingface"
            )
        resolved_model = model_name or _DEFAULT_LOCAL_EMBEDDING_MODEL
        logger.info(f"[EmbeddingFactory] local → {resolved_model}")
        return HuggingFaceEmbeddings(
            model_name=resolved_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )

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
        logger.info(f"[EmbeddingFactory] huggingface_api → {model_name}")
        return HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=api_key,
            repo_id=model_name,
        )

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
            logger.info(f"[EmbeddingFactory] commercial_api/OpenAI → {model_name}")
            return OpenAIEmbeddings(model=model_name, api_key=api_key)

        # Google Gemini: model names like 'models/embedding-001', 'gemini-…'
        if any(kw in model_lower for kw in ('gemini', 'embedding-001')):
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError(
                    "langchain-google-genai is not installed. "
                    "Run: pip install langchain-google-genai"
                )
            logger.info(f"[EmbeddingFactory] commercial_api/Gemini → {model_name}")
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
            )

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
        print("llama_cpp not installed")
        return None

        
    if not model_path or not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None

    if model_path in _llm_cache:
        return _llm_cache[model_path]

    print(f"Loading Local LLM from {model_path}...")
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
        _llm_cache[model_path] = llm
        print("Local LLM loaded!")
        return llm
    except Exception as e:
        print(f"Failed to load Local LLM: {e}")
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


def get_llm_client(provider: str, api_key: str = None, model_path: str = None) -> Any:
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
                print("OpenAI API Key missing")
                return None
            client = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.3)

        elif provider == 'gemini':
            if not api_key:
                print("Gemini API Key missing")
                return None
            client = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash", temperature=0.3)

        elif provider == 'anthropic':
            if not api_key:
                print("Anthropic API Key missing")
                return None
            client = ChatAnthropic(api_key=api_key, model="claude-3-haiku-20240307", temperature=0.3)

        elif provider == 'grok':
            if not api_key:
                print("Grok API Key missing")
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
                print("Local model path missing or invalid")
                return None
            client = "LOCAL:" + model_path

    except Exception as e:
        print(f"Error initializing LLM client for {provider}: {e}")
        return None

    if client:
        _llm_client_cache[cache_key] = client
    return client

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
        # Handle Local LLM (LlamaCpp)
        if isinstance(client, str) and client.startswith("LOCAL:"):
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
        print(f"Generation error ({provider}): {e}")
        return f"Error generating answer: {str(e)}"

def stream_ai_answer(context: str, question: str, provider: str, 
                     api_key: str = None, model_path: str = None, 
                     tensor_split: List[float] = None) -> Any:
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
    client = get_llm_client(provider, api_key, model_path)
    if not client:
        yield "Error: Could not initialize AI model. Check settings and API keys."
        return

    prompt_text = f"""You are a document search assistant. Answer the question using ONLY facts from the provided documents.

    IMPORTANT:
    1. Check Names: If the question is about a specific person (e.g., 'Siddhesh'), ensure you ONLY use documents describing that exact person. Do NOT confuse them with similar names like 'Siddharth'.
    2. Quote specific content, data, numbers, or key details from the documents.
    3. Reference which file the information comes from when possible.
    4. If the documents do not contain information for the specific person requested, state that clearly.

    Documents:
    {context}

    Question: {question}

    Answer (cite specific details from the documents):"""

    try:
        # Handle Local LLM (LlamaCpp)
        if isinstance(client, str) and client.startswith("LOCAL:"):
            real_model_path = client.split("LOCAL:")[1]
            llm = get_local_llm(real_model_path, tensor_split=tensor_split)
            if not llm:
                yield "Error: Local model failed to load."
                return

            MAX_CONTEXT_CHARS = 10000
            if len(context) > MAX_CONTEXT_CHARS:
                context = context[:MAX_CONTEXT_CHARS] + "... [Truncated to fit context window]"

            with _local_llm_lock:
                stream = llm.create_completion(
                    prompt_text,
                    max_tokens=512,
                    stop=["System:", "Question:", "Context:", "Documents:"],
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

            messages = [
                SystemMessage(content="""You are a precise document search assistant.
                CRITICAL: You must distinguish between similar names. If the question asks about 'Siddhesh', do NOT provide info about 'Siddharth'.
                Only answer based on the provided documents. Quote facts and reference file names."""),
                HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:")
            ]

            for chunk in client.stream(messages):
                 yield chunk.content

    except Exception as e:
        print(f"Streaming error ({provider}): {e}")
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
        print(f"[CACHE] Hit for AI answer on query: '{question[:30]}...'")
        return cached_text
    
    print(f"[CACHE] Miss for AI answer. Generating with {model_id}...")
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
        print(f"[CACHE] Hit for smart_summary in {file_name or 'unknown'}")
        return cached_text
    
    print(f"[CACHE] Miss for smart_summary in {file_name or 'unknown'}. Generating...")

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
        print(f"[AI] Error: get_llm_client returned None for provider {provider}")
        # Fallback to regex summary if no model available
        return summarize(text, provider, api_key, model_path, query)
    
    print(f"[AI] Smart Summary: Analyzing '{file_name or 'unnamed document'}' for query '{query}'")

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
        print(f"Smart summary error: {e}")
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
        print(f"Error: {e}")
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
