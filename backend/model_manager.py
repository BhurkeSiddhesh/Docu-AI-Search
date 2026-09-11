import os
import logging
import requests
import threading
import shutil
import psutil

_logger = logging.getLogger(__name__)

# Path configuration for new folder structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Expanded list of GGUF models with metadata
AVAILABLE_MODELS = [
    # ============================================================
    # COMPATIBLE WITH 8GB RAM (Fast, lightweight edge models)
    # ============================================================
    {
        "id": "gemma-2-2b-it.Q4_K_M",
        "name": "Gemma 2 2B Instruct (Google)",
        "description": "Google's ultra-fast 2.6B model. Exceptional instruction following, QA, and low latency.",
        "size": "1.7 GB",
        "size_bytes": 1708582752,
        "ram_required": 4,
        "category": "8gb",
        "ram_tier": "8GB RAM",
        "speed": "Ultra-Fast",
        "intelligence": "Very High",
        "quantization": "Q4_K_M",
        "use_case": "Fast QA, Document Search, Low-Resource Systems",
        "recommended": True,
        "url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
    },
    {
        "id": "Llama-3.2-3B-Instruct.Q4_K_M",
        "name": "Llama 3.2 3B Instruct (Meta)",
        "description": "Meta's flagship compact model. Fast, precise on document reasoning, multilingual.",
        "size": "2.0 GB",
        "size_bytes": 2020000000,
        "ram_required": 5,
        "category": "8gb",
        "ram_tier": "8GB RAM",
        "speed": "Ultra-Fast",
        "intelligence": "Very High",
        "quantization": "Q4_K_M",
        "use_case": "Fast Instructions, Summarization, Edge Search",
        "recommended": True,
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    },
    {
        "id": "Qwen2.5-3B-Instruct.Q4_K_M",
        "name": "Qwen 2.5 3B Instruct (Alibaba)",
        "description": "Best-in-class small model for tabular data, spreadsheets, numbers, and structured extraction.",
        "size": "2.1 GB",
        "size_bytes": 2150000000,
        "ram_required": 5,
        "category": "8gb",
        "ram_tier": "8GB RAM",
        "speed": "Ultra-Fast",
        "intelligence": "Exceptional",
        "quantization": "Q4_K_M",
        "use_case": "Spreadsheets, Financial Docs, Invoices, Numerical Q&A",
        "recommended": True,
        "url": "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    },
    {
        "id": "Phi-3.5-mini-instruct.Q4_K_M",
        "name": "Phi-3.5 Mini Instruct (Microsoft)",
        "description": "Microsoft's 3.8B model with 128k context support and strong multi-step analytical reasoning.",
        "size": "2.4 GB",
        "size_bytes": 2500000000,
        "ram_required": 6,
        "category": "8gb",
        "ram_tier": "8GB RAM",
        "speed": "Fast",
        "intelligence": "Exceptional",
        "quantization": "Q4_K_M",
        "use_case": "Multi-Step Logic, Long Document Reasoning",
        "recommended": True,
        "url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"
    },

    # ============================================================
    # COMPATIBLE WITH 16GB RAM (Balanced speed & frontier reasoning)
    # ============================================================
    {
        "id": "Qwen2.5-7B-Instruct.Q4_K_M",
        "name": "Qwen 2.5 7B Instruct (Alibaba)",
        "description": "Worldwide leader among open 7B models. Top-tier comprehension on documents, tables, and code.",
        "size": "4.7 GB",
        "size_bytes": 4680000000,
        "ram_required": 9,
        "category": "16gb",
        "ram_tier": "16GB RAM",
        "speed": "Fast",
        "intelligence": "State-of-the-Art",
        "quantization": "Q4_K_M",
        "use_case": "Deep Document Analysis, Tabular Synthesis, Technical Docs",
        "recommended": True,
        "url": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    },
    {
        "id": "Meta-Llama-3.1-8B-Instruct.Q4_K_M",
        "name": "Llama 3.1 8B Instruct (Meta)",
        "description": "Meta's flagship 8B model with 128k context. Excellent general intelligence and summarization.",
        "size": "4.9 GB",
        "size_bytes": 4920000000,
        "ram_required": 10,
        "category": "16gb",
        "ram_tier": "16GB RAM",
        "speed": "Fast",
        "intelligence": "State-of-the-Art",
        "quantization": "Q4_K_M",
        "use_case": "Complex Summarization, Fact Extraction, Multi-Document Synthesis",
        "recommended": True,
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    },
    {
        "id": "gemma-2-9b-it.Q4_K_M",
        "name": "Gemma 2 9B Instruct (Google)",
        "description": "Google's 9B powerhouse. Rivals many 30B+ models in factual accuracy, reasoning, and depth.",
        "size": "5.5 GB",
        "size_bytes": 5500000000,
        "ram_required": 11,
        "category": "16gb",
        "ram_tier": "16GB RAM",
        "speed": "Balanced",
        "intelligence": "State-of-the-Art",
        "quantization": "Q4_K_M",
        "use_case": "High-Accuracy Q&A, Complex Domain Understanding",
        "recommended": True,
        "url": "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf"
    },
    {
        "id": "Mistral-7B-Instruct-v0.3.Q4_K_M",
        "name": "Mistral 7B Instruct v0.3",
        "description": "Reliable, high-efficiency 7B model. Fast token generation and concise document answers.",
        "size": "4.4 GB",
        "size_bytes": 4370000000,
        "ram_required": 8,
        "category": "16gb",
        "ram_tier": "16GB RAM",
        "speed": "Fast",
        "intelligence": "High",
        "quantization": "Q4_K_M",
        "use_case": "Fast General Search, Direct Q&A",
        "recommended": False,
        "url": "https://huggingface.co/maziyarpanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
    },

    # ============================================================
    # COMPATIBLE WITH 32GB+ RAM (Maximum reasoning & fidelity)
    # ============================================================
    {
        "id": "Qwen2.5-14B-Instruct.Q4_K_M",
        "name": "Qwen 2.5 14B Instruct (Alibaba)",
        "description": "Frontier-level intelligence. Outperforms original GPT-4 on many benchmarks with deep reasoning.",
        "size": "9.0 GB",
        "size_bytes": 9000000000,
        "ram_required": 16,
        "category": "32gb",
        "ram_tier": "32GB+ RAM",
        "speed": "Balanced",
        "intelligence": "Frontier-Grade",
        "quantization": "Q4_K_M",
        "use_case": "Power Users, Multi-Document Research, Complex Analytics",
        "recommended": True,
        "url": "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    },
    {
        "id": "Meta-Llama-3.1-8B-Instruct.Q8_0",
        "name": "Llama 3.1 8B Instruct (Q8_0 High-Precision)",
        "description": "Maximum 8-bit precision with virtually zero quantization loss. Flawless syntax and facts.",
        "size": "8.5 GB",
        "size_bytes": 8540000000,
        "ram_required": 16,
        "category": "32gb",
        "ram_tier": "32GB+ RAM",
        "speed": "Fast",
        "intelligence": "State-of-the-Art (Full Precision)",
        "quantization": "Q8_0",
        "use_case": "Exact Numerical & Factual Extraction, Zero Precision Degradation",
        "recommended": False,
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    },
    {
        "id": "gemma-2-9b-it.Q8_0",
        "name": "Gemma 2 9B Instruct (Q8_0 High-Precision)",
        "description": "Uncompromised 8-bit quantization of Google's 9B model for 32GB+ memory workstations.",
        "size": "9.9 GB",
        "size_bytes": 9900000000,
        "ram_required": 18,
        "category": "32gb",
        "ram_tier": "32GB+ RAM",
        "speed": "Balanced",
        "intelligence": "Frontier-Grade (Full Precision)",
        "quantization": "Q8_0",
        "use_case": "Highest Quality Reasoning in the 9B Class",
        "recommended": False,
        "url": "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q8_0.gguf"
    }
]

download_status = {
    "downloading": False,
    "model_id": None,
    "progress": 0,
    "error": None,
    "bytes_downloaded": 0,
    "total_bytes": 0
}
_download_lock = threading.Lock()

def get_available_models():
    """
    Returns a list of all models available for download.

    Each model dictionary contains metadata like ID, name, description,
    size, RAM requirements, and download URL.

    Returns:
        list: A list of model configuration dictionaries.
    """
    return AVAILABLE_MODELS

def get_local_models():
    """
    Scans the models directory and returns a list of downloaded models.

    Matches downloaded GGUF files with metadata from the available models list
    to provide descriptive information in the UI.

    Returns:
        list: A list of dictionaries, each containing 'id', 'filename', 'path',
              'size', 'name', 'category', and 'ram_required'.
    """
    models = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".gguf"):
                filepath = os.path.join(MODELS_DIR, f)
                size = os.path.getsize(filepath)
                
                # Try to find metadata from AVAILABLE_MODELS
                model_id = f.replace(".gguf", "")
                available_model = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
                
                models.append({
                    "id": model_id,
                    "filename": f,
                    "path": os.path.abspath(filepath),
                    "size": size,
                    "name": available_model["name"] if available_model else f.replace(".gguf", "").replace("-", " ").replace(".", " "),
                    "category": available_model["category"] if available_model else "unknown",
                    "ram_required": available_model["ram_required"] if available_model else None,
                    "ram_tier": available_model.get("ram_tier") if available_model else None,
                    "speed": available_model.get("speed") if available_model else None,
                    "intelligence": available_model.get("intelligence") if available_model else None,
                })
    return models

def check_system_resources(model):
    """
    Validates if the system has enough disk space and RAM for a model.

    Checks:
        - Free disk space in the models directory (requires 10% buffer).
        - Available system RAM compared to the model's recommendation.

    Args:
        model (dict): The model metadata dictionary to check.

    Returns:
        tuple (bool, list):
            - bool: True if disk space is sufficient to proceed with download.
            - list: A list of warning strings (e.g., low disk, low RAM).
    """
    warnings = []
    can_download = True
    
    # Check available disk space
    disk_usage = shutil.disk_usage(MODELS_DIR)
    available_gb = disk_usage.free / (1024**3)
    required_gb = model.get("size_bytes", 0) / (1024**3) * 1.1  # 10% buffer
    
    if available_gb < required_gb:
        warnings.append(f"Low disk space: {available_gb:.1f}GB available, {required_gb:.1f}GB needed")
        can_download = False
    
    # Check available RAM
    ram_info = psutil.virtual_memory()
    available_ram_gb = ram_info.available / (1024**3)
    required_ram_gb = model.get("ram_required", 4)
    
    if available_ram_gb < required_ram_gb:
        warnings.append(f"Low RAM: {available_ram_gb:.1f}GB available, {required_ram_gb}GB recommended")
        # Don't block download for RAM, just warn
    
    return can_download, warnings

def download_file(url, filename, model_id, total_bytes=0):
    """
    Internal worker function that downloads a model file from a URL.

    Specifically handles:
        - Resuming partial downloads using HTTP Range headers.
        - Chunked transfer for large files (1MB blocks).
        - Real-time global status updates including progress percentage.
        - Moving the file from a .partial extension to the final name upon success.

    Args:
        url (str): The direct download URL for the GGUF model.
        filename (str): The local target filename.
        model_id (str): The unique identifier of the model being downloaded.
        total_bytes (int, optional): Expected file size for progress calculation. 
                                     Defaults to 0.

    Note:
        This function updates the global `download_status` variable which is
        shared with the API to report progress to the frontend.
    """
    import logging
    _logger = logging.getLogger(__name__)
    global download_status
    filepath = os.path.join(MODELS_DIR, filename)
    temp_filepath = filepath + ".partial"

    try:
        _logger.info("Starting download: %s", url)
        with _download_lock:
            download_status = {
                "downloading": True,
                "model_id": model_id,
                "progress": 0,
                "error": None,
                "bytes_downloaded": 0,
                "total_bytes": total_bytes
            }

        # Check for partial download (resume support)
        headers = {}
        downloaded = 0
        if os.path.exists(temp_filepath):
            downloaded = os.path.getsize(temp_filepath)
            headers["Range"] = f"bytes={downloaded}-"
            _logger.info("Resuming from byte %d", downloaded)

        # Retry transient connection/5xx failures; resume support above makes
        # a retried request continue from the last byte rather than restart.
        from backend.providers import _make_retry_session
        response = _make_retry_session().get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()

        # Get total size from headers
        if "content-range" in response.headers:
            total_size = int(response.headers.get("content-range", "").split("/")[-1])
        else:
            total_size = int(response.headers.get('content-length', 0)) + downloaded

        with _download_lock:
            download_status["total_bytes"] = total_size
        block_size = 1024 * 1024  # 1MB

        mode = 'ab' if downloaded > 0 else 'wb'
        with open(temp_filepath, mode) as file:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                file.write(data)
                if total_size > 0:
                    with _download_lock:
                        download_status["progress"] = int((downloaded / total_size) * 100)
                        download_status["bytes_downloaded"] = downloaded

        # Rename to final filename
        os.rename(temp_filepath, filepath)

        _logger.info("Download complete: %s", filename)
        with _download_lock:
            download_status = {
                "downloading": False,
                "model_id": None,
                "progress": 100,
                "error": None,
                "bytes_downloaded": downloaded,
                "total_bytes": total_size
            }

    except Exception as e:
        _logger.error("Download failed: %s", e, exc_info=True)
        with _download_lock:
            download_status = {
                "downloading": False,
                "model_id": model_id,
                "progress": 0,
                # Generic message only — exception details (which can carry
                # stack/internal info) stay in the server log.
                "error": "Download failed. Check the server log for details.",
                "bytes_downloaded": 0,
                "total_bytes": 0
            }
        # Keep partial file for resume

def start_download(model_id):
    """
    Initiates a background download thread for a specific model ID.

    Performs preliminary checks (already downloading, model existence,
    disk space) before spawning a daemon thread to handle the IO.

    Args:
        model_id (str): The ID of the model from AVAILABLE_MODELS to download.

    Returns:
        tuple (bool, str):
            - bool: True if the download thread was successfully started.
            - str: A success or error message for the caller.
    """
    global download_status

    model = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    if not model:
        return False, f"Model not found: {model_id}"

    filename = f"{model_id}.gguf"
    filepath = os.path.join(MODELS_DIR, filename)

    if os.path.exists(filepath):
        return False, "Model already downloaded"

    can_download, warnings = check_system_resources(model)
    if not can_download:
        return False, f"Cannot download: {'; '.join(warnings)}"

    with _download_lock:
        if download_status["downloading"]:
            return False, "Another download is in progress"
        download_status["downloading"] = True
        download_status["model_id"] = model_id

    thread = threading.Thread(
        target=download_file,
        args=(model["url"], filename, model_id, model.get("size_bytes", 0))
    )
    thread.daemon = True
    thread.start()
    
    warning_msg = f" (Warnings: {'; '.join(warnings)})" if warnings else ""
    return True, f"Download started{warning_msg}"

def get_download_status():
    """
    Retrieves the current background download status.

    Returns:
        dict: A dictionary containing 'downloading' (bool), 'model_id' (str),
              'progress' (int), 'error' (str), 'bytes_downloaded' (int),
              and 'total_bytes' (int).
    """
    with _download_lock:
        return dict(download_status)

def is_safe_model_path(path):
    """
    Validates that the path is strictly within the MODELS_DIR.

    This function is a security essential, preventing path traversal attacks
    and unauthorized file deletions outside the intended directory.

    Args:
        path (str): The file path to validate.

    Returns:
        bool: True if the path is safe (inside MODELS_DIR), False otherwise.
    """
    if not path:
        return False

    try:
        # Resolve absolute paths
        abs_path = os.path.abspath(path)
        abs_models_dir = os.path.abspath(MODELS_DIR)

        # Check if path is within MODELS_DIR
        # os.path.commonpath returns the longest common sub-path
        # We ensure the common path IS the models directory
        common = os.path.commonpath([abs_path, abs_models_dir])

        # 1. Must be inside models dir
        # 2. Must not BE the models dir itself (prevent deleting the folder)
        return common == abs_models_dir and abs_path != abs_models_dir

    except Exception:
        return False

def delete_model(model_path):
    """
    Deletes a downloaded model file from the disk.

    Uses `is_safe_model_path` to ensure security against path traversal
    before attempting the file removal.

    Args:
        model_path (str): The absolute or relative path to the GGUF file.

    Returns:
        bool: True if the file was successfully deleted or didn't exist,
              False if the path was unsafe or an error occurred.
    """
    if not is_safe_model_path(model_path):
        _logger.warning("Security: attempt to delete unsafe path: %s", model_path)
        return False

    # Resolve against the directory listing so the path handed to os.remove
    # always originates from MODELS_DIR enumeration, not from the request.
    # String-only normalisation: no filesystem call touches the raw input.
    try:
        candidate = os.path.normcase(os.path.abspath(os.path.normpath(model_path)))
    except (ValueError, TypeError):
        return False
    try:
        entries = os.listdir(MODELS_DIR)
    except OSError as e:
        _logger.error("Error listing models directory: %s", e)
        return False
    for name in entries:
        target = os.path.join(MODELS_DIR, name)
        if os.path.normcase(os.path.abspath(target)) == candidate:
            try:
                os.remove(target)
                return True
            except OSError as e:
                _logger.error("Error deleting model: %s", e)
                return False
    return False
