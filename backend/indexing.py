import os
import faiss
import json
import logging
import pickle
import numpy as np
import concurrent.futures
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)
from backend.llm_integration import get_embeddings, get_tags, smart_summary, summarize
from backend.file_processing import extract_text, SUPPORTED_EXTENSIONS
from backend import database
from backend.clustering import perform_global_clustering
from rank_bm25 import BM25Okapi
import string

# Metadata sidecar filename suffix
_META_SUFFIX = '_meta.json'

# Chunking strategy identifier, persisted in the metadata sidecar. When this
# changes (different splitter/size/overlap), cached chunks from a previous
# index can no longer be reused and a full re-chunk/re-embed is forced.
_CHUNKER_VERSION = 'recursive-1000-150'
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 150

# Checkpoint file for resume-on-failure support
_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'index_checkpoint.json')


def _load_checkpoint(fingerprint: str = None) -> dict:
    """Load the extraction checkpoint, discarding it if it belongs to a
    different file set (prevents stale text from a previous run leaking in)."""
    if os.path.exists(_CHECKPOINT_PATH):
        try:
            with open(_CHECKPOINT_PATH) as f:
                data = json.load(f)
            # New format: {"fingerprint": ..., "files": {...}}
            if isinstance(data, dict) and 'files' in data:
                if fingerprint is None or data.get('fingerprint') == fingerprint:
                    return data['files']
                logger.info("[Index] Checkpoint fingerprint mismatch — starting fresh.")
                return {}
            # Legacy flat format — only trust it if no fingerprint was requested
            if isinstance(data, dict) and fingerprint is None:
                return data
        except Exception:
            pass
    return {}


def _save_checkpoint(checkpoint: dict, fingerprint: str = ""):
    os.makedirs(os.path.dirname(_CHECKPOINT_PATH), exist_ok=True)
    with open(_CHECKPOINT_PATH, 'w') as f:
        json.dump({'fingerprint': fingerprint, 'files': checkpoint}, f)


def _clear_checkpoint():
    if os.path.exists(_CHECKPOINT_PATH):
        os.remove(_CHECKPOINT_PATH)

def _embed_batch_with_retry(model, batch, retries: int = 3):
    """Embed a single batch, retrying up to `retries` times with exponential back-off."""
    if model is None:
        raise ValueError("Embedding model is not initialized.")
    for attempt in range(retries):
        try:
            return model.embed_documents(batch)
        except (AttributeError, TypeError, ValueError):
            raise
        except Exception as e:
            logger.warning(
                "Embedding attempt %d/%d failed: %s. Retrying in %ds...",
                attempt + 1, retries, e, 2 ** attempt,
            )
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization for BM25 keyword matching.

    Removes punctuation and converts text to lowercase.

    Args:
        text (str): Input text string.

    Returns:
        List[str]: A list of cleaned tokens.
    """
    # Remove punctuation and lowercase
    translator = str.maketrans('', '', string.punctuation)
    return text.lower().translate(translator).split()

def safe_extract_text(filepath: str) -> Tuple[str, Optional[str]]:
    """
    Thread-safe wrapper for text extraction.

    Used for parallel execution to catch exceptions and prevent one bad file 
    from crashing the indexing process.

    Args:
        filepath (str): Path to the document.

    Returns:
        Tuple[str, Optional[str]]: A tuple of (path, extracted_text). 
            Text is None if extraction fails.
    """
    try:
        text = extract_text(filepath)
        return filepath, text
    except Exception as e:
        logger.info(f"Error reading {filepath}: {e}")
        return filepath, None

def _load_reusable_chunks(previous_index_path: str, current_files: List[str],
                          current_model_name: str) -> Dict[str, List[Tuple[str, Any]]]:
    """
    Load chunks + embedding vectors from the previous index for files that
    have not changed since it was built (same path, size, and mtime).

    Reuse is only safe when the chunking strategy and embedding model are
    unchanged; otherwise an empty dict is returned and the caller does a
    full re-index.

    Returns:
        Dict[str, List[Tuple[str, np.ndarray]]]: {filepath: [(chunk_text, vector), ...]}
    """
    try:
        base_path = os.path.splitext(previous_index_path)[0]
        meta_path = base_path + _META_SUFFIX
        docs_path = base_path + '_docs.pkl'
        if not (os.path.exists(previous_index_path) and os.path.exists(meta_path) and os.path.exists(docs_path)):
            return {}

        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if meta.get('chunker') != _CHUNKER_VERSION:
            logger.info("[Index] Chunking strategy changed — full re-index required.")
            return {}

        def _norm(name: str) -> str:
            return str(name or '').split('/')[-1].lower()
        prev_model = meta.get('model_name') or 'unknown'
        if prev_model == 'unknown' or _norm(prev_model) != _norm(current_model_name):
            logger.info("[Index] Embedding model changed (%s -> %s) — full re-index required.",
                        prev_model, current_model_name)
            return {}

        fingerprints = database.get_file_fingerprints()
        if not fingerprints:
            return {}

        with open(docs_path, 'rb') as f:
            prev_chunks = pickle.load(f)
        prev_index = faiss.read_index(previous_index_path)
        if not prev_chunks or prev_index.ntotal != len(prev_chunks):
            return {}
        all_vectors = prev_index.reconstruct_n(0, prev_index.ntotal)

        # Group previous chunks (position, text) by source file
        by_file: Dict[str, List[Tuple[int, str]]] = {}
        for pos, chunk in enumerate(prev_chunks):
            if not isinstance(chunk, dict):
                return {}
            by_file.setdefault(chunk.get('filepath'), []).append((pos, chunk.get('text', '')))

        current_set = set(current_files)
        reusable: Dict[str, List[Tuple[str, Any]]] = {}
        for filepath, entries in by_file.items():
            if filepath not in current_set or filepath not in fingerprints:
                continue
            try:
                stat = os.stat(filepath)
            except OSError:
                continue
            size_db, mtime_db = fingerprints[filepath]
            if int(stat.st_size) != int(size_db or -1):
                continue
            if abs(float(stat.st_mtime) - float(mtime_db or -1.0)) > 1e-6:
                continue
            reusable[filepath] = [(text, all_vectors[pos]) for pos, text in entries]
        return reusable
    except Exception as e:
        logger.warning("[Index] Incremental reuse unavailable (%s); doing a full re-index.", e)
        return {}


def create_index(folder_paths: List[str] | str, provider: str, api_key: str = None,
                 model_path: str = None, progress_callback: callable = None,
                 embedding_client: Any = None, previous_index_path: str = None) -> Tuple:
    """
    Creates a RAPTOR index (Global Clustering + Recursive Summarization).

    This pipeline handles:
    1. Parallel text extraction from various file types.
    2. Overlapping sliding-window chunking.
    3. High-dimensional vector embedding.
    4. BM25 keyword index construction.
    5. Hierarchical UMAP/K-Means clustering.
    6. LLM-based cluster summarization for multi-level RAG.

    Args:
        folder_paths (List[str] | str): One or more directories to scan.
        provider (str): LLM provider for summarization.
        api_key (str, optional): Secret key for cloud providers.
        model_path (str, optional): Path to GGUF file for local models.
        progress_callback (callable, optional): function(percent, total, status) for UI updates.
        embedding_client (Any, optional): A pre-resolved LangChain embedding client.
        previous_index_path (str, optional): Path to the existing index on disk.
            When provided, chunks + vectors of unchanged files are reused so
            only new/modified files are re-extracted and re-embedded.

    Returns:
        Tuple: (index_chunks, all_chunks, tags, index_summaries, cluster_summaries, final_cluster_map, bm25, meta)
            Note: Returns an empty state if no files are found.
    """
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]

    logger.info(f"Starting RAPTOR Indexing of folders: {folder_paths}")
    start_time = time.time()

    # 1. Collect Files — only supported types, skipping dot/junk directories so
    # indexing a project folder doesn't churn through node_modules or .git.
    _SKIP_DIRS = {'node_modules', '__pycache__', 'venv', 'venv_new', '.git', '.svn', '$RECYCLE.BIN', 'System Volume Information'}
    all_files = []
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            for dirpath, dirnames, filenames in os.walk(folder_path):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith('.')]
                for filename in filenames:
                    if os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS:
                        all_files.append(os.path.join(dirpath, filename))

    logger.info(f"Found {len(all_files)} supported files.")
    if not all_files:
        return None, None, None, None, None, None, None, {}

    # 2. Resolve the embedding client up front — its identity gates whether
    # cached vectors from the previous index may be reused.
    if embedding_client is not None:
        embeddings_model = embedding_client
        logger.info("[Index] Using pre-resolved embedding client from app.state.")
    else:
        embeddings_model = get_embeddings(provider, api_key, model_path)
    _model_name = getattr(embeddings_model, 'model_name', None) or getattr(embeddings_model, 'model', 'unknown')

    # 3. Incremental reuse — must run BEFORE the DB is cleared, because file
    # fingerprints (size/mtime) from the previous run live in the files table.
    reuse_map: Dict[str, List[Tuple[str, Any]]] = {}
    if previous_index_path:
        reuse_map = _load_reusable_chunks(previous_index_path, all_files, str(_model_name))
        if reuse_map:
            logger.info(f"[Index] Incremental: reusing chunks+vectors for "
                        f"{len(reuse_map)}/{len(all_files)} unchanged files.")

    # 4. Clear Database (rebuilt below from reused + fresh files)
    database.clear_files()
    database.clear_clusters()

    # Define stage weights
    # Extraction: 20%, Chunking: 5%, Embedding: 40%, Clustering: 5%, Summarization: 25%, Finalizing: 5%
    # We will accumulate progress_base to ensure monotonic increase

    # 5. Parallel Text Extraction (CPU Bound) - 0% to 20%
    logger.info("Step 1/5: Extracting Text (Parallel)...")
    valid_docs = [] # List of (filepath, text)

    # Load checkpoint to resume after a failure. The fingerprint ties the
    # checkpoint to this exact file set so leftovers from other runs are ignored.
    import hashlib
    _fingerprint = hashlib.sha256("\n".join(sorted(all_files)).encode('utf-8', 'replace')).hexdigest()[:16]
    checkpoint = _load_checkpoint(_fingerprint)
    files_to_extract = [f for f in all_files if f not in checkpoint and f not in reuse_map]
    # Restore already-extracted docs from checkpoint (reused files don't need text)
    for cached_path, cached_text in checkpoint.items():
        if cached_path in all_files and cached_text and cached_path not in reuse_map:
            valid_docs.append((cached_path, cached_text))

    # Use fewer workers for CPU bound tasks to keep UI responsive.
    # Process pools cost ~5s/worker to spawn on Windows (each re-imports the
    # backend stack), so only pay that for corpora large enough to benefit.
    if len(files_to_extract) >= 50:
        _executor_cls = concurrent.futures.ProcessPoolExecutor
    else:
        _executor_cls = concurrent.futures.ThreadPoolExecutor
    with _executor_cls(max_workers=4) as executor:
        future_to_file = {executor.submit(safe_extract_text, f): f for f in files_to_extract}

        total_files = len(all_files)
        compete_count = total_files - len(files_to_extract)  # checkpointed + reused files
        _since_save = 0
        for future in concurrent.futures.as_completed(future_to_file):
            filepath, text = future.result()
            if text:
                valid_docs.append((filepath, text))
            checkpoint[filepath] = text or ""
            # Batch checkpoint writes: rewriting the full JSON per file is O(n²) I/O
            _since_save += 1
            if _since_save >= 20:
                _save_checkpoint(checkpoint, _fingerprint)
                _since_save = 0

            compete_count += 1
            if progress_callback:
                # Map 0-total_files to 0-20%
                percent = int((compete_count / total_files) * 20)
                progress_callback(percent, 100, f"Extracting: {os.path.basename(filepath)}")
        # Flush any remaining entries
        if extracted_since_save > 0:
            _save_checkpoint(checkpoint)

        if _since_save:
            _save_checkpoint(checkpoint, _fingerprint)

    logger.info(f"Successfully extracted text from {len(valid_docs)} files.")

    # 6. Chunking - 20% to 25%
    if progress_callback: progress_callback(22, 100, "Chunking text...")
    logger.info("Step 2/5: Chunking Text...")
    # Recursive splitting honours the chunk budget even for text without blank
    # lines (PDF extractions often have none) by falling back through
    # paragraph -> line -> sentence -> word boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []      # List of chunk dicts (text, filepath, faiss_idx)
    chunk_strings = []   # Just the text (BM25 / KG / alignment)
    chunk_vectors = []   # Per chunk: reused vector or None (needs embedding)
    pending_texts = []   # Chunk texts that still need embedding

    extracted_texts = dict(valid_docs)
    current_faiss_idx = 0

    files_to_add = []
    for filepath in all_files:
        reused = reuse_map.get(filepath)
        if reused is not None:
            file_chunks = [t for t, _v in reused]
            file_vecs = [v for _t, v in reused]
        else:
            text = extracted_texts.get(filepath)
            if not text:
                continue
            file_chunks = text_splitter.split_text(text)
            file_vecs = [None] * len(file_chunks)

        if not file_chunks:
            continue

        try:
            file_stat = os.stat(filepath)
        except OSError as stat_err:
            logger.warning(f"Skipping {filepath}: {stat_err}")
            continue
        file_info = {
            'path': filepath,
            'filename': os.path.basename(filepath),
            'file_type': os.path.splitext(filepath)[1].lower(),
            'size': file_stat.st_size,
            'last_modified': file_stat.st_mtime, # Database expects float timestamp
            'faiss_start_idx': current_faiss_idx,
            'faiss_end_idx': current_faiss_idx + len(file_chunks) - 1,
            'tags': '[]' # Default empty tags
        }

        # Add to DB immediately
        files_to_add.append(file_info)
        if len(files_to_add) >= 500:
            database.add_files_batch(files_to_add)
            files_to_add = []

        for chunk, vec in zip(file_chunks, file_vecs):
            all_chunks.append({
                'text': chunk,
                'filepath': filepath,
                'faiss_idx': current_faiss_idx,
                'file_id': None # Could fetch, but relying on path match is okay for now
            })
            chunk_strings.append(chunk)
            chunk_vectors.append(vec)
            if vec is None:
                pending_texts.append(chunk)
            current_faiss_idx += 1

    if files_to_add:
        database.add_files_batch(files_to_add)
    logger.info(f"Generated {len(chunk_strings)} total chunks "
                f"({len(chunk_strings) - len(pending_texts)} reused, {len(pending_texts)} to embed).")

    if not chunk_strings:
        logger.info("Warning: No text chunks found in provided files.")
        return None, None, None, None, None, None, None, {}

    # 7. Parallel Embedding of new/changed chunks (I/O Bound) - 25% to 65%
    if progress_callback: progress_callback(25, 100, "Starting embeddings...")
    logger.info("Step 3/5: Embedding Chunks (Parallel)...")

    new_embeddings = []
    if pending_texts:
        # Embed in batches to be efficient but safe
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
        batches = [pending_texts[i:i + batch_size] for i in range(0, len(pending_texts), batch_size)]

        # Use ThreadPool for Network/GPU bound
        chunk_embeddings_map = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # submit returns a future object
            future_map = {executor.submit(_embed_batch_with_retry, embeddings_model, batch): i for i, batch in enumerate(batches)}

            completed = 0
            total_batches = len(batches)

            for future in concurrent.futures.as_completed(future_map):
                batch_idx = future_map[future]
                try:
                    result = future.result()
                    chunk_embeddings_map[batch_idx] = result
                except Exception as e:
                    logger.info(f"Error embedding batch {batch_idx}: {e}")
                    chunk_embeddings_map[batch_idx] = [] # Handle failure gracefully?

                completed += 1
                if progress_callback:
                    # Map 0-total_batches to 25-65% (range of 40)
                    percent = 25 + int((completed / total_batches) * 40)
                    progress_callback(percent, 100, f"Embedding batch {completed}/{total_batches}")
        # Reassemble in order
        for i in range(len(batches)):
            if i in chunk_embeddings_map:
                new_embeddings.extend(chunk_embeddings_map[i])

        # Fail fast on embedding failures rather than silently produce a corrupt index.
        # If even one batch returned empty, the FAISS vectors no longer line up 1:1
        # with chunk_strings and downstream search returns wrong chunks.
        if not new_embeddings:
            logger.error(
                "Indexing aborted: every embedding batch failed (0/%d). "
                "Check the embedding provider/API key.",
                len(batches),
            )
            _clear_checkpoint()
            return None, None, None, None, None, None, None, {}

        if len(new_embeddings) != len(pending_texts):
            logger.error(
                "Indexing aborted: embedding/chunk count mismatch (%d embeddings vs %d chunks). "
                "Some batches failed — aborting to avoid a misaligned index.",
                len(new_embeddings),
                len(pending_texts),
            )
            _clear_checkpoint()
            return None, None, None, None, None, None, None, {}

    # Merge reused vectors with fresh ones, preserving chunk order
    _new_iter = iter(new_embeddings)
    chunk_embeddings = [vec if vec is not None else next(_new_iter) for vec in chunk_vectors]

    # 5b. BM25 Indexing - 65% to 68%
    if progress_callback: progress_callback(66, 100, "Building Keyword Index...")
    logger.info("Step 3.5/5: Building BM25 Index...")
    tokenized_corpus = [tokenize(doc) for doc in chunk_strings]
    bm25 = BM25Okapi(tokenized_corpus)

    # 6. Build Knowledge Graph (Fast) - 68% to 95%
    if progress_callback: progress_callback(70, 100, "Building Knowledge Graph...")
    logger.info("Step 4/5: Building Knowledge Graph...")
    
    database.clear_graph()
    graph_nodes = []
    graph_edges = []
    
    # Group chunks by filepath
    doc_chunks_map = {}
    for i, chunk in enumerate(all_chunks):
        filepath = chunk['filepath']
        if filepath not in doc_chunks_map:
            doc_chunks_map[filepath] = []
        doc_chunks_map[filepath].append(i)
        
    doc_embeddings = {}
    from collections import Counter
    
    # Simple stop words for keyword extraction
    STOP_WORDS = set(["the", "and", "a", "to", "of", "in", "i", "is", "that", "it", "on", "you", "this", "for", "but", "with", "are", "have", "be", "at", "or", "as", "was", "so", "if", "out", "not", "we", "my", "from", "by", "an"])
    
    seen_keywords = set()
    for filepath, indices in doc_chunks_map.items():
        # Average chunk embeddings to get doc embedding
        chunk_embs = [chunk_embeddings[i] for i in indices if i < len(chunk_embeddings)]
        if chunk_embs:
            doc_emb = np.mean(chunk_embs, axis=0)
            doc_embeddings[filepath] = doc_emb

        # Add Node for Document
        filename = os.path.basename(filepath)
        graph_nodes.append({
            "id": filepath,
            "type": "document",
            "label": filename,
            "metadata": json.dumps({
                "file_type": os.path.splitext(filepath)[1].lower(),
                "chunks": len(indices),
            })
        })

        # Extract Keywords (TF-like approach)
        doc_text = " ".join([chunk_strings[i] for i in indices])
        tokens = tokenize(doc_text)
        filtered_tokens = [t for t in tokens if len(t) > 3 and t not in STOP_WORDS]
        counts = Counter(filtered_tokens)
        top_keywords = counts.most_common(5)

        for kw, kw_count in top_keywords:
            kw_id = f"kw_{kw}"
            if kw_id not in seen_keywords:
                seen_keywords.add(kw_id)
                graph_nodes.append({
                    "id": kw_id,
                    "type": "keyword",
                    "label": kw,
                    "metadata": "{}"
                })
            # Add Edge Document -> Keyword (weight = term frequency)
            graph_edges.append({
                "source_id": filepath,
                "target_id": kw_id,
                "weight": float(kw_count),
                "relation_type": "mentions"
            })

    # Compute Document Similarity Edges (vectorized cosine, top-k neighbors).
    # A fixed high threshold (e.g. 0.85) almost never fires between mean-pooled
    # documents, leaving the graph edge-less; top-k with a floor keeps it useful.
    filepaths = list(doc_embeddings.keys())
    if len(filepaths) > 1:
        emb_matrix = np.array([doc_embeddings[fp] for fp in filepaths], dtype='float32')
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = emb_matrix / norms
        sim_matrix = normalized @ normalized.T
        np.fill_diagonal(sim_matrix, -1.0)

        # Calibrated for MiniLM-class mean-pooled doc vectors: related topics
        # score ~0.35-0.55, unrelated docs < 0.3.
        _SIM_FLOOR = 0.35
        _TOP_K = 3
        seen_pairs = set()
        for i, fp1 in enumerate(filepaths):
            neighbor_idxs = np.argsort(sim_matrix[i])[::-1][:_TOP_K]
            for j in neighbor_idxs:
                sim = float(sim_matrix[i][j])
                if sim < _SIM_FLOOR:
                    continue
                pair = tuple(sorted((i, int(j))))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                fp2 = filepaths[int(j)]
                graph_edges.append({
                    "source_id": fp1,
                    "target_id": fp2,
                    "weight": sim,
                    "relation_type": "similar_to"
                })

    database.add_graph_data(graph_nodes, graph_edges)
    logger.info(f"Knowledge Graph created with {len(set(n['id'] for n in graph_nodes))} nodes and {len(graph_edges)} edges.")
    
    # Maintain existing return signature structure (dummy lists since we disabled RAPTOR)
    cluster_summaries = []
    final_cluster_map = {}
    
    # 8. Create FAISS Indices - 95% to 99%
    if progress_callback: progress_callback(97, 100, "Finalizing Indices...")
    logger.info("Finalizing Indices...")

    # Chunk Index
    try:
        chunk_emb_np = np.array(chunk_embeddings).astype('float32')
        if chunk_emb_np.ndim != 2:
            raise ValueError(f"ragged embedding matrix (shape {chunk_emb_np.shape})")
    except (ValueError, TypeError) as dim_err:
        logger.error(
            "Indexing aborted: cached vectors are incompatible with freshly "
            "embedded ones (%s). Delete data/index.faiss and re-index for a clean rebuild.",
            dim_err,
        )
        _clear_checkpoint()
        return None, None, None, None, None, None, None, {}
    index_chunks = faiss.IndexFlatL2(chunk_emb_np.shape[1])
    index_chunks.add(chunk_emb_np)
    
    # Summary Index
    if cluster_summaries:
        summary_embeddings = embeddings_model.embed_documents(cluster_summaries)
        summary_emb_np = np.array(summary_embeddings).astype('float32')
        index_summaries = faiss.IndexFlatL2(summary_emb_np.shape[1])
        index_summaries.add(summary_emb_np)
    else:
        index_summaries = None
    
    logger.info(f"Indexing Complete! Chunks: {len(chunk_strings)}, Clusters: {len(cluster_summaries)}.")
    logger.info(f"Total time: {time.time() - start_time:.2f}s")
    
    # Use empty tags list to maintain return signature
    tags = [""] * len(chunk_strings)

    # Store the embedding dimension so load_index can detect future model mismatches
    _embedding_dim = int(chunk_emb_np.shape[1])

    # Return both indices packaged (we'll need to modify save_index/load_index too)
    meta = {
        'model_name': _model_name,
        'embedding_dim': _embedding_dim,
        'chunker': _CHUNKER_VERSION,
    }
    # Atomically commit metadata: only clear old data after a successful build (#165)
    database.clear_files()
    if files_to_add:
        database.add_files_batch(files_to_add)
    database.clear_clusters()
    if clusters_batch_data:
        database.add_clusters_batch(clusters_batch_data)

    _clear_checkpoint()
    return index_chunks, all_chunks, tags, index_summaries, cluster_summaries, final_cluster_map, bm25, meta

def save_index(index_chunks: faiss.Index, all_chunks: List[Dict], tags: List[str], 
               filepath: str, index_summaries: faiss.Index = None,
               cluster_summaries: List[str] = None, cluster_map: Dict = None, 
               bm25: BM25Okapi = None, model_name: str = 'unknown', 
               embedding_dim: int = 0):
    """
    Persists the Dual FAISS + BM25 indices to disk.

    Saves vectors using FAISS binary format and metadata/BM25 using Pickle.
    A sidecar JSON file is also created to store model metadata for safety checks.

    Args:
        index_chunks (faiss.Index): The primary vector index.
        all_chunks (List[Dict]): Text chunks and associated file paths.
        tags (List[str]): User-defined tags (legacy).
        filepath (str): Destination path for the main .faiss file.
        index_summaries (faiss.Index, optional): The cluster summary vector index.
        cluster_summaries (List[str], optional): The LLM-generated summary texts.
        cluster_map (Dict, optional): Maps summary indices to chunk indices.
        bm25 (BM25Okapi, optional): The keyword search index.
        model_name (str): The name of the embedding model used.
        embedding_dim (int): The expected vector dimensionality.
    """
    faiss.write_index(index_chunks, filepath)
    base_path = os.path.splitext(filepath)[0]

    # ── Metadata sidecar ───────────────────────────────────────────────────
    # Stores the model name and the embedding dimension so the search layer can
    # detect a mismatch before querying FAISS (rather than crashing at runtime).
    dim = embedding_dim if embedding_dim else (index_chunks.d if index_chunks else 0)
    meta = {
        'model_name': model_name,
        'embedding_dim': dim,
        'chunker': _CHUNKER_VERSION,
    }
    with open(base_path + _META_SUFFIX, 'w') as f:
        json.dump(meta, f)
    # ───────────────────────────────────────────────────────────────────────

    # Use .pkl for everything as per AGENTS.md and for BM25 picklability
    with open(base_path + '_docs.pkl', 'wb') as f:
        pickle.dump(all_chunks, f)
    with open(base_path + '_tags.pkl', 'wb') as f:
        pickle.dump(tags, f)
        
    if index_summaries is not None:
        faiss.write_index(index_summaries, base_path + '_summary.index')
        with open(base_path + '_summaries.pkl', 'wb') as f:
            pickle.dump(cluster_summaries, f)
        with open(base_path + '_cluster_map.pkl', 'wb') as f:
            pickle.dump(cluster_map, f)
    else:
        # Remove stale summary artifacts from a previous run: load_index would
        # otherwise pair an old cluster_map with the NEW chunk ordering and
        # silently return wrong chunks for theme matches.
        for suffix in ('_summary.index', '_summaries.pkl', '_summaries.json',
                       '_cluster_map.pkl', '_cluster_map.json'):
            stale = base_path + suffix
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                    logger.info(f"Removed stale summary artifact: {stale}")
                except OSError as e:
                    logger.warning(f"Could not remove stale artifact {stale}: {e}")


    if bm25 is not None:
        with open(base_path + '_bm25.pkl', 'wb') as f:
            pickle.dump(bm25, f)
            
    logger.info(f"RAPTOR Index saved to {filepath} (Pickle format)")

def load_index(filepath: str) -> Tuple:
    """
    Loads the hierarchical RAG index from disk.

    Attempts to load FAISS indices, documentation chunks, and BM25 data. 
    Includes fallback logic for legacy JSON metadata and handles potential 
    PICKLE deserialization errors gracefully.

    Args:
        filepath (str): Path to the main .faiss index file.

    Returns:
        Tuple: (index_chunks, all_chunks, tags, index_summaries, cluster_summaries, cluster_map, bm25, meta)
            Returns empty values and an empty meta dict if the file is missing.
    """
    if not os.path.exists(filepath):
        return None, None, None, None, None, None, None, {}

    index_chunks = faiss.read_index(filepath)
    base_path = os.path.splitext(filepath)[0]

    # ── Load metadata sidecar (non-fatal if missing for legacy indices) ────
    meta = {}
    meta_path = base_path + _META_SUFFIX
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            logger.info(f"[Index] Warning: could not read metadata sidecar: {e}")
    # If no sidecar, infer dimension from the loaded index
    if 'embedding_dim' not in meta:
        meta['embedding_dim'] = index_chunks.d
    if 'model_name' not in meta:
        meta['model_name'] = 'unknown'
    # ───────────────────────────────────────────────────────────────────────

    docs_path_pkl = base_path + '_docs.pkl'
    docs_path_json = base_path + '_docs.json'
    tags_path_pkl = base_path + '_tags.pkl'
    tags_path_json = base_path + '_tags.json'

    all_chunks = []
    tags = []
    
    try:
        if os.path.exists(docs_path_pkl):
            with open(docs_path_pkl, 'rb') as f:
                all_chunks = pickle.load(f)
        elif os.path.exists(docs_path_json):
            with open(docs_path_json, 'r') as f:
                all_chunks = json.load(f)
        else:
            logger.info(f"Warning: No docs metadata found at {base_path}")
            return index_chunks, [], [], None, None, None, None, meta

        if os.path.exists(tags_path_pkl):
            with open(tags_path_pkl, 'rb') as f:
                tags = pickle.load(f)
        elif os.path.exists(tags_path_json):
            with open(tags_path_json, 'r') as f:
                tags = json.load(f)
    except Exception as e:
        logger.info(f"Error loading metadata: {e}")
        return index_chunks, [], [], None, None, None, None, meta
        
    index_summaries = None
    cluster_summaries = None
    cluster_map = None
    bm25 = None
    
    summary_idx_path = base_path + '_summary.index'
    if os.path.exists(summary_idx_path):
        try:
            index_summaries = faiss.read_index(summary_idx_path)
            
            # Load summaries
            sum_path_pkl = base_path + '_summaries.pkl'
            sum_path_json = base_path + '_summaries.json'
            if os.path.exists(sum_path_pkl):
                with open(sum_path_pkl, 'rb') as f:
                    cluster_summaries = pickle.load(f)
            elif os.path.exists(sum_path_json):
                with open(sum_path_json, 'r') as f:
                    cluster_summaries = json.load(f)
                    
            # Load cluster map
            map_path_pkl = base_path + '_cluster_map.pkl'
            map_path_json = base_path + '_cluster_map.json'
            if os.path.exists(map_path_pkl):
                with open(map_path_pkl, 'rb') as f:
                    cluster_map = pickle.load(f)
            elif os.path.exists(map_path_json):
                with open(map_path_json, 'r') as f:
                    cluster_map_raw = json.load(f)
                    cluster_map = {int(k): v for k, v in cluster_map_raw.items()}
        except Exception as e:
            logger.info(f"Error loading summary index: {e}")
                
    # Reconstruct or load BM25
    bm25_path = base_path + '_bm25.pkl'
    if os.path.exists(bm25_path):
        try:
            with open(bm25_path, 'rb') as f:
                bm25 = pickle.load(f)
            logger.info("Loaded BM25 from disk.")
        except Exception as e:
            logger.warning(f"BM25 index load failed ({type(e).__name__}: {e}); will reconstruct from corpus.")
             
    if bm25 is None and all_chunks:
        logger.info("Reconstructing BM25 Index...")
        chunk_strings = [chunk['text'] for chunk in all_chunks]
        tokenized_corpus = [tokenize(doc) for doc in chunk_strings]
        bm25 = BM25Okapi(tokenized_corpus)

    logger.info(f"Loaded RAPTOR Index: {len(all_chunks)} chunks, {len(cluster_summaries) if cluster_summaries else 0} clusters.")
    return index_chunks, all_chunks, tags, index_summaries, cluster_summaries, cluster_map, bm25, meta
