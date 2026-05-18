import os
import faiss
import hashlib
import json
import logging
import configparser
import pickle
import numpy as np
import concurrent.futures
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
from langchain_text_splitters import CharacterTextSplitter

logger = logging.getLogger(__name__)
from backend.llm_integration import get_embeddings, get_tags, smart_summary, summarize
from backend.file_processing import extract_text
from backend import database
from backend.clustering import perform_global_clustering
from rank_bm25 import BM25Okapi
import string

# Metadata sidecar filename suffix
_META_SUFFIX = '_meta.json'

# Checkpoint file for resume-on-failure support
_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'index_checkpoint.json')

# Save the extraction checkpoint every N files instead of every file. The
# checkpoint exists to allow resume after a crash mid-extraction; fsync'ing on
# every file produced hundreds of disk writes for typical corpora and gave the
# UI a "stuck" appearance during indexing.
_CHECKPOINT_EVERY = 25

# Path to the project config.ini, used to read the cluster-summarization flag.
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.ini')


def _load_checkpoint() -> dict:
    if os.path.exists(_CHECKPOINT_PATH):
        try:
            with open(_CHECKPOINT_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_checkpoint(checkpoint: dict):
    os.makedirs(os.path.dirname(_CHECKPOINT_PATH), exist_ok=True)
    with open(_CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f)


def _clear_checkpoint():
    if os.path.exists(_CHECKPOINT_PATH):
        os.remove(_CHECKPOINT_PATH)

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

def _compute_file_hash(filepath: str) -> Optional[str]:
    """
    Compute a streaming sha256 of the file bytes so incremental indexing can
    detect content changes without re-embedding. Returns None if the file
    can't be read (e.g. permission error, broken symlink).
    """
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        logger.info(f"[Index] Could not hash {filepath}: {e}")
        return None


def _clustering_enabled() -> bool:
    """
    Read [AdvancedRAG] cluster_summarization from config.ini. Default is False:
    cluster summarization makes one LLM call per cluster and is the single
    biggest cost on full-rebuilds. Users who want RAPTOR-style multi-level
    retrieval can opt back in via Settings.
    """
    config = configparser.ConfigParser()
    try:
        config.read(_CONFIG_PATH)
        return config.getboolean('AdvancedRAG', 'cluster_summarization', fallback=False)
    except Exception:
        return False


def _read_existing_meta(index_path: Optional[str]) -> Dict[str, Any]:
    """
    Peek at the _meta.json sidecar next to an on-disk index, so an incremental
    run that reuses all chunks can still report the original model_name without
    instantiating an embedding client.
    """
    if not index_path:
        return {}
    base = os.path.splitext(index_path)[0]
    meta_path = base + _META_SUFFIX
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


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

def create_index(folder_paths: List[str] | str, provider: str, api_key: str = None,
                 model_path: str = None, progress_callback: callable = None,
                 embedding_client: Any = None,
                 existing_index_path: Optional[str] = None) -> Tuple:
    """
    Build (or incrementally update) the hybrid FAISS + BM25 index for the given folders.

    Incremental behavior: when ``existing_index_path`` is provided and points to
    an existing index on disk, files whose sha256 content hash matches the row
    in the DB are *reused* — their chunks and embeddings are copied from the
    existing index without re-running extraction or calling the embedding API.
    Only new and changed files go through the full extract → chunk → embed
    pipeline. The final FAISS index is rebuilt fresh from `reused + new` so
    the index file on disk always reflects the current folder state.

    Cluster summarization (the LLM-heavy RAPTOR step) is gated behind
    ``[AdvancedRAG] cluster_summarization`` in config.ini and defaults to off.

    Args:
        folder_paths (List[str] | str): One or more directories to scan.
        provider (str): LLM provider for summarization.
        api_key (str, optional): Secret key for cloud providers.
        model_path (str, optional): Path to GGUF file for local models.
        progress_callback (callable, optional): function(percent, total, status) for UI updates.
        embedding_client (Any, optional): A pre-resolved LangChain embedding client.
        existing_index_path (str, optional): Path to a prior on-disk index whose
            unchanged-file vectors should be reused. None disables reuse (full rebuild).

    Returns:
        Tuple: (index_chunks, all_chunks, tags, index_summaries, cluster_summaries,
                final_cluster_map, bm25, meta).
            Returns the empty 8-tuple if no files are found or every embedding batch fails.
    """
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]

    logger.info(f"Starting RAPTOR Indexing of folders: {folder_paths}")
    start_time = time.time()

    # 1. Collect files in the requested folders.
    all_files: List[str] = []
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            for dirpath, _, filenames in os.walk(folder_path):
                for filename in filenames:
                    all_files.append(os.path.join(dirpath, filename))

    logger.info(f"Found {len(all_files)} total files.")
    if not all_files:
        database.clear_files()
        database.clear_clusters()
        return None, None, None, None, None, None, None, {}

    # 2. Hash files and load the prior DB state for the incremental classification.
    #    We do this BEFORE clearing DB rows so we can compare against last run.
    if progress_callback:
        progress_callback(1, 100, "Checking for changes...")
    current_hashes: Dict[str, Optional[str]] = {f: _compute_file_hash(f) for f in all_files}
    indexed_state = database.get_indexed_state()

    # 3. Optionally load the existing FAISS index so we can reuse vectors for
    #    unchanged files. If it can't be loaded, we silently fall back to a
    #    full rebuild — this is also the path on the very first indexing run.
    existing_index = None
    existing_docs: Optional[List[Dict]] = None
    if existing_index_path and os.path.exists(existing_index_path):
        try:
            loaded = load_index(existing_index_path)
            existing_index = loaded[0]
            existing_docs = loaded[1]
            if existing_index is None or existing_docs is None:
                existing_index, existing_docs = None, None
        except Exception as e:
            logger.info(f"[Index] Could not load existing index for reuse: {e}")
            existing_index, existing_docs = None, None

    can_reuse = existing_index is not None and existing_docs is not None

    # 4. Classify each file as new / changed / unchanged based on content_hash.
    #    Unchanged-but-can't-reuse (no existing index) is treated as new so we
    #    end up doing a full rebuild without special-casing the empty path.
    new_files: List[str] = []
    changed_files: List[str] = []
    unchanged_specs: List[Tuple[str, int, int]] = []  # (path, faiss_start_idx, faiss_end_idx)
    for path in all_files:
        prior = indexed_state.get(path)
        cur_hash = current_hashes.get(path)
        if (
            can_reuse
            and prior
            and prior['content_hash'] is not None
            and cur_hash is not None
            and prior['content_hash'] == cur_hash
            and prior['faiss_start_idx'] is not None
            and prior['faiss_end_idx'] is not None
        ):
            unchanged_specs.append((path, prior['faiss_start_idx'], prior['faiss_end_idx']))
        elif prior and prior['content_hash'] is not None:
            changed_files.append(path)
        else:
            new_files.append(path)

    files_to_process = new_files + changed_files

    logger.info(
        "Incremental classification: %d new, %d changed, %d unchanged (reused).",
        len(new_files), len(changed_files), len(unchanged_specs),
    )
    if progress_callback:
        progress_callback(
            3, 100,
            f"{len(unchanged_specs)} unchanged · {len(new_files)} new · {len(changed_files)} changed",
        )

    # 5. Rewrite DB rows from scratch with the new layout. Cluster summaries
    #    are repopulated below if the feature is enabled.
    database.clear_files()
    database.clear_clusters()

    all_chunks: List[Dict] = []
    chunk_strings: List[str] = []
    chunk_embeddings: List = []
    files_to_add_db: List[Dict] = []
    current_faiss_idx = 0

    # 6. Reuse chunks + vectors for unchanged files. `existing_index.reconstruct_n`
    #    pulls each file's contiguous run of vectors in one numpy slice.
    for path, old_start, old_end in unchanged_specs:
        if old_start is None or old_end is None or old_start > old_end:
            continue
        n_chunks = old_end - old_start + 1
        if old_end >= len(existing_docs) or old_end >= existing_index.ntotal:
            # Existing on-disk state is inconsistent with current_state — fall
            # back to re-extracting this file rather than silently corrupting
            # the new index.
            files_to_process.append(path)
            continue
        try:
            file_stat = os.stat(path)
        except OSError as e:
            logger.info(f"[Index] Skipping unchanged-stat for {path}: {e}")
            continue
        try:
            reused_vecs = existing_index.reconstruct_n(old_start, n_chunks)
        except Exception as e:
            logger.info(f"[Index] Could not reconstruct vectors for {path}: {e}; re-extracting.")
            files_to_process.append(path)
            continue
        file_start = current_faiss_idx
        for offset in range(n_chunks):
            old_chunk = existing_docs[old_start + offset]
            new_chunk = dict(old_chunk) if isinstance(old_chunk, dict) else {'text': str(old_chunk), 'filepath': path}
            new_chunk['faiss_idx'] = current_faiss_idx
            all_chunks.append(new_chunk)
            chunk_strings.append(new_chunk.get('text', ''))
            chunk_embeddings.append(reused_vecs[offset])
            current_faiss_idx += 1
        files_to_add_db.append({
            'path': path,
            'filename': os.path.basename(path),
            'file_type': os.path.splitext(path)[1].lower(),
            'size': file_stat.st_size,
            'last_modified': file_stat.st_mtime,
            'faiss_start_idx': file_start,
            'faiss_end_idx': current_faiss_idx - 1,
            'tags': '[]',
            'content_hash': current_hashes[path],
        })

    if files_to_add_db:
        database.add_files_batch(files_to_add_db)
        files_to_add_db = []

    # 7. Extract + embed new/changed files. The embedding client is only built
    #    if we actually need it, so an all-unchanged run doesn't hit the API.
    embeddings_model = None
    if files_to_process:
        logger.info("Step 1/5: Extracting Text (Parallel)...")
        valid_docs: List[Tuple[str, str]] = []

        checkpoint = _load_checkpoint()
        files_to_extract = [f for f in files_to_process if f not in checkpoint]
        for cached_path, cached_text in checkpoint.items():
            if cached_path in files_to_process and cached_text:
                valid_docs.append((cached_path, cached_text))

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(safe_extract_text, f): f for f in files_to_extract}
            complete_count = len(checkpoint)
            total_files = max(len(files_to_process), 1)
            ckpt_dirty = 0
            for future in concurrent.futures.as_completed(future_to_file):
                filepath, text = future.result()
                if text:
                    valid_docs.append((filepath, text))
                checkpoint[filepath] = text or ""
                ckpt_dirty += 1
                # Batched checkpoint persistence: a crash loses at most
                # _CHECKPOINT_EVERY files of extraction work, in exchange for
                # dropping hundreds of fsyncs per run.
                if ckpt_dirty >= _CHECKPOINT_EVERY:
                    _save_checkpoint(checkpoint)
                    ckpt_dirty = 0
                complete_count += 1
                if progress_callback:
                    percent = 5 + int((complete_count / total_files) * 15)  # 5%..20%
                    progress_callback(percent, 100, f"Extracting: {os.path.basename(filepath)}")
            if ckpt_dirty > 0:
                _save_checkpoint(checkpoint)

        logger.info(f"Successfully extracted text from {len(valid_docs)} files.")

        if progress_callback:
            progress_callback(22, 100, "Chunking text...")
        logger.info("Step 2/5: Chunking Text...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        new_chunks_by_file: List[Tuple[str, List[str]]] = []
        new_chunk_strings: List[str] = []
        for filepath, text in valid_docs:
            chunks = text_splitter.split_text(text)
            if not chunks:
                continue
            new_chunks_by_file.append((filepath, chunks))
            new_chunk_strings.extend(chunks)

        if new_chunk_strings:
            if progress_callback:
                progress_callback(25, 100, "Starting embeddings...")
            logger.info("Step 3/5: Embedding Chunks (Parallel)...")
            if embedding_client is not None:
                embeddings_model = embedding_client
                logger.info("[Index] Using pre-resolved embedding client from app.state.")
            else:
                embeddings_model = get_embeddings(provider, api_key, model_path)

            batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
            batches = [new_chunk_strings[i:i + batch_size] for i in range(0, len(new_chunk_strings), batch_size)]
            chunk_embeddings_map: Dict[int, List] = {}

            # Cloud embedding APIs are I/O-bound; 10 workers keeps the API
            # saturated without thrashing local CPU.
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_map = {
                    executor.submit(embeddings_model.embed_documents, batch): i
                    for i, batch in enumerate(batches)
                }
                completed = 0
                total_batches = max(len(batches), 1)
                for future in concurrent.futures.as_completed(future_map):
                    batch_idx = future_map[future]
                    try:
                        chunk_embeddings_map[batch_idx] = future.result()
                    except Exception as e:
                        logger.info(f"Error embedding batch {batch_idx}: {e}")
                        chunk_embeddings_map[batch_idx] = []
                    completed += 1
                    if progress_callback:
                        percent = 25 + int((completed / total_batches) * 40)
                        progress_callback(percent, 100, f"Embedding batch {completed}/{total_batches}")

            new_chunk_embeddings: List = []
            for i in range(len(batches)):
                if i in chunk_embeddings_map:
                    new_chunk_embeddings.extend(chunk_embeddings_map[i])

            # Fail fast on embedding failures rather than silently produce a
            # corrupt index. See the previous "embedding/chunk alignment guard"
            # commit (95fa775) for the failure mode this protects against.
            if not new_chunk_embeddings:
                logger.error(
                    "Indexing aborted: every embedding batch failed (0/%d). "
                    "Check the embedding provider/API key.",
                    len(batches),
                )
                _clear_checkpoint()
                return None, None, None, None, None, None, None, {}

            if len(new_chunk_embeddings) != len(new_chunk_strings):
                logger.error(
                    "Indexing aborted: embedding/chunk count mismatch (%d embeddings vs %d chunks). "
                    "Some batches failed — aborting to avoid a misaligned index.",
                    len(new_chunk_embeddings),
                    len(new_chunk_strings),
                )
                _clear_checkpoint()
                return None, None, None, None, None, None, None, {}

            emb_iter = iter(new_chunk_embeddings)
            for path, chunks in new_chunks_by_file:
                try:
                    file_stat = os.stat(path)
                except OSError as e:
                    logger.info(f"[Index] Skipping stat for {path}: {e}")
                    # Drain the iterator for this file's chunks so subsequent
                    # files don't pull mis-aligned embeddings.
                    for _ in chunks:
                        next(emb_iter)
                    continue
                file_start = current_faiss_idx
                for chunk in chunks:
                    all_chunks.append({
                        'text': chunk,
                        'filepath': path,
                        'faiss_idx': current_faiss_idx,
                        'file_id': None,
                    })
                    chunk_strings.append(chunk)
                    chunk_embeddings.append(next(emb_iter))
                    current_faiss_idx += 1
                files_to_add_db.append({
                    'path': path,
                    'filename': os.path.basename(path),
                    'file_type': os.path.splitext(path)[1].lower(),
                    'size': file_stat.st_size,
                    'last_modified': file_stat.st_mtime,
                    'faiss_start_idx': file_start,
                    'faiss_end_idx': current_faiss_idx - 1,
                    'tags': '[]',
                    'content_hash': current_hashes[path],
                })
            if files_to_add_db:
                database.add_files_batch(files_to_add_db)

    logger.info(
        f"Total chunks: {len(chunk_strings)} ({len(unchanged_specs)} files reused, "
        f"{len(new_files) + len(changed_files)} processed)."
    )

    if not chunk_strings:
        logger.info("Warning: No text chunks found in provided files.")
        return None, None, None, None, None, None, None, {}

    # 8. BM25 (always rebuilt — in-memory and cheap).
    if progress_callback:
        progress_callback(66, 100, "Building Keyword Index...")
    logger.info("Step 3.5/5: Building BM25 Index...")
    tokenized_corpus = [tokenize(doc) for doc in chunk_strings]
    bm25 = BM25Okapi(tokenized_corpus)

    # 9. Cluster summarization is gated behind config.ini. Off by default
    #    because each cluster fires an LLM call and these dominate wall time.
    cluster_summaries: List[str] = []
    final_cluster_map: Dict[int, List[int]] = {}
    index_summaries = None

    if _clustering_enabled():
        if progress_callback:
            progress_callback(70, 100, "Clustering content...")
        logger.info("Step 4/5: Performing Global Clustering...")
        cluster_map = perform_global_clustering(chunk_embeddings, max_cluster_size=20)
        logger.info(f"Created {len(cluster_map)} global clusters.")

        if progress_callback:
            progress_callback(75, 100, "Summarizing clusters...")
        logger.info("Step 5/5: Summarizing Clusters (Parallel)...")

        def process_cluster(cluster_id, indices):
            cluster_text = "\n\n".join([chunk_strings[i] for i in indices])
            if provider == 'local':
                summary = summarize(cluster_text, provider, api_key, model_path, f"Cluster {cluster_id}")
            else:
                summary = smart_summary(
                    cluster_text,
                    "Summarize the key themes and facts in this collection.",
                    provider, api_key, model_path, file_name=f"Cluster {cluster_id}",
                )
            return cluster_id, summary

        total_clusters = len(cluster_map)
        processed_clusters = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_cid = {
                executor.submit(process_cluster, cid, idxs): cid
                for cid, idxs in cluster_map.items()
            }
            clusters_batch_data = []
            for future in concurrent.futures.as_completed(future_to_cid):
                cid, summary = future.result()
                if summary:
                    clusters_batch_data.append((summary, 1))
                    current_summary_idx = len(cluster_summaries)
                    cluster_summaries.append(summary)
                    final_cluster_map[current_summary_idx] = cluster_map[cid]
                processed_clusters += 1
                if progress_callback:
                    percent = 75 + int((processed_clusters / max(total_clusters, 1)) * 20)
                    progress_callback(percent, 100, f"Summarizing cluster {processed_clusters}/{total_clusters}")
            if clusters_batch_data:
                database.add_clusters_batch(clusters_batch_data)
    else:
        logger.info("Cluster summarization disabled (config: AdvancedRAG.cluster_summarization) — skipping for faster indexing.")
        if progress_callback:
            progress_callback(95, 100, "Finalizing Indices...")

    # 10. Build the final FAISS chunk index from the merged vector list.
    if progress_callback:
        progress_callback(97, 100, "Finalizing Indices...")
    logger.info("Finalizing Indices...")
    chunk_emb_np = np.array(chunk_embeddings).astype('float32')
    index_chunks = faiss.IndexFlatL2(chunk_emb_np.shape[1])
    index_chunks.add(chunk_emb_np)

    if cluster_summaries:
        # Need an embedding client for the summary vectors. If we never built
        # one (all files unchanged), instantiate it lazily now.
        if embeddings_model is None:
            if embedding_client is not None:
                embeddings_model = embedding_client
            else:
                embeddings_model = get_embeddings(provider, api_key, model_path)
        summary_embeddings = embeddings_model.embed_documents(cluster_summaries)
        summary_emb_np = np.array(summary_embeddings).astype('float32')
        index_summaries = faiss.IndexFlatL2(summary_emb_np.shape[1])
        index_summaries.add(summary_emb_np)

    logger.info(f"Indexing Complete! Chunks: {len(chunk_strings)}, Clusters: {len(cluster_summaries)}.")
    logger.info(f"Total time: {time.time() - start_time:.2f}s")

    tags = [""] * len(chunk_strings)
    _embedding_dim = int(chunk_emb_np.shape[1])
    if embeddings_model is not None:
        _model_name = getattr(embeddings_model, 'model_name', None) or getattr(embeddings_model, 'model', 'unknown')
    else:
        # All unchanged — keep the model_name from the existing on-disk meta.
        _model_name = _read_existing_meta(existing_index_path).get('model_name', 'unknown')

    meta = {
        'model_name': _model_name,
        'embedding_dim': _embedding_dim,
    }
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
        except:
             pass
             
    if bm25 is None and all_chunks:
        logger.info("Reconstructing BM25 Index...")
        chunk_strings = [chunk['text'] for chunk in all_chunks]
        tokenized_corpus = [tokenize(doc) for doc in chunk_strings]
        bm25 = BM25Okapi(tokenized_corpus)

    logger.info(f"Loaded RAPTOR Index: {len(all_chunks)} chunks, {len(cluster_summaries) if cluster_summaries else 0} clusters.")
    return index_chunks, all_chunks, tags, index_summaries, cluster_summaries, cluster_map, bm25, meta
