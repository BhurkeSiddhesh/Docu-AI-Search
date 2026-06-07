import logging
import time
from typing import List, Dict, Any

from backend.llm_integration import generate_ai_answer

logger = logging.getLogger(__name__)

# Cache for query rewriting; capped at _CACHE_MAX entries (LRU-style eviction)
_QUERY_REWRITE_CACHE: Dict[str, str] = {}
_CACHE_MAX = 500

def rewrite_query(query: str, provider: str, api_key: str, model_path: str = "") -> str:
    """
    Uses an LLM to transform a conversational query into search keywords.

    This optimization step extracts core entities and intent while removing
    conversational filler, leading to higher quality vector and BM25 matches.

    Args:
        query (str): The raw conversational user query.
        provider (str): The LLM provider to use for rewriting (e.g., 'openai').
        api_key (str): API key for the LLM provider.
        model_path (str, optional): Path to the local model if using a local provider.

    Returns:
        str: An optimized, keyword-dense search query string.

    Note:
        The function uses a bounded memory cache (max 500 entries) to avoid
        redundant LLM calls for identical queries in the same session.
    """
    if query in _QUERY_REWRITE_CACHE:
        return _QUERY_REWRITE_CACHE[query]

    system_instruction = (
        "You are an expert search engine query optimizer. "
        "Convert the user's conversational question into a highly effective, keyword-dense search query. "
        "Extract the core entities, intent, and implied context. "
        "Remove filler words (like 'how', 'what', 'can you', 'tell me'). "
        "Return ONLY the optimized search keywords on a single line. Do not explain."
    )

    # We want a very fast response, so limit tokens and use low temp
    try:
        start_time = time.time()
        logger.debug("Rewriting query: '%s'", query)
        rewritten = generate_ai_answer(
            context="",
            question=query,
            provider=provider,
            api_key=api_key,
            model_path=model_path,
            raw=True,
            system_instruction=system_instruction,
            max_tokens=64,
            temperature=0.1
        )
        rewritten = rewritten.strip()
        elapsed = time.time() - start_time
        logger.debug("Query rewritten to: '%s' (%.2fs)", rewritten, elapsed)

        # Evict oldest entry if cache is full, then store
        if len(_QUERY_REWRITE_CACHE) >= _CACHE_MAX:
            oldest_key = next(iter(_QUERY_REWRITE_CACHE))
            del _QUERY_REWRITE_CACHE[oldest_key]
        _QUERY_REWRITE_CACHE[query] = rewritten
        return rewritten
    except Exception as e:
        logger.warning("Query rewrite failed: %s. Falling back to original query.", e)
        return query


# Global instance to avoid reloading the re-ranker on every search
_RERANKER_CACHE = {}

def rerank_results(query: str, chunks: List[Dict[str, Any]], reranker_model_name: str) -> List[Dict[str, Any]]:
    """
    Re-scores and re-orders search results using a Cross-Encoder model.

    Unlike Bi-Encoders (used for initial search), Cross-Encoders perform a
    deep semantic comparison between the query and each document chunk
    simultaneously, providing much higher ranking precision.

    Args:
        query (str): The search query used for comparison.
        chunks (List[Dict[str, Any]]): The list of retrieved chunks from initial search.
        reranker_model_name (str): The model name/HuggingFace ID for the Cross-Encoder.

    Returns:
        List[Dict[str, Any]]: The re-ordered list of chunks, sorted by `rerank_score`.

    Note:
        This step is computationally more expensive than initial retrieval.
        It is typically applied to a small candidate pool (e.g., top 20 results).
    """
    if not chunks:
        return []

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning("sentence-transformers not installed. Skipping re-ranking.")
        return chunks

    if reranker_model_name not in _RERANKER_CACHE:
        logger.info("Loading Cross-Encoder: %s", reranker_model_name)
        try:
            # Load locally or download automatically
            _RERANKER_CACHE[reranker_model_name] = CrossEncoder(reranker_model_name)
        except Exception as e:
            logger.error("Failed to load re-ranker model: %s", e)
            return chunks

    reranker = _RERANKER_CACHE[reranker_model_name]

    # Prepare inputs: list of (query, document) pairs
    pairs = [[query, chunk['document']] for chunk in chunks]

    logger.debug("Re-ranking %d candidate chunks", len(chunks))
    start_time = time.time()

    try:
        scores = reranker.predict(pairs)

        # Inject the new cross-encoder score and sort
        for i, chunk in enumerate(chunks):
            # We preserve the original FAISS/BM25 score, but sort by this one
            chunk['rerank_score'] = float(scores[i])

        # Higher score is better in Cross-Encoders
        ranked_chunks = sorted(chunks, key=lambda x: x.get('rerank_score', -999.0), reverse=True)

        elapsed = time.time() - start_time
        logger.debug("Re-ranking complete in %.2fs", elapsed)
        return ranked_chunks

    except Exception as e:
        logger.error("Re-ranking failed during prediction: %s", e)
        return chunks
