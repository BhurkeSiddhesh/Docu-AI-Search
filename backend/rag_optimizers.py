import time
from typing import List, Dict, Any

from backend.llm_integration import generate_ai_answer

# Cache for query rewriting to save latency on repeated identical queries
_QUERY_REWRITE_CACHE: Dict[str, str] = {}

def rewrite_query(query: str, provider: str, api_key: str, model_path: str = "") -> str:
    """
    Uses a fast LLM call to rewrite a conversational user query into
    an optimized, keyword-dense search query.
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
        print(f"[RAG OPTIMIZER] Rewriting Query: '{query}'")
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
        print(f"[RAG OPTIMIZER] Optimized to: '{rewritten}' (took {elapsed:.2f}s)")
        
        # Cache it
        _QUERY_REWRITE_CACHE[query] = rewritten
        return rewritten
    except Exception as e:
        print(f"[RAG OPTIMIZER] Query rewrite failed: {e}. Falling back to original query.")
        return query


# Global instance to avoid reloading the re-ranker on every search
_RERANKER_CACHE = {}

def rerank_results(query: str, chunks: List[Dict[str, Any]], reranker_model_name: str) -> List[Dict[str, Any]]:
    """
    Uses a Cross-Encoder to re-score and re-rank the retrieved chunks.
    It performs deep semantic comparison between the query and each chunk.
    """
    if not chunks:
        return []
        
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("[RAG OPTIMIZER] sentence-transformers not installed. Skipping re-ranking.")
        return chunks

    if reranker_model_name not in _RERANKER_CACHE:
        print(f"[RAG OPTIMIZER] Loading Cross-Encoder: {reranker_model_name}...")
        try:
            # Load locally or download automatically
            _RERANKER_CACHE[reranker_model_name] = CrossEncoder(reranker_model_name)
        except Exception as e:
            print(f"[RAG OPTIMIZER] Failed to load re-ranker model: {e}")
            return chunks

    reranker = _RERANKER_CACHE[reranker_model_name]
    
    # Prepare inputs: list of (query, document) pairs
    pairs = [[query, chunk['document']] for chunk in chunks]
    
    print(f"[RAG OPTIMIZER] Re-ranking {len(chunks)} candidate chunks...")
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
        print(f"[RAG OPTIMIZER] Re-ranking complete in {elapsed:.2f}s")
        return ranked_chunks
        
    except Exception as e:
        print(f"[RAG OPTIMIZER] Re-ranking failed during prediction: {e}")
        return chunks
