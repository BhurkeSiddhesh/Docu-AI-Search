import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import database
from backend.search import search
from backend.indexing import load_index
from backend.llm_integration import get_embeddings

QUERY = "which companies did siddhesh worked from fractal for providing services?"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(PROJECT_ROOT, 'data', 'index.faiss')

def debug_retrieval():
    """
    Performs a deep-dive search for a specific complex query and logs the results.

    This debugging tool:
    1. Loads the current FAISS index and BM25 metadata.
    2. Executes a search using the local embedding provider.
    3. Retrieves file metadata (like filenames) from the database for each search result.
    4. Writes detailed results (top 5) to 'retrieval_debug.txt' for manual inspection.
    """
    if not os.path.exists(INDEX_PATH):
        print("Index not found.")
        return

    res = load_index(INDEX_PATH)
    index, docs, tags, index_summaries, cluster_summaries, cluster_map, bm25 = res[:7]
    embeddings = get_embeddings(provider='local')
    
    results, context_snippets = search(
        QUERY, index, docs, tags, embeddings,
        index_summaries, cluster_summaries, cluster_map, bm25
    )
    
    with open("retrieval_debug.txt", "w", encoding="utf-8") as f:
        f.write(f"DEBUGGING RETRIEVAL FOR: {QUERY}\n\n")
        for i, res in enumerate(results[:5]):
            faiss_idx = res.get('faiss_idx')
            file_info = database.get_file_by_faiss_index(faiss_idx) if faiss_idx is not None else None
            filename = file_info['filename'] if file_info else "Unknown"
            
            output = f"--- Result {i+1} ---\n"
            output += f"File: {filename}\n"
            output += f"Content: {res['document']}\n"
            output += "-" * 20 + "\n\n"
            print(output[:500] + "...") # print summary to console
            f.write(output)
    print("\nFull debug output written to retrieval_debug.txt")

if __name__ == "__main__":
    database.init_database()
    debug_retrieval()
