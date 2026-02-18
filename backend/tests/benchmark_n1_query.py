import time
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import database

def setup_mock_db():
    # Initialize a temporary database
    database.DATABASE_PATH = "test_metadata.db"
    if os.path.exists(database.DATABASE_PATH):
        os.remove(database.DATABASE_PATH)
    database.init_database()

    # Add some dummy files
    for i in range(100):
        database.add_file(
            path=f"/path/to/file_{i}.txt",
            filename=f"file_{i}.txt",
            extension=".txt",
            size_bytes=1000,
            modified_date="2023-01-01",
            chunk_count=10,
            faiss_start_idx=i*10,
            faiss_end_idx=i*10 + 9
        )

def measure_n1(indices):
    start = time.time()
    results = []
    for idx in indices:
        file_info = database.get_file_by_faiss_index(idx)
        results.append(file_info)
    end = time.time()
    return end - start, results

def get_files_by_faiss_indices_optimized(faiss_indices):
    if not faiss_indices:
        return {}

    conn = database.get_connection()
    cursor = conn.cursor()

    # Simple batch query approach: find all files that cover any of these indices.
    # We can use OR clauses for a small number of indices.
    clauses = ["(faiss_start_idx <= ? AND faiss_end_idx >= ?)" for _ in faiss_indices]
    params = []
    for idx in faiss_indices:
        params.extend([idx, idx])

    query = f"SELECT * FROM files WHERE {' OR '.join(clauses)}"
    cursor.execute(query, params)
    rows = cursor.fetchall()

    # Map them back
    file_map = {}
    for row in rows:
        row_dict = dict(row)
        # A file can cover multiple indices in our list
        for idx in faiss_indices:
            if row_dict['faiss_start_idx'] <= idx <= row_dict['faiss_end_idx']:
                file_map[idx] = row_dict

    return file_map

def measure_batch(indices):
    start = time.time()
    results_map = database.get_files_by_faiss_indices(indices)
    results = [results_map.get(idx) for idx in indices]
    end = time.time()
    return end - start, results

if __name__ == "__main__":
    setup_mock_db()

    # Typical search results size
    test_indices = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

    print(f"Measuring performance for {len(test_indices)} lookups...")

    # Warm up
    measure_n1(test_indices)
    measure_batch(test_indices)

    n1_time, _ = measure_n1(test_indices)
    batch_time, _ = measure_batch(test_indices)

    print(f"N+1 time: {n1_time:.6f}s")
    print(f"Batch time: {batch_time:.6f}s")
    print(f"Improvement: {(n1_time / batch_time):.2f}x")

    # Cleanup
    if os.path.exists(database.DATABASE_PATH):
        os.remove(database.DATABASE_PATH)
