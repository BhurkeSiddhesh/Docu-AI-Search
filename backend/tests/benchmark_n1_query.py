import time
import os
import tempfile
import shutil

from backend import database

def setup_mock_db():
    """Initialize a temporary database and return cleanup function."""
    # Save original DATABASE_PATH
    original_path = database.DATABASE_PATH
    
    # Create temporary database in temp directory
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test_metadata.db")
    database.DATABASE_PATH = temp_db_path
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
    
    def cleanup():
        """Restore original DATABASE_PATH and remove temp directory."""
        database.DATABASE_PATH = original_path
        shutil.rmtree(temp_dir)
    
    return cleanup

def measure_n1(indices):
    start = time.time()
    results = []
    for idx in indices:
        file_info = database.get_file_by_faiss_index(idx)
        results.append(file_info)
    end = time.time()
    return end - start, results

def measure_batch(indices):
    start = time.time()
    results_map = database.get_files_by_faiss_indices(indices)
    results = [results_map.get(idx) for idx in indices]
    end = time.time()
    return end - start, results

if __name__ == "__main__":
    cleanup = setup_mock_db()
    
    try:
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
    finally:
        # Cleanup temp database and restore original path
        cleanup()
