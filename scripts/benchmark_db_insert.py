import time
import os
import sqlite3
from datetime import datetime
from backend import database

# Setup test database path
database.DATABASE_PATH = 'benchmark_test.db'
if os.path.exists(database.DATABASE_PATH):
    os.remove(database.DATABASE_PATH)
database.init_database()

def benchmark_single_inserts(n=1000):
    print(f"Benchmarking {n} single inserts...")
    start_time = time.time()
    for i in range(n):
        database.add_file(
            path=f'/test/path/{i}.txt',
            filename=f'{i}.txt',
            extension='.txt',
            size_bytes=1000,
            modified_date=datetime.now(),
            chunk_count=10,
            faiss_start_idx=i*10,
            faiss_end_idx=i*10+9
        )
    end_time = time.time()
    duration = end_time - start_time
    print(f"Single inserts took {duration:.4f} seconds ({n/duration:.2f} inserts/sec)")
    return duration

def benchmark_batch_inserts(n=1000):
    print(f"Benchmarking {n} batch inserts...")
    files_to_insert = []
    for i in range(n):
        files_to_insert.append({
            'path': f'/test/path_batch/{i}.txt',
            'filename': f'{i}.txt',
            'extension': '.txt',
            'size_bytes': 1000,
            'modified_date': datetime.now(),
            'chunk_count': 10,
            'faiss_start_idx': i*10,
            'faiss_end_idx': i*10+9
        })

    start_time = time.time()
    database.batch_add_files(files_to_insert)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Batch inserts took {duration:.4f} seconds ({n/duration:.2f} inserts/sec)")
    return duration

if __name__ == "__main__":
    t1 = benchmark_single_inserts(1000)
    t2 = benchmark_batch_inserts(1000)

    print(f"\nImprovement: {t1/t2:.2f}x faster")

    # Clean up
    if os.path.exists(database.DATABASE_PATH):
        os.remove(database.DATABASE_PATH)
