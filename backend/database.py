import sqlite3
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

# Path configuration for new folder structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(DATA_DIR, 'metadata.db')

# Thread-local storage for database connections
_local_storage = threading.local()

class PooledConnection:
    """
    A wrapper around sqlite3.Connection to support thread-local pooling.
    Intercepts close() to keep the connection open and resets state on close.
    """
    def __init__(self, connection):
        self._connection = connection

    def close(self):
        """
        Don't actually close the connection. Just rollback any uncommitted transaction.
        This resets the connection state for the next user.
        """
        try:
            self._connection.rollback()
        except sqlite3.ProgrammingError:
            # Connection might be closed or failed
            pass

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def cursor(self, *args, **kwargs):
        return self._connection.cursor(*args, **kwargs)

    def execute(self, *args, **kwargs):
        return self._connection.execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._connection.executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        return self._connection.executescript(*args, **kwargs)

    def create_function(self, *args, **kwargs):
        return self._connection.create_function(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._connection.rollback()
        else:
            self._connection.commit()
        # Do not close the underlying connection

def get_connection():
    """Create and return a thread-local database connection."""
    global _local_storage

    current_pid = os.getpid()

    # Check if we have a connection for this thread
    if not hasattr(_local_storage, "connection"):
        _local_storage.connection = None
        _local_storage.db_path = None
        _local_storage.pid = None

    # Check if the connection is valid and points to the correct database path
    # AND is owned by the current process (handling forks)
    if (_local_storage.connection is None or
        _local_storage.db_path != DATABASE_PATH or
        getattr(_local_storage, 'pid', None) != current_pid):

        # Close old connection if it exists AND belongs to this process
        if _local_storage.connection and getattr(_local_storage, 'pid', None) == current_pid:
            try:
                _local_storage.connection.close()
            except Exception:
                pass

        # Create new connection
        _local_storage.connection = sqlite3.connect(DATABASE_PATH)
        _local_storage.db_path = DATABASE_PATH
        _local_storage.pid = current_pid

    # Ensure clean transaction state
    # If a previous user didn't close properly, the connection might be in a transaction.
    try:
        if _local_storage.connection.in_transaction:
            _local_storage.connection.rollback()
    except Exception:
        # If checking failed (e.g. connection closed externally), recreate
        _local_storage.connection = sqlite3.connect(DATABASE_PATH)
        _local_storage.db_path = DATABASE_PATH
        _local_storage.pid = current_pid

    # Always ensure row_factory is set correctly
    _local_storage.connection.row_factory = sqlite3.Row

    return PooledConnection(_local_storage.connection)

def init_database():
    """Initialize the database schema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Files table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            extension TEXT,
            size_bytes INTEGER,
            modified_date DATETIME,
            indexed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            chunk_count INTEGER DEFAULT 0,
            faiss_start_idx INTEGER,
            faiss_end_idx INTEGER
        )
    """)
    
    # Search history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            result_count INTEGER,
            execution_time_ms INTEGER
        )
    """)
    
    # Preferences table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Folder history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS folder_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_used_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_indexed BOOLEAN DEFAULT 0
        )
    """)

    # Response Cache table (Persistent AI Answers)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS response_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_hash TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            model_id TEXT NOT NULL,
            response_type TEXT NOT NULL,
            response_text TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            hit_count INTEGER DEFAULT 0,
            last_accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(query_hash, context_hash, model_id, response_type)
        )
    """)

    # Clusters table (RAPTOR)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT NOT NULL,
            level INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Index for fast cache lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_lookup ON response_cache(query_hash, model_id)")
    
    # NEW: Index for fast file-to-chunk range lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_faiss_range ON files(faiss_start_idx, faiss_end_idx)")
    
    # NEW: Index for fast filename lookups (used by agent and search)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_name ON files(filename)")

    # Populate history from existing files if empty
    cursor.execute("SELECT COUNT(*) FROM folder_history")
    if cursor.fetchone()[0] == 0:
        try:
            cursor.execute("SELECT path FROM files")
            files = cursor.fetchall()
            seen_folders = set()
            timestamp = datetime.now()
            
            for row in files:
                folder = os.path.dirname(row['path'])
                if folder and folder not in seen_folders:
                    seen_folders.add(folder)
                    cursor.execute("INSERT OR IGNORE INTO folder_history (path, added_at, last_used_at) VALUES (?, ?, ?)", 
                                  (folder, timestamp, timestamp))
            if seen_folders:
                print(f"Migrated {len(seen_folders)} folders to history.")
        except Exception as e:
            print(f"Migration failed: {e}")
            
    conn.commit()
    conn.close()

def add_file(path: str, filename: str, extension: str, size_bytes: int, 
             modified_date: datetime, chunk_count: int, 
             faiss_start_idx: int, faiss_end_idx: int) -> int:
    """Add a file to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO files 
        (path, filename, extension, size_bytes, modified_date, chunk_count, faiss_start_idx, faiss_end_idx)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (path, filename, extension, size_bytes, modified_date, chunk_count, faiss_start_idx, faiss_end_idx))
    
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def add_files_batch(files_data: List[Dict]):
    """Batch insert files for performance."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.executemany('''
            INSERT OR REPLACE INTO files
            (path, filename, extension, size_bytes, modified_date, chunk_count, faiss_start_idx, faiss_end_idx)
            VALUES (:path, :filename, :extension, :size_bytes, :modified_date, :chunk_count, :faiss_start_idx, :faiss_end_idx)
        ''', files_data)
        conn.commit()
    except Exception as e:
        print(f"Error adding batch files to DB: {e}")
    finally:
        conn.close()

def get_all_files(limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
    """Get all indexed files."""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM files ORDER BY indexed_date DESC, id DESC"
    params = []

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    elif offset > 0:
        query += " LIMIT -1"

    if offset > 0:
        query += " OFFSET ?"
        params.append(offset)

    cursor.execute(query, tuple(params))
    files = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return files

def get_file_by_path(path: str) -> Optional[Dict]:
    """Get a file by its path."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM files WHERE path = ?", (path,))
    row = cursor.fetchone()
    
    conn.close()
    return dict(row) if row else None

def get_file_by_name(filename: str) -> Optional[Dict]:
    """Get a file by its name."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM files WHERE filename = ?", (filename,))
    row = cursor.fetchone()
    
    conn.close()
    return dict(row) if row else None

def delete_file(file_id: int):
    """Delete a file from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
    
    conn.commit()
    conn.close()

def clear_files():
    """Clear all files from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM files")
    conn.commit()
    conn.close()

def clear_all_files():
    """Alias for clear_files."""
    clear_files()

def add_search_history(query: str, result_count: int, execution_time_ms: int):
    """Add a search to history."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO search_history (query, result_count, execution_time_ms)
        VALUES (?, ?, ?)
    """, (query, result_count, execution_time_ms))
    
    conn.commit()
    conn.close()

def get_search_history(limit: int = 20) -> List[Dict]:
    """Get recent search history."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM search_history 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    
    history = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return history

def clear_all_files():
    """Clear all files from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM files")
    
    conn.commit()
    conn.close()

def get_preference(key: str) -> Optional[str]:
    """Get a preference value."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT value FROM preferences WHERE key = ?", (key,))
    row = cursor.fetchone()
    
    conn.close()
    return row['value'] if row else None

def set_preference(key: str, value: str):
    """Set a preference value."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO preferences (key, value)
        VALUES (?, ?)
    """, (key, value))
    
    conn.commit()
    conn.close()

def get_file_by_faiss_index(faiss_idx: int) -> Optional[Dict]:
    """Get the file that contains a specific FAISS chunk index."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM files 
        WHERE faiss_start_idx <= ? AND faiss_end_idx >= ?
    """, (faiss_idx, faiss_idx))
    row = cursor.fetchone()
    
    conn.close()
    return dict(row) if row else None

def delete_search_history_item(history_id: int) -> bool:
    """Delete a single search history item."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM search_history WHERE id = ?", (history_id,))
    deleted = cursor.rowcount > 0
    
    conn.commit()
    conn.close()
    return deleted

def delete_all_search_history() -> int:
    """Delete all search history. Returns count of deleted items."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM search_history")
    count = cursor.fetchone()[0]
    
    cursor.execute("DELETE FROM search_history")
    
    conn.commit()
    conn.close()
    return count

def add_folder_to_history(path: str):
    """Add or update a folder in history."""
    conn = get_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now()
    
    # Check if exists
    cursor.execute("SELECT id FROM folder_history WHERE path = ?", (path,))
    row = cursor.fetchone()
    
    if row:
        cursor.execute("UPDATE folder_history SET last_used_at = ? WHERE id = ?", (timestamp, row['id']))
    else:
        cursor.execute("INSERT INTO folder_history (path, added_at, last_used_at) VALUES (?, ?, ?)", 
                      (path, timestamp, timestamp))
    
    conn.commit()
    conn.close()

def get_folder_history(limit: int = 20, indexed_only: bool = False) -> List[Dict]:
    """Get recent folder history."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if indexed_only:
        cursor.execute("SELECT * FROM folder_history WHERE is_indexed = 1 ORDER BY last_used_at DESC LIMIT ?", (limit,))
    else:
        cursor.execute("SELECT * FROM folder_history ORDER BY last_used_at DESC LIMIT ?", (limit,))
    
    history = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return history

def mark_folder_indexed(path: str):
    """Mark a folder as successfully indexed."""
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now()
    
    # Upsert with is_indexed = 1
    cursor.execute("SELECT id FROM folder_history WHERE path = ?", (path,))
    row = cursor.fetchone()
    
    if row:
        cursor.execute("UPDATE folder_history SET is_indexed = 1, last_used_at = ? WHERE id = ?", (timestamp, row['id']))
    else:
        cursor.execute("INSERT INTO folder_history (path, added_at, last_used_at, is_indexed) VALUES (?, ?, ?, 1)", 
                      (path, timestamp, timestamp))
    
    conn.commit()
    conn.close()

def delete_folder_history_item(path: str) -> bool:
    """Delete a single folder from history. Returns True if deleted."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM folder_history WHERE path = ?", (path,))
    deleted = cursor.rowcount > 0
    
    conn.commit()
    conn.close()
    return deleted

def clear_folder_history() -> int:
    """Clear all folder history. Returns count of deleted items."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM folder_history")
    count = cursor.fetchone()[0]
    
    cursor.execute("DELETE FROM folder_history")
    
    conn.commit()
    conn.close()
    return count

# Database initialization moved to api.py startup_event
# to prevent side effects on import (file locking)

# -----------------------------------------------------------------------------
# Response Cache Functions
# -----------------------------------------------------------------------------

def get_cached_response(query_hash: str, context_hash: str, model_id: str, response_type: str) -> Optional[str]:
    """
    Retrieve a cached response if it exists.
    Updates hit_count and last_accessed_at on hit.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT response_text, hit_count FROM response_cache 
            WHERE query_hash = ? AND context_hash = ? AND model_id = ? AND response_type = ?
        """, (query_hash, context_hash, model_id, response_type))
        
        result = cursor.fetchone()
        if result:
            response_text, hit_count = result
            # Update stats asynchronously-ish (fire and forget update)
            cursor.execute("""
                UPDATE response_cache 
                SET hit_count = hit_count + 1, last_accessed_at = CURRENT_TIMESTAMP 
                WHERE query_hash = ? AND context_hash = ? AND model_id = ? AND response_type = ?
            """, (query_hash, context_hash, model_id, response_type))
            conn.commit()
            return response_text
        return None
    except Exception as e:
        print(f"Cache lookup failed: {e}")
        return None
    finally:
        conn.close()

def cache_response(query_hash: str, context_hash: str, model_id: str, response_type: str, response_text: str):
    """
    Store a new response in the persistent cache.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO response_cache 
            (query_hash, context_hash, model_id, response_type, response_text, hit_count, last_accessed_at)
            VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
        """, (query_hash, context_hash, model_id, response_type, response_text))
        conn.commit()
    except Exception as e:
        print(f"Cache storage failed: {e}")
    finally:
        conn.close()

def clear_response_cache():
    """Clear all cached AI responses."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM response_cache")
        count = cursor.rowcount
        conn.commit()
        return count
    except Exception as e:
        print(f"Cache clear failed: {e}")
        return 0
    finally:
        conn.close()

def get_cache_stats():
    """Get statistics about the response cache."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*), SUM(hit_count) FROM response_cache")
        total_entries, total_hits = cursor.fetchone()
        return {
            "total_entries": total_entries or 0,
            "total_hits": total_hits or 0
        }
    except Exception as e:
        print(f"Cache stats failed: {e}")
        return {"total_entries": 0, "total_hits": 0}
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# Cluster Functions (RAPTOR)
# -----------------------------------------------------------------------------

def add_clusters_batch(clusters_data: List[Tuple[str, int]]):
    """Batch insert cluster summaries."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.executemany('''
            INSERT INTO clusters (summary, level)
            VALUES (?, ?)
        ''', clusters_data)
        conn.commit()
    except Exception as e:
        print(f"Error adding batch clusters to DB: {e}")
    finally:
        conn.close()

def add_cluster(summary: str, level: int) -> int:
    """Add a cluster summary to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO clusters (summary, level)
        VALUES (?, ?)
    """, (summary, level))
    
    cluster_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return cluster_id

def get_clusters_by_level(level: int) -> List[Dict]:
    """Get all clusters at a specific level."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM clusters WHERE level = ?", (level,))
    clusters = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return clusters

def clear_clusters():
    """Clear all clusters from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM clusters")
    
    conn.commit()
    conn.close()


def cleanup_test_data() -> Dict[str, int]:
    """
    Remove test data from the production database.
    Cleans up paths that look like test paths (containing /test/, temp directories, etc.)
    Returns counts of cleaned items.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    counts = {
        'files': 0,
        'folders': 0,
        'search_history': 0
    }
    
    # Patterns that indicate test data
    test_patterns = [
        '%/test/%',
        r'%\test\%',
        r'%\Temp\%',
        '%/tmp/%',
        '%/var/folders/%',  # macOS temp
        r'%AppData\Local\Temp%',  # Windows temp
    ]
    
    # Clean files table
    for pattern in test_patterns:
        cursor.execute("DELETE FROM files WHERE path LIKE ?", (pattern,))
        counts['files'] += cursor.rowcount
    
    # Clean folder_history table  
    for pattern in test_patterns:
        cursor.execute("DELETE FROM folder_history WHERE path LIKE ?", (pattern,))
        counts['folders'] += cursor.rowcount
    
    # Clean search_history with test-like queries
    cursor.execute("DELETE FROM search_history WHERE query LIKE '%test%' OR query LIKE 'delete%' OR query LIKE 'structure%'")
    counts['search_history'] = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    total = sum(counts.values())
    if total > 0:
        print(f"[CLEANUP] Removed {counts['files']} test files, {counts['folders']} test folders, {counts['search_history']} test searches")
    
    return counts

# N+1 Optimization for Search Metadata
MAX_INDICES = 100  # SQLite limit is actually 999 vars, keeping conservative for tests

def get_files_by_faiss_indices(indices: list[int]) -> dict[int, dict]:
    """
    Batch retrieve file metadata for multiple FAISS indices in a single query.
    Returns a dictionary mapping faiss_idx -> file_info dict.
    """
    if not indices:
        return {}

    # Deduplicate and validate
    unique_indices = sorted(list(set(indices)))

    if len(unique_indices) > MAX_INDICES:
        raise ValueError(f"Too many indices for single query (max {MAX_INDICES}). Provided: {len(unique_indices)}")

    conn = get_connection()
    try:
        # Build query: SELECT * FROM files WHERE (faiss_start_idx <= ? AND faiss_end_idx >= ?) OR ...
        query_parts = []
        params = []
        for idx in unique_indices:
            query_parts.append("(faiss_start_idx <= ? AND faiss_end_idx >= ?)")
            params.extend([idx, idx])

        sql = f"SELECT * FROM files WHERE {' OR '.join(query_parts)}"

        cursor = conn.execute(sql, params)
        files = [dict(row) for row in cursor.fetchall()]

        result = {}
        for idx in unique_indices:
            for file in files:
                if file['faiss_start_idx'] <= idx <= file['faiss_end_idx']:
                    result[idx] = file
                    break

        return result
    except Exception as e:
        print(f"Error getting files by faiss indices: {e}")
        return {}
    finally:
        conn.close()
