import sqlite3
import threading
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATABASE_PATH = os.path.join(DATA_DIR, 'metadata.db')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Thread-local storage for database connections
thread_local = threading.local()

class PooledConnection:
    """
    A wrapper to prevent explicit closing of thread-local sqlite3 connections.

    This ensures that the connection remains open for reuse by the same thread,
    while still providing a 'close' method that resets transaction state.
    """
    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize the PooledConnection wrapper.

        Args:
            connection (sqlite3.Connection): The raw sqlite3 connection object.
        """
        self._connection = connection

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the underlying connection.

        Args:
            name (str): Attribute name.

        Returns:
            Any: The attribute or method from the sqlite3 connection.
        """
        return getattr(self._connection, name)

    def close(self):
        # Don't actually close, just rollback any uncommitted transaction
        # to leave the connection in a clean state for the next user.
        self._connection.rollback()
        pass

def get_connection():
    """
    Get or create a thread-local database connection.

    Reuses connection if it exists for the current thread and matches the 
    current DATABASE_PATH. Enables WAL mode for improved concurrency on initialization.

    Returns:
        PooledConnection: A thread-safe connection wrapper.
    """
    if not hasattr(thread_local, "connection") or thread_local.db_path != DATABASE_PATH:
        # Create new connection
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        thread_local.connection = conn
        thread_local.db_path = DATABASE_PATH

        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")

    return PooledConnection(thread_local.connection)

def init_database():
    """
    Initialize the database schema if tables do not exist.

    Creates tables for:
    - files: Document metadata.
    - search_history: Past queries.
    - folder_history: Indexed and visited paths.
    - preferences: Key-value settings.
    - response_cache: Cached AI answers.
    - clusters: RAPTOR document clustering summaries.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Files table - stores metadata about indexed files
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT,
            size INTEGER,
            last_modified FLOAT,
            faiss_start_idx INTEGER,
            faiss_end_idx INTEGER,
            tags TEXT
        )
    ''')

    # Indices for faster lookup
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_faiss_start ON files(faiss_start_idx)')

    # Migration: Add missing columns if they don't exist in older databases
    cursor.execute("PRAGMA table_info(files)")
    existing_columns = [col[1] for col in cursor.fetchall()]
    if 'file_type' not in existing_columns:
        cursor.execute("ALTER TABLE files ADD COLUMN file_type TEXT")
    if 'tags' not in existing_columns:
        cursor.execute("ALTER TABLE files ADD COLUMN tags TEXT")
    
    # Search history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            result_count INTEGER,
            execution_time_ms INTEGER
        )
    ''')

    # Folder history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS folder_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_indexed BOOLEAN DEFAULT 0
        )
    ''')

    # User preferences table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS preferences (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    # Response Cache Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS response_cache (
            query_hash TEXT,
            context_hash TEXT,
            model_id TEXT,
            response_type TEXT,
            response_text TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            hit_count INTEGER DEFAULT 0,
            PRIMARY KEY (query_hash, context_hash, model_id, response_type)
        )
    ''')

    # Cluster Table (RAPTOR)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT,
            level INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

# -----------------------------------------------------------------------------
# File Operations
# -----------------------------------------------------------------------------

def add_file(path: str, filename: str, file_type: str, size: int, last_modified: float,
             faiss_start_idx: int, faiss_end_idx: int, tags: List[str] = None):
    """
    Insert or replace a single file's metadata in the database.

    Args:
        path (str): Absolute file path.
        filename (str): Name of the file.
        file_type (str): Extension/type identifier.
        size (int): File size in bytes.
        last_modified (float): Timestamp of last modification.
        faiss_start_idx (int): Beginning index in the vector database.
        faiss_end_idx (int): Ending index in the vector database.
        tags (List[str], optional): List of descriptive tags. Defaults to None.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    tags_json = json.dumps(tags) if tags else '[]'
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO files
            (path, filename, file_type, size, last_modified, faiss_start_idx, faiss_end_idx, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (path, filename, file_type, size, last_modified, faiss_start_idx, faiss_end_idx, tags_json))
        conn.commit()
    except Exception as e:
        print(f"Error adding file to DB: {e}")
    finally:
        conn.close()

def add_files_batch(files_data: List[Dict]):
    """
    Batch insert multiple files for better performance.

    Args:
        files_data (List[Dict]): A list of dictionaries containing file metadata 
            keys matching the 'files' table columns.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.executemany('''
            INSERT OR REPLACE INTO files
            (path, filename, file_type, size, last_modified, faiss_start_idx, faiss_end_idx, tags)
            VALUES (:path, :filename, :file_type, :size, :last_modified, :faiss_start_idx, :faiss_end_idx, :tags)
        ''', files_data)
        conn.commit()
    except Exception as e:
        print(f"Error adding batch files to DB: {e}")
    finally:
        conn.close()

def get_all_files() -> List[Dict]:
    """
    Retrieve all indexed files from the database.

    Returns:
        List[Dict]: A list of rows representing indexed files.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM files ORDER BY filename')
    files = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return files

def get_file_by_path(path: str) -> Optional[Dict]:
    """
    Retrieve metadata for a specific file by its path.

    Args:
        path (str): The absolute path of the file.

    Returns:
        Optional[Dict]: The file details if found, else None.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM files WHERE path = ?', (path,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_file_by_faiss_index(idx: int) -> Optional[Dict]:
    """
    Find the file associated with a specific embedding index.

    Args:
        idx (int): The index in the FAISS vector database.

    Returns:
        Optional[Dict]: The file containing that index, or None.
    """
    conn = get_connection()
    cursor = conn.cursor()
    # Find the file where the index falls within the start/end range
    cursor.execute('''
        SELECT * FROM files
        WHERE ? BETWEEN faiss_start_idx AND faiss_end_idx
    ''', (idx,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def clear_files():
    """
    Delete all entries from the files table.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM files')
    conn.commit()
    conn.close()

# -----------------------------------------------------------------------------
# Search History Operations
# -----------------------------------------------------------------------------

def add_search_history(query: str, result_count: int, execution_time_ms: int):
    """
    Log a search query and its performance metrics.

    Args:
        query (str): The search query text.
        result_count (int): Number of findings returned.
        execution_time_ms (int): Retrieval time in milliseconds.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO search_history (query, result_count, execution_time_ms)
            VALUES (?, ?, ?)
        ''', (query, result_count, execution_time_ms))
        conn.commit()
    except Exception as e:
        print(f"Error adding search history: {e}")
    finally:
        conn.close()

def get_search_history(limit: int = 50) -> List[Dict]:
    """
    Retrieve the most recent search history items.

    Args:
        limit (int): Maximum number of history items to return. Defaults to 50.

    Returns:
        List[Dict]: List of recent search events.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM search_history ORDER BY timestamp DESC LIMIT ?', (limit,))
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return history

def delete_search_history_item(history_id: int) -> bool:
    """
    Remove a specific item from search history.

    Args:
        history_id (int): The database ID of the history entry.

    Returns:
        bool: True if an item was deleted, False otherwise.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM search_history WHERE id = ?', (history_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted

def delete_all_search_history() -> int:
    """
    Clear the entire search history.

    Returns:
        int: The number of items deleted.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM search_history')
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count

# -----------------------------------------------------------------------------
# Folder History Operations
# -----------------------------------------------------------------------------

def add_folder_to_history(path: str):
    """
    Add a folder to the path history or update its last accessed time.

    Args:
        path (str): The directory path.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Update last_accessed_at if exists, otherwise insert
        cursor.execute('''
            INSERT INTO folder_history (path, last_accessed_at)
            VALUES (?, CURRENT_TIMESTAMP)
            ON CONFLICT(path) DO UPDATE SET last_accessed_at = CURRENT_TIMESTAMP
        ''', (path,))
        conn.commit()
    except Exception as e:
        print(f"Error adding folder history: {e}")
    finally:
        conn.close()

def mark_folder_indexed(path: str):
    """
    Update a folder's status to indicate it has been indexed.

    Args:
        path (str): The directory path.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE folder_history
            SET is_indexed = 1
            WHERE path = ?
        ''', (path,))
        conn.commit()
    except Exception as e:
        print(f"Error marking folder indexed: {e}")
    finally:
        conn.close()

def get_folder_history(indexed_only: bool = False) -> List[Dict]:
    """
    Retrieve recently added target folders.

    Args:
        indexed_only (bool): If True, only return folders marked as indexed. Defaults to False.

    Returns:
        List[Dict]: List of folder data dictionaries.
    """
    conn = get_connection()
    cursor = conn.cursor()
    if indexed_only:
        cursor.execute('SELECT * FROM folder_history WHERE is_indexed = 1 ORDER BY last_accessed_at DESC')
    else:
        cursor.execute('SELECT * FROM folder_history ORDER BY last_accessed_at DESC')
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return history

def delete_folder_history_item(path: str) -> bool:
    """
    Remove a folder path from the history.

    Args:
        path (str): The folder path to delete.

    Returns:
        bool: True if the folder was removed, False otherwise.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM folder_history WHERE path = ?', (path,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted

def clear_folder_history() -> int:
    """
    Delete all folder history entries.

    Returns:
        int: The number of rows deleted.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM folder_history')
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count

# -----------------------------------------------------------------------------
# User Preferences Operations
# -----------------------------------------------------------------------------

def get_preference(key: str, default: str = None) -> str:
    """
    Retrieve a persistent user preference.

    Args:
        key (str): Preference key.
        default (str, optional): Default value if key is not found. Defaults to None.

    Returns:
        str: The preference value.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM preferences WHERE key = ?', (key,))
    row = cursor.fetchone()
    conn.close()
    return row['value'] if row else default

def set_preference(key: str, value: str):
    """
    Save or update a persistent user preference.

    Args:
        key (str): Preference key.
        value (str): Preference value.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO preferences (key, value)
        VALUES (?, ?)
    ''', (key, value))
    conn.commit()
    conn.close()

# -----------------------------------------------------------------------------
# Response Cache Functions
# -----------------------------------------------------------------------------

def get_cached_response(query_hash: str, context_hash: str, model_id: str, response_type: str) -> Optional[str]:
    """
    Retrieve a cached AI response if available.

    Updates the hit count and last accessed timestamp if a hit occurs.

    Args:
        query_hash (str): Hash of the user's query.
        context_hash (str): Hash of the retrieval context.
        model_id (str): ID/Name of the model used.
        response_type (str): Type of response (e.g., 'summary', 'answer').

    Returns:
        Optional[str]: The cached text if found, else None.
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
    Persist an AI response to the cache for future reuse.

    Args:
        query_hash (str): Hash of the user's query.
        context_hash (str): Hash of the retrieval context.
        model_id (str): ID/Name of the model used.
        response_type (str): Type of response.
        response_text (str): The raw text to store.
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

def clear_response_cache() -> int:
    """
    Flush all entries from the AI response cache.

    Returns:
        int: Total number of cache entries cleared.
    """
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

def get_cache_stats() -> Dict[str, int]:
    """
    Calculate usage metrics for the AI response cache.

    Returns:
        Dict[str, int]: A dictionary with 'total_entries' and 'total_hits'.
    """
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
    """
    Batch insert multiple cluster summaries for RAPTOR indexing.

    Args:
        clusters_data (List[Tuple[str, int]]): List of (summary_text, level) tuples.
    """
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
    """
    Insert a single cluster summary into the database.

    Args:
        summary (str): The summary text of the cluster.
        level (int): The hierarchy level (0 being the most granular).

    Returns:
        int: The ID of the newly created cluster.
    """
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
    """
    Retrieve all cluster summaries at a specific hierarchy level.

    Args:
        level (int): The hierarchy level to filter by.

    Returns:
        List[Dict]: List of cluster rows.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM clusters WHERE level = ?", (level,))
    clusters = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return clusters

def clear_clusters():
    """
    Delete all RAPTOR cluster summaries from the database.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM clusters")
    
    conn.commit()
    conn.close()

def cleanup_test_data() -> Dict[str, int]:
    """
    Purge test-related entries from metadata tables.

    Cleans up file paths, folders, and query history that match known test 
    directories (like /tmp/, /test/, etc.).

    Returns:
        Dict[str, int]: Counts of items removed from 'files', 'folders', and 'search_history'.
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
MAX_INDICES = 900  # SQLite limit is usually 999 vars, keeping safe margin

def get_files_by_faiss_indices(indices: list[int]) -> dict[int, dict]:
    """
    Batch retrieve file metadata for multiple FAISS indices in a single query.

    Optimization to prevent N+1 query problems when fetching metadata for search results.

    Args:
        indices (list[int]): List of vector indices from FAISS.

    Returns:
        dict[int, dict]: Mapping of faiss_idx to file metadata dictionary.

    Raises:
        ValueError: If too many indices are provided for a single SQLite query.
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
