"""
Test Database Module

Tests for the database module including file tracking,
search history, and metadata storage.
"""

import unittest
import os
import tempfile
import shutil
from backend import database

# Initialize database for unittest execution

# Initialize database for unittest execution
_original_db_path = None
_temp_db_file = None

def setUpModule():
    """Initialize database schema in a temp file."""
    global _original_db_path, _temp_db_file

    # Save original path
    _original_db_path = database.DATABASE_PATH

    # Create temp file
    _temp_db_file = tempfile.NamedTemporaryFile(delete=False)
    _temp_db_file.close() # Close so sqlite can open it

    # Override path
    database.DATABASE_PATH = _temp_db_file.name

    # Init DB
    database.init_database()

def tearDownModule():
    """Cleanup temp database."""
    global _original_db_path, _temp_db_file

    # Restore path
    if _original_db_path:
        database.DATABASE_PATH = _original_db_path

    # Remove file
    if _temp_db_file and os.path.exists(_temp_db_file.name):
        try:
            os.unlink(_temp_db_file.name)
        except:
            pass


class TestDatabaseBase(unittest.TestCase):
    """Base class for database tests ensuring clean state."""
    
    def setUp(self):
        """Clear relevant tables before each test."""
        # Note: We are using the session-scoped DB from conftest.py
        # We just need to ensure it's clean for our tests.
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM files")
        cursor.execute("DELETE FROM search_history")
        cursor.execute("DELETE FROM folder_history")
        cursor.execute("DELETE FROM preferences")
        conn.commit()
        conn.close()

class TestDatabaseFolderOperations(TestDatabaseBase):
    """Tests for folder history operations."""
    
    def test_folder_history(self):
        """Test adding and retrieving folder history."""
        path = "/test/path/folder"
        
        # Add folder
        database.add_folder_to_history(path)
        
        # Retrieve history
        history = database.get_folder_history()
        
        self.assertIsInstance(history, list)
        if history:
            self.assertEqual(history[0]['path'], path)
            self.assertIn('added_at', history[0])
            self.assertIn('last_used_at', history[0])

    def test_clear_folder_history(self):
        """Test clearing folder history."""
        # Add some items
        database.add_folder_to_history("/path/1")
        database.add_folder_to_history("/path/2")
        
        count = database.clear_folder_history()
        self.assertTrue(count >= 2)
        
        # Verify empty
        hist = database.get_folder_history()
        self.assertEqual(len(hist), 0)

    def test_database_initialization(self):
        """Test that database is properly initialized."""
        # Should not raise
        database.init_database()
        
        # Check that get_connection exists
        self.assertTrue(
            hasattr(database, 'get_connection'),
            "Database should have get_connection function"
        )

    def test_add_file_metadata(self):
        """Test adding file metadata to database."""
        from datetime import datetime
        
        # Should not raise
        try:
            database.add_file(
                path='/test/path/document.pdf',
                filename='document.pdf',
                extension='.pdf',
                size_bytes=1024,
                modified_date=datetime.now(),
                chunk_count=1,
                faiss_start_idx=0,
                faiss_end_idx=0
            )
        except Exception as e:
            self.fail(f"Failed to add file: {e}")
    
    def test_get_file_by_faiss_index(self):
        """Test retrieving file by FAISS index."""
        from datetime import datetime
        
        # First add a file
        test_path = '/test/retrieval/test.txt'
        database.add_file(
            path=test_path,
            filename='test.txt',
            extension='.txt',
            size_bytes=512,
            modified_date=datetime.now(),
            chunk_count=1,
            faiss_start_idx=999,
            faiss_end_idx=999
        )
        
        # Try to retrieve it
        file_info = database.get_file_by_faiss_index(999)
        
        if file_info:
            self.assertEqual(file_info['path'], test_path)
            self.assertEqual(file_info['filename'], 'test.txt')
    
    def test_add_search_history(self):
        """Test adding search history entries."""
        # Should not raise - using correct parameter names
        try:
            database.add_search_history(
                query="test query",
                result_count=5,
                execution_time_ms=100
            )
        except Exception as e:
            self.fail(f"Failed to add search history: {e}")
    
    def test_get_search_history(self):
        """Test retrieving search history."""
        # Add a search entry
        database.add_search_history("history test query", 3, 50)
        
        # Get history
        history = database.get_search_history(limit=10)
        self.assertIsInstance(history, list)
    
    def test_get_all_files(self):
        """Test getting all indexed files."""
        files = database.get_all_files()
        self.assertIsInstance(files, list)
    
    def test_clear_files(self):
        """Test clearing all file entries."""
        # Should not raise
        try:
            database.clear_all_files()
        except Exception as e:
            self.fail(f"Failed to clear files: {e}")

    def test_delete_folder_history_item(self):
        """Test deleting a single folder from history."""
        path = "/test/path/delete"
        database.add_folder_to_history(path)

        # Verify added
        history = database.get_folder_history()
        found = any(item['path'] == path for item in history)
        self.assertTrue(found)

        # Delete
        result = database.delete_folder_history_item(path)
        self.assertTrue(result)

        # Verify deleted
        history = database.get_folder_history()
        found = any(item['path'] == path for item in history)
        self.assertFalse(found)

    def test_delete_nonexistent_folder_history_item(self):
        """Test deleting a folder that doesn't exist."""
        result = database.delete_folder_history_item("/nonexistent/path")
        self.assertFalse(result)


class TestDatabaseSearchHistory(TestDatabaseBase):
    """Tests specifically for search history functionality."""
    
    def test_history_structure(self):
        """Test that search history entries have correct structure."""
        # Add an entry
        database.add_search_history("structure test", 2, 30)
        
        history = database.get_search_history(limit=1)
        
        if history:
            entry = history[0]
            self.assertIn('query', entry)
            self.assertIn('result_count', entry)
            self.assertIn('execution_time_ms', entry)
    
    def test_delete_search_history(self):
        """Test deleting search history."""
        if hasattr(database, 'delete_all_search_history'):
            deleted_count = database.delete_all_search_history()
            self.assertIsInstance(deleted_count, int)
    
    def test_delete_single_history_item(self):
        """Test deleting a single search history item."""
        # Add a search entry
        database.add_search_history("delete single test", 1, 10)
        
        # Get the most recent entry
        history = database.get_search_history(limit=1)
        if history:
            history_id = history[0]['id']
            result = database.delete_search_history_item(history_id)
            self.assertTrue(result)
    
    def test_delete_nonexistent_history_item(self):
        """Test deleting a history item that doesn't exist."""
        # Try to delete a non-existent ID
        result = database.delete_search_history_item(999999)
        self.assertFalse(result)


class TestDatabaseFileOperations(TestDatabaseBase):
    """Tests for file CRUD operations."""
    
    def test_get_file_by_path_existing(self):
        """Test retrieving a file by path that exists."""
        from datetime import datetime
        
        test_path = '/test/get_by_path/existing.pdf'
        database.add_file(
            path=test_path,
            filename='existing.pdf',
            extension='.pdf',
            size_bytes=2048,
            modified_date=datetime.now(),
            chunk_count=2,
            faiss_start_idx=100,
            faiss_end_idx=101
        )
        
        file_info = database.get_file_by_path(test_path)
        
        self.assertIsNotNone(file_info)
        self.assertEqual(file_info['path'], test_path)
        self.assertEqual(file_info['filename'], 'existing.pdf')
        self.assertEqual(file_info['size_bytes'], 2048)
    
    def test_get_file_by_path_nonexistent(self):
        """Test retrieving a file by path that doesn't exist."""
        file_info = database.get_file_by_path('/nonexistent/path/file.txt')
        self.assertIsNone(file_info)
    
    def test_delete_file(self):
        """Test deleting a file from the database."""
        from datetime import datetime
        
        test_path = '/test/delete/todelete.pdf'
        file_id = database.add_file(
            path=test_path,
            filename='todelete.pdf',
            extension='.pdf',
            size_bytes=512,
            modified_date=datetime.now(),
            chunk_count=1,
            faiss_start_idx=200,
            faiss_end_idx=200
        )
        
        # Verify file exists
        file_info = database.get_file_by_path(test_path)
        self.assertIsNotNone(file_info)
        
        # Delete the file
        database.delete_file(file_info['id'])
        
        # Verify file is gone
        file_info_after = database.get_file_by_path(test_path)
        self.assertIsNone(file_info_after)
    
    def test_get_file_by_faiss_index_not_found(self):
        """Test retrieving file by FAISS index that doesn't exist."""
        file_info = database.get_file_by_faiss_index(888888)
        self.assertIsNone(file_info)

    def test_get_files_by_faiss_indices(self):
        """Test batch retrieving files by FAISS indices."""
        from backend import database
        from datetime import datetime

        # Add two files
        database.add_file(
            path='/test/batch/1.txt',
            filename='1.txt',
            extension='.txt',
            size_bytes=100,
            modified_date=datetime.now(),
            chunk_count=5,
            faiss_start_idx=1000,
            faiss_end_idx=1004
        )
        database.add_file(
            path='/test/batch/2.txt',
            filename='2.txt',
            extension='.txt',
            size_bytes=100,
            modified_date=datetime.now(),
            chunk_count=5,
            faiss_start_idx=1005,
            faiss_end_idx=1009
        )

        # Test batch retrieval
        indices = [1002, 1007, 9999]
        results = database.get_files_by_faiss_indices(indices)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[1002]['filename'], '1.txt')
        self.assertEqual(results[1007]['filename'], '2.txt')
        self.assertNotIn(9999, results)

    def test_get_files_by_faiss_indices_exceeds_max(self):
        """Test that ValueError is raised when exceeding MAX_INDICES limit."""
        from backend import database
        
        # Create a list of 101 indices (exceeds MAX_INDICES=100)
        indices = list(range(101))
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            database.get_files_by_faiss_indices(indices)
        
        self.assertIn("Too many indices", str(context.exception))



class TestDatabasePreferences(TestDatabaseBase):
    """Tests for preferences storage."""
    
    def test_set_and_get_preference(self):
        """Test setting and getting a preference."""
        database.set_preference('test_key', 'test_value')
        value = database.get_preference('test_key')
        
        self.assertEqual(value, 'test_value')
    
    def test_get_nonexistent_preference(self):
        """Test getting a preference that doesn't exist."""
        value = database.get_preference('nonexistent_key_12345')
        self.assertIsNone(value)
    
    def test_update_preference(self):
        """Test updating an existing preference."""
        database.set_preference('update_key', 'original_value')
        database.set_preference('update_key', 'updated_value')
        
        value = database.get_preference('update_key')
        self.assertEqual(value, 'updated_value')
    
    def test_preference_with_special_characters(self):
        """Test preferences with special characters."""
        special_value = "path/with/slashes and spaces & symbols!"
        database.set_preference('special_key', special_value)
        
        value = database.get_preference('special_key')
        self.assertEqual(value, special_value)


class TestGetFilesByFaissIndices(unittest.TestCase):
    """Tests for get_files_by_faiss_indices batch lookup function."""

    def test_empty_list_returns_empty_dict(self):
        """Empty input returns an empty dict without touching the DB."""
        from backend import database
        result = database.get_files_by_faiss_indices([])
        self.assertEqual(result, {})

    def test_exceeding_max_indices_raises_value_error(self):
        """More than MAX_INDICES unique indices raises ValueError."""
        from backend import database
        indices = list(range(database.MAX_INDICES + 1))
        with self.assertRaises(ValueError) as ctx:
            database.get_files_by_faiss_indices(indices)
        error_msg = str(ctx.exception)
        self.assertIn(str(database.MAX_INDICES), error_msg)
        self.assertIn(str(len(indices)), error_msg)

    def test_exactly_max_indices_does_not_raise(self):
        """Exactly MAX_INDICES unique indices does not raise."""
        from backend import database
        # Only unique_indices count matters; use all-same to get 1 unique
        indices = list(range(database.MAX_INDICES))
        # Should not raise (result may be empty if no matching rows)
        result = database.get_files_by_faiss_indices(indices)
        self.assertIsInstance(result, dict)

    def test_duplicates_do_not_inflate_unique_count(self):
        """Duplicate indices are deduplicated before checking the limit."""
        from backend import database
        # 101 items but all the same value → only 1 unique → no ValueError
        indices = [42] * (database.MAX_INDICES + 1)
        result = database.get_files_by_faiss_indices(indices)
        self.assertIsInstance(result, dict)


class TestDatabaseConnection(unittest.TestCase):
    """Tests for database connection handling."""
    
    def test_get_connection_returns_valid_connection(self):
        """Test that get_connection returns a valid SQLite connection."""
        conn = database.get_connection()
        self.assertIsNotNone(conn)
        
        # Test that we can execute a query
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        self.assertEqual(result[0], 1)
        
        conn.close()
    
    def test_multiple_connections(self):
        """Test that multiple connections work correctly."""
        conn1 = database.get_connection()
        conn2 = database.get_connection()
        
        self.assertIsNotNone(conn1)
        self.assertIsNotNone(conn2)
        
        conn1.close()
        conn2.close()


class TestDatabaseResponseCache(unittest.TestCase):
    """Test response cache functionality."""

    def setUp(self):
        """Clean cache before each test."""
        from backend import database
        database.init_database()
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM response_cache")
        conn.commit()
        conn.close()

    def test_cache_response(self):
        """Test caching a response."""
        from backend import database
        database.cache_response(
            query_hash="hash1",
            context_hash="ctx1",
            model_id="model1",
            response_type="summary",
            response_text="Test response"
        )

        # Verify cached
        cached = database.get_cached_response("hash1", "ctx1", "model1", "summary")
        self.assertEqual(cached, "Test response")

    def test_cache_hit_increments_count(self):
        """Test that cache hits increment the hit count."""
        from backend import database
        database.cache_response("hash2", "ctx2", "model2", "answer", "Answer text")

        # First hit
        database.get_cached_response("hash2", "ctx2", "model2", "answer")
        # Second hit
        database.get_cached_response("hash2", "ctx2", "model2", "answer")

        # Verify hit count increased
        stats = database.get_cache_stats()
        self.assertGreater(stats['total_hits'], 0)

    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        from backend import database
        cached = database.get_cached_response("nonexistent", "ctx", "model", "type")
        self.assertIsNone(cached)

    def test_clear_response_cache(self):
        """Test clearing the cache."""
        from backend import database
        # Add entries
        database.cache_response("h1", "c1", "m1", "t1", "text1")
        database.cache_response("h2", "c2", "m2", "t2", "text2")

        # Clear
        count = database.clear_response_cache()
        self.assertEqual(count, 2)

        # Verify empty
        stats = database.get_cache_stats()
        self.assertEqual(stats['total_entries'], 0)


class TestDatabaseClusters(unittest.TestCase):
    """Test cluster (RAPTOR) functionality."""

    def setUp(self):
        """Clean clusters before each test."""
        from backend import database
        database.init_database()
        database.clear_clusters()

    def test_add_cluster(self):
        """Test adding a single cluster."""
        from backend import database
        cluster_id = database.add_cluster("Test cluster summary", level=1)
        self.assertIsInstance(cluster_id, int)
        self.assertGreater(cluster_id, 0)

    def test_get_clusters_by_level(self):
        """Test retrieving clusters by level."""
        from backend import database
        database.add_cluster("Level 1 cluster", level=1)
        database.add_cluster("Another level 1", level=1)
        database.add_cluster("Level 2 cluster", level=2)

        level1_clusters = database.get_clusters_by_level(1)
        self.assertEqual(len(level1_clusters), 2)

        level2_clusters = database.get_clusters_by_level(2)
        self.assertEqual(len(level2_clusters), 1)

    def test_add_clusters_batch(self):
        """Test batch adding clusters."""
        from backend import database
        clusters_data = [
            ("Cluster 1", 1),
            ("Cluster 2", 1),
            ("Cluster 3", 2)
        ]

        database.add_clusters_batch(clusters_data)

        all_level1 = database.get_clusters_by_level(1)
        self.assertEqual(len(all_level1), 2)

    def test_clear_clusters(self):
        """Test clearing all clusters."""
        from backend import database
        database.add_cluster("Cluster to clear", level=1)
        database.clear_clusters()

        clusters = database.get_clusters_by_level(1)
        self.assertEqual(len(clusters), 0)


class TestDatabaseCleanup(unittest.TestCase):
    """Test cleanup functionality."""

    def test_cleanup_test_data(self):
        """Test cleaning up test data from production database."""
        from backend import database
        from datetime import datetime

        database.init_database()

        # Add some test-like paths
        database.add_file(
            path='/test/path/file.pdf',
            filename='file.pdf',
            extension='.pdf',
            size_bytes=1024,
            modified_date=datetime.now(),
            chunk_count=1,
            faiss_start_idx=0,
            faiss_end_idx=0
        )

        database.add_folder_to_history('/test/folder')
        database.add_search_history('test query', 0, 100)

        # Run cleanup
        counts = database.cleanup_test_data()

        # Verify cleanup happened
        self.assertIsInstance(counts, dict)
        self.assertIn('files', counts)
        self.assertIn('folders', counts)
        self.assertIn('search_history', counts)


class TestDatabaseConcurrency(unittest.TestCase):
    """Test thread safety of database connections."""

    def test_multiple_connections_different_threads(self):
        """Test that connections work across threads."""
        from backend import database
        import threading

        database.init_database()
        results = []

        def worker():
            conn = database.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            results.append(result[0])
            conn.close()

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should succeed
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r == 1 for r in results))


class TestDatabaseMarkFolderIndexed(unittest.TestCase):
    """Test marking folders as indexed."""

    def setUp(self):
        """Clean folder history before each test."""
        from backend import database
        database.init_database()
        database.clear_folder_history()

    def test_mark_folder_indexed_new_folder(self):
        """Test marking a new folder as indexed."""
        from backend import database

        database.mark_folder_indexed('/new/folder')

        history = database.get_folder_history(indexed_only=True)
        paths = [item['path'] for item in history]
        self.assertIn('/new/folder', paths)

    def test_mark_folder_indexed_existing_folder(self):
        """Test marking an existing folder as indexed."""
        from backend import database

        database.add_folder_to_history('/existing/folder')
        database.mark_folder_indexed('/existing/folder')

        history = database.get_folder_history(indexed_only=True)
        paths = [item['path'] for item in history]
        self.assertIn('/existing/folder', paths)


class TestDatabaseBatchOperations(unittest.TestCase):
    """Test batch database operations performance."""

    def setUp(self):
        """Set up test environment."""
        from backend import database
        database.init_database()
        database.clear_all_files()

    def test_add_files_batch_performance(self):
        """Test that batch insert is efficient."""
        from backend import database
        from datetime import datetime

        # Create 100 file records
        files_data = [
            {
                'path': f'/test/file{i}.txt',
                'filename': f'file{i}.txt',
                'extension': '.txt',
                'size_bytes': 1024,
                'modified_date': datetime.now(),
                'chunk_count': 1,
                'faiss_start_idx': i * 10,
                'faiss_end_idx': i * 10 + 9
            }
            for i in range(100)
        ]

        # Should complete without error
        database.add_files_batch(files_data)

        # Verify all added
        all_files = database.get_all_files()
        self.assertGreaterEqual(len(all_files), 100)


class TestDatabaseFileByName(unittest.TestCase):
    """Test file lookup by name functionality."""

    def setUp(self):
        """Set up test environment."""
        from backend import database
        database.init_database()
        database.clear_all_files()

    def test_get_file_by_name_exists(self):
        """Test getting file by name when it exists."""
        from backend import database
        from datetime import datetime

        database.add_file(
            path='/test/unique_file.pdf',
            filename='unique_file.pdf',
            extension='.pdf',
            size_bytes=2048,
            modified_date=datetime.now(),
            chunk_count=1,
            faiss_start_idx=0,
            faiss_end_idx=0
        )

        file_info = database.get_file_by_name('unique_file.pdf')
        self.assertIsNotNone(file_info)
        self.assertEqual(file_info['filename'], 'unique_file.pdf')

    def test_get_file_by_name_not_exists(self):
        """Test getting file by name when it doesn't exist."""
        from backend import database

        file_info = database.get_file_by_name('nonexistent_file.pdf')
        self.assertIsNone(file_info)


class TestDatabaseEdgeCases(unittest.TestCase):
    """Test database edge cases and boundary conditions."""

    def test_get_files_with_limit_and_offset(self):
        """Test pagination with limit and offset."""
        from backend import database
        from datetime import datetime

        database.init_database()
        database.clear_all_files()

        # Add 10 files
        for i in range(10):
            database.add_file(
                path=f'/test/file{i}.txt',
                filename=f'file{i}.txt',
                extension='.txt',
                size_bytes=100,
                modified_date=datetime.now(),
                chunk_count=1,
                faiss_start_idx=i,
                faiss_end_idx=i
            )

        # Test limit
        files = database.get_all_files(limit=5)
        self.assertEqual(len(files), 5)

        # Test offset
        files_offset = database.get_all_files(limit=5, offset=5)
        self.assertEqual(len(files_offset), 5)

    def test_add_file_with_special_characters_in_path(self):
        """Test adding file with special characters in path."""
        from backend import database
        from datetime import datetime

        database.init_database()

        special_path = "/test/folder (2024) [data]/file & name.pdf"
        database.add_file(
            path=special_path,
            filename='file & name.pdf',
            extension='.pdf',
            size_bytes=1024,
            modified_date=datetime.now(),
            chunk_count=1,
            faiss_start_idx=0,
            faiss_end_idx=0
        )

        file_info = database.get_file_by_path(special_path)
        self.assertIsNotNone(file_info)
        self.assertEqual(file_info['path'], special_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)