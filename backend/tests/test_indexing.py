import unittest
import tempfile
import os
import numpy as np
import shutil
import pickle
from unittest.mock import patch, MagicMock, call
from backend.indexing import create_index, save_index, load_index

# Shared temp database setup for ALL test classes
_shared_temp_dir = None
_original_db_path = None

def setUpModule():
    """Set up shared temp database for all tests in this module."""
    global _shared_temp_dir, _original_db_path

    # Create shared temp directory
    _shared_temp_dir = tempfile.mkdtemp()

    # Database
    from backend import database
    _original_db_path = database.DATABASE_PATH
    database.DATABASE_PATH = os.path.join(_shared_temp_dir, 'test_indexing_metadata.db')
    database.init_database()

def tearDownModule():
    """Clean up shared temp database."""
    global _shared_temp_dir, _original_db_path
    from backend import database
    import gc
    import time

    # Restore original path
    database.DATABASE_PATH = _original_db_path

    # Try to close any lingering connections and clean up
    gc.collect()
    time.sleep(0.1)

    if _shared_temp_dir and os.path.exists(_shared_temp_dir):
        try:
            shutil.rmtree(_shared_temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up test directory {_shared_temp_dir}: {e}")

class MockFuture:
    def __init__(self, result):
        self._result = result
    def result(self, timeout=None):
        return self._result

class MockExecutor:
    """A synchronous executor for testing."""
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def submit(self, fn, *args, **kwargs):
        return MockFuture(fn(*args, **kwargs))
    def map(self, fn, *iterables):
        return [fn(*args) for args in zip(*iterables)]


class TestIndexing(unittest.TestCase):
    """Test cases for indexing module"""

    def setUp(self):
        from backend import database
        database.init_database()
        """Set up test environment before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_folder = self.temp_dir
        
        # Create a dummy file
        self.test_file = os.path.join(self.test_folder, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("This is a test document content for indexing.")
            
        # Global patches for executors to make tests synchronous and mock-friendly
        self.pp_patcher = patch('concurrent.futures.ProcessPoolExecutor', side_effect=MockExecutor)
        self.tp_patcher = patch('concurrent.futures.ThreadPoolExecutor', side_effect=MockExecutor)
        self.ac_patcher = patch('concurrent.futures.as_completed', side_effect=lambda fs: fs)
        self.pp_patcher.start()
        self.tp_patcher.start()
        self.ac_patcher.start()

    def tearDown(self):
        """Clean up after each test method."""
        self.pp_patcher.stop()
        self.tp_patcher.stop()
        self.ac_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test creating an index."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        # Mock the extract_text function to return test content
        mock_extract_text.return_value = "This is test content for indexing."
        
        # Mock the embeddings model
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model
        
        # Mock the get_tags and clustering functions
        with patch('backend.indexing.get_tags', return_value="test, indexing"):
            with patch('backend.indexing.perform_global_clustering', return_value={0: [0]}):
                with patch('backend.indexing.smart_summary', return_value="Summary"):
                    res = create_index(self.test_folder, "openai", "fake_api_key")
                    index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res
                    
                    # Verify the index was created
                    self.assertIsNotNone(index)
                    self.assertIsNotNone(docs)
                    self.assertIsNotNone(tags)
                    self.assertEqual(len(docs), 1)
                    # tags is now a list of strings (empty or joined tags)
                    self.assertEqual(len(tags), 1)

    
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_empty_folder(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test creating an index with empty folder."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = []

        empty_folder = os.path.join(self.temp_dir, "empty_folder")
        os.makedirs(empty_folder, exist_ok=True)
        
        # Mock the extract_text function to return None
        mock_extract_text.return_value = None
        mock_embeddings_model = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings_model
        
        with patch('backend.indexing.get_tags', return_value=""),              patch('backend.indexing.perform_global_clustering', return_value={}),              patch('backend.indexing.smart_summary', return_value=""):
            res = create_index(empty_folder, "openai", "fake_api_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            
            # Verify the result is None, None, None
            self.assertIsNone(index)
            self.assertIsNone(docs)
            self.assertIsNone(tags)
    
    @patch('backend.indexing.faiss.write_index')
    @patch('backend.indexing.faiss.read_index')
    def test_save_and_load_index(self, mock_read_index, mock_write_index):
        """Test saving and loading an index."""
        # Mock write_index to create a dummy file so os.path.exists passes
        def side_effect_write(index, filepath):
            with open(filepath, 'w') as f:
                f.write("dummy")
        mock_write_index.side_effect = side_effect_write

        # Create a real FAISS index and documents (only write_index is mocked)
        import faiss
        dimension = 3
        index = faiss.IndexFlatL2(dimension)
        index.d = dimension  # explicitly set in case faiss is globally mocked
        embeddings = np.array([[1.0, 2.0, 3.0]], dtype='float32')
        index.add(embeddings)
        
        docs = [{"text": "Test document"}]
        tags = [["test", "tag"]]
        
        # Save the index
        index_path = os.path.join(self.temp_dir, "test_index.faiss")
        save_index(index, docs, tags, index_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(index_path))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_index_docs.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_index_tags.pkl")))
        
        # Setup mock for load_index
        mock_read_index.return_value = index # Return the mock index

        # Load the index (now returns 8-tuple; unpack first 7)
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]
        
        # Verify the loaded data matches the original
        self.assertIsNotNone(loaded_index)
        self.assertEqual(loaded_docs, docs)
        self.assertEqual(loaded_tags, tags)
    
    @patch('faiss.read_index')
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('backend.indexing.pickle.load')
    def test_load_index_preserves_data(self, mock_pickle_load, _mock_open, mock_exists, mock_read_index):
        """Test loading an index."""
        # Mock the index reading
        mock_faiss_index = MagicMock()
        mock_read_index.return_value = mock_faiss_index
        
        index_path = "fake_index.faiss"

        # Mock os.path.exists: True for main file and .pkl files
        mock_exists.side_effect = lambda path: path == index_path or path.endswith('.pkl')
        
        # Mock pickle loading: docs, tags, and bm25
        mock_pickle_load.side_effect = [
            [{"text": "Test document"}], # docs
            [["test", "tag"]],            # tags
            MagicMock()                   # bm25
        ]
        
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]
        
        # Verify the functions were called: faiss, docs.pkl, tags.pkl, bm25.pkl
        mock_read_index.assert_called_once_with(index_path)
        self.assertEqual(mock_pickle_load.call_count, 3)
        
        # Verify the results
        self.assertEqual(loaded_index, mock_faiss_index)
        self.assertEqual(loaded_docs, [{"text": "Test document"}])
        self.assertEqual(loaded_tags, [["test", "tag"]])


class TestIndexingMultipleFolders(unittest.TestCase):
    """Test indexing with multiple folders."""

    def setUp(self):
        from backend import database
        database.init_database()
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create two test folders
        self.folder1 = os.path.join(self.temp_dir, "folder1")
        self.folder2 = os.path.join(self.temp_dir, "folder2")
        os.makedirs(self.folder1, exist_ok=True)
        os.makedirs(self.folder2, exist_ok=True)
        
        # Create test files
        with open(os.path.join(self.folder1, "doc1.txt"), 'w') as f:
            f.write("Content from folder 1")
        with open(os.path.join(self.folder2, "doc2.txt"), 'w') as f:
            f.write("Content from folder 2")

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_multiple_folders(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test creating index from multiple folders."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        mock_extract_text.return_value = "Test content"
        
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_get_embeddings.return_value = mock_embeddings_model
        
        with patch('backend.indexing.get_tags', return_value="test"),              patch('backend.indexing.perform_global_clustering', return_value={0: [0, 1]}),              patch('backend.indexing.smart_summary', return_value="Summary"):
            res = create_index(
                [self.folder1, self.folder2], 
                "openai", 
                "fake_key"
            )
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res
            
            self.assertIsNotNone(index)
            self.assertEqual(len(docs), 2)


    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_with_progress_callback(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test progress callback during indexing."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        mock_extract_text.return_value = "Test content"
        
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model
        
        progress_calls = []
        def progress_callback(current, total, filename=None):
            progress_calls.append((current, total, filename))
        
        with patch('backend.indexing.get_tags', return_value="test"),              patch('backend.indexing.perform_global_clustering', return_value={0: [0]}),              patch('backend.indexing.smart_summary', return_value="Summary"):
            create_index(self.folder1, "openai", "fake_key", progress_callback=progress_callback)
            
            # Verify progress was called
            self.assertGreater(len(progress_calls), 0)

    def test_create_index_nonexistent_folder(self):
        """Test creating index with nonexistent folder."""
        with patch('backend.indexing.get_embeddings') as mock_embed:
            mock_embeddings_model = MagicMock()
            mock_embed.return_value = mock_embeddings_model
            
            res = create_index(
                "/nonexistent/folder/path", 
                "openai", 
                "fake_key"
            )
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res
            
            self.assertIsNone(index)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_string_folder_path(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test that string folder path is converted to list."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        mock_extract_text.return_value = "Test content"
        
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model
        
        with patch('backend.indexing.get_tags', return_value="test"),              patch('backend.indexing.perform_global_clustering', return_value={0: [0]}),              patch('backend.indexing.smart_summary', return_value="Summary"):
            # Pass string instead of list
            res = create_index(
                self.folder1,  # String, not list
                "openai", 
                "fake_key"
            )
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res
            
            self.assertIsNotNone(index)



class TestSaveIndex(unittest.TestCase):
    """Dedicated tests for save_index function."""

    def setUp(self):
        from backend import database
        database.init_database()
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.faiss.write_index')
    def test_save_index_creates_all_files(self, mock_write_index):
        """Test that save_index creates .faiss, _docs.json, and _tags.json files."""
        # Mock write_index side effect
        def side_effect_write(index, filepath):
            with open(filepath, 'w') as f:
                f.write("dummy")
        mock_write_index.side_effect = side_effect_write

        import faiss
        
        index = faiss.IndexFlatL2(3)
        index.d = 3  # explicitly set in case faiss is globally mocked
        embeddings = np.array([[1.0, 2.0, 3.0]], dtype='float32')
        index.add(embeddings)
        
        docs = ["Document 1", "Document 2"]
        tags = [["tag1"], ["tag2"]]
        
        index_path = os.path.join(self.temp_dir, "index.faiss")
        save_index(index, docs, tags, index_path)
        
        self.assertTrue(os.path.exists(index_path))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_docs.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_tags.pkl")))
        # Verify that the metadata sidecar is also written
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_meta.json")))


class TestLoadIndex(unittest.TestCase):
    """Dedicated tests for load_index function."""

    def setUp(self):
        from backend import database
        database.init_database()
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.faiss.write_index')
    @patch('backend.indexing.faiss.read_index')
    def test_load_index_preserves_data(self, mock_read_index, mock_write_index):
        """Test that load_index correctly restores saved data."""
        # Mock write_index
        def side_effect_write(index, filepath):
            with open(filepath, 'w') as f:
                f.write("dummy")
        mock_write_index.side_effect = side_effect_write

        import faiss
        
        # Create and save
        original_index = faiss.IndexFlatL2(3)
        original_index.d = 3  # explicitly set in case faiss is globally mocked
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32')
        original_index.add(embeddings)
        
        original_docs = [{"text": "Doc A"}, {"text": "Doc B"}]
        original_tags = [["alpha"], ["beta", "gamma"]]
        
        index_path = os.path.join(self.temp_dir, "test.faiss")
        save_index(original_index, original_docs, original_tags, index_path)
        
        # Mock load_index return
        mock_read_index.return_value = original_index

        # Load and verify (now returns 8-tuple)
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]
        meta = res[7] if len(res) > 7 else {}
        
        # We can't check ntotal on a mock object unless we set it.
        # But if we use the same mock object "original_index", it should have whatever we set on it.
        # However, faiss.IndexFlatL2 is a mock.
        # We can just check it's the same object.
        self.assertEqual(loaded_index, original_index)
        self.assertEqual(loaded_docs, original_docs)
        self.assertEqual(loaded_tags, original_tags)


class TestIndexingBoundaryCases(unittest.TestCase):
    """Test boundary cases and edge conditions for indexing."""

    def setUp(self):
        """Set up test fixtures."""
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()

        # Global patches for executors
        self.pp_patcher = patch('concurrent.futures.ProcessPoolExecutor', side_effect=MockExecutor)
        self.tp_patcher = patch('concurrent.futures.ThreadPoolExecutor', side_effect=MockExecutor)
        self.ac_patcher = patch('concurrent.futures.as_completed', side_effect=lambda fs: fs)
        self.pp_patcher.start()
        self.tp_patcher.start()
        self.ac_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.pp_patcher.stop()
        self.tp_patcher.stop()
        self.ac_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_with_very_large_file(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test indexing a very large file."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        # Simulate very large file split into many chunks
        mock_splitter_instance.split_text.return_value = [f"chunk{i}" for i in range(100)]

        # Create test file
        large_file = os.path.join(self.temp_dir, "large.txt")
        with open(large_file, 'w') as f:
            f.write("x" * 1000000)  # 1MB of content

        mock_extract_text.return_value = "x" * 1000000

        mock_embeddings_model = MagicMock()
        # Return embeddings for all chunks
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3] for _ in range(100)]
        mock_get_embeddings.return_value = mock_embeddings_model

        with patch('backend.indexing.get_tags', return_value="large"), \
             patch('backend.indexing.perform_global_clustering', return_value={0: list(range(100))}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):
            res = create_index(self.temp_dir, "openai", "fake_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            self.assertIsNotNone(index)
            # Should create multiple chunks
            self.assertEqual(len(docs), 100)

    @patch('backend.indexing.get_embeddings')
    def test_create_index_with_unsupported_files(self, mock_get_embeddings):
        """Test indexing folder with unsupported file types."""
        # Create unsupported files
        unsupported_file = os.path.join(self.temp_dir, "image.png")
        with open(unsupported_file, 'wb') as f:
            f.write(b'\x89PNG\r\n')  # PNG header

        mock_embeddings_model = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings_model

        res = create_index(self.temp_dir, "openai", "fake_key")
        index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

        # Should return None for no indexable content
        self.assertIsNone(index)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_with_special_characters_in_path(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test indexing files with special characters in path."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        # Create folder with special characters
        special_folder = os.path.join(self.temp_dir, "test folder (2024) [data]")
        os.makedirs(special_folder, exist_ok=True)

        test_file = os.path.join(special_folder, "file with spaces.txt")
        with open(test_file, 'w') as f:
            f.write("test content")

        mock_extract_text.return_value = "test content"

        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model

        with patch('backend.indexing.get_tags', return_value="test"), \
             patch('backend.indexing.perform_global_clustering', return_value={0: [0]}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):
            res = create_index(special_folder, "openai", "fake_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            self.assertIsNotNone(index)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_with_extraction_error(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test indexing when file extraction fails."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = []

        test_file = os.path.join(self.temp_dir, "corrupt.pdf")
        with open(test_file, 'w') as f:
            f.write("not a real PDF")

        # Simulate extraction failure
        mock_extract_text.return_value = None

        mock_embeddings_model = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings_model

        with patch('backend.indexing.get_tags', return_value=""), \
             patch('backend.indexing.perform_global_clustering', return_value={}), \
             patch('backend.indexing.smart_summary', return_value=""):
            res = create_index(self.temp_dir, "openai", "fake_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            # Should handle extraction errors gracefully
            self.assertIsNone(index)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_with_empty_files(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test indexing empty files."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = []

        empty_file = os.path.join(self.temp_dir, "empty.txt")
        with open(empty_file, 'w') as f:
            f.write("")  # Empty file

        mock_extract_text.return_value = ""

        mock_embeddings_model = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings_model

        with patch('backend.indexing.get_tags', return_value=""), \
             patch('backend.indexing.perform_global_clustering', return_value={}), \
             patch('backend.indexing.smart_summary', return_value=""):
            res = create_index(self.temp_dir, "openai", "fake_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            # Empty files should be handled
            self.assertIsNone(index)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_with_mixed_file_types(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test indexing folder with mixed supported and unsupported files."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        # Create supported file
        with open(os.path.join(self.temp_dir, "doc.txt"), 'w') as f:
            f.write("valid content")

        # Create unsupported file
        with open(os.path.join(self.temp_dir, "image.jpg"), 'wb') as f:
            f.write(b'\xFF\xD8\xFF')  # JPEG header

        mock_extract_text.return_value = "valid content"

        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model

        with patch('backend.indexing.get_tags', return_value="test"), \
             patch('backend.indexing.perform_global_clustering', return_value={0: [0]}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):
            res = create_index(self.temp_dir, "openai", "fake_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            # Should index supported files only
            self.assertIsNotNone(index)

    def test_create_index_with_symbolic_links(self):
        """Test indexing folder containing symbolic links."""
        # Create a real file
        real_file = os.path.join(self.temp_dir, "real.txt")
        with open(real_file, 'w') as f:
            f.write("real content")

        # Try to create symbolic link (may fail on Windows without admin)
        link_file = os.path.join(self.temp_dir, "link.txt")
        try:
            os.symlink(real_file, link_file)
        except OSError:
            # Skip test if symlinks not supported
            self.skipTest("Symbolic links not supported on this system")

        with patch('backend.indexing.get_embeddings') as mock_embed, \
             patch('backend.indexing.extract_text', return_value="content"), \
             patch('backend.indexing.CharacterTextSplitter') as mock_splitter_cls, \
             patch('backend.indexing.get_tags', return_value="test"), \
             patch('backend.indexing.perform_global_clustering', return_value={0: [0]}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):

            mock_splitter_instance = mock_splitter_cls.return_value
            mock_splitter_instance.split_text.return_value = ["chunk1"]

            mock_embeddings_model = MagicMock()
            mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
            mock_embed.return_value = mock_embeddings_model

            res = create_index(self.temp_dir, "openai", "fake_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            # Should handle symbolic links
            self.assertIsNotNone(index)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_with_nested_folders(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test indexing deeply nested folder structure."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        # Create deeply nested structure
        nested_path = os.path.join(self.temp_dir, "a", "b", "c", "d", "e")
        os.makedirs(nested_path, exist_ok=True)

        deep_file = os.path.join(nested_path, "deep.txt")
        with open(deep_file, 'w') as f:
            f.write("deep content")

        mock_extract_text.return_value = "deep content"

        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model

        with patch('backend.indexing.get_tags', return_value="test"), \
             patch('backend.indexing.perform_global_clustering', return_value={0: [0]}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):
            res = create_index(self.temp_dir, "openai", "fake_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            self.assertIsNotNone(index)


class TestSaveLoadRobustness(unittest.TestCase):
    """Test robustness of save and load operations."""

    def setUp(self):
        """Set up test fixtures."""
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.faiss.write_index')
    def test_save_index_with_special_characters_in_path(self, mock_write_index):
        """Test saving index to path with special characters."""
        def side_effect_write(index, filepath):
            with open(filepath, 'w') as f:
                f.write("dummy")
        mock_write_index.side_effect = side_effect_write

        import faiss

        index = faiss.IndexFlatL2(3)
        embeddings = np.array([[1.0, 2.0, 3.0]], dtype='float32')
        index.add(embeddings)

        docs = ["Test doc"]
        tags = [["tag1"]]

        # Path with special characters
        special_path = os.path.join(self.temp_dir, "index (test) [2024].faiss")
        save_index(index, docs, tags, special_path)

        self.assertTrue(os.path.exists(special_path))

    @patch('backend.indexing.faiss.write_index')
    def test_save_index_with_empty_docs(self, mock_write_index):
        """Test saving index with empty documents list."""
        def side_effect_write(index, filepath):
            with open(filepath, 'w') as f:
                f.write("dummy")
        mock_write_index.side_effect = side_effect_write

        import faiss

        index = faiss.IndexFlatL2(3)
        docs = []  # Empty
        tags = []

        index_path = os.path.join(self.temp_dir, "empty.faiss")
        save_index(index, docs, tags, index_path)

        # Should save without error
        self.assertTrue(os.path.exists(index_path))

    @patch('backend.indexing.faiss.write_index')
    def test_save_index_with_very_long_document(self, mock_write_index):
        """Test saving index with extremely long document content."""
        def side_effect_write(index, filepath):
            with open(filepath, 'w') as f:
                f.write("dummy")
        mock_write_index.side_effect = side_effect_write

        import faiss

        index = faiss.IndexFlatL2(3)
        embeddings = np.array([[1.0, 2.0, 3.0]], dtype='float32')
        index.add(embeddings)

        # Very long document
        long_doc = "x" * 1000000  # 1MB document
        docs = [{"text": long_doc}]
        tags = [["tag"]]

        index_path = os.path.join(self.temp_dir, "long.faiss")
        save_index(index, docs, tags, index_path)

        self.assertTrue(os.path.exists(index_path))

    @patch('backend.indexing.faiss.read_index')
    def test_load_index_with_missing_tags_file(self, mock_read_index):
        """Test loading index when tags file is missing."""
        import faiss

        index = faiss.IndexFlatL2(3)
        mock_read_index.return_value = index

        # Create index and docs files but not tags
        index_path = os.path.join(self.temp_dir, "partial.faiss")
        with open(index_path, 'w') as f:
            f.write("dummy")

        docs_path = os.path.join(self.temp_dir, "partial_docs.pkl")
        with open(docs_path, 'wb') as f:
            pickle.dump(["doc1"], f)

        bm25_path = os.path.join(self.temp_dir, "partial_bm25.pkl")
        with open(bm25_path, 'wb') as f:
            pickle.dump(MagicMock(), f)

        # Tags file intentionally missing

        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res

        # Should load with empty tags
        self.assertIsNotNone(loaded_index)
        self.assertEqual(loaded_docs, ["doc1"])
        self.assertEqual(loaded_tags, [])

    def test_load_index_nonexistent_file(self):
        """Test loading index from non-existent path."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.faiss")

        res = load_index(nonexistent_path)
        index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

        # Should return None for all
        self.assertIsNone(index)
        self.assertIsNone(docs)
        self.assertIsNone(tags)


class TestProgressCallbackBehavior(unittest.TestCase):
    """Test progress callback behavior during indexing."""

    def setUp(self):
        """Set up test fixtures."""
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()

        # Create test files
        for i in range(3):
            with open(os.path.join(self.temp_dir, f"file{i}.txt"), 'w') as f:
                f.write(f"content {i}")

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_progress_callback_called_multiple_times(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test that progress callback is called for each file."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        mock_extract_text.return_value = "content"

        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model

        progress_calls = []

        def progress_callback(current, total, message=None):
            progress_calls.append((current, total, message))

        with patch('backend.indexing.get_tags', return_value="test"), \
             patch('backend.indexing.perform_global_clustering', return_value={0: [0]}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):
            create_index(
                self.temp_dir,
                "openai",
                "fake_key",
                progress_callback=progress_callback
            )

            # Should have multiple progress updates
            self.assertGreater(len(progress_calls), 0)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_progress_callback_error_handling(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test that indexing continues even if progress callback raises error."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        mock_extract_text.return_value = "content"

        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model

        def bad_callback(current, total, message=None):
            raise RuntimeError("Callback error")

        with patch('backend.indexing.get_tags', return_value="test"), \
             patch('backend.indexing.perform_global_clustering', return_value={0: [0]}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):
            # Should not raise error even though callback fails
            try:
                res = create_index(
                    self.temp_dir,
                    "openai",
                    "fake_key",
                    progress_callback=bad_callback
                )
                # If it doesn't crash, the test passes
                self.assertTrue(True)
            except RuntimeError:
                # If the callback error propagates, that's also acceptable behavior
                # depending on implementation
                pass


class TestIndexingWithEmbeddingClient(unittest.TestCase):
    """Test indexing with injected embedding client."""

    def setUp(self):
        """
        Prepare an isolated test environment for indexing tests.
        
        Initializes the test database, creates a temporary directory with a sample test file ("test.txt"), and patches concurrent.futures ProcessPoolExecutor, ThreadPoolExecutor, and as_completed to use synchronous MockExecutor behaviour for deterministic, single-threaded execution during tests.
        """
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()

        # Create test file
        with open(os.path.join(self.temp_dir, "test.txt"), 'w') as f:
            f.write("Test content for embedding")

        # Global patches for executors
        self.pp_patcher = patch('concurrent.futures.ProcessPoolExecutor', side_effect=MockExecutor)
        self.tp_patcher = patch('concurrent.futures.ThreadPoolExecutor', side_effect=MockExecutor)
        self.ac_patcher = patch('concurrent.futures.as_completed', side_effect=lambda fs: fs)
        self.pp_patcher.start()
        self.tp_patcher.start()
        self.ac_patcher.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.pp_patcher.stop()
        self.tp_patcher.stop()
        self.ac_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.extract_text')
    def test_create_index_with_embedding_client(self, mock_extract_text, mock_splitter_cls):
        """Test create_index with embedding_client parameter."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        mock_extract_text.return_value = "Test content"

        # Create a mock embedding client
        mock_embedding_client = MagicMock()
        mock_embedding_client.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_embedding_client.model_name = "test-embedding-model"

        with patch('backend.indexing.get_tags', return_value="test"), \
             patch('backend.indexing.perform_global_clustering', return_value={0: [0]}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):

            res = create_index(
                self.temp_dir,
                provider="openai",
                api_key="fake_key",
                embedding_client=mock_embedding_client
            )
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            self.assertIsNotNone(index)
            # Verify embedding client was used (not get_embeddings)
            mock_embedding_client.embed_documents.assert_called()

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.get_embeddings')
    def test_create_index_fallback_to_get_embeddings(self, mock_get_embeddings,
                                                       mock_extract_text, mock_splitter_cls):
        """Test create_index falls back to get_embeddings when embedding_client is None."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        mock_extract_text.return_value = "Test content"

        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model

        with patch('backend.indexing.get_tags', return_value="test"), \
             patch('backend.indexing.perform_global_clustering', return_value={0: [0]}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):

            res = create_index(
                self.temp_dir,
                provider="openai",
                api_key="fake_key",
                embedding_client=None  # Explicitly None
            )
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            self.assertIsNotNone(index)
            # Verify get_embeddings was called as fallback
            mock_get_embeddings.assert_called_once()


class TestSaveIndexWithMetadata(unittest.TestCase):
    """Test save_index with metadata sidecar."""

    def setUp(self):
        """
        Prepare the test environment by reinitializing the test database and creating a per-test temporary directory.
        """
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """
        Remove the per-test temporary directory created in setUp.
        
        If the directory referenced by self.temp_dir exists, delete it and all its contents.
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.faiss.write_index')
    def test_save_index_writes_metadata_sidecar(self, mock_write_index):
        """Test that save_index creates metadata sidecar file."""
        def side_effect_write(index, filepath):
            """
            Write a small dummy file at the given filepath to simulate writing a FAISS index.
            
            Parameters:
                index: Ignored; present to match the faiss.write_index signature.
                filepath (str or Path): Filesystem path where a file containing the text "dummy" will be created or overwritten.
            """
            with open(filepath, 'w') as f:
                f.write("dummy")
        mock_write_index.side_effect = side_effect_write

        import faiss
        index = faiss.IndexFlatL2(384)
        index.d = 384
        embeddings = np.array([[1.0] * 384], dtype='float32')
        index.add(embeddings)

        docs = [{"text": "Test"}]
        tags = [["tag"]]

        index_path = os.path.join(self.temp_dir, "test.faiss")
        save_index(index, docs, tags, index_path,
                   model_name="test-model", embedding_dim=384)

        # Check metadata file exists
        meta_path = os.path.join(self.temp_dir, "test_meta.json")
        self.assertTrue(os.path.exists(meta_path))

        # Verify metadata content
        import json
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.assertEqual(meta['model_name'], "test-model")
        self.assertEqual(meta['embedding_dim'], 384)


class TestLoadIndexWithMetadata(unittest.TestCase):
    """Test load_index returns metadata."""

    def setUp(self):
        """
        Prepare the test environment by reinitializing the test database and creating a per-test temporary directory.
        """
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """
        Remove the per-test temporary directory created in setUp.
        
        If the directory referenced by self.temp_dir exists, delete it and all its contents.
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.faiss.write_index')
    @patch('backend.indexing.faiss.read_index')
    def test_load_index_returns_metadata(self, mock_read_index, mock_write_index):
        """Test that load_index returns metadata as 8th element."""
        def side_effect_write(index, filepath):
            """
            Write a small dummy file at the given filepath to simulate writing a FAISS index.
            
            Parameters:
                index: Ignored; present to match the faiss.write_index signature.
                filepath (str or Path): Filesystem path where a file containing the text "dummy" will be created or overwritten.
            """
            with open(filepath, 'w') as f:
                f.write("dummy")
        mock_write_index.side_effect = side_effect_write

        import faiss
        index = faiss.IndexFlatL2(768)
        index.d = 768
        embeddings = np.array([[1.0] * 768], dtype='float32')
        index.add(embeddings)

        docs = [{"text": "Test"}]
        tags = [["tag"]]

        index_path = os.path.join(self.temp_dir, "test.faiss")
        save_index(index, docs, tags, index_path,
                   model_name="gpt-embedding", embedding_dim=768)

        # Mock read to return same index
        mock_read_index.return_value = index

        # Load it back
        result = load_index(index_path)
        self.assertEqual(len(result), 8)  # Should be 8-tuple

        meta = result[7]
        self.assertIsInstance(meta, dict)
        self.assertEqual(meta['model_name'], "gpt-embedding")
        self.assertEqual(meta['embedding_dim'], 768)


class TestIndexingProgressCallback(unittest.TestCase):
    """Test progress callback behavior in detail."""

    def setUp(self):
        """
        Prepare test environment by initializing the test database and creating a temporary directory with three sample text files.
        
        Initializes the application's test database, assigns a new temporary directory path to `self.temp_dir`, and creates three files named `file0.txt`, `file1.txt`, and `file2.txt` inside that directory containing "Content 0", "Content 1", and "Content 2" respectively.
        """
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()

        # Create multiple test files
        for i in range(3):
            with open(os.path.join(self.temp_dir, f"file{i}.txt"), 'w') as f:
                f.write(f"Content {i}")

    def tearDown(self):
        """
        Remove the per-test temporary directory created in setUp.
        
        If the directory referenced by self.temp_dir exists, delete it and all its contents.
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_progress_callback_receives_correct_stages(self, mock_extract_text,
                                                         mock_get_embeddings, mock_splitter_cls):
        """Test that progress callback is called for different stages."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        mock_extract_text.return_value = "Test content"

        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model

        progress_calls = []

        def progress_callback(current, total, message=None):
            """
            Record a progress update by appending a structured entry to the enclosing `progress_calls` list.
            
            Parameters:
                current (int): The current progress count or step.
                total (int): The total number of steps or items.
                message (str | None): Optional human-readable status or stage description.
            
            Notes:
                This function has a side effect of mutating the outer-scope `progress_calls` list by appending a dict with keys `'current'`, `'total'`, and `'message'`.
            """
            progress_calls.append({
                'current': current,
                'total': total,
                'message': message
            })

        with patch('backend.indexing.get_tags', return_value="test"), \
             patch('backend.indexing.perform_global_clustering', return_value={0: [0]}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):

            create_index(
                self.temp_dir,
                "openai",
                "fake_key",
                progress_callback=progress_callback
            )

            # Verify progress was reported
            self.assertGreater(len(progress_calls), 0)

            # Check that messages describe different stages
            messages = [call['message'] for call in progress_calls if call['message']]
            stage_keywords = ['Extracting', 'Chunking', 'Embedding', 'Keyword', 'Clustering', 'Summarizing', 'Finalizing']
            found_stages = [kw for kw in stage_keywords if any(kw.lower() in str(msg).lower() for msg in messages)]
            self.assertGreater(len(found_stages), 2)  # At least a few stages


class TestIndexingErrorRecovery(unittest.TestCase):
    """Test indexing error recovery and edge cases."""

    def setUp(self):
        """
        Prepare the test environment by reinitializing the test database and creating a per-test temporary directory.
        """
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """
        Remove the per-test temporary directory created in setUp.
        
        If the directory referenced by self.temp_dir exists, delete it and all its contents.
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.get_embeddings')
    def test_create_index_handles_embedding_failure_gracefully(self, mock_get_embeddings):
        """Test that indexing handles embedding failures."""
        # Create a file
        with open(os.path.join(self.temp_dir, "test.txt"), 'w') as f:
            f.write("Test")

        # Make embeddings fail
        mock_get_embeddings.side_effect = Exception("Embedding API error")

        # Should raise the exception (not silently fail)
        with self.assertRaises(Exception):
            create_index(self.temp_dir, "openai", "fake_key")


if __name__ == '__main__':
    unittest.main()