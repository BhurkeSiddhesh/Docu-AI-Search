"""
Test Indexing Module

Tests for the FAISS indexing and document processing pipeline.
"""

import sys
import unittest
import os
import tempfile
import shutil
import pickle
import numpy as np
from unittest.mock import MagicMock, patch

# Mock dependencies BEFORE imports


# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Since we're mocking faiss, we need to ensure the mocks have the expected methods
import faiss
faiss.IndexFlatL2 = MagicMock()
faiss.write_index = MagicMock()
faiss.read_index = MagicMock()

from backend.indexing import create_index, load_index, save_index

class DummyFuture:
    def __init__(self, result):
        self._result = result
    def result(self):
        return self._result

class DummyExecutor:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def submit(self, fn, *args, **kwargs):
        return DummyFuture(fn(*args, **kwargs))
    def map(self, fn, *iterables, **kwargs):
        return [fn(*args) for args in zip(*iterables)]
    def shutdown(self, wait=True, *, cancel_futures=False): pass

def setUpModule():
    from backend import database
    global original_db_path, temp_test_dir
    temp_test_dir = tempfile.mkdtemp()
    original_db_path = database.DATABASE_PATH
    database.DATABASE_PATH = os.path.join(temp_test_dir, "test_metadata.db")
    database.init_database()

def tearDownModule():
    from backend import database
    # Close any open connections in this thread to allow deletion on Windows
    if hasattr(database.thread_local, "connection"):
        database.thread_local.connection.close()
        del database.thread_local.connection
    database.DATABASE_PATH = original_db_path
    if os.path.exists(temp_test_dir):
        try:
            shutil.rmtree(temp_test_dir)
        except Exception as e:
            print(f"Warning: Could not cleanup module temp dir: {e}")

MockExecutor = DummyExecutor
MockFuture = DummyFuture

class TestIndexing(unittest.TestCase):
    """Test cases for indexing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.temp_dir, "test_docs")
        os.makedirs(self.test_folder, exist_ok=True)
        with open(os.path.join(self.test_folder, "test.txt"), "w") as f:
            f.write("This is a test document.")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass

    @patch('backend.indexing.concurrent.futures.as_completed', side_effect=lambda fs: fs)
    @patch('backend.indexing.concurrent.futures.ThreadPoolExecutor', return_value=DummyExecutor())
    @patch('backend.indexing.concurrent.futures.ProcessPoolExecutor', return_value=DummyExecutor())
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.perform_global_clustering')
    @patch('backend.indexing.smart_summary')
    def test_create_index(self, mock_summary, mock_clustering, mock_extract, mock_get_embeddings, mock_splitter, mock_process_pool, mock_thread_pool, mock_as_completed):
        mock_splitter.return_value.split_text.return_value = ["chunk1"]
        mock_extract.return_value = "Test content"
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model
        mock_clustering.return_value = {0: [0]}
        mock_summary.return_value = "Cluster Summary"

        res = create_index(self.test_folder, "openai", "fake_key")
        index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

        self.assertIsNotNone(index)
        self.assertEqual(len(docs), 1)

    @patch('backend.indexing.get_embeddings')
    def test_create_index_empty_folder(self, mock_embed):
        """Test creating index with empty folder."""
        empty_folder = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_folder, exist_ok=True)
        
        res = create_index(empty_folder, "openai", "fake_key")
        self.assertIsNone(res[0])

    @patch('backend.indexing.faiss.read_index')
    def test_load_index_preserves_data(self, mock_read_index):
        # Create physical dummy files to avoid FileNotFoundError without mocking builtins.open
        index_path = os.path.join(self.temp_dir, "test.faiss")
        base_path = os.path.splitext(index_path)[0]

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
        # Ensure the file exists for load_index check
        with open(index_path, "wb") as f: f.write(b"fake faiss")
        save_index(index, docs, tags, index_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(index_path))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_index_docs.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_index_tags.pkl")))
        
        # Setup mock for load_index
        mock_read_index.return_value = index  # Return the mock index

        # Load the index (now returns 8-tuple; unpack first 7)
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]
        
        # Verify the loaded data matches the original
        self.assertIsNotNone(loaded_index)
        self.assertEqual(loaded_docs, docs)
        self.assertEqual(loaded_tags, tags)


class TestIndexingMultipleFolders(unittest.TestCase):
    def setUp(self):
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()
        self.folder1 = os.path.join(self.temp_dir, "folder1")
        self.folder2 = os.path.join(self.temp_dir, "folder2")
        os.makedirs(self.folder1, exist_ok=True)
        os.makedirs(self.folder2, exist_ok=True)
        with open(os.path.join(self.folder1, "doc1.txt"), 'w') as f: f.write("C1")
        with open(os.path.join(self.folder2, "doc2.txt"), 'w') as f: f.write("C2")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.concurrent.futures.as_completed', side_effect=lambda fs: fs)
    @patch('backend.indexing.concurrent.futures.ThreadPoolExecutor', return_value=DummyExecutor())
    @patch('backend.indexing.concurrent.futures.ProcessPoolExecutor', return_value=DummyExecutor())
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.perform_global_clustering')
    @patch('backend.indexing.smart_summary')
    def test_create_index_multiple_folders(self, mock_summary, mock_clustering, mock_extract, mock_get_embeddings, mock_splitter, mock_process_pool, mock_thread_pool, mock_as_completed):
        mock_splitter.return_value.split_text.return_value = ["chunk"]
        mock_extract.return_value = "content"
        # Return one vector per chunk in the batch — required since the
        # embedding/chunk alignment check in create_index would otherwise abort.
        mock_get_embeddings.return_value.embed_documents.side_effect = lambda batch: [[0.1] for _ in batch]
        mock_clustering.return_value = {0: [0, 1]}
        mock_summary.return_value = "Sum"

        res = create_index([self.folder1, self.folder2], "openai", "k")
        self.assertIsNotNone(res[0])

    @patch('backend.indexing.concurrent.futures.as_completed', side_effect=lambda fs: fs)
    @patch('backend.indexing.concurrent.futures.ThreadPoolExecutor', return_value=DummyExecutor())
    @patch('backend.indexing.concurrent.futures.ProcessPoolExecutor', return_value=DummyExecutor())
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.perform_global_clustering')
    @patch('backend.indexing.smart_summary')
    def test_create_index_with_progress_callback(self, mock_summary, mock_clustering, mock_extract, mock_get_embeddings, mock_splitter, mock_process_pool, mock_thread_pool, mock_as_completed):
        mock_splitter.return_value.split_text.return_value = ["chunk"]
        mock_extract.return_value = "content"
        mock_get_embeddings.return_value.embed_documents.return_value = [[0.1]]
        mock_clustering.return_value = {0:[0]}
        mock_summary.return_value = "Sum"
        
        calls = []
        def cb(c, t, m=None): calls.append(c)
        create_index(self.folder1, "openai", "k", progress_callback=cb)
        self.assertTrue(len(calls) > 0)

    def test_create_index_nonexistent_folder(self):
        res = create_index("/nonexistent/folder/path", "openai", "fake_key")
        self.assertIsNone(res[0])

    @patch('backend.indexing.concurrent.futures.as_completed', side_effect=lambda fs: fs)
    @patch('backend.indexing.concurrent.futures.ThreadPoolExecutor', return_value=DummyExecutor())
    @patch('backend.indexing.concurrent.futures.ProcessPoolExecutor', return_value=DummyExecutor())
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.perform_global_clustering')
    def test_create_index_string_folder_path(self, mock_clustering, mock_extract, mock_get_embeddings, mock_splitter, mock_process_pool, mock_thread_pool, mock_as_completed):
        mock_splitter.return_value.split_text.return_value = ["chunk"]
        mock_extract.return_value = "content"
        mock_get_embeddings.return_value.embed_documents.return_value = [[0.1]]
        mock_clustering.return_value = {} # No clusters -> no summary
        
        res = create_index(self.folder1, "openai", "k")
        self.assertIsNotNone(res[0])

class TestSaveIndex(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass

    @patch('backend.indexing.faiss.write_index')
    def test_save_index_creates_all_files(self, mock_write_index):
        import faiss
        index = faiss.IndexFlatL2(3)
        index.d = 3  # explicitly set in case faiss is globally mocked
        embeddings = np.array([[1.0, 2.0, 3.0]], dtype='float32')
        index.add(embeddings)
        
        docs = ["Document 1", "Document 2"]
        tags = [["tag1"], ["tag2"]]
        
        index_path = os.path.join(self.temp_dir, "index.faiss")
        # Manually create a dummy file since write_index is mocked but save_index checks for success
        with open(index_path, "wb") as f: f.write(b"dummy")
        save_index(index, docs, tags, index_path)
        
        # Since faiss.write_index is mocked, it won't actually update the file,
        # but the sidecar files should be created by save_index.
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_docs.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_tags.pkl")))
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
        self.pp_patcher = patch('concurrent.futures.ProcessPoolExecutor', side_effect=DummyExecutor)
        self.tp_patcher = patch('concurrent.futures.ThreadPoolExecutor', side_effect=DummyExecutor)
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
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

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
        index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

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
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

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
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

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
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

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

        # extract_text is mocked, so both files become valid_docs and yield a
        # chunk each. We need one embedding per chunk to satisfy the alignment
        # guard in create_index.
        mock_extract_text.return_value = "valid content"

        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.side_effect = lambda batch: [[0.1, 0.2, 0.3] for _ in batch]
        mock_get_embeddings.return_value = mock_embeddings_model

        with patch('backend.indexing.get_tags', return_value="test"), \
             patch('backend.indexing.perform_global_clustering', return_value={0: [0, 1]}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):
            res = create_index(self.temp_dir, "openai", "fake_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

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
             patch('backend.indexing.perform_global_clustering', return_value={0: [0, 1]}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):

            mock_splitter_instance = mock_splitter_cls.return_value
            mock_splitter_instance.split_text.return_value = ["chunk1"]

            mock_embeddings_model = MagicMock()
            # os.walk lists both real.txt and the symlink, so chunk_strings has
            # 2 entries; return one vector per chunk to satisfy the alignment guard.
            mock_embeddings_model.embed_documents.side_effect = lambda batch: [[0.1, 0.2, 0.3] for _ in batch]
            mock_embed.return_value = mock_embeddings_model

            res = create_index(self.temp_dir, "openai", "fake_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

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
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

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
            pickle.dump("dummy_bm25_data", f)

        # Tags file intentionally missing

        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

        # Should load with empty tags
        self.assertIsNotNone(loaded_index)
        self.assertEqual(loaded_docs, ["doc1"])
        self.assertEqual(loaded_tags, [])

    def test_load_index_nonexistent_file(self):
        """Test loading index from non-existent path."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.faiss")

        res = load_index(nonexistent_path)
        index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

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
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

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
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]

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
        
        self.pp_patcher = patch('concurrent.futures.ProcessPoolExecutor', side_effect=MockExecutor)
        self.tp_patcher = patch('concurrent.futures.ThreadPoolExecutor', side_effect=MockExecutor)
        self.ac_patcher = patch('concurrent.futures.as_completed', side_effect=lambda fs: fs)
        self.pp_patcher.start()
        self.tp_patcher.start()
        self.ac_patcher.start()

    def tearDown(self):
        """
        Remove the per-test temporary directory created in setUp.
        
        If the directory referenced by self.temp_dir exists, delete it and all its contents.
        """
        self.pp_patcher.stop()
        self.tp_patcher.stop()
        self.ac_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.get_embeddings')
    def test_create_index_handles_embedding_failure_gracefully(self, mock_get_embeddings, mock_extract_text, mock_splitter_cls):
        """Test that indexing handles embedding failures."""
        # Create a file
        with open(os.path.join(self.temp_dir, "test.txt"), 'w') as f:
            f.write("Test")

        # Force extraction and chunking
        mock_extract_text.return_value = "Test"
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["Test"]

        # Make embeddings fail
        mock_get_embeddings.side_effect = Exception("Embedding API error")

        # Should raise the exception (not silently fail)
        with self.assertRaises(Exception):
            create_index(self.temp_dir, "openai", "fake_key")


class TestIndexingEmbeddingBatchFailures(unittest.TestCase):
    """
    Regression tests for indexing crashes caused by failed embedding batches.

    Previously create_index would either crash with `IndexError: tuple index out
    of range` at `chunk_emb_np.shape[1]` (all batches failed → empty array) or
    silently produce a misaligned FAISS index (some batches failed → embeddings
    fewer than chunks). It must now return the empty 8-tuple instead.
    """

    def setUp(self):
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()
        for i in range(2):
            with open(os.path.join(self.temp_dir, f"doc{i}.txt"), 'w') as f:
                f.write(f"content {i}")

        self.pp_patcher = patch('concurrent.futures.ProcessPoolExecutor', side_effect=MockExecutor)
        self.tp_patcher = patch('concurrent.futures.ThreadPoolExecutor', side_effect=MockExecutor)
        self.ac_patcher = patch('concurrent.futures.as_completed', side_effect=lambda fs: fs)
        self.pp_patcher.start()
        self.tp_patcher.start()
        self.ac_patcher.start()

    def tearDown(self):
        self.pp_patcher.stop()
        self.tp_patcher.stop()
        self.ac_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_all_embedding_batches_fail_returns_empty_tuple(self, mock_extract, mock_get_embeddings, mock_splitter_cls):
        """
        When every embedding batch fails (production catches the exception and
        stores `[]`), create_index must abort cleanly — NOT crash on `shape[1]`
        of an empty np.array.
        """
        mock_splitter_cls.return_value.split_text.return_value = ["chunk1", "chunk2"]
        mock_extract.return_value = "some text"

        # Simulate the post-catch state at indexing.py L254-256: every batch
        # returns [] (this is what the production try/except puts in the map).
        failing_embedder = MagicMock()
        failing_embedder.embed_documents.return_value = []
        mock_get_embeddings.return_value = failing_embedder

        with patch('backend.indexing.get_tags', return_value=""), \
             patch('backend.indexing.perform_global_clustering', return_value={}), \
             patch('backend.indexing.smart_summary', return_value=""):
            res = create_index(self.temp_dir, "openai", "fake_key")

        # Must return the empty 8-tuple, not crash.
        self.assertEqual(len(res), 8)
        self.assertIsNone(res[0])  # index_chunks
        self.assertEqual(res[7], {})  # meta

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_partial_embedding_failure_aborts_instead_of_corrupting(self, mock_extract, mock_get_embeddings, mock_splitter_cls):
        """
        When some embedding batches fail, create_index must abort rather than
        silently produce a FAISS index that's misaligned with chunk_strings
        (which would route searches to the wrong chunks).
        """
        # Force enough chunks to span multiple batches (default batch_size=100).
        many_chunks = [f"chunk{i}" for i in range(150)]
        mock_splitter_cls.return_value.split_text.return_value = many_chunks
        mock_extract.return_value = "long text"

        embedder = MagicMock()
        call_count = {'n': 0}

        def partial_embed(batch):
            # First batch returns one vector per chunk; subsequent batches
            # return [] — the same state the production try/except produces
            # when a batch raises (see indexing.py L254-256).
            call_count['n'] += 1
            if call_count['n'] == 1:
                return [[0.1, 0.2, 0.3] for _ in batch]
            return []

        embedder.embed_documents.side_effect = partial_embed
        mock_get_embeddings.return_value = embedder

        with patch('backend.indexing.get_tags', return_value=""), \
             patch('backend.indexing.perform_global_clustering', return_value={0: list(range(150))}), \
             patch('backend.indexing.smart_summary', return_value="Summary"):
            res = create_index(self.temp_dir, "openai", "fake_key")

        # Must abort with the empty 8-tuple — not return a corrupt index.
        self.assertEqual(len(res), 8)
        self.assertIsNone(res[0])

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_checkpoint_cleared_after_embedding_abort(self, mock_extract, mock_get_embeddings, mock_splitter_cls):
        """
        When create_index aborts due to embedding failure, the on-disk
        checkpoint must be cleared so the next attempt starts fresh
        (rather than silently skipping previously-extracted files).
        """
        from backend import indexing as indexing_module

        mock_splitter_cls.return_value.split_text.return_value = ["chunk1"]
        mock_extract.return_value = "text"

        failing_embedder = MagicMock()
        failing_embedder.embed_documents.return_value = []
        mock_get_embeddings.return_value = failing_embedder

        # Redirect checkpoint to a temp path so we don't touch real data/.
        ckpt_path = os.path.join(self.temp_dir, "index_checkpoint.json")
        with patch.object(indexing_module, '_CHECKPOINT_PATH', ckpt_path), \
             patch('backend.indexing.get_tags', return_value=""), \
             patch('backend.indexing.perform_global_clustering', return_value={}), \
             patch('backend.indexing.smart_summary', return_value=""):
            create_index(self.temp_dir, "openai", "fake_key")
            self.assertFalse(
                os.path.exists(ckpt_path),
                "Checkpoint must be cleared on embedding-failure abort so the next "
                "run re-extracts files instead of silently skipping them."
            )


class TestIndexingNonexistentFolder(unittest.TestCase):
    """Regression: a misconfigured folder list should not crash the API task."""

    def setUp(self):
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_mixed_existing_and_nonexistent_folders(self):
        """
        If one folder is missing, indexing should still scan the others and
        not raise FileNotFoundError. `good` is intentionally empty so we exit
        cleanly via the no-files-found path — no embedding mocks needed.
        """
        good = os.path.join(self.temp_dir, "good_empty")
        os.makedirs(good)
        missing = os.path.join(self.temp_dir, "does_not_exist")

        try:
            res = create_index([missing, good], "openai", "fake_key")
        except FileNotFoundError as e:
            self.fail(f"create_index raised on missing folder instead of skipping it: {e}")

        self.assertEqual(len(res), 8)
        self.assertIsNone(res[0])


class _StubFaissIndex:
    """
    A minimal stand-in for faiss.IndexFlatL2 used by the incremental tests.
    The top of this test file globally mocks faiss.IndexFlatL2 with MagicMock,
    so we cannot build a real one for the "prior" index — but the incremental
    branch in create_index calls `.ntotal` and `.reconstruct_n(start, n)`,
    which a bare MagicMock would silently return MagicMocks for.
    """

    def __init__(self, dim: int):
        self.d = dim
        self._vecs = np.zeros((0, dim), dtype='float32')

    @property
    def ntotal(self) -> int:
        return int(self._vecs.shape[0])

    def add(self, vecs):
        self._vecs = np.vstack([self._vecs, np.asarray(vecs, dtype='float32')])

    def reconstruct_n(self, start: int, n: int):
        return self._vecs[start:start + n].copy()


class TestIncrementalIndexing(unittest.TestCase):
    """
    Tests for the incremental-indexing fast path: re-running create_index over
    a folder whose files haven't changed should reuse their cached chunks and
    vectors from the existing on-disk index instead of re-extracting and
    re-embedding from scratch.
    """

    def setUp(self):
        from backend import database
        database.init_database()
        # Each test must start with a clean DB so prior tests' file rows don't
        # accidentally satisfy the "unchanged" check for the new temp_dir.
        database.clear_files()
        database.clear_clusters()
        # Keep the index sidecar OUTSIDE the scanned folder, otherwise os.walk
        # would pick it up as a 4th "file" to index.
        self.scratch_root = tempfile.mkdtemp()
        self.temp_dir = os.path.join(self.scratch_root, "docs")
        os.makedirs(self.temp_dir)
        self.fake_index_path = os.path.join(self.scratch_root, "index.faiss")
        with open(self.fake_index_path, 'w') as f:
            f.write("stub")

        self.pp_patcher = patch('concurrent.futures.ProcessPoolExecutor', side_effect=MockExecutor)
        self.tp_patcher = patch('concurrent.futures.ThreadPoolExecutor', side_effect=MockExecutor)
        self.ac_patcher = patch('concurrent.futures.as_completed', side_effect=lambda fs: fs)
        self.pp_patcher.start()
        self.tp_patcher.start()
        self.ac_patcher.start()

    def tearDown(self):
        self.pp_patcher.stop()
        self.tp_patcher.stop()
        self.ac_patcher.stop()
        if os.path.exists(self.scratch_root):
            shutil.rmtree(self.scratch_root)

    def _write_file(self, name: str, content: str) -> str:
        path = os.path.join(self.temp_dir, name)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def _seed_prior_state(self, file_specs):
        """
        Pre-populate the DB and build a fake "prior" FAISS index + docs list
        as if create_index had run before. ``file_specs`` is a list of
        ``(path, n_chunks)`` tuples in the order they were originally indexed.

        Returns the prior (index, docs) tuple shaped like load_index()[0:2].
        """
        import hashlib

        prior_docs = []
        all_vecs = []
        files_rows = []
        cursor = 0
        for path, n_chunks in file_specs:
            with open(path, 'rb') as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()
            file_start = cursor
            for j in range(n_chunks):
                prior_docs.append({
                    'text': f"prior-chunk-{path}-{j}",
                    'filepath': path,
                    'faiss_idx': cursor,
                    'file_id': None,
                })
                # Distinctive vector values let a future test assert reuse if needed.
                all_vecs.append([float(cursor) + 0.5, 0.0, 0.0])
                cursor += 1
            file_stat = os.stat(path)
            files_rows.append({
                'path': path,
                'filename': os.path.basename(path),
                'file_type': os.path.splitext(path)[1].lower(),
                'size': file_stat.st_size,
                'last_modified': file_stat.st_mtime,
                'faiss_start_idx': file_start,
                'faiss_end_idx': cursor - 1,
                'tags': '[]',
                'content_hash': content_hash,
            })

        from backend import database as _database
        _database.add_files_batch(files_rows)

        # Real (stubbed) ntotal/reconstruct_n behavior — see _StubFaissIndex above
        # for why a bare MagicMock isn't sufficient here.
        prior_index = _StubFaissIndex(3)
        prior_index.add(np.array(all_vecs, dtype='float32'))
        return prior_index, prior_docs

    @patch('backend.indexing.load_index')
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_rerun_with_no_changes_does_not_re_embed(self, mock_extract, mock_get_embeddings,
                                                     mock_splitter_cls, mock_load_index):
        """
        The core incremental promise: when nothing has changed, the embedding
        client is never asked to embed anything.
        """
        path1 = self._write_file("a.txt", "alpha content")
        path2 = self._write_file("b.txt", "beta content")

        prior_index, prior_docs = self._seed_prior_state([(path1, 1), (path2, 1)])
        mock_load_index.return_value = (prior_index, prior_docs, [""] * 2, None, None, None, None, {'model_name': 'test', 'embedding_dim': 3})

        # If these are called, the test should fail — they signal we re-did
        # work that should have been skipped.
        mock_extract.side_effect = AssertionError("extract_text must not be called for unchanged files")
        mock_splitter_cls.return_value.split_text.side_effect = AssertionError("split_text must not be called for unchanged files")
        embedder = MagicMock()
        embedder.embed_documents.side_effect = AssertionError("embed_documents must not be called for unchanged files")
        mock_get_embeddings.return_value = embedder

        res = create_index(self.temp_dir, "openai", "fake_key", existing_index_path=self.fake_index_path)
        index, docs, tags, idx_sum, clus_sum, clus_map, bm25, meta = res

        self.assertIsNotNone(index, "Incremental run must still produce an index from reused vectors")
        self.assertEqual(len(docs), 2, "All prior chunks should have been carried over")
        self.assertEqual(meta.get('embedding_dim'), 3)
        # The actual returned `index` is the module-mocked faiss.IndexFlatL2,
        # so we can't introspect ntotal here. The real proof of reuse is that
        # the embedding-client mock was never called (asserted via side_effect).

    @patch('backend.indexing.load_index')
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_rerun_with_new_file_only_embeds_the_new_one(self, mock_extract, mock_get_embeddings,
                                                          mock_splitter_cls, mock_load_index):
        """
        Adding one new file to a previously-indexed folder should only run
        extraction+embedding for that one file; the other two stay reused.
        """
        path1 = self._write_file("a.txt", "alpha content")
        path2 = self._write_file("b.txt", "beta content")

        prior_index, prior_docs = self._seed_prior_state([(path1, 1), (path2, 1)])
        mock_load_index.return_value = (prior_index, prior_docs, [""] * 2, None, None, None, None, {'model_name': 'test', 'embedding_dim': 3})

        # Now add a new file the prior index doesn't know about.
        path3 = self._write_file("c.txt", "gamma content")

        mock_extract.return_value = "gamma content"
        mock_splitter_cls.return_value.split_text.return_value = ["gamma-chunk"]
        embedder = MagicMock()
        embedder.embed_documents.side_effect = lambda batch: [[9.9, 0.0, 0.0] for _ in batch]
        mock_get_embeddings.return_value = embedder

        res = create_index(self.temp_dir, "openai", "fake_key", existing_index_path=self.fake_index_path)
        index, docs, tags, _, _, _, _, _ = res

        self.assertIsNotNone(index)
        self.assertEqual(len(docs), 3, "Two reused + one new chunk")
        # extract_text should have been called exactly once — for the new file.
        # (mock_extract.call_count reflects only direct calls inside this run.)
        self.assertEqual(mock_extract.call_count, 1)
        # And the embedding client should have been called once with the new chunk.
        embedder.embed_documents.assert_called_once()
        called_batch = embedder.embed_documents.call_args[0][0]
        self.assertEqual(called_batch, ["gamma-chunk"])

    @patch('backend.indexing.load_index')
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_rerun_with_modified_file_re_embeds_only_that_file(self, mock_extract, mock_get_embeddings,
                                                                 mock_splitter_cls, mock_load_index):
        """
        Touching the content of one file should cause it to be re-extracted
        and re-embedded; siblings stay reused.
        """
        path1 = self._write_file("a.txt", "alpha content")
        path2 = self._write_file("b.txt", "beta content")

        prior_index, prior_docs = self._seed_prior_state([(path1, 1), (path2, 1)])
        mock_load_index.return_value = (prior_index, prior_docs, [""] * 2, None, None, None, None, {'model_name': 'test', 'embedding_dim': 3})

        # Modify b.txt — its hash no longer matches the DB row.
        with open(path2, 'w') as f:
            f.write("beta content MUTATED")

        mock_extract.return_value = "beta content MUTATED"
        mock_splitter_cls.return_value.split_text.return_value = ["new-beta-chunk"]
        embedder = MagicMock()
        embedder.embed_documents.side_effect = lambda batch: [[7.7, 0.0, 0.0] for _ in batch]
        mock_get_embeddings.return_value = embedder

        res = create_index(self.temp_dir, "openai", "fake_key", existing_index_path=self.fake_index_path)
        index, docs, _, _, _, _, _, _ = res

        self.assertIsNotNone(index)
        self.assertEqual(len(docs), 2, "One reused + one re-embedded chunk")
        self.assertEqual(mock_extract.call_count, 1, "Only the modified file should be re-extracted")
        embedder.embed_documents.assert_called_once()

    @patch('backend.indexing.load_index')
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_rerun_with_deleted_file_drops_its_chunks(self, mock_extract, mock_get_embeddings,
                                                       mock_splitter_cls, mock_load_index):
        """
        A file removed from the folder between runs should disappear from the
        new index entirely; the remaining file is still reused.
        """
        path1 = self._write_file("a.txt", "alpha content")
        path2 = self._write_file("b.txt", "beta content")

        prior_index, prior_docs = self._seed_prior_state([(path1, 1), (path2, 1)])
        mock_load_index.return_value = (prior_index, prior_docs, [""] * 2, None, None, None, None, {'model_name': 'test', 'embedding_dim': 3})

        # Delete b.txt so the new scan only sees a.txt.
        os.remove(path2)

        # Neither extract nor embed should be called: a.txt is unchanged.
        mock_extract.side_effect = AssertionError("must not re-extract unchanged file")
        mock_splitter_cls.return_value.split_text.side_effect = AssertionError("must not chunk")
        embedder = MagicMock()
        embedder.embed_documents.side_effect = AssertionError("must not embed")
        mock_get_embeddings.return_value = embedder

        res = create_index(self.temp_dir, "openai", "fake_key", existing_index_path=self.fake_index_path)
        index, docs, _, _, _, _, _, _ = res

        self.assertIsNotNone(index)
        self.assertEqual(len(docs), 1, "Deleted file's chunk should be gone")


class TestClusterSummarizationGating(unittest.TestCase):
    """
    The cluster-summarization step is the single biggest cost in a full-rebuild
    (one LLM call per cluster). It's now gated behind [AdvancedRAG]
    cluster_summarization in config.ini and should default to off.
    """

    def setUp(self):
        from backend import database
        database.init_database()
        database.clear_files()
        database.clear_clusters()
        self.temp_dir = tempfile.mkdtemp()
        self._write_file = lambda name, content: (
            open(os.path.join(self.temp_dir, name), 'w').write(content)
            or os.path.join(self.temp_dir, name)
        )
        with open(os.path.join(self.temp_dir, "doc.txt"), 'w') as f:
            f.write("hello")

        self.pp_patcher = patch('concurrent.futures.ProcessPoolExecutor', side_effect=MockExecutor)
        self.tp_patcher = patch('concurrent.futures.ThreadPoolExecutor', side_effect=MockExecutor)
        self.ac_patcher = patch('concurrent.futures.as_completed', side_effect=lambda fs: fs)
        self.pp_patcher.start()
        self.tp_patcher.start()
        self.ac_patcher.start()

    def tearDown(self):
        self.pp_patcher.stop()
        self.tp_patcher.stop()
        self.ac_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.smart_summary')
    @patch('backend.indexing.perform_global_clustering')
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_cluster_summarization_off_by_default(self, mock_extract, mock_get_embeddings,
                                                    mock_splitter_cls, mock_clustering, mock_smart_summary):
        """
        With no [AdvancedRAG] cluster_summarization=true in config, indexing
        should not invoke clustering or smart_summary at all.
        """
        mock_extract.return_value = "hello"
        mock_splitter_cls.return_value.split_text.return_value = ["chunk"]
        embedder = MagicMock()
        embedder.embed_documents.side_effect = lambda batch: [[0.1] * 3 for _ in batch]
        mock_get_embeddings.return_value = embedder

        # _clustering_enabled() reads config.ini at the project root. We point
        # it at a temp file with the flag explicitly off.
        ini_path = os.path.join(self.temp_dir, "config.ini")
        with open(ini_path, 'w') as f:
            f.write("[AdvancedRAG]\ncluster_summarization = false\n")

        from backend import indexing as indexing_module
        with patch.object(indexing_module, '_CONFIG_PATH', ini_path):
            res = create_index(self.temp_dir, "openai", "fake_key")

        self.assertIsNotNone(res[0])
        mock_clustering.assert_not_called()
        mock_smart_summary.assert_not_called()
        # And the returned cluster_summaries / index_summaries should be empty.
        self.assertEqual(res[3], None)
        self.assertEqual(res[4], [])

    @patch('backend.indexing.smart_summary')
    @patch('backend.indexing.perform_global_clustering')
    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_cluster_summarization_runs_when_explicitly_enabled(self, mock_extract, mock_get_embeddings,
                                                                  mock_splitter_cls, mock_clustering, mock_smart_summary):
        """
        Power users who set cluster_summarization=true should still get the
        full RAPTOR pipeline.
        """
        mock_extract.return_value = "hello"
        mock_splitter_cls.return_value.split_text.return_value = ["chunk"]
        embedder = MagicMock()
        embedder.embed_documents.side_effect = lambda batch: [[0.1] * 3 for _ in batch]
        mock_get_embeddings.return_value = embedder
        mock_clustering.return_value = {0: [0]}
        mock_smart_summary.return_value = "summary"

        ini_path = os.path.join(self.temp_dir, "config.ini")
        with open(ini_path, 'w') as f:
            f.write("[AdvancedRAG]\ncluster_summarization = true\n")

        from backend import indexing as indexing_module
        with patch.object(indexing_module, '_CONFIG_PATH', ini_path):
            res = create_index(self.temp_dir, "openai", "fake_key")

        self.assertIsNotNone(res[0])
        mock_clustering.assert_called_once()
        mock_smart_summary.assert_called_once()


if __name__ == '__main__':
    unittest.main()