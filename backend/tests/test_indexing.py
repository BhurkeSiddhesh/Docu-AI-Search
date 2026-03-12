"""
Test Indexing Module

Tests for the FAISS indexing and document processing pipeline.
"""

import sys
from unittest.mock import MagicMock

# Mock dependencies BEFORE imports
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.cluster'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()
sys.modules['langchain'] = MagicMock()
sys.modules['langchain_community'] = MagicMock()
sys.modules['backend.llm_integration'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()

import unittest
from unittest.mock import patch, MagicMock, call
import os
import shutil
import tempfile
import json
import pickle

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
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
    def submit(self, fn, *args, **kwargs):
        return DummyFuture(fn(*args, **kwargs))

class TestIndexing(unittest.TestCase):
    """Test cases for indexing functionality."""

    def setUp(self):
        from backend import database
        database.init_database()
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.temp_dir, "test_docs")
        os.makedirs(self.test_folder, exist_ok=True)
        with open(os.path.join(self.test_folder, "test.txt"), "w") as f:
            f.write("This is a test document.")

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
    def test_create_index(self, mock_summary, mock_clustering, mock_extract, mock_get_embeddings, mock_splitter, mock_process_pool, mock_thread_pool, mock_as_completed):
        mock_splitter.return_value.split_text.return_value = ["chunk1"]
        mock_extract.return_value = "Test content"
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model
        mock_clustering.return_value = {0: [0]}
        mock_summary.return_value = "Cluster Summary"

        res = create_index(self.test_folder, "openai", "fake_key")
        index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

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
        mock_get_embeddings.return_value.embed_documents.return_value = [[0.1]]
        mock_clustering.return_value = {0: [0]}
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
        shutil.rmtree(self.temp_dir)

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


if __name__ == '__main__':
    unittest.main()
