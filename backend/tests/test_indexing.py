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
from unittest.mock import patch
import os
import shutil
import tempfile
import json

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

        # Touch faiss index
        with open(index_path, 'w') as f: f.write("dummy")

        # Create JSON files
        with open(base_path + '_docs.json', 'w') as f:
            json.dump([{"text": "Test document"}], f)
        with open(base_path + '_tags.json', 'w') as f:
            json.dump([["test", "tag"]], f)

        # Optional files (summ) - not needed for basic test, we'll leave them absent to test fallback

        mock_faiss_index = MagicMock()
        mock_read_index.return_value = mock_faiss_index
        
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res
        
        self.assertEqual(loaded_index, mock_faiss_index)
        self.assertEqual(loaded_docs, [{"text": "Test document"}])
        self.assertEqual(loaded_tags, [["test", "tag"]])


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
        index_path = os.path.join(self.temp_dir, "index.faiss")
        save_index(index, [], [], index_path)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_docs.json")))

if __name__ == '__main__':
    unittest.main()
