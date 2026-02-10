import unittest
import sys
import tempfile
import os
import shutil
import json
from unittest.mock import patch, MagicMock, call

# Mock missing dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()
sys.modules['docx'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['pptx'] = MagicMock()
sys.modules['openpyxl'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.cluster'] = MagicMock()
sys.modules['sklearn.mixture'] = MagicMock()

# Setup mocks before importing backend.indexing
sys.modules['langchain_text_splitters'].CharacterTextSplitter.return_value.split_text.return_value = ["chunk1"]

import numpy as np
from backend.indexing import create_index, save_index, load_index

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
        """Set up test environment before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_folder = self.temp_dir
        
        # Create a dummy file
        self.test_file = os.path.join(self.test_folder, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("This is a test document content for indexing.")
            
        # Global patches for executors
        self.pp_patcher = patch('concurrent.futures.ProcessPoolExecutor', side_effect=MockExecutor)
        self.tp_patcher = patch('concurrent.futures.ThreadPoolExecutor', side_effect=MockExecutor)
        self.ac_patcher = patch('concurrent.futures.as_completed', side_effect=lambda fs: fs)
        self.pp_patcher.start()
        self.tp_patcher.start()
        self.ac_patcher.start()

        # Patch database calls
        self.db_clear_files_patcher = patch('backend.database.clear_all_files')
        self.db_clear_clusters_patcher = patch('backend.database.clear_clusters')
        self.db_add_file_patcher = patch('backend.database.add_file')
        self.db_add_cluster_patcher = patch('backend.database.add_cluster')

        self.mock_clear_files = self.db_clear_files_patcher.start()
        self.mock_clear_clusters = self.db_clear_clusters_patcher.start()
        self.mock_add_file = self.db_add_file_patcher.start()
        self.mock_add_cluster = self.db_add_cluster_patcher.start()

        sys.modules['langchain_text_splitters'].CharacterTextSplitter.return_value.split_text.return_value = ["chunk1"]

    def tearDown(self):
        """Clean up after each test method."""
        self.pp_patcher.stop()
        self.tp_patcher.stop()
        self.ac_patcher.stop()

        self.db_clear_files_patcher.stop()
        self.db_clear_clusters_patcher.stop()
        self.db_add_file_patcher.stop()
        self.db_add_cluster_patcher.stop()

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index(self, mock_extract_text, mock_get_embeddings):
        """Test creating an index."""
        mock_extract_text.return_value = "This is test content for indexing."
        
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model
        
        with patch('backend.indexing.get_tags', return_value="test, indexing"),              patch('backend.indexing.perform_global_clustering', return_value={0: [0]}),              patch('backend.indexing.smart_summary', return_value="Summary"):
            res = create_index(self.test_folder, "openai", "fake_api_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res
            
            self.assertIsNotNone(index)
            self.assertIsNotNone(docs)
            self.assertIsNotNone(tags)
            self.assertEqual(len(docs), 1)
            self.assertEqual(len(tags), 1)

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_empty_folder(self, mock_extract_text, mock_get_embeddings):
        """Test creating an index with empty folder."""
        empty_folder = os.path.join(self.temp_dir, "empty_folder")
        os.makedirs(empty_folder, exist_ok=True)
        
        mock_extract_text.return_value = None
        mock_embeddings_model = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings_model
        
        with patch('backend.indexing.get_tags', return_value=""),              patch('backend.indexing.perform_global_clustering', return_value={}),              patch('backend.indexing.smart_summary', return_value="Summary"):
            res = create_index(empty_folder, "openai", "fake_api_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            self.assertIsNone(index)
            self.assertIsNone(docs)
            self.assertIsNone(tags)
    
    @patch('backend.indexing.faiss')
    def test_save_and_load_index(self, mock_faiss):
        """Test saving and loading an index."""
        index = MagicMock()
        index.ntotal = 1

        # Configure write_index to touch file
        def side_effect_write(idx, path):
            with open(path, 'w') as f:
                f.write("dummy")
        
        mock_faiss.write_index.side_effect = side_effect_write
        mock_faiss.read_index.return_value = index

        docs = [{"text": "Test document"}]
        tags = [["test", "tag"]]
        
        index_path = os.path.join(self.temp_dir, "test_index.faiss")

        # Manually create dummy index so load_index finds it (in case save_index logic changes or side_effect issues)
        # But here we rely on side_effect which is set on mock_faiss

        save_index(index, docs, tags, index_path)
        
        # Verify write_index called
        mock_faiss.write_index.assert_called_with(index, index_path)

        # Verify JSONs exist
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_index_docs.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_index_tags.json")))
        
        # Load the index
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res
        
        self.assertIsNotNone(loaded_index)
        self.assertEqual(loaded_docs, docs)
        self.assertEqual(loaded_tags, tags)
    
    @patch('backend.indexing.faiss')
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('json.load')
    def test_load_index(self, mock_json_load, mock_open, mock_exists, mock_faiss):
        """Test loading an index."""
        mock_faiss_index = MagicMock()
        mock_faiss.read_index.return_value = mock_faiss_index
        
        mock_exists.return_value = True
        
        mock_json_load.side_effect = [
             [{"text": "Test document"}],
             [["test", "tag"]],
             ["summary"],
             {"0": [0]}
        ]
        
        index_path = "fake_index.faiss"
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res
        
        mock_faiss.read_index.assert_called()
        self.assertEqual(mock_json_load.call_count, 4)
        
        self.assertEqual(loaded_index, mock_faiss_index)
        self.assertEqual(loaded_docs, [{"text": "Test document"}])
        self.assertEqual(loaded_tags, [["test", "tag"]])
        self.assertEqual(clus_map, {0: [0]})

if __name__ == '__main__':
    unittest.main()


class TestIndexingMultipleFolders(unittest.TestCase):
    """Test indexing with multiple folders."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.folder1 = os.path.join(self.temp_dir, "folder1")
        self.folder2 = os.path.join(self.temp_dir, "folder2")
        os.makedirs(self.folder1, exist_ok=True)
        os.makedirs(self.folder2, exist_ok=True)
        
        with open(os.path.join(self.folder1, "doc1.txt"), 'w') as f:
            f.write("Content from folder 1")
        with open(os.path.join(self.folder2, "doc2.txt"), 'w') as f:
            f.write("Content from folder 2")

        self.db_clear_files_patcher = patch('backend.database.clear_all_files')
        self.db_clear_clusters_patcher = patch('backend.database.clear_clusters')
        self.db_add_file_patcher = patch('backend.database.add_file')
        self.db_add_cluster_patcher = patch('backend.database.add_cluster')

        self.mock_clear_files = self.db_clear_files_patcher.start()
        self.mock_clear_clusters = self.db_clear_clusters_patcher.start()
        self.mock_add_file = self.db_add_file_patcher.start()
        self.mock_add_cluster = self.db_add_cluster_patcher.start()

        sys.modules['langchain_text_splitters'].CharacterTextSplitter.return_value.split_text.return_value = ["chunk1"]


    def tearDown(self):
        self.db_clear_files_patcher.stop()
        self.db_clear_clusters_patcher.stop()
        self.db_add_file_patcher.stop()
        self.db_add_cluster_patcher.stop()
        shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_multiple_folders(self, mock_extract_text, mock_get_embeddings):
        """Test creating index from multiple folders."""
        mock_extract_text.return_value = "Test content"
        
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_get_embeddings.return_value = mock_embeddings_model
        
        # We expect 2 chunks (one per file). Clustering expects indices 0 and 1.
        with patch('backend.indexing.get_tags', return_value="test"),              patch('backend.indexing.perform_global_clustering', return_value={0: [0, 1]}),              patch('backend.indexing.smart_summary', return_value="Summary"):
            res = create_index(
                [self.folder1, self.folder2], 
                "openai", 
                "fake_key"
            )
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res
            
            self.assertIsNotNone(index)
            self.assertEqual(len(docs), 2)


    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_with_progress_callback(self, mock_extract_text, mock_get_embeddings):
        """Test progress callback during indexing."""
        mock_extract_text.return_value = "Test content"
        
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model
        
        progress_calls = []
        def progress_callback(current, total, filename):
            progress_calls.append((current, total, filename))
        
        with patch('backend.indexing.get_tags', return_value="test"),              patch('backend.indexing.perform_global_clustering', return_value={0: [0]}),              patch('backend.indexing.smart_summary', return_value="Summary"):
            create_index(self.folder1, "openai", "fake_key", progress_callback=progress_callback)
            
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
            self.assertIsNone(res[0])

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index_string_folder_path(self, mock_extract_text, mock_get_embeddings):
        """Test that string folder path is converted to list."""
        mock_extract_text.return_value = "Test content"
        
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model
        
        with patch('backend.indexing.get_tags', return_value="test"),              patch('backend.indexing.perform_global_clustering', return_value={0: [0]}),              patch('backend.indexing.smart_summary', return_value="Summary"):
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
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.faiss')
    def test_save_index_creates_all_files(self, mock_faiss):
        """Test that save_index creates .faiss, _docs.json, and _tags.json files."""
        index = MagicMock()
        
        docs = ["Document 1", "Document 2"]
        tags = [["tag1"], ["tag2"]]
        
        index_path = os.path.join(self.temp_dir, "index.faiss")
        save_index(index, docs, tags, index_path)
        
        mock_faiss.write_index.assert_called_with(index, index_path)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_docs.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_tags.json")))


class TestLoadIndex(unittest.TestCase):
    """Dedicated tests for load_index function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.faiss')
    def test_load_index_preserves_data(self, mock_faiss):
        """Test that load_index correctly restores saved data."""
        mock_faiss.read_index.return_value = MagicMock()
        mock_faiss.read_index.return_value.ntotal = 2
        
        original_index = MagicMock()
        
        original_docs = [{"text": "Doc A"}, {"text": "Doc B"}]
        original_tags = [["alpha"], ["beta", "gamma"]]
        
        index_path = os.path.join(self.temp_dir, "test.faiss")

        # Touch index file
        with open(index_path, 'w') as f:
            f.write("dummy")

        save_index(original_index, original_docs, original_tags, index_path)
        
        # Load and verify
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res
        
        self.assertIsNotNone(loaded_index)
        self.assertEqual(loaded_docs, original_docs)
        self.assertEqual(loaded_tags, original_tags)
