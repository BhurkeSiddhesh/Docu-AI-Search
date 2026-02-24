"""
Test Indexing Module

Tests for the FAISS indexing and document processing pipeline.
"""

import sys
from unittest.mock import MagicMock

# Mock sklearn BEFORE any other imports to prevent ModuleNotFoundError
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.cluster'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()
sys.modules['langchain'] = MagicMock()
sys.modules['langchain_community'] = MagicMock()
sys.modules['backend.llm_integration'] = MagicMock()

import unittest
from unittest.mock import patch
import os
import shutil
import tempfile
import numpy as np

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Since we're mocking faiss, we need to ensure the mocks have the expected methods
import faiss
faiss.IndexFlatL2 = MagicMock()
faiss.write_index = MagicMock()
faiss.read_index = MagicMock()

from backend.indexing import create_index, load_index, save_index

class TestIndexing(unittest.TestCase):
    """Test cases for indexing functionality."""

    def setUp(self):
        from backend import database
        database.init_database()
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.temp_dir, "test_docs")
        os.makedirs(self.test_folder, exist_ok=True)
        
        # Create a dummy file
        with open(os.path.join(self.test_folder, "test.txt"), "w") as f:
            f.write("This is a test document.")

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.CharacterTextSplitter')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    def test_create_index(self, mock_extract_text, mock_get_embeddings, mock_splitter_cls):
        """Test creating an index from a folder."""
        # Mock text splitting
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1"]

        mock_extract_text.return_value = "Test content"
        
        # Mock embeddings
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model
        
        # Mock other dependencies
        with patch('backend.indexing.get_tags', return_value="test"),              patch('backend.indexing.perform_global_clustering', return_value={0: [0]}),              patch('backend.indexing.smart_summary', return_value="Summary"):

            res = create_index(
                self.test_folder,
                "openai",
                "fake_key"
            )
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            self.assertIsNotNone(index)
            self.assertEqual(len(docs), 1)
            self.assertEqual(len(tags), 1)

    @patch('backend.indexing.get_embeddings')
    def test_create_index_empty_folder(self, mock_embed):
        """Test creating index with empty folder."""
        empty_folder = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_folder, exist_ok=True)
        
        mock_embeddings_model = MagicMock()
        mock_embed.return_value = mock_embeddings_model
        
        res = create_index(empty_folder, "openai", "fake_key")
        index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res
        
        self.assertIsNone(index)
        self.assertEqual(len(docs), 0)

    @patch('json.load')
    @patch('backend.indexing.faiss.read_index')
    @patch('os.path.exists')
    def test_load_index_preserves_data(self, mock_exists, mock_read_index, mock_json_load):
        """Test loading index."""
        # Mock the index reading
        mock_faiss_index = MagicMock()
        mock_read_index.return_value = mock_faiss_index
        
        index_path = "fake_index.faiss"

        # Mock os.path.exists: True for main file
        mock_exists.side_effect = lambda path: True # Assume all exist for simplicity
        
        # Mock json loading
        # The function loads: _docs.json, _tags.json, _summ_index.json, _summ_docs.json, _cluster_map.json
        mock_json_load.side_effect = [
            [{"text": "Test document"}], # docs
            [["test", "tag"]], # tags
            None, # summ_index
            [], # summ_docs
            {} # cluster_map
        ]
        
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res
        
        # Verify the functions were called
        mock_read_index.assert_called_once_with(index_path)
        
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
        embeddings = np.array([[1.0, 2.0, 3.0]], dtype='float32')
        index.add(embeddings)
        
        docs = ["Document 1", "Document 2"]
        tags = [["tag1"], ["tag2"]]
        
        index_path = os.path.join(self.temp_dir, "index.faiss")
        save_index(index, docs, tags, index_path)
        
        self.assertTrue(os.path.exists(index_path))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_docs.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_tags.json")))


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
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32')
        original_index.add(embeddings)
        
        original_docs = [{"text": "Doc A"}, {"text": "Doc B"}]
        original_tags = [["alpha"], ["beta", "gamma"]]
        
        index_path = os.path.join(self.temp_dir, "test.faiss")
        save_index(original_index, original_docs, original_tags, index_path)
        
        # Mock load_index return
        mock_read_index.return_value = original_index

        # Load and verify
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res
        
        # We can't check ntotal on a mock object unless we set it.
        # But if we use the same mock object "original_index", it should have whatever we set on it.
        # However, faiss.IndexFlatL2 is a mock.
        # We can just check it's the same object.
        self.assertEqual(loaded_index, original_index)
        self.assertEqual(loaded_docs, original_docs)
        self.assertEqual(loaded_tags, original_tags)


if __name__ == '__main__':
    unittest.main()
