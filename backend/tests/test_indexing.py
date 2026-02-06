"""
Test Indexing Module

Tests for the indexing.py module, including FAISS index creation,
document processing, and RAPTOR clustering integration.
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from backend.indexing import create_index, save_index, load_index

class TestIndexing(unittest.TestCase):
    """Test indexing functionality."""

    def setUp(self):
        """Set up test directory and files."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.temp_dir, "test_docs")
        os.makedirs(self.test_folder, exist_ok=True)
        
        # Create a dummy test file
        with open(os.path.join(self.test_folder, "test.txt"), "w") as f:
            f.write("This is a test document for indexing.")

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.database')
    def test_create_index(self, mock_database, mock_extract_text, mock_get_embeddings):
        """Test creating an index from a folder."""
        # Mock dependencies
        mock_extract_text.return_value = "This is a test document."
        
        mock_embeddings_model = MagicMock()
        # Return dummy embeddings (list of lists)
        mock_embeddings_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embeddings.return_value = mock_embeddings_model
        
        # Mock internal functions to avoid complex logic
        with patch('backend.indexing.get_tags', return_value="tag1, tag2"),              patch('backend.indexing.perform_global_clustering', return_value={0: [0]}),              patch('backend.indexing.smart_summary', return_value="Summary"):

            res = create_index(self.test_folder, "openai", "fake_api_key")

            # Unpack results
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res
            
            # Verify index creation
            self.assertIsNotNone(index)
            self.assertEqual(index.ntotal, 1)
            self.assertEqual(len(docs), 1)

            # Verify database calls
            mock_database.clear_all_files.assert_called()
            mock_database.add_file.assert_called()

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.database')
    def test_create_index_empty_folder(self, mock_database, mock_extract_text, mock_get_embeddings):
        """Test creating an index with empty folder."""
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
    
    def test_save_and_load_index(self):
        """Test saving and loading an index."""
        # Create a mock FAISS index and documents
        import faiss
        dimension = 3
        index = faiss.IndexFlatL2(dimension)
        embeddings = np.array([[1.0, 2.0, 3.0]], dtype='float32')
        index.add(embeddings)
        
        docs = ["Test document"]
        tags = [["test", "tag"]]
        
        # Save the index
        index_path = os.path.join(self.temp_dir, "test_index.faiss")
        save_index(index, docs, tags, index_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(index_path))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_index_docs.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_index_tags.pkl")))
        
        # Load the index
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res
        
        # Verify the loaded data matches the original
        self.assertIsNotNone(loaded_index)
        self.assertEqual(loaded_docs, docs)
        self.assertEqual(loaded_tags, tags)
    
    @patch('faiss.read_index')
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_load_index(self, mock_pickle_load, mock_open, mock_exists, mock_read_index):
        """Test loading an index."""
        # Mock the index reading
        mock_faiss_index = MagicMock()
        mock_read_index.return_value = mock_faiss_index
        
        # Mock os.path.exists: True for main file, False for others
        mock_exists.side_effect = lambda path: path == index_path
        
        # Mock pickle loading
        mock_pickle_load.side_effect = [
            ["Test document"],
            [["test", "tag"]]
        ]
        
        index_path = "fake_index.faiss"
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res
        
        # Verify the functions were called
        mock_read_index.assert_called_once_with(index_path)
        self.assertEqual(mock_pickle_load.call_count, 2)
        
        # Verify the results
        self.assertEqual(loaded_index, mock_faiss_index)
        self.assertEqual(loaded_docs, ["Test document"])
        self.assertEqual(loaded_tags, [["test", "tag"]])

if __name__ == '__main__':
    unittest.main()


class TestIndexingMultipleFolders(unittest.TestCase):
    """Test indexing with multiple folders."""

    def setUp(self):
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

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.database')
    def test_create_index_multiple_folders(self, mock_database, mock_extract_text, mock_get_embeddings):
        """Test creating index from multiple folders."""
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


    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.database')
    def test_create_index_with_progress_callback(self, mock_database, mock_extract_text, mock_get_embeddings):
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
            
            # Verify progress was called
            self.assertGreater(len(progress_calls), 0)

    @patch('backend.indexing.database')
    def test_create_index_nonexistent_folder(self, mock_database):
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

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.database')
    def test_create_index_string_folder_path(self, mock_database, mock_extract_text, mock_get_embeddings):
        """Test that string folder path is converted to list."""
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
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_save_index_creates_all_files(self):
        """Test that save_index creates .faiss, _docs.pkl, and _tags.pkl files."""
        import faiss
        
        index = faiss.IndexFlatL2(3)
        embeddings = np.array([[1.0, 2.0, 3.0]], dtype='float32')
        index.add(embeddings)
        
        docs = ["Document 1", "Document 2"]
        tags = [["tag1"], ["tag2"]]
        
        index_path = os.path.join(self.temp_dir, "index.faiss")
        save_index(index, docs, tags, index_path)
        
        self.assertTrue(os.path.exists(index_path))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_docs.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "index_tags.pkl")))


class TestLoadIndex(unittest.TestCase):
    """Dedicated tests for load_index function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_load_index_preserves_data(self):
        """Test that load_index correctly restores saved data."""
        import faiss
        
        # Create and save
        original_index = faiss.IndexFlatL2(3)
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32')
        original_index.add(embeddings)
        
        original_docs = ["Doc A", "Doc B"]
        original_tags = [["alpha"], ["beta", "gamma"]]
        
        index_path = os.path.join(self.temp_dir, "test.faiss")
        save_index(original_index, original_docs, original_tags, index_path)
        
        # Load and verify
        res = load_index(index_path)
        loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res
        
        self.assertEqual(loaded_index.ntotal, 2)
        self.assertEqual(loaded_docs, original_docs)
        self.assertEqual(loaded_tags, original_tags)
