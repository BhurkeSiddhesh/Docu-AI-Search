"""
Test Indexing Module

Tests for the indexing functionality (FAISS, text extraction).
"""

import unittest
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
import sys
import concurrent.futures

# Mock modules that might be missing or hard to install
sys.modules['faiss'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.cluster'] = MagicMock()

# Now we can import
from backend.indexing import create_index

class TestIndexing(unittest.TestCase):
    """Test cases for indexing."""

    def setUp(self):
        """Set up temporary directory for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.test_dir, "test_files")
        os.makedirs(self.test_folder)
        
        # Create dummy files
        with open(os.path.join(self.test_folder, "test1.txt"), "w") as f:
            f.write("This is a test document.")

        with open(os.path.join(self.test_folder, "test2.txt"), "w") as f:
            f.write("Another test document for indexing.")

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.safe_extract_text')
    @patch('backend.indexing.perform_global_clustering')
    @patch('backend.indexing.summarize')
    @patch('backend.indexing.database')
    @patch('concurrent.futures.ProcessPoolExecutor')
    def test_create_index(self, mock_executor_cls, mock_db, mock_summarize, mock_cluster, mock_extract, mock_get_embeddings):
        """Test creating an index."""
        # Mock ProcessPoolExecutor to run synchronously or mock return values
        # Since we can't pickle MagicMock, we avoid actual multiprocessing
        mock_executor = mock_executor_cls.return_value
        mock_executor.__enter__.return_value = mock_executor

        # When submit is called, return a Future with a predictable result
        def side_effect_submit(func, *args, **kwargs):
            future = concurrent.futures.Future()
            # args[0] is filepath
            future.set_result((args[0], f"Content of {os.path.basename(args[0])}"))
            return future
            
        mock_executor.submit.side_effect = side_effect_submit

        # Mock embeddings
        mock_embed_model = MagicMock()
        mock_embed_model.embed_documents.return_value = [[0.1] * 384] * 2
        mock_get_embeddings.return_value = mock_embed_model
        
        # Mock clustering
        mock_cluster.return_value = {0: [0]} # Cluster ID 0 contains chunk index 0
        
        # Mock summarization
        mock_summarize.return_value = "Cluster Summary"

        # Mock FAISS index
        mock_index = MagicMock()
        mock_index.ntotal = 2
        
        with patch('faiss.IndexFlatL2', return_value=mock_index):
            index, docs, tags, summ_index, summ_docs, cluster_map, bm25 = create_index(
                [self.test_folder], "openai", "fake_api_key"
            )

            self.assertIsNotNone(index)
            # 2 files, assuming 1 chunk per file (short content)
            self.assertEqual(len(docs), 2)
            self.assertEqual(len(tags), 2)

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.database')
    def test_create_index_empty_folder(self, mock_db, mock_get_embeddings):
        """Test indexing an empty folder."""
        empty_folder = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_folder)
        
        index, docs, tags, summ_index, summ_docs, cluster_map, bm25 = create_index(
            [empty_folder], "openai", "fake_api_key"
        )
        
        self.assertIsNone(index)
        self.assertIsNone(docs)


class TestIndexingMultipleFolders(unittest.TestCase):
    """Test cases for indexing multiple folders."""

    def setUp(self):
        """Set up temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.folder1 = os.path.join(self.test_dir, "folder1")
        self.folder2 = os.path.join(self.test_dir, "folder2")
        os.makedirs(self.folder1)
        os.makedirs(self.folder2)
        
        with open(os.path.join(self.folder1, "f1.txt"), "w") as f:
            f.write("File in folder 1")
        
        with open(os.path.join(self.folder2, "f2.txt"), "w") as f:
            f.write("File in folder 2")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.safe_extract_text')
    @patch('backend.indexing.perform_global_clustering')
    @patch('backend.indexing.summarize')
    @patch('backend.indexing.database')
    @patch('concurrent.futures.ProcessPoolExecutor')
    def test_create_index_multiple_folders(self, mock_executor_cls, mock_db, mock_summarize, mock_cluster, mock_extract, mock_embed):
        """Test indexing multiple folders."""
        # Mock Executor
        mock_executor = mock_executor_cls.return_value
        mock_executor.__enter__.return_value = mock_executor

        def side_effect_submit(func, *args, **kwargs):
            future = concurrent.futures.Future()
            future.set_result((args[0], "Content"))
            return future
        mock_executor.submit.side_effect = side_effect_submit

        # Mock Embeddings
        mock_embed_model = MagicMock()
        mock_embed_model.embed_documents.return_value = [[0.1] * 384] * 2
        mock_embed.return_value = mock_embed_model

        # Mock clustering
        mock_cluster.return_value = {0: [0]}
        mock_summarize.return_value = "Cluster Summary"

        mock_index = MagicMock()
        mock_index.ntotal = 2

        with patch('faiss.IndexFlatL2', return_value=mock_index):
            res = create_index(
                [self.folder1, self.folder2], 
                "openai", "fake_key"
            )
            
            self.assertIsNotNone(res[0]) # index
            self.assertEqual(len(res[1]), 2) # docs

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.database')
    @patch('concurrent.futures.ProcessPoolExecutor')
    def test_create_index_with_progress_callback(self, mock_executor_cls, mock_db, mock_embed):
        """Test indexing with a progress callback."""
        mock_executor = mock_executor_cls.return_value
        mock_executor.__enter__.return_value = mock_executor
        def side_effect_submit(func, *args, **kwargs):
            future = concurrent.futures.Future()
            future.set_result((args[0], "Content"))
            return future
        mock_executor.submit.side_effect = side_effect_submit

        mock_embed_model = MagicMock()
        mock_embed_model.embed_documents.return_value = [[0.1] * 384]
        mock_embed.return_value = mock_embed_model

        progress_callback = MagicMock()

        with patch('faiss.IndexFlatL2'), \
             patch('backend.indexing.perform_global_clustering', return_value={}), \
             patch('backend.indexing.summarize', return_value="Summary"):

            create_index(self.folder1, "openai", "fake_key", progress_callback=progress_callback)
            
            # Should have been called at least once
            progress_callback.assert_called()

    @patch('backend.indexing.database')
    def test_create_index_nonexistent_folder(self, mock_db):
        """Test indexing a folder that doesn't exist."""
        res = create_index(
            ["/nonexistent/folder/path"],
            "openai", "fake_key"
        )

        self.assertIsNone(res[0])

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.database')
    @patch('concurrent.futures.ProcessPoolExecutor')
    def test_create_index_string_folder_path(self, mock_executor_cls, mock_db, mock_embed):
        """Test passing a single string instead of list (backward compatibility)."""
        mock_executor = mock_executor_cls.return_value
        mock_executor.__enter__.return_value = mock_executor
        def side_effect_submit(func, *args, **kwargs):
            future = concurrent.futures.Future()
            future.set_result((args[0], "Content"))
            return future
        mock_executor.submit.side_effect = side_effect_submit

        mock_embed_model = MagicMock()
        mock_embed_model.embed_documents.return_value = [[0.1] * 384]
        mock_embed.return_value = mock_embed_model

        with patch('faiss.IndexFlatL2'), \
             patch('backend.indexing.perform_global_clustering', return_value={}), \
             patch('backend.indexing.summarize', return_value="Summary"):

            res = create_index(
                self.folder1, # Passing string directly
                "openai", "fake_key"
            )
            
            self.assertIsNotNone(res[0])

if __name__ == '__main__':
    unittest.main()
