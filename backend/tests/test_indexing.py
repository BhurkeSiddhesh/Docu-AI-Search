import unittest
import os
import shutil
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from backend.indexing import create_index, save_index, load_index

class TestIndexing(unittest.TestCase):
    """Test cases for indexing functionality."""

    def setUp(self):
        # Create a temporary test folder with some files
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_folder = self.test_dir.name
        
        # Create dummy files
        with open(os.path.join(self.test_folder, "test1.txt"), "w") as f:
            f.write("This is a test document.")
            
        with open(os.path.join(self.test_folder, "test2.txt"), "w") as f:
            f.write("Another test document.")

        # Initialize test database
        from backend import database
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.original_db_path = database.DATABASE_PATH
        database.DATABASE_PATH = self.db_path
        database.init_database()

    def tearDown(self):
        # Cleanup
        self.test_dir.cleanup()
        from backend import database
        database.DATABASE_PATH = self.original_db_path
        os.close(self.db_fd)
        os.remove(self.db_path)

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.perform_global_clustering')
    @patch('backend.indexing.summarize')
    def test_create_index(self, mock_summarize, mock_clustering, mock_get_embeddings):
        """Test creating an index."""
        # Mock embeddings model
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_documents.return_value = np.random.rand(2, 384).astype('float32')
        mock_get_embeddings.return_value = mock_embedding_model
        
        # Mock clustering
        mock_clustering.return_value = {0: [0], 1: [1]}
        
        mock_summarize.return_value = "Summary"

        res = create_index(self.test_folder, "openai", "fake_api_key")

        self.assertIsNotNone(res)
        index, docs, tags, summ_index, summ_docs, cluster_map, bm25 = res

        self.assertIsNotNone(index)
        self.assertEqual(len(docs), 2)

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.perform_global_clustering')
    def test_create_index_empty_folder(self, mock_clustering, mock_get_embeddings):
        """Test creating an index with empty folder."""
        empty_folder = os.path.join(self.test_folder, "empty_folder")
        os.makedirs(empty_folder, exist_ok=True)
        
        res = create_index(empty_folder, "openai", "fake_api_key")
        
        self.assertIsNotNone(res)
        self.assertIsNone(res[0])

    def test_save_and_load_index(self):
        """Test saving and loading an index."""
        # Create a dummy index
        import faiss
        d = 64
        nb = 100
        index = faiss.IndexFlatL2(d)
        xb = np.random.random((nb, d)).astype('float32')
        index.add(xb)
        
        docs = ["doc" + str(i) for i in range(nb)]
        tags = ["tag" + str(i) for i in range(nb)]
        
        summ_index = faiss.IndexFlatL2(d)
        summ_index.add(xb[:10])
        summ_docs = ["summ" + str(i) for i in range(10)]
        cluster_map = {0: "Cluster 0"}
        
        # Create BM25 dummy
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([["test"]])

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test_index.faiss")

            save_index(index, docs, tags, index_path, summ_index, summ_docs, cluster_map, bm25)

            self.assertTrue(os.path.exists(index_path))

            loaded = load_index(index_path)

            self.assertIsNotNone(loaded)
            l_index, l_docs, l_tags, l_summ_index, l_summ_docs, l_cluster_map, l_bm25 = loaded

            self.assertEqual(l_index.ntotal, nb)
            self.assertEqual(len(l_docs), nb)
            self.assertEqual(len(l_tags), nb)
            self.assertEqual(len(l_summ_docs), 10)
            self.assertEqual(l_cluster_map, cluster_map)
            self.assertIsNotNone(l_bm25)


class TestIndexingMultipleFolders(unittest.TestCase):
    """Test cases for indexing multiple folders."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.folder1 = os.path.join(self.test_dir.name, "folder1")
        self.folder2 = os.path.join(self.test_dir.name, "folder2")
        os.makedirs(self.folder1)
        os.makedirs(self.folder2)

        with open(os.path.join(self.folder1, "doc1.txt"), "w") as f:
            f.write("Content of doc 1")
        with open(os.path.join(self.folder2, "doc2.txt"), "w") as f:
            f.write("Content of doc 2")

        # Initialize test database
        from backend import database
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.original_db_path = database.DATABASE_PATH
        database.DATABASE_PATH = self.db_path
        database.init_database()

    def tearDown(self):
        self.test_dir.cleanup()
        from backend import database
        database.DATABASE_PATH = self.original_db_path
        os.close(self.db_fd)
        os.remove(self.db_path)

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.perform_global_clustering')
    @patch('backend.indexing.summarize')
    def test_create_index_multiple_folders(self, mock_summarize, mock_clustering, mock_get_embeddings):
        """Test creating index from multiple folders."""
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_documents.return_value = np.random.rand(2, 384).astype('float32')
        mock_get_embeddings.return_value = mock_embedding_model

        mock_clustering.return_value = {0: [0], 1: [1]}
        mock_summarize.return_value = "Summary"

        res = create_index(
            [self.folder1, self.folder2],
            "openai",
            "fake_key"
        )

        self.assertIsNotNone(res)
        index, docs, tags, _, _, _, _ = res
        self.assertEqual(len(docs), 2)

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.perform_global_clustering')
    @patch('backend.indexing.summarize')
    def test_create_index_with_progress_callback(self, mock_summarize, mock_clustering, mock_get_embeddings):
        """Test progress callback during indexing."""
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_documents.return_value = np.random.rand(1, 384).astype('float32')
        mock_get_embeddings.return_value = mock_embedding_model
        
        mock_clustering.return_value = {0: [0]}
        mock_summarize.return_value = "Summary"

        progress_callback = MagicMock()
        
        create_index(self.folder1, "openai", "fake_key", progress_callback=progress_callback)
        
        self.assertTrue(progress_callback.called)

    def test_create_index_nonexistent_folder(self):
        """Test creating index with nonexistent folder."""
        res = create_index(
            ["/nonexistent/folder/path"],
            "openai",
            "fake_key"
        )
        self.assertIsNotNone(res)
        self.assertIsNone(res[0])

    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.perform_global_clustering')
    @patch('backend.indexing.summarize')
    def test_create_index_string_folder_path(self, mock_summarize, mock_clustering, mock_get_embeddings):
        """Test that string folder path is converted to list."""
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_documents.return_value = np.random.rand(1, 384).astype('float32')
        mock_get_embeddings.return_value = mock_embedding_model

        mock_clustering.return_value = {0: [0]}
        mock_summarize.return_value = "Summary"

        res = create_index(
            self.folder1, # Pass as string
            "openai",
            "fake_key"
        )
        
        self.assertIsNotNone(res)

if __name__ == '__main__':
    unittest.main()
