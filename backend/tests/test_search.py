import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()

from backend.search import search

class TestSearch(unittest.TestCase):
    def setUp(self):
        # Mock embeddings model
        self.mock_embeddings_model = MagicMock()
        self.mock_embeddings_model.embed_query.return_value = np.zeros(384)

        # Mock ThreadPoolExecutor
        self.patcher = patch('concurrent.futures.ThreadPoolExecutor')
        self.mock_executor_cls = self.patcher.start()
        self.mock_executor = self.mock_executor_cls.return_value
        self.mock_executor.__enter__.return_value = self.mock_executor

        # Setup common mock future behavior
        self.mock_future = MagicMock()
        self.mock_executor.submit.return_value = self.mock_future

    def tearDown(self):
        self.patcher.stop()

    def test_search_basic(self):
        """Test basic search functionality."""
        query = "test query"
        index = MagicMock()
        docs = ["doc1", "doc2"]
        tags = ["tag1", "tag2"]

        # Mock faiss index search result via executor
        # distances, indices
        # search() function logic likely processes this.
        # It typically gets 2D arrays.
        dists = np.array([[0.1, 0.2]])
        idxs = np.array([[0, 1]])
        self.mock_future.result.return_value = (dists, idxs)

        # Assuming search signature is search(query, index, docs, tags, embedding_model)
        # and it returns (results, context_string)
        results, context = search(query, index, docs, tags, self.mock_embeddings_model)

        self.assertEqual(len(results), 2)
        self.assertIn("doc1", context)

    def test_search_empty_index(self):
        """Test search with an empty index."""
        query = "test query"
        index = MagicMock()
        docs = []
        tags = []
        
        # Empty result
        self.mock_future.result.return_value = (np.empty((1, 0)), np.empty((1, 0)))

        results, context = search(query, index, docs, tags, self.mock_embeddings_model)
        
        self.assertEqual(len(results), 0)
        self.assertEqual(context, "")

if __name__ == '__main__':
    unittest.main()
