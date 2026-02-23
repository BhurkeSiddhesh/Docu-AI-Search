import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock dependencies BEFORE import
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()
sys.modules['numpy'] = MagicMock()

import numpy as np

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.search import search

class TestSearch(unittest.TestCase):
    def setUp(self):
        # Mock embeddings model
        self.mock_embeddings_model = MagicMock()
        # Mock embed_query to return a valid numpy array-like structure if needed
        # But since we mocked numpy, np.array will be a Mock.

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
        docs = [{"text": "doc1", "filepath": "path1"}, {"text": "doc2", "filepath": "path2"}]
        tags = ["tag1", "tag2"]

        # Since numpy is mocked, np.array(...) returns a MagicMock.
        # search.py does .
        # And .

        # We need to ensure  is iterable and yields 0, 1.
        # If  is mocked,  is mocked.
        # We can set side_effect or return_value.

        mock_dists = MagicMock()
        mock_idxs = MagicMock()

        # idxs[0] should be [0, 1]
        mock_idxs.__getitem__.return_value = [0, 1]

        # dists[0] should be [0.1, 0.2]
        mock_dists.__getitem__.return_value = [0.1, 0.2]

        self.mock_future.result.return_value = (mock_dists, mock_idxs)

        results, context = search(query, index, docs, tags, self.mock_embeddings_model)

        self.assertEqual(len(results), 2)
        # Context is a list of strings
        self.assertEqual(context[0], "doc1")
        self.assertEqual(context[1], "doc2")

    def test_search_empty_index(self):
        """Test search with an empty index."""
        query = "test query"
        index = MagicMock()
        docs = []
        tags = []
        
        mock_dists = MagicMock()
        mock_idxs = MagicMock()

        # idxs[0] should be [-1]
        mock_idxs.__getitem__.return_value = [-1]

        self.mock_future.result.return_value = (mock_dists, mock_idxs)

        results, context = search(query, index, docs, tags, self.mock_embeddings_model)
        
        self.assertEqual(len(results), 0)
        self.assertEqual(context, [])

if __name__ == '__main__':
    unittest.main()
