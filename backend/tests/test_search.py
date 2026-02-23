import unittest
import sys
from unittest.mock import MagicMock

# Mock heavy/missing dependencies at the module level
sys.modules['faiss'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

# Now import the module under test
from backend.search import search

class TestSearch(unittest.TestCase):
    """Test cases for search module"""

    def test_search_basic(self):
        """Test basic search functionality."""
        # Create mock embeddings model
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_query.return_value = [0.5, 0.5, 0.5]
        
        # Mock index behavior
        mock_index = MagicMock()
        # Mock search method: returns (distances, indices)
        # Assuming search returns 3 results
        mock_index.search.return_value = ([[0.1, 0.2, 0.3]], [[0, 1, 2]])
        
        # Documents and tags
        docs = ["Document 1", "Document 2", "Document 3"]
        tags = ["tag1", "tag2", "tag3"]
        
        # Perform search
        query = "test query"
        results, context = search(query, mock_index, docs, tags, mock_embeddings_model)
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results, list)
        self.assertIsInstance(context, list)
        
        # Check structure
        for result in results:
            self.assertIn("document", result)
            self.assertIn("distance", result)
            self.assertIn("tags", result)
    
    def test_search_with_more_documents_than_k(self):
        """Test search when there are more documents than k."""
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_query.return_value = [0.5] * 3

        mock_index = MagicMock()
        # Return 6 results
        mock_index.search.return_value = ([[0.1]*6], [[0,1,2,3,4,5]])
        
        docs = [f"Document {i}" for i in range(6)]
        tags = [f"tag{i}" for i in range(6)]
        
        query = "test query"
        results, context = search(query, mock_index, docs, tags, mock_embeddings_model)
        
        self.assertEqual(len(results), 6)
    
    def test_search_with_insufficient_documents(self):
        """Test search when there are fewer documents than requested."""
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_query.return_value = [0.5] * 3
        
        mock_index = MagicMock()
        # Return 1 result
        mock_index.search.return_value = ([[0.1]], [[0]])
        
        docs = ["Single Document"]
        tags = ["single_tag"]
        
        query = "test query"
        results, context = search(query, mock_index, docs, tags, mock_embeddings_model)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document"], "Single Document")
    
    def test_search_empty_index(self):
        """Test search with an empty index."""
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_query.return_value = [0.5] * 3
        
        mock_index = MagicMock()
        # Empty search results
        mock_index.search.return_value = ([[]], [[]])
        
        docs = []
        tags = []
        
        query = "test query"
        results, context = search(query, mock_index, docs, tags, mock_embeddings_model)
        
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()
