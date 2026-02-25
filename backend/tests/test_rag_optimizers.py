import unittest
from unittest.mock import patch, MagicMock

from backend.rag_optimizers import rewrite_query, rerank_results, _QUERY_REWRITE_CACHE

class TestRagOptimizers(unittest.TestCase):
    def setUp(self):
        # Clear cache before each test
        _QUERY_REWRITE_CACHE.clear()

    @patch('backend.rag_optimizers.generate_ai_answer')
    def test_rewrite_query_uses_llm_and_caches(self, mock_generate):
        mock_generate.return_value = "work experience london"
        
        # Call 1 (should hit LLM)
        query = "What did Siddhesh do in london?"
        result1 = rewrite_query(query, "openai", "test-key", "")
        self.assertEqual(result1, "work experience london")
        mock_generate.assert_called_once()
        
        # Call 2 (should hit cache)
        result2 = rewrite_query(query, "openai", "test-key", "")
        self.assertEqual(result2, "work experience london")
        self.assertEqual(mock_generate.call_count, 1) # Still 1

    @patch('backend.rag_optimizers.generate_ai_answer')
    def test_rewrite_query_fallback_on_error(self, mock_generate):
        mock_generate.side_effect = Exception("LLM Error")
        
        query = "What did Siddhesh do in london?"
        result = rewrite_query(query, "openai", "test-key", "")
        self.assertEqual(result, query) # Should return original query

    @patch('sentence_transformers.CrossEncoder')
    def test_rerank_results(self, mock_cross_encoder):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1, 0.5] # Assuming three pairs
        mock_cross_encoder.return_value = mock_model
        
        query = "test"
        chunks = [
            {'document': 'Doc A', 'id': 1},
            {'document': 'Doc B', 'id': 2},
            {'document': 'Doc C', 'id': 3}
        ]
        
        reranked = rerank_results(query, chunks, "mock-model")
        
        print(f"Reranked Output: {[(c['id'], c.get('rerank_score')) for c in reranked]}")
        
        # Check if they are sorted by score descending (1 -> 3 -> 2)
        self.assertEqual(reranked[0]['id'], 1)  # Score 0.9
        self.assertEqual(reranked[1]['id'], 3)  # Score 0.5
        self.assertEqual(reranked[2]['id'], 2)  # Score 0.1
        
        # Check that original fields are preserved
        self.assertEqual(reranked[0]['document'], 'Doc A')
        
    def test_rerank_results_empty(self):
        result = rerank_results("query", [], "test-model")
        self.assertEqual(len(result), 0)

if __name__ == '__main__':
    unittest.main()
