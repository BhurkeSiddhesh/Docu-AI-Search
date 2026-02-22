import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.api import app
import sys
import io

class TestSecurityLogging(unittest.TestCase):
    """
    Security tests to ensure sensitive data (PII) is not logged to stdout/stderr.
    """

    def setUp(self):
        self.client = TestClient(app)
        self.sensitive_query = "SUPER_SECRET_PASSWORD_123"

    @patch('backend.database.add_search_history')
    @patch('backend.api.search')
    @patch('backend.api.summarize')
    @patch('backend.api.load_config')
    @patch('backend.api.get_embeddings')
    @patch('backend.llm_integration.get_llm_client')
    def test_search_endpoint_redacts_query(self, mock_get_client, mock_get_embeddings,
                                          mock_load_config, mock_summarize, mock_search, mock_add_history):
        """
        Test that the search endpoint and underlying functions do NOT print the raw query.
        """
        mock_config = MagicMock()
        mock_config.get.return_value = 'local'
        mock_load_config.return_value = mock_config

        mock_search.return_value = (
            [{'document': 'content', 'tags': ['tag1'], 'faiss_idx': 0, 'file_path': 'test.txt'}],
            ['context snippet']
        )
        mock_summarize.return_value = "Summary"

        with patch('backend.api.index', MagicMock()),              patch('backend.api.docs', []),              patch('backend.api.tags', []),              patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:

            response = self.client.post("/api/search", json={
                "query": self.sensitive_query
            })

            logs = mock_stdout.getvalue()
            self.assertEqual(response.status_code, 200)

            if self.sensitive_query in logs:
                print(f"\nSECURITY FAILURE: Found sensitive query in logs:\n{logs}")

            self.assertNotIn(self.sensitive_query, logs,
                             f"Sensitive query '{self.sensitive_query}' leaked in logs!")

    @patch('backend.llm_integration.get_llm_client')
    def test_llm_integration_logging(self, mock_get_client):
        """Test specific functions in llm_integration directly for logging leaks."""
        from backend.llm_integration import cached_generate_ai_answer, cached_smart_summary

        # 1. Test Cache Hit Log
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch('backend.database.get_cached_response', return_value="Cached Answer"):
                 cached_generate_ai_answer("context", self.sensitive_query, "local")

            logs = mock_stdout.getvalue()
            self.assertNotIn(self.sensitive_query, logs, "cached_generate_ai_answer leaked query in logs (cache hit)")

        # 2. Test Smart Summary Log
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
             # Force cache miss to trigger generation log
             with patch('backend.database.get_cached_response', return_value=None):
                 mock_get_client.return_value = MagicMock()
                 cached_smart_summary("text", self.sensitive_query, "local")

             logs = mock_stdout.getvalue()
             self.assertNotIn(self.sensitive_query, logs, "smart_summary leaked query in logs")

    def test_search_function_logging(self):
        """Test the search function in backend/search.py specifically."""
        from backend.search import search

        mock_index = MagicMock()
        mock_index.search.return_value = ([[1.0]], [[0]])

        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = [0.5]

        docs = [{'text': 'doc', 'filepath': 'path'}]
        tags = ['tag']

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            search(
                query=self.sensitive_query,
                index=mock_index,
                docs=docs,
                tags=tags,
                embeddings_model=MagicMock(),
                bm25=mock_bm25
            )

            logs = mock_stdout.getvalue()
            self.assertNotIn(self.sensitive_query, logs, "search() leaked query in logs")

if __name__ == '__main__':
    unittest.main()
