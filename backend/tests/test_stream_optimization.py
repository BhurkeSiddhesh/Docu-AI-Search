import unittest
from unittest.mock import patch, MagicMock
import sys
from fastapi.testclient import TestClient

class TestStreamOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Patch modules before importing api
        cls.module_patcher = patch.dict(sys.modules, {
            'backend.search': MagicMock(),
            'backend.llm_integration': MagicMock(),
            'backend.database': MagicMock()
        })
        cls.module_patcher.start()

        # Import app here safely
        from backend.api import app
        cls.app = app

    @classmethod
    def tearDownClass(cls):
        cls.module_patcher.stop()

    def setUp(self):
        self.client = TestClient(self.app)

    @patch('backend.api.index', 'dummy_index')
    @patch('backend.api.search')
    @patch('backend.api.stream_ai_answer')
    @patch('backend.api.load_config')
    def test_stream_answer_with_context_skips_search(self, mock_config, mock_stream, mock_search):
        # Setup mocks
        mock_config.return_value = MagicMock()

        def mock_generator(*args, **kwargs):
            yield "answer"
        mock_stream.side_effect = mock_generator

        # Call with context
        response = self.client.post("/api/stream-answer", json={
            "query": "test",
            "context": ["context1", "context2"]
        })

        # Verify response
        self.assertEqual(response.status_code, 200)

        # Verify search was NOT called
        mock_search.assert_not_called()

        # Verify stream_ai_answer WAS called with the context
        mock_stream.assert_called()
        args = mock_stream.call_args[0]
        context_text = args[0]
        self.assertIn("context1", context_text)
        self.assertIn("context2", context_text)

    @patch('backend.api.index', 'dummy_index')
    @patch('backend.api.search')
    @patch('backend.api.stream_ai_answer')
    @patch('backend.api.summarize')
    @patch('backend.api.load_config')
    @patch('backend.api.database')
    def test_stream_answer_without_context_calls_search(self, mock_db, mock_config, mock_summarize, mock_stream, mock_search):
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_search.return_value = ([{'document': 'doc1', 'faiss_idx': 1}], [])
        mock_summarize.return_value = "summary"
        mock_db.get_files_by_faiss_indices.return_value = {}

        def mock_generator(*args, **kwargs):
            yield "answer"
        mock_stream.side_effect = mock_generator

        # Call without context
        response = self.client.post("/api/stream-answer", json={
            "query": "test"
        })

        # Verify response
        self.assertEqual(response.status_code, 200)

        # Verify search WAS called
        mock_search.assert_called()

        # Verify stream_ai_answer WAS called
        mock_stream.assert_called()

if __name__ == '__main__':
    unittest.main()
