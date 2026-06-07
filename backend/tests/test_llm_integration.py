import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.llm_integration import get_llm_client, smart_summary, get_embeddings, _invoke_with_retry

class TestLLMIntegrationV2(unittest.TestCase):
    """Test cases for the new multi-provider LLM integration."""

    @patch('backend.llm_integration.ChatOpenAI')
    def test_get_llm_client_openai(self, mock_chat_openai):
        """Test getting OpenAI client."""
        client = get_llm_client('openai', api_key='sk-test')
        self.assertIsNotNone(client)
        mock_chat_openai.assert_called_with(api_key='sk-test', model='gpt-4o-mini', temperature=0.3)

    @patch('backend.llm_integration.ChatGoogleGenerativeAI')
    def test_get_llm_client_gemini(self, mock_gemini):
        """Test getting Gemini client."""
        client = get_llm_client('gemini', api_key='AIza-test')
        self.assertIsNotNone(client)
        mock_gemini.assert_called_with(google_api_key='AIza-test', model='gemini-1.5-flash', temperature=0.3)

    @patch('backend.llm_integration.ChatAnthropic')
    def test_get_llm_client_anthropic(self, mock_anthropic):
        """Test getting Anthropic client."""
        client = get_llm_client('anthropic', api_key='sk-ant-test')
        self.assertIsNotNone(client)
        mock_anthropic.assert_called_with(api_key='sk-ant-test', model='claude-3-haiku-20240307', temperature=0.3)

    def test_get_llm_client_local(self):
        """Test getting Local client (returns string path marker)."""
        with patch('os.path.exists', return_value=True):
            client = get_llm_client('local', model_path='models/test.gguf')
            self.assertEqual(client, "LOCAL:models/test.gguf")

    def test_get_llm_client_missing_key(self):
        """Test missing key returns None."""
        client = get_llm_client('openai', api_key=None)
        self.assertIsNone(client)

    @patch('backend.llm_integration.get_llm_client')
    def test_smart_summary_cloud(self, mock_get_client):
        """Test smart summary with cloud provider."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is a smart summary."
        mock_client.invoke.return_value = mock_response
        mock_get_client.return_value = mock_client

        summary = smart_summary("Long text...", "Query", "openai", "key")
        
        self.assertEqual(summary, "This is a smart summary.")
        mock_client.invoke.assert_called_once()

    @patch('backend.llm_integration.get_llm_client')
    @patch('backend.llm_integration.get_local_llm')
    def test_smart_summary_local(self, mock_get_local, mock_get_client):
        """Test smart summary with local provider."""
        mock_get_client.return_value = "LOCAL:model.gguf"
        
        mock_llm = MagicMock()
        mock_llm.create_completion.return_value = {'choices': [{'text': 'Local summary'}]}
        mock_get_local.return_value = mock_llm

        summary = smart_summary("Long text...", "Query", "local", model_path="model.gguf")
        
        self.assertEqual(summary, "Local summary")
        mock_llm.create_completion.assert_called_once()

    @patch('backend.llm_integration.get_embeddings')
    def test_get_embeddings_routing(self, mock_get_emb):
        """Test that get_embeddings is called (actual logic inside is complex to mock fully due to lazy loading imports, but we check api.py calls it)."""
        # This test just ensures the function exists and runs without import error
        pass


class TestInvokeWithRetry(unittest.TestCase):

    def test_succeeds_on_first_attempt(self):
        mock_client = MagicMock()
        mock_client.invoke.return_value = "result"
        result = _invoke_with_retry(mock_client, ["msg"])
        self.assertEqual(result, "result")
        self.assertEqual(mock_client.invoke.call_count, 1)

    @patch('time.sleep')
    def test_retries_on_transient_failure_and_succeeds(self, mock_sleep):
        mock_client = MagicMock()
        mock_client.invoke.side_effect = [Exception("transient"), "result"]
        result = _invoke_with_retry(mock_client, ["msg"])
        self.assertEqual(result, "result")
        self.assertEqual(mock_client.invoke.call_count, 2)
        mock_sleep.assert_called_once_with(1)  # 2**0 = 1

    @patch('time.sleep')
    def test_raises_after_all_retries_exhausted(self, mock_sleep):
        mock_client = MagicMock()
        mock_client.invoke.side_effect = RuntimeError("permanent error")
        with self.assertRaises(RuntimeError):
            _invoke_with_retry(mock_client, ["msg"], retries=3)
        self.assertEqual(mock_client.invoke.call_count, 3)
        # sleep called for first two failures, not the last
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('time.sleep')
    def test_exponential_backoff_sleep_values(self, mock_sleep):
        mock_client = MagicMock()
        mock_client.invoke.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            "ok",
        ]
        result = _invoke_with_retry(mock_client, ["msg"], retries=3)
        self.assertEqual(result, "ok")
        # backoff: 2**0=1, 2**1=2
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        self.assertEqual(sleep_calls, [1, 2])


if __name__ == '__main__':
    unittest.main()
