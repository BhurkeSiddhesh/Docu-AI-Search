"""
backend/tests/test_providers.py
-------------------------------
Unit tests for the LLM provider abstraction (backend/providers.py).

All HTTP calls are mocked — no real Ollama or LM Studio required.
"""

import json
import unittest
from unittest.mock import MagicMock, patch, Mock

from backend.providers import (
    OllamaProvider,
    OpenAICompatibleProvider,
    get_provider,
    clear_provider_cache,
    PROVIDER_OLLAMA,
    PROVIDER_LMSTUDIO,
)


class TestOllamaProvider(unittest.TestCase):
    """Tests for OllamaProvider."""

    def setUp(self):
        clear_provider_cache()
        self.provider = OllamaProvider(base_url="http://localhost:11434", model="llama3")

    # -- generate --

    @patch("backend.providers.requests.post")
    def test_generate_returns_response(self, mock_post):
        """generate() extracts the 'response' field from Ollama JSON."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "  Hello, world!  ", "done": True}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = self.provider.generate("Say hello")
        self.assertEqual(result, "Hello, world!")
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        self.assertEqual(payload["model"], "llama3")
        self.assertFalse(payload["stream"])

    @patch("backend.providers.requests.post")
    def test_generate_with_system_prompt(self, mock_post):
        """generate() passes system prompt to Ollama."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok", "done": True}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        self.provider.generate("Test", system_prompt="Be concise")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertEqual(payload["system"], "Be concise")

    @patch("backend.providers.requests.post")
    def test_generate_connection_error(self, mock_post):
        """generate() raises ConnectionError with helpful message when Ollama is down."""
        import requests as real_requests
        mock_post.side_effect = real_requests.ConnectionError("refused")

        with self.assertRaises(ConnectionError) as ctx:
            self.provider.generate("Hello")
        self.assertIn("ollama serve", str(ctx.exception).lower())

    # -- stream --

    @patch("backend.providers.requests.post")
    def test_stream_yields_tokens(self, mock_post):
        """stream() yields individual tokens from NDJSON lines."""
        lines = [
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": " world", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_post.return_value = mock_resp

        tokens = list(self.provider.stream("Say hello"))
        self.assertEqual(tokens, ["Hello", " world"])

    # -- list_models --

    @patch("backend.providers.requests.get")
    def test_list_models_returns_parsed_list(self, mock_get):
        """list_models() parses Ollama /api/tags response."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3:latest", "size": 4000000000, "modified_at": "2024-01-01"},
                {"name": "mistral:latest", "size": 3000000000, "modified_at": "2024-01-02"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        models = self.provider.list_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["id"], "llama3:latest")
        self.assertEqual(models[1]["name"], "mistral:latest")

    @patch("backend.providers.requests.get")
    def test_list_models_connection_error(self, mock_get):
        """list_models() raises ConnectionError when Ollama is offline."""
        import requests as real_requests
        mock_get.side_effect = real_requests.ConnectionError("refused")

        with self.assertRaises(ConnectionError):
            self.provider.list_models()

    # -- health_check --

    @patch("backend.providers.requests.get")
    def test_health_check_ok(self, mock_get):
        """health_check() returns ok when Ollama responds."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "m1"}, {"name": "m2"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = self.provider.health_check()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["models_available"], 2)

    @patch("backend.providers.requests.get")
    def test_health_check_error(self, mock_get):
        """health_check() returns error when Ollama is unreachable."""
        import requests as real_requests
        mock_get.side_effect = real_requests.ConnectionError("refused")

        result = self.provider.health_check()
        self.assertEqual(result["status"], "error")
        self.assertIn("ollama", result["message"].lower())


class TestOpenAICompatibleProvider(unittest.TestCase):
    """Tests for OpenAICompatibleProvider (LM Studio, vLLM, etc.)."""

    def setUp(self):
        clear_provider_cache()
        self.provider = OpenAICompatibleProvider(
            base_url="http://localhost:1234/v1",
            model="local-model",
        )

    # -- generate --

    @patch("backend.providers.requests.post")
    def test_generate_returns_content(self, mock_post):
        """generate() extracts message content from OpenAI response format."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "  Generated text  "}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = self.provider.generate("Test prompt")
        self.assertEqual(result, "Generated text")

    @patch("backend.providers.requests.post")
    def test_generate_connection_error(self, mock_post):
        """generate() raises ConnectionError when LM Studio is down."""
        import requests as real_requests
        mock_post.side_effect = real_requests.ConnectionError("refused")

        with self.assertRaises(ConnectionError) as ctx:
            self.provider.generate("Hello")
        self.assertIn("lm studio", str(ctx.exception).lower())

    # -- stream --

    @patch("backend.providers.requests.post")
    def test_stream_yields_tokens_from_sse(self, mock_post):
        """stream() parses SSE format (data: {...}) correctly."""
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            'data: [DONE]',
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_post.return_value = mock_resp

        tokens = list(self.provider.stream("Test"))
        self.assertEqual(tokens, ["Hello", " world"])

    # -- list_models --

    @patch("backend.providers.requests.get")
    def test_list_models_returns_parsed_list(self, mock_get):
        """list_models() parses OpenAI /v1/models response."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"id": "llama-3-8b", "owned_by": "local"},
                {"id": "mistral-7b", "owned_by": "local"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        models = self.provider.list_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["id"], "llama-3-8b")

    # -- health_check --

    @patch("backend.providers.requests.get")
    def test_health_check_ok(self, mock_get):
        """health_check() returns ok when server responds."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "m1"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = self.provider.health_check()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["models_available"], 1)


class TestProviderFactory(unittest.TestCase):
    """Tests for the get_provider() factory function."""

    def setUp(self):
        clear_provider_cache()

    def test_ollama_provider(self):
        """get_provider('ollama') returns OllamaProvider."""
        provider = get_provider(PROVIDER_OLLAMA, {"model": "llama3"})
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.model, "llama3")

    def test_lmstudio_provider(self):
        """get_provider('lmstudio') returns OpenAICompatibleProvider."""
        provider = get_provider(PROVIDER_LMSTUDIO, {"model": "local-model"})
        self.assertIsInstance(provider, OpenAICompatibleProvider)

    def test_custom_base_url(self):
        """get_provider() respects custom base_url."""
        provider = get_provider(PROVIDER_OLLAMA, {
            "base_url": "http://192.168.1.100:11434",
            "model": "llama3",
        })
        self.assertEqual(provider.base_url, "http://192.168.1.100:11434")

    def test_unknown_provider_raises(self):
        """get_provider() raises ValueError for unknown providers."""
        with self.assertRaises(ValueError):
            get_provider("magic_cloud", {})

    def test_caching(self):
        """Provider instances are cached by type+url+model."""
        p1 = get_provider(PROVIDER_OLLAMA, {"model": "llama3"})
        p2 = get_provider(PROVIDER_OLLAMA, {"model": "llama3"})
        self.assertIs(p1, p2)

    def test_clear_cache(self):
        """clear_provider_cache() removes all cached instances."""
        p1 = get_provider(PROVIDER_OLLAMA, {"model": "llama3"})
        clear_provider_cache()
        p2 = get_provider(PROVIDER_OLLAMA, {"model": "llama3"})
        self.assertIsNot(p1, p2)


if __name__ == "__main__":
    unittest.main()
