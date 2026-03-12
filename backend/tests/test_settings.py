"""
backend/tests/test_settings.py
-------------------------------
Unit tests for the embedding settings router (backend/settings.py).

All LangChain / embedding factory calls are mocked — no real model loading.
"""

import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend.api import app


class TestEmbeddingSettingsRouter(unittest.TestCase):
    """Tests for GET /api/settings/embeddings and POST /api/settings/embeddings."""

    def setUp(self):
        self.client = TestClient(app)
        # Provide a clean, isolated embedding config on app.state for each test
        app.state.embedding_config = {
            "provider_type": "local",
            "model_name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "api_key": "",
        }

    # ------------------------------------------------------------------
    # GET tests
    # ------------------------------------------------------------------

    def test_get_embeddings_config_returns_200(self):
        """GET returns 200 with the three expected fields."""
        response = self.client.get("/api/settings/embeddings")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("provider_type", data)
        self.assertIn("model_name", data)
        self.assertIn("api_key_set", data)

    def test_get_embeddings_config_reads_from_state(self):
        """GET reflects the config stored in app.state."""
        app.state.embedding_config = {
            "provider_type": "commercial_api",
            "model_name": "text-embedding-3-small",
            "api_key": "sk-secret",
        }
        response = self.client.get("/api/settings/embeddings")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["provider_type"], "commercial_api")
        self.assertEqual(data["model_name"], "text-embedding-3-small")
        self.assertTrue(data["api_key_set"])
        # Raw key must never be returned
        self.assertNotIn("api_key", data)

    def test_get_embeddings_config_key_not_set_flag(self):
        """api_key_set is False when no key is stored."""
        # app.state already has api_key="" from setUp
        response = self.client.get("/api/settings/embeddings")
        data = response.json()
        self.assertFalse(data["api_key_set"])

    @patch("backend.settings._read_embedding_section")
    def test_get_embeddings_config_falls_back_to_ini(self, mock_read):
        """When app.state has no embedding_config, GET falls back to config.ini."""
        app.state.embedding_config = None
        mock_read.return_value = {
            "provider_type": "huggingface_api",
            "model_name": "BAAI/bge-large-en-v1.5",
            "api_key": "hf-token",
        }
        response = self.client.get("/api/settings/embeddings")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["provider_type"], "huggingface_api")
        self.assertTrue(data["api_key_set"])

    # ------------------------------------------------------------------
    # POST – success cases
    # ------------------------------------------------------------------

    @patch("backend.settings._write_embedding_section")
    def test_post_embeddings_config_local(self, mock_write):
        """POST with provider_type='local' succeeds without an api_key."""
        payload = {
            "provider_type": "local",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }
        response = self.client.post("/api/settings/embeddings", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["provider_type"], "local")
        # app.state should be updated
        self.assertEqual(app.state.embedding_config["model_name"],
                         "sentence-transformers/all-MiniLM-L6-v2")
        mock_write.assert_called_once()

    @patch("backend.settings._write_embedding_section")
    def test_post_embeddings_config_huggingface_api(self, mock_write):
        """POST with provider_type='huggingface_api' succeeds when api_key is supplied."""
        payload = {
            "provider_type": "huggingface_api",
            "model_name": "BAAI/bge-large-en-v1.5",
            "api_key": "hf-secret-token",
        }
        response = self.client.post("/api/settings/embeddings", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["provider_type"], "huggingface_api")
        mock_write.assert_called_once()

    @patch("backend.settings._write_embedding_section")
    def test_post_embeddings_config_commercial_api(self, mock_write):
        """POST with provider_type='commercial_api' (OpenAI model) succeeds."""
        payload = {
            "provider_type": "commercial_api",
            "model_name": "text-embedding-3-small",
            "api_key": "sk-openai-key",
        }
        response = self.client.post("/api/settings/embeddings", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider_type"], "commercial_api")

    # ------------------------------------------------------------------
    # POST – validation failures
    # ------------------------------------------------------------------

    def test_post_embeddings_config_invalid_provider(self):
        """POST with an unknown provider_type returns 422."""
        payload = {
            "provider_type": "magic_cloud",
            "model_name": "some-model",
            "api_key": "key",
        }
        response = self.client.post("/api/settings/embeddings", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_post_embeddings_config_missing_api_key_for_hf_api(self):
        """POST for 'huggingface_api' without api_key returns 422."""
        payload = {
            "provider_type": "huggingface_api",
            "model_name": "BAAI/bge-large-en-v1.5",
            # api_key intentionally omitted
        }
        response = self.client.post("/api/settings/embeddings", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_post_embeddings_config_missing_api_key_for_commercial(self):
        """POST for 'commercial_api' without api_key returns 422."""
        payload = {
            "provider_type": "commercial_api",
            "model_name": "text-embedding-3-small",
        }
        response = self.client.post("/api/settings/embeddings", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_post_embeddings_config_empty_model_name(self):
        """POST with an empty model_name returns 422."""
        payload = {
            "provider_type": "local",
            "model_name": "   ",  # whitespace-only → invalid after strip
        }
        response = self.client.post("/api/settings/embeddings", json=payload)
        self.assertEqual(response.status_code, 422)


class TestGetActiveEmbeddingClient(unittest.TestCase):
    """Tests for the get_active_embedding_client() helper."""

    def _make_app(self, embedding_cfg=None):
        """Return a minimal mock that mirrods app.state."""
        mock_app = MagicMock()
        mock_app.state.embedding_config = embedding_cfg
        return mock_app

    @patch("backend.settings.get_embedding_client" if False else "backend.llm_integration.get_embedding_client")
    def test_returns_client_from_state(self, _mock_factory):
        """Helper calls get_embedding_client() with state values when state is set."""
        from backend.settings import get_active_embedding_client

        mock_client = MagicMock()
        with patch("backend.llm_integration.get_embedding_client", return_value=mock_client) as mock_factory:
            mock_app = self._make_app({
                "provider_type": "local",
                "model_name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                "api_key": "",
            })
            result = get_active_embedding_client(mock_app)
            mock_factory.assert_called_once_with(
                provider_type="local",
                model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                api_key=None,
            )
            self.assertEqual(result, mock_client)

    def test_fallback_when_no_state(self):
        """Helper falls back to get_embeddings() when app.state has no config."""
        from backend.settings import get_active_embedding_client

        mock_client = MagicMock()
        mock_app = self._make_app(embedding_cfg=None)

        with patch("backend.llm_integration.get_embeddings", return_value=mock_client) as mock_legacy:
            result = get_active_embedding_client(mock_app)
            mock_legacy.assert_called_once_with(provider="local")
            self.assertEqual(result, mock_client)


if __name__ == "__main__":
    unittest.main()
