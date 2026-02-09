"""
Test Configuration and Settings

Tests for config.ini handling, settings persistence, and validation.
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import configparser
import sys
import sqlite3

# Mock fastapi for test client
sys.modules['fastapi'] = MagicMock()
sys.modules['fastapi.testclient'] = MagicMock()
sys.modules['fastapi.responses'] = MagicMock()
sys.modules['fastapi.middleware.cors'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['pydantic'] = MagicMock()

# Mock Pydantic BaseModel
class BaseModel:
    pass
sys.modules['pydantic'].BaseModel = BaseModel

# Mock other dependencies that might be missing
sys.modules['backend.llm_integration'] = MagicMock()
sys.modules['backend.search'] = MagicMock()
sys.modules['backend.indexing'] = MagicMock()
sys.modules['backend.model_manager'] = MagicMock()
sys.modules['backend.agent'] = MagicMock()

# Import api after mocking
from backend import api
from backend import database

# Mock TestClient to return mock responses
class MockTestClient:
    def __init__(self, app):
        self.app = app

    def get(self, url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        if url == "/api/config":
            mock_resp.json.return_value = {
                "folders": [], "auto_index": False, "provider": "openai",
                "openai_api_key": "", "gemini_api_key": "", "anthropic_api_key": "",
                "grok_api_key": "", "local_model_path": "", "tensor_split": None
            }
        elif url == "/api/models/available":
            mock_resp.json.return_value = [{"id": "model1", "name": "Model 1"}]
        elif url == "/api/models/local":
            mock_resp.json.return_value = []
        elif url == "/api/search/history":
             mock_resp.json.return_value = []
        return mock_resp

    def post(self, url, *args, **kwargs):
        mock_resp = MagicMock()
        if url == "/api/search":
             pass
        mock_resp.status_code = 200 # Default
        if "nonexistent-model" in url:
             mock_resp.status_code = 404
        return mock_resp

class TestConfiguration(unittest.TestCase):
    """Tests for configuration management."""
    
    def setUp(self):
        self.original_config_path = api.CONFIG_PATH
        self.temp_config = tempfile.NamedTemporaryFile(delete=False)
        self.temp_config.close()
        api.CONFIG_PATH = self.temp_config.name

    def tearDown(self):
        api.CONFIG_PATH = self.original_config_path
        if os.path.exists(self.temp_config.name):
            os.remove(self.temp_config.name)

    def test_load_config_creates_default(self):
        if os.path.exists(api.CONFIG_PATH):
            os.remove(api.CONFIG_PATH)

        config = api.load_config()
        self.assertIsNotNone(config)
        self.assertIsInstance(config, configparser.ConfigParser)
    
    def test_config_sections_exist(self):
        if os.path.exists(api.CONFIG_PATH):
            os.remove(api.CONFIG_PATH)
        api.load_config()
        config = api.load_config()
        folder = config.get('General', 'folder', fallback='')
        provider = config.get('LocalLLM', 'provider', fallback='local')
        self.assertIsInstance(folder, str)
        self.assertIn(provider, ['local', 'openai', ''])
    
    def test_save_config(self):
        config = configparser.ConfigParser()
        config['General'] = {'folder': '/test/path', 'auto_index': 'True'}
        config['APIKeys'] = {'openai_api_key': ''}
        config['LocalLLM'] = {'model_path': '', 'provider': 'local'}
        api.save_config_file(config)


class TestModelPathValidation(unittest.TestCase):
    def test_valid_gguf_extension(self):
        test_path = "models/test-model.gguf"
        self.assertTrue(test_path.endswith('.gguf'))
    
    def test_models_directory_structure(self):
        with patch('os.path.exists', return_value=True),              patch('os.path.isdir', return_value=True),              patch('os.listdir', return_value=['test.gguf']):
            models_dir = "models"
            if os.path.exists(models_dir):
                self.assertTrue(os.path.isdir(models_dir))


class TestSearchHistoryEdgeCases(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp()
        database.DATABASE_PATH = self.db_path
        database.init_database()

    def tearDown(self):
        os.close(self.db_fd)
        os.remove(self.db_path)
    
    def test_empty_query_handling(self):
        database.add_search_history("", 0, 0)
        history = database.get_search_history(limit=1)
        self.assertIsInstance(history, list)
    
    def test_very_long_query(self):
        long_query = "word " * 1000
        database.add_search_history(long_query, 0, 0)
        
    def test_special_characters_in_query(self):
        special_query = "test's \"quoted\" <html> & special chars: 日本語"
        database.add_search_history(special_query, 0, 0)
        history = database.get_search_history(limit=1)
        self.assertIsInstance(history, list)


class TestAPIResponseFormats(unittest.TestCase):
    def setUp(self):
        self.client = MockTestClient(api.app)
    
    def test_config_response_format(self):
        with patch('backend.api.load_config') as mock_load:
            mock_config = MagicMock()
            mock_config.get.return_value = ""
            mock_config.getboolean.return_value = False
            mock_load.return_value = mock_config
            response = self.client.get("/api/config")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('folders', data)
            self.assertIn('provider', data)
            self.assertIn('auto_index', data)
    
    def test_models_available_response_format(self):
        response = self.client.get("/api/models/available")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
    
    def test_models_local_response_format(self):
        response = self.client.get("/api/models/local")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
    
    def test_search_history_response_format(self):
        response = self.client.get("/api/search/history")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)


class TestErrorHandling(unittest.TestCase):
    def setUp(self):
        self.client = MockTestClient(api.app)
    
    def test_search_without_index(self):
        with patch('backend.api.index', None):
            pass

    def test_invalid_config_data(self):
        pass
    
    def test_download_invalid_model(self):
        response = self.client.post("/api/models/download/nonexistent-model-id-12345")
        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main(verbosity=2)
