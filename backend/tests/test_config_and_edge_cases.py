import unittest
import os
import shutil
import tempfile
import sqlite3
import configparser # FIX: Added missing import
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.api import app
from backend import database

class TestSearchHistoryEdgeCases(unittest.TestCase):
    def setUp(self):
        # Setup temporary database
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        os.close(self.temp_db_fd)
        
        # Patch database path
        self.db_patcher = patch('backend.database.DATABASE_PATH', self.temp_db_path)
        self.db_patcher.start()
        
        # Initialize database schema
        database.init_database()
        
    def tearDown(self):
        self.db_patcher.stop()
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_empty_query_handling(self):
        """Test handling of empty search queries."""
        # Should handle empty string gracefully
        try:
            database.add_search_history("", 0, 0)
        except Exception as e:
            self.fail(f"add_search_history raised exception for empty query: {e}")

        history = database.get_search_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['query'], "")

    def test_very_long_query(self):
        """Test handling of very long search queries."""
        long_query = "a" * 10000
        try:
            database.add_search_history(long_query, 0, 0)
        except Exception as e:
            self.fail(f"add_search_history raised exception for long query: {e}")

        history = database.get_search_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['query'], long_query)

    def test_special_characters_in_query(self):
        """Test handling of special characters in queries."""
        special_query = "QUERY with 'quotes' and \"double quotes\" and emoji 🔍"
        try:
            database.add_search_history(special_query, 0, 0)
        except Exception as e:
            self.fail(f"add_search_history raised exception for special chars: {e}")

        history = database.get_search_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['query'], special_query)


class TestAPIResponseFormats(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Setup temporary database for API tests too
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        os.close(self.temp_db_fd)
        self.db_patcher = patch('backend.database.DATABASE_PATH', self.temp_db_path)
        self.db_patcher.start()
        database.init_database()

    def tearDown(self):
        self.db_patcher.stop()
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_config_response_format(self):
        """Test /api/config returns correct format."""
        with patch('backend.api.verify_local_request', return_value=None):
            response = self.client.get("/api/config")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            expected_keys = {
                "folders", "auto_index", "openai_api_key", "gemini_api_key",
                "anthropic_api_key", "grok_api_key", "local_model_path",
                "provider", "tensor_split"
            }
            self.assertTrue(expected_keys.issubset(data.keys()))
            self.assertIsInstance(data['folders'], list)
            self.assertIsInstance(data['auto_index'], bool)

    def test_models_available_response_format(self):
        """Test /api/models/available returns correct format."""
        with patch('backend.api.get_available_models') as mock_get:
            mock_get.return_value = [{"id": "model1", "name": "Model 1"}]
            response = self.client.get("/api/models/available")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]['id'], "model1")

    def test_models_local_response_format(self):
        """Test /api/models/local returns correct format."""
        with patch('backend.api.get_local_models') as mock_get:
            mock_get.return_value = [{"name": "local.gguf", "path": "/path/to/local.gguf"}]
            response = self.client.get("/api/models/local")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIsInstance(data, list)

    def test_search_history_response_format(self):
        """Test /api/search/history returns correct format."""
        # Add some history first
        database.add_search_history("test", 5, 100)
        
        response = self.client.get("/api/search/history")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        item = data[0]
        self.assertIn('id', item)
        self.assertIn('query', item)
        self.assertIn('timestamp', item)
        self.assertIn('result_count', item)


class TestConfigPersistence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'config.ini')
        self.config_patcher = patch('backend.api.CONFIG_PATH', self.config_path)
        self.config_patcher.start()

        # Also need DB for verify_local_request/dependencies if they touch it
        # though config logic mostly touches file system
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        os.close(self.temp_db_fd)
        self.db_patcher = patch('backend.database.DATABASE_PATH', self.temp_db_path)
        self.db_patcher.start()
        database.init_database()

        self.client = TestClient(app)

    def tearDown(self):
        self.config_patcher.stop()
        self.db_patcher.stop()
        shutil.rmtree(self.temp_dir)
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_load_config_creates_default(self):
        """Test that load_config creates a default config file if missing."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

        with patch('backend.api.verify_local_request', return_value=None):
            response = self.client.get("/api/config")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(os.path.exists(self.config_path))

    def test_save_config(self):
        """Test saving configuration."""
        payload = {
            "folders": ["/test/path"],
            "auto_index": True,
            "provider": "openai"
        }
        with patch('backend.api.verify_local_request', return_value=None):
            response = self.client.post("/api/config", json=payload)
            self.assertEqual(response.status_code, 200)
            
            config = configparser.ConfigParser()
            config.read(self.config_path)
            self.assertEqual(config.get('General', 'folders'), '/test/path')
            self.assertEqual(config.get('General', 'auto_index'), 'True')

    def test_invalid_config_data(self):
        """Test posting invalid config data."""
        with patch('backend.api.verify_local_request', return_value=None):
            response = self.client.post("/api/config", json={"folders": "not a list"})
            self.assertEqual(response.status_code, 422) # Validation error

    def test_config_sections_exist(self):
        """Ensure all required sections exist in generated config."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

        with patch('backend.api.verify_local_request', return_value=None):
            self.client.get("/api/config")

            config = configparser.ConfigParser()
            config.read(self.config_path)
            self.assertTrue(config.has_section('General'))
            self.assertTrue(config.has_section('APIKeys'))
            self.assertTrue(config.has_section('LocalLLM'))

if __name__ == '__main__':
    unittest.main()
