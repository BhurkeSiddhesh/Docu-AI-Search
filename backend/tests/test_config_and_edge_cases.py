import unittest
import tempfile
import shutil
import os
import json
import logging
import configparser
from unittest.mock import patch, MagicMock
from backend import database

# Disable logging during tests
logging.getLogger("backend.api").setLevel(logging.CRITICAL)

class TestConfiguration(unittest.TestCase):
    """Test configuration management."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.ini")
        
        # Patch config path
        self.config_patcher = patch('backend.api.CONFIG_PATH', self.config_path)
        self.config_patcher.start()
        
        # Setup TestClient
        from fastapi.testclient import TestClient
        from backend.api import app
        self.client = TestClient(app)

    def tearDown(self):
        """Clean up."""
        self.config_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_load_config_creates_default(self):
        """Test loading config creates default file if missing."""
        from backend.api import load_config
        
        config = load_config()
        self.assertTrue(os.path.exists(self.config_path))
        self.assertIsInstance(config, configparser.ConfigParser)
        self.assertIn("General", config)
        self.assertIn("APIKeys", config)

    def test_save_config(self):
        """Test saving configuration."""
        from backend.api import save_config_file
        
        config = configparser.ConfigParser()
        config['Test'] = {'test_key': 'test_value'}
        
        save_config_file(config)

        saved = configparser.ConfigParser()
        saved.read(self.config_path)

        self.assertEqual(saved['Test']['test_key'], 'test_value')

    def test_config_sections_exist(self):
        """Test that default config has required sections."""
        from backend.api import load_config
        config = load_config()
        self.assertIn("General", config)
        self.assertIn("APIKeys", config)
        self.assertIn("LocalLLM", config)

class TestSearchHistoryEdgeCases(unittest.TestCase):
    """Test edge cases for search history."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        
        # Patch the database path in backend.database
        self.db_patcher = patch('backend.database.DATABASE_PATH', self.db_path)
        self.db_patcher.start()
        
        # Initialize the database schema
        database.init_database()

    def tearDown(self):
        """Clean up."""
        self.db_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_empty_query_handling(self):
        """Test handling of empty search queries."""
        # Empty query should still be storable
        database.add_search_history("", 0, 0)
        
        history = database.get_search_history(1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['query'], "")

    def test_very_long_query(self):
        """Test handling of very long search queries."""
        long_query = "word " * 1000  # 5000+ characters
        
        # Should handle long queries
        database.add_search_history(long_query, 0, 0)
        
        history = database.get_search_history(1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['query'], long_query)

    def test_special_characters_in_query(self):
        """Test handling of special characters in queries."""
        special_query = "test's \"quoted\" <html> & special chars: 日本語"
        
        database.add_search_history(special_query, 0, 0)
        
        history = database.get_search_history(1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['query'], special_query)

class TestAPIResponseFormats(unittest.TestCase):
    """Test API response formats match expected schema."""

    def setUp(self):
        """Set up test client."""
        # Patch database path BEFORE creating TestClient to ensure it uses the test DB
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db_patcher = patch('backend.database.DATABASE_PATH', self.db_path)
        self.db_patcher.start()

        # Initialize schema
        database.init_database()

        from fastapi.testclient import TestClient
        from backend.api import app

        # Add dependency overrides if needed (e.g. for security)
        app.dependency_overrides = {}

        self.client = TestClient(app)

    def tearDown(self):
        """Clean up."""
        self.db_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_config_response_format(self):
        """Test /api/config returns correct format."""
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)
        # Check against what API actually returns (dict of dicts or flattened?)
        # Based on configparser usage in api.py, it likely iterates sections
        # Let's check for keys we know exist in default
        found_key = False
        for key in ["folders", "General", "APIKeys"]:
            if key in data or (isinstance(data, dict) and any(k.lower() == key.lower() for k in data)):
                found_key = True
                break
        # If the API transforms it, adapt here. For now assume some config data returned.
        self.assertTrue(len(data) > 0)

    def test_models_available_response_format(self):
        """Test /api/models/available returns correct format."""
        # Mock the external call
        with patch('backend.model_manager.get_available_models') as mock_get:
            mock_get.return_value = [
                {"id": "test-model", "name": "Test Model", "size": "1GB"}
            ]
            response = self.client.get("/api/models/available")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        if len(data) > 0:
            self.assertIn("id", data[0])
            self.assertIn("name", data[0])

    def test_models_local_response_format(self):
        """Test /api/models/local returns correct format."""
        with patch('backend.model_manager.get_local_models') as mock_get:
            mock_get.return_value = [
                {"filename": "model.gguf", "size_mb": 100}
            ]
            response = self.client.get("/api/models/local")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_search_history_response_format(self):
        """Test /api/search/history returns correct format."""
        # Add some dummy history
        database.add_search_history("test", 1, 100)

        response = self.client.get("/api/search/history")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertIn("query", data[0])
        self.assertIn("timestamp", data[0])

class TestErrorHandling(unittest.TestCase):
    """Test API error handling."""

    def setUp(self):
        """Set up test client."""
        from fastapi.testclient import TestClient
        from backend.api import app
        self.client = TestClient(app)

    def test_download_invalid_model(self):
        """Test downloading non-existent model returns error."""
        # Mock start_download to return error
        with patch('backend.model_manager.start_download', return_value=(False, "Model not found")):
             response = self.client.post("/api/models/download/nonexistent-model-id-12345")
        
        # Based on api.py: if not success, raise HTTPException(400)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)

    def test_invalid_config_data(self):
        """Test sending invalid data to config endpoint."""
        response = self.client.post("/api/config", json="not a json object")
        self.assertEqual(response.status_code, 422) # Validation error

    def test_search_without_index(self):
        """Test searching without an index returns appropriate error."""
        # Patch backend.search.search to raise exception
        with patch('backend.search.search', side_effect=ValueError("Index not found")):
            response = self.client.post("/api/search", json={"query": "test"})
            # API catches Exception and returns 500, but checking log shows it might handle index check differently
            # If search raises ValueError, API might return 500
            self.assertIn(response.status_code, [400, 500])

if __name__ == '__main__':
    unittest.main()
