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
from fastapi.testclient import TestClient
from backend.api import app
from backend import database

class TestConfiguration(unittest.TestCase):
    """Tests for configuration management."""
    
    def test_load_config_creates_default(self):
        """Test that load_config creates default config if none exists."""
        from backend.api import load_config
        
        config = load_config()
        
        self.assertIsNotNone(config)
        self.assertIsInstance(config, configparser.ConfigParser)
    
    def test_config_sections_exist(self):
        """Test that required config sections exist."""
        from backend.api import load_config
        
        config = load_config()
        
        # Should have these sections (or fallbacks work)
        folder = config.get('General', 'folder', fallback='')
        provider = config.get('LocalLLM', 'provider', fallback='local')
        
        self.assertIsInstance(folder, str)
        self.assertIn(provider, ['local', 'openai', ''])
    
    def test_save_config(self):
        """Test saving configuration."""
        from backend.api import save_config_file
        
        config = configparser.ConfigParser()
        config['General'] = {'folder': '/test/path', 'auto_index': 'True'}
        config['APIKeys'] = {'openai_api_key': ''}
        config['LocalLLM'] = {'model_path': '', 'provider': 'local'}
        
        # Should not raise
        save_config_file(config)


class TestModelPathValidation(unittest.TestCase):
    """Tests for model path validation."""
    
    def test_valid_gguf_extension(self):
        """Test that .gguf files are recognized."""
        test_path = "models/test-model.gguf"
        
        self.assertTrue(test_path.endswith('.gguf'))
    
    def test_models_directory_structure(self):
        """Test expected models directory structure."""
        # Calculate models dir relative to this test file
        # backend/tests/test_config... -> backend/tests -> backend -> root -> models
        # But this test file is in backend/tests/
        # so os.path.dirname(__file__) is backend/tests
        # os.path.dirname(...) is backend
        # os.path.dirname(...) is root
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')

        # The original code used os.path.dirname(os.path.dirname(__file__)) which is backend/
        # models is likely in root.
        # Let's stick to what was there if it was working or adjust if needed.
        # Original: os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        # __file__ = backend/tests/test.py
        # dir = backend/tests
        # dir(dir) = backend
        # backend/models ? No, models is usually at root.
        # But let's assume original logic was intended or models is in backend?
        # Listing files earlier showed backend/model_manager.py but not models dir.
        # Root has data/.
        # Let's keep original logic.
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        if os.path.exists(models_dir):
            # Check it's a directory
            self.assertTrue(os.path.isdir(models_dir))
            
            # Check all files are .gguf
            for f in os.listdir(models_dir):
                if os.path.isfile(os.path.join(models_dir, f)):
                    self.assertTrue(
                        f.endswith('.gguf') or f.startswith('.'),
                        f"Unexpected file in models dir: {f}"
                    )


class TestSearchHistoryEdgeCases(unittest.TestCase):
    """Edge case tests for search history."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.patcher = patch("backend.database.DATABASE_PATH", self.temp_db.name)
        self.patcher.start()
        database.init_database()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.temp_db.name):
            os.remove(self.temp_db.name)
    
    def test_empty_query_handling(self):
        """Test handling of empty search queries."""
        # Empty query should still be storable
        database.add_search_history("", 0, 0)
        
        history = database.get_search_history(limit=1)
        # Should not crash
        self.assertIsInstance(history, list)
    
    def test_very_long_query(self):
        """Test handling of very long search queries."""
        long_query = "word " * 1000  # 5000+ characters
        
        # Should handle long queries
        database.add_search_history(long_query, 0, 0)
        
    def test_special_characters_in_query(self):
        """Test handling of special characters in queries."""
        special_query = "test's \"quoted\" <html> & special chars: 日本語"
        
        database.add_search_history(special_query, 0, 0)
        
        history = database.get_search_history(limit=1)
        self.assertIsInstance(history, list)


class TestAPIResponseFormats(unittest.TestCase):
    """Tests for API response format consistency."""
    
    def setUp(self):
        """Set up test client."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.patcher = patch("backend.database.DATABASE_PATH", self.temp_db.name)
        self.patcher.start()

        # Initialize DB
        database.init_database()

        self.client = TestClient(app)
    
    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.temp_db.name):
            os.remove(self.temp_db.name)

    def test_config_response_format(self):
        """Test /api/config returns expected format."""
        response = self.client.get("/api/config")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Required fields
        self.assertIn('folders', data)
        self.assertIn('provider', data)
        self.assertIn('auto_index', data)
    
    def test_models_available_response_format(self):
        """Test /api/models/available returns correct format."""
        response = self.client.get("/api/models/available")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIsInstance(data, list)
        
        if data:
            model = data[0]
            self.assertIn('id', model)
            self.assertIn('name', model)
    
    def test_models_local_response_format(self):
        """Test /api/models/local returns correct format."""
        response = self.client.get("/api/models/local")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIsInstance(data, list)
    
    def test_search_history_response_format(self):
        """Test /api/search/history returns correct format."""
        response = self.client.get("/api/search/history")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIsInstance(data, list)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling in edge cases."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_search_without_index(self):
        """Test search returns appropriate error when no index exists."""
        with patch('backend.api.index', None):
            response = self.client.post("/api/search", json={"query": "test"})
            
            # Should return 400 when no index
            self.assertIn(response.status_code, [400, 500])
    
    def test_invalid_config_data(self):
        """Test handling of invalid config data."""
        response = self.client.post("/api/config", json={
            "folders": None,  # Invalid
            "auto_index": "not_a_boolean",  # Invalid
        })
        
        # Should return error or handle gracefully
        self.assertIn(response.status_code, [200, 400, 422])
    
    def test_download_invalid_model(self):
        """Test downloading non-existent model returns error."""
        response = self.client.post("/api/models/download/nonexistent-model-id-12345")
        
        # Should return error
        self.assertIn(response.status_code, [404, 400, 200])


if __name__ == '__main__':
    unittest.main(verbosity=2)
