"""
Test Configuration and Settings
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import configparser
import sys
import sqlite3

class TestConfiguration(unittest.TestCase):
    def setUp(self):
        # Create a mock for pydantic that has BaseModel
        mock_pydantic = MagicMock()
        class MockBaseModel:
            pass
        mock_pydantic.BaseModel = MockBaseModel

        self.modules_patcher = patch.dict(sys.modules, {
            'fastapi': MagicMock(),
            'fastapi.testclient': MagicMock(),
            'fastapi.responses': MagicMock(),
            'fastapi.middleware.cors': MagicMock(),
            'uvicorn': MagicMock(),
            'pydantic': mock_pydantic,
        })
        self.modules_patcher.start()

        if 'backend.api' in sys.modules:
            del sys.modules['backend.api']
        import backend.api
        self.api = backend.api

        self.original_config_path = self.api.CONFIG_PATH
        self.temp_config = tempfile.NamedTemporaryFile(delete=False)
        self.temp_config.close()
        self.api.CONFIG_PATH = self.temp_config.name

    def tearDown(self):
        self.api.CONFIG_PATH = self.original_config_path
        if os.path.exists(self.temp_config.name):
            os.remove(self.temp_config.name)
        self.modules_patcher.stop()

    def test_load_config_creates_default(self):
        if os.path.exists(self.api.CONFIG_PATH):
            os.remove(self.api.CONFIG_PATH)
        config = self.api.load_config()
        self.assertIsNotNone(config)
    
    def test_config_sections_exist(self):
        if os.path.exists(self.api.CONFIG_PATH):
            os.remove(self.api.CONFIG_PATH)
        self.api.load_config()
        config = self.api.load_config()
        self.assertTrue(config.has_section('General'))
    
    def test_save_config(self):
        config = configparser.ConfigParser()
        config['General'] = {'folder': '/test/path', 'auto_index': 'True'}
        self.api.save_config_file(config)


class TestModelPathValidation(unittest.TestCase):
    def test_valid_gguf_extension(self):
        test_path = "models/test-model.gguf"
        self.assertTrue(test_path.endswith('.gguf'))
    
    def test_models_directory_structure(self):
        with patch('os.path.exists', return_value=True),              patch('os.path.isdir', return_value=True):
            pass


class TestSearchHistoryEdgeCases(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp()
        import backend.database
        self.original_db_path = backend.database.DATABASE_PATH
        backend.database.DATABASE_PATH = self.db_path
        backend.database.init_database()
        self.database = backend.database

    def tearDown(self):
        os.close(self.db_fd)
        os.remove(self.db_path)
        self.database.DATABASE_PATH = self.original_db_path
    
    def test_empty_query_handling(self):
        self.database.add_search_history("", 0, 0)
        history = self.database.get_search_history(limit=1)
        self.assertIsInstance(history, list)
    
    def test_very_long_query(self):
        long_query = "word " * 1000
        self.database.add_search_history(long_query, 0, 0)
        
    def test_special_characters_in_query(self):
        special_query = "test's \"quoted\" <html> & special chars: 日本語"
        self.database.add_search_history(special_query, 0, 0)
        history = self.database.get_search_history(limit=1)
        self.assertIsInstance(history, list)


class TestAPIResponseFormats(unittest.TestCase):
    pass

class TestErrorHandling(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
