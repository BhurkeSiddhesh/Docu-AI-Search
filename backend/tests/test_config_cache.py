import sys; import os; sys.path.append(os.getcwd())
import unittest
import os
import time
import configparser
from unittest.mock import patch, MagicMock
from backend import api

class TestConfigCache(unittest.TestCase):
    def setUp(self):
        # Reset cache before each test
        # Accessing private variables for testing purposes
        if hasattr(api, '_config_cache'):
            api._config_cache = None
        if hasattr(api, '_config_mtime'):
            api._config_mtime = 0

        # Create a temporary config file
        self.test_config_path = "test_config_cache.ini"
        self.original_config_path = api.CONFIG_PATH
        api.CONFIG_PATH = self.test_config_path

        config = configparser.ConfigParser()
        config['Test'] = {'key': 'value1'}
        with open(self.test_config_path, 'w') as f:
            config.write(f)

    def tearDown(self):
        # Restore original path and cleanup
        api.CONFIG_PATH = self.original_config_path
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def test_cache_hit(self):
        """Test that subsequent calls return the same object if file unchanged."""
        config1 = api.load_config()
        self.assertEqual(config1['Test']['key'], 'value1')

        config2 = api.load_config()
        self.assertIs(config1, config2, "Should return the same config object")

    def test_cache_invalidation(self):
        """Test that modifying the file invalidates the cache."""
        config1 = api.load_config()
        self.assertEqual(config1['Test']['key'], 'value1')

        # Ensure mtime changes
        mtime1 = os.path.getmtime(self.test_config_path)

        # Modify file content
        config = configparser.ConfigParser()
        config['Test'] = {'key': 'value2'}

        # Force mtime update
        new_time = mtime1 + 1
        with open(self.test_config_path, 'w') as f:
            config.write(f)
        os.utime(self.test_config_path, (new_time, new_time))

        config2 = api.load_config()
        self.assertIsNot(config1, config2, "Should return a new config object")
        self.assertEqual(config2['Test']['key'], 'value2')

    def test_config_creation(self):
        """Test that config is created if missing."""
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

        config = api.load_config()
        self.assertTrue(os.path.exists(self.test_config_path))
        self.assertIn('General', config)

    @patch('os.path.getmtime')
    def test_getmtime_error(self, mock_getmtime):
        """Test fallback when getmtime fails."""
        # Load once to populate cache
        config1 = api.load_config()

        # Simulate error on second call
        mock_getmtime.side_effect = OSError("Access denied")

        config2 = api.load_config()
        # Should return cached config
        self.assertIs(config1, config2)

if __name__ == '__main__':
    unittest.main()
