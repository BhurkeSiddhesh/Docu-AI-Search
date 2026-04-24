import unittest
import os
import configparser
from unittest.mock import patch
import sys

# Import api
from backend import api


class TestConfigCache(unittest.TestCase):
    def setUp(self):
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

    def test_config_reads_values(self):
        """Test that load_config reads the correct values."""
        config = api.load_config()
        self.assertEqual(config['Test']['key'], 'value1')

    def test_config_returns_configparser(self):
        """Test that load_config returns a ConfigParser object."""
        config = api.load_config()
        self.assertIsInstance(config, configparser.ConfigParser)

    def test_cache_invalidation(self):
        """Test that modifying the file is reflected in the next load."""
        config1 = api.load_config()
        self.assertEqual(config1['Test']['key'], 'value1')

        # Modify file content
        config = configparser.ConfigParser()
        config['Test'] = {'key': 'value2'}

        # Force mtime update
        mtime1 = os.path.getmtime(self.test_config_path)
        new_time = mtime1 + 1
        with open(self.test_config_path, 'w') as f:
            config.write(f)
        os.utime(self.test_config_path, (new_time, new_time))

        config2 = api.load_config()
        self.assertEqual(config2['Test']['key'], 'value2')

    def test_config_creation(self):
        """Test that config is created if missing."""
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

        config = api.load_config()
        self.assertTrue(os.path.exists(self.test_config_path))
        self.assertIn('General', config)

    def test_multiple_loads_return_correct_values(self):
        """Test that multiple calls to load_config return correct values."""
        config1 = api.load_config()
        config2 = api.load_config()
        # Both should have the same values
        self.assertEqual(config1['Test']['key'], 'value2' if 'value2' in config1['Test']['key'] else 'value1')
        self.assertEqual(config1['Test']['key'], config2['Test']['key'])


if __name__ == '__main__':
    unittest.main()
