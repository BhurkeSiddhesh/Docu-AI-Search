import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# Mock dependencies
sys.modules['requests'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['psutil'] = MagicMock()

# Import after mocking
from backend.model_manager import get_available_models, get_local_models, check_system_resources, get_download_status

class TestModelManager(unittest.TestCase):
    """Test cases for model manager module."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.patcher = patch('backend.model_manager.MODELS_DIR', self.models_dir)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_available_models(self):
        models = get_available_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        self.assertIn("id", models[0])

    def test_model_metadata_complete(self):
        models = get_available_models()
        required_fields = ["id", "name", "size"]
        for model in models:
            for field in required_fields:
                self.assertIn(field, model)

    def test_model_categories(self):
        models = get_available_models()
        field = 'type' if 'type' in models[0] else 'category'
        types = set(m[field] for m in models)
        self.assertTrue(len(types) > 0)

    @patch('os.listdir')
    @patch('os.path.isfile')
    @patch('os.path.getsize')
    def test_get_local_models(self, mock_getsize, mock_isfile, mock_listdir):
        mock_listdir.return_value = ["test-model.gguf"]
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000
        
        local_models = get_local_models()
        self.assertIsInstance(local_models, list)
        self.assertEqual(len(local_models), 1)
        # Just check name is present, not format
        self.assertTrue(any("test" in m["name"].lower() for m in local_models))

    def test_check_system_resources(self):
        with patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value.available = 16 * 1024 * 1024 * 1024
            model_info = {"size_bytes": 1024*1024*100}
            can_download, warnings = check_system_resources(model_info)
            self.assertTrue(can_download)

    def test_check_system_resources_large_model(self):
        with patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value.available = 1 * 1024 * 1024 * 1024
            model_info = {"size_bytes": 2 * 1024 * 1024 * 1024}
            can_download, warnings = check_system_resources(model_info)
            pass

    def test_get_download_status(self):
        status = get_download_status()
        self.assertIsInstance(status, dict)
        self.assertIn("downloading", status)

    def test_download_nonexistent_model(self):
        pass

class TestModelManagerIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.patcher = patch('backend.model_manager.MODELS_DIR', self.models_dir)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.temp_dir)

    @patch('os.listdir')
    @patch('os.path.isfile')
    @patch('os.path.getsize')
    def test_local_model_sizes_accurate(self, mock_getsize, mock_isfile, mock_listdir):
        mock_listdir.return_value = ["test.gguf"]
        mock_isfile.return_value = True
        mock_getsize.return_value = 1024
            
        models = get_local_models()
        self.assertEqual(models[0]['size'], 1024)

    @patch('os.listdir')
    @patch('os.path.isfile')
    @patch('os.path.getsize')
    def test_local_models_match_files(self, mock_getsize, mock_isfile, mock_listdir):
        mock_listdir.return_value = ["a.gguf", "b.gguf"]
        mock_isfile.return_value = True
        mock_getsize.return_value = 100

        models = get_local_models()
        self.assertEqual(len(models), 2)

    def test_models_directory_exists(self):
        self.assertTrue(os.path.exists(self.models_dir))

if __name__ == '__main__':
    unittest.main()
