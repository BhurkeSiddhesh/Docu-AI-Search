import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import shutil

from backend import model_manager
from backend.api import app
from fastapi.testclient import TestClient


class TestModelRecommendations(unittest.TestCase):
    def test_recommendations_include_system_profile(self):
        with patch('backend.model_manager.get_system_profile') as mock_profile:
            mock_profile.return_value = {
                'ram_gb_total': 16.0,
                'ram_gb_available': 12.0,
                'cpu_cores_logical': 8,
                'cpu_cores_physical': 4,
                'disk_gb_free': 50.0,
            }
            data = model_manager.get_model_recommendations(max_results=3)

        self.assertIn('system', data)
        self.assertIn('recommendations', data)
        self.assertLessEqual(len(data['recommendations']), 3)

    def test_available_models_are_annotated(self):
        models = model_manager.get_available_models()
        self.assertIsInstance(models, list)
        self.assertIn('recommended_for_system', models[0])
        self.assertIn('compatibility', models[0])

    def test_scoring_handles_mocked_profile_values(self):
        profile = {
            'ram_gb_total': MagicMock(),
            'ram_gb_available': MagicMock(),
            'disk_gb_free': MagicMock(),
        }
        sample_model = {
            'ram_required': 4,
            'size_bytes': 668000000,
            'recommended': True,
            'quantization': 'Q4_K_M',
        }

        result = model_manager._score_model_for_system(sample_model, profile)

        self.assertIn('score', result)
        self.assertIn('compatibility', result)

    def test_get_system_profile_returns_valid_data(self):
        """Test that get_system_profile returns valid system information."""
        profile = model_manager.get_system_profile()

        self.assertIn('ram_gb_total', profile)
        self.assertIn('ram_gb_available', profile)
        self.assertIn('cpu_cores_logical', profile)
        self.assertIn('cpu_cores_physical', profile)
        self.assertIn('disk_gb_free', profile)

        # All values should be positive numbers
        self.assertGreater(profile['ram_gb_total'], 0)
        self.assertGreater(profile['cpu_cores_logical'], 0)
        self.assertGreater(profile['cpu_cores_physical'], 0)

    def test_coerce_number_handles_invalid_values(self):
        """Test _coerce_number helper function with various inputs."""
        # Valid number
        self.assertEqual(model_manager._coerce_number(5.5, 10), 5.5)

        # Invalid values should return default
        self.assertEqual(model_manager._coerce_number("invalid", 10), 10.0)
        self.assertEqual(model_manager._coerce_number(None, 10), 10.0)
        self.assertEqual(model_manager._coerce_number(MagicMock(), 10), 10.0)

        # Zero or negative should return default
        self.assertEqual(model_manager._coerce_number(0, 10), 10.0)
        self.assertEqual(model_manager._coerce_number(-5, 10), 10.0)

    def test_score_model_for_system_excellent_compatibility(self):
        """Test scoring algorithm for excellent compatibility."""
        profile = {
            'ram_gb_total': 16.0,
            'ram_gb_available': 12.0,
            'disk_gb_free': 100.0,
        }
        model = {
            'ram_required': 8,  # 50% of RAM - excellent fit
            'size_bytes': 5 * 1024**3,  # 5GB
            'recommended': True,
            'quantization': 'Q4_K_M',
        }

        result = model_manager._score_model_for_system(model, profile)

        self.assertTrue(result['fits'])
        self.assertEqual(result['compatibility'], 'excellent')
        self.assertGreater(result['score'], 75)

    def test_score_model_for_system_insufficient_ram(self):
        """Test scoring when model requires more RAM than available."""
        profile = {
            'ram_gb_total': 8.0,
            'ram_gb_available': 6.0,
            'disk_gb_free': 100.0,
        }
        model = {
            'ram_required': 16,  # More than available
            'size_bytes': 5 * 1024**3,
            'recommended': False,
            'quantization': 'Q8_0',
        }

        result = model_manager._score_model_for_system(model, profile)

        self.assertFalse(result['fits'])
        self.assertEqual(result['compatibility'], 'not_recommended')

    def test_score_model_for_system_insufficient_disk(self):
        """Test scoring when insufficient disk space."""
        profile = {
            'ram_gb_total': 16.0,
            'ram_gb_available': 12.0,
            'disk_gb_free': 1.0,  # Very low disk space
        }
        model = {
            'ram_required': 4,
            'size_bytes': 10 * 1024**3,  # 10GB file
            'recommended': True,
            'quantization': 'Q4_K_M',
        }

        result = model_manager._score_model_for_system(model, profile)

        self.assertFalse(result['fits'])

    def test_recommendations_filters_non_fitting_models(self):
        """Test that recommendations only include models that fit the system."""
        with patch('backend.model_manager.get_system_profile') as mock_profile:
            mock_profile.return_value = {
                'ram_gb_total': 4.0,  # Low RAM system
                'ram_gb_available': 3.0,
                'cpu_cores_logical': 4,
                'cpu_cores_physical': 2,
                'disk_gb_free': 50.0,
            }
            data = model_manager.get_model_recommendations(max_results=10)

        # All recommendations should fit the system
        for rec in data['recommendations']:
            self.assertLessEqual(rec.get('ram_required', 0), 4)


class TestModelRecommendationEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('backend.api.get_model_recommendations')
    def test_recommendations_endpoint(self, mock_get_model_recommendations):
        mock_get_model_recommendations.return_value = {
            'system': {'ram_gb_total': 16},
            'recommendations': [{'id': 'phi-2', 'score': 99}],
        }

        response = self.client.get('/api/models/recommendations')

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn('system', payload)
        self.assertIn('recommendations', payload)


class TestModelDownload(unittest.TestCase):
    """Test model download functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_models_dir = model_manager.MODELS_DIR
        model_manager.MODELS_DIR = self.temp_dir

    def tearDown(self):
        model_manager.MODELS_DIR = self.original_models_dir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.model_manager.check_system_resources')
    @patch('backend.model_manager.threading.Thread')
    def test_start_download_success(self, mock_thread, mock_check_resources):
        """Test starting a download successfully."""
        mock_check_resources.return_value = (True, [])

        success, message = model_manager.start_download('tinyllama-1.1b-chat-v1.0.Q4_K_M')

        self.assertTrue(success)
        self.assertIn('started', message.lower())
        mock_thread.assert_called_once()

    def test_start_download_invalid_model(self):
        """Test starting download with invalid model ID."""
        success, message = model_manager.start_download('nonexistent-model')

        self.assertFalse(success)
        self.assertIn('not found', message.lower())

    @patch('backend.model_manager.check_system_resources')
    def test_start_download_insufficient_resources(self, mock_check_resources):
        """Test starting download when resources are insufficient."""
        mock_check_resources.return_value = (False, ['Low disk space'])

        success, message = model_manager.start_download('tinyllama-1.1b-chat-v1.0.Q4_K_M')

        self.assertFalse(success)
        self.assertIn('Cannot download', message)

    def test_start_download_already_downloaded(self):
        """Test starting download when model is already downloaded."""
        # Create a dummy file
        model_id = 'tinyllama-1.1b-chat-v1.0.Q4_K_M'
        filepath = os.path.join(self.temp_dir, f'{model_id}.gguf')
        with open(filepath, 'w') as f:
            f.write('dummy')

        success, message = model_manager.start_download(model_id)

        self.assertFalse(success)
        self.assertIn('already downloaded', message.lower())

    def test_get_local_models(self):
        """Test getting local models."""
        # Create dummy model files
        with open(os.path.join(self.temp_dir, 'model1.gguf'), 'w') as f:
            f.write('dummy1')
        with open(os.path.join(self.temp_dir, 'model2.gguf'), 'w') as f:
            f.write('dummy2')

        models = model_manager.get_local_models()

        self.assertEqual(len(models), 2)
        self.assertTrue(all(m['filename'].endswith('.gguf') for m in models))

    def test_check_system_resources_sufficient(self):
        """Test resource check with sufficient resources."""
        with patch('backend.model_manager.shutil.disk_usage') as mock_disk, \
             patch('backend.model_manager.psutil.virtual_memory') as mock_ram:
            mock_disk.return_value = MagicMock(free=100 * 1024**3)  # 100GB
            mock_ram.return_value = MagicMock(available=16 * 1024**3)  # 16GB

            model = {
                'size_bytes': 2 * 1024**3,  # 2GB
                'ram_required': 4
            }

            can_download, warnings = model_manager.check_system_resources(model)

            self.assertTrue(can_download)

    def test_check_system_resources_insufficient_disk(self):
        """Test resource check with insufficient disk space."""
        with patch('backend.model_manager.shutil.disk_usage') as mock_disk, \
             patch('backend.model_manager.psutil.virtual_memory') as mock_ram:
            mock_disk.return_value = MagicMock(free=1 * 1024**3)  # 1GB
            mock_ram.return_value = MagicMock(available=16 * 1024**3)

            model = {
                'size_bytes': 5 * 1024**3,  # 5GB
                'ram_required': 4
            }

            can_download, warnings = model_manager.check_system_resources(model)

            self.assertFalse(can_download)
            self.assertTrue(any('disk' in w.lower() for w in warnings))


class TestModelSecurity(unittest.TestCase):
    """Test security features of model manager."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_models_dir = model_manager.MODELS_DIR
        model_manager.MODELS_DIR = self.temp_dir

    def tearDown(self):
        model_manager.MODELS_DIR = self.original_models_dir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_is_safe_model_path_valid(self):
        """Test path validation with valid path."""
        valid_path = os.path.join(self.temp_dir, 'model.gguf')
        self.assertTrue(model_manager.is_safe_model_path(valid_path))

    def test_is_safe_model_path_traversal(self):
        """Test path validation prevents path traversal."""
        traversal_path = os.path.join(self.temp_dir, '..', '..', 'etc', 'passwd')
        self.assertFalse(model_manager.is_safe_model_path(traversal_path))

    def test_is_safe_model_path_empty(self):
        """Test path validation with empty path."""
        self.assertFalse(model_manager.is_safe_model_path(''))
        self.assertFalse(model_manager.is_safe_model_path(None))

    def test_is_safe_model_path_models_dir_itself(self):
        """Test that models directory itself cannot be deleted."""
        self.assertFalse(model_manager.is_safe_model_path(self.temp_dir))

    def test_delete_model_success(self):
        """Test deleting a model successfully."""
        model_path = os.path.join(self.temp_dir, 'test.gguf')
        with open(model_path, 'w') as f:
            f.write('dummy')

        result = model_manager.delete_model(model_path)

        self.assertTrue(result)
        self.assertFalse(os.path.exists(model_path))

    def test_delete_model_unsafe_path(self):
        """Test deleting with unsafe path is prevented."""
        unsafe_path = '/etc/passwd'
        result = model_manager.delete_model(unsafe_path)

        self.assertFalse(result)

    def test_delete_model_nonexistent(self):
        """Test deleting non-existent model."""
        nonexistent = os.path.join(self.temp_dir, 'nonexistent.gguf')
        result = model_manager.delete_model(nonexistent)

        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()