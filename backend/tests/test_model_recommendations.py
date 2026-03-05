import unittest
from unittest.mock import MagicMock, patch

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


if __name__ == '__main__':
    unittest.main()
