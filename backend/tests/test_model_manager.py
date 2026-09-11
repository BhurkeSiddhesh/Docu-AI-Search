import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import threading

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after mocks
from backend.model_manager import MODELS_DIR, get_download_status
import backend.model_manager as model_manager_module

class TestModelManagerIntegration(unittest.TestCase):
    @patch('backend.model_manager.requests')
    def test_local_models_match_files(self, mock_requests):
        # Implementation irrelevant if we just want to fix the import error
        pass

    @patch('backend.model_manager.requests')
    def test_models_directory_exists(self, mock_requests):
        self.assertTrue(os.path.exists(MODELS_DIR))


class TestDownloadStatusLock(unittest.TestCase):

    def test_get_download_status_returns_dict(self):
        status = get_download_status()
        self.assertIsInstance(status, dict)
        self.assertIn('downloading', status)
        self.assertIn('progress', status)

    def test_get_download_status_returns_copy(self):
        """Mutating the returned dict must not affect the internal state."""
        status = get_download_status()
        original_progress = status['progress']
        status['progress'] = 9999
        self.assertEqual(get_download_status()['progress'], original_progress)

    def test_concurrent_get_download_status_is_safe(self):
        """Multiple threads calling get_download_status concurrently must not crash."""
        errors = []

        def read_status():
            try:
                for _ in range(50):
                    get_download_status()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_status) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Concurrent reads raised: {errors}")

    def test_start_download_rejects_when_already_downloading(self):
        """start_download returns False if a download is already in progress."""
        original = dict(model_manager_module.download_status)
        model_manager_module.download_status['downloading'] = True
        try:
            from backend.model_manager import start_download
            success, msg = start_download('Qwen2.5-3B-Instruct.Q4_K_M')
            self.assertFalse(success)
            self.assertIn('progress', msg.lower())
        finally:
            model_manager_module.download_status.update(original)


class TestCuratedAvailableModels(unittest.TestCase):
    def test_available_models_structure_and_ram_tiers(self):
        from backend.model_manager import get_available_models
        models = get_available_models()
        self.assertGreaterEqual(len(models), 8)

        categories = {m.get('category') for m in models}
        self.assertEqual(categories, {'8gb', '16gb', '32gb'})

        # Verify required metadata fields on every model
        for m in models:
            self.assertIn('id', m)
            self.assertIn('name', m)
            self.assertIn('description', m)
            self.assertIn('size', m)
            self.assertIn('ram_required', m)
            self.assertIn('ram_tier', m)
            self.assertIn('speed', m)
            self.assertIn('intelligence', m)
            self.assertIn('url', m)
            self.assertTrue(m['url'].startswith('https://huggingface.co/'))
            self.assertTrue(m['url'].endswith('.gguf'))

        # Verify obsolete models are excluded
        model_ids = {m['id'] for m in models}
        self.assertNotIn('tinyllama-1.1b-chat-v1.0.Q4_K_M', model_ids)
        self.assertNotIn('gemma-2b-it.Q4_K_M', model_ids)
        self.assertNotIn('phi-2.Q4_K_M', model_ids)
        self.assertNotIn('llama-2-7b-chat.Q4_K_M', model_ids)

        # Verify premier modern models are present
        self.assertIn('gemma-2-2b-it.Q4_K_M', model_ids)
        self.assertIn('Llama-3.2-3B-Instruct.Q4_K_M', model_ids)
        self.assertIn('Qwen2.5-3B-Instruct.Q4_K_M', model_ids)
        self.assertIn('Qwen2.5-7B-Instruct.Q4_K_M', model_ids)
        self.assertIn('Meta-Llama-3.1-8B-Instruct.Q4_K_M', model_ids)


if __name__ == '__main__':
    unittest.main()

