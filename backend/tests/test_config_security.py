import unittest
import os
import tempfile
import configparser
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.api import app, mask_key, verify_local_request

class TestConfigSecurity(unittest.TestCase):
    def setUp(self):
        # Patch dependencies to avoid side effects
        self.mock_load_config = patch('backend.api.load_config').start()
        self.mock_save_config = patch('backend.api.save_config_file').start()

        # Setup mock config
        self.config_mock = MagicMock()
        self.config_mock.get.side_effect = lambda section, key, fallback='': {
            ('General', 'folder'): '/test/folder',
            ('General', 'folders'): '/test/folder',
            ('General', 'auto_index'): 'False',
            ('APIKeys', 'openai_api_key'): 'sk-proj-1234567890abcdef',
            ('APIKeys', 'gemini_api_key'): 'AIzaSyDn1234567890',
            ('APIKeys', 'anthropic_api_key'): '',
            ('LocalLLM', 'model_path'): '',
            ('LocalLLM', 'provider'): 'openai'
        }.get((section, key), fallback)

        self.config_mock.getboolean.side_effect = lambda section, key, fallback=False: {
            ('General', 'auto_index'): False
        }.get((section, key), fallback)

        self.mock_load_config.return_value = self.config_mock

        # Override dependency to allow TestClient requests
        app.dependency_overrides[verify_local_request] = lambda: None

        # TestClient
        self.client = TestClient(app)

    def tearDown(self):
        # Clear overrides
        app.dependency_overrides = {}
        patch.stopall()

    def test_mask_key_helper(self):
        """Test the mask_key helper function directly."""
        self.assertEqual(mask_key(""), "")
        self.assertEqual(mask_key("abc"), "") # Less than 5 chars -> empty (as per implementation)
        self.assertEqual(mask_key("1234"), "")
        self.assertEqual(mask_key("12345"), "********")
        self.assertEqual(mask_key("sk-proj-1234567890abcdef"), "********")

    def test_get_config_returns_masked_keys(self):
        """Test that GET /api/config returns masked API keys."""
        response = self.client.get("/api/config")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check masking
        self.assertEqual(data['openai_api_key'], "********")
        self.assertEqual(data['gemini_api_key'], "********")
        self.assertEqual(data['anthropic_api_key'], "") # Was empty in mock

        # Check other fields preserved
        self.assertEqual(data['folders'], ['/test/folder'])

    def test_update_config_preserves_masked_keys(self):
        """Test that POST /api/config preserves keys if they are masked."""
        # Payload with masked keys
        payload = {
            "folders": ["/new/folder"],
            "auto_index": True,
            "openai_api_key": "********", # Should be preserved
            "gemini_api_key": "AIzaNewKey", # Should be updated
            "anthropic_api_key": "",
            "local_model_path": "",
            "provider": "openai"
        }

        response = self.client.post("/api/config", json=payload)
        self.assertEqual(response.status_code, 200)

        # Verify what was saved
        # Get the config object passed to save_config_file
        args, _ = self.mock_save_config.call_args
        saved_config = args[0]

        # OpenAI key was "********", should be preserved as 'sk-proj-1234567890abcdef' (from setUp mock)
        self.assertEqual(saved_config['APIKeys']['openai_api_key'], 'sk-proj-1234567890abcdef')

        # Gemini key was changed
        self.assertEqual(saved_config['APIKeys']['gemini_api_key'], 'AIzaNewKey')

    def test_verify_local_request_blocks_remote(self):
        """Test that remote IPs are blocked."""
        from fastapi import Request, HTTPException

        # Create a mock request
        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "192.168.1.50"

        # verify_local_request is async
        import asyncio
        loop = asyncio.new_event_loop()

        # Test remote IP
        with self.assertRaises(HTTPException) as cm:
             loop.run_until_complete(verify_local_request(mock_request))
        self.assertEqual(cm.exception.status_code, 403)

        # Test local IP
        mock_request.client.host = "127.0.0.1"
        try:
            loop.run_until_complete(verify_local_request(mock_request))
        except HTTPException:
            self.fail("verify_local_request raised HTTPException for 127.0.0.1")

        loop.close()

if __name__ == '__main__':
    unittest.main()
