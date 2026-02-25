import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.api import app
import os

class TestCommandInjection(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Verify local request might not be present or needed depending on api.py state
        # If it was used, we would need to override it.
        # But grep showed it's not in api.py, so we assume it's not used.

    def tearDown(self):
        pass

    @patch('backend.database.get_file_by_path')
    @patch('os.path.exists')
    @patch('subprocess.run')
    @patch('os.startfile', create=True)  # create=True for non-Windows envs
    def test_open_file_with_leading_dash_rejected(self, mock_startfile, mock_subprocess, mock_exists, mock_get_file):
        """
        Security Test: Files starting with '-' should be rejected to prevent
        argument injection in subprocess calls (e.g. 'open -n').
        """
        # Setup: Mimic a valid indexed file that happens to start with a dash
        dangerous_filename = "-dangerous.txt"
        mock_get_file.return_value = {'path': dangerous_filename}
        mock_exists.return_value = True

        # Action: Attempt to open it
        # The endpoint expects JSON body with 'path' key based on: file_path = request.get('path', '')
        response = self.client.post("/api/open-file", json={"path": dangerous_filename})

        # Check result
        if response.status_code == 200:
             print("VULNERABILITY REPRODUCED: Endpoint accepted leading dash.")
        else:
             print(f"Endpoint returned {response.status_code}: {response.json()}")

        # For the fix verification, we assert that it FAILS (400)
        self.assertEqual(response.status_code, 400, "Should return 400 Bad Request for filenames starting with '-'")
        self.assertIn("Invalid filename", response.json().get('detail', ''), "Error message should mention invalid filename")

        # Crucially, ensure no system command was executed
        mock_subprocess.assert_not_called()
        mock_startfile.assert_not_called()

    @patch('backend.database.get_file_by_path')
    @patch('os.path.exists')
    @patch('subprocess.run')
    @patch('os.startfile', create=True)
    def test_open_normal_file_allowed(self, mock_startfile, mock_subprocess, mock_exists, mock_get_file):
        """Verify normal files still open correctly."""
        safe_filename = "safe_document.pdf"
        mock_get_file.return_value = {'path': safe_filename}
        mock_exists.return_value = True

        response = self.client.post("/api/open-file", json={"path": safe_filename})

        self.assertEqual(response.status_code, 200)

        # Verify SOMETHING was called (platform dependent)
        import platform
        if platform.system() == 'Windows':
            mock_startfile.assert_called_once()
        else:
            mock_subprocess.assert_called_once()

if __name__ == '__main__':
    unittest.main()
