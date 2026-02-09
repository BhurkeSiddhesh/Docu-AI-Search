import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException, Request
from backend.api import app, verify_local_request
import os

class TestAPISecurity(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Ensure no overrides are active by default for these tests
        app.dependency_overrides = {}

    def tearDown(self):
        app.dependency_overrides = {}

    @patch('backend.database.get_file_by_path')
    @patch('os.path.exists')
    def test_open_file_remote_access_denied_integration(self, mock_exists, mock_get_file):
        """Test that remote requests (default TestClient) are denied by the endpoint."""
        mock_get_file.return_value = {'path': '/test/doc.pdf'}
        mock_exists.return_value = True

        # TestClient uses 'testclient' host by default, which should be denied
        response = self.client.post("/api/open-file", json={"path": "/test/doc.pdf"})

        self.assertEqual(response.status_code, 403)
        self.assertIn("Access denied", response.json()['detail'])

    def test_verify_local_request_logic(self):
        """Unit test for the security dependency function."""

        # Case 1: Localhost (IPv4)
        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "127.0.0.1"
        try:
            verify_local_request(mock_request)
        except HTTPException:
            self.fail("verify_local_request raised HTTPException for 127.0.0.1")

        # Case 2: Localhost (IPv6)
        mock_request.client.host = "::1"
        try:
            verify_local_request(mock_request)
        except HTTPException:
            self.fail("verify_local_request raised HTTPException for ::1")

        # Case 3: Localhost (hostname)
        mock_request.client.host = "localhost"
        try:
            verify_local_request(mock_request)
        except HTTPException:
            self.fail("verify_local_request raised HTTPException for localhost")

        # Case 4: Remote IP
        mock_request.client.host = "192.168.1.1"
        with self.assertRaises(HTTPException) as cm:
            verify_local_request(mock_request)
        self.assertEqual(cm.exception.status_code, 403)

        # Case 5: TestClient default
        mock_request.client.host = "testclient"
        with self.assertRaises(HTTPException) as cm:
            verify_local_request(mock_request)
        self.assertEqual(cm.exception.status_code, 403)

    @patch('backend.database.get_file_by_path')
    @patch('os.path.exists')
    @patch('subprocess.run')
    @patch('os.startfile', create=True)
    def test_endpoint_works_when_authorized(self, mock_startfile, mock_subprocess, mock_exists, mock_get_file):
        """Verify the endpoint works when dependency is overridden (authorized)."""
        mock_get_file.return_value = {'path': '/test/doc.pdf'}
        mock_exists.return_value = True

        # Override to allow access
        app.dependency_overrides[verify_local_request] = lambda: None

        response = self.client.post("/api/open-file", json={"path": "/test/doc.pdf"})
        self.assertEqual(response.status_code, 200)

    @patch('backend.database.get_file_by_path')
    @patch('os.path.exists')
    def test_open_file_invalid_extension(self, mock_exists, mock_get_file):
        """Test that invalid file extensions are blocked even if authorized."""
        mock_get_file.return_value = {'path': '/test/malware.exe'}
        mock_exists.return_value = True

        # Override to allow access (simulate local request)
        app.dependency_overrides[verify_local_request] = lambda: None

        response = self.client.post("/api/open-file", json={"path": "/test/malware.exe"})

        self.assertEqual(response.status_code, 403)
        self.assertIn("File type not allowed", response.json()['detail'])

if __name__ == '__main__':
    unittest.main()
