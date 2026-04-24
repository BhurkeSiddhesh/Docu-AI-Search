import unittest
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)


class TestCorsConfiguration(unittest.TestCase):

    def test_cors_configuration_valid(self):
        # Simulate a VALID preflight request
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, Authorization",
        }
        response = client.options("/", headers=headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )

        allow_methods = response.headers.get("access-control-allow-methods", "")
        print(f"Valid Request - Allowed Methods: {allow_methods}")

        allow_headers = response.headers.get("access-control-allow-headers", "")
        print(f"Valid Request - Allowed Headers: {allow_headers}")

        # The app uses allow_methods='*', so all standard methods should be permitted
        self.assertIn("GET", allow_methods)
        self.assertIn("POST", allow_methods)

        # Check headers - both should be present since allow_headers='*'
        self.assertIn("Content-Type", allow_headers)
        self.assertIn("Authorization", allow_headers)

    def test_cors_configuration_from_allowed_origins(self):
        # Simulate a preflight from the allowed frontend origin
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        }
        response = client.options("/", headers=headers)

        self.assertEqual(response.status_code, 200)
        # Should reflect the requesting origin back
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )

    def test_cors_configuration_from_unknown_origin(self):
        # Simulate a request from an unknown origin
        headers = {
            "Origin": "http://evil-site.com",
            "Access-Control-Request-Method": "POST",
        }
        response = client.options("/", headers=headers)

        # CORS middleware should not echo back unknown origins
        allow_origin = response.headers.get("access-control-allow-origin", "")
        print(f"Unknown origin response: {allow_origin}")
        self.assertNotEqual(allow_origin, "http://evil-site.com",
                           "Unknown origins should not be reflected back")
