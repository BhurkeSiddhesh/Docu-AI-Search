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

        allowed_methods_list = [m.strip() for m in allow_methods.split(",")]
        self.assertIn("GET", allowed_methods_list)
        self.assertIn("POST", allowed_methods_list)
        self.assertNotIn("PUT", allowed_methods_list)
        self.assertNotIn("PATCH", allowed_methods_list)

        # Check headers
        self.assertIn("Content-Type", allow_headers)
        self.assertIn("Authorization", allow_headers)

    def test_cors_configuration_invalid_header(self):
        # Simulate an INVALID preflight request (bad header)
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, X-Custom-Bad-Header",
        }
        response = client.options("/", headers=headers)

        # Note: FastAPI's CORSMiddleware in this configuration returns 400 for
        # disallowed headers. While some CORS implementations return 200 and rely
        # on browser enforcement, this implementation actively rejects invalid headers.
        print(f"Invalid Request Status: {response.status_code}")
        self.assertEqual(response.status_code, 400)
        
        # Even with 400 status, the disallowed header should not be in the allow list
        allow_headers = response.headers.get("access-control-allow-headers", "")
        print(f"Invalid Request - Allowed Headers: {allow_headers}")
        self.assertNotIn("X-Custom-Bad-Header", allow_headers)
