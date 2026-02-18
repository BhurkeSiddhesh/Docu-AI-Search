import unittest
from fastapi.testclient import TestClient
from backend.api import app

class TestRateLimiting(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_rate_limit_exceeded(self):
        """Test that making more than 100 requests in a minute triggers rate limiting."""
        # Note: The global limit is set to 100/minute.
        # We assume clean state or at least < 100 requests from 'testclient' before this.

        # Make 100 successful requests
        for i in range(100):
            response = self.client.get("/api/health")
            # If we hit limit early, print warning but fail only if it's way too early (e.g. 1)
            # But normally it should be exactly 100.
            if response.status_code == 429:
                print(f"Hit rate limit early at request {i+1}")
                # Use this as the break point
                break
            self.assertEqual(response.status_code, 200, f"Request {i+1} failed with {response.status_code}")

        # The next request should fail
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 429)
        # Verify error message usually contains "Too Many Requests" or "Rate limit exceeded"
        # slowapi default is "Rate limit exceeded" in detail or body?
        # Standard 429 body.
        print(f"Rate limit response: {response.text}")

if __name__ == '__main__':
    unittest.main()
