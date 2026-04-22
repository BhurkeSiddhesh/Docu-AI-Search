"""
backend/tests/test_system_prompts.py
-------------------------------------
Unit tests for the system prompts CRUD and API endpoints.
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.api import app
from backend import database


class TestSystemPromptsAPI(unittest.TestCase):
    """Tests for /api/system-prompts endpoints."""

    def setUp(self):
        self.client = TestClient(app)
        # Reset the database table for each test
        database.init_database()
        # Clear existing prompts to start fresh
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM system_prompts")
        conn.commit()
        conn.close()

    def test_list_system_prompts_empty(self):
        """GET /api/system-prompts returns empty list when no prompts exist."""
        response = self.client.get("/api/system-prompts")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_create_system_prompt(self):
        """POST /api/system-prompts creates a new prompt."""
        payload = {
            "name": "Test Prompt",
            "content": "You are a test assistant.",
            "category": "test",
        }
        response = self.client.post("/api/system-prompts", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("id", data)

    def test_create_and_list(self):
        """Created prompts appear in the list."""
        self.client.post("/api/system-prompts", json={
            "name": "Alpha", "content": "Content A", "category": "general",
        })
        self.client.post("/api/system-prompts", json={
            "name": "Beta", "content": "Content B", "category": "general",
        })

        response = self.client.get("/api/system-prompts")
        self.assertEqual(response.status_code, 200)
        prompts = response.json()
        self.assertEqual(len(prompts), 2)
        names = [p["name"] for p in prompts]
        self.assertIn("Alpha", names)
        self.assertIn("Beta", names)

    def test_delete_system_prompt(self):
        """DELETE /api/system-prompts/{id} removes the prompt."""
        # Create one
        resp = self.client.post("/api/system-prompts", json={
            "name": "Deletable", "content": "Will be deleted", "category": "test",
        })
        prompt_id = resp.json()["id"]

        # Delete it
        del_resp = self.client.delete(f"/api/system-prompts/{prompt_id}")
        self.assertEqual(del_resp.status_code, 200)

        # Verify gone
        list_resp = self.client.get("/api/system-prompts")
        self.assertEqual(list_resp.json(), [])

    def test_delete_nonexistent_prompt(self):
        """DELETE for non-existent ID returns 404."""
        response = self.client.delete("/api/system-prompts/99999")
        self.assertEqual(response.status_code, 404)

    def test_create_missing_fields(self):
        """POST without required fields returns 422."""
        response = self.client.post("/api/system-prompts", json={"name": "Incomplete"})
        self.assertEqual(response.status_code, 422)

    def test_seed_default_prompts(self):
        """seed_default_prompts() populates the table when empty."""
        from backend.system_prompts import seed_default_prompts, get_system_prompts, DEFAULT_PROMPTS

        seed_default_prompts()
        prompts = get_system_prompts()
        self.assertEqual(len(prompts), len(DEFAULT_PROMPTS))

    def test_seed_does_not_duplicate(self):
        """seed_default_prompts() is idempotent — doesn't add duplicates."""
        from backend.system_prompts import seed_default_prompts, get_system_prompts, DEFAULT_PROMPTS

        seed_default_prompts()
        seed_default_prompts()  # Called twice
        prompts = get_system_prompts()
        self.assertEqual(len(prompts), len(DEFAULT_PROMPTS))

    def test_filter_by_category(self):
        """GET with category query param filters results."""
        self.client.post("/api/system-prompts", json={
            "name": "Dev", "content": "Dev content", "category": "development",
        })
        self.client.post("/api/system-prompts", json={
            "name": "Gen", "content": "Gen content", "category": "general",
        })

        response = self.client.get("/api/system-prompts?category=development")
        self.assertEqual(response.status_code, 200)
        prompts = response.json()
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]["name"], "Dev")


if __name__ == "__main__":
    unittest.main()
