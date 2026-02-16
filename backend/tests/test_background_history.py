import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.api import app
import backend.database as database

class TestSearchBackground(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch('backend.api.search')
    @patch('backend.api.get_embeddings')
    @patch('backend.api.summarize')
    @patch('backend.api.load_config')
    @patch('backend.database.add_search_history')
    @patch('fastapi.BackgroundTasks.add_task')
    def test_search_uses_background_task(self, mock_add_task, mock_add_history, mock_load_config, mock_summarize, mock_embeddings, mock_search):
        """Verify that add_search_history is dispatched as a background task."""

        # Setup mocks
        mock_config = MagicMock()
        mock_config.get.return_value = 'openai'
        mock_load_config.return_value = mock_config

        mock_search.return_value = ([], []) # Empty results
        mock_summarize.return_value = "summary"

        # Mock index to avoid "Index not loaded" error
        with patch('backend.api.index', MagicMock()), \
             patch('backend.api.docs', []), \
             patch('backend.api.tags', []):

            response = self.client.post("/api/search", json={"query": "background test"})

            self.assertEqual(response.status_code, 200)

            # Verify that BackgroundTasks.add_task was called with database.add_search_history
            # Note: TestClient executes background tasks, so mock_add_history is also called,
            # but we want to ensure it was added via add_task, not called directly.

            # Check if add_task was called
            # Since TestClient might optimize or bypass, we might need to check if the framework did it.
            # But here we are patching `fastapi.BackgroundTasks.add_task`.
            # Wait, `BackgroundTasks` is instantiated inside the endpoint?
            # No, it's a dependency injection.
            # FastAPIs dependency injection creates a BackgroundTasks instance.
            # Patching `fastapi.BackgroundTasks.add_task` (method) should work if we patch it where it is used?
            # Or patch the class method?

            # Actually, `fastapi.BackgroundTasks` is imported in `backend.api`.
            # But the dependency injection uses the class from `fastapi`.

            # Let's see if `mock_add_task` captured the call.
            # Since `background_tasks.add_task(...)` is called on an instance,
            # patching the class method `add_task` should work for all instances.

            # Verify at least one call to add_task with database.add_search_history as the first arg
            found = False
            for call in mock_add_task.call_args_list:
                args, _ = call
                if args[0] == database.add_search_history:
                    found = True
                    break

            if not found:
                # Debug info
                print("\nCalls to add_task:", mock_add_task.call_args_list)

            self.assertTrue(found, "database.add_search_history was not added to background tasks")

if __name__ == '__main__':
    unittest.main()
