import unittest
import os
import tempfile
import shutil
from backend.model_manager import delete_model, is_safe_model_path as is_safe_path

# Mocking MODELS_DIR for the module would be tricky with unittest parallel execution.
# However, delete_model and is_safe_path take explicit paths or use global MODELS_DIR.
# Ideally, we should patch backend.model_manager.MODELS_DIR

from unittest.mock import patch

class TestSecurityFix(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.patcher = patch("backend.model_manager.MODELS_DIR", self.temp_dir)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_is_safe_path_valid(self):
        file_path = os.path.join(self.temp_dir, "file.txt")
        # MODELS_DIR is mocked to self.temp_dir, so this should be safe
        self.assertTrue(is_safe_path(file_path))

    def test_is_safe_path_traversal(self):
        # Create a sibling directory
        sibling_dir = self.temp_dir + "_sibling"
        os.makedirs(sibling_dir, exist_ok=True)
        try:
            # Using .. to go out of MODELS_DIR (mocked to self.temp_dir)
            traversal_path = os.path.join(self.temp_dir, "..", os.path.basename(sibling_dir), "secret.txt")

            # Should be False because it resolves to sibling_dir which is not under MODELS_DIR
            self.assertFalse(is_safe_path(traversal_path))
        finally:
            if os.path.exists(sibling_dir):
                shutil.rmtree(sibling_dir)

    def test_delete_model_valid(self):
        # Create a file inside temp_models_dir
        model_path = os.path.join(self.temp_dir, "test_model.gguf")
        with open(model_path, "w") as f:
            f.write("data")

        self.assertTrue(delete_model(model_path))
        self.assertFalse(os.path.exists(model_path))

    def test_delete_model_outside(self):
        # Create a file outside
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            outside_path = tmp.name
            tmp.write(b"data")

        try:
            # Attempt to delete explicit outside path
            self.assertFalse(delete_model(outside_path))
            self.assertTrue(os.path.exists(outside_path))
        finally:
            if os.path.exists(outside_path):
                os.remove(outside_path)

    def test_delete_model_traversal(self):
        # Create a file outside
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            outside_path = tmp.name
            tmp.write(b"data")

        try:
            # Construct path using traversal from temp_models_dir
            rel_path = os.path.relpath(outside_path, self.temp_dir)
            traversal_path = os.path.join(self.temp_dir, rel_path)

            # But delete_model should reject it
            self.assertFalse(delete_model(traversal_path))
            self.assertTrue(os.path.exists(outside_path))
        finally:
            if os.path.exists(outside_path):
                os.remove(outside_path)

    def test_delete_model_nonexistent(self):
        path = os.path.join(self.temp_dir, "fake.gguf")
        self.assertFalse(delete_model(path))

if __name__ == '__main__':
    unittest.main()
