import os
import unittest
from unittest.mock import patch

from scripts.debug_retrieval import prepare_output_path


class TestDebugRetrieval(unittest.TestCase):
    @patch("scripts.debug_retrieval.os.makedirs")
    def test_prepare_output_path_uses_data_directory(self, mock_makedirs):
        """Debug output should be written inside the repo data directory."""
        temp_root = os.path.join("C:\\repo", "example")
        output_path = prepare_output_path(temp_root)

        self.assertEqual(
            output_path,
            os.path.join(temp_root, "data", "retrieval_debug.txt"),
        )
        mock_makedirs.assert_called_once_with(
            os.path.join(temp_root, "data"),
            exist_ok=True,
        )


if __name__ == "__main__":
    unittest.main()
