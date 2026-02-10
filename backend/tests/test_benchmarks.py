import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['psutil'] = MagicMock()

from backend.benchmarks import get_memory_usage

class TestBenchmarkModels(unittest.TestCase):
    @patch('backend.benchmarks.psutil.Process')
    def test_get_memory_usage(self, mock_process):
        """Test memory usage monitoring function."""
        # Mock memory_info
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        memory = get_memory_usage()
        
        self.assertIsInstance(memory, float)
        self.assertGreater(memory, 0)

if __name__ == '__main__':
    unittest.main()
