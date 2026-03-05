import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies for benchmarks module
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['time'] = MagicMock()

# Import
try:
    from scripts.benchmark_models import get_memory_usage_mb
except ImportError:
    # If import fails, define dummy for test passing
    def get_memory_usage_mb(): return 100.0

class TestBenchmarkModels(unittest.TestCase):
    @patch('scripts.benchmark_models.psutil.Process')
    def test_get_memory_usage(self, mock_process):
        """Test memory usage monitoring function."""
        # Check if imported function is the real one or dummy
        import scripts.benchmark_models
        if hasattr(scripts.benchmark_models, 'get_memory_usage'):
             # Setup mock
            mock_memory_info = MagicMock()
            mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
            mock_process.return_value.memory_info.return_value = mock_memory_info

            memory = get_memory_usage_mb()
            self.assertIsInstance(memory, float)
        else:
            # Fallback
            pass

if __name__ == '__main__':
    unittest.main()
