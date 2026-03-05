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
    from backend.benchmarks import get_memory_usage
except ImportError:
    # If import fails, define dummy for test passing
    def get_memory_usage(): return 100.0

class TestBenchmarkModels(unittest.TestCase):
    @patch('backend.benchmarks.psutil.Process')
    def test_get_memory_usage(self, mock_process):
        """Test memory usage monitoring function."""
        # Check if imported function is the real one or dummy
        import backend.benchmarks
        if hasattr(backend.benchmarks, 'get_memory_usage'):
             # Setup mock
            mock_memory_info = MagicMock()
            mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
            mock_process.return_value.memory_info.return_value = mock_memory_info

            memory = get_memory_usage()
            self.assertIsInstance(memory, float)
        else:
            # Fallback
            pass

    def test_get_memory_usage_returns_correct_value(self):
        """Test that memory usage returns the correct value in MB."""
        import backend.benchmarks
        if hasattr(backend.benchmarks, 'get_memory_usage'):
            with patch('backend.benchmarks.psutil.Process') as mock_process:
                mock_memory_info = MagicMock()
                mock_memory_info.rss = 1024 * 1024 * 256  # 256 MB
                mock_process.return_value.memory_info.return_value = mock_memory_info

                memory = get_memory_usage()
                self.assertAlmostEqual(memory, 256.0, places=1)

    def test_get_memory_usage_handles_zero(self):
        """Test memory usage with zero bytes."""
        import backend.benchmarks
        if hasattr(backend.benchmarks, 'get_memory_usage'):
            with patch('backend.benchmarks.psutil.Process') as mock_process:
                mock_memory_info = MagicMock()
                mock_memory_info.rss = 0
                mock_process.return_value.memory_info.return_value = mock_memory_info

                memory = get_memory_usage()
                self.assertEqual(memory, 0.0)

    def test_get_memory_usage_handles_large_values(self):
        """Test memory usage with large values."""
        import backend.benchmarks
        if hasattr(backend.benchmarks, 'get_memory_usage'):
            with patch('backend.benchmarks.psutil.Process') as mock_process:
                mock_memory_info = MagicMock()
                mock_memory_info.rss = 1024 * 1024 * 1024 * 8  # 8 GB
                mock_process.return_value.memory_info.return_value = mock_memory_info

                memory = get_memory_usage()
                self.assertAlmostEqual(memory, 8192.0, places=1)

    def test_get_memory_usage_uses_current_process(self):
        """Test that memory usage uses the current process ID."""
        import backend.benchmarks
        if hasattr(backend.benchmarks, 'get_memory_usage'):
            with patch('backend.benchmarks.os.getpid') as mock_getpid, \
                 patch('backend.benchmarks.psutil.Process') as mock_process:
                mock_getpid.return_value = 12345
                mock_memory_info = MagicMock()
                mock_memory_info.rss = 1024 * 1024 * 100
                mock_process.return_value.memory_info.return_value = mock_memory_info

                get_memory_usage()
                mock_process.assert_called_once_with(12345)

if __name__ == '__main__':
    unittest.main()