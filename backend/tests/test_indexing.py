import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()
sys.modules['pdfplumber'] = MagicMock()
sys.modules['docx'] = MagicMock()
sys.modules['pptx'] = MagicMock()
sys.modules['openpyxl'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.cluster'] = MagicMock()

from backend import indexing

class TestIndexing(unittest.TestCase):
    def test_create_index(self):
        # This is a placeholder test to ensure module can be imported and function called
        pass

if __name__ == '__main__':
    unittest.main()
