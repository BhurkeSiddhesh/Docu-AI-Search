import unittest
import sys
import os
import tempfile
import shutil
import json
from unittest.mock import MagicMock

# Mock missing dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()
sys.modules['docx'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['pptx'] = MagicMock()
sys.modules['openpyxl'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.cluster'] = MagicMock()
sys.modules['sklearn.mixture'] = MagicMock()

# Now import the target module
from backend.indexing import save_index, load_index

class TestSecurityFixVerification(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = os.path.join(self.temp_dir, "test")
        self.index_path = self.base_path + ".index"

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_files_securely(self):
        # Mock inputs
        index_chunks = MagicMock()
        index_chunks.write_index = MagicMock()

        all_chunks = [{'text': 'chunk1', 'filepath': 'f1', 'faiss_idx': 0, 'file_id': None}]
        tags = ['tag1']
        cluster_summaries = ['summary1']
        cluster_map = {0: [0]}
        bm25 = "mock_bm25"

        # Call save_index
        print(f"Calling save_index with path: {self.index_path}")
        save_index(index_chunks, all_chunks, tags, self.index_path,
                   index_summaries=MagicMock(),
                   cluster_summaries=cluster_summaries,
                   cluster_map=cluster_map,
                   bm25=bm25)

        # Check for files
        files = os.listdir(self.temp_dir)
        print("Files created:", files)

        # Verify JSONs exist (Post-fix check)
        self.assertTrue(os.path.exists(self.base_path + '_docs.json'), "Docs JSON not found")
        self.assertTrue(os.path.exists(self.base_path + '_tags.json'), "Tags JSON not found")
        self.assertTrue(os.path.exists(self.base_path + '_summaries.json'), "Summaries JSON not found")
        self.assertTrue(os.path.exists(self.base_path + '_cluster_map.json'), "Cluster map JSON not found")

        # Verify pickles DO NOT exist
        self.assertFalse(os.path.exists(self.base_path + '_docs.pkl'), "Docs pickle found (SECURITY FAIL)")
        self.assertFalse(os.path.exists(self.base_path + '_tags.pkl'), "Tags pickle found (SECURITY FAIL)")
        self.assertFalse(os.path.exists(self.base_path + '_summaries.pkl'), "Summaries pickle found (SECURITY FAIL)")
        self.assertFalse(os.path.exists(self.base_path + '_cluster_map.pkl'), "Cluster map pickle found (SECURITY FAIL)")
        self.assertFalse(os.path.exists(self.base_path + '_bm25.pkl'), "BM25 pickle found (SECURITY FAIL)")

if __name__ == '__main__':
    unittest.main()
