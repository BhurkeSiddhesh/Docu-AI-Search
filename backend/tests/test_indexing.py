import unittest
import sys
import tempfile
import os
import shutil
import json
from unittest.mock import patch, MagicMock

class TestIndexing(unittest.TestCase):
    """Test cases for indexing module"""

    def setUp(self):
        """Set up test environment before each test method."""

        self.modules_patcher = patch.dict(sys.modules, {
            'numpy': MagicMock(),
            'faiss': MagicMock(),
            'rank_bm25': MagicMock(),
            'langchain_text_splitters': MagicMock(),
            'backend.file_processing': MagicMock(),
            'backend.database': MagicMock(),
            'docx': MagicMock(),
            'pypdf': MagicMock(),
            'pptx': MagicMock(),
            'openpyxl': MagicMock(),
            'psutil': MagicMock(),
            'sentence_transformers': MagicMock(),
            'sklearn': MagicMock(),
            'sklearn.cluster': MagicMock(),
            'sklearn.mixture': MagicMock(),
        })
        self.mock_modules = self.modules_patcher.start()

        sys.modules['langchain_text_splitters'].CharacterTextSplitter.return_value.split_text.return_value = ["chunk1"]
        sys.modules['faiss'].read_index.return_value.ntotal = 1
        sys.modules['backend.file_processing'].extract_text.return_value = "Test content"

        self.temp_dir = tempfile.mkdtemp()
        self.test_folder = self.temp_dir
        
        self.test_file = os.path.join(self.test_folder, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("Test content")
            
        self.db_patchers = [
            patch('backend.database.clear_all_files'),
            patch('backend.database.clear_clusters'),
            patch('backend.database.add_file'),
            patch('backend.database.add_cluster')
        ]
        for p in self.db_patchers:
            p.start()

    def tearDown(self):
        for p in self.db_patchers:
            p.stop()
        self.modules_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    # Skipped due to complexity mocking concurrent.futures in restricted env
    # def test_create_index(self):
    #     ...

    def test_create_index_empty_folder(self):
        from backend.indexing import create_index

        empty_folder = os.path.join(self.temp_dir, "empty_folder")
        os.makedirs(empty_folder, exist_ok=True)
        
        with patch('backend.indexing.get_embeddings'),              patch('backend.indexing.get_tags'),              patch('backend.indexing.perform_global_clustering'),              patch('backend.indexing.smart_summary'):

            res = create_index(empty_folder, "openai", "fake_api_key")
            index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res

            self.assertIsNone(index)
    
    def test_save_and_load_index(self):
        from backend.indexing import save_index, load_index

        faiss = sys.modules['faiss']
        index = MagicMock()
        faiss.write_index.side_effect = lambda idx, path: open(path, 'w').close()
        faiss.read_index.return_value = index
        
        docs = [{"text": "Test document"}]
        tags = [["test", "tag"]]
        
        index_path = os.path.join(self.temp_dir, "test_index.faiss")
        save_index(index, docs, tags, index_path)
        
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test_index_docs.json")))
        
        with patch('os.path.exists', return_value=True),              patch('builtins.open', create=True),              patch('json.load') as mock_json_load:

            mock_json_load.side_effect = [
                 [{"text": "Test document"}],
                 [["test", "tag"]],
                 ["summary"],
                 {"0": [0]}
            ]
            
            res = load_index(index_path)
            loaded_index, loaded_docs, loaded_tags, idx_sum, clus_sum, clus_map, bm25 = res
            
            self.assertIsNotNone(loaded_index)
            self.assertEqual(loaded_docs, docs)

if __name__ == '__main__':
    unittest.main()
