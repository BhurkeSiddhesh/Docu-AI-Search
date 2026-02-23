import unittest
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
from backend.indexing import create_index, save_index, load_index
from backend import database

class TestIndexing(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.test_dir, "test_docs")
        os.makedirs(self.test_folder)
        
        # Create dummy file
        with open(os.path.join(self.test_folder, "test.txt"), "w") as f:
            f.write("This is a test document.")
            
        # Patch database
        self.db_path = os.path.join(self.test_dir, 'test_metadata.db')
        self.patcher = patch('backend.database.DATABASE_PATH', self.db_path)
        self.patcher.start()
        database.init_database()
            
    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)

    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.CharacterTextSplitter')
    def test_create_index(self, mock_splitter, mock_get_embeddings, mock_extract):
        # Mock dependencies
        mock_extract.return_value = "This is a test document."
        
        # Mock embeddings model
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = [[0.1] * 384]
        mock_get_embeddings.return_value = mock_model
        
        # Mock splitter
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text.return_value = ["This is a test document."]
        mock_splitter.return_value = mock_splitter_instance
        
        # Run indexing
        index, docs, tags, summ_idx, summ_docs, cluster_map, bm25 = create_index([self.test_folder], "openai", "fake_api_key")
        
        self.assertIsNotNone(index)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]['text'], "This is a test document.")

    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.get_embeddings')
    def test_create_index_empty_folder(self, mock_get_embeddings, mock_extract):
        empty_folder = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_folder)
        
        res = create_index([empty_folder], "openai", "fake_api_key")
        
        # Should return empty result
        self.assertIsNone(res[0])

class TestIndexingMultipleFolders(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.folder1 = os.path.join(self.test_dir, "folder1")
        self.folder2 = os.path.join(self.test_dir, "folder2")
        os.makedirs(self.folder1)
        os.makedirs(self.folder2)

        with open(os.path.join(self.folder1, "doc1.txt"), "w") as f:
            f.write("Doc 1 content")
        with open(os.path.join(self.folder2, "doc2.txt"), "w") as f:
            f.write("Doc 2 content")
            
        # Patch database
        self.db_path = os.path.join(self.test_dir, 'test_metadata.db')
        self.patcher = patch('backend.database.DATABASE_PATH', self.db_path)
        self.patcher.start()
        database.init_database()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)

    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.CharacterTextSplitter')
    def test_create_index_multiple_folders(self, mock_splitter, mock_get_embeddings, mock_extract):
        mock_extract.side_effect = ["Doc 1 content", "Doc 2 content"]
        
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = [[0.1] * 384]
        mock_get_embeddings.return_value = mock_model
        
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text.return_value = ["Chunk"]
        mock_splitter.return_value = mock_splitter_instance
        
        res = create_index(
            [self.folder1, self.folder2],
            "openai", "fake_key"
        )

        index, docs, tags, _, _, _, _ = res
        self.assertIsNotNone(index)
        # Should have at least 2 docs (1 from each file)
        self.assertGreaterEqual(len(docs), 2)

    @patch('backend.indexing.extract_text')
    def test_create_index_with_progress_callback(self, mock_extract):
        mock_extract.return_value = "Content"
        progress_callback = MagicMock()
        
        # Patch get_embeddings and splitter to avoid real work
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = [[0.1] * 384]

        with patch('backend.indexing.get_embeddings', return_value=mock_model),              patch('backend.indexing.CharacterTextSplitter') as MockSplitter:

            MockSplitter.return_value.split_text.return_value = ["Content"]

            create_index([self.folder1], "openai", "fake_key", progress_callback=progress_callback)

            # Callback should have been called
            self.assertTrue(progress_callback.called)

    def test_create_index_nonexistent_folder(self):
        res = create_index(
            ["/nonexistent/folder/path"],
            "openai", "fake_key"
        )
        # Should return None/empty
        self.assertIsNone(res[0])

    @patch('backend.indexing.extract_text')
    @patch('backend.indexing.get_embeddings')
    @patch('backend.indexing.CharacterTextSplitter')
    def test_create_index_string_folder_path(self, mock_splitter, mock_get_embeddings, mock_extract):
        # Should handle string input (convert to list)
        mock_extract.return_value = "Content"
        
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = [[0.1] * 384]
        mock_get_embeddings.return_value = mock_model
        
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_text.return_value = ["Chunk"]
        mock_splitter.return_value = mock_splitter_instance
        
        res = create_index(
            self.folder1, # Pass string instead of list
            "openai", "fake_key"
        )
        
        self.assertIsNotNone(res[0])

if __name__ == '__main__':
    unittest.main()
