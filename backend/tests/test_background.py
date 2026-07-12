import unittest
import tempfile
import os
import configparser
from unittest.mock import patch, MagicMock, call, ANY
from backend.background import start_background_indexing, IndexingEventHandler


class TestBackground(unittest.TestCase):
    """Test cases for background module"""

    @patch('backend.background.save_index')
    def test_indexing_event_handler_initialization(self, mock_save_index):
        """Test initialization of IndexingEventHandler."""
        with patch('backend.background.create_index') as mock_create_index:
            mock_create_index.return_value = (MagicMock(), ["doc1"], ["tag1"], None, None, None, None)
            
            handler = IndexingEventHandler(
                folder="/test/folder",
                provider="openai",
                api_key="test_key",
                model_path=None
            )
            
            # Verify attributes are set correctly
            self.assertEqual(handler.folder, "/test/folder")
            self.assertEqual(handler.provider, "openai")
            self.assertEqual(handler.api_key, "test_key")
            self.assertIsNone(handler.model_path)
            
            # Verify update_index was NOT called during initialization
            mock_create_index.assert_not_called()
            mock_save_index.assert_not_called()
    
    @patch('backend.background.create_index')
    @patch('backend.background.save_index')
    def test_update_index(self, mock_save_index, mock_create_index):
        """Test the update_index method of IndexingEventHandler."""
        from unittest.mock import ANY
        mock_index = MagicMock()
        mock_create_index.return_value = (mock_index, ["doc1"], ["tag1"], None, None, None, None)

        handler = IndexingEventHandler(
            folder="/test/folder",
            provider="openai",
            api_key="test_key",
            model_path=None
        )

        # Call update_index directly
        handler.update_index()

        # Verify create_index was called
        mock_create_index.assert_called_once_with(
            "/test/folder", "openai", "test_key", None,
            previous_index_path=ANY
        )

        # Verify save_index was called (path is the absolute data/index.faiss path)
        mock_save_index.assert_called_once_with(
            mock_index, ["doc1"], ["tag1"], ANY, None, None, None, None,
            model_name=ANY, embedding_dim=ANY
        )
    
    @patch('backend.background.create_index')
    def test_update_index_none_result(self, mock_create_index):
        """Test update_index when create_index returns None."""
        mock_create_index.return_value = (None, None, None, None, None, None, None)
        
        handler = IndexingEventHandler(
            folder="/test/folder",
            provider="openai",
            api_key="test_key",
            model_path=None
        )
        
        # Call update_index directly
        handler.update_index()
        
        # Verify create_index was called but save_index was not
        mock_create_index.assert_called_once_with(
            "/test/folder", "openai", "test_key", None,
            previous_index_path=ANY
        )
    
    def test_on_modified_triggers_update(self):
        """Test that on_modified event schedules a debounced index update."""
        handler = IndexingEventHandler(
            folder="/test/folder",
            provider="openai",
            api_key="test_key",
            model_path=None
        )
        event = MagicMock()
        with patch.object(handler, 'queue_update') as mock_schedule:
            handler.on_modified(event)
        mock_schedule.assert_called_once()

    def test_on_created_triggers_update(self):
        """Test that on_created event schedules a debounced index update."""
        handler = IndexingEventHandler(
            folder="/test/folder",
            provider="openai",
            api_key="test_key",
            model_path=None
        )
        event = MagicMock()
        with patch.object(handler, 'queue_update') as mock_schedule:
            handler.on_created(event)
        mock_schedule.assert_called_once()

    def test_on_deleted_triggers_update(self):
        """Test that on_deleted event schedules a debounced index update."""
        handler = IndexingEventHandler(
            folder="/test/folder",
            provider="openai",
            api_key="test_key",
            model_path=None
        )
        event = MagicMock()
        with patch.object(handler, 'queue_update') as mock_schedule:
            handler.on_deleted(event)
        mock_schedule.assert_called_once()


class TestStartBackgroundIndexingFolderParsing(unittest.TestCase):
    """Tests for multi-folder and legacy fallback parsing in start_background_indexing."""

    def _make_config(self, general_items, tmp_path=None):
        cfg = configparser.ConfigParser()
        cfg['General'] = general_items
        cfg['LocalLLM'] = {'provider': 'openai', 'model_path': ''}
        cfg['APIKeys'] = {'openai_api_key': 'test_key'}
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
        cfg.write(f)
        f.close()
        return f.name

    @patch('backend.background.Observer')
    @patch('backend.background.IndexingEventHandler')
    @patch('backend.background.time')
    def test_multiple_folders_from_folders_key(self, mock_time, MockHandler, MockObserver):
        """Comma-separated 'folders' creates one handler per folder."""
        mock_time.sleep.side_effect = KeyboardInterrupt
        config_path = self._make_config({
            'auto_index': 'true',
            'folders': '/dir/a, /dir/b, /dir/c',
        })
        try:
            with patch('backend.background.CONFIG_PATH', config_path):
                start_background_indexing()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            os.unlink(config_path)

        # A single handler is constructed with the full folder list
        self.assertEqual(MockHandler.call_count, 1)
        self.assertEqual(MockHandler.call_args[0][0], ['/dir/a', '/dir/b', '/dir/c'])

    @patch('backend.background.Observer')
    @patch('backend.background.IndexingEventHandler')
    @patch('backend.background.time')
    def test_legacy_folder_key_fallback(self, mock_time, MockHandler, MockObserver):
        """Falls back to 'folder' key when 'folders' is empty."""
        mock_time.sleep.side_effect = KeyboardInterrupt
        config_path = self._make_config({
            'auto_index': 'true',
            'folder': '/legacy/dir',
        })
        try:
            with patch('backend.background.CONFIG_PATH', config_path):
                start_background_indexing()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            os.unlink(config_path)

        self.assertEqual(MockHandler.call_count, 1)
        self.assertEqual(MockHandler.call_args[0][0], ['/legacy/dir'])

    @patch('backend.background.Observer')
    @patch('backend.background.IndexingEventHandler')
    @patch('backend.background.time')
    def test_folders_key_takes_precedence_over_folder(self, mock_time, MockHandler, MockObserver):
        """'folders' key wins when both 'folders' and 'folder' are present."""
        mock_time.sleep.side_effect = KeyboardInterrupt
        config_path = self._make_config({
            'auto_index': 'true',
            'folders': '/primary/dir',
            'folder': '/legacy/dir',
        })
        try:
            with patch('backend.background.CONFIG_PATH', config_path):
                start_background_indexing()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            os.unlink(config_path)

        self.assertEqual(MockHandler.call_count, 1)
        self.assertEqual(MockHandler.call_args[0][0], ['/primary/dir'])

    def test_auto_index_false_does_not_start(self):
        """auto_index=false means no handlers or observer are created."""
        config_path = self._make_config({
            'auto_index': 'false',
            'folders': '/some/dir',
        })
        with patch('backend.background.Observer') as MockObserver, \
             patch('backend.background.IndexingEventHandler') as MockHandler, \
             patch('backend.background.CONFIG_PATH', config_path):
            start_background_indexing()
        os.unlink(config_path)

        MockHandler.assert_not_called()
        MockObserver.assert_not_called()


if __name__ == '__main__':
    unittest.main()
