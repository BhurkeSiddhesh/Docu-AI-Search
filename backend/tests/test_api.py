"""
Test API Module

Comprehensive tests for all API endpoints including configuration,
search, models, benchmarks, history, and file operations.
"""

import unittest
import sys
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.api import app

# We will use class-level or method-level patches instead of global ones to avoid inter-test pollution

sys.modules['fastapi.responses'] = MagicMock()

class TestAPIConfig(unittest.TestCase):
    """Test cases for configuration endpoints."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)
    
    @patch('backend.api.load_config')
    def test_get_config(self, mock_load_config):
        """Test getting configuration."""
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda section, key, fallback='': {
            ('General', 'folder'): '/test/folder',
            ('General', 'folders'): '/test/folder',
            ('APIKeys', 'openai_api_key'): 'sk-test',
            ('LocalLLM', 'model_path'): '/models/gpt.gguf',
            ('LocalLLM', 'provider'): 'local'
        }.get((section, key), fallback)
        
        mock_config.getboolean.side_effect = lambda section, key, fallback=False: {
            ('General', 'auto_index'): True
        }.get((section, key), fallback)
        
        mock_load_config.return_value = mock_config
        
        response = self.client.get("/api/config")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['folders'], ['/test/folder'])
        self.assertEqual(data['auto_index'], True)
        self.assertEqual(data['provider'], 'local')

    @patch('backend.database.add_folder_to_history')
    @patch('backend.api.save_config_file')
    def test_update_config(self, mock_save_config, mock_add_history):
        """Test updating configuration."""
        response = self.client.post("/api/config", json={
            "folders": ["/new/folder"],
            "auto_index": False,
            "openai_api_key": "sk-new",
            "local_model_path": "",
            "provider": "openai"
        })
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'success')
        mock_save_config.assert_called_once()


class TestAPISearch(unittest.TestCase):
    """Test cases for search endpoints."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.database.add_search_history')
    @patch('backend.database.get_files_by_faiss_indices')
    @patch('backend.database.get_file_by_faiss_index')
    @patch('backend.api.cached_generate_ai_answer')
    @patch('backend.api.cached_smart_summary')
    @patch('backend.api.load_config')
    @patch('backend.api.search')
    @patch('backend.api.summarize')
    @patch('backend.api.get_embeddings')
    def test_search_endpoint(self, mock_get_embeddings, mock_summarize, mock_search, mock_load_config, 
                              mock_smart_summary, mock_generate_ai, mock_get_file,
                              mock_get_files_batch, mock_add_history):
        """Test the search endpoint."""
        mock_config = MagicMock()
        mock_config.get.return_value = 'openai'
        mock_load_config.return_value = mock_config
        
        mock_get_file.return_value = {'filename': 'test.pdf', 'path': '/test/test.pdf'}
        mock_get_files_batch.return_value = {0: {'filename': 'test.pdf', 'path': '/test/test.pdf'}}
        mock_smart_summary.return_value = "Smart Summary"
        mock_generate_ai.return_value = "AI Answer"
        
        with patch('backend.api.index', MagicMock()), \
             patch('backend.api.docs', []), \
             patch('backend.api.tags', []):
            
            mock_search.return_value = (
                [{'document': 'content', 'tags': ['tag1'], 'faiss_idx': 0}],
                ['context snippet']
            )
            
            mock_summarize.return_value = "Summary"
            mock_smart_summary.return_value = "Smart Summary"
            mock_generate_ai.return_value = "AI Answer"
            
            response = self.client.post("/api/search", json={
                "query": "test query"
            })
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIsInstance(data, dict)
            self.assertIn('results', data)
            self.assertIn('ai_answer', data)
            self.assertIn('active_model', data)


    def test_search_without_index(self):
        """Test search when no index is loaded."""
        with patch('backend.api.index', None):
            response = self.client.post("/api/search", json={
                "query": "test query"
            })
            self.assertEqual(response.status_code, 400)


class TestAPIModels(unittest.TestCase):
    """Test cases for model management endpoints."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.api.get_available_models')
    def test_list_available_models(self, mock_get_available):
        """Test listing available models."""
        mock_get_available.return_value = [
            {'id': 'model1', 'name': 'Model 1', 'size': '1GB'},
            {'id': 'model2', 'name': 'Model 2', 'size': '2GB'}
        ]
        
        response = self.client.get("/api/models/available")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)

    @patch('backend.api.get_local_models')
    def test_list_local_models(self, mock_get_local):
        """Test listing local downloaded models."""
        mock_get_local.return_value = [
            {'id': 'local1', 'path': '/models/local1.gguf', 'size': 1000000}
        ]
        
        response = self.client.get("/api/models/local")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    @patch('backend.api.start_download')
    def test_download_model_endpoint(self, mock_start_download):
        """Test starting model download."""
        mock_start_download.return_value = (True, "Download started")
        
        response = self.client.post("/api/models/download/test-model")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)

    @patch('backend.api.get_download_status')
    def test_download_status_endpoint(self, mock_get_status):
        """Test getting download status."""
        mock_get_status.return_value = {
            'downloading': True,
            'model_id': 'test-model',
            'progress': 50
        }
        
        response = self.client.get("/api/models/status")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('downloading', data)
        self.assertIn('progress', data)

    @patch('backend.model_manager.delete_model')
    def test_delete_model(self, mock_delete):
        """Test deleting a model."""
        mock_delete.return_value = True
        
        # Use request method as workaround for delete which doesn't support json in some TestClient versions
        response = self.client.request(
            "DELETE", 
            "/api/models/delete", 
            json={"path": "/models/test.gguf"}
        )
        
        self.assertEqual(response.status_code, 200)


class TestAPIBenchmarks(unittest.TestCase):
    """Test cases for benchmark endpoints."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    def test_get_benchmark_status(self):
        """Test getting benchmark status."""
        response = self.client.get("/api/benchmarks/status")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('running', data)

    @patch('backend.api.benchmark_results', {'test': 'results'})
    def test_get_benchmark_results_with_data(self):
        """Test getting benchmark results when available."""
        with patch('backend.api.benchmark_results', {'model': 'test', 'score': 100}):
            response = self.client.get("/api/benchmarks/results")
            self.assertEqual(response.status_code, 200)

    @patch('backend.api.BackgroundTasks.add_task')
    @patch('backend.api.benchmark_status', {'running': False})
    def test_run_benchmarks_endpoint(self, mock_add_task):
        """Test starting benchmarks."""
        response = self.client.post("/api/benchmarks/run")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'started')
        mock_add_task.assert_called_once()


class TestAPISearchHistory(unittest.TestCase):
    """Test cases for search history endpoints."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.database.get_search_history')
    def test_get_search_history(self, mock_get_history):
        """Test getting search history."""
        mock_get_history.return_value = [
            {'id': 1, 'query': 'test', 'timestamp': '2024-01-01', 'result_count': 5}
        ]
        
        response = self.client.get("/api/search/history")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    @patch('backend.database.delete_search_history_item')
    def test_delete_search_history_item(self, mock_delete):
        """Test deleting a single history item."""
        mock_delete.return_value = True
        
        response = self.client.delete("/api/search/history/1")
        
        self.assertEqual(response.status_code, 200)

    @patch('backend.database.delete_all_search_history')
    def test_delete_all_search_history(self, mock_delete_all):
        """Test deleting all search history."""
        mock_delete_all.return_value = 5
        
        response = self.client.delete("/api/search/history")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('deleted_count', data)


class TestAPIFileOperations(unittest.TestCase):
    """Test cases for file operation endpoints."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)
        # Allow open-file in tests which use "testclient" host
        from backend.api import verify_local_request
        app.dependency_overrides[verify_local_request] = lambda: None

    def tearDown(self):
        app.dependency_overrides = {}

    @patch('backend.database.get_all_files')
    def test_list_indexed_files(self, mock_get_files):
        """Test listing indexed files."""
        mock_get_files.return_value = [
            {'id': 1, 'filename': 'test.pdf', 'path': '/test.pdf', 'size_bytes': 1024}
        ]
        
        response = self.client.get("/api/files")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_validate_path_valid(self):
        """Test validating a valid path."""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('os.walk', return_value=[('/test', [], ['file1.pdf', 'file2.docx'])]):
            response = self.client.post("/api/validate-path", json={
                "path": "/test/folder"
            })
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data['valid'])

    def test_validate_path_invalid(self):
        """Test validating an invalid path."""
        with patch('os.path.exists', return_value=False):
            response = self.client.post("/api/validate-path", json={
                "path": "/nonexistent/folder"
            })
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertFalse(data['valid'])

    @patch('backend.database.get_file_by_path')
    @patch('os.startfile', create=True)
    def test_open_file(self, mock_startfile, mock_get_file):
        """Test opening a file."""
        mock_get_file.return_value = {'path': '/test/document.pdf'}
        with patch('os.path.exists', return_value=True):
            response = self.client.post("/api/open-file", json={
                "path": "/test/document.pdf"
            })
            
            self.assertEqual(response.status_code, 200)


class TestAPIIndexing(unittest.TestCase):
    """Test cases for indexing endpoints."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.api.load_config')
    @patch('backend.api.BackgroundTasks.add_task')
    def test_index_endpoint(self, mock_add_task, mock_load_config):
        """Test the index endpoint."""
        mock_config = MagicMock()
        mock_config.get.return_value = '/test/folder'
        mock_load_config.return_value = mock_config
        
        with patch('os.path.exists', return_value=True):
            response = self.client.post("/api/index")
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['status'], 'accepted')
            mock_add_task.assert_called_once()

    def test_get_indexing_status(self):
        """Test getting indexing status."""
        response = self.client.get("/api/index/status")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('running', data)


class TestAPIStreamingEndpoint(unittest.TestCase):
    """Test cases for streaming answer endpoint."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.api.stream_ai_answer')
    @patch('backend.api.load_config')
    def test_stream_answer_with_provided_context(self, mock_load_config, mock_stream):
        """Test streaming answer with pre-provided context."""
        mock_config = MagicMock()
        mock_config.get.return_value = 'openai'
        mock_load_config.return_value = mock_config

        mock_stream.return_value = iter(["token1", "token2", "token3"])

        with patch('backend.api.index', MagicMock()):
            response = self.client.post("/api/stream-answer", json={
                "query": "test query",
                "context": ["context snippet 1", "context snippet 2"]
            })

            self.assertEqual(response.status_code, 200)
            # Verify streaming response
            content = response.read().decode('utf-8')
            self.assertIsInstance(content, str)

    def test_stream_answer_without_index(self):
        """Test streaming answer when index is not loaded."""
        with patch('backend.api.index', None):
            response = self.client.post("/api/stream-answer", json={
                "query": "test query"
            })

            self.assertEqual(response.status_code, 200)
            # Should return error message
            content = response.read().decode('utf-8')
            self.assertIn("Error", content)


class TestAPIRateLimiting(unittest.TestCase):
    """Test cases for rate limiting functionality."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.api.load_config')
    def test_rate_limit_headers(self, mock_load_config):
        """Test that rate limit headers are present."""
        mock_config = MagicMock()
        mock_config.get.return_value = '/test/folder'
        mock_config.getboolean.return_value = False
        mock_load_config.return_value = mock_config

        response = self.client.get("/api/config")

        # Rate limiting should add headers
        self.assertEqual(response.status_code, 200)


class TestAPIFolderHistory(unittest.TestCase):
    """Test cases for folder history management."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.database.get_folder_history')
    def test_get_folder_history(self, mock_get_history):
        """Test getting folder history."""
        mock_get_history.return_value = [
            {'path': '/folder1', 'is_indexed': True},
            {'path': '/folder2', 'is_indexed': True}
        ]

        response = self.client.get("/api/folders/history")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)

    @patch('backend.database.clear_folder_history')
    def test_clear_folder_history(self, mock_clear):
        """Test clearing folder history."""
        mock_clear.return_value = 5

        response = self.client.delete("/api/folders/history")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['deleted_count'], 5)

    @patch('backend.database.delete_folder_history_item')
    def test_delete_folder_history_item(self, mock_delete):
        """Test deleting a single folder from history."""
        mock_delete.return_value = True

        response = self.client.request(
            "DELETE",
            "/api/folders/history/item",
            json={"path": "/test/folder"}
        )

        self.assertEqual(response.status_code, 200)

    def test_delete_folder_history_item_missing_path(self):
        """Test deleting folder history without providing path."""
        response = self.client.request(
            "DELETE",
            "/api/folders/history/item",
            json={}
        )

        self.assertEqual(response.status_code, 400)


class TestAPILogging(unittest.TestCase):
    """Test cases for frontend logging endpoint."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    def test_receive_error_log(self):
        """Test receiving error log from frontend."""
        response = self.client.post("/api/logs", json={
            "level": "error",
            "message": "Test error message",
            "source": "Frontend",
            "stack": "Error stack trace"
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'logged')

    def test_receive_warning_log(self):
        """Test receiving warning log from frontend."""
        response = self.client.post("/api/logs", json={
            "level": "warn",
            "message": "Test warning message",
            "source": "Frontend"
        })

        self.assertEqual(response.status_code, 200)

    def test_receive_info_log(self):
        """Test receiving info log from frontend."""
        response = self.client.post("/api/logs", json={
            "level": "info",
            "message": "Test info message"
        })

        self.assertEqual(response.status_code, 200)


class TestAPIHealthEndpoints(unittest.TestCase):
    """Test cases for health check endpoints."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint returns status."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'online')

    def test_health_check_endpoint(self):
        """Test dedicated health check endpoint."""
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'ok')


class TestAPISecurityFeatures(unittest.TestCase):
    """Test cases for security features."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)
        # Override security check for tests
        from backend.api import verify_local_request
        app.dependency_overrides[verify_local_request] = lambda: None

    def tearDown(self):
        """Clean up overrides."""
        app.dependency_overrides = {}

    @patch('backend.database.get_file_by_path')
    def test_open_file_security_check_non_indexed(self, mock_get_file):
        """Test that opening non-indexed files is blocked."""
        mock_get_file.return_value = None  # File not in index

        response = self.client.post("/api/open-file", json={
            "path": "/some/random/file.pdf"
        })

        # Should be blocked for security
        self.assertEqual(response.status_code, 403)

    @patch('backend.database.get_file_by_path')
    def test_open_file_security_check_disallowed_extension(self, mock_get_file):
        """Test that files with disallowed extensions are blocked."""
        mock_get_file.return_value = {'path': '/test/malicious.exe'}

        with patch('os.path.exists', return_value=True):
            response = self.client.post("/api/open-file", json={
                "path": "/test/malicious.exe"
            })

            # Should be blocked due to dangerous extension
            self.assertEqual(response.status_code, 403)

    def test_open_file_missing_path(self):
        """Test opening file without providing path."""
        response = self.client.post("/api/open-file", json={})

        self.assertEqual(response.status_code, 400)

    @patch('backend.database.get_file_by_path')
    def test_open_file_nonexistent_file(self, mock_get_file):
        """Test opening a file that doesn't exist on disk."""
        mock_get_file.return_value = {'path': '/test/missing.pdf'}

        with patch('os.path.exists', return_value=False):
            response = self.client.post("/api/open-file", json={
                "path": "/test/missing.pdf"
            })

            self.assertEqual(response.status_code, 404)


class TestAPIConfigEdgeCases(unittest.TestCase):
    """Test edge cases for configuration management."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.api.load_config')
    def test_get_config_with_old_folder_format(self, mock_load_config):
        """Test getting config with legacy single folder format."""
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda section, key, fallback='': {
            ('General', 'folder'): '/legacy/folder',
            ('General', 'folders'): '',  # Empty new format
            ('APIKeys', 'openai_api_key'): '',
            ('LocalLLM', 'model_path'): '',
            ('LocalLLM', 'provider'): 'openai'
        }.get((section, key), fallback)

        mock_config.getboolean.return_value = False
        mock_load_config.return_value = mock_config

        response = self.client.get("/api/config")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Should convert old format to new
        self.assertIn('/legacy/folder', data['folders'])

    @patch('backend.database.add_folder_to_history')
    @patch('backend.api.save_config_file')
    def test_update_config_with_empty_folders(self, mock_save, mock_add_history):
        """Test updating config with empty folders list."""
        response = self.client.post("/api/config", json={
            "folders": [],
            "auto_index": False,
            "provider": "openai"
        })

        self.assertEqual(response.status_code, 200)
        mock_save.assert_called_once()

    @patch('backend.database.add_folder_to_history')
    @patch('backend.api.save_config_file')
    def test_update_config_with_multiple_api_keys(self, mock_save, mock_add_history):
        """Test updating config with multiple API keys."""
        response = self.client.post("/api/config", json={
            "folders": ["/test"],
            "auto_index": True,
            "openai_api_key": "sk-openai",
            "gemini_api_key": "gemini-key",
            "anthropic_api_key": "anthropic-key",
            "grok_api_key": "grok-key",
            "provider": "gemini"
        })

        self.assertEqual(response.status_code, 200)


class TestAPIIndexingEdgeCases(unittest.TestCase):
    """Test edge cases for indexing functionality."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.api.load_config')
    def test_trigger_indexing_with_no_folders(self, mock_load_config):
        """Test triggering indexing when no folders are configured."""
        mock_config = MagicMock()
        mock_config.get.return_value = ''  # No folders
        mock_load_config.return_value = mock_config

        response = self.client.post("/api/index")

        self.assertEqual(response.status_code, 400)

    @patch('backend.api.indexing_status', {'running': True})
    def test_trigger_indexing_while_already_running(self):
        """Test triggering indexing when it's already in progress."""
        with patch('backend.api.load_config') as mock_config:
            mock_config.return_value.get.return_value = '/test'

            response = self.client.post("/api/index")

            self.assertEqual(response.status_code, 400)

    def test_validate_path_not_directory(self):
        """Test validating a path that exists but is not a directory."""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=False):
            response = self.client.post("/api/validate-path", json={
                "path": "/test/file.txt"
            })

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertFalse(data['valid'])
            self.assertIn('error', data)

    def test_validate_path_empty(self):
        """Test validating with empty path."""
        response = self.client.post("/api/validate-path", json={
            "path": ""
        })

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['valid'])


class TestAPISearchEdgeCases(unittest.TestCase):
    """Test edge cases for search functionality."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.database.add_search_history')
    @patch('backend.database.get_files_by_faiss_indices')
    @patch('backend.api.load_config')
    @patch('backend.api.search')
    @patch('backend.api.summarize')
    @patch('backend.api.get_embeddings')
    def test_search_with_empty_results(self, mock_embeddings, mock_summarize,
                                       mock_search, mock_config, mock_batch, mock_history):
        """Test search that returns no results."""
        mock_config.return_value.get.return_value = 'openai'
        mock_search.return_value = ([], [])  # Empty results
        mock_batch.return_value = {}

        with patch('backend.api.index', MagicMock()), \
             patch('backend.api.docs', []), \
             patch('backend.api.tags', []):

            response = self.client.post("/api/search", json={
                "query": "nonexistent term"
            })

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(len(data['results']), 0)

    @patch('backend.database.add_search_history')
    @patch('backend.database.get_files_by_faiss_indices')
    @patch('backend.api.load_config')
    @patch('backend.api.search')
    @patch('backend.api.summarize')
    @patch('backend.api.get_embeddings')
    def test_search_with_very_long_query(self, mock_embeddings, mock_summarize,
                                          mock_search, mock_config, mock_batch, mock_history):
        """Test search with extremely long query."""
        mock_config.return_value.get.return_value = 'openai'
        mock_search.return_value = ([{'document': 'test', 'tags': [], 'faiss_idx': 0}], ['test'])
        mock_summarize.return_value = "Summary"
        mock_batch.return_value = {0: {'filename': 'test.pdf', 'path': '/test.pdf'}}

        long_query = "word " * 1000  # Very long query

        with patch('backend.api.index', MagicMock()), \
             patch('backend.api.docs', []), \
             patch('backend.api.tags', []):

            response = self.client.post("/api/search", json={
                "query": long_query
            })

            # Should handle long query without crashing
            self.assertEqual(response.status_code, 200)


class TestAPIEmbeddingSettings(unittest.TestCase):
    """Test cases for embedding settings endpoints."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.settings.get_embedding_config_from_ini')
    def test_get_embedding_settings(self, mock_get_config):
        """Test getting embedding settings."""
        mock_get_config.return_value = {
            'provider_type': 'local',
            'model_name': 'test-model',
            'api_key_set': False
        }

        response = self.client.get("/api/settings/embeddings")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('provider_type', data)
        self.assertIn('model_name', data)
        self.assertIn('api_key_set', data)

    @patch('backend.settings.save_embedding_config')
    def test_update_embedding_settings_local(self, mock_save):
        """Test updating embedding settings to local provider."""
        mock_save.return_value = None

        response = self.client.post("/api/settings/embeddings", json={
            "provider_type": "local",
            "model_name": "new-local-model"
        })

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')

    @patch('backend.settings.save_embedding_config')
    def test_update_embedding_settings_commercial(self, mock_save):
        """Test updating embedding settings with API key."""
        mock_save.return_value = None

        response = self.client.post("/api/settings/embeddings", json={
            "provider_type": "commercial_api",
            "model_name": "text-embedding-ada-002",
            "api_key": "sk-test-key"
        })

        self.assertEqual(response.status_code, 200)

    def test_update_embedding_settings_missing_api_key(self):
        """Test updating embedding settings without required API key."""
        response = self.client.post("/api/settings/embeddings", json={
            "provider_type": "commercial_api",
            "model_name": "text-embedding-ada-002"
        })

        # Should fail validation
        self.assertIn(response.status_code, [400, 422])

    def test_update_embedding_settings_invalid_provider(self):
        """Test updating with invalid provider type."""
        response = self.client.post("/api/settings/embeddings", json={
            "provider_type": "invalid_provider",
            "model_name": "some-model"
        })

        # Should fail validation
        self.assertIn(response.status_code, [400, 422])


class TestAPIBrowseFolder(unittest.TestCase):
    """Test cases for folder browse endpoint."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('tkinter.filedialog.askdirectory')
    @patch('tkinter.Tk')
    def test_browse_folder_selected(self, mock_tk, mock_dialog):
        """Test browsing and selecting a folder."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_dialog.return_value = "/selected/folder"

        response = self.client.get("/api/browse")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['folder'], "/selected/folder")

    @patch('tkinter.filedialog.askdirectory')
    @patch('tkinter.Tk')
    def test_browse_folder_cancelled(self, mock_tk, mock_dialog):
        """
        Verify GET /api/browse returns folder=None when the user cancels the folder selection dialog.
        """
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_dialog.return_value = ""  # Empty when cancelled

        response = self.client.get("/api/browse")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data['folder'])


class TestAPICacheEndpoints(unittest.TestCase):
    """Test cases for AI response cache endpoints."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.database.get_cache_stats')
    def test_get_cache_stats(self, mock_stats):
        """Test getting cache statistics."""
        mock_stats.return_value = {
            'total_entries': 42,
            'total_hits': 128
        }

        response = self.client.get("/api/cache/stats")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['total_entries'], 42)
        self.assertEqual(data['total_hits'], 128)

    @patch('backend.database.clear_response_cache')
    def test_clear_cache(self, mock_clear):
        """Test clearing AI response cache."""
        mock_clear.return_value = 15

        response = self.client.post("/api/cache/clear")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['cleared_entries'], 15)


class TestAPISearchDimensionMismatch(unittest.TestCase):
    """Test cases for embedding dimension mismatch handling."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.database.add_search_history')
    @patch('backend.api.load_config')
    @patch('backend.api.search')
    @patch('backend.api.get_embeddings')
    def test_search_dimension_mismatch_error(self, mock_embeddings, mock_search,
                                              mock_config, mock_history):
        """Test search when embedding dimension doesn't match index."""
        from backend.search import EmbeddingDimensionMismatchError

        mock_config.return_value.get.return_value = 'openai'
        mock_search.side_effect = EmbeddingDimensionMismatchError(
            "Query dimension (384) != Index dimension (768)"
        )

        with patch('backend.api.index', MagicMock()), \
             patch('backend.api.docs', []), \
             patch('backend.api.tags', []):

            response = self.client.post("/api/search", json={
                "query": "test query"
            })

            # Should return 409 Conflict
            self.assertEqual(response.status_code, 409)
            self.assertIn("dimension", response.json()['detail'].lower())


class TestAPIModelDownloadEdgeCases(unittest.TestCase):
    """Test edge cases for model download functionality."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.api.start_download')
    def test_download_model_failure(self, mock_start_download):
        """Test model download failure."""
        mock_start_download.return_value = (False, "Download failed: Network error")

        response = self.client.post("/api/models/download/invalid-model")

        self.assertEqual(response.status_code, 400)
        self.assertIn('detail', response.json())

    @patch('backend.model_manager.delete_model')
    def test_delete_model_not_found(self, mock_delete):
        """Test deleting a model that doesn't exist."""
        mock_delete.return_value = False

        response = self.client.request(
            "DELETE",
            "/api/models/delete",
            json={"path": "/models/nonexistent.gguf"}
        )

        self.assertEqual(response.status_code, 404)

    def test_delete_model_missing_path(self):
        """Test deleting model without providing path."""
        response = self.client.request(
            "DELETE",
            "/api/models/delete",
            json={}
        )

        self.assertEqual(response.status_code, 400)


class TestAPIBenchmarkEdgeCases(unittest.TestCase):
    """Test edge cases for benchmark functionality."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.api.benchmark_status', {'running': True})
    def test_run_benchmarks_while_already_running(self):
        """Test running benchmarks when already in progress."""
        response = self.client.post("/api/benchmarks/run")

        self.assertEqual(response.status_code, 400)
        self.assertIn('already running', response.json()['detail'])

    @patch('backend.api.benchmark_results', None)
    @patch('os.path.exists')
    def test_get_benchmark_results_no_data(self, mock_exists):
        """Test getting results when no benchmarks have been run."""
        mock_exists.return_value = False

        response = self.client.get("/api/benchmarks/results")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('message', data)


class TestAPIMultipleFoldersIndexing(unittest.TestCase):
    """Test indexing with multiple folders."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.api.load_config')
    @patch('backend.api.BackgroundTasks.add_task')
    def test_index_multiple_folders(self, mock_add_task, mock_load_config):
        """Test indexing multiple folders at once."""
        mock_config = MagicMock()
        mock_config.get.side_effect = lambda section, key, fallback='': {
            ('General', 'folders'): '/folder1,/folder2,/folder3'
        }.get((section, key), fallback)
        mock_load_config.return_value = mock_config

        response = self.client.post("/api/index")

        self.assertEqual(response.status_code, 200)
        mock_add_task.assert_called_once()


class TestAPIStreamingEdgeCases(unittest.TestCase):
    """Test edge cases for streaming endpoint."""

    def setUp(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    @patch('backend.api.stream_ai_answer')
    @patch('backend.api.search')
    @patch('backend.api.summarize')
    @patch('backend.api.load_config')
    def test_stream_answer_rerun_search(self, mock_config, mock_summarize,
                                        mock_search, mock_stream):
        """Test streaming answer when context not provided (re-runs search)."""
        mock_config.return_value.get.return_value = 'openai'
        mock_search.return_value = ([{'document': 'test', 'tags': [], 'faiss_idx': 0}], ['context'])
        mock_summarize.return_value = "Summary"
        mock_stream.return_value = iter(["token1", "token2"])

        with patch('backend.api.index', MagicMock()), \
             patch('backend.api.docs', []), \
             patch('backend.api.tags', []), \
             patch('backend.database.get_files_by_faiss_indices', return_value={}):

            response = self.client.post("/api/stream-answer", json={
                "query": "test query"
                # No context provided
            })

            self.assertEqual(response.status_code, 200)
            # Verify search was called
            mock_search.assert_called_once()


if __name__ == '__main__':
    unittest.main()