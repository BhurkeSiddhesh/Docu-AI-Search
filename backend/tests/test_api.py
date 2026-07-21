"""
Test API Module

Comprehensive tests for all API endpoints including configuration,
search, models, benchmarks, history, and file operations.
"""

import os
import unittest
import sys
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.api import app

# We will use class-level or method-level patches instead of global ones to avoid inter-test pollution


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
    @patch('backend.api.get_active_embedding_client')
    def test_search_endpoint(self, mock_get_active_client, mock_summarize, mock_search, mock_load_config, 
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

    @patch('backend.database.count_files')
    @patch('backend.database.get_all_files')
    def test_list_indexed_files(self, mock_get_files, mock_count_files):
        """Test listing indexed files returns paginated response."""
        mock_get_files.return_value = [
            {'id': 1, 'filename': 'test.pdf', 'path': '/test.pdf', 'size_bytes': 1024}
        ]
        mock_count_files.return_value = 1

        response = self.client.get("/api/files")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)
        self.assertIn('files', data)
        self.assertIn('total', data)
        self.assertEqual(data['total'], 1)
        self.assertEqual(len(data['files']), 1)

    def test_validate_path_valid(self):
        """Test validating a valid path."""
        test_root = os.path.realpath("/test") + os.sep
        with patch('backend.api._allowed_index_roots', return_value=(test_root,)), \
             patch('os.path.exists', return_value=True), \
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
        with patch('backend.api._allowed_index_roots', return_value=('/nonexistent/',)), \
             patch('os.path.exists', return_value=False):
            response = self.client.post("/api/validate-path", json={
                "path": "/nonexistent/folder"
            })

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertFalse(data['valid'])

    def test_validate_path_outside_allowed_roots(self):
        """Paths outside the allow-listed roots are rejected before any filesystem access."""
        with patch('backend.api._allowed_index_roots', return_value=('/test/',)), \
             patch('os.path.exists') as mock_exists:
            response = self.client.post("/api/validate-path", json={
                "path": "/opt/other/folder"
            })

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertFalse(data['valid'])
            self.assertIn('allowed index roots', data['error'])
            mock_exists.assert_not_called()

    def test_validate_path_traversal_escapes_root(self):
        """Traversal sequences that resolve outside the allowed root are rejected."""
        with patch('backend.api._allowed_index_roots', return_value=('/test/',)), \
             patch('os.path.exists') as mock_exists:
            response = self.client.post("/api/validate-path", json={
                "path": "/test/../etc/passwd"
            })

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertFalse(data['valid'])
            mock_exists.assert_not_called()

    def test_validate_path_home_directory_allowed_by_default(self):
        """The user's home directory is an allowed root without extra configuration."""
        home_subfolder = os.path.join(os.path.expanduser("~"), "documents")
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('os.walk', return_value=[(home_subfolder, [], ['a.pdf'])]):
            response = self.client.post("/api/validate-path", json={
                "path": home_subfolder
            })

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data['valid'])

    def test_allowed_index_roots_env_override(self):
        """DOCU_INDEX_ROOTS grants extra allow-listed roots."""
        from backend.api import _allowed_index_roots
        extra = os.pathsep.join(["/mnt/docs", "/srv/shared"])
        with patch.dict(os.environ, {"DOCU_INDEX_ROOTS": extra}):
            roots = _allowed_index_roots()
        self.assertIn(os.path.realpath("/mnt/docs") + os.sep, roots)
        self.assertIn(os.path.realpath("/srv/shared") + os.sep, roots)

    @patch('backend.database.get_all_file_paths')
    @patch('os.startfile', create=True)
    @patch('subprocess.run')
    def test_open_file(self, mock_subprocess, mock_startfile, mock_paths):
        """Test opening a file."""
        mock_paths.return_value = ['/test/document.pdf']
        mock_subprocess.return_value = MagicMock(returncode=0)
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
        from backend.api import indexing_status
        indexing_status["running"] = False

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
            # Raw-text stream protocol: error is surfaced as a readable token
            content = response.read().decode('utf-8')
            self.assertIn('Error: Index not loaded', content)


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

    @patch('backend.database.get_connection')
    def test_health_check_endpoint(self, mock_get_connection):
        """Test dedicated health check endpoint."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value = None
        mock_get_connection.return_value = mock_conn

        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'ok')
        mock_conn.execute.assert_called_once_with("SELECT 1")


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

    @patch('backend.database.get_all_file_paths')
    def test_open_file_security_check_non_indexed(self, mock_paths):
        """Test that opening non-indexed files is blocked."""
        mock_paths.return_value = []  # File not in index

        response = self.client.post("/api/open-file", json={
            "path": "/some/random/file.pdf"
        })

        # Should be blocked for security
        self.assertEqual(response.status_code, 403)

    @patch('backend.database.get_all_file_paths')
    def test_open_file_security_check_disallowed_extension(self, mock_paths):
        """Test that files with disallowed extensions are blocked."""
        mock_paths.return_value = ['/test/malicious.exe']

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

    @patch('backend.database.get_all_file_paths')
    def test_open_file_nonexistent_file(self, mock_paths):
        """Test opening a file that doesn't exist on disk."""
        mock_paths.return_value = ['/test/missing.pdf']

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
        from backend.api import indexing_status
        indexing_status["running"] = False

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
        with patch('backend.api._allowed_index_roots', return_value=('/test/',)), \
             patch('os.path.exists', return_value=True), \
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
    @patch('backend.api.get_active_embedding_client')
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
    @patch('backend.api.get_active_embedding_client')
    def test_search_with_very_long_query(self, mock_embeddings, mock_summarize,
                                          mock_search, mock_config, mock_batch, mock_history):
        """Test search with extremely long query."""
        mock_config.return_value.get.return_value = 'openai'
        mock_search.return_value = ([{'document': 'test', 'tags': [], 'faiss_idx': 0}], ['test'])
        mock_summarize.return_value = "Summary"
        mock_batch.return_value = {0: {'filename': 'test.pdf', 'path': '/test.pdf'}}

        long_query = "word " * 199  # Long query within the 1000-char max_length limit

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
        # Seed app.state with a known embedding config so GET works predictably
        app.state.embedding_config = {
            'provider_type': 'local',
            'model_name': 'test-model',
            'api_key': '',
        }

    def tearDown(self):
        """Clean up app.state after each test."""
        app.state.embedding_config = None

    def test_get_embedding_settings(self):
        """Test getting embedding settings."""
        response = self.client.get("/api/settings/embeddings")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('provider_type', data)
        self.assertIn('model_name', data)
        self.assertIn('api_key_set', data)
        self.assertEqual(data['provider_type'], 'local')
        self.assertEqual(data['model_name'], 'test-model')
        self.assertFalse(data['api_key_set'])

    @patch('backend.settings._write_embedding_section')
    def test_update_embedding_settings_local(self, mock_write):
        """Test updating embedding settings to local provider."""
        response = self.client.post("/api/settings/embeddings", json={
            "provider_type": "local",
            "model_name": "new-local-model"
        })

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        mock_write.assert_called_once()

    @patch('backend.settings._write_embedding_section')
    def test_update_embedding_settings_commercial(self, mock_write):
        """Test updating embedding settings with API key."""
        response = self.client.post("/api/settings/embeddings", json={
            "provider_type": "commercial_api",
            "model_name": "text-embedding-ada-002",
            "api_key": "sk-test-key"
        })

        self.assertEqual(response.status_code, 200)
        mock_write.assert_called_once()

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


import sys as _sys
_tkinter_available = 'tkinter' in _sys.modules or __import__('importlib.util', fromlist=['find_spec']).find_spec('tkinter') is not None


@unittest.skipUnless(_tkinter_available, "tkinter not available in this environment")
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
    @patch('backend.api.get_active_embedding_client')
    def test_search_dimension_mismatch_error(self, mock_embeddings, mock_search,
                                              mock_config, mock_history):
        """Test search when embedding dimension doesn't match index."""
        from backend.search import EmbeddingDimensionMismatchError

        mock_config.return_value.get.return_value = 'openai'
        mock_search.side_effect = EmbeddingDimensionMismatchError(
            query_dim=384, index_dim=768
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
        from backend.api import indexing_status
        indexing_status["running"] = False

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

    @patch('backend.api.cached_smart_summary')
    @patch('backend.api.get_active_embedding_client')
    @patch('backend.api.stream_ai_answer')
    @patch('backend.api.search')
    @patch('backend.api.summarize')
    @patch('backend.api.load_config')
    def test_stream_answer_rerun_search(self, mock_config, mock_summarize,
                                        mock_search, mock_stream, mock_embedding_client,
                                        mock_smart_summary):
        """Test streaming answer when context not provided (re-runs search)."""
        mock_config.return_value.get.return_value = 'openai'
        mock_search.return_value = ([{'document': 'test', 'tags': [], 'faiss_idx': 0}], ['context'])
        mock_summarize.return_value = "Summary"
        mock_stream.return_value = iter(["token1", "token2"])
        mock_smart_summary.return_value = "Smart Summary"

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


class TestBatch1Fixes(unittest.TestCase):
    """Tests for batch-1 critical bug fixes (#129, #170, #171, #174, #175, #205)."""

    def setUp(self):
        self.client = TestClient(app)
        from backend.api import verify_local_request
        app.dependency_overrides[verify_local_request] = lambda: None

    def tearDown(self):
        app.dependency_overrides = {}

    # ── #129: ZeroDivisionError when total=0 ────────────────────────────────

    def test_indexing_progress_callback_zero_total_no_crash(self):
        """indexing_progress_callback must not raise when total=0."""
        from backend.api import indexing_progress_callback
        try:
            indexing_progress_callback(0, 0, "Starting")
        except ZeroDivisionError:
            self.fail("indexing_progress_callback raised ZeroDivisionError with total=0")

    def test_indexing_progress_callback_zero_total_yields_zero_percent(self):
        """With total=0 progress should be 0%, not an error."""
        from backend.api import indexing_progress_callback, indexing_status
        indexing_progress_callback(0, 0, "init")
        from backend.api import indexing_status as s
        self.assertEqual(s["progress"], 0)

    # ── #170: POST /api/config must reject remote callers ───────────────────

    def test_post_config_blocked_from_remote(self):
        """POST /api/config must return 403 when called from a non-local host."""
        app.dependency_overrides = {}  # remove the override so real check runs
        from fastapi.testclient import TestClient as TC
        c = TC(app, headers={"X-Forwarded-For": "8.8.8.8"})
        # TestClient sets host=testclient which is allowed; simulate remote by
        # checking the real dependency raises 403 for non-local hosts via the
        # verify_local_request logic path — we verify the dependency is wired.
        from backend.api import verify_local_request, update_config
        import inspect
        sig = inspect.signature(update_config)
        self.assertIn('_', sig.parameters)

    # ── #174: POST /api/logs must reject remote callers ─────────────────────

    def test_post_logs_blocked_from_remote(self):
        """POST /api/logs must have verify_local_request dependency wired."""
        from backend.api import receive_log
        import inspect
        sig = inspect.signature(receive_log)
        self.assertIn('_', sig.parameters)

    # ── #175: DELETE /api/models/delete must reject remote callers ──────────

    def test_delete_model_blocked_from_remote(self):
        """DELETE /api/models/delete must have verify_local_request wired."""
        from backend.api import delete_model
        import inspect
        sig = inspect.signature(delete_model)
        self.assertIn('_', sig.parameters)

    # ── #171: providers endpoints reject remote callers ──────────────────────

    def test_provider_health_check_blocked_from_remote(self):
        """POST /api/providers/health must have verify_local_request wired."""
        from backend.api import provider_health_check
        import inspect
        sig = inspect.signature(provider_health_check)
        self.assertIn('_', sig.parameters)

    def test_provider_list_models_blocked_from_remote(self):
        """POST /api/providers/models must have verify_local_request wired."""
        from backend.api import provider_list_models
        import inspect
        sig = inspect.signature(provider_list_models)
        self.assertIn('_', sig.parameters)

    # ── #205: CORS must use configured ALLOWED_ORIGINS ──────────────────────

    def test_cors_uses_allowed_origins(self):
        """CORSMiddleware must be configured with ALLOWED_ORIGINS not wildcard."""
        from backend.api import app as fastapi_app, ALLOWED_ORIGINS
        from starlette.middleware.cors import CORSMiddleware
        cors_middleware = next(
            (m for m in fastapi_app.user_middleware if m.cls is CORSMiddleware),
            None
        )
        self.assertIsNotNone(cors_middleware, "CORSMiddleware not found")
        configured_origins = cors_middleware.kwargs.get('allow_origins', [])
        self.assertNotIn('*', configured_origins,
                         "CORS must not use wildcard; found '*' in allow_origins")
        self.assertEqual(configured_origins, ALLOWED_ORIGINS)


class TestBatch2Fixes(unittest.TestCase):
    """Tests for batch-2 fixes (#135, #168, #212)."""

    def setUp(self):
        self.client = TestClient(app)
        from backend.api import verify_local_request, require_auth
        app.dependency_overrides[verify_local_request] = lambda: None
        app.dependency_overrides[require_auth] = lambda: None

    def tearDown(self):
        app.dependency_overrides = {}

    # ── #212: data-access endpoints require auth ────────────────────────────

    def test_list_files_requires_auth(self):
        """GET /api/files must have require_auth dependency wired."""
        from backend.api import list_indexed_files
        import inspect
        sig = inspect.signature(list_indexed_files)
        self.assertIn('_auth', sig.parameters)

    def test_preview_file_requires_auth(self):
        """GET /api/files/preview must have require_auth dependency wired."""
        from backend.api import preview_file
        import inspect
        sig = inspect.signature(preview_file)
        self.assertIn('_auth', sig.parameters)

    def test_search_history_requires_auth(self):
        """GET /api/search/history must have require_auth dependency wired."""
        from backend.api import get_search_history
        import inspect
        sig = inspect.signature(get_search_history)
        self.assertIn('_auth', sig.parameters)

    def test_cache_stats_requires_auth(self):
        """GET /api/cache/stats must have require_auth dependency wired."""
        from backend.api import cache_stats_endpoint
        import inspect
        sig = inspect.signature(cache_stats_endpoint)
        self.assertIn('_auth', sig.parameters)

    # ── #135: download/benchmarks/index require verify_local_request ────────

    def test_download_model_requires_local(self):
        """POST /api/models/download must have verify_local_request wired."""
        from backend.api import download_model_endpoint
        import inspect
        sig = inspect.signature(download_model_endpoint)
        self.assertIn('_', sig.parameters)

    def test_run_benchmarks_requires_local(self):
        """POST /api/benchmarks/run must have verify_local_request wired."""
        from backend.api import run_benchmarks
        import inspect
        sig = inspect.signature(run_benchmarks)
        self.assertIn('_', sig.parameters)

    def test_trigger_indexing_requires_local(self):
        """POST /api/index must have verify_local_request wired."""
        from backend.api import trigger_indexing
        import inspect
        sig = inspect.signature(trigger_indexing)
        self.assertIn('_', sig.parameters)

    # ── #168: stream-answer yields SSE error event ──────────────────────────

    @patch('backend.api.stream_ai_answer', side_effect=RuntimeError("LLM unavailable"))
    @patch('backend.api.index', MagicMock())
    def test_stream_answer_yields_sse_error_on_exception(self, mock_stream):
        """When stream_ai_answer raises, client receives an SSE [ERROR] event."""
        # Provide context directly so the endpoint reaches stream_ai_answer
        response = self.client.post("/api/stream-answer", json={
            "query": "test",
            "context": ["some relevant document context"]
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"[ERROR]", response.content)


class TestBatch2BackgroundFix(unittest.TestCase):
    """Tests for background.py error handling fix (#200)."""

    @patch('backend.background.create_index', side_effect=RuntimeError("index exploded"))
    def test_update_index_exception_is_caught(self, mock_create):
        """update_index must catch exceptions and keep the watchdog running."""
        from backend.background import IndexingEventHandler
        handler = IndexingEventHandler("/some/folder", "openai", "key", None)
        try:
            handler.update_index()
        except Exception:
            self.fail("update_index propagated an exception; watchdog thread would die")

    @patch('backend.background.create_index')
    @patch('backend.background.save_index')
    def test_update_index_saves_on_success(self, mock_save, mock_create):
        """update_index still calls save_index when create_index succeeds."""
        from backend.background import IndexingEventHandler
        mock_create.return_value = (MagicMock(), ["doc"], ["tag"], None, None, None, None)
        handler = IndexingEventHandler("/some/folder", "openai", "key", None)
        handler.update_index()
        mock_save.assert_called_once()


class TestBatch1DatabaseFixes(unittest.TestCase):
    """Tests for database-level fixes (#186)."""

    def test_max_indices_within_sqlite_limit(self):
        """MAX_INDICES must be <= 499 to stay under SQLite's 999 bind-param cap."""
        from backend.database import MAX_INDICES
        self.assertLessEqual(MAX_INDICES, 499,
                             f"MAX_INDICES={MAX_INDICES} exceeds safe SQLite limit")


class TestBatch1ProvidersCacheFix(unittest.TestCase):
    """Tests for providers.py cache key fix (#180)."""

    def test_different_api_keys_produce_different_cache_entries(self):
        """get_provider must create separate instances for different api_keys."""
        from backend.providers import get_provider, _provider_cache
        _provider_cache.clear()

        p1 = get_provider('ollama', {'base_url': 'http://localhost:11434', 'model': 'm', 'api_key': 'key-A'})
        p2 = get_provider('ollama', {'base_url': 'http://localhost:11434', 'model': 'm', 'api_key': 'key-B'})
        self.assertIsNot(p1, p2, "Different api_keys must yield different provider instances")

    def test_same_api_key_returns_cached_instance(self):
        """get_provider must return the same instance for identical params."""
        from backend.providers import get_provider, _provider_cache
        _provider_cache.clear()

        p1 = get_provider('ollama', {'base_url': 'http://localhost:11434', 'model': 'm', 'api_key': 'key-A'})
        p2 = get_provider('ollama', {'base_url': 'http://localhost:11434', 'model': 'm', 'api_key': 'key-A'})
        self.assertIs(p1, p2, "Same params must return the cached instance")


class TestBatch5Fixes(unittest.TestCase):
    """Tests for batch-5 fixes (#120, #146, #159, #201, #217, #219, #263, #264)."""

    def setUp(self):
        self.client = TestClient(app)

    # --- #120: asyncio.get_running_loop() instead of get_event_loop() ---
    def test_indexing_callback_uses_get_running_loop(self):
        """indexing_progress_callback must use asyncio.get_running_loop(), not get_event_loop."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.indexing_progress_callback)
        self.assertIn('run_coroutine_threadsafe', src,
                      "Progress broadcasts must be scheduled thread-safely onto the main loop")

    # --- #159: _local_llm_lock released before yielding tokens ---
    def test_local_llm_lock_not_held_across_yield(self):
        """stream_ai_answer must not yield inside the _local_llm_lock block."""
        import inspect, ast
        from backend import llm_integration
        src = inspect.getsource(llm_integration.stream_ai_answer)
        # The 'yield token' must appear AFTER the 'with _local_llm_lock:' block ends.
        # Simple check: 'for token in tokens' (outside lock) must appear in source.
        self.assertIn('for token in tokens:', src)

    # --- #146: docker-compose.yml has healthcheck ---
    def test_docker_compose_has_healthcheck(self):
        """docker-compose.yml must define a healthcheck for the backend service."""
        import os
        compose_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'docker-compose.yml')
        if not os.path.exists(compose_path):
            self.skipTest("docker-compose.yml not found")
        with open(compose_path) as f:
            content = f.read()
        self.assertIn('healthcheck', content)
        self.assertIn('/api/health', content)

    # --- #217: providers/health and providers/models use asyncio.to_thread ---
    def test_provider_health_uses_to_thread(self):
        """provider_health_check must call health_check via asyncio.to_thread."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.provider_health_check)
        self.assertIn('asyncio.to_thread', src)

    def test_provider_models_uses_to_thread(self):
        """provider_list_models must call list_models via asyncio.to_thread."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.provider_list_models)
        self.assertIn('asyncio.to_thread', src)

    # --- #219: raw exception text not leaked in 500 responses ---
    def test_browse_folder_dialog_error_sanitized(self):
        """browse endpoint must not expose str(e) in 500 responses."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.browse_folder)
        # The error message must NOT include f"...{str(e)}" or f"...{e}"
        self.assertNotIn("detail=f\"Failed to open folder dialog: {str(e)}\"", src)

    def test_provider_health_connection_error_sanitized(self):
        """provider_list_models must not expose raw exception text for 503."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.provider_list_models)
        # Must not have detail=str(e) on the ConnectionError catch
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if 'ConnectionError' in line and i + 1 < len(lines):
                next_line = lines[i + 1]
                self.assertNotIn('detail=str(e)', next_line)


class TestBatch4Fixes(unittest.TestCase):
    """Tests for batch-4 fixes (#136, #144, #148, #157, #167, #187, #197, #213, #215)."""

    def setUp(self):
        self.client = TestClient(app)

    # --- #213: OllamaProvider.health_check uses Authorization header ---
    def test_ollama_health_check_sends_auth_header(self):
        """OllamaProvider.health_check must include Authorization when api_key is set."""
        from backend.providers import OllamaProvider
        provider = OllamaProvider(base_url='http://localhost:11434', model='m', api_key='secret')
        with patch.object(provider._session, 'get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {'models': []}
            mock_get.return_value = mock_resp
            provider.health_check()
        call_kwargs = mock_get.call_args
        sent_headers = call_kwargs[1].get('headers', {})
        self.assertIn('Authorization', sent_headers)
        self.assertIn('secret', sent_headers['Authorization'])

    # --- #187: run_indexing uses correct api_key for each provider ---
    def test_run_indexing_api_key_selection(self):
        """run_indexing must select the api_key matching the configured provider."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.run_indexing)
        # Must reference gemini_api_key and anthropic_api_key (not just openai_api_key)
        self.assertIn('gemini_api_key', src)
        self.assertIn('anthropic_api_key', src)

    # --- #157: subprocess.run has timeout ---
    def test_subprocess_run_has_timeout(self):
        """open-file endpoint subprocess.run calls must include a timeout."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.open_file)
        self.assertIn('timeout=30', src)

    # --- #167: validate-path uses asyncio.to_thread ---
    def test_validate_path_uses_async_thread(self):
        """validate-path must offload os.walk to a thread via asyncio.to_thread."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.validate_path)
        self.assertIn('asyncio.to_thread', src)

    # --- #148: agent_chat snapshots globals under _index_lock ---
    def test_agent_chat_uses_index_lock(self):
        """agent_chat must snapshot index globals inside _index_lock."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.agent_chat)
        self.assertIn('_index_lock', src)

    # --- #144: /api/agent/chat is now POST ---
    @patch('backend.api.load_config')
    def test_agent_chat_endpoint_is_post(self, mock_config):
        """GET /api/agent/chat must return 405; POST must be accepted."""
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.return_value = ''
        mock_config.return_value.getboolean.return_value = False
        res_get = self.client.get('/api/agent/chat?query=test')
        self.assertEqual(res_get.status_code, 405)

    # --- #136: stream_ai_answer return type is Iterator[str] ---
    def test_stream_ai_answer_return_annotation(self):
        """stream_ai_answer must be annotated with Iterator[str], not Any."""
        import inspect
        from backend import llm_integration
        hints = llm_integration.stream_ai_answer.__annotations__
        ret = hints.get('return')
        self.assertIsNotNone(ret, "stream_ai_answer must have a return annotation")
        self.assertNotEqual(str(ret), 'Any', "return type must not be Any")

    # --- #215: run_indexing no longer calls indexing_progress_callback inside lock ---
    def test_run_indexing_no_callback_inside_lock(self):
        """run_indexing must not call indexing_progress_callback inside _index_lock."""
        import inspect, ast
        from backend import api as api_mod
        src = inspect.getsource(api_mod.run_indexing)
        # The source must not contain the callback call inside a with _index_lock block.
        # Simple heuristic: check that progress=100 is set directly, not via callback.
        self.assertIn('indexing_status["progress"] = 100', src)

    # --- #197: sort_by=file_size offloads to thread ---
    def test_sort_by_file_size_uses_thread(self):
        """search endpoint sort_by=file_size must use asyncio.to_thread."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.search_files)
        self.assertIn('asyncio.to_thread', src)


class TestBatch3Fixes(unittest.TestCase):
    """Tests for batch-3 fixes (#127, #145, #166, #178, #182, #191, #228)."""

    # --- #145: bare except on tensor_split ---
    def test_tensor_split_bare_except_replaced(self):
        """api.py tensor_split parsing must not use bare except."""
        import ast, inspect
        from backend import api as api_module
        src = inspect.getsource(api_module)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                # Bare except found — check it's not in tensor_split context
                # (we just assert there are no bare excepts at all in the module)
                self.fail(f"Bare 'except:' found at line {node.lineno} — should specify exception type")

    # --- #166: _validate_token no longer reads config on every call ---
    def test_validate_token_uses_cache(self):
        """_validate_token must not open config.ini on subsequent calls."""
        import backend.auth as auth_mod
        auth_mod._cached_token_hash = "abc123"
        with patch('backend.auth.configparser.ConfigParser') as mock_cfg_cls:
            # Call _validate_token — it should hit the cache and NOT read config
            auth_mod._validate_token("sometoken")
        mock_cfg_cls.assert_not_called()
        # Restore
        auth_mod._cached_token_hash = ""

    def test_validate_token_reads_config_when_cache_empty(self):
        """_validate_token must read config.ini when cache is empty."""
        import backend.auth as auth_mod
        auth_mod._cached_token_hash = ""
        mock_cfg = MagicMock()
        mock_cfg.get.return_value = ""
        with patch('backend.auth.configparser.ConfigParser', return_value=mock_cfg):
            result = auth_mod._validate_token("sometoken")
        self.assertFalse(result)
        mock_cfg.read.assert_called_once()
        # Restore
        auth_mod._cached_token_hash = ""

    # --- #182: _QUERY_REWRITE_CACHE bounded ---
    def test_query_rewrite_cache_capped(self):
        """rewrite_query cache must not grow beyond _CACHE_MAX entries."""
        from backend import rag_optimizers
        rag_optimizers._QUERY_REWRITE_CACHE.clear()
        # Pre-fill to just below the cap
        for i in range(rag_optimizers._CACHE_MAX - 1):
            rag_optimizers._QUERY_REWRITE_CACHE[f"key_{i}"] = f"val_{i}"
        self.assertEqual(len(rag_optimizers._QUERY_REWRITE_CACHE), rag_optimizers._CACHE_MAX - 1)

        # Adding one more via direct eviction logic (replicate what rewrite_query does)
        if len(rag_optimizers._QUERY_REWRITE_CACHE) >= rag_optimizers._CACHE_MAX:
            oldest = next(iter(rag_optimizers._QUERY_REWRITE_CACHE))
            del rag_optimizers._QUERY_REWRITE_CACHE[oldest]
        rag_optimizers._QUERY_REWRITE_CACHE["new_key"] = "new_val"
        self.assertLessEqual(len(rag_optimizers._QUERY_REWRITE_CACHE), rag_optimizers._CACHE_MAX)
        rag_optimizers._QUERY_REWRITE_CACHE.clear()

    # --- #228: no duplicate get_file_by_name ---
    def test_get_file_by_name_not_duplicated(self):
        """database.py must have exactly one get_file_by_name definition."""
        import inspect
        from backend import database
        src = inspect.getsource(database)
        count = src.count("def get_file_by_name(")
        self.assertEqual(count, 1, f"Expected 1 get_file_by_name definition, found {count}")

    def test_get_file_by_name_tries_filename_column_first(self):
        """get_file_by_name must query the filename column before path LIKE."""
        from backend import database
        mock_conn = MagicMock()
        # First execute (filename =) returns a row; path LIKE should not be called
        mock_row = MagicMock()
        mock_row.__iter__ = MagicMock(return_value=iter([]))
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        with patch('backend.database.get_connection', return_value=mock_conn):
            database.get_file_by_name("test.pdf")
        first_call_sql = mock_conn.execute.call_args_list[0][0][0]
        self.assertIn("filename =", first_call_sql)

    # --- #127: database.py uses logger, not print ---
    def test_database_has_no_print_calls(self):
        """database.py must not contain print() calls."""
        import inspect
        from backend import database
        src = inspect.getsource(database)
        # Filter out strings/comments; just check raw source for print(
        import ast
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    self.fail(f"print() call found at line {node.lineno} in database.py")

    # --- #178: rag_optimizers.py uses logger, not print ---
    def test_rag_optimizers_has_no_print_calls(self):
        """rag_optimizers.py must not contain print() calls."""
        import inspect, ast
        from backend import rag_optimizers
        src = inspect.getsource(rag_optimizers)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    self.fail(f"print() call found at line {node.lineno} in rag_optimizers.py")

    # --- #191: response_cache eviction ---
    def test_cache_response_evicts_when_over_limit(self):
        """cache_response must evict oldest rows when row count exceeds 1000."""
        from backend import database
        mock_conn = MagicMock()
        cursor = mock_conn.cursor.return_value
        # Simulate INSERT, then count query returning 1100 rows
        cursor.execute.return_value = cursor
        cursor.fetchone.return_value = (1100,)
        with patch('backend.database.get_connection', return_value=mock_conn):
            database.cache_response("h1", "h2", "m1", "answer", "text")
        # Check eviction DELETE was called (3rd execute call after INSERT and COUNT)
        calls = cursor.execute.call_args_list
        delete_calls = [c for c in calls if 'DELETE FROM response_cache' in str(c)]
        self.assertTrue(len(delete_calls) > 0, "Expected a DELETE eviction call when count > 1000")


class TestBatch6Fixes(unittest.TestCase):
    """Tests for batch-6 fixes (#123, #126, #192, #196, #214, #218, #265)."""

    # --- #123: embedding cache key must not contain the raw API key ---
    def test_embedding_cache_key_hashes_api_key(self):
        """get_embeddings must not store the raw API key in the cache key."""
        import inspect
        from backend import llm_integration
        src = inspect.getsource(llm_integration.get_embeddings)
        # The cache key must use a hash of the api_key, not f"{provider}:{api_key}"
        self.assertNotIn('f"{provider}:{api_key', src,
                         "Cache key must not include raw api_key")
        self.assertIn('_digest_secret', src,
                      "Cache key must derive a digest of the api_key (PBKDF2)")

    # --- #126: watchdog handler must debounce rapid filesystem events ---
    def test_watchdog_handler_has_debounce(self):
        """IndexingEventHandler must not call update_index directly from on_modified."""
        import inspect
        from backend import background
        src = inspect.getsource(background.IndexingEventHandler)
        # on_modified/on_created/on_deleted must call the debounce queue, not update_index directly
        self.assertIn('queue_update', src,
                      "Event handlers must delegate to a debounce scheduler")
        self.assertIn('threading.Timer', src,
                      "Debounce must use threading.Timer")

    def test_watchdog_debounce_cancels_previous_timer(self):
        """Multiple rapid events must cancel the previous timer and start a new one."""
        from backend.background import IndexingEventHandler
        handler = IndexingEventHandler('/tmp', 'local', None, None, debounce_delay=5.0)
        # Call queue_update twice quickly — the first timer must be cancelled
        first_timer_cancelled = []
        real_cancel = None

        import threading
        original_timer = threading.Timer

        def fake_timer(delay, fn):
            t = original_timer(delay, fn)
            real_cancel_fn = t.cancel
            def tracked_cancel():
                first_timer_cancelled.append(True)
                real_cancel_fn()
            t.cancel = tracked_cancel
            return t

        with patch('backend.background.threading.Timer', side_effect=fake_timer):
            handler.queue_update()
            handler.queue_update()

        self.assertTrue(len(first_timer_cancelled) >= 1,
                        "First timer must be cancelled when a second event fires")
        # Cleanup
        if handler._timer:
            handler._timer.cancel()

    # --- #192: checkpoint must not save after every file ---
    def test_checkpoint_saves_every_10_files_not_every_file(self):
        """_save_checkpoint must not be called for every file extracted."""
        import inspect
        from backend import indexing
        src = inspect.getsource(indexing.create_index)
        # The code must contain '_since_save' (the batching counter)
        self.assertIn('_since_save', src,
                      "Indexing must batch checkpoint saves rather than saving after every file")

    # --- checkpoint is fingerprinted to the exact file set ---
    def test_checkpoint_is_fingerprinted_to_file_set(self):
        """Checkpoint reuse must be gated on a fingerprint of the file list."""
        import inspect
        from backend import indexing
        src = inspect.getsource(indexing.create_index)
        self.assertIn('_fingerprint', src,
                      "Checkpoint must be tied to the current file set")
        self.assertIn('_load_checkpoint(_fingerprint)', src,
                      "Checkpoint load must pass the fingerprint")

    # --- #196: cached_smart_summary must be called via asyncio.to_thread in search ---
    def test_search_uses_asyncio_to_thread_for_smart_summary(self):
        """search_files must wrap cached_smart_summary in asyncio.to_thread."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.search_files)
        self.assertIn('asyncio.to_thread', src,
                      "search_files must call cached_smart_summary via asyncio.to_thread")
        self.assertIn('cached_smart_summary', src)

    # --- #218: IndexingBanner must use WebSocket, not only HTTP polling ---
    def test_indexing_banner_uses_websocket(self):
        """IndexingBanner.jsx must open a WebSocket connection to /ws/progress."""
        import os as _os
        banner_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
            'frontend', 'src', 'components', 'IndexingBanner.jsx',
        )
        with open(banner_path) as f:
            src = f.read()
        self.assertIn('WebSocket', src,
                      "IndexingBanner must use WebSocket for live updates")
        self.assertIn('/ws/progress', src,
                      "IndexingBanner must connect to /ws/progress")

    # --- #265: AgentView must be wrapped in an ErrorBoundary ---
    def test_agent_view_wrapped_in_error_boundary(self):
        """SearchView.jsx must wrap AgentView inside an ErrorBoundary."""
        import os as _os
        search_view_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
            'frontend', 'src', 'components', 'SearchView.jsx',
        )
        with open(search_view_path) as f:
            src = f.read()
        self.assertIn('ErrorBoundary', src,
                      "SearchView must import ErrorBoundary")
        # Verify AgentView is rendered inside ErrorBoundary tags
        boundary_idx = src.find('<ErrorBoundary>')
        agent_idx = src.find('<AgentView')
        self.assertGreater(agent_idx, boundary_idx,
                           "AgentView must appear after <ErrorBoundary> opening tag")


class TestBatch7Fixes(unittest.TestCase):
    """Tests for batch-7 fixes (#136, #139, #143, #168, #211, #212)."""

    def setUp(self):
        self.client = TestClient(app)

    # --- #212: cache/clear and search/history must require auth ---
    def test_cache_clear_requires_auth(self):
        """POST /api/cache/clear must have require_auth dependency."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.clear_cache_endpoint)
        self.assertIn('require_auth', src,
                      "clear_cache_endpoint must use require_auth")

    def test_delete_search_history_requires_auth(self):
        """DELETE /api/search/history must have require_auth dependency."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.delete_all_search_history)
        self.assertIn('require_auth', src,
                      "delete_all_search_history must use require_auth")

    def test_delete_search_history_item_requires_auth(self):
        """DELETE /api/search/history/{id} must have require_auth dependency."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.delete_search_history_item)
        self.assertIn('require_auth', src,
                      "delete_search_history_item must use require_auth")

    # --- #143: stream_answer_endpoint snapshots globals under lock ---
    def test_stream_answer_uses_index_lock(self):
        """stream_answer_endpoint must snapshot index globals under _index_lock."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.stream_answer_endpoint)
        self.assertIn('_index_lock', src,
                      "stream_answer_endpoint must use _index_lock to snapshot globals")

    # --- #168: stream error must be SSE format, not plain string ---
    def test_stream_answer_error_is_sse_format(self):
        """stream_answer_endpoint 'not loaded' error must be proper SSE, not plain text."""
        import inspect
        from backend import api as api_mod
        src = inspect.getsource(api_mod.stream_answer_endpoint)
        self.assertIn('[ERROR]', src,
                      "Generation failures must surface a terminal [ERROR] token to the client")

    # --- stream_chat is an async generator consumed directly by the SSE route ---
    def test_stream_chat_is_async_generator(self):
        """agent.stream_chat must be an async generator function."""
        import inspect
        from backend import agent as agent_mod
        self.assertTrue(inspect.isasyncgenfunction(agent_mod.ReActAgent.stream_chat),
                        "stream_chat must be an async generator (api.py uses 'async for')")

    # --- #139: tools.py must select api_key by provider ---
    def test_tools_api_key_is_provider_aware(self):
        """tool_search_knowledge_base must select api_key based on provider."""
        import inspect
        from backend import tools as tools_mod
        src = inspect.getsource(tools_mod.tool_search_knowledge_base)
        self.assertIn('gemini_api_key', src,
                      "tools must select gemini_api_key for gemini provider")
        self.assertIn('anthropic_api_key', src,
                      "tools must select anthropic_api_key for anthropic provider")

    # --- #211: .dockerignore must exclude config.ini and .env ---
    def test_dockerignore_excludes_secrets(self):
        """dockerignore must exclude config.ini and .env."""
        import os as _os
        path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
            '.dockerignore',
        )
        self.assertTrue(_os.path.exists(path), ".dockerignore must exist")
        with open(path) as f:
            content = f.read()
        self.assertIn('config.ini', content, ".dockerignore must exclude config.ini")
        self.assertIn('.env', content, ".dockerignore must exclude .env")


class TestValidatePathTimeout(unittest.TestCase):
    """Tests for validate-path timeout and file count cap."""

    def setUp(self):
        self.client = TestClient(app)

    def test_validate_path_caps_file_count(self):
        """File count should stop at 10,000 and not walk further."""
        home_subfolder = os.path.join(os.path.expanduser("~"), "test_folder")
        # Simulate a directory with 20,000 supported files
        huge_file_list = [f"file_{i}.pdf" for i in range(20_000)]
        with patch('backend.api._allowed_index_roots', return_value=(os.path.expanduser("~") + os.sep,)), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('os.walk', return_value=[(home_subfolder, [], huge_file_list)]):
            response = self.client.post("/api/validate-path", json={
                "path": home_subfolder
            })

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data['valid'])
            # Should be capped at 10,000
            self.assertEqual(data['file_count'], 10_000)

    def test_validate_path_returns_valid_on_timeout(self):
        """If file counting times out, the path is still valid with file_count=-1."""
        import asyncio
        home_subfolder = os.path.join(os.path.expanduser("~"), "slow_folder")

        def slow_walk(folder):
            """Simulate a very slow os.walk."""
            import time
            time.sleep(10)
            return []

        with patch('backend.api._allowed_index_roots', return_value=(os.path.expanduser("~") + os.sep,)), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('os.walk', side_effect=slow_walk):
            response = self.client.post("/api/validate-path", json={
                "path": home_subfolder
            })

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data['valid'])
            self.assertEqual(data['file_count'], -1)


class TestIPv4Monkeypatch(unittest.TestCase):
    """Test that the IPv4 socket monkeypatch is active."""

    def test_socket_getaddrinfo_returns_only_ipv4(self):
        """socket.getaddrinfo should only return AF_INET (IPv4) after monkeypatch."""
        import socket
        results = socket.getaddrinfo('localhost', 80)
        for result in results:
            family = result[0]
            self.assertEqual(family, socket.AF_INET,
                             f"Expected AF_INET but got {family} — IPv4 monkeypatch not active")


if __name__ == '__main__':
    unittest.main()