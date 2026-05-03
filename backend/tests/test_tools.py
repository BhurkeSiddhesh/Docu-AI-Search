"""
Tests for backend/tools.py

Covers tool_search_knowledge_base, tool_read_file, and tool_list_files.
All external dependencies (search, database, file_processing) are mocked.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from backend import tools


class TestToolListFiles(unittest.TestCase):
    """Tests for tool_list_files()."""

    @patch("backend.tools.database.get_all_files")
    def test_returns_comma_separated_names(self, mock_get_all):
        """Returns filenames joined by comma when files exist."""
        mock_get_all.return_value = [
            {"filename": "alpha.pdf"},
            {"filename": "beta.docx"},
        ]
        result = tools.tool_list_files()
        self.assertIn("alpha.pdf", result)
        self.assertIn("beta.docx", result)

    @patch("backend.tools.database.get_all_files")
    def test_no_files_returns_message(self, mock_get_all):
        """Returns descriptive message when no files are indexed."""
        mock_get_all.return_value = []
        result = tools.tool_list_files()
        self.assertEqual(result, "No files indexed.")

    @patch("backend.tools.database.get_all_files")
    def test_limits_to_50_files(self, mock_get_all):
        """Only the first 50 filenames are included."""
        mock_get_all.return_value = [{"filename": f"file{i}.pdf"} for i in range(100)]
        result = tools.tool_list_files()
        names = result.split(", ")
        self.assertLessEqual(len(names), 50)

    @patch("backend.tools.database.get_all_files")
    def test_query_param_accepted(self, mock_get_all):
        """Optional query parameter is accepted without error."""
        mock_get_all.return_value = [{"filename": "doc.pdf"}]
        result = tools.tool_list_files(query="anything")
        self.assertIn("doc.pdf", result)


class TestToolReadFile(unittest.TestCase):
    """Tests for tool_read_file()."""

    def test_empty_path_returns_error(self):
        """Empty file_path returns an error string."""
        result = tools.tool_read_file("")
        self.assertIn("Error", result)

    @patch("backend.tools.database.get_file_by_path", return_value=None)
    @patch("backend.database.get_file_by_name", return_value=None, create=True)
    def test_file_not_in_db_returns_access_denied(self, mock_name, mock_path):
        """Returns access denied error when file is not in the knowledge base."""
        result = tools.tool_read_file("/some/random/path.pdf")
        self.assertIn("Access denied", result)

    @patch("backend.tools.database.get_file_by_path")
    def test_file_not_on_disk_returns_error(self, mock_path):
        """Returns error when file is in DB but missing from disk."""
        mock_path.return_value = {"path": "/nonexistent/file.pdf"}
        result = tools.tool_read_file("/nonexistent/file.pdf")
        self.assertIn("Error", result)

    @patch("backend.tools.database.get_file_by_path")
    @patch("backend.file_processing.extract_text")
    def test_reads_file_content(self, mock_extract, mock_db):
        """Returns extracted file content up to 5000 characters."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmp_path = f.name
        try:
            mock_db.return_value = {"path": tmp_path}
            mock_extract.return_value = "Hello document content"

            result = tools.tool_read_file(tmp_path)
            self.assertEqual(result, "Hello document content")
        finally:
            os.unlink(tmp_path)

    @patch("backend.tools.database.get_file_by_path")
    @patch("backend.file_processing.extract_text")
    def test_content_capped_at_5000_chars(self, mock_extract, mock_db):
        """Content is truncated to 5000 characters max."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmp_path = f.name
        try:
            mock_db.return_value = {"path": tmp_path}
            mock_extract.return_value = "x" * 10_000

            result = tools.tool_read_file(tmp_path)
            self.assertEqual(len(result), 5000)
        finally:
            os.unlink(tmp_path)

    @patch("backend.tools.database.get_file_by_path")
    @patch("backend.file_processing.extract_text")
    def test_empty_extracted_text_returns_message(self, mock_extract, mock_db):
        """Returns descriptive message when extracted text is empty."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmp_path = f.name
        try:
            mock_db.return_value = {"path": tmp_path}
            mock_extract.return_value = ""

            result = tools.tool_read_file(tmp_path)
            self.assertIn("empty", result.lower())
        finally:
            os.unlink(tmp_path)

    @patch("backend.tools.database.get_file_by_path", return_value=None)
    @patch("backend.database.get_file_by_name", create=True)
    @patch("backend.file_processing.extract_text")
    def test_fallback_to_lookup_by_name(self, mock_extract, mock_name, mock_path):
        """Falls back to get_file_by_name when get_file_by_path returns None."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmp_path = f.name
        try:
            mock_name.return_value = {"path": tmp_path}
            mock_extract.return_value = "Found by name"

            result = tools.tool_read_file(os.path.basename(tmp_path))
            self.assertEqual(result, "Found by name")
        finally:
            os.unlink(tmp_path)


class TestToolSearchKnowledgeBase(unittest.TestCase):
    """Tests for tool_search_knowledge_base()."""

    def _make_state(self, has_index=True):
        config = MagicMock()
        config.get.return_value = "local"
        return {
            "config": config,
            "index": MagicMock() if has_index else None,
            "docs": [],
            "tags": [],
            "index_summaries": None,
            "cluster_summaries": None,
            "cluster_map": None,
            "bm25": None,
        }

    def test_no_index_returns_error(self):
        """Returns error string when no index is loaded."""
        state = self._make_state(has_index=False)
        result = tools.tool_search_knowledge_base("anything", state)
        self.assertIn("Error", result)

    @patch("backend.tools.search.search")
    @patch("backend.tools.llm_integration.get_embeddings")
    def test_no_results_returns_message(self, mock_embed, mock_search):
        """Returns 'no results' message when search yields nothing."""
        mock_search.return_value = ([], {})
        mock_embed.return_value = MagicMock()
        state = self._make_state()

        result = tools.tool_search_knowledge_base("query", state)
        self.assertEqual(result, "No relevant information found.")

    @patch("backend.tools.search.search")
    @patch("backend.tools.llm_integration.get_embeddings")
    def test_formats_results_correctly(self, mock_embed, mock_search):
        """Results are formatted with source and content."""
        mock_search.return_value = (
            [{"file_name": "report.pdf", "document": "Revenue grew by 20% in Q3."}],
            {}
        )
        mock_embed.return_value = MagicMock()
        state = self._make_state()

        result = tools.tool_search_knowledge_base("revenue", state)
        self.assertIn("report.pdf", result)
        self.assertIn("Revenue grew", result)

    @patch("backend.tools.search.search")
    @patch("backend.tools.llm_integration.get_embeddings")
    def test_limits_to_five_results(self, mock_embed, mock_search):
        """Only the top 5 results are included in the output."""
        results = [{"file_name": f"file{i}.pdf", "document": f"content {i}"} for i in range(10)]
        mock_search.return_value = (results, {})
        mock_embed.return_value = MagicMock()
        state = self._make_state()

        output = tools.tool_search_knowledge_base("test", state)
        # Each result has "Source: fileN.pdf" — count occurrences
        source_count = output.count("Source:")
        self.assertLessEqual(source_count, 5)

    @patch("backend.tools.search.search")
    @patch("backend.tools.llm_integration.get_embeddings")
    def test_document_text_truncated_to_500_chars(self, mock_embed, mock_search):
        """Long document text is truncated to 500 characters per result."""
        long_content = "z" * 2000
        mock_search.return_value = (
            [{"file_name": "big.pdf", "document": long_content}],
            {}
        )
        mock_embed.return_value = MagicMock()
        state = self._make_state()

        result = tools.tool_search_knowledge_base("query", state)
        content_section = result.split("Content:")[-1]
        self.assertLessEqual(len(content_section.strip()), 510)


class TestAvailableTools(unittest.TestCase):
    """Tests for the AVAILABLE_TOOLS registry."""

    def test_all_expected_tools_registered(self):
        """All three tools are present in AVAILABLE_TOOLS."""
        self.assertIn("search_knowledge_base", tools.AVAILABLE_TOOLS)
        self.assertIn("read_file", tools.AVAILABLE_TOOLS)
        self.assertIn("list_files", tools.AVAILABLE_TOOLS)

    def test_tools_are_callable(self):
        """Each registered tool is a callable."""
        for name, fn in tools.AVAILABLE_TOOLS.items():
            self.assertTrue(callable(fn), f"Tool '{name}' is not callable")


if __name__ == "__main__":
    unittest.main()
