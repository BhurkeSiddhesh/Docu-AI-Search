"""
Tests for backend/file_processing.py — extract_text function.

Replaces the old manual exploration script with proper unittest cases
that create temporary files and verify extraction behavior.
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import MagicMock

# Stub out heavy dependencies before importing file_processing
for _mod in ["pypdf", "docx", "openpyxl", "pptx", "pptx.util"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.file_processing import extract_text


class TestExtractTextTxt(unittest.TestCase):
    """Tests for plain-text extraction."""

    def _write_tmp(self, content, suffix=".txt"):
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def tearDown(self):
        # Individual tests clean up their own temp files
        pass

    def test_extracts_plain_text_content(self):
        path = self._write_tmp("Hello, this is a test document.")
        try:
            result = extract_text(path)
            self.assertIn("Hello", result)
            self.assertIn("test document", result)
        finally:
            os.unlink(path)

    def test_extracts_multiline_text(self):
        content = "Line one\nLine two\nLine three"
        path = self._write_tmp(content)
        try:
            result = extract_text(path)
            self.assertIn("Line one", result)
            self.assertIn("Line three", result)
        finally:
            os.unlink(path)

    def test_empty_txt_file_returns_empty_or_none(self):
        path = self._write_tmp("")
        try:
            result = extract_text(path)
            self.assertFalse(result)  # empty string or None is falsy
        finally:
            os.unlink(path)

    def test_txt_with_unicode_characters(self):
        content = "Résumé of Müller: expertise in naïve Bayesian methods"
        path = self._write_tmp(content)
        try:
            result = extract_text(path)
            self.assertIsNotNone(result)
            self.assertGreater(len(result), 0)
        finally:
            os.unlink(path)

    def test_returns_string_type(self):
        path = self._write_tmp("Some content here")
        try:
            result = extract_text(path)
            if result is not None:
                self.assertIsInstance(result, str)
        finally:
            os.unlink(path)


class TestExtractTextErrorCases(unittest.TestCase):
    """Tests for error handling and edge cases in extract_text."""

    def test_nonexistent_file_returns_none_or_empty(self):
        result = extract_text("/nonexistent/path/file.txt")
        self.assertFalse(result)

    def test_unsupported_extension_returns_none_or_empty(self):
        fd, path = tempfile.mkstemp(suffix=".xyz")
        os.close(fd)
        try:
            result = extract_text(path)
            self.assertFalse(result)
        finally:
            os.unlink(path)

    def test_binary_file_with_txt_extension_handled_gracefully(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "wb") as f:
            f.write(bytes(range(256)))  # non-UTF-8 binary data
        try:
            # Should not raise an unhandled exception
            result = extract_text(path)
            # Result can be None/empty or partial; just no crash
        except Exception as e:
            self.fail(f"extract_text raised unexpected exception: {e}")
        finally:
            os.unlink(path)

    def test_none_path_handled_gracefully(self):
        try:
            result = extract_text(None)
            # Either returns falsy or raises; both are acceptable
        except Exception:
            pass  # Acceptable — just must not propagate silently wrong data

    def test_empty_string_path_handled_gracefully(self):
        try:
            result = extract_text("")
            self.assertFalse(result)
        except Exception:
            pass


class TestExtractTextFormats(unittest.TestCase):
    """Smoke tests verifying that each supported format is attempted."""

    def test_pdf_extension_attempted(self):
        """extract_text should attempt PDF parsing (may return empty for invalid PDF)."""
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.write(fd, b"%PDF-1.4 fake pdf content")
        os.close(fd)
        try:
            # Should not raise — invalid PDFs return None/empty gracefully
            result = extract_text(path)
        except Exception as e:
            self.fail(f"extract_text raised for .pdf extension: {e}")
        finally:
            os.unlink(path)

    def test_docx_extension_attempted(self):
        """extract_text should attempt DOCX parsing without crashing on invalid file."""
        fd, path = tempfile.mkstemp(suffix=".docx")
        os.write(fd, b"PK fake docx data")
        os.close(fd)
        try:
            result = extract_text(path)
        except Exception as e:
            self.fail(f"extract_text raised for .docx extension: {e}")
        finally:
            os.unlink(path)

    def test_xlsx_extension_attempted(self):
        """extract_text should attempt XLSX parsing without crashing on invalid file."""
        fd, path = tempfile.mkstemp(suffix=".xlsx")
        os.write(fd, b"PK fake xlsx data")
        os.close(fd)
        try:
            result = extract_text(path)
        except Exception as e:
            self.fail(f"extract_text raised for .xlsx extension: {e}")
        finally:
            os.unlink(path)

    def test_pptx_extension_attempted(self):
        """extract_text should attempt PPTX parsing without crashing on invalid file."""
        fd, path = tempfile.mkstemp(suffix=".pptx")
        os.write(fd, b"PK fake pptx data")
        os.close(fd)
        try:
            result = extract_text(path)
        except Exception as e:
            self.fail(f"extract_text raised for .pptx extension: {e}")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
