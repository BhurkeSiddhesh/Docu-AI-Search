"""
Test Database Module

Tests for the database module including file tracking,
search history, and metadata storage.
"""

import unittest
import os
import tempfile
import shutil
from backend import database

# Initialize database for unittest execution
_original_db_path = None
_temp_db_file = None

def setUpModule():
    """Initialize database schema in a temp file."""
    global _original_db_path, _temp_db_file

    # Save original path
    _original_db_path = database.DATABASE_PATH

    # Create temp file
    _temp_db_file = tempfile.NamedTemporaryFile(delete=False)
    _temp_db_file.close()  # Close so sqlite can open it

    # Override path
    database.DATABASE_PATH = _temp_db_file.name

    # Init DB
    database.init_database()

def tearDownModule():
    """Cleanup temp database."""
    global _original_db_path, _temp_db_file

    # Restore path
    if _original_db_path:
        database.DATABASE_PATH = _original_db_path

    # Remove file
    if _temp_db_file and os.path.exists(_temp_db_file.name):
        try:
            os.unlink(_temp_db_file.name)
        except OSError:
            pass


class TestDatabaseBase(unittest.TestCase):
    """Base class for database tests ensuring clean state."""

    def setUp(self):
        """Clear relevant tables before each test."""
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM files")
        cursor.execute("DELETE FROM search_history")
        cursor.execute("DELETE FROM folder_history")
        cursor.execute("DELETE FROM preferences")
        conn.commit()
        conn.close()
