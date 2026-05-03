"""
Tests for backend/auth.py

Covers token creation, validation, and the require_auth FastAPI dependency.
"""

import hashlib
import os
import secrets
import tempfile
import configparser
import unittest
from unittest.mock import patch, MagicMock

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


class TestValidateToken(unittest.TestCase):
    """Tests for the _validate_token() helper."""

    def _write_token_hash(self, config_path: str, token: str) -> None:
        config = configparser.ConfigParser()
        config.read(config_path)
        if not config.has_section("Auth"):
            config.add_section("Auth")
        config.set("Auth", "token_hash", hashlib.sha256(token.encode()).hexdigest())
        with open(config_path, "w") as f:
            config.write(f)

    def test_valid_token_returns_true(self):
        """Correct token passes validation."""
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
            config_path = f.name

        try:
            token = "mysecrettoken123"
            self._write_token_hash(config_path, token)

            with patch("backend.auth._CONFIG_PATH", config_path):
                from backend.auth import _validate_token
                self.assertTrue(_validate_token(token))
        finally:
            os.unlink(config_path)

    def test_wrong_token_returns_false(self):
        """Incorrect token fails validation."""
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
            config_path = f.name

        try:
            self._write_token_hash(config_path, "correct_token")

            with patch("backend.auth._CONFIG_PATH", config_path):
                from backend.auth import _validate_token
                self.assertFalse(_validate_token("wrong_token"))
        finally:
            os.unlink(config_path)

    def test_no_stored_hash_returns_false(self):
        """Validation returns False when no token hash is in config."""
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
            config_path = f.name

        try:
            with patch("backend.auth._CONFIG_PATH", config_path):
                from backend.auth import _validate_token
                self.assertFalse(_validate_token("anytoken"))
        finally:
            os.unlink(config_path)

    def test_empty_token_returns_false(self):
        """Empty string token fails validation."""
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
            config_path = f.name

        try:
            self._write_token_hash(config_path, "real_token")

            with patch("backend.auth._CONFIG_PATH", config_path):
                from backend.auth import _validate_token
                self.assertFalse(_validate_token(""))
        finally:
            os.unlink(config_path)


class TestGetOrCreateToken(unittest.TestCase):
    """Tests for _get_or_create_token()."""

    def test_creates_token_when_none_exists(self):
        """A new token is generated and hash written to config.ini."""
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
            config_path = f.name

        try:
            with patch("backend.auth._CONFIG_PATH", config_path):
                from backend.auth import _get_or_create_token
                token = _get_or_create_token()

            self.assertIsInstance(token, str)
            self.assertGreater(len(token), 0)

            config = configparser.ConfigParser()
            config.read(config_path)
            stored_hash = config.get("Auth", "token_hash", fallback="")
            expected = hashlib.sha256(token.encode()).hexdigest()
            self.assertEqual(stored_hash, expected)
        finally:
            os.unlink(config_path)

    def test_returns_empty_string_when_token_already_exists(self):
        """Returns empty sentinel when a hash already exists (plaintext unrecoverable)."""
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="w") as f:
            config_path = f.name

        try:
            config = configparser.ConfigParser()
            config.add_section("Auth")
            config.set("Auth", "token_hash", hashlib.sha256(b"existing").hexdigest())
            with open(config_path, "w") as f:
                config.write(f)

            with patch("backend.auth._CONFIG_PATH", config_path):
                from backend.auth import _get_or_create_token
                result = _get_or_create_token()

            self.assertEqual(result, "")
        finally:
            os.unlink(config_path)


class TestRequireAuth(unittest.IsolatedAsyncioTestCase):
    """Tests for the require_auth FastAPI dependency."""

    async def test_auth_disabled_allows_all(self):
        """When AUTH_ENABLED is False, any request is allowed."""
        with patch("backend.auth.AUTH_ENABLED", False):
            from backend.auth import require_auth
            mock_request = MagicMock()
            result = await require_auth(request=mock_request, credentials=None)
            self.assertIsNone(result)

    async def test_auth_enabled_valid_token_passes(self):
        """Valid Bearer token passes when auth is enabled."""
        with patch("backend.auth.AUTH_ENABLED", True), \
             patch("backend.auth._validate_token", return_value=True):
            from backend.auth import require_auth
            mock_request = MagicMock()
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
            result = await require_auth(request=mock_request, credentials=credentials)
            self.assertIsNone(result)

    async def test_auth_enabled_invalid_token_raises_401(self):
        """Invalid Bearer token raises 401 when auth is enabled."""
        with patch("backend.auth.AUTH_ENABLED", True), \
             patch("backend.auth._validate_token", return_value=False):
            from backend.auth import require_auth
            mock_request = MagicMock()
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad_token")
            with self.assertRaises(HTTPException) as ctx:
                await require_auth(request=mock_request, credentials=credentials)
            self.assertEqual(ctx.exception.status_code, 401)

    async def test_auth_enabled_no_credentials_raises_401(self):
        """Missing credentials raise 401 when auth is enabled."""
        with patch("backend.auth.AUTH_ENABLED", True), \
             patch("backend.auth._validate_token", return_value=False):
            from backend.auth import require_auth
            mock_request = MagicMock()
            with self.assertRaises(HTTPException) as ctx:
                await require_auth(request=mock_request, credentials=None)
            self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
