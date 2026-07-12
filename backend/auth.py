"""
Lightweight API key authentication for Docu-AI-Search.

Set AUTH_ENABLED=true to require a Bearer token on all sensitive endpoints.
The token is auto-generated on first run and written to config.ini.
Retrieve it with: GET /api/auth/token (localhost-only).
"""

import hashlib
import os
import secrets
import configparser
import logging
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

_security = HTTPBearer(auto_error=False)

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"

# Path to config.ini (mirrors api.py)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_BASE_DIR, "config.ini")

# Module-level cache for the stored token hash — avoids a config.ini read on
# every authenticated request; refreshed whenever a new token is generated.
_cached_token_hash: str = ""


def _load_token_hash() -> str:
    """Read and cache the stored token hash from config.ini."""
    global _cached_token_hash
    if _cached_token_hash:
        return _cached_token_hash
    config = configparser.ConfigParser()
    config.read(_CONFIG_PATH)
    _cached_token_hash = config.get("Auth", "token_hash", fallback="")
    return _cached_token_hash


def _get_or_create_token() -> str:
    """Return the plaintext token, creating it if it doesn't exist yet."""
    global _cached_token_hash
    config = configparser.ConfigParser()
    config.read(_CONFIG_PATH)
    if not config.has_section("Auth"):
        config.add_section("Auth")
    token_hash = config.get("Auth", "token_hash", fallback="")
    if not token_hash:
        token = secrets.token_hex(32)
        new_hash = hashlib.sha256(token.encode()).hexdigest()
        config.set("Auth", "token_hash", new_hash)
        import tempfile, os as _os
        _dir = _os.path.dirname(_CONFIG_PATH)
        with tempfile.NamedTemporaryFile("w", dir=_dir, delete=False, suffix=".tmp") as _tmp:
            config.write(_tmp)
            _tmp_path = _tmp.name
        _os.replace(_tmp_path, _CONFIG_PATH)
        _cached_token_hash = new_hash
        logger.info("Generated new API auth token. Retrieve it via GET /api/auth/token (localhost only).")
        return token
    # Token was already created; cache the hash and return sentinel.
    _cached_token_hash = token_hash
    return ""


def _validate_token(token: str) -> bool:
    stored_hash = _load_token_hash()
    if not stored_hash:
        return False
    candidate_hash = hashlib.sha256(token.encode()).hexdigest()
    return secrets.compare_digest(candidate_hash, stored_hash)


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_security),
):
    """FastAPI dependency — validates Bearer token when AUTH_ENABLED=true."""
    if not AUTH_ENABLED:
        return  # Auth disabled; allow all requests.
    if credentials is None or not _validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


# Ensure a token exists on module import when auth is enabled
if AUTH_ENABLED:
    _get_or_create_token()
