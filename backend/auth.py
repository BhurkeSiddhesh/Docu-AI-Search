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

# PBKDF2 parameters for token storage. The token itself is 256-bit random, so
# the KDF is defence-in-depth; iteration cost is only paid on the first request
# after startup thanks to the validated-token cache below.
_PBKDF2_ITERATIONS = 100_000


def _hash_token(token: str, salt_hex: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256", token.encode(), bytes.fromhex(salt_hex), _PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt_hex}${dk.hex()}"


# Module-level cache for the stored token hash — avoids a config.ini read on
# every authenticated request; refreshed whenever a new token is generated.
# _validated_token holds the plaintext of the last successfully validated
# token so subsequent requests skip the KDF (constant-time compare only).
_cached_token_hash: str = ""
_validated_token = None


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
    if not token_hash or "$" not in token_hash:
        # No token yet, or a legacy unsalted hash from an older release —
        # regenerate so the stored credential always uses the current KDF.
        token = secrets.token_hex(32)
        new_hash = _hash_token(token, secrets.token_hex(16))
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
    global _validated_token
    stored_hash = _load_token_hash()
    if not stored_hash:
        return False
    if _validated_token is not None and secrets.compare_digest(token, _validated_token):
        return True
    try:
        _scheme, _iters, salt_hex, _digest = stored_hash.split("$", 3)
        candidate = _hash_token(token, salt_hex)
    except ValueError:
        return False  # legacy/corrupt hash format — token must be regenerated
    if secrets.compare_digest(candidate, stored_hash):
        _validated_token = token
        return True
    return False


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
