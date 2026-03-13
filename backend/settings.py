"""
backend/settings.py
-------------------
FastAPI router that manages the embedding provider configuration.

Endpoints
---------
GET  /api/settings/embeddings  – Return the active embedding config (key masked).
POST /api/settings/embeddings  – Persist a new embedding config to config.ini
                                 and refresh app.state so all subsequent calls
                                 use the updated settings without a restart.

Helper
------
get_active_embedding_client(app) – Returns a ready-to-use LangChain embeddings
                                   object, reading from app.state first and
                                   falling back to config.ini / legacy path.
"""

from __future__ import annotations

import configparser
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers (mirrors api.py so both read the same config.ini)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.ini")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_PROVIDER_TYPES = {"local", "huggingface_api", "commercial_api"}

_DEFAULT_EMBEDDING_CONFIG = {
    "provider_type": "local",
    "model_name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "api_key": "",
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EmbeddingConfig(BaseModel):
    """Payload accepted by POST /api/settings/embeddings."""

    provider_type: str
    model_name: str
    api_key: Optional[str] = None

    @field_validator("provider_type")
    @classmethod
    def validate_provider_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in VALID_PROVIDER_TYPES:
            raise ValueError(
                f"Invalid provider_type '{v}'. "
                f"Must be one of: {sorted(VALID_PROVIDER_TYPES)}"
            )
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("model_name must not be empty.")
        return v


class EmbeddingConfigResponse(BaseModel):
    """Shape returned by GET /api/settings/embeddings (key masked)."""

    provider_type: str
    model_name: str
    api_key_set: bool  # True if a non-empty key is stored


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/settings", tags=["settings"])


# ── GET ────────────────────────────────────────────────────────────────────

@router.get("/embeddings", response_model=EmbeddingConfigResponse)
async def get_embedding_config(request: Request):
    """
    Retrieves the current embedding provider configuration.

    The response provides the provider type, model name, and a flag indicating
    if an API key is configured. To maintain security, the raw API key is
    never exposed through this endpoint.

    Args:
        request (Request): The FastAPI request object, used to access app state.

    Returns:
        EmbeddingConfigResponse: The current configuration with the API key status.
    """
    # Prefer app.state if available (already seeded on startup)
    state_cfg = getattr(request.app.state, "embedding_config", None)
    if state_cfg:
        return EmbeddingConfigResponse(
            provider_type=state_cfg.get("provider_type", _DEFAULT_EMBEDDING_CONFIG["provider_type"]),
            model_name=state_cfg.get("model_name", _DEFAULT_EMBEDDING_CONFIG["model_name"]),
            api_key_set=bool(state_cfg.get("api_key", "")),
        )

    # Fall back to reading config.ini directly
    cfg = _read_embedding_section()
    return EmbeddingConfigResponse(
        provider_type=cfg["provider_type"],
        model_name=cfg["model_name"],
        api_key_set=bool(cfg["api_key"]),
    )


# ── POST ───────────────────────────────────────────────────────────────────

@router.post("/embeddings")
async def update_embedding_config(body: EmbeddingConfig, request: Request):
    """
    Updates and persists the embedding provider configuration.

    This endpoint:
        1. Validates the input payload.
        2. Ensures an API key is provided for non-local providers.
        3. Persists the configuration to `config.ini`.
        4. Updates the in-memory cache in `app.state` to avoid a restart.

    Args:
        body (EmbeddingConfig): The new configuration details.
        request (Request): The FastAPI request object to update app state.

    Returns:
        dict: A success status with the updated configuration details.

    Raises:
        HTTPException: 422 Unprocessable Entity if an API key is missing for
                       cloud-based providers.
    """
    # --- Extra validation: api_key is mandatory for non-local providers ---
    if body.provider_type in ("huggingface_api", "commercial_api"):
        if not body.api_key:
            raise HTTPException(
                status_code=422,
                detail=f"'api_key' is required for provider_type='{body.provider_type}'.",
            )

    new_cfg = {
        "provider_type": body.provider_type,
        "model_name": body.model_name,
        "api_key": body.api_key or "",
    }

    # Persist to config.ini
    _write_embedding_section(new_cfg)

    # Refresh app.state (hot-reload without restart)
    request.app.state.embedding_config = new_cfg

    logger.info(
        "[Settings] Embedding config updated: provider_type=%s, model_name=%s",
        new_cfg["provider_type"],
        new_cfg["model_name"],
    )
    return {
        "status": "success",
        "message": "Embedding configuration saved.",
        "provider_type": new_cfg["provider_type"],
        "model_name": new_cfg["model_name"],
    }


# ---------------------------------------------------------------------------
# Public helper – used by api.py search & indexing paths
# ---------------------------------------------------------------------------

def get_active_embedding_client(app):
    """
    Resolves and returns a LangChain embedding client based on current settings.

    The function follows a strict resolution order:
        1. Checks in-memory `app.state.embedding_config` for the newest settings.
        2. Falls back to reading the `[Embeddings]` section in `config.ini`.
        3. If neither contains custom settings, it falls back to the legacy 
           `get_embeddings(provider='local')` path.

    Args:
        app: The FastAPI application instance.

    Returns:
        Embeddings: A concrete LangChain embeddings object (e.g., HuggingFace, OpenAI).
    """
    from backend.llm_integration import get_embedding_client, get_embeddings  # lazy

    # 1. Try app.state
    state_cfg = getattr(app.state, "embedding_config", None)
    if state_cfg:
        return _build_client_from_cfg(state_cfg, get_embedding_client)

    # 2. Try config.ini [Embeddings] section
    cfg = _read_embedding_section()
    if cfg["provider_type"] != _DEFAULT_EMBEDDING_CONFIG["provider_type"] or cfg["api_key"]:
        # Non-default settings were found in config; use the factory
        return _build_client_from_cfg(cfg, get_embedding_client)

    # 3. Legacy fallback – keeps existing behaviour intact
    logger.debug("[Settings] No embedding config in state/ini. Using legacy get_embeddings().")
    return get_embeddings(provider="local")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_client_from_cfg(cfg: dict, factory_fn):
    """
    Helper to bridge the configuration dictionary to the embedding factory.

    Maps 'provider_type', 'model_name', and 'api_key' keys to the
    corresponding arguments in `backend.llm_integration.get_embedding_client`.

    Args:
        cfg (dict): Configuration dictionary containing provider details.
        factory_fn (callable): The `get_embedding_client` factory function.

    Returns:
        Embeddings: The instantiated LangChain embedding client.
    """
    return factory_fn(
        provider_type=cfg["provider_type"],
        model_name=cfg.get("model_name") or _DEFAULT_EMBEDDING_CONFIG["model_name"],
        api_key=cfg.get("api_key") or None,
    )


def _read_embedding_section() -> dict:
    """
    Reads the `[Embeddings]` section from `config.ini`.

    If the section or specific keys are missing, it returns the global 
    application defaults for embedding providers.

    Returns:
        dict: A dictionary containing 'provider_type', 'model_name', and 'api_key'.
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    if not config.has_section("Embeddings"):
        return dict(_DEFAULT_EMBEDDING_CONFIG)

    return {
        "provider_type": config.get("Embeddings", "provider_type",
                                    fallback=_DEFAULT_EMBEDDING_CONFIG["provider_type"]),
        "model_name": config.get("Embeddings", "model_name",
                                 fallback=_DEFAULT_EMBEDDING_CONFIG["model_name"]),
        "api_key": config.get("Embeddings", "api_key",
                              fallback=_DEFAULT_EMBEDDING_CONFIG["api_key"]),
    }


def _write_embedding_section(cfg: dict) -> None:
    """
    Writes the embedding configuration to the `[Embeddings]` section in `config.ini`.

    This modification is non-destructive and preserves all other sections
    present in the configuration file.

    Args:
        cfg (dict): The configuration values to persist.
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    if not config.has_section("Embeddings"):
        config.add_section("Embeddings")

    config.set("Embeddings", "provider_type", cfg["provider_type"])
    config.set("Embeddings", "model_name", cfg["model_name"])
    config.set("Embeddings", "api_key", cfg["api_key"])

    with open(CONFIG_PATH, "w") as fh:
        config.write(fh)


def seed_app_state(app) -> None:
    """
    Seeds the in-memory application state with current embedding settings.

    This should be called during the FastAPI startup event to ensure
    downstream logic (like specialized search or re-indexing) has access
     to the active configuration without disk IO overhead.

    Args:
        app: The FastAPI application instance to seed.
    """
    cfg = _read_embedding_section()
    app.state.embedding_config = cfg
    logger.info(
        "[Settings] Seeded embedding config on startup: provider_type=%s, model_name=%s",
        cfg["provider_type"],
        cfg["model_name"],
    )
