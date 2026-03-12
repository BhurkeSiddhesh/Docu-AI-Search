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
    Return the current embedding configuration.
    The raw API key is never sent over the wire; only a boolean
    ``api_key_set`` flag is exposed.
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
    Persist the new embedding configuration.

    * ``'huggingface_api'`` requires ``api_key``.
    * ``'commercial_api'`` requires ``api_key``.
    * ``'local'`` does not require ``api_key``.

    The updated config is written to ``config.ini`` **and** cached on
    ``app.state.embedding_config`` so downstream code picks it up
    immediately without reading from disk.
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
    Return a LangChain embeddings object for the currently configured provider.

    Resolution order:
      1. ``app.state.embedding_config`` (set on startup / POST)
      2. ``[Embeddings]`` section in config.ini
      3. Legacy fallback via ``get_embeddings()`` (local HuggingFace)

    No API keys are ever logged.
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
    """Call the factory with the resolved config dict."""
    return factory_fn(
        provider_type=cfg["provider_type"],
        model_name=cfg.get("model_name") or _DEFAULT_EMBEDDING_CONFIG["model_name"],
        api_key=cfg.get("api_key") or None,
    )


def _read_embedding_section() -> dict:
    """Read ``[Embeddings]`` from config.ini, returning defaults if absent."""
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
    """Write ``[Embeddings]`` section to config.ini (preserves other sections)."""
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
    Called once from ``api.py``'s startup event to pre-populate
    ``app.state.embedding_config`` from disk so the first search/index
    request doesn't have to read config.ini.
    """
    cfg = _read_embedding_section()
    app.state.embedding_config = cfg
    logger.info(
        "[Settings] Seeded embedding config on startup: provider_type=%s, model_name=%s",
        cfg["provider_type"],
        cfg["model_name"],
    )
