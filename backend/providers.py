"""
backend/providers.py
--------------------
Unified LLM provider abstraction layer.

Defines an ``LLMProvider`` ABC that normalises access to:
* Local GGUF models (via llama-cpp-python)
* Ollama (REST API at localhost:11434)
* LM Studio / any OpenAI-compatible server (REST API at localhost:1234/v1)
* Cloud APIs (OpenAI, Gemini, Anthropic, Grok — via LangChain wrappers)

Usage
-----
>>> provider = get_provider("ollama", {"base_url": "http://localhost:11434", "model": "llama3"})
>>> answer   = provider.generate("Summarise this document", system_prompt="You are a helpful assistant")
>>> for token in provider.stream("Explain quantum computing"):
...     print(token, end="")
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Common interface that every LLM backend must implement."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Return the full completion as a string."""

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """Yield tokens one-by-one as they arrive."""

    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """Return available models from the backend."""

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return ``{"status": "ok", ...}`` or ``{"status": "error", "message": ...}``."""


# ---------------------------------------------------------------------------
# Ollama  (http://localhost:11434)
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Connector for a running Ollama instance.

    Default endpoint: ``http://localhost:11434``
    Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._timeout = 120  # seconds

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # -- generate -----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if stop:
            payload["options"]["stop"] = stop

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please make sure Ollama is running (`ollama serve`)."
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e.response.status_code} — {e.response.text}")

    # -- stream -------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if stop:
            payload["options"]["stop"] = stop

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        return
                except json.JSONDecodeError:
                    continue
        except requests.ConnectionError:
            yield f"[Error] Cannot connect to Ollama at {self.base_url}. Is it running?"

    # -- list_models --------------------------------------------------------

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                headers=self._headers(),
                timeout=10,
            )
            resp.raise_for_status()
            models_raw = resp.json().get("models", [])
            return [
                {
                    "id": m.get("name", "unknown"),
                    "name": m.get("name", "unknown"),
                    "size": m.get("size", 0),
                    "modified_at": m.get("modified_at", ""),
                    "details": m.get("details", {}),
                }
                for m in models_raw
            ]
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please make sure Ollama is running (`ollama serve`)."
            )
        except Exception as e:
            logger.error("Failed to list Ollama models: %s", e)
            return []

    # -- health_check -------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", headers=self._headers(), timeout=5)
            resp.raise_for_status()
            model_count = len(resp.json().get("models", []))
            return {"status": "ok", "provider": "ollama", "models_available": model_count}
        except requests.ConnectionError:
            return {
                "status": "error",
                "provider": "ollama",
                "message": f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running (`ollama serve`).",
            }
        except Exception as e:
            return {"status": "error", "provider": "ollama", "message": str(e)}


# ---------------------------------------------------------------------------
# OpenAI-Compatible  (LM Studio, vLLM, text-generation-webui, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider(LLMProvider):
    """Connector for LM Studio and other OpenAI-compatible servers.

    Supports two API formats:
    - **LM Studio native**: ``/api/v1/chat`` with ``input`` / ``system_prompt`` fields.
    - **OpenAI-compatible**: ``/v1/chat/completions`` with ``messages`` array.

    Auto-detection: If ``base_url`` does NOT already end with ``/v1``, the provider
    uses the LM Studio native API.  If it ends with ``/v1``, it uses OpenAI format.

    Default endpoint: ``http://127.0.0.1:1234`` (LM Studio default).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:1234", model: str = "", api_key: str = "lm-studio"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "lm-studio"
        self._timeout = 180  # Large models may need more time
        # Auto-detect API format: if URL ends with /v1, use OpenAI format
        self._use_openai_format = self.base_url.endswith("/v1")

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # -- generate -----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Optional[List[str]] = None,
    ) -> str:
        if self._use_openai_format:
            return self._generate_openai(prompt, system_prompt=system_prompt,
                                         max_tokens=max_tokens, temperature=temperature, stop=stop)
        return self._generate_native(prompt, system_prompt=system_prompt,
                                     max_tokens=max_tokens, temperature=temperature, stop=stop)

    def _generate_native(self, prompt, *, system_prompt=None, max_tokens=512, temperature=0.3, stop=None) -> str:
        """LM Studio native API: POST /api/v1/chat

        Note: The native `/api/v1/chat` endpoint only accepts `model`, `input`,
        `system_prompt`, and `stream`. Advanced sampling params (max_tokens,
        temperature, stop) are NOT supported on this path — use the OpenAI-compat
        `/v1/chat/completions` endpoint for those.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt

        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/chat",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # LM Studio native returns choices[].message.content (same shape)
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            # Fallback: direct "content" or "response" key
            return (data.get("content") or data.get("response") or "").strip()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}. "
                "Please make sure LM Studio is running."
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"LM Studio HTTP error: {e.response.status_code} — {e.response.text}")

    def _generate_openai(self, prompt, *, system_prompt=None, max_tokens=512, temperature=0.3, stop=None) -> str:
        """OpenAI-compatible API: POST /chat/completions"""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(prompt, system_prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to the OpenAI-compatible server at {self.base_url}. "
                "Please make sure LM Studio (or your server) is running."
            )
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected response format from {self.base_url}")
        except requests.HTTPError as e:
            raise RuntimeError(f"HTTP error from {self.base_url}: {e.response.status_code} — {e.response.text}")

    # -- stream -------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        if self._use_openai_format:
            yield from self._stream_openai(prompt, system_prompt=system_prompt,
                                           max_tokens=max_tokens, temperature=temperature, stop=stop)
        else:
            yield from self._stream_native(prompt, system_prompt=system_prompt,
                                           max_tokens=max_tokens, temperature=temperature, stop=stop)

    def _stream_native(self, prompt, *, system_prompt=None, max_tokens=512, temperature=0.3, stop=None):
        """LM Studio native streaming: POST /api/v1/chat with stream=True"""
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "stream": True,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt

        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/chat",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except requests.ConnectionError:
            yield f"[Error] Cannot connect to LM Studio at {self.base_url}. Is it running?"

    def _stream_openai(self, prompt, *, system_prompt=None, max_tokens=512, temperature=0.3, stop=None):
        """OpenAI-compatible streaming: POST /chat/completions with stream=True"""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(prompt, system_prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except requests.ConnectionError:
            yield f"[Error] Cannot connect to server at {self.base_url}. Is it running?"

    # -- list_models --------------------------------------------------------

    def list_models(self) -> List[Dict[str, Any]]:
        """Fetch models from LM Studio /api/v1/models or OpenAI /v1/models."""
        if self._use_openai_format:
            url = f"{self.base_url}/models"
        else:
            url = f"{self.base_url}/api/v1/models"

        try:
            resp = requests.get(url, headers=self._headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # LM Studio native uses "models" key; OpenAI uses "data" key
            models_raw = data.get("data") or data.get("models", [])
            results = []
            for m in models_raw:
                # LM Studio native: model has 'key' (e.g. qwen2.5-0.5b-instruct)
                # OpenAI-compat:    model has 'id'
                model_id = m.get("id") or m.get("key") or m.get("path") or "unknown"
                model_name = m.get("display_name") or model_id
                results.append({
                    "id": model_id,
                    "name": model_name,
                    "owned_by": m.get("owned_by") or m.get("publisher", ""),
                    "type": m.get("type", "llm"),
                    "format": m.get("format", ""),
                    "size_bytes": m.get("size_bytes", 0),
                })
            return results
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to server at {self.base_url}. "
                "Please make sure LM Studio (or your server) is running."
            )
        except Exception as e:
            logger.error("Failed to list models from %s: %s", self.base_url, e)
            return []

    # -- health_check -------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        if self._use_openai_format:
            url = f"{self.base_url}/models"
        else:
            url = f"{self.base_url}/api/v1/models"

        try:
            resp = requests.get(url, headers=self._headers(), timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models_list = data.get("data") or data.get("models", [])
            model_count = len(models_list)
            return {"status": "ok", "provider": "lmstudio", "models_available": model_count}
        except requests.ConnectionError:
            return {
                "status": "error",
                "provider": "lmstudio",
                "message": f"Cannot connect to LM Studio at {self.base_url}. Is it running?",
            }
        except Exception as e:
            return {"status": "error", "provider": "lmstudio", "message": str(e)}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Provider type constants
PROVIDER_OLLAMA = "ollama"
PROVIDER_LMSTUDIO = "lmstudio"
PROVIDER_OPENAI_COMPAT = "openai_compatible"

# Default endpoint URLs
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LMSTUDIO_URL = "http://127.0.0.1:1234"

# Provider cache to avoid re-creating instances
_provider_cache: Dict[str, LLMProvider] = {}


def get_provider(provider_type: str, config: Optional[Dict[str, Any]] = None) -> LLMProvider:
    """Factory that returns the correct ``LLMProvider`` for the given type.

    Args:
        provider_type: One of ``'ollama'``, ``'lmstudio'``, ``'openai_compatible'``.
        config: Dict with optional keys:
            - ``base_url``: Override the default endpoint.
            - ``model``: The model name/tag to use.
            - ``api_key``: API key (optional for local, required for some remote).

    Returns:
        An ``LLMProvider`` instance (cached by ``provider_type + base_url``).

    Raises:
        ValueError: If ``provider_type`` is not recognised.
    """
    config = config or {}
    base_url = config.get("base_url", "")
    model = config.get("model", "")
    api_key = config.get("api_key", "")

    if provider_type == PROVIDER_OLLAMA:
        url = base_url or DEFAULT_OLLAMA_URL
        cache_key = f"ollama:{url}:{model}:{api_key}"
        if cache_key not in _provider_cache:
            _provider_cache[cache_key] = OllamaProvider(base_url=url, model=model, api_key=api_key)
        else:
            _provider_cache[cache_key].model = model
        return _provider_cache[cache_key]

    if provider_type in (PROVIDER_LMSTUDIO, PROVIDER_OPENAI_COMPAT):
        url = base_url or DEFAULT_LMSTUDIO_URL
        cache_key = f"openai_compat:{url}:{model}:{api_key}"
        if cache_key not in _provider_cache:
            _provider_cache[cache_key] = OpenAICompatibleProvider(base_url=url, model=model, api_key=api_key)
        else:
            _provider_cache[cache_key].model = model
        return _provider_cache[cache_key]

    raise ValueError(
        f"Unknown provider_type '{provider_type}'. "
        f"Supported: {PROVIDER_OLLAMA}, {PROVIDER_LMSTUDIO}, {PROVIDER_OPENAI_COMPAT}"
    )


def clear_provider_cache() -> None:
    """Remove all cached provider instances (useful in tests)."""
    _provider_cache.clear()
