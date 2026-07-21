"""
backend/tests/test_bugfixes_streaming_indexing.py
-------------------------------------------------
Regression tests for the 2026-07 fixes:

* Local/Gemma empty-response handling in stream_ai_answer (empty chat stream
  must fall back to raw completion, and a fully-empty generation must emit a
  visible sentinel instead of a silent blank).
* Gemma chat-message construction (no `system` role).
* LM Studio / external provider streaming surfaces HTTP errors.
* mark_folder_indexed upserts and normalizes paths.

All LLM calls are mocked — no models or servers required.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend import llm_integration
from backend.llm_integration import (
    _is_gemma_model,
    _build_local_chat_messages,
    stream_ai_answer,
)


def _fake_stream(tokens):
    """Build a llama-cpp-style chat stream of delta chunks."""
    return [{"choices": [{"delta": {"content": t}}]} for t in tokens]


def _fake_completion_stream(tokens):
    """Build a llama-cpp-style raw completion stream."""
    return [{"choices": [{"text": t}]} for t in tokens]


class TestGemmaMessageBuilding(unittest.TestCase):
    def test_is_gemma_model(self):
        self.assertTrue(_is_gemma_model("/models/gemma-2-2b-it.Q4_K_M.gguf"))
        self.assertTrue(_is_gemma_model("GEMMA-9b.gguf"))
        self.assertFalse(_is_gemma_model("/models/llama-3-8b-instruct.gguf"))
        self.assertFalse(_is_gemma_model(""))

    def test_gemma_folds_system_into_user(self):
        """Gemma has no system role — the system prompt is prepended to user."""
        msgs = _build_local_chat_messages("SYS", "USER", "/m/gemma-2-2b-it.gguf")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertIn("SYS", msgs[0]["content"])
        self.assertIn("USER", msgs[0]["content"])

    def test_non_gemma_keeps_system_role(self):
        msgs = _build_local_chat_messages("SYS", "USER", "/m/llama-3.gguf")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


class TestLocalStreamEmptyHandling(unittest.TestCase):
    """stream_ai_answer local branch: empty output must not be a silent blank."""

    def _run_with(self, chat_tokens, raw_tokens):
        fake_llm = MagicMock()
        fake_llm.create_chat_completion.return_value = _fake_stream(chat_tokens)
        fake_llm.create_completion.return_value = _fake_completion_stream(raw_tokens)
        with patch.object(llm_integration, "get_llm_client", return_value="LOCAL:/m/gemma-2b.gguf"), \
             patch.object(llm_integration, "get_local_llm", return_value=fake_llm):
            return list(stream_ai_answer("ctx", "q", "local", model_path="/m/gemma-2b.gguf")), fake_llm

    def test_empty_chat_falls_back_to_raw(self):
        """A successful-but-empty chat stream ([]) must trigger the raw fallback."""
        tokens, fake_llm = self._run_with(chat_tokens=[], raw_tokens=["Hello ", "world"])
        self.assertEqual("".join(tokens), "Hello world")
        fake_llm.create_completion.assert_called_once()

    def test_non_empty_chat_used_directly(self):
        tokens, fake_llm = self._run_with(chat_tokens=["Answer"], raw_tokens=["SHOULD_NOT_APPEAR"])
        self.assertEqual("".join(tokens), "Answer")
        fake_llm.create_completion.assert_not_called()

    def test_all_empty_emits_sentinel(self):
        """When both chat and raw yield nothing, a visible sentinel is emitted."""
        tokens, _ = self._run_with(chat_tokens=[], raw_tokens=[])
        joined = "".join(tokens)
        self.assertTrue(joined.startswith("[No answer generated"), joined)


class TestExternalConfigModelName(unittest.TestCase):
    """_build_external_provider_config must resolve model by NAME, never a path."""

    def test_lmstudio_uses_model_override_name(self):
        cfg = llm_integration._build_external_provider_config(
            "lmstudio", base_url_override="http://localhost:1234/v1", model_override="my-model"
        )
        self.assertEqual(cfg["model"], "my-model")
        self.assertEqual(cfg["base_url"], "http://localhost:1234/v1")


class TestExternalStreamHttpError(unittest.TestCase):
    """A bad model id (HTTP 4xx) must surface as an [Error] token, not silence."""

    def test_stream_openai_surfaces_http_error(self):
        import requests
        from backend.providers import OpenAICompatibleProvider, clear_provider_cache
        clear_provider_cache()
        provider = OpenAICompatibleProvider(base_url="http://localhost:1234/v1", model="bad-model")

        err_resp = MagicMock()
        err_resp.status_code = 404
        err_resp.text = "model 'bad-model' not found"
        http_err = requests.HTTPError(response=err_resp)
        post_resp = MagicMock()
        post_resp.raise_for_status.side_effect = http_err

        with patch.object(provider._session, "post", return_value=post_resp):
            tokens = list(provider.stream("hi"))
        joined = "".join(tokens)
        self.assertIn("[Error]", joined)
        self.assertIn("404", joined)


class TestMarkFolderIndexed(unittest.TestCase):
    """mark_folder_indexed upserts and normalizes paths (D3)."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from backend import database
        cls.database = database
        cls._orig_path = database.DATABASE_PATH
        cls._tmp = tempfile.NamedTemporaryFile(delete=False)
        cls._tmp.close()
        database.DATABASE_PATH = cls._tmp.name
        database.init_database()

    @classmethod
    def tearDownClass(cls):
        cls.database.DATABASE_PATH = cls._orig_path
        if os.path.exists(cls._tmp.name):
            os.unlink(cls._tmp.name)

    def _indexed_paths(self):
        return {row["path"] for row in self.database.get_folder_history(indexed_only=True)}

    def test_upserts_folder_absent_from_history(self):
        """A folder never added to history still becomes indexed (upsert)."""
        self.database.mark_folder_indexed("/data/reports")
        self.assertIn("/data/reports", self._indexed_paths())

    def test_normalizes_trailing_slash_and_whitespace(self):
        """A trailing slash / whitespace must match the same folder row."""
        self.database.add_folder_to_history("/data/docs")
        self.database.mark_folder_indexed("  /data/docs/  ")
        indexed = self._indexed_paths()
        self.assertIn("/data/docs", indexed)
        # No duplicate un-normalized row was created.
        self.assertNotIn("/data/docs/", indexed)


if __name__ == "__main__":
    unittest.main()
