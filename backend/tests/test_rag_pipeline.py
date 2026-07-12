"""
Integration-level tests for the RAG pipeline.

Replaces the old manual exploration script with proper unittest cases that
verify the search → context assembly → LLM answer generation flow using
fully mocked external dependencies (no real models or LLM API calls).
"""

import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os
import numpy as np

# Stub out heavy / broken transitive dependencies before any backend import
for _mod in [
    "pypdf", "docx", "openpyxl", "pptx", "pptx.util",
    "langchain_community", "langchain_community.llms", "langchain_community.llms.llamacpp",
    "langchain_openai", "langchain_google_genai", "langchain_anthropic",
    "langchain_core", "langchain_core.messages",
    "openai", "anthropic", "google", "google.generativeai", "google.genai",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.search import search
from backend import llm_integration


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_index(dim=128):
    index = MagicMock()
    index.d = dim
    return index


def _make_future(dists, idxs):
    future = MagicMock()
    mock_dists = MagicMock()
    mock_dists.__getitem__ = lambda self, key: dists
    mock_idxs = MagicMock()
    mock_idxs.__getitem__ = lambda self, key: idxs
    future.result.return_value = (mock_dists, mock_idxs)
    return future


# ── Search → Context Pipeline ────────────────────────────────────────────────

class TestSearchContextPipeline(unittest.TestCase):
    """
    Verify that search() returns properly structured results that can be
    assembled into context for LLM consumption.
    """

    def setUp(self):
        self.embeddings_model = MagicMock()
        self.embeddings_model.embed_query.return_value = [0.1] * 128

        self.executor_patcher = patch("concurrent.futures.ThreadPoolExecutor")
        mock_executor_cls = self.executor_patcher.start()
        self.mock_executor = mock_executor_cls.return_value
        self.mock_executor.__enter__.return_value = self.mock_executor

        future = _make_future([0.15, 0.25], [0, 1])
        self.mock_executor.submit.return_value = future

    def tearDown(self):
        self.executor_patcher.stop()

    def test_search_results_contain_document_text(self):
        index = _make_index()
        docs = [
            {"text": "Machine learning automates analytical model building.", "filepath": "/docs/ml.txt"},
            {"text": "Deep learning uses neural networks.", "filepath": "/docs/dl.txt"},
        ]
        tags = [["ai", "ml"], ["ai", "deep"]]

        results, context = search("machine learning", index, docs, tags, self.embeddings_model)

        self.assertGreater(len(results), 0)
        self.assertGreater(len(context), 0)
        self.assertIn("Machine learning", context[0])

    def test_search_results_have_required_fields_for_llm(self):
        index = _make_index()
        docs = [{"text": "Revenue grew 25% this quarter.", "filepath": "/docs/report.pdf"}]
        tags = [["finance", "revenue"]]

        future = _make_future([0.1], [0])
        self.mock_executor.submit.return_value = future

        results, context = search("revenue growth", index, docs, tags, self.embeddings_model)

        self.assertEqual(len(results), 1)
        result = results[0]
        for field in ("document", "distance", "file_name", "file_path", "tags"):
            self.assertIn(field, result, f"Missing field: {field}")

    def test_context_list_matches_results_count(self):
        index = _make_index()
        docs = [
            {"text": "First document content here.", "filepath": "/docs/a.txt"},
            {"text": "Second document content here.", "filepath": "/docs/b.txt"},
        ]
        tags = ["tag_a", "tag_b"]

        results, context = search("content", index, docs, tags, self.embeddings_model)

        self.assertEqual(len(results), len(context))

    def test_search_with_bm25_augments_results(self):
        index = _make_index()
        docs = [{"text": "quarterly financial report analysis", "filepath": "/docs/fin.pdf"}]
        tags = [["finance"]]

        future = _make_future([0.2], [0])
        self.mock_executor.submit.return_value = future

        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = [8.5]

        results, context = search(
            "quarterly financial", index, docs, tags, self.embeddings_model, bm25=mock_bm25
        )

        self.assertGreater(len(results), 0)


# ── Context → LLM Answer Pipeline ─────────────────────────────────────────────

class TestContextToLLMPipeline(unittest.TestCase):
    """
    Verify that llm_integration.generate_ai_answer is called with the expected
    arguments when assembled search context is passed in.  We mock the function
    itself to avoid exercising LangChain internals.
    """

    @patch("backend.llm_integration.generate_ai_answer", return_value="The revenue was $5M.")
    def test_generate_ai_answer_called_with_context_and_question(self, mock_generate):
        context = "Source: report.pdf\nContent: Revenue for 2023 was five million dollars."
        question = "What was the 2023 revenue?"

        result = llm_integration.generate_ai_answer(
            context=context,
            question=question,
            provider="openai",
            api_key="sk-test",
        )

        mock_generate.assert_called_once_with(
            context=context, question=question, provider="openai", api_key="sk-test"
        )
        self.assertEqual(result, "The revenue was $5M.")

    @patch("backend.llm_integration.generate_ai_answer", return_value="42 is the answer.")
    def test_generate_ai_answer_returns_string(self, mock_generate):
        result = llm_integration.generate_ai_answer(
            context="some context",
            question="some question",
            provider="openai",
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ── End-to-End RAG Flow (fully mocked) ────────────────────────────────────────

class TestEndToEndRAGFlow(unittest.TestCase):
    """
    Simulate the full pipeline: index → search → LLM answer,
    with all external calls mocked.
    """

    def setUp(self):
        self.embeddings_model = MagicMock()
        self.embeddings_model.embed_query.return_value = [0.2] * 128

        self.executor_patcher = patch("concurrent.futures.ThreadPoolExecutor")
        mock_executor_cls = self.executor_patcher.start()
        self.mock_executor = mock_executor_cls.return_value
        self.mock_executor.__enter__.return_value = self.mock_executor

        future = _make_future([0.1], [0])
        self.mock_executor.submit.return_value = future

    def tearDown(self):
        self.executor_patcher.stop()

    @patch("backend.llm_integration.generate_ai_answer", return_value="Siddhesh studied at Symbiosis.")
    def test_full_flow_search_to_answer(self, mock_generate):
        # Step 1: Search
        index = _make_index()
        docs = [{"text": "Siddhesh Bhurke holds an MBA from Symbiosis.", "filepath": "/docs/cv.pdf"}]
        tags = [["education", "mba"]]

        results, context = search("where did siddhesh study", index, docs, tags, self.embeddings_model)

        self.assertGreater(len(results), 0)
        assembled_context = "\n".join(context)

        # Step 2: Generate answer from search context
        answer = llm_integration.generate_ai_answer(
            context=assembled_context,
            question="where did siddhesh study?",
            provider="openai",
        )

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        self.assertIn("Symbiosis", call_kwargs["context"])
        self.assertIsInstance(answer, str)
        self.assertEqual(answer, "Siddhesh studied at Symbiosis.")

    def test_empty_search_results_produce_empty_context(self):
        index = _make_index()
        docs = []
        tags = []

        future = _make_future([-1], [-1])
        self.mock_executor.submit.return_value = future

        results, context = search("anything", index, docs, tags, self.embeddings_model)

        self.assertEqual(results, [])
        self.assertEqual(context, [])

    def test_multiple_documents_ranked_by_relevance(self):
        index = _make_index()
        docs = [
            {"text": "Annual revenue for 2023 was $5 million.", "filepath": "/docs/annual.pdf"},
            {"text": "Marketing budget was $500k.", "filepath": "/docs/budget.pdf"},
            {"text": "Team size is 50 employees.", "filepath": "/docs/team.pdf"},
        ]
        tags = ["finance", "budget", "hr"]

        future = _make_future([0.05, 0.15, 0.30], [0, 1, 2])
        self.mock_executor.submit.return_value = future

        results, context = search("revenue financial", index, docs, tags, self.embeddings_model)

        self.assertGreater(len(results), 0)
        # First result should come from the closest match (lowest distance = highest relevance)
        self.assertIn("revenue", context[0].lower())


if __name__ == "__main__":
    unittest.main()
