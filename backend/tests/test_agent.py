"""
Comprehensive tests for the ReActAgent class in backend/agent.py.

Covers initialization, internal helper methods, and the full stream_chat
async generator loop including tool execution, forced searches, error
handling, and max-step termination.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import asyncio
import sys
import os

# ── Stub out heavy / broken transitive dependencies before any backend import ──
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

from backend.agent import ReActAgent, _DIRECT_ANSWER_PREFIXES


# ── Helpers ──────────────────────────────────────────────────────────────────

def collect_events(agent, query):
    """Drain the stream_chat async generator and return a list of events."""
    async def _collect():
        events = []
        async for event in agent.stream_chat(query):
            events.append(event)
        return events
    return asyncio.run(_collect())


def _make_config(provider="openai", model_path="models/test.gguf", api_key="test_key"):
    config = MagicMock()

    def get_side_effect(section, key, fallback=None):
        if section == "LocalLLM" and key == "provider":
            return provider
        if section == "LocalLLM" and key == "model_path":
            return model_path
        if section == "APIKeys":
            return api_key
        return fallback

    config.get.side_effect = get_side_effect
    return config


def _make_agent(provider="openai"):
    return ReActAgent({"config": _make_config(provider=provider)})


# ── Initialization ────────────────────────────────────────────────────────────

class TestReActAgentInit(unittest.TestCase):

    def test_local_provider_uses_reduced_limits(self):
        agent = _make_agent(provider="local")
        self.assertTrue(agent.is_local)
        self.assertEqual(agent.max_steps, 4)
        self.assertEqual(agent.max_tokens, 384)
        self.assertEqual(agent.obs_window, 350)

    def test_cloud_provider_uses_full_limits(self):
        agent = _make_agent(provider="openai")
        self.assertFalse(agent.is_local)
        self.assertEqual(agent.max_steps, 5)
        self.assertEqual(agent.max_tokens, 512)
        self.assertEqual(agent.obs_window, 800)

    def test_gemini_provider_treated_as_cloud(self):
        agent = _make_agent(provider="gemini")
        self.assertFalse(agent.is_local)
        self.assertEqual(agent.max_steps, 5)

    def test_global_state_reference_stored(self):
        state = {"config": _make_config(), "index": "idx"}
        agent = ReActAgent(state)
        self.assertIs(agent.global_state, state)

    def test_local_model_uses_simplified_prompt(self):
        local_agent = _make_agent(provider="local")
        cloud_agent = _make_agent(provider="openai")
        self.assertNotEqual(local_agent.system_prompt, cloud_agent.system_prompt)
        self.assertIn("STRICT RULES", local_agent.system_prompt)

    def test_api_key_extracted_from_config(self):
        config = _make_config(provider="openai", api_key="sk-test-abc")
        agent = ReActAgent({"config": config})
        self.assertEqual(agent.api_key, "sk-test-abc")


# ── _extract_final_answer ─────────────────────────────────────────────────────

class TestExtractFinalAnswer(unittest.TestCase):

    def setUp(self):
        self.agent = _make_agent()

    def test_standard_final_answer_prefix(self):
        text = "Thought: I found it.\nFinal Answer: The company was founded in 1990."
        result = self.agent._extract_final_answer(text)
        self.assertEqual(result, "The company was founded in 1990.")

    def test_lowercase_final_answer_prefix(self):
        text = "Thinking...\nfinal answer: revenue grew by 25%"
        result = self.agent._extract_final_answer(text)
        self.assertEqual(result, "revenue grew by 25%")

    def test_answer_colon_prefix_at_line_start(self):
        text = "Answer: The headquarters is in New York."
        result = self.agent._extract_final_answer(text)
        self.assertEqual(result, "The headquarters is in New York.")

    def test_returns_none_when_no_final_answer(self):
        text = "I need to search for more information about this topic."
        self.assertIsNone(self.agent._extract_final_answer(text))

    def test_returns_none_for_empty_string(self):
        self.assertIsNone(self.agent._extract_final_answer(""))

    def test_returns_last_occurrence_when_multiple(self):
        text = "Final Answer: first\nFinal Answer: second"
        result = self.agent._extract_final_answer(text)
        self.assertEqual(result, "second")

    def test_strips_whitespace_from_result(self):
        text = "Final Answer:   padded answer   "
        result = self.agent._extract_final_answer(text)
        self.assertEqual(result, "padded answer")


# ── _is_grounded_direct_answer ────────────────────────────────────────────────

class TestIsGroundedDirectAnswer(unittest.TestCase):

    def setUp(self):
        self.agent = _make_agent()

    def test_based_on_document_prefix_accepted(self):
        text = "Based on the document, the company revenue was $5M in 2023 and grew steadily."
        self.assertTrue(self.agent._is_grounded_direct_answer(text))

    def test_according_to_file_prefix_accepted(self):
        text = "According to the file, the project deadline is March 31st and the budget is $100k."
        self.assertTrue(self.agent._is_grounded_direct_answer(text))

    def test_from_the_document_prefix_accepted(self):
        text = "From the document, Siddhesh holds an MBA and worked at three companies."
        self.assertTrue(self.agent._is_grounded_direct_answer(text))

    def test_the_document_says_prefix_accepted(self):
        text = "The document says the policy was updated in 2024 and applies globally now."
        self.assertTrue(self.agent._is_grounded_direct_answer(text))

    def test_too_short_text_rejected(self):
        text = "Based on the document, yes."
        self.assertFalse(self.agent._is_grounded_direct_answer(text))

    def test_contains_action_colon_rejected(self):
        text = "Based on the document, I need to take action: search for more information about this."
        self.assertFalse(self.agent._is_grounded_direct_answer(text))

    def test_no_grounded_prefix_rejected(self):
        text = "This is a long enough text that does not start with any recognized grounded prefix at all."
        self.assertFalse(self.agent._is_grounded_direct_answer(text))

    def test_all_defined_prefixes_accepted(self):
        suffix = " detailed answer that is certainly longer than sixty characters in total."
        for prefix in _DIRECT_ANSWER_PREFIXES:
            text = prefix.capitalize() + suffix
            with self.subTest(prefix=prefix):
                self.assertTrue(self.agent._is_grounded_direct_answer(text))


# ── _force_search_action ──────────────────────────────────────────────────────

class TestForceSearchAction(unittest.TestCase):

    def setUp(self):
        self.agent = _make_agent()

    def test_always_returns_search_knowledge_base(self):
        action, _ = self.agent._force_search_action("anything here")
        self.assertEqual(action, "search_knowledge_base")

    def test_strips_what_question_word(self):
        _, search_input = self.agent._force_search_action("what is machine learning?")
        self.assertNotIn("what", search_input.lower())
        self.assertIn("machine", search_input.lower())

    def test_strips_who_question_word(self):
        _, search_input = self.agent._force_search_action("who founded the company")
        self.assertNotIn("who", search_input.lower())

    def test_strips_how_question_word(self):
        _, search_input = self.agent._force_search_action("how does this work exactly")
        self.assertNotIn("how", search_input.lower())

    def test_falls_back_to_full_query_when_stripped_too_short(self):
        _, search_input = self.agent._force_search_action("why")
        self.assertEqual(search_input, "why")

    def test_preserves_meaningful_content_after_stripping(self):
        _, search_input = self.agent._force_search_action("where is the revenue report")
        self.assertIn("revenue", search_input.lower())
        self.assertIn("report", search_input.lower())


# ── stream_chat ───────────────────────────────────────────────────────────────

class TestStreamChat(unittest.TestCase):

    @patch("backend.llm_integration.generate_ai_answer")
    @patch("backend.tools.AVAILABLE_TOOLS")
    def test_search_then_final_answer_yields_answer_event(self, mock_tools, mock_generate):
        mock_tools.get.return_value = MagicMock(
            return_value="Source: doc.pdf\nContent: relevant information found here"
        )
        mock_generate.side_effect = [
            "Thought: I need to search.\nAction: search_knowledge_base\nAction Input: machine learning",
            "Thought: Found info.\nFinal Answer: Machine learning is a subset of AI.",
        ]

        events = collect_events(_make_agent(), "what is machine learning?")
        types = [e["type"] for e in events]

        self.assertIn("observation", types)
        self.assertIn("answer", types)
        answer_events = [e for e in events if e["type"] == "answer"]
        self.assertIn("Machine learning", answer_events[0]["content"])

    @patch("backend.llm_integration.generate_ai_answer")
    @patch("backend.tools.AVAILABLE_TOOLS")
    def test_answer_event_comes_after_observation(self, mock_tools, mock_generate):
        mock_tools.get.return_value = MagicMock(return_value="search results here")
        mock_generate.side_effect = [
            "Action: search_knowledge_base\nAction Input: query",
            "Final Answer: Answer based on results.",
        ]

        events = collect_events(_make_agent(), "test question")
        types = [e["type"] for e in events]

        obs_idx = types.index("observation")
        ans_idx = types.index("answer")
        self.assertGreater(ans_idx, obs_idx)

    @patch("backend.llm_integration.generate_ai_answer")
    def test_llm_error_prefix_yields_error_event(self, mock_generate):
        mock_generate.return_value = "Error: LLM is unavailable"

        events = collect_events(_make_agent(), "test question")
        types = [e["type"] for e in events]

        self.assertIn("error", types)
        error_events = [e for e in events if e["type"] == "error"]
        self.assertIn("LLM Error", error_events[0]["content"])

    @patch("backend.llm_integration.generate_ai_answer")
    def test_generate_raises_exception_yields_error_event(self, mock_generate):
        mock_generate.side_effect = RuntimeError("network timeout")

        events = collect_events(_make_agent(), "test question")
        types = [e["type"] for e in events]

        self.assertIn("error", types)

    @patch("backend.llm_integration.generate_ai_answer")
    @patch("backend.tools.AVAILABLE_TOOLS")
    def test_premature_final_answer_forces_search_first(self, mock_tools, mock_generate):
        """Agent must not emit 'answer' if no search has happened yet."""
        mock_tools.get.return_value = MagicMock(return_value="relevant docs here")
        mock_generate.side_effect = [
            "Final Answer: I already know the answer.",      # blocked — no search yet
            "Final Answer: Answer derived from the search.",  # accepted after search
        ]

        events = collect_events(_make_agent(), "what is the revenue?")
        types = [e["type"] for e in events]

        # A 'thought' should have been emitted to announce the forced search
        self.assertIn("thought", types)
        # Eventually an answer arrives (after the forced search)
        self.assertIn("answer", types)
        # Observation must precede the answer
        self.assertIn("observation", types)

    @patch("backend.llm_integration.generate_ai_answer")
    @patch("backend.tools.AVAILABLE_TOOLS")
    def test_max_steps_exceeded_yields_error(self, mock_tools, mock_generate):
        """After exhausting all steps without a final answer, emit an error."""
        mock_tools.get.return_value = MagicMock(return_value="some results")
        # Always produce a non-terminal response (short thought, no action after search)
        mock_generate.return_value = "I'm still thinking about this short response."

        agent = _make_agent(provider="local")  # max_steps=4 for faster termination
        events = collect_events(agent, "unanswerable question")
        types = [e["type"] for e in events]

        self.assertIn("error", types)
        error_event = next(e for e in events if e["type"] == "error")
        self.assertIn("unable to find", error_event["content"].lower())

    @patch("backend.llm_integration.generate_ai_answer")
    @patch("backend.tools.AVAILABLE_TOOLS")
    def test_tool_exception_yields_error_observation_not_crash(self, mock_tools, mock_generate):
        """If a tool raises, the agent should record it as an observation and keep going."""
        mock_tools.get.return_value = MagicMock(side_effect=Exception("Tool crashed"))
        mock_generate.return_value = (
            "Thought: Search.\nAction: search_knowledge_base\nAction Input: query"
        )

        agent = _make_agent(provider="local")  # max_steps=4
        events = collect_events(agent, "test question")
        types = [e["type"] for e in events]

        self.assertIn("observation", types)
        obs_events = [e for e in events if e["type"] == "observation"]
        self.assertTrue(any("Error" in e["content"] for e in obs_events))

    @patch("backend.llm_integration.generate_ai_answer")
    @patch("backend.tools.AVAILABLE_TOOLS")
    def test_list_files_tool_is_dispatched(self, mock_tools, mock_generate):
        mock_list_files = MagicMock(return_value="report.pdf, budget.xlsx")

        def get_tool(name):
            return mock_list_files if name == "list_files" else None

        mock_tools.get.side_effect = get_tool
        mock_generate.side_effect = [
            "Thought: List files.\nAction: list_files\nAction Input: ",
            "Final Answer: The indexed files are report.pdf and budget.xlsx.",
        ]

        events = collect_events(_make_agent(), "what files are indexed?")
        types = [e["type"] for e in events]

        self.assertIn("observation", types)
        obs = [e for e in events if e["type"] == "observation"][0]["content"]
        self.assertIn("report.pdf", obs)

    @patch("backend.llm_integration.generate_ai_answer")
    @patch("backend.tools.AVAILABLE_TOOLS")
    def test_fallback_function_call_syntax_parsed(self, mock_tools, mock_generate):
        """Agent should parse tool-call style syntax: search_knowledge_base('query')."""
        mock_tools.get.return_value = MagicMock(return_value="fallback results here")
        mock_generate.side_effect = [
            "search_knowledge_base('annual revenue 2023')",
            "Final Answer: Revenue was $5M.",
        ]

        events = collect_events(_make_agent(), "what is the revenue?")
        types = [e["type"] for e in events]

        self.assertIn("observation", types)

    @patch("backend.llm_integration.generate_ai_answer")
    @patch("backend.tools.AVAILABLE_TOOLS")
    def test_unknown_tool_yields_error_observation(self, mock_tools, mock_generate):
        """Calling an unregistered tool should produce an error observation."""
        mock_tools.get.return_value = None  # Tool not found
        mock_generate.side_effect = [
            "Action: nonexistent_tool\nAction Input: something",
            "Final Answer: Based on the results obtained from the document search.",
        ]

        events = collect_events(_make_agent(), "test question")
        obs_events = [e for e in events if e["type"] == "observation"]

        self.assertTrue(len(obs_events) > 0)
        self.assertTrue(any("not found" in e["content"].lower() for e in obs_events))

    @patch("backend.llm_integration.generate_ai_answer")
    @patch("backend.tools.AVAILABLE_TOOLS")
    def test_generate_called_with_correct_parameters(self, mock_tools, mock_generate):
        """Verify that generate_ai_answer receives the expected keyword arguments."""
        mock_tools.get.return_value = MagicMock(return_value="results")
        mock_generate.side_effect = [
            "Action: search_knowledge_base\nAction Input: test",
            "Final Answer: Done.",
        ]

        agent = _make_agent(provider="openai")
        collect_events(agent, "test query")

        first_call_kwargs = mock_generate.call_args_list[0].kwargs
        self.assertTrue(first_call_kwargs.get("raw"))
        self.assertEqual(first_call_kwargs.get("provider"), "openai")
        self.assertEqual(first_call_kwargs.get("temperature"), 0.1)

    @patch("backend.llm_integration.generate_ai_answer")
    @patch("backend.tools.AVAILABLE_TOOLS")
    def test_grounded_direct_answer_accepted_after_search(self, mock_tools, mock_generate):
        """A long grounded answer (no 'Final Answer:' token) should be accepted after search."""
        mock_tools.get.return_value = MagicMock(return_value="search results from docs")

        long_grounded = (
            "Based on the document, the quarterly report shows revenue of $10M, "
            "representing a 15% increase year-over-year driven by product expansion."
        )
        mock_generate.side_effect = [
            "Action: search_knowledge_base\nAction Input: revenue",
            long_grounded,
        ]

        events = collect_events(_make_agent(), "what is the revenue?")
        types = [e["type"] for e in events]

        self.assertIn("answer", types)
        answer_content = next(e["content"] for e in events if e["type"] == "answer")
        self.assertIn("revenue", answer_content.lower())


if __name__ == "__main__":
    unittest.main()
