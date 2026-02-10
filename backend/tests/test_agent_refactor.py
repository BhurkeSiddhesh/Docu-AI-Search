import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import asyncio

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock heavy dependencies BEFORE import
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()
sys.modules['pdfplumber'] = MagicMock()
sys.modules['docx'] = MagicMock()
sys.modules['pptx'] = MagicMock()
sys.modules['openpyxl'] = MagicMock()
sys.modules['llama_cpp'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Mock langchain related modules
mock_langchain_core = MagicMock()
sys.modules['langchain_core'] = mock_langchain_core
sys.modules['langchain_core.messages'] = mock_langchain_core

# Define dummy classes for messages
class SystemMessage:
    def __init__(self, content):
        self.content = content

class HumanMessage:
    def __init__(self, content):
        self.content = content

# Attach to mock so import works inside llm_integration
mock_langchain_core.SystemMessage = SystemMessage
mock_langchain_core.HumanMessage = HumanMessage

from backend import llm_integration

class TestAgentRefactor(unittest.TestCase):

    @patch('backend.llm_integration.get_llm_client')
    @patch('backend.llm_integration.get_local_llm')
    def test_generate_ai_answer_raw_local(self, mock_get_local, mock_get_client):
        """Test generate_ai_answer with raw=True for Local provider."""
        # Setup mocks
        mock_get_client.return_value = "LOCAL:model.gguf"
        mock_llm = MagicMock()
        mock_llm.create_completion.return_value = {'choices': [{'text': 'Raw response'}]}
        mock_get_local.return_value = mock_llm

        # Call function
        response = llm_integration.generate_ai_answer(
            context="Ignored",
            question="User Prompt",
            provider="local",
            model_path="model.gguf",
            raw=True,
            system_instruction="System Prompt",
            stop=["Stop1"],
            max_tokens=100,
            temperature=0.5
        )

        # Verify response
        self.assertEqual(response, "Raw response")

        # Verify call to create_completion
        mock_llm.create_completion.assert_called_once()
        args, kwargs = mock_llm.create_completion.call_args

        # Expected prompt: System Prompt + \n\n + User Prompt
        expected_prompt = "System Prompt\n\nUser Prompt"
        self.assertEqual(args[0], expected_prompt)
        self.assertEqual(kwargs['max_tokens'], 100)
        self.assertEqual(kwargs['stop'], ["Stop1"])
        self.assertEqual(kwargs['temperature'], 0.5)

    @patch('backend.llm_integration.get_llm_client')
    def test_generate_ai_answer_raw_cloud(self, mock_get_client):
        """Test generate_ai_answer with raw=True for Cloud provider."""
        # Setup mocks
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Cloud response"

        # Mock bind().invoke()
        mock_bind = MagicMock()
        mock_bind.invoke.return_value = mock_response
        mock_client.bind.return_value = mock_bind

        # Also mock invoke directly in case bind is skipped or fails
        mock_client.invoke.return_value = mock_response

        mock_get_client.return_value = mock_client

        # Call function
        response = llm_integration.generate_ai_answer(
            context="Ignored",
            question="User Prompt",
            provider="openai",
            raw=True,
            system_instruction="System Prompt",
            stop=["Stop1"],
            max_tokens=100,
            temperature=0.5
        )

        self.assertEqual(response, "Cloud response")

        # Verify bind was called with stop
        mock_client.bind.assert_called_with(stop=["Stop1"])
        mock_bind.invoke.assert_called_once()

        # Verify messages passed to invoke
        args, _ = mock_bind.invoke.call_args
        messages = args[0]
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertEqual(messages[0].content, "System Prompt")
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertEqual(messages[1].content, "User Prompt")

    @patch('backend.llm_integration.generate_ai_answer')
    def test_agent_calls_generate_ai_answer_correctly(self, mock_generate):
        """Verify ReActAgent calls generate_ai_answer with correct params."""
        # Setup
        mock_generate.return_value = "Thought: I should search.\nAction: search_knowledge_base\nAction Input: query"
        # Mock config object
        mock_config = MagicMock()
        def get_side_effect(section, key, fallback=None):
            if section == "LocalLLM" and key == "provider": return "local"
            if section == "LocalLLM" and key == "model_path": return "model.gguf"
            if section == "APIKeys": return ""
            return fallback
        mock_config.get.side_effect = get_side_effect

        global_state = {"config": mock_config}

        # We need to import ReActAgent inside the test method or ensure it is imported after mocks
        from backend.agent import ReActAgent
        agent = ReActAgent(global_state)

        # Run one step of stream_chat
        async def run_agent():
            gen = agent.stream_chat("Hello")
            try:
                # Need to iterate enough to trigger the call
                async for _ in gen:
                    break
            except Exception:
                pass

        asyncio.run(run_agent())

        # Verify mock call
        mock_generate.assert_called()
        kwargs = mock_generate.call_args.kwargs

        self.assertTrue(kwargs['raw'])
        self.assertIn("AVAILABLE TOOLS", kwargs['system_instruction'])
        self.assertIn("Question: Hello", kwargs['question'])
        self.assertEqual(kwargs['stop'], ["Observation:", "Definition:", "Thought:"])
        self.assertEqual(kwargs['max_tokens'], 256)
        self.assertEqual(kwargs['temperature'], 0.1)

if __name__ == '__main__':
    unittest.main()
