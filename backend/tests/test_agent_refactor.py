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
sys.modules['numpy'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['psutil'] = MagicMock()

# Mock langchain related modules
mock_langchain_core = MagicMock()
sys.modules['langchain_core'] = mock_langchain_core
sys.modules['langchain_core.messages'] = mock_langchain_core

# Define dummy classes
class SystemMessage:
    def __init__(self, content):
        self.content = content

class HumanMessage:
    def __init__(self, content):
        self.content = content

# Attach to mock
mock_langchain_core.SystemMessage = SystemMessage
mock_langchain_core.HumanMessage = HumanMessage

from backend import llm_integration

class TestAgentRefactor(unittest.TestCase):

    @patch('backend.llm_integration.get_llm_client')
    @patch('backend.llm_integration.get_local_llm')
    def test_generate_ai_answer_raw_local(self, mock_get_local, mock_get_client):
        """Test generate_ai_answer with raw=True for Local provider."""
        mock_get_client.return_value = "LOCAL:model.gguf"
        mock_llm = MagicMock()
        mock_llm.create_completion.return_value = {'choices': [{'text': 'Raw response'}]}
        mock_get_local.return_value = mock_llm

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

        self.assertEqual(response, "Raw response")

        args, kwargs = mock_llm.create_completion.call_args
        self.assertEqual(kwargs['max_tokens'], 100)

    @patch('backend.llm_integration.get_llm_client')
    def test_generate_ai_answer_raw_cloud(self, mock_get_client):
        """Test generate_ai_answer with raw=True for Cloud provider."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Cloud response"

        mock_bind = MagicMock()
        mock_bind.invoke.return_value = mock_response
        mock_client.bind.return_value = mock_bind
        mock_client.invoke.return_value = mock_response

        mock_get_client.return_value = mock_client

        # IMPORTANT: Patch the message classes at the import location (lazy import)
        with patch('langchain_core.messages.SystemMessage', SystemMessage), \
             patch('langchain_core.messages.HumanMessage', HumanMessage):

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

        # Verify messages passed to invoke (from bind or client)
        if mock_bind.invoke.called:
            args, _ = mock_bind.invoke.call_args
        else:
            args, _ = mock_client.invoke.call_args

        messages = args[0]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "System Prompt")
        self.assertEqual(messages[1].content, "User Prompt")

    @patch('backend.llm_integration.generate_ai_answer')
    def test_agent_calls_generate_ai_answer_correctly(self, mock_generate):
        """Verify ReActAgent calls generate_ai_answer with correct params."""
        mock_generate.return_value = "Thought: I should search.\nAction: search_knowledge_base\nAction Input: query"
        mock_config = MagicMock()
        def get_side_effect(section, key, fallback=None):
            if section == "LocalLLM" and key == "provider": return "local"
            if section == "LocalLLM" and key == "model_path": return "model.gguf"
            if section == "APIKeys": return ""
            return fallback
        mock_config.get.side_effect = get_side_effect

        global_state = {"config": mock_config}

        from backend.agent import ReActAgent
        agent = ReActAgent(global_state)

        async def run_agent():
            gen = agent.stream_chat("Hello")
            try:
                async for _ in gen:
                    break
            except Exception:
                pass

        asyncio.run(run_agent())

        mock_generate.assert_called()
        kwargs = mock_generate.call_args.kwargs
        self.assertTrue(kwargs['raw'])

if __name__ == '__main__':
    unittest.main()
