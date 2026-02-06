import sys
from unittest.mock import MagicMock, patch

# Mock dependencies
sys.modules['faiss'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['docx'] = MagicMock()
sys.modules['PyPDF2'] = MagicMock()

# Mock backend modules
sys.modules['backend.search'] = MagicMock()
sys.modules['backend.database'] = MagicMock()
sys.modules['backend.file_processing'] = MagicMock()

# Mock langchain_core
mock_lc_core = MagicMock()
sys.modules['langchain_core'] = mock_lc_core
sys.modules['langchain_core.messages'] = mock_lc_core

class MockMessage:
    def __init__(self, content):
        self.content = content

mock_lc_core.HumanMessage = MockMessage
mock_lc_core.SystemMessage = MockMessage

import unittest
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.llm_integration import generate_ai_answer
from backend.agent import ReActAgent

class TestAgentRefactor(unittest.TestCase):

    @patch('backend.llm_integration.get_llm_client')
    @patch('backend.llm_integration.get_local_llm')
    def test_generate_ai_answer_raw_local(self, mock_get_local, mock_get_client):
        mock_get_client.return_value = "LOCAL:test.gguf"

        mock_llm = MagicMock()
        mock_llm.create_completion.return_value = {'choices': [{'text': 'Raw response'}]}
        mock_get_local.return_value = mock_llm

        prompt = "System: ...\nUser: ..."
        response = generate_ai_answer(
            context=prompt,
            question=None,
            provider="local",
            model_path="test.gguf",
            raw=True,
            max_tokens=100,
            stop=["Stop:"],
            temperature=0.5,
            repeat_penalty=1.2
        )

        self.assertEqual(response, "Raw response")

        mock_llm.create_completion.assert_called_once()
        args, kwargs = mock_llm.create_completion.call_args
        self.assertEqual(args[0], prompt)
        self.assertEqual(kwargs['max_tokens'], 100)
        self.assertEqual(kwargs['stop'], ["Stop:"])
        self.assertEqual(kwargs['temperature'], 0.5)
        self.assertEqual(kwargs['repeat_penalty'], 1.2)
        self.assertEqual(kwargs['echo'], False)

    @patch('backend.llm_integration.get_llm_client')
    def test_generate_ai_answer_raw_cloud(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.invoke.return_value.content = "Cloud response"
        mock_get_client.return_value = mock_client

        prompt = "System: ...\nUser: ..."

        response = generate_ai_answer(
            context=prompt,
            question=None,
            provider="openai",
            api_key="sk-test",
            raw=True,
            stop=["Stop:"]
        )

        self.assertEqual(response, "Cloud response")

        mock_client.invoke.assert_called_once()
        args, kwargs = mock_client.invoke.call_args

        messages = args[0]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, prompt)
        self.assertIsInstance(messages[0], MockMessage)

        self.assertEqual(kwargs['stop'], ["Stop:"])

    @patch('backend.llm_integration.generate_ai_answer')
    def test_agent_calls_generate_ai_answer(self, mock_generate):
        mock_generate.return_value = "Thought: I need to search.\nAction: search_knowledge_base"

        mock_config = MagicMock()
        def get_side_effect(section, key, fallback=''):
            if key == 'provider': return 'local'
            if key == 'model_path': return 'test.gguf'
            if key == 'local_api_key': return ''
            return fallback
        mock_config.get.side_effect = get_side_effect

        mock_state = {'config': mock_config}

        agent = ReActAgent(mock_state)

        async def run_agent():
            gen = agent.stream_chat("Test question")
            try:
                async for item in gen:
                    pass
            except Exception as e:
                pass

        asyncio.run(run_agent())

        mock_generate.assert_called()
        args, kwargs = mock_generate.call_args

        self.assertTrue(kwargs['raw'])
        self.assertEqual(kwargs['provider'], "local")
        self.assertEqual(kwargs['stop'], ["Observation:", "Definition:", "Thought:"])
        self.assertEqual(kwargs['max_tokens'], 256)
        self.assertEqual(kwargs['temperature'], 0.1)
        self.assertEqual(kwargs['repeat_penalty'], 1.1)

if __name__ == '__main__':
    unittest.main()
