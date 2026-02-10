import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()
sys.modules['llama_cpp'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.messages'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['langchain_anthropic'] = MagicMock()
sys.modules['langchain_huggingface'] = MagicMock()

from backend.llm_integration import generate_ai_answer

class TestGenerateAIAnswer(unittest.TestCase):
    @patch('backend.llm_integration.get_llm_client')
    def test_generate_ai_answer_success(self, mock_get_client):
        """Test generating AI answer successfully."""
        mock_client = MagicMock()
        # Ensure invoke returns an object with content attribute that is a string
        mock_response = MagicMock()
        mock_response.content = "Generated answer"
        mock_client.invoke.return_value = mock_response
        
        # Also need to mock bind().invoke() just in case raw mode or other logic calls it
        mock_client.bind.return_value.invoke.return_value = mock_response

        mock_get_client.return_value = mock_client
        
        result = generate_ai_answer("context", "question", "openai", "key")
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Generated answer")

if __name__ == '__main__':
    unittest.main()
