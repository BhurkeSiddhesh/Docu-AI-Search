import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

from backend.portfolio import service, importers, tools
from backend.portfolio.schema import connection
from backend.agent import ReActAgent

class TestPortfolioIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(delete=False)
        self.tmp.close()
        os.environ["DOCU_PORTFOLIO_DB"] = self.tmp.name
        with connection() as conn:
            pass

    def tearDown(self):
        os.environ.pop("DOCU_PORTFOLIO_DB", None)
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_importer_csv_tables(self):
        csv_content = "trade_date,symbol,side,quantity,net_amount\n2025-01-01,AAPL,buy,10,1500"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(csv_content.encode("utf-8"))
            f.flush()
            tables = importers.parse_file(f.name)
            self.assertTrue(len(tables) >= 1)
        os.unlink(f.name)

    def test_agent_routes_portfolio_queries(self):
        class MockConfig:
            def get(self, section, key, fallback=None):
                return fallback
        agent = ReActAgent({"config": MockConfig()})
        async def run():
            responses = []
            async for chunk in agent.stream_chat("How much profit did Siddhesh earn?"):
                responses.append(chunk)
            return responses

        import asyncio
        responses = asyncio.run(run())
        self.assertTrue(any(r.get("type") == "answer" for r in responses))
        ans = next(r["content"] for r in responses if r.get("type") == "answer")
        self.assertIn("No portfolio transactions", ans)

    def test_tools_is_portfolio_question(self):
        self.assertTrue(tools.is_portfolio_question("How much profit did I make on AAPL?"))
        self.assertTrue(tools.is_portfolio_question("Which stock was most profitable?"))
        self.assertFalse(tools.is_portfolio_question("explain the budget report"))
