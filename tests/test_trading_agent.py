import unittest
from agents.trading_agent import TradingAgent

class TestTradingAgent(unittest.TestCase):
    def setUp(self):
        self.agent = TradingAgent()

    def test_strategy_generation(self):
        result = self.agent.generate_strategy("sample.json")
        self.assertIsNotNone(result)
        self.assertIn("strategy", result)

if __name__ == "__main__":
    unittest.main()