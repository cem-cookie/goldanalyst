"""
test_trading_agent.py - Pytest tests for TradingAgent
Tests initialization, parameter validation, error handling, and edge cases.
"""
import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from agents.trading_agent import TradingAgent


@pytest.fixture
def sample_news_data():
    """Create a temporary news JSON file for testing."""
    return [
        {
            "source": "YahooRSS",
            "title": "Gold prices surge on inflation fears",
            "full_text": "Gold prices surged today as inflation concerns grow.",
            "selected": True,
            "publish_date": "2026-04-25"
        },
        {
            "source": "InvestingRSS",
            "title": "Fed signals rate hold",
            "full_text": "The Federal Reserve signaled it will hold rates steady.",
            "selected": True,
            "publish_date": "2026-04-25"
        },
        {
            "source": "YahooRSS",
            "title": "No full text article",
            "selected": True,
            "publish_date": "2026-04-25"
            # No full_text - should be filtered out
        }
    ]


class TestTradingAgentInit:
    """Tests for TradingAgent __init__."""

    def test_init_with_defaults(self):
        """Default constructor should set reasonable defaults."""
        agent = TradingAgent(api_key="test-key")
        assert agent.name == "Agent"
        assert agent.max_alloc == 0.5
        assert agent.fee_bps == 10
        assert agent.json_path == "data/gold_news.json"

    def test_init_with_custom_params(self):
        """Custom parameters should override defaults."""
        agent = TradingAgent(
            name="GoldMaster",
            api_key="custom-key",
            max_alloc=0.3,
            fee_bps=5,
            json_path="data/custom_news.json",
            persist_outputs=False
        )
        assert agent.name == "GoldMaster"
        assert agent.max_alloc == 0.3
        assert agent.fee_bps == 5
        assert agent.json_path == "data/custom_news.json"
        assert agent.persist_outputs is False

    def test_init_with_context(self):
        """Context dict should be accepted by constructor."""
        context = {"strategy": "Scalping", "investment_level": "Aggressive"}
        agent = TradingAgent(api_key="test-key", context=context)
        # Context is consumed but not stored directly as attribute
        assert agent.name == "Agent"

    def test_init_without_api_key(self):
        """Should work without API key (client will use env var or be None)."""
        agent = TradingAgent()
        assert agent.api_key is None
        # Client should be created but may fail on API calls
        assert agent.client is not None


class TestLoadNews:
    """Tests for load_news method."""

    def test_load_news_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        agent = TradingAgent(api_key="test-key", json_path="data/nonexistent_file.json")
        with pytest.raises(FileNotFoundError):
            agent.load_news()

    def test_load_news_with_valid_data(self, sample_news_data):
        """Should load and filter news with full_text and selected=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test_news.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sample_news_data, f, ensure_ascii=False)
            agent = TradingAgent(api_key="test-key", json_path=json_path)
            result = agent.load_news()
            assert len(result) == 2  # Only 2 items have both selected=True and full_text
            assert all("full_text" in item for item in result)
            assert all(item["selected"] for item in result)

    def test_load_news_with_dict_format(self):
        """Should handle news wrapped in a dict with 'news' key."""
        data = {"news": [{"title": "Test", "full_text": "Content", "selected": True}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test_news2.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            agent = TradingAgent(api_key="test-key", json_path=json_path)
            result = agent.load_news()
            assert len(result) == 1
            assert result[0]["title"] == "Test"

    def test_load_news_handles_unexpected_format(self):
        """Should return empty list for unexpected format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "bad_format.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump("not a list or dict with news key", f)
            agent = TradingAgent(api_key="test-key", json_path=json_path)
            result = agent.load_news()
            assert result == []


class TestGenerateTradeDecision:
    """Tests for generate_trade_decision method."""

    def test_generate_trade_decision_error_handling(self):
        """Should return a safe default decision on error (no API call)."""
        agent = TradingAgent(api_key="invalid-key")
        decision = agent.generate_trade_decision("Buy gold now!")
        assert decision["action"] == "HOLD"
        assert "confidence" in decision
        assert "reasoning" in decision

    @patch("agents.trading_agent.OpenAI")
    def test_generate_trade_decision_with_mock_response(self, mock_openai):
        """Should parse valid JSON response from LLM."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"action": "BUY", "reasoning": "Strong momentum", "confidence": 8, "risk_level": "low"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        agent = TradingAgent(api_key="test-key")
        agent.client = mock_client
        decision = agent.generate_trade_decision("Market is bullish")
        assert decision["action"] == "BUY"
        assert decision["confidence"] == 8
        assert decision["reasoning"] == "Strong momentum"


class TestRun:
    """Tests for the run() method."""

    def test_run_with_no_news_file(self):
        """Should return default decision when news file not found."""
        agent = TradingAgent(api_key="test-key", json_path="data/nonexistent.json")
        result = agent.run()
        assert result["recommendation"]["action"] == "HOLD"
        assert "asset" in result
        assert result["asset"] == "XAU/USD"

    def test_run_with_empty_news(self):
        """Should return default decision when no selected news found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "empty_news.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            agent = TradingAgent(api_key="test-key", json_path=json_path)
            result = agent.run()
            assert result["recommendation"]["action"] == "HOLD"
            assert "No selected news" in result["market_summary"]["comment"]
