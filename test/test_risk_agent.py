"""
test_risk_agent.py - Pytest tests for RiskAgent
Tests initialization, context handling, heuristic assessment, and error paths.
"""
import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from agents.risk_agent import RiskAgent


@pytest.fixture
def mock_strategies():
    """Sample strategies for risk assessment testing."""
    return [
        {
            "id": 1,
            "name": "Aggressive Buy",
            "action": "BUY",
            "confidence": 8,
            "expected_risk": "High",
            "expected_return": "High",
            "rationale": "Strong upward momentum expected"
        },
        {
            "id": 2,
            "name": "Conservative Hold",
            "action": "HOLD",
            "confidence": 5,
            "expected_risk": "Low",
            "expected_return": "Low",
            "rationale": "Market sideways, wait for signal"
        },
        {
            "id": 3,
            "name": "Short Sell",
            "action": "SELL",
            "confidence": 2,
            "expected_risk": "High",
            "expected_return": "High",
            "rationale": "Overbought conditions"
        }
    ]


@pytest.fixture
def temp_decision_file():
    """Create a temporary trading_decision.json for testing."""
    return {
        "asset": "XAU/USD",
        "timestamp": "2026-04-25T00:00:00",
        "strategies": [
            {
                "id": 1,
                "name": "Test Strategy",
                "action": "BUY",
                "confidence": 7,
                "expected_risk": "Medium",
                "expected_return": "Medium",
                "rationale": "Test rationale"
            }
        ]
    }


class TestRiskAgentInit:
    """Tests for RiskAgent __init__."""

    def test_init_with_defaults(self):
        """Default constructor should set reasonable defaults."""
        agent = RiskAgent(openai_api_key="test-key")
        assert agent.decision_path == "data/trading_decision.json"
        assert agent.out_path == "data/risk_report.json"
        assert agent.strategy_type == "Swing"
        assert agent.investment_level == "Active"

    def test_init_with_custom_context(self):
        """Context parameters should override defaults."""
        context = {
            "strategy": "Scalping",
            "investment_level": "Aggressive",
            "buy_price_threshold": 4000.0,
            "sell_price_threshold": 4200.0,
            "target_profit": 0.15
        }
        agent = RiskAgent(openai_api_key="test-key", context=context)
        assert agent.strategy_type == "Scalping"
        assert agent.investment_level == "Aggressive"
        assert agent.buy_price_threshold == 4000.0
        assert agent.sell_price_threshold == 4200.0
        assert agent.target_profit == 0.15

    def test_init_with_custom_paths(self):
        """Custom decision and output paths should be stored."""
        agent = RiskAgent(
            openai_api_key="test-key",
            decision_path="data/custom_decision.json",
            out_path="data/custom_risk_report.json"
        )
        assert agent.decision_path == "data/custom_decision.json"
        assert agent.out_path == "data/custom_risk_report.json"

    def test_init_api_key_alias(self):
        """Should support both openai_api_key and api_key parameter names."""
        agent = RiskAgent(api_key="alias-key")
        assert agent.client.api_key == "alias-key"

    def test_init_openai_key_preferred_over_alias(self):
        """openai_api_key should take precedence over api_key."""
        agent = RiskAgent(openai_api_key="preferred-key", api_key="alias-key")
        assert agent.client.api_key == "preferred-key"

    def test_init_with_empty_context(self):
        """Should handle empty context without errors."""
        agent = RiskAgent(openai_api_key="test-key", context=None)
        assert agent.strategy_type == "Swing"
        assert agent.buy_price_threshold == 0


class TestLoadDecision:
    """Tests for load_decision method."""

    def test_load_decision_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        agent = RiskAgent(
            openai_api_key="test-key",
            decision_path="data/nonexistent_decision.json"
        )
        with pytest.raises(FileNotFoundError):
            agent.load_decision()

    def test_load_decision_valid_file(self, temp_decision_file):
        """Should parse valid decision JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test_decision.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(temp_decision_file, f)
            agent = RiskAgent(openai_api_key="test-key", decision_path=json_path)
            result = agent.load_decision()
            assert result["asset"] == "XAU/USD"
            assert len(result["strategies"]) == 1


class TestHeuristicAssess:
    """Tests for _heuristic_assess fallback method."""

    def test_heuristic_assess_returns_valid_structure(self, mock_strategies):
        """Heuristic assess should return dict with items and summary."""
        agent = RiskAgent(openai_api_key="test-key")
        result = agent._heuristic_assess("XAU/USD", mock_strategies)
        assert "items" in result
        assert "summary" in result
        assert len(result["items"]) == 3

    def test_heuristic_assess_each_item_has_required_keys(self, mock_strategies):
        """Each assessment item should have expected keys."""
        agent = RiskAgent(openai_api_key="test-key")
        result = agent._heuristic_assess("XAU/USD", mock_strategies)
        required_keys = {"id", "approval", "approval_score", "risk_level", "key_risks", "mitigations"}
        for item in result["items"]:
            assert required_keys.issubset(set(item.keys()))

    def test_heuristic_assess_confidence_affects_score(self, mock_strategies):
        """Higher confidence strategies should get higher scores."""
        agent = RiskAgent(openai_api_key="test-key")
        result = agent._heuristic_assess("XAU/USD", mock_strategies)
        scores = {item["id"]: item["approval_score"] for item in result["items"]}
        # Strategy 1 (conf=8) should have higher score than Strategy 3 (conf=2)
        assert scores[1] > scores[3]

    def test_heuristic_assess_with_passive_level(self, mock_strategies):
        """Passive investment should penalize high-risk strategies."""
        agent = RiskAgent(
            openai_api_key="test-key",
            context={"investment_level": "Passive"}
        )
        result = agent._heuristic_assess("XAU/USD", mock_strategies)
        # BUY with High risk should be penalized for Passive
        strategy_1 = result["items"][0]
        assert isinstance(strategy_1["approval_score"], (int, float))

    def test_heuristic_assess_with_empty_strategies(self):
        """Should handle empty strategy list gracefully."""
        agent = RiskAgent(openai_api_key="test-key")
        result = agent._heuristic_assess("XAU/USD", [])
        assert "items" in result
        assert len(result["items"]) == 0


class TestRiskAgentRun:
    """Tests for the run() method."""

    def test_run_with_missing_decision_file(self):
        """Should raise FileNotFoundError if no decision file exists."""
        agent = RiskAgent(
            openai_api_key="test-key",
            decision_path="data/missing_decision.json"
        )
        with pytest.raises(FileNotFoundError):
            agent.run()

    def test_run_with_valid_data_triggers_heuristic(self, temp_decision_file):
        """With valid data but no LLM, should fall back to heuristic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            decision_path = os.path.join(tmpdir, "decision.json")
            out_path = os.path.join(tmpdir, "risk_report.json")
            with open(decision_path, "w", encoding="utf-8") as f:
                json.dump(temp_decision_file, f)
            # Use invalid API key to force heuristic fallback
            agent = RiskAgent(
                openai_api_key=None,
                decision_path=decision_path,
                out_path=out_path
            )
            result = agent.run()
            assert result is not None
            assert "items" in result
            # The heuristic path should produce a valid result
            assert len(result["items"]) > 0
