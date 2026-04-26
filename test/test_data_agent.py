"""
test_data_agent.py - Pytest tests for DataAgent
Tests initialization, config loading, quality scoring, and helper methods.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from agents.data_agent import DataAgent


@pytest.fixture
def mock_config():
    """Realistic YAML config for testing."""
    return {
        "sources": {
            "YahooRSS": {
                "base_score": 75,
                "weight_factors": [
                    {"has_full_text": 10},
                    {"article_length_min": 500},
                    {"article_length_bonus": 5}
                ]
            },
            "InvestingRSS": {
                "base_score": 80,
                "weight_factors": {
                    "has_full_text": 10,
                    "article_length_min": 500,
                    "article_length_bonus": 5
                }
            },
            "Unknown": {"base_score": 40}
        },
        "thresholds": {
            "minimum_score": 50,
            "high_quality": 75
        },
        "llm_selection": {
            "selected_bonus": 15,
            "not_selected_penalty": -10
        },
        "content_quality": {
            "keywords_positive": ["gold", "inflation", "central bank"],
            "keywords_negative": ["shocking", "scam"],
            "positive_keyword_bonus": 3,
            "negative_keyword_penalty": -5
        }
    }


@pytest.fixture
def sample_news_item():
    """Minimal news item for quality scoring tests."""
    return {
        "source": "YahooRSS",
        "title": "Gold prices surge on inflation fears",
        "summary": "Gold reached new highs as inflation concerns grow",
        "full_text": "Gold prices surged today after the Federal Reserve signaled continued inflation concerns. "
                      "Investors are flocking to safe-haven assets. " * 10,
        "selected": True,
        "full_text_len": 1200
    }


class TestDataAgentInit:
    """Tests for DataAgent __init__."""

    @patch("agents.data_agent.yaml.safe_load")
    @patch("agents.data_agent.os.path.exists", return_value=True)
    def test_init_with_custom_config_path(self, mock_exists, mock_yaml, mock_config):
        """__init__ should load config from provided path."""
        mock_yaml.return_value = mock_config
        with patch("builtins.open", create=True):
            agent = DataAgent(openai_api_key=None, config_path="/custom/path.yaml")
        assert agent.config is not None
        assert agent.config["sources"]["YahooRSS"]["base_score"] == 75

    @patch("agents.data_agent.yaml.safe_load")
    @patch("agents.data_agent.os.path.exists", return_value=True)
    def test_init_default_config_path_resolves(self, mock_exists, mock_yaml, mock_config):
        """Default config_path should resolve to config/news_quality_scorer.yaml."""
        mock_yaml.return_value = mock_config
        with patch("builtins.open", create=True):
            agent = DataAgent(openai_api_key=None)
        assert agent.config is not None

    @patch("agents.data_agent.yaml.safe_load")
    def test_init_uses_default_config_on_load_failure(self, mock_yaml):
        """If YAML fails to load, should fall back to _default_config()."""
        mock_yaml.side_effect = FileNotFoundError("Config not found")
        agent = DataAgent(openai_api_key=None)
        assert "sources" in agent.config
        assert agent.config["llm_selection"]["selected_bonus"] == 15

    def test_init_with_explicit_api_key(self):
        """API key passed to constructor should be used."""
        agent = DataAgent(openai_api_key="test-key-123")
        assert agent.client.api_key == "test-key-123"


class TestNormalizeWeightFactors:
    """Tests for _normalize_weight_factors."""

    @patch("agents.data_agent.yaml.safe_load")
    def test_handles_list_format(self, mock_yaml, mock_config):
        """Should convert list format weight_factors to dict."""
        mock_yaml.return_value = mock_config
        with patch("builtins.open", create=True):
            agent = DataAgent(openai_api_key=None)
        raw_list = [{"has_full_text": 10}, {"article_length_min": 500}]
        result = agent._normalize_weight_factors(raw_list)
        assert isinstance(result, dict)
        assert result["has_full_text"] == 10
        assert result["article_length_min"] == 500

    @patch("agents.data_agent.yaml.safe_load")
    def test_handles_dict_format(self, mock_yaml, mock_config):
        """Should pass through dict format unchanged."""
        mock_yaml.return_value = mock_config
        with patch("builtins.open", create=True):
            agent = DataAgent(openai_api_key=None)
        raw_dict = {"has_full_text": 10, "article_length_min": 500}
        result = agent._normalize_weight_factors(raw_dict)
        assert result == raw_dict

    @patch("agents.data_agent.yaml.safe_load")
    def test_handles_none(self, mock_yaml, mock_config):
        """Should return empty dict for None input."""
        mock_yaml.return_value = mock_config
        with patch("builtins.open", create=True):
            agent = DataAgent(openai_api_key=None)
        result = agent._normalize_weight_factors(None)
        assert result == {}

    @patch("agents.data_agent.yaml.safe_load")
    def test_handles_invalid_type(self, mock_yaml, mock_config):
        """Should return empty dict for non-list, non-dict input."""
        mock_yaml.return_value = mock_config
        with patch("builtins.open", create=True):
            agent = DataAgent(openai_api_key=None)
        result = agent._normalize_weight_factors("invalid")
        assert result == {}


class TestDefaultConfig:
    """Tests for _default_config."""

    def test_default_config_structure(self):
        """_default_config should return expected keys."""
        with patch("agents.data_agent.yaml.safe_load", side_effect=Exception("fail")):
            agent = DataAgent(openai_api_key=None)
        config = agent._default_config()
        assert "sources" in config
        assert "thresholds" in config
        assert "llm_selection" in config


class TestCalculateQualityScore:
    """Tests for _calculate_quality_score."""

    @patch("agents.data_agent.yaml.safe_load")
    def test_score_includes_base_score(self, mock_yaml, mock_config, sample_news_item):
        """Quality score should start with base_score from source config."""
        mock_yaml.return_value = mock_config
        with patch("builtins.open", create=True):
            agent = DataAgent(openai_api_key=None)
        result = agent._calculate_quality_score(sample_news_item)
        assert result["score"] >= 75  # YahooRSS base_score is 75

    @patch("agents.data_agent.yaml.safe_load")
    def test_score_uses_unknown_for_missing_source(self, mock_yaml, mock_config):
        """Should use Unknown source config for unrecognized sources."""
        mock_yaml.return_value = mock_config
        with patch("builtins.open", create=True):
            agent = DataAgent(openai_api_key=None)
        item = {"source": "NonExistentSource", "title": "Test", "selected": True}
        result = agent._calculate_quality_score(item)
        assert result["score"] >= 30  # Unknown base_score 40 + not_selected_penalty -10 = 30

    @patch("agents.data_agent.yaml.safe_load")
    def test_selected_item_gets_bonus(self, mock_yaml, mock_config):
        """Selected items should get llm_selection bonus."""
        mock_yaml.return_value = mock_config
        with patch("builtins.open", create=True):
            agent = DataAgent(openai_api_key=None)
        item = {"source": "YahooRSS", "title": "Test", "selected": True}
        score_selected = agent._calculate_quality_score(item)["score"]
        item["selected"] = False
        score_not_selected = agent._calculate_quality_score(item)["score"]
        assert score_selected > score_not_selected

    @patch("agents.data_agent.yaml.safe_load")
    def test_score_returns_dict_with_score_and_quality(self, mock_yaml, mock_config, sample_news_item):
        """Score result should be a dict with 'score' and 'quality' keys."""
        mock_yaml.return_value = mock_config
        with patch("builtins.open", create=True):
            agent = DataAgent(openai_api_key=None)
        result = agent._calculate_quality_score(sample_news_item)
        assert isinstance(result, dict)
        assert "score" in result
        assert "quality" in result
        assert isinstance(result["score"], (int, float))
        assert result["quality"] in ("high", "medium", "low")
