"""
test_history_agent.py - Pytest tests for HistoryAgent
Tests initialization, CSV loading, summary generation, and formatting helpers.
"""
import os
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from pathlib import Path
from agents.history_agent import HistoryAgent


@pytest.fixture
def sample_trades_df():
    """Create a sample trades DataFrame for testing."""
    return pd.DataFrame({
        "side": ["BUY", "SELL", "BUY", "SELL", "SELL"],
        "qty": [10, 10, 5, 5, 10],
        "price": [3900.0, 3950.0, 3910.0, 3960.0, 3920.0],
        "notional": [39000, 39500, 19550, 19800, 39200],
        "fee": [10, 12, 8, 9, 15],
        "realized_pnl": [0, 500.0, 0, 250.0, -100.0],
        "position_qty": [10, 0, 5, 0, -10],
        "avg_cost": [3900, 0, 3910, 0, 3920],
        "cash": [61000, 61488, 41930, 42167, 81352],
        "equity": [100000, 101000, 100700, 101200, 100800],
        "unrealized_pnl": [0, 0, 50, 0, -200],
        "total_pnl": [0, 500, 50, 750, 550]
    })


@pytest.fixture
def empty_trades_df():
    """Return an empty DataFrame with all required columns."""
    return pd.DataFrame(columns=[
        "side", "qty", "price", "notional", "fee", "realized_pnl",
        "position_qty", "avg_cost", "cash", "equity", "unrealized_pnl", "total_pnl"
    ])


class TestHistoryAgentInit:
    """Tests for HistoryAgent __init__."""

    def test_init_with_defaults(self):
        """Default paths should point to standard data files."""
        agent = HistoryAgent()
        assert os.path.normpath(str(agent.trades_csv)) == os.path.normpath("data/trades_history.csv")
        assert os.path.normpath(str(agent.ledger_json)) == os.path.normpath("data/ledger.json")

    def test_init_with_custom_paths(self):
        """Custom paths should override defaults."""
        agent = HistoryAgent(
            trades_csv="custom/trades.csv",
            ledger_json="custom/ledger.json"
        )
        assert os.path.normpath(str(agent.trades_csv)) == os.path.normpath("custom/trades.csv")
        assert os.path.normpath(str(agent.ledger_json)) == os.path.normpath("custom/ledger.json")


class TestSafeFloat:
    """Tests for _safe_float helper."""

    def test_valid_integer(self):
        """Integer should be converted to float."""
        agent = HistoryAgent()
        result = agent._safe_float(5)
        assert result == 5.0
        assert isinstance(result, float)

    def test_valid_float(self):
        """Float should pass through unchanged."""
        agent = HistoryAgent()
        result = agent._safe_float(100.5)
        assert result == 100.5

    def test_none_value(self):
        """None should return None."""
        agent = HistoryAgent()
        result = agent._safe_float(None)
        assert result is None

    def test_nan_value(self):
        """NaN (np.nan) should return None since pd.notna(np.nan) is False."""
        agent = HistoryAgent()
        result = agent._safe_float(np.nan)
        assert result is None

    def test_invalid_string(self):
        """Invalid string should return None."""
        agent = HistoryAgent()
        result = agent._safe_float("not_a_number")
        assert result is None

    def test_numeric_string(self):
        """Numeric string should be converted to float."""
        agent = HistoryAgent()
        result = agent._safe_float("42.5")
        assert result == 42.5
        assert isinstance(result, float)


class TestFmtCurrency:
    """Tests for _fmt_currency formatter."""

    def test_valid_number(self):
        """Valid number should be formatted as currency."""
        agent = HistoryAgent()
        result = agent._fmt_currency(1500.75)
        assert result == "$1500.75"

    def test_none_value(self):
        """None should return em dash."""
        agent = HistoryAgent()
        result = agent._fmt_currency(None)
        assert result == "—"

    def test_invalid_string(self):
        """Invalid string should return em dash."""
        agent = HistoryAgent()
        result = agent._fmt_currency("invalid")
        assert result == "—"

    def test_zero_value(self):
        """Zero should be formatted correctly."""
        agent = HistoryAgent()
        result = agent._fmt_currency(0.0)
        assert result == "$0.00"


class TestFmtNumber:
    """Tests for _fmt_number formatter."""

    def test_valid_number(self):
        """Valid number should be formatted to 2 decimal places."""
        agent = HistoryAgent()
        result = agent._fmt_number(42.5)
        assert result == "42.50"

    def test_none_value(self):
        """None should return em dash."""
        agent = HistoryAgent()
        result = agent._fmt_number(None)
        assert result == "—"


class TestLoad:
    """Tests for load method."""

    def test_load_file_not_found(self):
        """Should raise FileNotFoundError for missing CSV."""
        agent = HistoryAgent(trades_csv="data/nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            agent.load()

    def test_load_valid_csv(self, sample_trades_df):
        """Should load and parse valid CSV data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "trades.csv")
            sample_trades_df.to_csv(csv_path, index=False)
            agent = HistoryAgent(trades_csv=csv_path)
            result = agent.load()
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 5
            # Numeric columns should be properly typed
            assert pd.api.types.is_numeric_dtype(result["equity"])

    def test_load_with_encoding(self):
        """Should handle UTF-8 encoded CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "trades_utf8.csv")
            df = pd.DataFrame({"side": ["BUY"], "equity": [100000]})
            df.to_csv(csv_path, index=False, encoding="utf-8")
            agent = HistoryAgent(trades_csv=csv_path)
            result = agent.load()
            assert len(result) == 1


class TestSummary:
    """Tests for summary method."""

    def test_summary_from_ledger(self):
        """Should return ledger data when ledger.json exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = os.path.join(tmpdir, "ledger.json")
            trades_path = os.path.join(tmpdir, "trades.csv")
            ledger_data = {
                "num_trades": 42,
                "win_rate": 65.0,
                "net_realized": 5000.0,
                "final_equity": 105000.0
            }
            with open(ledger_path, "w", encoding="utf-8") as f:
                json.dump(ledger_data, f)
            agent = HistoryAgent(trades_csv=trades_path, ledger_json=ledger_path)
            result = agent.summary()
            assert result["num_trades"] == 42
            assert result["win_rate"] == 65.0

    def test_summary_from_csv(self, sample_trades_df):
        """Should calculate summary from CSV when no ledger exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "trades.csv")
            ledger_path = os.path.join(tmpdir, "ledger_missing.json")
            sample_trades_df.to_csv(csv_path, index=False)
            agent = HistoryAgent(trades_csv=csv_path, ledger_json=ledger_path)
            result = agent.summary()
            assert "num_trades" in result
            assert "win_rate" in result
            assert "net_realized" in result
            # Total trades should count BUY and SELL
            assert result["num_trades"] == 5

    def test_summary_empty_dataframe(self, empty_trades_df):
        """Should raise ValueError when DataFrame is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "empty.csv")
            ledger_path = os.path.join(tmpdir, "empty_ledger.json")
            empty_trades_df.to_csv(csv_path, index=False)
            agent = HistoryAgent(trades_csv=csv_path, ledger_json=ledger_path)
            with pytest.raises(ValueError, match="No trades to analyze"):
                agent.summary()

    def test_summary_returns_numeric_values(self, sample_trades_df):
        """Summary values should be numeric, not None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "trades.csv")
            ledger_path = os.path.join(tmpdir, "ledger_no.json")
            sample_trades_df.to_csv(csv_path, index=False)
            agent = HistoryAgent(trades_csv=csv_path, ledger_json=ledger_path)
            result = agent.summary()
            assert isinstance(result["win_rate"], (int, float))
            assert isinstance(result["net_realized"], (int, float))
            assert result["win_rate"] is not None

    def test_summary_win_rate_only_sell_trades(self, sample_trades_df):
        """Win rate should be based on SELL trades with positive realized_pnl."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "trades.csv")
            ledger_path = os.path.join(tmpdir, "ledger_no.json")
            sample_trades_df.to_csv(csv_path, index=False)
            agent = HistoryAgent(trades_csv=csv_path, ledger_json=ledger_path)
            result = agent.summary()
            # 3 SELL trades: realized_pnl values = [500, 250, -100]
            # 2 positive out of 3 = ~66.67%
            assert round(result["win_rate"], 1) == round(2 / 3 * 100, 1)


class TestReportText:
    """Tests for report_text method."""

    def test_report_text_returns_string(self, sample_trades_df):
        """report_text should return a string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "trades.csv")
            ledger_path = os.path.join(tmpdir, "ledger_no.json")
            sample_trades_df.to_csv(csv_path, index=False)
            agent = HistoryAgent(trades_csv=csv_path, ledger_json=ledger_path)
            result = agent.report_text()
            assert isinstance(result, str)
            assert len(result) > 0
            assert "Total trades" in result or "Total trades" in result.lower()

    def test_report_markdown_returns_string(self, sample_trades_df):
        """report_markdown should return a string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "trades.csv")
            ledger_path = os.path.join(tmpdir, "ledger_no.json")
            sample_trades_df.to_csv(csv_path, index=False)
            agent = HistoryAgent(trades_csv=csv_path, ledger_json=ledger_path)
            result = agent.report_markdown()
            assert isinstance(result, str)
            assert "**Total trades:**" in result
            assert "**Win rate:**" in result
