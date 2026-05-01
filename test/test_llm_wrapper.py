"""Tests for LLM timeout wrapper utilities."""
import time
import pytest
from utils.llm_wrapper import (
    LLM_CONFIG,
    call_with_timeout,
    call_with_retry,
    call_with_timeout_and_retry,
)


class TestLLMConfig:
    """Tests for LLM config loading."""

    def test_config_has_defaults(self):
        """Config should have required keys."""
        assert "llm_timeout_seconds" in LLM_CONFIG
        assert "llm_max_retries" in LLM_CONFIG
        assert "llm_base_delay_seconds" in LLM_CONFIG

    def test_config_values(self):
        """Config values should be sensible."""
        assert LLM_CONFIG["llm_timeout_seconds"] == 15
        assert LLM_CONFIG["llm_max_retries"] == 3
        assert LLM_CONFIG["llm_base_delay_seconds"] == 1


class TestCallWithTimeout:
    """Tests for timeout functionality."""

    def test_timeout_success(self):
        """Should return result on success."""
        def fast_func():
            return "success"
        
        result = call_with_timeout(fast_func, timeout_seconds=5)
        assert result == "success"

    def test_timeout_applies(self):
        """Should timeout if function takes too long."""
        def slow_func():
            time.sleep(2)
            return "slow"
        
        result = call_with_timeout(slow_func, timeout_seconds=1, default_return="timed_out")
        assert result == "timed_out"


class TestCallWithRetry:
    """Tests for retry functionality."""

    def test_retry_success_first_try(self):
        """Should succeed on first try."""
        call_count = [0]
        
        def succeed_eventually():
            call_count[0] += 1
            return "success"
        
        result = call_with_retry(succeed_eventually, max_retries=3)
        assert result == "success"
        assert call_count[0] == 1

    def test_retry_succeeds_later(self):
        """Should retry and succeed."""
        call_count = [0]
        
        def fail_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("temp error")
            return "success"
        
        result = call_with_retry(fail_twice, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count[0] == 3

    def test_retry_exhausted(self):
        """Should raise when retries exhausted."""
        call_count = [0]
        
        def always_fail():
            call_count[0] += 1
            raise ValueError("always fails")
        
        with pytest.raises(ValueError):
            call_with_retry(always_fail, max_retries=2, base_delay=0.01)
        assert call_count[0] == 3  # initial + 2 retries


class TestCallWithTimeoutAndRetry:
    """Tests for combined timeout + retry."""

    def test_timeout_and_retry_success(self):
        """Should succeed without needing retry."""
        def func():
            return "ok"
        
        result = call_with_timeout_and_retry(
            func, 
            timeout_seconds=5,
            max_retries=2,
            base_delay=0.01,
            default_return="default"
        )
        assert result == "ok"

    def test_timeout_returns_default(self):
        """Should return default on timeout."""
        def slow_func():
            time.sleep(2)
            return "slow"
        
        result = call_with_timeout_and_retry(
            slow_func, 
            timeout_seconds=1,
            max_retries=1,
            base_delay=0.01,
            default_return="default"
        )
        assert result == "default"