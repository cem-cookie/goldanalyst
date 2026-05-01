"""Tests for validation utilities."""
import pytest
from utils.validation import (
    ensure_int,
    ensure_float,
    ensure_str,
    ensure_in_range,
    ensure_positive,
    ensure_list,
)


class TestEnsureInt:
    """Tests for ensure_int."""

    def test_valid_int(self):
        assert ensure_int(5, "val") == 5

    def test_cast_from_float(self):
        assert ensure_int(5.7, "val") == 5

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="must be an integer"):
            ensure_int("abc", "val")

    def test_below_min(self):
        with pytest.raises(ValueError, match="must be >= 10"):
            ensure_int(5, "val", min_val=10)

    def test_above_max(self):
        with pytest.raises(ValueError, match="must be <= 10"):
            ensure_int(15, "val", max_val=10)


class TestEnsureFloat:
    """Tests for ensure_float."""

    def test_valid_float(self):
        assert ensure_float(5.5, "val") == 5.5

    def test_cast_from_int(self):
        assert ensure_float(5, "val") == 5.0

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="must be a number"):
            ensure_float([1, 2], "val")

    def test_in_range(self):
        assert ensure_float(0.5, "val", min_val=0.0, max_val=1.0) == 0.5


class TestEnsureStr:
    """Tests for ensure_str."""

    def test_valid_str(self):
        assert ensure_str("hello", "val") == "hello"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="must be a string"):
            ensure_str(123, "val")

    def test_below_min_len(self):
        with pytest.raises(ValueError, match="at least 5 characters"):
            ensure_str("abc", "val", min_len=5)


class TestEnsureInRange:
    """Tests for ensure_in_range."""

    def test_valid_range(self):
        assert ensure_in_range(0.5, "val") == 0.5

    def test_at_boundaries(self):
        assert ensure_in_range(0.0, "val") == 0.0
        assert ensure_in_range(1.0, "val") == 1.0

    def test_below_min(self):
        with pytest.raises(ValueError, match="must be >= 0.0"):
            ensure_in_range(-0.1, "val")

    def test_above_max(self):
        with pytest.raises(ValueError, match="must be <= 1.0"):
            ensure_in_range(1.1, "val")


class TestEnsurePositive:
    """Tests for ensure_positive."""

    def test_valid_positive(self):
        assert ensure_positive(5.0, "val") == 5.0

    def test_zero_fails(self):
        with pytest.raises(ValueError, match="must be positive"):
            ensure_positive(0, "val")

    def test_negative_fails(self):
        with pytest.raises(ValueError, match="must be positive"):
            ensure_positive(-1, "val")


class TestEnsureList:
    """Tests for ensure_list."""

    def test_valid_list(self):
        assert ensure_list([1, 2, 3], "val") == [1, 2, 3]

    def test_not_a_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            ensure_list("abc", "val")

    def test_min_length(self):
        with pytest.raises(ValueError, match="at least 2 items"):
            ensure_list([1], "val", min_len=2)