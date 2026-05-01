"""Tests for error handler utilities."""
import pytest
from utils.error_handler import (
    ErrorHandler,
    handle_error,
    get_traceback,
    set_user_message,
    DEFAULT_USER_MESSAGE,
)


def test_default_user_message():
    """Default message should be generic."""
    assert DEFAULT_USER_MESSAGE == "An unexpected error occurred. Please try again."


def test_error_handler_returns_user_message():
    """Handler should return user-safe message."""
    handler = ErrorHandler()
    result = handler.handle(ValueError("test error"), log_full=False)
    assert result == "An unexpected error occurred. Please try again."


def test_error_handler_captures_traceback():
    """Handler should capture traceback."""
    handler = ErrorHandler()
    try:
        raise ValueError("test error")
    except ValueError as e:
        handler.handle(e, log_full=False)
    
    tb = handler.get_last_traceback()
    assert tb is not None
    assert "ValueError" in tb
    assert "test error" in tb


def test_handle_error_function():
    """handle_error convenience function should work."""
    try:
        raise RuntimeError("test runtime")
    except RuntimeError as e:
        msg = handle_error(e, log_full=False)
    
    assert msg == "An unexpected error occurred. Please try again."


def test_custom_user_message():
    """Should support custom user messages."""
    custom_msg = "Something went wrong. Contact support."
    handler = ErrorHandler(user_message=custom_msg)
    result = handler.handle(IOError("disk full"), log_full=False)
    assert result == custom_msg


def test_set_user_message():
    """set_user_message should update global handler."""
    new_msg = "Custom error occurred!"
    set_user_message(new_msg)
    
    try:
        raise ValueError("test")
    except ValueError as e:
        msg = handle_error(e, log_full=False)
    
    assert msg == new_msg