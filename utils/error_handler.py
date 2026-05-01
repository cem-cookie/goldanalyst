"""Error handling utilities - centralized error handling for user-facing apps."""
import traceback
import sys
from typing import Optional


# Default user-friendly message
DEFAULT_USER_MESSAGE = "An unexpected error occurred. Please try again."


class ErrorHandler:
    """Centralized error handler for displaying safe messages to users."""
    
    def __init__(self, user_message: str = DEFAULT_USER_MESSAGE):
        self.user_message = user_message
        self._last_traceback: Optional[str] = None
    
    def handle(self, exc: Exception, log_full: bool = True) -> str:
        """
        Handle an exception and return a user-safe message.
        
        Args:
            exc: The exception that was raised
            log_full: Whether to print full traceback to console
            
        Returns:
            User-safe message string
        """
        # Capture full traceback
        tb = traceback.format_exc()
        self._last_traceback = tb
        
        # Log full details to console (for debugging)
        if log_full:
            print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        
        # Return user-friendly message
        return self.user_message
    
    def get_last_traceback(self) -> Optional[str]:
        """Get the last captured traceback."""
        return self._last_traceback
    
    def log_and_return(self, exc: Exception) -> str:
        """Log the error and return user message (convenience method)."""
        return self.handle(exc, log_full=True)


# Global handler instance
_default_handler = ErrorHandler()


def handle_error(exc: Exception, log_full: bool = True) -> str:
    """
    Handle an exception with safe user message.
    
    Args:
        exc: The exception
        log_full: Whether to log full traceback
        
    Returns:
        User-safe message
    """
    return _default_handler.handle(exc, log_full)


def get_traceback() -> Optional[str]:
    """Get last traceback for logging."""
    return _default_handler.get_last_traceback()


def set_user_message(message: str) -> None:
    """Set custom user message for all errors."""
    _default_handler.user_message = message