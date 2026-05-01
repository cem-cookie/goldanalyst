"""Input validation utilities - lightweight validators."""
from typing import Any


def ensure_int(value: Any, param_name: str, min_val: int | None = None, max_val: int | None = None) -> int:
    """Ensure value is an integer within optional range."""
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{param_name} must be an integer, got {type(value).__name__}")
    
    if min_val is not None and result < min_val:
        raise ValueError(f"{param_name} must be >= {min_val}, got {result}")
    if max_val is not None and result > max_val:
        raise ValueError(f"{param_name} must be <= {max_val}, got {result}")
    
    return result


def ensure_float(value: Any, param_name: str, min_val: float | None = None, max_val: float | None = None) -> float:
    """Ensure value is a float within optional range."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{param_name} must be a number, got {type(value).__name__}")
    
    if min_val is not None and result < min_val:
        raise ValueError(f"{param_name} must be >= {min_val}, got {result}")
    if max_val is not None and result > max_val:
        raise ValueError(f"{param_name} must be <= {max_val}, got {result}")
    
    return result


def ensure_str(value: Any, param_name: str, min_len: int | None = None, max_len: int | None = None) -> str:
    """Ensure value is a string within optional length."""
    if not isinstance(value, str):
        raise ValueError(f"{param_name} must be a string, got {type(value).__name__}")
    
    length = len(value)
    if min_len is not None and length < min_len:
        raise ValueError(f"{param_name} must be at least {min_len} characters, got {length}")
    if max_len is not None and length > max_len:
        raise ValueError(f"{param_name} must be at most {max_len} characters, got {length}")
    
    return value


def ensure_in_range(value: float, param_name: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Ensure value is within [min_val, max_val] range (common for normalized values)."""
    result = ensure_float(value, param_name, min_val, max_val)
    if result < min_val or result > max_val:
        raise ValueError(f"{param_name} must be between {min_val} and {max_val}, got {result}")
    return result


def ensure_positive(value: float, param_name: str) -> float:
    """Ensure value is positive."""
    if value <= 0:
        raise ValueError(f"{param_name} must be positive, got {value}")
    return value


def ensure_list(value: Any, param_name: str, min_len: int = 0, max_len: int | None = None) -> list:
    """Ensure value is a list with optional length constraints."""
    if not isinstance(value, list):
        raise ValueError(f"{param_name} must be a list, got {type(value).__name__}")
    
    length = len(value)
    if length < min_len:
        raise ValueError(f"{param_name} must have at least {min_len} items, got {length}")
    if max_len is not None and length > max_len:
        raise ValueError(f"{param_name} must have at most {max_len} items, got {length}")
    
    return value