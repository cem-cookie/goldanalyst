"""LLM helper utilities - timeout, retry logic, and config loading."""
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _load_llm_config() -> dict:
    """Load LLM configuration from config/llm.yaml with defaults."""
    config_path = Path(__file__).parent.parent / "config" / "llm.yaml"
    defaults = {
        "llm_timeout_seconds": 15,
        "llm_max_retries": 3,
        "llm_base_delay_seconds": 1,
    }
    if config_path.exists():
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg:
                defaults.update(cfg)
        except Exception as e:
            print(f"[WARN] Failed to load LLM config: {e}")
    return defaults


LLM_CONFIG = _load_llm_config()


def call_with_timeout(
    func: Callable[..., T],
    timeout_seconds: int | None = None,
    default_return: Any = None,
    *args,
    **kwargs,
) -> T | Any:
    """
    Call a function with timeout handling.
    
    Args:
        func: Function to call
        timeout_seconds: Timeout in seconds (uses config default if None)
        default_return: Value to return on timeout
        *args, **kwargs: Arguments passed to func
        
    Returns:
        Function result or default_return on timeout
    """
    import threading
    
    timeout = timeout_seconds or LLM_CONFIG.get("llm_timeout_seconds", 15)
    result = [default_return]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"[WARN] LLM call timed out after {timeout}s")
        return default_return
    if exception[0]:
        raise exception[0]
    return result[0]


def call_with_retry(
    func: Callable[..., T],
    max_retries: int | None = None,
    base_delay: float | None = None,
    *args,
    **kwargs,
) -> T | Any:
    """
    Call a function with exponential backoff retry.
    
    Args:
        func: Function to call
        max_retries: Maximum retry attempts (uses config default if None)
        base_delay: Base delay in seconds (uses config default if None)
        *args, **kwargs: Arguments passed to func
        
    Returns:
        Function result
    """
    max_retries = max_retries or LLM_CONFIG.get("llm_max_retries", 3)
    base_delay = base_delay or LLM_CONFIG.get("llm_base_delay_seconds", 1)
    
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"[INFO] LLM retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                time.sleep(delay)
    
    print(f"[WARN] LLM call failed after {max_retries + 1} attempts")
    raise last_exception


def call_with_timeout_and_retry(
    func: Callable[..., T],
    timeout_seconds: int | None = None,
    max_retries: int | None = None,
    base_delay: float | None = None,
    default_return: Any = None,
    *args,
    **kwargs,
) -> T | Any:
    """
    Call a function with both timeout and retry handling.
    First applies retry logic, then timeout per attempt.
    
    Args:
        func: Function to call
        timeout_seconds: Timeout per attempt
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        default_return: Value to return on timeout
        *args, **kwargs: Arguments passed to func
        
    Returns:
        Function result or default_return
    """
    max_retries = max_retries or LLM_CONFIG.get("llm_max_retries", 3)
    base_delay = base_delay or LLM_CONFIG.get("llm_base_delay_seconds", 1)
    timeout = timeout_seconds or LLM_CONFIG.get("llm_timeout_seconds", 15)
    
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            result = call_with_timeout(func, timeout, None, *args, **kwargs)
            if result is not None:
                return result
        except Exception as e:
            last_exception = e
        
        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            print(f"[INFO] LLM retry {attempt + 1}/{max_retries} after {delay}s")
            time.sleep(delay)
    
    print(f"[WARN] LLM call failed after {max_retries + 1} attempts, returning default")
    return default_return