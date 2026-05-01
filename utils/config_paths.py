"""Config path utilities - central helper for config file resolution."""
from pathlib import Path


def get_config_path(filename: str) -> Path:
    """
    Get the full path to a config file.
    
    Args:
        filename: Config filename (e.g., 'trading.yaml', 'llm.yaml')
        
    Returns:
        Full Path to config file
    """
    return Path(__file__).parent.parent / "config" / filename


def get_data_path(filename: str) -> Path:
    """
    Get the full path to a data file.
    
    Args:
        filename: Data filename (e.g., 'gold_news.json')
        
    Returns:
        Full Path to data file
    """
    return Path(__file__).parent.parent / "data" / filename


def ensure_data_dir() -> Path:
    """Ensure data directory exists."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def ensure_config_dir() -> Path:
    """Ensure config directory exists."""
    config_dir = Path(__file__).parent.parent / "config"
    config_dir.mkdir(exist_ok=True)
    return config_dir