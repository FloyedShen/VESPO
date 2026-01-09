"""
Proxy storage for cross-process communication

Uses temporary files to pass Manager proxies across spawn boundaries.
Optimized to use /dev/shm (tmpfs) for in-memory storage when available.
"""

import os
import pickle
import tempfile
from typing import Optional

# Environment variable for proxy file path
ENV_PROXY_FILE = "VLLM_DUAL_ENGINE_PROXY_FILE"
ENV_SESSION_ID = "VLLM_DUAL_ENGINE_SESSION_ID"

# Preferred storage directory (tmpfs for faster access)
PREFERRED_STORAGE_DIR = "/dev/shm"


def _get_storage_dir() -> str:
    """
    Get the best storage directory for proxy files

    Returns:
        Path to storage directory (prefers /dev/shm if available)
    """
    # Check if /dev/shm exists and is writable (tmpfs - in-memory)
    if os.path.exists(PREFERRED_STORAGE_DIR) and os.access(PREFERRED_STORAGE_DIR, os.W_OK):
        return PREFERRED_STORAGE_DIR

    # Fallback to default temp directory
    return tempfile.gettempdir()


def save_proxy_to_file(proxy: dict) -> str:
    """
    Save Manager proxy to a temporary file (preferably in memory)

    Args:
        proxy: Manager proxy dict from get_proxy()

    Returns:
        Path to the temporary file
    """
    storage_dir = _get_storage_dir()

    # Create temporary file in the chosen directory
    fd, path = tempfile.mkstemp(
        suffix='.pkl',
        prefix='vllm_dual_engine_',
        dir=storage_dir
    )
    os.close(fd)

    # Pickle the proxy
    with open(path, 'wb') as f:
        pickle.dump(proxy, f)

    return path


def load_proxy_from_file(path: Optional[str] = None) -> Optional[dict]:
    """
    Load Manager proxy from file

    Args:
        path: File path (if None, reads from environment variable)

    Returns:
        Manager proxy dict or None if not found
    """
    if path is None:
        path = os.environ.get(ENV_PROXY_FILE)

    if path is None or not os.path.exists(path):
        return None

    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def set_proxy_file_env(path: str):
    """Set proxy file path in environment variable"""
    os.environ[ENV_PROXY_FILE] = path


def clear_proxy_file():
    """Remove proxy file and clear environment variable"""
    path = os.environ.get(ENV_PROXY_FILE)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except:
            pass
    if ENV_PROXY_FILE in os.environ:
        del os.environ[ENV_PROXY_FILE]


def set_session_id_env(session_id: str):
    """Set session ID in environment variable"""
    os.environ[ENV_SESSION_ID] = session_id


def get_session_id_env() -> Optional[str]:
    """Get session ID from environment variable"""
    return os.environ.get(ENV_SESSION_ID)
