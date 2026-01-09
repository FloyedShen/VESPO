"""
Global Registry for Cross-Process State Sharing

Uses multiprocessing.Manager to store state that persists across spawn boundaries.
The session ID and Manager address are passed via environment variables.
"""

import os
import multiprocessing as mp
from typing import Dict, Optional, Any

# Manager instance (created once in main process)
_MANAGER: Optional[mp.managers.SyncManager] = None

# Manager dict for storing session state (shared across processes)
_REGISTRY_DICT: Optional[Dict[str, Dict[str, Any]]] = None

# Environment variable names
ENV_SESSION_ID = "VLLM_DUAL_ENGINE_SESSION_ID"


def initialize_manager():
    """Initialize the global Manager (call once in main process)"""
    global _MANAGER, _REGISTRY_DICT
    if _MANAGER is None:
        _MANAGER = mp.Manager()
        _REGISTRY_DICT = _MANAGER.dict()


def register_session(session_id: str, sync_proxy: dict, alpha_computer,
                     enable_optimal_sampling: bool = True):
    """
    Register a session's shared state in the Manager registry

    Args:
        session_id: Unique session identifier
        sync_proxy: Manager proxy dict from DualEngineSyncStateV2.get_proxy()
        alpha_computer: AlphaComputer instance
        enable_optimal_sampling: Whether to enable optimal sampling
    """
    if _REGISTRY_DICT is None:
        raise RuntimeError("Manager not initialized. Call initialize_manager() first.")

    _REGISTRY_DICT[session_id] = {
        'sync_proxy': sync_proxy,
        'alpha_computer': alpha_computer,
        'enable_optimal_sampling': enable_optimal_sampling
    }


def get_session_state(session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve session state from Manager registry

    Args:
        session_id: Session ID (if None, reads from environment variable)

    Returns:
        Dict with 'sync_proxy', 'alpha_computer', 'enable_optimal_sampling'
        or None if not found
    """
    # Get Manager dict (reconnect if needed)
    global _REGISTRY_DICT
    if _REGISTRY_DICT is None:
        # In subprocess: try to reconnect to parent's Manager
        # The Manager connection is preserved through pickling
        return None

    if session_id is None:
        # Try to get from environment variable
        session_id = os.environ.get(ENV_SESSION_ID)

    if session_id is None or session_id not in _REGISTRY_DICT:
        return None

    return dict(_REGISTRY_DICT[session_id])


def unregister_session(session_id: str):
    """Remove session from registry"""
    if _REGISTRY_DICT is not None and session_id in _REGISTRY_DICT:
        del _REGISTRY_DICT[session_id]


def set_session_id_env(session_id: str):
    """Set session ID in environment variable"""
    os.environ[ENV_SESSION_ID] = session_id


def clear_session_id_env():
    """Clear session ID from environment variable"""
    if ENV_SESSION_ID in os.environ:
        del os.environ[ENV_SESSION_ID]


def shutdown_manager():
    """Shutdown the global Manager"""
    global _MANAGER, _REGISTRY_DICT
    if _MANAGER is not None:
        try:
            _MANAGER.shutdown()
        except:
            pass
        _MANAGER = None
        _REGISTRY_DICT = None
