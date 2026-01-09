"""
Dual-Engine Synchronization State V2 - Multiprocessing Support

Uses multiprocessing.Manager for cross-process synchronization.
This version works with vLLM's spawn multiprocessing mode.
"""

import multiprocessing as mp
from multiprocessing import managers
import time
import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging
import pickle

logger = logging.getLogger(__name__)


class DualEngineSyncStateV2:
    """
    Shared state for synchronizing two vLLM engines across processes

    Uses multiprocessing.Manager for cross-process communication.
    Works with vLLM V1's spawn multiprocessing mode.

    Architecture:
    - Manager provides shared dict and synchronization primitives
    - Logits are serialized via pickle (Manager handles this)
    - Barriers ensure both engines sync at each token

    Usage:
        # In main process (before creating LLMs)
        sync_state = DualEngineSyncStateV2.create()

        # Pass to both engines
        SyncLogitsProcessorA.set_shared_state(sync_state.get_proxy(), ...)
        SyncLogitsProcessorB.set_shared_state(sync_state.get_proxy(), ...)
    """

    def __init__(self):
        """
        Initialize synchronization state

        IMPORTANT: Call create() classmethod instead of __init__ directly
        """
        # Create manager
        self.manager = mp.Manager()

        # Shared dictionaries (accessible across processes)
        self.logits_dict = self.manager.dict()  # request_id -> serialized logits
        self.barriers_dict = self.manager.dict()  # request_id -> barrier_info
        self.stats_dict = self.manager.dict()  # Sync statistics
        self.registry_dict = self.manager.dict()  # Session registry for cross-process state

        # Global lock for ALL barrier operations (simpler than per-request locks)
        self.barrier_lock = self.manager.Lock()

        # Initialize stats
        self.stats_dict['sync_count'] = 0
        self.stats_dict['timeout_count'] = 0
        self.stats_dict['total_wait_time'] = 0.0

        # Configuration
        self.timeout = 5.0
        self.enable_logging = False

        logger.info("DualEngineSyncStateV2 initialized with Manager")

    @classmethod
    def create(cls, timeout: float = 5.0, enable_logging: bool = False):
        """
        Create a new sync state instance

        Args:
            timeout: Sync timeout in seconds
            enable_logging: Enable verbose logging

        Returns:
            DualEngineSyncStateV2 instance
        """
        instance = cls()
        instance.timeout = timeout
        instance.enable_logging = enable_logging
        return instance

    def get_proxy(self):
        """
        Get a proxy object that can be passed to subprocesses

        Returns:
            Dict containing manager proxies (can be pickled)
        """
        return {
            'logits_dict': self.logits_dict,
            'barriers_dict': self.barriers_dict,
            'stats_dict': self.stats_dict,
            'registry_dict': self.registry_dict,  # Include registry for session state
            'barrier_lock': self.barrier_lock,  # Global lock for atomic barrier updates
            'timeout': self.timeout,
            'enable_logging': self.enable_logging
        }

    @staticmethod
    def sync_and_exchange(
        proxy: dict,
        engine_id: str,
        request_id: str,
        logits: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Synchronization point: block until both engines have logits ready

        This is a static method so it can be called from subprocess.

        Args:
            proxy: Proxy dict from get_proxy()
            engine_id: 'A' or 'B'
            request_id: Unique request identifier
            logits: This engine's logits [vocab_size]

        Returns:
            Other engine's logits [vocab_size], or None if sync failed
        """
        logits_dict = proxy['logits_dict']
        barriers_dict = proxy['barriers_dict']
        stats_dict = proxy['stats_dict']
        barrier_lock = proxy['barrier_lock']  # Global lock for all barriers
        timeout = proxy['timeout']
        enable_logging = proxy.get('enable_logging', False)

        start_time = time.time()

        try:
            # Create barrier for this request if first time
            barrier_key = f"{request_id}_barrier"

            # Initialize barrier if needed (with lock to avoid race)
            with barrier_lock:
                if barrier_key not in barriers_dict:
                    barriers_dict[barrier_key] = {
                        'count': 0,
                        'ready': False,
                        'completed_by': []  # Track which engines have finished
                    }

            # Convert logits to CPU numpy for pickling
            logits_np = logits.detach().cpu().numpy()

            # Store my logits
            key_my = f"{request_id}_{engine_id}"
            logits_dict[key_my] = logits_np

            if enable_logging:
                logger.debug(f"[{engine_id}] Request {request_id}: Logits stored")

            # ATOMIC: Increment barrier count with global lock
            with barrier_lock:
                barrier_info = dict(barriers_dict[barrier_key])
                barrier_info['count'] += 1
                if barrier_info['count'] >= 2:
                    barrier_info['ready'] = True
                barriers_dict[barrier_key] = barrier_info

                if enable_logging:
                    logger.debug(
                        f"[{engine_id}] Request {request_id}: Barrier count={barrier_info['count']}, "
                        f"ready={barrier_info['ready']}"
                    )

            # Wait for both engines to arrive
            wait_start = time.time()
            while True:
                # Check if barrier exists (other engine might have cleaned up)
                if barrier_key not in barriers_dict:
                    # Barrier deleted means other engine finished → ready=True
                    break

                barrier_info = dict(barriers_dict[barrier_key])
                if barrier_info.get('ready', False):
                    break

                # Check timeout
                if time.time() - wait_start > timeout:
                    # Update timeout count
                    stats_dict['timeout_count'] = stats_dict.get('timeout_count', 0) + 1
                    logger.warning(
                        f"[{engine_id}] Request {request_id}: Sync timeout after {timeout}s"
                    )
                    return None

                # Sleep briefly to avoid busy waiting
                time.sleep(0.001)  # 1ms

            # Both engines arrived, read other's logits
            other_id = 'B' if engine_id == 'A' else 'A'
            key_other = f"{request_id}_{other_id}"

            # Wait for other's logits to be available
            wait_start = time.time()
            while key_other not in logits_dict:
                if time.time() - wait_start > timeout:
                    logger.error(
                        f"[{engine_id}] Request {request_id}: Missing logits from {other_id}"
                    )
                    return None
                time.sleep(0.001)

            other_logits_np = logits_dict[key_other]
            other_logits = torch.from_numpy(other_logits_np).to(
                device=logits.device,
                dtype=logits.dtype
            )

            # Update statistics
            wait_time = time.time() - start_time
            stats_dict['sync_count'] = stats_dict.get('sync_count', 0) + 1
            stats_dict['total_wait_time'] = stats_dict.get('total_wait_time', 0.0) + wait_time

            if enable_logging:
                logger.debug(
                    f"[{engine_id}] Request {request_id}: Sync complete "
                    f"(waited {wait_time*1000:.1f}ms)"
                )

            # Cleanup with reference counting
            # Only delete barrier when BOTH engines have finished
            with barrier_lock:
                if barrier_key in barriers_dict:
                    barrier_info = dict(barriers_dict[barrier_key])
                    completed_by = barrier_info.get('completed_by', [])

                    # Mark this engine as completed
                    if engine_id not in completed_by:
                        completed_by.append(engine_id)
                        barrier_info['completed_by'] = completed_by
                        barriers_dict[barrier_key] = barrier_info

                    # If both engines completed, cleanup
                    if len(completed_by) >= 2:
                        try:
                            del logits_dict[key_my]
                        except:
                            pass
                        try:
                            del logits_dict[key_other]
                        except:
                            pass
                        try:
                            del barriers_dict[barrier_key]
                        except:
                            pass

            return other_logits

        except Exception as e:
            logger.error(
                f"[{engine_id}] Request {request_id}: Sync failed with error: {e}"
            )
            return None

    def get_statistics(self) -> dict:
        """
        Get synchronization statistics

        Returns:
            Dictionary with sync metrics
        """
        sync_count = self.stats_dict.get('sync_count', 0)
        timeout_count = self.stats_dict.get('timeout_count', 0)
        total_wait = self.stats_dict.get('total_wait_time', 0.0)

        if sync_count == 0:
            avg_wait = 0.0
        else:
            avg_wait = total_wait / sync_count

        total_attempts = sync_count + timeout_count
        if total_attempts == 0:
            timeout_rate = 0.0
        else:
            timeout_rate = timeout_count / total_attempts

        return {
            'sync_count': sync_count,
            'timeout_count': timeout_count,
            'avg_wait_time': avg_wait,
            'timeout_rate': timeout_rate,
            'total_wait_time': total_wait
        }

    def reset_statistics(self):
        """Reset all statistics counters"""
        self.stats_dict['sync_count'] = 0
        self.stats_dict['timeout_count'] = 0
        self.stats_dict['total_wait_time'] = 0.0

    def cleanup(self):
        """Cleanup manager resources"""
        try:
            self.manager.shutdown()
        except:
            pass

    def __repr__(self):
        stats = self.get_statistics()
        return (
            f"DualEngineSyncStateV2("
            f"syncs={stats['sync_count']}, "
            f"timeouts={stats['timeout_count']}, "
            f"avg_wait={stats['avg_wait_time']*1000:.1f}ms)"
        )

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()
