"""
Dual-Engine Synchronization State

Core synchronization mechanism for dual-engine optimal sampling.
Enables two vLLM engines to synchronize at LogitsProcessor layer and exchange logits.

Key Features:
- Per-request synchronization (handles batch size mismatch)
- Timeout protection (prevents deadlocks)
- Thread-safe operations
- Automatic cleanup
- Graceful degradation on failures
"""

import threading
import time
import torch
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RequestLogits:
    """Logits for a specific request at a specific token position"""
    tensor: torch.Tensor  # [vocab_size]
    timestamp: float
    ready: threading.Event


class DualEngineSyncState:
    """
    Shared state for synchronizing two vLLM engines

    Architecture:
    - Two engines (A and B) run independently
    - Each maintains its own KV cache (O(n) per token)
    - At LogitsProcessor, they synchronize and exchange logits
    - No recomputation needed (eliminates O(n²) overhead)

    Thread Safety:
    - Uses threading.RLock for state protection
    - Uses threading.Barrier for synchronization
    - Timeout-protected to prevent deadlocks

    Usage:
        sync_state = DualEngineSyncState(timeout=5.0)

        # In Engine A's LogitsProcessor
        other_logits = sync_state.sync_and_exchange('A', request_id, my_logits)

        # In Engine B's LogitsProcessor (blocks until A arrives)
        other_logits = sync_state.sync_and_exchange('B', request_id, my_logits)
    """

    def __init__(self, timeout: float = 5.0, enable_logging: bool = True):
        """
        Initialize synchronization state

        Args:
            timeout: Maximum wait time at sync point (seconds)
            enable_logging: Whether to log sync events
        """
        self.timeout = timeout
        self.enable_logging = enable_logging

        # Logits storage: engine_id -> request_id -> RequestLogits
        self.logits: Dict[str, Dict[str, RequestLogits]] = {
            'A': {},
            'B': {}
        }

        # Per-request barriers for synchronization
        # request_id -> threading.Barrier(2)
        self.barriers: Dict[str, threading.Barrier] = {}

        # Global lock for state modification
        self.lock = threading.RLock()

        # Statistics
        self.sync_count = 0
        self.timeout_count = 0
        self.total_wait_time = 0.0

    def sync_and_exchange(
        self,
        engine_id: str,
        request_id: str,
        logits: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Synchronization point: block until both engines have logits ready,
        then exchange logits.

        Args:
            engine_id: 'A' or 'B' (identifies which engine is calling)
            request_id: Unique request identifier (must match across engines)
            logits: This engine's logits [vocab_size]

        Returns:
            Other engine's logits [vocab_size], or None if sync failed

        Raises:
            TimeoutError: If sync takes longer than timeout
        """
        start_time = time.time()

        try:
            # Create barrier if first time seeing this request
            with self.lock:
                if request_id not in self.barriers:
                    self.barriers[request_id] = threading.Barrier(
                        parties=2,  # Two engines must sync
                        timeout=self.timeout
                    )
                barrier = self.barriers[request_id]

            # Store my logits
            with self.lock:
                self.logits[engine_id][request_id] = RequestLogits(
                    tensor=logits.clone().detach(),  # Clone to avoid sharing
                    timestamp=time.time(),
                    ready=threading.Event()
                )
                self.logits[engine_id][request_id].ready.set()

            if self.enable_logging:
                logger.debug(f"[{engine_id}] Request {request_id}: Logits ready, waiting at barrier")

            # Wait at barrier for both engines to arrive
            try:
                barrier.wait()
            except threading.BrokenBarrierError as e:
                # Barrier broken (timeout or other engine crashed)
                self._cleanup_request(request_id)
                raise TimeoutError(
                    f"Sync timeout for request {request_id} (engine {engine_id})"
                ) from e

            # Both engines arrived, read other's logits
            other_id = 'B' if engine_id == 'A' else 'A'

            with self.lock:
                if request_id not in self.logits[other_id]:
                    raise RuntimeError(
                        f"Missing logits from engine {other_id} for request {request_id}"
                    )

                other_logits = self.logits[other_id][request_id].tensor

            # Update statistics
            wait_time = time.time() - start_time
            with self.lock:
                self.sync_count += 1
                self.total_wait_time += wait_time

            if self.enable_logging:
                logger.debug(
                    f"[{engine_id}] Request {request_id}: Sync complete "
                    f"(waited {wait_time*1000:.1f}ms)"
                )

            return other_logits

        except TimeoutError as e:
            # Log timeout and increment counter
            with self.lock:
                self.timeout_count += 1

            logger.warning(
                f"[{engine_id}] Request {request_id}: Sync timeout after {self.timeout}s"
            )

            # Return None to signal fallback to single-engine mode
            return None

        except Exception as e:
            logger.error(
                f"[{engine_id}] Request {request_id}: Sync failed with error: {e}"
            )
            self._cleanup_request(request_id)
            return None

    def _cleanup_request(self, request_id: str):
        """
        Remove completed request from state

        Called after sync completes or on error.
        """
        with self.lock:
            self.logits['A'].pop(request_id, None)
            self.logits['B'].pop(request_id, None)

            # Remove barrier (will be recreated if needed)
            barrier = self.barriers.pop(request_id, None)
            if barrier is not None:
                try:
                    barrier.abort()  # Wake up any waiting threads
                except:
                    pass

    def cleanup_completed_request(self, request_id: str):
        """
        Public method to cleanup a completed request

        Should be called when a request finishes generation.
        """
        self._cleanup_request(request_id)

    def get_statistics(self) -> dict:
        """
        Get synchronization statistics

        Returns:
            Dictionary with sync metrics:
            - sync_count: Total number of successful syncs
            - timeout_count: Number of timeouts
            - avg_wait_time: Average wait time per sync (seconds)
            - timeout_rate: Timeout percentage
        """
        with self.lock:
            if self.sync_count == 0:
                avg_wait = 0.0
            else:
                avg_wait = self.total_wait_time / self.sync_count

            total_attempts = self.sync_count + self.timeout_count
            if total_attempts == 0:
                timeout_rate = 0.0
            else:
                timeout_rate = self.timeout_count / total_attempts

            return {
                'sync_count': self.sync_count,
                'timeout_count': self.timeout_count,
                'avg_wait_time': avg_wait,
                'timeout_rate': timeout_rate,
                'total_wait_time': self.total_wait_time
            }

    def reset_statistics(self):
        """Reset all statistics counters"""
        with self.lock:
            self.sync_count = 0
            self.timeout_count = 0
            self.total_wait_time = 0.0

    def __repr__(self):
        stats = self.get_statistics()
        return (
            f"DualEngineSyncState("
            f"syncs={stats['sync_count']}, "
            f"timeouts={stats['timeout_count']}, "
            f"avg_wait={stats['avg_wait_time']*1000:.1f}ms)"
        )
