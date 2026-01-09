"""
Base Sync LogitsProcessor - Common functionality for dual-engine architecture

Provides common implementation for Engine A and Engine B LogitsProcessors
to eliminate code duplication.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate

from .sync_state_v2 import DualEngineSyncStateV2
from .alpha_computer import AlphaComputer
from . import proxy_storage

logger = logging.getLogger(__name__)


class BaseSyncLogitsProcessor(LogitsProcessor):
    """
    Base class for dual-engine synchronization LogitsProcessors

    Provides common functionality for both Engine A and Engine B:
    - Request state management
    - Proxy loading and session configuration
    - Batch update handling
    - Alpha statistics tracking
    - Parallel synchronization logic

    Subclasses only need to:
    - Define self.engine_id ('A' or 'B')
    - Call super().__init__() with appropriate engine_id
    """

    def __init__(
        self,
        engine_id: str,
        vllm_config,
        device: torch.device,
        is_pin_memory: bool
    ):
        """
        Initialize base sync processor

        Args:
            engine_id: 'A' or 'B' to identify this engine
            vllm_config: vLLM configuration
            device: torch device
            is_pin_memory: Whether to pin memory
        """
        # Note: Do NOT call super().__init__() - it raises NotImplementedError

        self.engine_id = engine_id

        # Load sync proxy from file storage (set by main process)
        sync_proxy = proxy_storage.load_proxy_from_file()
        if sync_proxy is None:
            raise RuntimeError(
                f"SyncLogitsProcessor{engine_id}: Failed to load proxy from file. "
                "Ensure proxy_storage.save_proxy_to_file() was called before creating LLM."
            )

        self.sync_proxy = sync_proxy

        # Get session configuration from proxy's registry
        session_id = proxy_storage.get_session_id_env()
        if session_id is None:
            raise RuntimeError(
                f"SyncLogitsProcessor{engine_id}: Session ID not set in environment. "
                "Call proxy_storage.set_session_id_env() before creating LLM."
            )

        registry_dict = sync_proxy['registry_dict']
        if session_id not in registry_dict:
            raise RuntimeError(
                f"SyncLogitsProcessor{engine_id}: Session {session_id} not found in registry. "
                f"Available sessions: {list(registry_dict.keys())}"
            )

        session_config = registry_dict[session_id]
        self.alpha_computer = session_config['alpha_computer']
        self.enable_optimal_sampling = session_config['enable_optimal_sampling']

        # Request state tracking
        # batch_index -> (prompt_tokens, output_tokens, sampling_params, request_id)
        self.request_states: Dict[int, Tuple] = {}

        # Alpha history for statistics
        # request_id -> List[float]
        self.alpha_history: Dict[str, List[float]] = {}

        logger.info(f"[Engine {engine_id}] SyncLogitsProcessor initialized")

    def is_argmax_invariant(self) -> bool:
        """
        Optimal sampling modifies logits and can change argmax.
        Return False to ensure it's applied even in greedy sampling.
        """
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """
        Update internal state based on batch changes

        Called by vLLM before each generation step to notify about:
        - New requests added (batch_update.added)
        - Requests finished (batch_update.removed)
        - Batch reorganization (batch_update.moved)
        """
        logger.info(
            f"[Engine {self.engine_id}] update_state() called, "
            f"batch_update={'None' if batch_update is None else 'exists'}"
        )

        if batch_update is None:
            logger.info(f"[Engine {self.engine_id}] batch_update is None, returning early")
            return

        logger.info(
            f"[Engine {self.engine_id}] batch_update.added: "
            f"{len(batch_update.added) if batch_update.added else 0} items"
        )
        logger.info(
            f"[Engine {self.engine_id}] batch_update.removed: "
            f"{len(batch_update.removed) if batch_update.removed else 0} items"
        )
        logger.info(
            f"[Engine {self.engine_id}] batch_update.moved: "
            f"{len(batch_update.moved) if batch_update.moved else 0} items"
        )

        # Handle new requests
        if batch_update.added is not None:
            for item in batch_update.added:
                # Unpack batch_update item
                idx = item[0]
                params = item[1]
                prompt_tok_ids = item[2]
                output_tok_ids = item[3]

                # Extract request_id from params.extra_args
                request_id = self._extract_request_id(params, idx)

                # Store request state
                self.request_states[idx] = (
                    prompt_tok_ids,
                    output_tok_ids,
                    params,
                    request_id
                )

                logger.info(f"[Engine {self.engine_id}] Request {request_id} added at index {idx}")

        # Handle removed requests (finished generation)
        if batch_update.removed is not None:
            for idx in batch_update.removed:
                if idx in self.request_states:
                    _, _, _, request_id = self.request_states[idx]
                    logger.info(f"[Engine {self.engine_id}] Request {request_id} removed from index {idx}")

                    # Remove from local state
                    del self.request_states[idx]
                    # Keep alpha history for debugging

        # Handle batch reorganization (index changes)
        if batch_update.moved is not None and len(batch_update.moved) > 0:
            new_states = {}
            for move in batch_update.moved:
                old_idx = move[0]
                new_idx = move[1]
                if old_idx in self.request_states:
                    new_states[new_idx] = self.request_states[old_idx]
                    logger.info(f"[Engine {self.engine_id}] Moved request from {old_idx} -> {new_idx}")
            self.request_states = new_states

        logger.info(
            f"[Engine {self.engine_id}] After update_state, request_states has "
            f"{len(self.request_states)} items: {list(self.request_states.keys())}"
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply optimal sampling by synchronizing with the other engine (PARALLEL SYNC)

        Args:
            logits: [batch_size, vocab_size] logits from this engine

        Returns:
            Mixed logits [batch_size, vocab_size] based on optimal sampling
        """
        logger.info(f"[Engine {self.engine_id}] apply() called with logits shape: {logits.shape}")
        logger.info(f"[Engine {self.engine_id}] enable_optimal_sampling: {self.enable_optimal_sampling}")
        logger.info(f"[Engine {self.engine_id}] request_states: {list(self.request_states.keys())}")

        if not self.enable_optimal_sampling:
            logger.info(f"[Engine {self.engine_id}] Optimal sampling DISABLED, returning original logits")
            return logits

        batch_size = logits.shape[0]
        logger.info(f"[Engine {self.engine_id}] Processing batch_size={batch_size} with PARALLEL sync")

        # Step 1: Collect all sync tasks
        sync_tasks = []  # List of (batch_idx, request_id)
        for batch_idx in range(batch_size):
            if batch_idx not in self.request_states:
                logger.info(
                    f"[Engine {self.engine_id}] batch_idx={batch_idx} NOT in request_states, skipping"
                )
                continue

            _, _, _, request_id = self.request_states[batch_idx]
            sync_tasks.append((batch_idx, request_id))
            logger.info(
                f"[Engine {self.engine_id}] Added sync task: batch_idx={batch_idx}, "
                f"request_id={request_id}"
            )

        if not sync_tasks:
            logger.info(f"[Engine {self.engine_id}] No sync tasks, returning original logits")
            return logits

        logger.info(f"[Engine {self.engine_id}] Launching {len(sync_tasks)} parallel sync requests...")

        # Step 2: Parallel launch sync requests using ThreadPoolExecutor
        futures_map = {}  # future -> (batch_idx, request_id)

        with ThreadPoolExecutor(max_workers=min(len(sync_tasks), 32)) as executor:
            # Submit all sync requests
            for batch_idx, request_id in sync_tasks:
                future = executor.submit(
                    DualEngineSyncStateV2.sync_and_exchange,
                    proxy=self.sync_proxy,
                    engine_id=self.engine_id,
                    request_id=request_id,
                    logits=logits[batch_idx]  # [vocab_size]
                )
                futures_map[future] = (batch_idx, request_id)
                logger.debug(f"[Engine {self.engine_id}] Submitted sync for request_id={request_id}")

            # Step 3: Parallel wait - collect results as they complete
            logger.info(
                f"[Engine {self.engine_id}] Waiting for {len(futures_map)} "
                "sync operations to complete..."
            )

            for future in as_completed(futures_map):
                batch_idx, request_id = futures_map[future]

                try:
                    # Get sync result (other engine's logits)
                    other_logits = future.result()
                    logger.debug(
                        f"[Engine {self.engine_id}] Request {request_id}: Sync completed, "
                        f"other_logits={'None' if other_logits is None else 'tensor'}"
                    )

                    if other_logits is None:
                        # Sync failed (timeout), fall back to this engine only
                        logger.warning(
                            f"[Engine {self.engine_id}] Request {request_id}: "
                            f"Sync failed, using Engine {self.engine_id} only"
                        )
                        continue

                    # Step 4: Mix logits
                    my_logits = logits[batch_idx]
                    mixed_logits = self._mix_logits(my_logits, other_logits, request_id)
                    logits[batch_idx] = mixed_logits

                except Exception as e:
                    logger.error(
                        f"[Engine {self.engine_id}] Request {request_id}: "
                        f"Error during sync/mixing: {e}"
                    )
                    # Fall back to this engine only (no modification to logits)
                    continue

        logger.info(f"[Engine {self.engine_id}] Parallel sync completed for batch_size={batch_size}")
        return logits

    def _mix_logits(
        self,
        my_logits: torch.Tensor,
        other_logits: torch.Tensor,
        request_id: str
    ) -> torch.Tensor:
        """
        Mix logits from two engines using optimal sampling

        Args:
            my_logits: [vocab_size] Logits from this engine
            other_logits: [vocab_size] Logits from other engine
            request_id: Request identifier for statistics

        Returns:
            Mixed logits [vocab_size]
        """
        # Convert to probabilities
        my_probs = F.softmax(my_logits, dim=-1)
        other_probs = F.softmax(other_logits, dim=-1)

        # Determine which is theta (smaller) and which is teacher (larger)
        # Engine A is typically Teacher, Engine B is typically Theta
        if self.engine_id == 'A':
            probs_t = my_probs  # Teacher
            probs_theta = other_probs  # Theta
        else:  # Engine B
            probs_theta = my_probs  # Theta
            probs_t = other_probs  # Teacher

        # Compute alpha (teacher weight)
        alpha = self.alpha_computer.compute(
            probs_theta=probs_theta.unsqueeze(0),  # [1, vocab_size]
            probs_t=probs_t.unsqueeze(0)  # [1, vocab_size]
        ).item()

        # Track alpha for statistics
        if request_id not in self.alpha_history:
            self.alpha_history[request_id] = []
        self.alpha_history[request_id].append(alpha)

        # Compute optimal distribution q*
        q_star = self.alpha_computer.compute_q_star(
            probs_theta=probs_theta.unsqueeze(0),
            probs_t=probs_t.unsqueeze(0),
            alpha=alpha
        ).squeeze(0)  # [vocab_size]

        # Convert back to logits (for vLLM's sampler)
        mixed_logits = torch.log(q_star + 1e-10)

        logger.debug(
            f"[Engine {self.engine_id}] Request {request_id}: "
            f"Mixed logits with alpha={alpha:.3f}"
        )

        return mixed_logits

    def _extract_request_id(self, params, default_idx: int) -> str:
        """
        Extract request_id from sampling parameters

        Args:
            params: SamplingParams object
            default_idx: Default index if no ID found

        Returns:
            request_id string
        """
        if hasattr(params, 'extra_args') and params.extra_args is not None:
            if 'request_id' in params.extra_args:
                return params.extra_args['request_id']

        # Fall back to index-based ID
        return f"req_{default_idx}"

    def get_alpha_stats(self, request_id: str) -> Optional[Dict]:
        """
        Get alpha statistics for a specific request

        Args:
            request_id: Request identifier

        Returns:
            Dictionary with alpha statistics or None if not available
        """
        if request_id not in self.alpha_history:
            return None

        history = self.alpha_history[request_id]
        if len(history) == 0:
            return None

        import numpy as np
        return {
            'mean': float(np.mean(history)),
            'std': float(np.std(history)),
            'min': float(np.min(history)),
            'max': float(np.max(history)),
            'count': len(history),
            'history': history
        }

    def get_sync_statistics(self) -> Dict:
        """Get synchronization statistics from sync_state via proxy"""
        stats_dict = self.sync_proxy['stats_dict']
        sync_count = stats_dict.get('sync_count', 0)
        timeout_count = stats_dict.get('timeout_count', 0)
        total_wait = stats_dict.get('total_wait_time', 0.0)

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
