# Copyright 2024 IS Reshape Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
IS Reshape Megatron Actor

This module is kept for backwards compatibility but the actual implementation
is in megatron_workers.py via monkey-patching.

Note: The update_policy method is monkey-patched in ISReshapeMegatronActorRolloutRefWorker.init_model()
with _is_reshape_update_policy from megatron_workers.py. NoMu support is implemented there.
"""

from verl.workers.actor.megatron_actor import MegatronPPOActor

# Import to register "is_reshape" and related policy loss functions
import recipe.vespo.code.core_algos  # noqa: F401

__all__ = ["ISReshapeMegatronPPOActor"]


class ISReshapeMegatronPPOActor(MegatronPPOActor):
    """
    Extended MegatronPPOActor for IS Reshape experiments.

    Note: The actual update_policy implementation with per-update metrics tracking
    and NoMu support is monkey-patched from megatron_workers.py.
    See _is_reshape_update_policy in megatron_workers.py for the actual implementation.
    """
    pass
