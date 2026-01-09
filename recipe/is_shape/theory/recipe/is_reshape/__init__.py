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
IS Reshape: A Unified Framework Connecting SFT and RL

This module implements the IS Reshape algorithm, which provides a continuous
interpolation between Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL)
through importance sampling weight reshaping.

Key Features:
- Unified objective function: L_γ(θ) = (1/γ)(E_μ[w^γ r] - E_μ[r])
- Adaptive γ selection via closed-form solution: γ* = min(1, √(-log ρ_min / σ²))
- Variance control through Rényi divergence
- Geometric interpolation of target distributions

References:
- IS Reshape Unified Framework (2024)
"""

from .policy_loss import compute_policy_loss_is_reshape

__all__ = ["compute_policy_loss_is_reshape"]
