"""
OptimalSamplingModel: 最优采样分布的实现

支持特性:
- 两种backend: transformers 和 VLLM
- 多种alpha计算方法: fixed, kl_symmetry, entropy
- q* 分布计算和采样
- 完整的诊断信息 (ESS, KL散度等)

核心概念:
- π_θ: Base模型（如Llama-2-7b）
- π_t: Teacher模型（如Llama-2-7b-chat，通常是Instruct模型）
- q*: 最优混合分布 q*(x) = π_θ(x)^(1-α) × π_t(x)^α
- α: **Teacher模型的权重** (α=0→Base, α=1→Teacher, α>0.5→更接近Teacher)

使用示例:
    model = OptimalSamplingModel(
        model_theta_path="meta-llama/Llama-2-7b-hf",        # Base model
        model_t_path="meta-llama/Llama-2-7b-chat-hf",       # Teacher/Instruct model
        backend="transformers",
        alpha_method="kl_symmetry"
    )

    outputs = model.generate(
        prompts=["Hello, how are you?"],
        max_new_tokens=100,
        temperature=1.0
    )

    # 通常期望 alpha > 0.5，因为Teacher模型质量更高
    print(f"Average α: {outputs.alpha_values.mean():.3f}")
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Literal
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


@dataclass
class SamplingOutput:
    """采样输出结果"""
    generated_texts: List[str]  # decode后的文本（如果skip_decode=False）
    generated_ids: torch.Tensor  # 生成的token IDs [batch, seq_len]
    alpha_values: torch.Tensor  # Alpha值 [batch, seq_len]
    ess_ratios: torch.Tensor    # ESS比例 [batch, seq_len]
    diagnostics: Dict[str, any]  # 诊断信息
    logits: Optional[Dict[str, torch.Tensor]] = None  # ✨ 新增：每一步的logits {"theta": [...], "t": [...]}
    q_star_probs: Optional[torch.Tensor] = None  # ✨ 新增：q*概率分布 [batch, seq_len, vocab_size]


class AlphaComputer:
    """Alpha参数计算器（增强数值稳定性版本）

    ✅ 重要：Alpha语义定义
    ==================
    α 表示 **Teacher模型 (π_t)** 的权重：
    - α = 0 → 完全使用 Base模型 (π_θ)
    - α = 1 → 完全使用 Teacher模型 (π_t)
    - α > 0.5 → 更接近 Teacher（符合直觉，因为Teacher质量更高）

    混合公式：q*(x) = π_θ(x)^(1-α) × π_t(x)^α
    """

    def __init__(self, method: str = "kl_symmetry", fixed_alpha: float = 0.5,
                 tol: float = 1e-6, max_iter: int = 12,
                 constraint_to_target: bool = False,
                 target_top_k: int = -1,
                 target_top_p: float = 1.0):
        """
        Args:
            method: alpha计算方法 ["fixed", "kl_symmetry", "reverse_kl_symmetry", "ess_balance", "entropy"]
            fixed_alpha: 当method="fixed"时使用的固定值（α=0.5表示均匀混合）
            tol: KL对称/ESS平衡求解的容差
            max_iter: 最大迭代次数
            constraint_to_target: 是否限制在π_t的support上（推荐True）
            target_top_k: π_t的top-k限制（-1表示不限制）
            target_top_p: π_t的top-p限制（1.0表示不限制）

        Note:
            α表示Teacher (π_t)的权重。通常期望α > 0.5，因为Teacher模型质量更高。
        """
        self.method = method
        self.fixed_alpha = fixed_alpha
        self.tol = tol
        self.max_iter = max_iter
        self.eps = 1e-10

        # ✅ 新增：Support约束
        self.constraint_to_target = constraint_to_target
        self.target_top_k = target_top_k
        self.target_top_p = target_top_p

    def compute(self, probs_theta: torch.Tensor, probs_t: torch.Tensor) -> torch.Tensor:
        """
        计算alpha（✨ 改进：不再施加support约束，保持alpha计算纯粹）

        Args:
            probs_theta: [batch, vocab_size] 当前策略概率
            probs_t: [batch, vocab_size] 目标策略概率

        Returns:
            alpha: [batch] alpha值

        Note:
            ✨ 新改进：Support约束现在在q*计算之后施加，不影响alpha计算。
            这样可以避免约束让两个分布变得相似，导致alpha偏小的问题。
        """
        # ✅ 移除：不再在alpha计算时施加constraint
        # 原因：constraint会让probs_theta和probs_t变得更相似，影响alpha计算
        # 新方案：先算alpha，再对q*施加constraint

        if self.method == "fixed":
            return self._fixed(probs_theta)
        elif self.method == "kl_symmetry":
            return self._kl_symmetry(probs_theta, probs_t)
        elif self.method == "reverse_kl_symmetry":
            return self._reverse_kl_symmetry(probs_theta, probs_t)
        elif self.method == "ess_balance":
            return self._ess_balance(probs_theta, probs_t)
        elif self.method == "entropy":
            return self._entropy(probs_theta, probs_t)
        else:
            raise ValueError(f"Unknown alpha method: {self.method}")

    def _apply_support_constraint(self, probs_theta: torch.Tensor, probs_t: torch.Tensor) -> tuple:
        """
        限制在π_t的support上（只在π_t认为合理的token上混合）

        这是一个非常重要的数值稳定技巧：
        - 避免Base model的异常token
        - 只在Instruct model支持的空间上做混合
        - 大幅提升数值稳定性

        Args:
            probs_theta: [batch, vocab_size]
            probs_t: [batch, vocab_size]

        Returns:
            (probs_theta_masked, probs_t_masked): 约束后的概率分布
        """
        batch_size, vocab_size = probs_t.shape

        # 创建mask（标记π_t支持的token）
        mask = torch.ones_like(probs_t, dtype=torch.bool)

        # Top-k 约束
        if self.target_top_k > 0:
            k = min(self.target_top_k, vocab_size)
            _, top_k_indices = torch.topk(probs_t, k=k, dim=-1)

            # 创建top-k mask
            mask_k = torch.zeros_like(probs_t, dtype=torch.bool)
            mask_k.scatter_(-1, top_k_indices, True)
            mask = mask & mask_k

        # Top-p 约束
        if self.target_top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs_t, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # 找到累积概率超过top_p的位置
            indices_to_keep = cumsum_probs <= self.target_top_p
            # 至少保留第一个token
            indices_to_keep[..., 0] = True

            # 创建top-p mask
            mask_p = torch.zeros_like(probs_t, dtype=torch.bool)
            mask_p.scatter_(-1, sorted_indices, indices_to_keep)
            mask = mask & mask_p

        # 应用mask
        probs_theta_masked = probs_theta * mask.float()
        probs_t_masked = probs_t * mask.float()

        # 重新归一化
        probs_theta_masked = probs_theta_masked / (probs_theta_masked.sum(dim=-1, keepdim=True) + self.eps)
        probs_t_masked = probs_t_masked / (probs_t_masked.sum(dim=-1, keepdim=True) + self.eps)

        return probs_theta_masked, probs_t_masked

    def apply_constraint_to_q_star(self, q_star: torch.Tensor, probs_t: torch.Tensor) -> torch.Tensor:
        """
        ✨ 新方法：对q*施加support约束（限制在π_t的support上）

        这是改进版的约束应用方式：
        - 先在完整空间计算alpha（避免约束影响alpha）
        - 计算完整的q*
        - 最后对q*施加约束（只保留π_t支持的token）

        Args:
            q_star: [batch, vocab_size] q*概率分布
            probs_t: [batch, vocab_size] π_t概率分布（用于确定support）

        Returns:
            q_star_constrained: [batch, vocab_size] 约束后的q*分布
        """
        if not self.constraint_to_target or (self.target_top_k <= 0 and self.target_top_p >= 1.0):
            # 不需要约束，直接返回
            return q_star

        batch_size, vocab_size = probs_t.shape

        # 创建mask（标记π_t支持的token）
        mask = torch.ones_like(probs_t, dtype=torch.bool)

        # Top-k 约束
        if self.target_top_k > 0:
            k = min(self.target_top_k, vocab_size)
            _, top_k_indices = torch.topk(probs_t, k=k, dim=-1)

            # 创建top-k mask
            mask_k = torch.zeros_like(probs_t, dtype=torch.bool)
            mask_k.scatter_(-1, top_k_indices, True)
            mask = mask & mask_k

        # Top-p 约束
        if self.target_top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs_t, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # 找到累积概率超过top_p的位置
            indices_to_keep = cumsum_probs <= self.target_top_p
            # 至少保留第一个token
            indices_to_keep[..., 0] = True

            # 创建top-p mask
            mask_p = torch.zeros_like(probs_t, dtype=torch.bool)
            mask_p.scatter_(-1, sorted_indices, indices_to_keep)
            mask = mask & mask_p

        # 应用mask到q*
        q_star_masked = q_star * mask.float()

        # 重新归一化
        q_star_masked = q_star_masked / (q_star_masked.sum(dim=-1, keepdim=True) + self.eps)

        return q_star_masked


    def _fixed(self, probs_theta: torch.Tensor) -> torch.Tensor:
        """固定alpha"""
        batch_size = probs_theta.shape[0]
        return torch.full((batch_size,), self.fixed_alpha,
                         device=probs_theta.device, dtype=probs_theta.dtype)

    def _kl_symmetry(self, probs_theta: torch.Tensor, probs_t: torch.Tensor) -> torch.Tensor:
        """
        二分法求解KL对称条件（增强数值稳定性版本）

        目标: D_KL(q||π_θ) = D_KL(q||π_t)
        等价于: E_q[log(π_t/π_θ)] = 0

        改进：
        - ✅ Clamp输入概率避免log(0)
        - ✅ 在log-space计算，避免数值溢出
        - ✅ 检测极端情况并提前返回
        - ✅ 使用logsumexp确保数值稳定
        - ✨ 新增：收敛检测和智能fallback

        注意: 这是ESS平衡条件的一阶近似
        """
        batch_size = probs_theta.shape[0]
        device = probs_theta.device

        # ✅ 数值稳定性保护1：Clamp输入概率
        probs_theta = torch.clamp(probs_theta, min=self.eps, max=1.0)
        probs_t = torch.clamp(probs_t, min=self.eps, max=1.0)

        # ✨ 增强1：检测分布是否几乎相同
        max_diff = (probs_theta - probs_t).abs().max()
        if max_diff < 1e-6:
            # 两个分布几乎完全相同，直接返回0.5
            return torch.full((batch_size,), 0.5, device=device, dtype=probs_theta.dtype)

        # 二分搜索
        alpha_low = torch.full((batch_size,), 0.0, device=device)
        alpha_high = torch.full((batch_size,), 1.0, device=device)

        # ✅ 预计算log概率（避免重复计算）
        log_probs_theta = torch.log(probs_theta)
        log_probs_t = torch.log(probs_t)
        log_ratio = log_probs_t - log_probs_theta  # log(π_t/π_θ)

        # ✨ 增强2：跟踪收敛状态
        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for iteration in range(self.max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2

            # ✅ 在log-space计算 q_alpha
            # ✅ 修改：log q = (1-α) log π_θ + α log π_t （α现在是π_t的权重）
            alpha_expanded = alpha_mid.unsqueeze(-1)
            log_q_unnormalized = (1 - alpha_expanded) * log_probs_theta + alpha_expanded * log_probs_t

            # ✅ 使用logsumexp归一化（数值稳定）
            log_q = log_q_unnormalized - torch.logsumexp(log_q_unnormalized, dim=-1, keepdim=True)
            q_alpha = torch.exp(log_q)

            # 计算 Δ(α) = E_q[log(π_t/π_θ)]
            delta = (q_alpha * log_ratio).sum(dim=-1)

            # ✨ 增强3：检查delta是否有效（NaN/Inf检测）
            invalid = torch.isnan(delta) | torch.isinf(delta)
            if invalid.any():
                # 对无效的样本，标记为已收敛并使用entropy fallback
                converged = converged | invalid
                delta = torch.where(invalid, torch.tensor(0.0, device=device), delta)

            # ✅ 修改：更新区间（反转不等式，因为α含义变了）
            # delta > 0 → q偏向π_t → 需要减小α（减小π_t权重）
            mask = delta > 0
            alpha_high = torch.where(mask, alpha_mid, alpha_high)  # 反转！
            alpha_low = torch.where(mask, alpha_low, alpha_mid)    # 反转！

            # ✨ 增强4：检查收敛
            width = alpha_high - alpha_low
            newly_converged = width < self.tol
            converged = converged | newly_converged

            if converged.all():
                break

        alpha_result = (alpha_low + alpha_high) / 2

        # # ✨ 增强5：对未收敛的样本使用entropy fallback
        # if not converged.all():
        #     num_not_converged = (~converged).sum().item()
        #     if num_not_converged > 0:
        #         print(f"⚠️  Warning: {num_not_converged}/{batch_size} samples did not converge in KL symmetry")
        #         print(f"   Using entropy-based fallback for these samples")
        #
        #         # Entropy-based alpha
        #         h_theta = -(probs_theta * torch.log(probs_theta + self.eps)).sum(dim=-1)
        #         h_t = -(probs_t * torch.log(probs_t + self.eps)).sum(dim=-1)
        #         alpha_entropy = h_theta / (h_theta + h_t + self.eps)
        #
        #         # 对未收敛的样本使用entropy fallback
        #         alpha_result = torch.where(converged, alpha_result, alpha_entropy)

        # ✅ 数值稳定性保护6：最终clamp和NaN检查
        alpha_result = torch.clamp(alpha_result, min=0.0, max=1.0)

        # ✨ 增强6：最终NaN检查
        if torch.isnan(alpha_result).any():
            nan_count = torch.isnan(alpha_result).sum().item()
            print(f"❌ CRITICAL: Alpha has {nan_count} NaN after KL symmetry computation")
            print(f"   Falling back to α=0.75 for all NaN positions")
            alpha_result = torch.where(torch.isnan(alpha_result),
                                        torch.tensor(0.75, device=device),
                                        alpha_result)

        return alpha_result

    def _reverse_kl_symmetry(self, probs_theta: torch.Tensor, probs_t: torch.Tensor) -> torch.Tensor:
        """
        反向KL对称条件（增强数值稳定性版本）

        目标: D_KL(π_θ||q) = D_KL(π_t||q)
        等价于: E_{π_θ}[log q] = E_{π_t}[log q]
        或: Σ (π_θ - π_t) log q = 0

        理论对比：
        - 前向KL (kl_symmetry): D_KL(q||π_θ) = D_KL(q||π_t)
          * Mode-seeking（模式追踪）
          * q倾向于集中在π的单一模式上
          * 从q采样的视角（符合Importance Sampling）

        - 反向KL (reverse_kl_symmetry): D_KL(π_θ||q) = D_KL(π_t||q)
          * Mode-covering（模式覆盖）
          * q倾向于覆盖π的所有模式
          * 从π采样的视角（更探索性）

        注意：反向KL对称不是ESS平衡的直接近似，
        而是一个独立的准则。它会产生更分散、entropy更高的q*。
        """
        batch_size = probs_theta.shape[0]
        device = probs_theta.device

        # ✅ 数值稳定性保护1：Clamp输入概率
        probs_theta = torch.clamp(probs_theta, min=self.eps, max=1.0)
        probs_t = torch.clamp(probs_t, min=self.eps, max=1.0)

        # ✅ 数值稳定性保护2：检测极端情况
        max_ratio = (probs_theta / probs_t).max(dim=-1)[0]
        min_ratio = (probs_theta / probs_t).min(dim=-1)[0]
        nearly_identical = (max_ratio < 1.1) & (min_ratio > 0.9)

        if nearly_identical.any():
            alpha_result = torch.full((batch_size,), 0.5, device=device, dtype=probs_theta.dtype)
            if nearly_identical.all():
                return alpha_result

        # 二分搜索
        alpha_low = torch.full((batch_size,), 0.1, device=device)
        alpha_high = torch.full((batch_size,), 0.9, device=device)

        # ✅ 预计算log概率（避免重复计算）
        log_probs_theta = torch.log(probs_theta)
        log_probs_t = torch.log(probs_t)

        for iteration in range(self.max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2

            # ✅ 在log-space计算 q_alpha
            # ✅ 修改：log q = (1-α) log π_θ + α log π_t （α现在是π_t的权重）
            alpha_expanded = alpha_mid.unsqueeze(-1)
            log_q_unnormalized = (1 - alpha_expanded) * log_probs_theta + alpha_expanded * log_probs_t

            # ✅ 使用logsumexp归一化（数值稳定）
            log_q = log_q_unnormalized - torch.logsumexp(log_q_unnormalized, dim=-1, keepdim=True)

            # 计算 Δ(α) = E_{π_θ}[log q] - E_{π_t}[log q]
            #           = Σ π_θ(x) log q(x) - Σ π_t(x) log q(x)
            #           = Σ (π_θ(x) - π_t(x)) log q(x)
            delta = ((probs_theta - probs_t) * log_q).sum(dim=-1)

            # ✅ 数值稳定性保护3：检查delta是否有效
            invalid = torch.isnan(delta) | torch.isinf(delta)
            if invalid.any():
                alpha_mid = torch.where(invalid, torch.tensor(0.5, device=device), alpha_mid)
                delta = torch.where(invalid, torch.tensor(0.0, device=device), delta)

            # ✅ 修改：更新区间（反转不等式，因为α含义变了）
            # 当 delta > 0 时，说明 E_{π_θ}[log q] > E_{π_t}[log q]
            # 意味着q在π_θ认为可能的区域给予了更高的概率
            # 需要增大π_t的权重，即增大α（新定义）
            mask = delta > 0
            alpha_low = torch.where(mask, alpha_mid, alpha_low)   # 反转！
            alpha_high = torch.where(mask, alpha_high, alpha_mid) # 反转！

            # 检查收敛
            if (alpha_high - alpha_low).max() < self.tol:
                break

        alpha_result = (alpha_low + alpha_high) / 2

        # ✅ 数值稳定性保护4：最终clamp
        alpha_result = torch.clamp(alpha_result, min=0.1, max=0.9)

        # 处理之前检测到的nearly_identical情况
        if nearly_identical.any():
            alpha_result = torch.where(nearly_identical, torch.tensor(0.5, device=device), alpha_result)

        return alpha_result

    def _ess_balance(self, probs_theta: torch.Tensor, probs_t: torch.Tensor) -> torch.Tensor:
        """
        二分法求解ESS平衡条件 (精确条件，带数值稳定性保护)

        目标: ESS_θ(q) = ESS_t(q)
        等价于: Σ(π_θ²/q) = Σ(π_t²/q)
        或: Σ(π_θ²/q) - Σ(π_t²/q) = 0

        这是理论上的精确条件，KL对称只是它的一阶近似。
        根据 theory/proof_final.md:589-592，两者差异通常 < 2%。

        包含6层数值稳定性保护（见ESS_BALANCE_STABILITY.md）。

        注意：
        - α=0时 q=π_θ (只用Base)，α=1时 q=π_t (只用Teacher)
        - 二分搜索找到使ESS平衡的α值
        """
        batch_size = probs_theta.shape[0]
        device = probs_theta.device

        # ========================================
        # 第1步：检查边界条件
        # ========================================
        # 计算 α=0 (只用Base) 和 α=1 (只用Teacher) 时的 ESS 和 sum_sq
        alpha_zero = torch.zeros(batch_size, device=device)
        alpha_one = torch.ones(batch_size, device=device)

        q_0 = self._geometric_mean(probs_theta, probs_t, alpha_zero)
        sum_theta_sq_0 = ((probs_theta ** 2) / (q_0 + self.eps)).sum(dim=-1)
        sum_t_sq_0 = ((probs_t ** 2) / (q_0 + self.eps)).sum(dim=-1)
        ess_theta_0 = 1.0 / (sum_theta_sq_0 + self.eps)
        ess_t_0 = 1.0 / (sum_t_sq_0 + self.eps)
        delta_0 = sum_t_sq_0 - sum_theta_sq_0

        q_1 = self._geometric_mean(probs_theta, probs_t, alpha_one)
        sum_theta_sq_1 = ((probs_theta ** 2) / (q_1 + self.eps)).sum(dim=-1)
        sum_t_sq_1 = ((probs_t ** 2) / (q_1 + self.eps)).sum(dim=-1)
        ess_theta_1 = 1.0 / (sum_theta_sq_1 + self.eps)
        ess_t_1 = 1.0 / (sum_t_sq_1 + self.eps)
        delta_1 = sum_t_sq_1 - sum_theta_sq_1

        # ========================================
        # 第2步：多重检查，判断是否需要回退
        # ========================================
        need_fallback = (
            (delta_0 * delta_1 > 0) |           # 零点不存在（同号）
            (ess_theta_0 < 0.01) | (ess_t_0 < 0.01) |  # ESS过小
            (ess_theta_1 < 0.01) | (ess_t_1 < 0.01) |
            (sum_theta_sq_0 > 100) | (sum_t_sq_0 > 100) |  # sum_sq过大
            (sum_theta_sq_1 > 100) | (sum_t_sq_1 > 100)
        )

        # ========================================
        # 第3步：自适应搜索范围
        # ========================================
        # 如果检测到问题，缩小搜索范围到[0.2, 0.8]
        alpha_low = torch.where(need_fallback,
                               torch.full((batch_size,), 0.2, device=device),
                               torch.zeros(batch_size, device=device))
        alpha_high = torch.where(need_fallback,
                                torch.full((batch_size,), 0.8, device=device),
                                torch.ones(batch_size, device=device))

        # ========================================
        # 第4步：二分搜索
        # ========================================
        for _ in range(self.max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2

            # 计算 q_alpha
            q_alpha = self._geometric_mean(probs_theta, probs_t, alpha_mid)

            # 计算 ESS 差值
            sum_theta_sq = ((probs_theta ** 2) / (q_alpha + self.eps)).sum(dim=-1)
            sum_t_sq = ((probs_t ** 2) / (q_alpha + self.eps)).sum(dim=-1)
            delta = sum_t_sq - sum_theta_sq

            # 更新区间
            mask = delta < 0
            alpha_low = torch.where(mask, alpha_mid, alpha_low)
            alpha_high = torch.where(mask, alpha_high, alpha_mid)

            # 检查收敛
            if (alpha_high - alpha_low).max() < self.tol:
                break

        alpha_result = (alpha_low + alpha_high) / 2

        # ========================================
        # 第5步：对有问题的样本回退到KL对称
        # ========================================
        if need_fallback.any():
            alpha_fallback = self._kl_symmetry(probs_theta, probs_t)
            alpha_result = torch.where(need_fallback, alpha_fallback, alpha_result)

        # ========================================
        # 第6步：最终限制到[0.1, 0.9]
        # ========================================
        return torch.clamp(alpha_result, 0.1, 0.9)

    def _entropy(self, probs_theta: torch.Tensor, probs_t: torch.Tensor) -> torch.Tensor:
        """熵公式快速近似

        启发式：
        - h_theta 高 → Base模型不确定 → 应该更依赖Teacher → α应该高
        - h_t 高 → Teacher模型不确定 → 应该更依赖Base → α应该低

        因此：α = h_theta / (h_theta + h_t)

        注意：这是一个经验公式，不是理论推导的精确解。
        """
        h_theta = -(probs_theta * torch.log(probs_theta + self.eps)).sum(dim=-1)
        h_t = -(probs_t * torch.log(probs_t + self.eps)).sum(dim=-1)
        alpha = h_theta / (h_theta + h_t + self.eps)
        return torch.clamp(alpha, 0.0, 1.0)

    def _geometric_mean(self, p1: torch.Tensor, p2: torch.Tensor,
                       alpha: torch.Tensor) -> torch.Tensor:
        """
        计算几何平均

        ✅ 重要改动：α 现在表示 p2 (通常是 π_t/Teacher) 的权重
        - α = 0 → 完全使用 p1 (π_θ/Base)
        - α = 1 → 完全使用 p2 (π_t/Teacher)
        - α > 0.5 → 更接近 Teacher（符合直觉）

        公式：q = p1^(1-α) × p2^α
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        # ✅ 修改：α 现在是 p2 的权重
        log_q = (1 - alpha) * torch.log(p1 + self.eps) + alpha * torch.log(p2 + self.eps)
        return F.softmax(log_q, dim=-1)


class DiagnosticComputer:
    """诊断信息计算器（增强数值稳定性版本）"""

    def __init__(self):
        self.eps = 1e-10

    def compute(self, probs_theta: torch.Tensor, probs_t: torch.Tensor,
                q_star: torch.Tensor, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算诊断信息（带数值稳定性保护）

        Returns:
            dict with keys: ess_theta, ess_t, ess_ratio, kl_theta, kl_t, kl_diff
        """
        # ✅ 数值稳定性保护：Clamp 所有概率分布
        probs_theta = torch.clamp(probs_theta, min=self.eps, max=1.0)
        probs_t = torch.clamp(probs_t, min=self.eps, max=1.0)
        q_star = torch.clamp(q_star, min=self.eps, max=1.0)

        # ✅ 检查输入是否有效
        if torch.isnan(probs_theta).any() or torch.isnan(probs_t).any() or torch.isnan(q_star).any():
            # 返回全 nan 的诊断信息
            batch_size = probs_theta.shape[0]
            nan_tensor = torch.full((batch_size,), float('nan'), device=probs_theta.device)
            return {
                "alpha": alpha,
                "ess_theta": nan_tensor,
                "ess_t": nan_tensor,
                "ess_ratio": nan_tensor,
                "kl_theta": nan_tensor,
                "kl_t": nan_tensor,
                "kl_diff": nan_tensor,
            }

        # ESS (Effective Sample Size)
        # ESS = 1 / Σ(p²/q)
        sum_theta_sq = ((probs_theta ** 2) / q_star).sum(dim=-1)
        sum_t_sq = ((probs_t ** 2) / q_star).sum(dim=-1)

        # ✅ 避免除以 0 或过大的值
        sum_theta_sq = torch.clamp(sum_theta_sq, min=self.eps, max=1e6)
        sum_t_sq = torch.clamp(sum_t_sq, min=self.eps, max=1e6)

        ess_theta = 1.0 / sum_theta_sq
        ess_t = 1.0 / sum_t_sq
        ess_ratio = ess_theta / (ess_t + self.eps)

        # KL散度：D_KL(q||p) = Σ q(x) log(q(x)/p(x))
        # ✅ 使用 log-space 计算，避免 log(0)
        log_q = torch.log(q_star)
        log_probs_theta = torch.log(probs_theta)
        log_probs_t = torch.log(probs_t)

        kl_theta = (q_star * (log_q - log_probs_theta)).sum(dim=-1)
        kl_t = (q_star * (log_q - log_probs_t)).sum(dim=-1)

        # ✅ 最终检查：替换任何剩余的 inf/nan
        ess_theta = torch.where(torch.isfinite(ess_theta), ess_theta, torch.zeros_like(ess_theta))
        ess_t = torch.where(torch.isfinite(ess_t), ess_t, torch.zeros_like(ess_t))
        ess_ratio = torch.where(torch.isfinite(ess_ratio), ess_ratio, torch.ones_like(ess_ratio))
        kl_theta = torch.where(torch.isfinite(kl_theta), kl_theta, torch.zeros_like(kl_theta))
        kl_t = torch.where(torch.isfinite(kl_t), kl_t, torch.zeros_like(kl_t))

        return {
            "alpha": alpha,
            "ess_theta": ess_theta,
            "ess_t": ess_t,
            "ess_ratio": ess_ratio,
            "kl_theta": kl_theta,
            "kl_t": kl_t,
            "kl_diff": (kl_theta - kl_t).abs(),
        }


class OptimalSamplingModel:
    """最优采样模型"""

    def __init__(
        self,
        model_theta_path: str,
        model_t_path: Optional[str] = None,
        alpha_method: str = "kl_symmetry",
        fixed_alpha: float = 0.5,
        alpha_tol: float = 1e-6,
        constraint_to_target: bool = False,
        target_top_k: int = -1,
        target_top_p: float = 1.0,
        force_target_for_special_tokens: bool = True,
        force_target_for_first_token: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        """
        Args:
            model_theta_path: π_θ 模型路径
            model_t_path: π_t 模型路径 (如果为None, 则使用model_theta_path)
            alpha_method: alpha计算方法 ["fixed", "kl_symmetry", "reverse_kl_symmetry", "entropy", "ess_balance"]
            fixed_alpha: 固定alpha值 (当alpha_method="fixed"时)
            alpha_tol: KL对称求解容差
            constraint_to_target: ✨ 是否限制在π_t的support上（推荐True，提升数值稳定性）
            target_top_k: ✨ π_t的top-k限制（-1表示不限制）
            target_top_p: ✨ π_t的top-p限制（1.0表示不限制）
            force_target_for_special_tokens: ✨ 对special tokens强制使用π_t（从tokenizer获取，推荐True）
            force_target_for_first_token: ✨ 强制第一个token使用π_t（推荐True）
            device: 设备
            dtype: 数据类型
            **kwargs: 传递给transformers的额外参数（如load_in_4bit, attn_implementation等）
        """
        self.device = device
        self.dtype = dtype

        print(device, dtype)

        # 存储special token处理参数
        self.force_target_for_special_tokens = force_target_for_special_tokens
        self.force_target_for_first_token = force_target_for_first_token

        # 初始化模型（仅支持transformers backend）
        self._init_transformers(model_theta_path, model_t_path, **kwargs)

        # 初始化alpha计算器（带support约束）
        self.alpha_computer = AlphaComputer(
            method=alpha_method,
            fixed_alpha=fixed_alpha,
            tol=alpha_tol,
            constraint_to_target=constraint_to_target,
            target_top_k=target_top_k,
            target_top_p=target_top_p
        )

        # 初始化诊断计算器
        self.diagnostic_computer = DiagnosticComputer()

        print(f"✓ OptimalSamplingModel initialized")
        print(f"  Backend: transformers")
        print(f"  Alpha method: {alpha_method}")
        if constraint_to_target:
            print(f"  ✨ Support constraint: ENABLED")
            if target_top_k > 0:
                print(f"     - Target top-k: {target_top_k}")
            if target_top_p < 1.0:
                print(f"     - Target top-p: {target_top_p}")
        if force_target_for_special_tokens and not self.same_model:
            print(f"  ✨ Special token handling: ENABLED")
            print(f"     - For special tokens from tokenizer config, use π_t directly")
        if force_target_for_first_token:
            print(f"  ✨ First token forcing: ENABLED")
            print(f"     - First token will use π_t directly")
        print(f"  π_θ: {model_theta_path}")
        print(f"  π_t: {model_t_path or model_theta_path}")

    def _init_transformers(self, model_theta_path: str, model_t_path: Optional[str], **kwargs):
        """初始化transformers backend（支持不同tokenizer）"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ✅ 移除kwargs中的torch_dtype（避免与self.dtype冲突）
        kwargs.pop('torch_dtype', None)

        # ✅ 默认启用 Flash Attention 2（如果用户没有指定）
        if "attn_implementation" not in kwargs:
            try:
                import flash_attn
                kwargs["attn_implementation"] = "flash_attention_2"
                print("✅ Flash Attention 2 auto-enabled (detected flash_attn package)")
            except ImportError:
                print("ℹ️  Flash Attention 2 not available, using default attention")
                print("   Install with: pip install flash-attn --no-build-isolation")

        # ✨ 打印传递的参数（用于调试）
        if kwargs:
            print(f"\n📦 Model loading parameters:")
            for key, value in kwargs.items():
                if key in ["attn_implementation", "load_in_4bit", "load_in_8bit", "device_map"]:
                    print(f"   - {key}: {value}")
                elif key.startswith("bnb_"):
                    print(f"   - {key}: {value}")
            print()

        # ========================================
        # 加载tokenizer（两个模型可能有不同的tokenizer）
        # ========================================
        print(f"Loading tokenizer for π_θ from {model_theta_path}...")
        self.tokenizer_theta = AutoTokenizer.from_pretrained(model_theta_path)
        if self.tokenizer_theta.pad_token is None:
            self.tokenizer_theta.pad_token = self.tokenizer_theta.eos_token

        # 加载 π_θ
        print(f"Loading π_θ from {model_theta_path}...")
        self.model_theta = AutoModelForCausalLM.from_pretrained(
            model_theta_path,
            torch_dtype=self.dtype,
            device_map=self.device if self.device != "cuda" else "auto",
            **kwargs
        )
        self.model_theta.eval()

        # 加载 π_t 和它的tokenizer
        if model_t_path is None or model_t_path == model_theta_path:
            print(f"Using π_θ as π_t (same model)")
            self.model_t = self.model_theta
            self.tokenizer_t = self.tokenizer_theta
            self.same_model = True
            self.same_tokenizer = True
        else:
            print(f"Loading tokenizer for π_t from {model_t_path}...")
            self.tokenizer_t = AutoTokenizer.from_pretrained(model_t_path)
            if self.tokenizer_t.pad_token is None:
                self.tokenizer_t.pad_token = self.tokenizer_t.eos_token

            print(f"Loading π_t from {model_t_path}...")
            self.model_t = AutoModelForCausalLM.from_pretrained(
                model_t_path,
                torch_dtype=self.dtype,
                device_map=self.device if self.device != "cuda" else "auto",
                **kwargs
            )
            self.model_t.eval()
            self.same_model = False

            # 检查tokenizer是否相同
            self.same_tokenizer = self._check_tokenizer_compatibility()

        # 保持向后兼容：self.tokenizer指向θ的tokenizer
        self.tokenizer = self.tokenizer_theta

        # 如果tokenizer不同，建立vocabulary映射
        if not self.same_tokenizer:
            print("\n⚠️ different tokenizer detected，build vocabulary mapping...")
            self._build_vocab_mapping()

        # 检测special tokens（π_t有但π_θ可能没见过的）
        if self.force_target_for_special_tokens and not self.same_model:
            self._detect_special_tokens()
        else:
            self.special_token_mask = None

    def _check_tokenizer_compatibility(self) -> bool:
        """检查两个tokenizer是否相同/兼容"""
        # 方法1：检查vocab size
        if len(self.tokenizer_theta) != len(self.tokenizer_t):
            print(f"  Tokenizer vocab size: θ={len(self.tokenizer_theta)}, t={len(self.tokenizer_t)}")
            return False

        # 方法2：检查特殊token
        # if (self.tokenizer_theta.eos_token_id != self.tokenizer_t.eos_token_id or
        #     self.tokenizer_theta.bos_token_id != self.tokenizer_t.bos_token_id):
        #     print(f"  Tokenizer special token different: θ(eos={self.tokenizer_theta.eos_token_id}, bos={self.tokenizer_theta.bos_token_id}), ")
        #     return False

        # 方法3：采样检查一些token
        sample_tokens = ["hello", "world", "the", "a", "is"]
        for token_str in sample_tokens:
            id_theta = self.tokenizer_theta.encode(token_str, add_special_tokens=False)
            id_t = self.tokenizer_t.encode(token_str, add_special_tokens=False)
            if id_theta != id_t:
                print(f"  Tokenizer encode diff: '{token_str}' -> θ={id_theta}, t={id_t}")
                return False

        print("  ✓ Tokenizers are compatible")
        return True

    def _build_vocab_mapping(self):
        """
        建立两个vocabulary之间的映射

        核心思想：
        1. 对于每个θ的token ID，找到t中对应的token string
        2. 建立 ID映射: vocab_map_theta_to_t[id_theta] = id_t
        3. 在计算q*时，需要对齐两个模型的logits
        """
        vocab_size_theta = len(self.tokenizer_theta)
        vocab_size_t = len(self.tokenizer_t)

        print(f"  Building vocab mapping: θ({vocab_size_theta}) -> t({vocab_size_t})")

        # 映射: theta_id -> t_id
        self.vocab_map_theta_to_t = {}
        self.vocab_map_t_to_theta = {}

        # 对于θ的每个token，找到t中的对应
        unmapped_count = 0
        for id_theta in range(vocab_size_theta):
            try:
                # Decode token
                token_str = self.tokenizer_theta.decode([id_theta], skip_special_tokens=False)

                # Encode到t的vocabulary
                ids_t = self.tokenizer_t.encode(token_str, add_special_tokens=False)

                if len(ids_t) == 1:
                    # 1对1映射
                    self.vocab_map_theta_to_t[id_theta] = ids_t[0]
                elif len(ids_t) > 1:
                    # 1对多映射（θ的一个token对应t的多个token）
                    # 简化：取第一个
                    self.vocab_map_theta_to_t[id_theta] = ids_t[0]
                else:
                    # 无法映射
                    unmapped_count += 1

            except Exception as e:
                unmapped_count += 1

        # 反向映射
        for id_t in range(vocab_size_t):
            try:
                token_str = self.tokenizer_t.decode([id_t], skip_special_tokens=False)
                ids_theta = self.tokenizer_theta.encode(token_str, add_special_tokens=False)

                if len(ids_theta) == 1:
                    self.vocab_map_t_to_theta[id_t] = ids_theta[0]
                elif len(ids_theta) > 1:
                    self.vocab_map_t_to_theta[id_t] = ids_theta[0]

            except Exception:
                pass

        mapped_ratio = len(self.vocab_map_theta_to_t) / vocab_size_theta
        print(f"  ✓ Mapped {len(self.vocab_map_theta_to_t)}/{vocab_size_theta} tokens ({mapped_ratio:.1%})")

        if unmapped_count > vocab_size_theta * 0.1:
            print(f"  ⚠️  警告: {unmapped_count} tokens无法映射 ({unmapped_count/vocab_size_theta:.1%})")
            print(f"     这可能导致生成质量下降")

        # 存储vocab size以便后续使用
        self.vocab_size_theta = vocab_size_theta
        self.vocab_size_t = vocab_size_t

    def _detect_special_tokens(self):
        """
        检测special tokens（从tokenizer config中获取）

        策略：
        1. 获取π_t的所有special tokens（从tokenizer.all_special_tokens）
        2. 获取π_θ的所有special tokens
        3. 找到所有special tokens（保守策略：使用并集）
        4. 创建mask标记这些token ID

        这些token在Base model中可能没见过（如<|im_start|>等），
        应该直接使用π_t的概率。
        """
        print("\n" + "="*60)
        print("Detecting special tokens...")
        print("="*60)

        # ✅ 使用模型的实际vocab size，而不是tokenizer的
        vocab_size_theta_tokenizer = len(self.tokenizer_theta)
        vocab_size_t_tokenizer = len(self.tokenizer_t)

        # 从模型config获取实际的vocab size
        if hasattr(self.model_theta.config, 'vocab_size'):
            vocab_size_theta = self.model_theta.config.vocab_size
        else:
            vocab_size_theta = vocab_size_theta_tokenizer

        if hasattr(self.model_t.config, 'vocab_size'):
            vocab_size_t = self.model_t.config.vocab_size
        else:
            vocab_size_t = vocab_size_t_tokenizer

        # 使用较大的vocab size（确保能容纳所有可能的token）
        vocab_size = max(vocab_size_theta, vocab_size_t)

        print(f"Tokenizer vocab size: θ={vocab_size_theta_tokenizer}, t={vocab_size_t_tokenizer}")
        print(f"Model vocab size: θ={vocab_size_theta}, t={vocab_size_t}")
        print(f"Using vocab size: {vocab_size}")

        # 初始化mask（False表示正常token，True表示special token）
        self.special_token_mask = torch.zeros(vocab_size, dtype=torch.bool)

        # 获取π_t的special tokens
        special_tokens_t = set()
        if hasattr(self.tokenizer_t, 'all_special_tokens'):
            for token in self.tokenizer_t.all_special_tokens:
                try:
                    token_ids = self.tokenizer_t.encode(token, add_special_tokens=False)
                    special_tokens_t.update(token_ids)
                except:
                    pass

        # 获取π_θ的special tokens
        special_tokens_theta = set()
        if hasattr(self.tokenizer_theta, 'all_special_tokens'):
            for token in self.tokenizer_theta.all_special_tokens:
                try:
                    token_ids = self.tokenizer_theta.encode(token, add_special_tokens=False)
                    special_tokens_theta.update(token_ids)
                except:
                    pass

        # 方案：保守策略 - 所有special tokens都强制使用π_t
        # 原因：即使θ和t都有EOS，Base model对EOS的处理可能不够好
        all_special_tokens = special_tokens_theta | special_tokens_t

        print(f"π_θ special tokens: {len(special_tokens_theta)} unique IDs")
        print(f"π_t special tokens: {len(special_tokens_t)} unique IDs")
        print(f"All special tokens: {len(all_special_tokens)} unique IDs")

        # 标记这些token
        for token_id in all_special_tokens:
            if token_id < vocab_size:
                self.special_token_mask[token_id] = True

        # 打印一些示例
        print(f"\nSpecial token examples:")
        count = 0
        for token_id in sorted(list(all_special_tokens))[:10]:
            if token_id < vocab_size:
                try:
                    token_str_theta = self.tokenizer_theta.decode([token_id])
                    token_str_t = self.tokenizer_t.decode([token_id])
                    in_theta = token_id in special_tokens_theta
                    in_t = token_id in special_tokens_t
                    print(f"  ID {token_id}: θ='{token_str_theta}' ({'✓' if in_theta else '✗'}), "
                          f"t='{token_str_t}' ({'✓' if in_t else '✗'})")
                    count += 1
                except:
                    pass

        if len(all_special_tokens) > 10:
            print(f"  ... and {len(all_special_tokens) - count} more")

        print(f"\n✓ Special token mask created: {self.special_token_mask.sum().item()} tokens marked")
        print("="*60 + "\n")

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        prompts_t: Optional[List[str]] = None,  # ✅ 新增：π_t的prompt（可选）
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        return_diagnostics: bool = True,
        skip_decode: bool = False,  # ✨ 新增：跳过decode，返回空的generated_texts
        return_logits: bool = False,  # ✨ 新增：返回每步的logits
        return_q_star_probs: bool = False,  # ✨ 新增：返回q*概率分布
        **kwargs
    ) -> SamplingOutput:
        """
        使用q*采样生成文本

        Args:
            prompts: π_θ的输入prompts列表
            prompts_t: π_t的输入prompts列表（可选）
                      如果为None，则使用prompts（默认行为，向后兼容）
                      如果提供，则两个模型看到不同的输入
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            return_diagnostics: 是否返回诊断信息
            skip_decode: 是否跳过内部decode（默认False，如果True则generated_texts为空列表）
            return_logits: 是否返回logits（默认False）
            return_q_star_probs: 是否返回q*概率分布（默认False）

        Returns:
            SamplingOutput对象

        Examples:
            # 场景1: 两个模型看相同的prompt（默认）
            >>> outputs = model.generate(prompts=["Hello"])

            # 场景2: 两个模型看不同的prompt
            >>> outputs = model.generate(
            ...     prompts=["Answer briefly: What is AI?"],  # π_θ看简洁版
            ...     prompts_t=["Answer in detail: What is AI?"]  # π_t看详细版
            ... )

            # 场景3: 返回logits并跳过decode（外部decode）
            >>> outputs = model.generate(
            ...     prompts=["Hello"],
            ...     skip_decode=True,
            ...     return_logits=True,
            ...     return_q_star_probs=True
            ... )
            >>> # outputs.generated_texts 为空列表
            >>> # outputs.logits 包含 {"theta": [...], "t": [...]}
            >>> # 在外部decode
            >>> decoded = model.tokenizer.batch_decode(outputs.generated_ids, skip_special_tokens=True)
        """
        return self._generate_transformers(
            prompts, prompts_t, max_new_tokens, temperature, top_p, top_k,
            return_diagnostics, skip_decode, return_logits, return_q_star_probs, **kwargs
        )

    def _generate_transformers(
        self,
        prompts: List[str],
        prompts_t: Optional[List[str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        return_diagnostics: bool,
        skip_decode: bool,  # ✨ 新增参数
        return_logits: bool,  # ✨ 新增参数
        return_q_star_probs: bool,  # ✨ 新增参数
        use_kv_cache: bool = True,  # ✅ 新增参数
        stopping_criteria: Optional[any] = None,  # ✅ 支持自定义停止条件
        **kwargs
    ) -> SamplingOutput:
        """
        Transformers backend的生成（支持双prompt）

        Args:
            prompts: π_θ的prompts
            prompts_t: π_t的prompts（可选）
            skip_decode: 是否跳过decode
            return_logits: 是否返回logits
            return_q_star_probs: 是否返回q*概率
            use_kv_cache: 是否使用KV cache加速（默认True）
            stopping_criteria: transformers.StoppingCriteriaList对象
        """
        batch_size = len(prompts)

        # ========================================
        # 第1步：判断是否使用不同的prompt
        # ========================================
        use_different_prompts = (prompts_t is not None) and (prompts_t != prompts)

        if use_different_prompts:
            if len(prompts_t) != batch_size:
                raise ValueError(f"prompts_t长度({len(prompts_t)})必须与prompts长度({batch_size})相同")

        # ========================================
        # 第2步：Tokenize（可能不同）
        # ========================================
        # Tokenize π_θ的prompt
        inputs_theta = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # ✅ 防止超长输入
        )
        input_ids_theta = inputs_theta["input_ids"].to(self.model_theta.device)
        attention_mask_theta = inputs_theta["attention_mask"].to(self.model_theta.device)

        # Tokenize π_t的prompt（如果不同）
        if use_different_prompts:
            # 检查tokenizer是否相同
            if self.same_model:
                # 同一个模型，tokenizer肯定相同
                inputs_t = self.tokenizer(
                    prompts_t,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )
            else:
                inputs_t = self.tokenizer_t(
                    prompts_t,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )

            input_ids_t = inputs_t["input_ids"].to(self.model_t.device)
            attention_mask_t = inputs_t["attention_mask"].to(self.model_t.device)
        else:
            # 使用相同的prompt（默认行为）
            input_ids_t = input_ids_theta
            attention_mask_t = attention_mask_theta

        # ========================================
        # 准备存储
        # ========================================
        all_generated_ids = []
        all_alpha_values = []
        all_ess_ratios = []
        all_diagnostics = []

        # ✨ 新增：存储 logits 和 q_star_probs
        all_logits_theta = [] if return_logits else None
        all_logits_t = [] if return_logits else None
        all_q_star_probs = [] if return_q_star_probs else None

        # ✅ 停止条件：为每个样本维护finished状态
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.model_theta.device)
        eos_token_id = self.tokenizer_theta.eos_token_id  # 使用θ的EOS（生成token在θ的vocab）

        if use_kv_cache:
            # ========================================
            # ✅ 使用KV cache的高效实现（支持双prompt）
            # ========================================

            # Prefill阶段：处理完整prompt，初始化KV cache
            # π_θ 使用 input_ids_theta
            outputs_theta = self.model_theta(
                input_ids=input_ids_theta,
                attention_mask=attention_mask_theta,
                use_cache=True  # ✅ 启用KV cache
            )
            past_key_values_theta = outputs_theta.past_key_values
            logits_theta = outputs_theta.logits[:, -1, :]

            # π_t 使用 input_ids_t（可能不同）
            if not self.same_model:
                outputs_t = self.model_t(
                    input_ids=input_ids_t,
                    attention_mask=attention_mask_t,
                    use_cache=True
                )
                past_key_values_t = outputs_t.past_key_values
                logits_t = outputs_t.logits[:, -1, :]
            else:
                past_key_values_t = None
                logits_t = logits_theta

            # Decode阶段：逐token生成，复用KV cache
            for step in tqdm(range(max_new_tokens), desc=kwargs['tqdm_desc']):
                # ✅ 对齐logits并计算概率（处理不同tokenizer）
                probs_theta, probs_t = self._align_logits(logits_theta, logits_t, temperature)

                # ✅ 强制第一个token使用π_t
                if step == 0 and self.force_target_for_first_token:
                    # 第一个token直接使用π_t，不进行混合
                    q_star = probs_t
                    # ✅ 修改：α=1 表示完全使用π_t（Teacher）
                    alpha = torch.ones(batch_size, device=probs_theta.device)
                else:
                    # 后续token正常计算alpha和q*
                    # ✨ 改进：先计算alpha，再计算q*，最后施加约束
                    alpha = self.alpha_computer.compute(probs_theta, probs_t)
                    q_star = self._compute_q_star(probs_theta, probs_t, alpha)

                    # ✨ 新增：对q*施加support约束（限制在π_t的support上）
                    # 这样可以避免约束影响alpha计算
                    q_star = self.alpha_computer.apply_constraint_to_q_star(q_star, probs_t)

                # 应用 top-p / top-k
                if top_p < 1.0 or top_k > 0:
                    q_star = self._apply_sampling_filters(q_star, top_p, top_k)

                # ✅ 采样前安全检查
                if torch.isnan(q_star).any() or torch.isinf(q_star).any() or (q_star < 0).any():
                    print(f"⚠️  Warning: Invalid q_star at step {step}, using \\pi_t fallback")
                    # q_star = torch.ones_like(q_star) / q_star.size(-1)
                    q_star = probs_t

                # 从 q* 采样
                try:
                    next_tokens = torch.multinomial(q_star, num_samples=1).squeeze(-1)
                except RuntimeError as e:
                    print(f"⚠️  Sampling failed at step {step}: {e}")
                    print(f"   q_star stats: min={q_star.min():.6f}, max={q_star.max():.6f}, sum={q_star.sum(dim=-1)}")
                    # 回退到argmax
                    next_tokens = q_star.argmax(dim=-1)

                # ✅ 检查本步是否有样本生成EOS
                if eos_token_id is not None:
                    finished = finished | (next_tokens == eos_token_id)

                # ✅ 对已完成的样本，使用pad token（防止继续生成）
                if self.tokenizer_theta.pad_token_id is not None:
                    next_tokens = torch.where(finished,
                                             torch.tensor(self.tokenizer_theta.pad_token_id, device=next_tokens.device),
                                             next_tokens)

                # 计算诊断信息
                if return_diagnostics:
                    diag = self.diagnostic_computer.compute(probs_theta, probs_t, q_star, alpha)
                    all_alpha_values.append(alpha.cpu())
                    all_ess_ratios.append(diag["ess_ratio"].cpu())
                    all_diagnostics.append({k: v.cpu() for k, v in diag.items()})

                # ✨ 新增：收集 logits 和 q_star_probs
                if return_logits:
                    all_logits_theta.append(logits_theta.cpu())
                    all_logits_t.append(logits_t.cpu())
                if return_q_star_probs:
                    all_q_star_probs.append(q_star.cpu())

                # 保存生成的token
                all_generated_ids.append(next_tokens.unsqueeze(-1))

                # ✅ 检查停止条件
                if stopping_criteria is not None:
                    generated_so_far = torch.cat(all_generated_ids, dim=-1)
                    current_input_ids = torch.cat([input_ids_theta, generated_so_far], dim=-1)
                    if stopping_criteria(current_input_ids, None):
                        break

                # ✅ 所有样本都完成时停止
                if finished.all():
                    break

                # 更新attention_mask（各自独立）
                attention_mask_theta = torch.cat([
                    attention_mask_theta,
                    torch.ones((batch_size, 1), device=attention_mask_theta.device)
                ], dim=-1)

                if use_different_prompts:
                    attention_mask_t = torch.cat([
                        attention_mask_t,
                        torch.ones((batch_size, 1), device=attention_mask_t.device)
                    ], dim=-1)
                else:
                    attention_mask_t = attention_mask_theta

                # ✅ Forward新token（使用past_key_values和各自的attention_mask）
                outputs_theta = self.model_theta(
                    input_ids=next_tokens.unsqueeze(-1),  # ✅ 只传入新token（相同）
                    attention_mask=attention_mask_theta,   # ✅ 使用各自的mask
                    past_key_values=past_key_values_theta,  # ✅ 使用cache
                    use_cache=True
                )
                past_key_values_theta = outputs_theta.past_key_values
                logits_theta = outputs_theta.logits[:, -1, :]

                if not self.same_model:
                    outputs_t = self.model_t(
                        input_ids=next_tokens.unsqueeze(-1),
                        attention_mask=attention_mask_t,     # ✅ 使用各自的mask
                        past_key_values=past_key_values_t,
                        use_cache=True
                    )
                    past_key_values_t = outputs_t.past_key_values
                    logits_t = outputs_t.logits[:, -1, :]
                else:
                    logits_t = logits_theta

        else:
            # ========================================
            # 原始实现（不使用KV cache，用于对比/调试，支持双prompt）
            # ========================================
            current_ids_theta = input_ids_theta
            current_attention_mask_theta = attention_mask_theta

            current_ids_t = input_ids_t
            current_attention_mask_t = attention_mask_t

            for step in range(max_new_tokens):
                # 获取 π_θ 的logits
                outputs_theta = self.model_theta(
                    input_ids=current_ids_theta,
                    attention_mask=current_attention_mask_theta,
                    use_cache=False
                )
                logits_theta = outputs_theta.logits[:, -1, :]

                # 获取 π_t 的logits
                if self.same_model:
                    logits_t = logits_theta
                else:
                    outputs_t = self.model_t(
                        input_ids=current_ids_t,
                        attention_mask=current_attention_mask_t,
                        use_cache=False
                    )
                    logits_t = outputs_t.logits[:, -1, :]

                # ✅ 对齐logits并计算概率（处理不同tokenizer）
                probs_theta, probs_t = self._align_logits(logits_theta, logits_t, temperature)

                # ✅ 强制第一个token使用π_t
                if step == 0 and self.force_target_for_first_token:
                    # 第一个token直接使用π_t，不进行混合
                    q_star = probs_t
                    # ✅ 修改：α=1 表示完全使用π_t（Teacher）
                    alpha = torch.ones(batch_size, device=probs_theta.device)
                else:
                    # 后续token正常计算alpha和q*
                    # ✨ 改进：先计算alpha，再计算q*，最后施加约束
                    alpha = self.alpha_computer.compute(probs_theta, probs_t)
                    q_star = self._compute_q_star(probs_theta, probs_t, alpha)

                    # ✨ 新增：对q*施加support约束（限制在π_t的support上）
                    # 这样可以避免约束影响alpha计算
                    q_star = self.alpha_computer.apply_constraint_to_q_star(q_star, probs_t)

                # 应用 top-p / top-k
                if top_p < 1.0 or top_k > 0:
                    q_star = self._apply_sampling_filters(q_star, top_p, top_k)

                # ✅ 采样前安全检查
                if torch.isnan(q_star).any() or torch.isinf(q_star).any() or (q_star < 0).any():
                    print(f"⚠️  Warning: Invalid q_star at step {step}, using q_t fallback")
                    # q_star = torch.ones_like(q_star) / q_star.size(-1)
                    q_star = probs_t

                # 从 q* 采样
                try:
                    next_tokens = torch.multinomial(q_star, num_samples=1).squeeze(-1)
                except RuntimeError as e:
                    print(f"⚠️  Sampling failed at step {step}: {e}")
                    print(f"   q_star stats: min={q_star.min():.6f}, max={q_star.max():.6f}, sum={q_star.sum(dim=-1)}")
                    # 回退到argmax
                    next_tokens = q_star.argmax(dim=-1)

                # ✅ 检查本步是否有样本生成EOS
                if eos_token_id is not None:
                    finished = finished | (next_tokens == eos_token_id)

                # ✅ 对已完成的样本，使用pad token（防止继续生成）
                if self.tokenizer_theta.pad_token_id is not None:
                    next_tokens = torch.where(finished,
                                             torch.tensor(self.tokenizer_theta.pad_token_id, device=next_tokens.device),
                                             next_tokens)

                # 计算诊断信息
                if return_diagnostics:
                    diag = self.diagnostic_computer.compute(probs_theta, probs_t, q_star, alpha)
                    all_alpha_values.append(alpha.cpu())
                    all_ess_ratios.append(diag["ess_ratio"].cpu())
                    all_diagnostics.append({k: v.cpu() for k, v in diag.items()})

                # ✨ 新增：收集 logits 和 q_star_probs
                if return_logits:
                    all_logits_theta.append(logits_theta.cpu())
                    all_logits_t.append(logits_t.cpu())
                if return_q_star_probs:
                    all_q_star_probs.append(q_star.cpu())

                # 更新序列（各自独立）
                all_generated_ids.append(next_tokens.unsqueeze(-1))
                current_ids_theta = torch.cat([current_ids_theta, next_tokens.unsqueeze(-1)], dim=-1)
                current_attention_mask_theta = torch.cat([
                    current_attention_mask_theta,
                    torch.ones((batch_size, 1), device=current_attention_mask_theta.device)
                ], dim=-1)

                current_ids_t = torch.cat([current_ids_t, next_tokens.unsqueeze(-1)], dim=-1)
                current_attention_mask_t = torch.cat([
                    current_attention_mask_t,
                    torch.ones((batch_size, 1), device=current_attention_mask_t.device)
                ], dim=-1)

                # ✅ 检查停止条件
                if stopping_criteria is not None:
                    if stopping_criteria(current_ids_theta, None):
                        break

                # ✅ 所有样本都完成时停止
                if finished.all():
                    break

        # 组装结果
        generated_ids = torch.cat(all_generated_ids, dim=-1)

        # ✨ 新增：根据 skip_decode 决定是否 decode
        if skip_decode:
            generated_texts = []  # 返回空列表
        else:
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 组装诊断信息
        diagnostics = {}
        if return_diagnostics and all_diagnostics:
            alpha_values = torch.stack(all_alpha_values, dim=1)  # [batch, seq_len]
            ess_ratios = torch.stack(all_ess_ratios, dim=1)

            # 聚合所有step的诊断信息
            for key in all_diagnostics[0].keys():
                values = torch.stack([d[key] for d in all_diagnostics], dim=1)
                diagnostics[key] = values
        else:
            alpha_values = torch.zeros((batch_size, 0))
            ess_ratios = torch.zeros((batch_size, 0))

        # ✨ 新增：组装 logits 和 q_star_probs
        logits = None
        if return_logits and all_logits_theta:
            logits = {
                "theta": torch.stack(all_logits_theta, dim=1),  # [batch, seq_len, vocab_size]
                "t": torch.stack(all_logits_t, dim=1)
            }

        q_star_probs = None
        if return_q_star_probs and all_q_star_probs:
            q_star_probs = torch.stack(all_q_star_probs, dim=1)  # [batch, seq_len, vocab_size]

        return SamplingOutput(
            generated_texts=generated_texts,
            generated_ids=generated_ids,
            alpha_values=alpha_values,
            ess_ratios=ess_ratios,
            diagnostics=diagnostics,
            logits=logits,
            q_star_probs=q_star_probs
        )

    def _align_logits(
        self,
        logits_theta: torch.Tensor,
        logits_t: torch.Tensor,
        temperature: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对齐两个模型的logits（当tokenizer不同时）

        策略：
        1. 如果tokenizer相同：直接使用
        2. 如果tokenizer不同：
           - 将θ的logits映射到统一空间（使用θ的vocabulary）
           - 将t的logits也映射到θ的vocabulary
           - 返回对齐后的概率分布

        Args:
            logits_theta: [batch, vocab_size_theta]
            logits_t: [batch, vocab_size_t]
            temperature: 温度

        Returns:
            (probs_theta, probs_t_aligned): 两个在相同vocabulary上的概率分布
        """
        if self.same_tokenizer:
            # Tokenizer相同，直接计算概率
            probs_theta = F.softmax(logits_theta / temperature, dim=-1)
            probs_t = F.softmax(logits_t / temperature, dim=-1)
            return probs_theta, probs_t

        # Tokenizer不同，需要对齐
        batch_size = logits_theta.shape[0]

        # θ的概率分布（保持不变）
        probs_theta = F.softmax(logits_theta / temperature, dim=-1)

        # 将t的概率映射到θ的vocabulary
        probs_t_aligned = torch.zeros_like(probs_theta)

        # 对于θ的每个token，找到t中的对应token并复制概率
        for id_theta, id_t in self.vocab_map_theta_to_t.items():
            probs_t_aligned[:, id_theta] = F.softmax(logits_t / temperature, dim=-1)[:, id_t]

        # 重新归一化（因为可能有unmapped tokens）
        probs_t_aligned = probs_t_aligned / (probs_t_aligned.sum(dim=-1, keepdim=True) + 1e-10)

        return probs_theta, probs_t_aligned

    def _map_token_id(self, token_id: int, from_model: str = "theta") -> int:
        """
        将token ID从一个vocabulary映射到另一个

        Args:
            token_id: 源token ID
            from_model: "theta" 或 "t"

        Returns:
            目标token ID
        """
        if self.same_tokenizer:
            return token_id

        if from_model == "theta":
            # θ -> t
            return self.vocab_map_theta_to_t.get(token_id, self.tokenizer_t.unk_token_id)
        else:
            # t -> θ
            return self.vocab_map_t_to_theta.get(token_id, self.tokenizer_theta.unk_token_id)

    def _compute_q_star(self, probs_theta: torch.Tensor, probs_t: torch.Tensor,
                        alpha: torch.Tensor) -> torch.Tensor:
        """
        计算q*分布（支持对special tokens直接使用π_t）

        ✅ 重要改动：α 现在表示 π_t (Teacher) 的权重！
        - α = 0 → 完全使用 π_θ (Base)
        - α = 1 → 完全使用 π_t (Teacher)
        - α > 0.5 → 更接近 Teacher（符合直觉）

        核心改进：
        对于从tokenizer config检测到的special tokens（如EOS、chat template等），
        这些token在Base model中可能没见过或处理不好。
        对这些token，我们直接使用π_t的概率，不进行几何平均。

        数学上等价于：对special token设置α=1（完全使用π_t）

        注意：probs_theta和probs_t应该已经通过_align_logits对齐到相同vocabulary
        """
        eps = 1e-10

        # ========================================
        # ✨ 增强1: 输入验证和NaN早期检测
        # ========================================
        if torch.isnan(probs_theta).any():
            nan_count = torch.isnan(probs_theta).sum().item()
            print(f"⚠️  Warning: probs_theta contains {nan_count} NaN values before q* computation")
            probs_theta = torch.where(torch.isnan(probs_theta),
                                       torch.tensor(eps, device=probs_theta.device),
                                       probs_theta)

        if torch.isnan(probs_t).any():
            nan_count = torch.isnan(probs_t).sum().item()
            print(f"⚠️  Warning: probs_t contains {nan_count} NaN values before q* computation")
            probs_t = torch.where(torch.isnan(probs_t),
                                   torch.tensor(eps, device=probs_t.device),
                                   probs_t)

        if torch.isnan(alpha).any():
            nan_positions = torch.where(torch.isnan(alpha))
            print(f"⚠️  Warning: alpha contains NaN at positions {nan_positions}")
            print(f"   Falling back to α=0.5 for NaN positions")
            alpha = torch.where(torch.isnan(alpha),
                                torch.tensor(0.5, device=alpha.device),
                                alpha)

        # ========================================
        # ✨ 增强2: 确保输入在有效范围内
        # ========================================
        probs_theta = torch.clamp(probs_theta, min=eps, max=1.0)
        probs_t = torch.clamp(probs_t, min=eps, max=1.0)
        alpha = torch.clamp(alpha, min=0.0, max=1.0)

        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)  # [batch, 1]

        # 计算log概率（添加eps避免log(0)，即使clamp之后也保险）
        log_probs_theta = torch.log(probs_theta + eps)
        log_probs_t = torch.log(probs_t + eps)

        # ========================================
        # ✨ 增强3: 验证log概率的有效性（只检查NaN，-inf是正常的）
        # ========================================
        # 注意：log(极小概率)会产生-inf，这是正常的，softmax会正确处理
        # 只有NaN才是真正的问题
        if torch.isnan(log_probs_theta).any():
            nan_count = torch.isnan(log_probs_theta).sum().item()
            print(f"❌ Error: log_probs_theta has {nan_count} NaN after log operation")
            # 强制修复：将NaN替换为一个安全的log值
            log_probs_theta = torch.where(torch.isnan(log_probs_theta),
                                           torch.tensor(-23.0, device=log_probs_theta.device),
                                           log_probs_theta)

        if torch.isnan(log_probs_t).any():
            nan_count = torch.isnan(log_probs_t).sum().item()
            print(f"❌ Error: log_probs_t has {nan_count} NaN after log operation")
            log_probs_t = torch.where(torch.isnan(log_probs_t),
                                       torch.tensor(-23.0, device=log_probs_t.device),
                                       log_probs_t)

        if self.force_target_for_special_tokens and self.special_token_mask is not None:
            # 使用special token mask
            # special_token_mask: [vocab_size] bool tensor
            # True表示special token，应该使用π_t

            # 将mask移到正确的device并扩展到batch
            device = probs_theta.device
            special_mask = self.special_token_mask.to(device)  # [vocab_size]
            special_mask = special_mask.unsqueeze(0)  # [1, vocab_size]

            # 计算两种情况的log q
            # ✅ 修改：α 现在是 π_t 的权重
            # 正常token: log q = (1-α) log π_θ + α log π_t
            # Special token: log q = log π_t （α=1）

            log_q_normal = (1 - alpha) * log_probs_theta + alpha * log_probs_t
            log_q_special = log_probs_t  # 直接使用π_t

            # 使用mask选择
            log_q = torch.where(special_mask, log_q_special, log_q_normal)
        else:
            # 原始方法：统一的alpha
            # ✅ 修改：α 现在是 π_t 的权重
            log_q = (1 - alpha) * log_probs_theta + alpha * log_probs_t

        # ========================================
        # ✨ 增强4: 检查log_q的有效性（只检查NaN）
        # ========================================
        # 注意：log_q可能包含-inf（表示极小概率），这是正常的
        # softmax会将-inf正确处理为0概率
        if torch.isnan(log_q).any():
            nan_count = torch.isnan(log_q).sum().item()
            print(f"❌ Error: log_q has {nan_count} NaN after geometric mean")
            print(f"   Alpha value: {alpha.squeeze() if alpha.dim() > 1 else alpha}")
            print(f"   Falling back to probs_t (Teacher model)")
            return probs_t.clone()

        # 归一化
        q_star = F.softmax(log_q, dim=-1)

        # ========================================
        # ✨ 增强5: 最终验证q*的有效性
        # ========================================
        if torch.isnan(q_star).any():
            nan_count = torch.isnan(q_star).sum().item()
            print(f"❌ CRITICAL: q_star has {nan_count} NaN after softmax!")
            print(f"   This should not happen if log_q was finite")
            print(f"   Falling back to q_t distribution")
            # q_star = torch.ones_like(q_star) / q_star.size(-1)
            q_star = probs_t.clone()

        return q_star

    def _apply_sampling_filters(self, probs: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
        """应用top-p和top-k过滤"""
        # Top-k
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, top_k_indices, top_k_probs)

        # Top-p (nucleus sampling)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # 移除累积概率超过top_p的token
            sorted_indices_to_remove = cumsum_probs > top_p
            # 保留第一个超过阈值的token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # 创建mask
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            probs = probs.masked_fill(indices_to_remove, 0.0)

        # 重新归一化
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        return probs

    # ============================================
    # Speculative Decoding 核心方法
    # ============================================

    @torch.no_grad()
    def _draft_k_tokens_simple(
        self,
        current_ids_theta: torch.Tensor,
        current_ids_t: torch.Tensor,
        attention_mask_theta: torch.Tensor,
        attention_mask_t: torch.Tensor,
        k: int,
        temperature: float = 1.0,
        use_different_prompts: bool = False
    ) -> tuple:
        """
        使用π_θ简单地生成k个draft tokens（不使用past_kv，保证正确性）

        Returns:
            draft_tokens: [batch, k]
            all_probs_theta: [batch, k, vocab]
            all_probs_t: [batch, k, vocab]
        """
        batch_size = current_ids_theta.shape[0]
        device = current_ids_theta.device

        draft_tokens = []
        all_probs_theta = []
        all_probs_t = []

        for i in range(k):
            # Forward π_θ
            outputs_theta = self.model_theta(
                input_ids=current_ids_theta,
                attention_mask=attention_mask_theta,
                use_cache=False  # 不使用cache，简单直接
            )
            logits_theta = outputs_theta.logits[:, -1, :]

            # Forward π_t
            if not self.same_model:
                outputs_t = self.model_t(
                    input_ids=current_ids_t,
                    attention_mask=attention_mask_t,
                    use_cache=False
                )
                logits_t = outputs_t.logits[:, -1, :]
            else:
                logits_t = logits_theta

            # 计算概率
            probs_theta = F.softmax(logits_theta / temperature, dim=-1)
            probs_t = F.softmax(logits_t / temperature, dim=-1)

            # 从π_θ采样draft token
            next_token = torch.multinomial(probs_theta, num_samples=1)  # [batch, 1]

            draft_tokens.append(next_token)
            all_probs_theta.append(probs_theta)
            all_probs_t.append(probs_t)

            # 更新序列
            current_ids_theta = torch.cat([current_ids_theta, next_token], dim=-1)
            if use_different_prompts:
                current_ids_t = torch.cat([current_ids_t, next_token], dim=-1)
            else:
                current_ids_t = current_ids_theta

            attention_mask_theta = torch.cat([
                attention_mask_theta,
                torch.ones((batch_size, 1), device=device)
            ], dim=-1)
            if use_different_prompts:
                attention_mask_t = torch.cat([
                    attention_mask_t,
                    torch.ones((batch_size, 1), device=device)
                ], dim=-1)
            else:
                attention_mask_t = attention_mask_theta

        draft_tokens = torch.cat(draft_tokens, dim=-1)  # [batch, k]
        all_probs_theta = torch.stack(all_probs_theta, dim=1)  # [batch, k, vocab]
        all_probs_t = torch.stack(all_probs_t, dim=1)  # [batch, k, vocab]

        return draft_tokens, all_probs_theta, all_probs_t

    def _verify_and_accept_batch(
        self,
        input_ids_theta: torch.Tensor,
        input_ids_t: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        attention_mask_theta: torch.Tensor,
        attention_mask_t: torch.Tensor,
        past_key_values_theta,
        past_key_values_t,
        temperature: float,
        use_different_prompts: bool
    ) -> tuple[list, list, list, any, any, torch.Tensor, torch.Tensor]:
        """
        批量验证draft tokens并决定接受/拒绝

        核心逻辑：
        1. 批量forward: 一次性处理k个draft tokens
        2. 对每个位置i:
           - 计算真实的q*(i)
           - 计算接受概率: min(1, q*[draft[i]] / q_draft[draft[i]])
           - 接受或拒绝（从修正分布重采样）
        3. 遇到第一个拒绝就停止

        Returns:
            accepted_tokens: 接受的token列表
            alpha_values: 对应的alpha值列表
            diagnostics: 诊断信息列表
            updated_past_kv_theta: 更新后的π_θ KV cache
            updated_past_kv_t: 更新后的π_t KV cache
            final_attention_mask_theta: 最终的attention mask (θ)
            final_attention_mask_t: 最终的attention mask (t)
        """
        batch_size, k = draft_tokens.shape
        device = draft_tokens.device

        # ========================================
        # Step 1: 批量forward（关键优化！）
        # ========================================
        # 将k个draft token拼接到输入后面
        extended_ids_theta = torch.cat([input_ids_theta, draft_tokens], dim=-1)
        extended_mask_theta = torch.cat([
            attention_mask_theta,
            torch.ones((batch_size, k), device=device)
        ], dim=-1)

        # Forward π_θ（批量处理k个token）
        outputs_theta = self.model_theta(
            input_ids=extended_ids_theta,
            attention_mask=extended_mask_theta,
            use_cache=True
        )
        logits_theta_all = outputs_theta.logits  # [batch, seq_len + k, vocab_size]

        # Forward π_t（批量处理k个token）
        if not self.same_model:
            extended_ids_t = torch.cat([input_ids_t, draft_tokens], dim=-1)
            extended_mask_t = torch.cat([
                attention_mask_t,
                torch.ones((batch_size, k), device=device)
            ], dim=-1)

            outputs_t = self.model_t(
                input_ids=extended_ids_t,
                attention_mask=extended_mask_t,
                use_cache=True
            )
            logits_t_all = outputs_t.logits
        else:
            logits_t_all = logits_theta_all
            extended_mask_t = extended_mask_theta

        # ========================================
        # Step 2: 逐个验证每个draft token
        # ========================================
        accepted_tokens = []
        alpha_values = []
        diagnostics_list = []

        # 跟踪当前位置的past_kv和attention_mask
        current_past_kv_theta = past_key_values_theta
        current_past_kv_t = past_key_values_t
        current_mask_theta = attention_mask_theta
        current_mask_t = attention_mask_t

        for i in range(k):
            # 获取第i个位置的logits（对应draft_token[i]之前的状态）
            # 如果extended_ids = [A, B, C, D, E], draft_tokens = [D, E]
            # logits[seq_len-1] 预测 D (draft[0])
            # logits[seq_len] 预测 E (draft[1])
            # 用负索引：draft[i] 的logits是 logits[-(k-i+1)]
            logits_theta_i = logits_theta_all[:, -(k - i + 1), :]
            logits_t_i = logits_t_all[:, -(k - i + 1), :]

            # 计算概率
            probs_theta_i, probs_t_i = self._align_logits(logits_theta_i, logits_t_i, temperature)

            # 计算α和q*
            alpha = self.alpha_computer.compute(probs_theta_i, probs_t_i)
            q_star = self._compute_q_star(probs_theta_i, probs_t_i, alpha)
            q_star = self.alpha_computer.apply_constraint_to_q_star(q_star, probs_t_i)

            # 获取draft token和对应的draft概率
            draft_token_i = draft_tokens[:, i]  # [batch]
            q_draft_i = draft_probs[:, i, :]    # [batch, vocab_size]

            # ========================================
            # 核心：Speculative Sampling验证
            # ========================================
            # 接受概率: accept_prob = min(1, q*(draft) / q_draft(draft))
            q_star_at_draft = q_star.gather(-1, draft_token_i.unsqueeze(-1)).squeeze(-1)  # [batch]
            q_draft_at_draft = q_draft_i.gather(-1, draft_token_i.unsqueeze(-1)).squeeze(-1)  # [batch]

            accept_prob = torch.clamp(
                q_star_at_draft / (q_draft_at_draft + 1e-10),
                max=1.0
            )

            # 采样决定是否接受
            accept = torch.rand_like(accept_prob) < accept_prob

            if accept.all():
                # 所有样本都接受这个draft token
                accepted_tokens.append(draft_token_i)
                alpha_values.append(alpha)

                # 计算诊断信息
                diag = self.diagnostic_computer.compute(probs_theta_i, probs_t_i, q_star, alpha)
                diagnostics_list.append(diag)

                # 更新past_kv和attention_mask（这里需要逐步更新）
                # 由于我们已经有了完整的forward结果，可以截取对应的past_kv
                # 简化：直接使用最终的past_kv（因为我们批量forward了）
                current_mask_theta = torch.cat([
                    current_mask_theta,
                    torch.ones((batch_size, 1), device=device)
                ], dim=-1)
                if use_different_prompts:
                    current_mask_t = torch.cat([
                        current_mask_t,
                        torch.ones((batch_size, 1), device=device)
                    ], dim=-1)
                else:
                    current_mask_t = current_mask_theta

            else:
                # 至少有一个样本拒绝 → 需要重采样
                # 修正分布：q'(x) = max(0, q*(x) - q_draft(x)) / Z
                corrected_probs = torch.clamp(q_star - q_draft_i, min=0.0)
                corrected_probs = corrected_probs / (corrected_probs.sum(dim=-1, keepdim=True) + 1e-10)

                # 重采样
                resampled_token = torch.multinomial(corrected_probs, num_samples=1).squeeze(-1)

                # 对接受的样本用draft，对拒绝的样本用resampled
                final_token = torch.where(accept, draft_token_i, resampled_token)

                accepted_tokens.append(final_token)
                alpha_values.append(alpha)

                diag = self.diagnostic_computer.compute(probs_theta_i, probs_t_i, q_star, alpha)
                diagnostics_list.append(diag)

                # 更新mask
                current_mask_theta = torch.cat([
                    current_mask_theta,
                    torch.ones((batch_size, 1), device=device)
                ], dim=-1)
                if use_different_prompts:
                    current_mask_t = torch.cat([
                        current_mask_t,
                        torch.ones((batch_size, 1), device=device)
                    ], dim=-1)
                else:
                    current_mask_t = current_mask_theta

                # ❌ 拒绝：停止验证后续draft tokens
                break

        # ========================================
        # Step 3: 更新past_kv（incrementally forward accepted tokens）
        # ========================================
        # 从past_key_values开始，逐个forward accepted tokens
        if len(accepted_tokens) > 0:
            current_past_kv_theta = past_key_values_theta
            current_past_kv_t = past_key_values_t

            for token in accepted_tokens:
                # Forward π_θ one token at a time
                outputs_theta = self.model_theta(
                    input_ids=token.unsqueeze(-1),
                    attention_mask=None,  # attention_mask已经在current_mask_theta中累积了
                    past_key_values=current_past_kv_theta,
                    use_cache=True
                )
                current_past_kv_theta = outputs_theta.past_key_values

                if not self.same_model:
                    outputs_t = self.model_t(
                        input_ids=token.unsqueeze(-1),
                        attention_mask=None,
                        past_key_values=current_past_kv_t,
                        use_cache=True
                    )
                    current_past_kv_t = outputs_t.past_key_values
                else:
                    current_past_kv_t = current_past_kv_theta

            final_past_kv_theta = current_past_kv_theta
            final_past_kv_t = current_past_kv_t
        else:
            final_past_kv_theta = past_key_values_theta
            final_past_kv_t = past_key_values_t

        return (
            accepted_tokens,
            alpha_values,
            diagnostics_list,
            final_past_kv_theta,
            final_past_kv_t,
            current_mask_theta,
            current_mask_t
        )

    @torch.no_grad()
    def generate_speculative(
        self,
        prompts: List[str],
        prompts_t: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        k: int = 4,  # 每次预测k个draft tokens
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        return_diagnostics: bool = True,
        skip_decode: bool = False,
        return_logits: bool = False,
        return_q_star_probs: bool = False,
        stopping_criteria: Optional[any] = None,
        **kwargs
    ) -> SamplingOutput:
        """
        使用Speculative Decoding加速的生成方法

        核心思想：
        1. 用π_θ快速生成k个draft tokens（autoregressive）
        2. 用π_θ和π_t批量验证这k个tokens（并行forward）
        3. 接受/拒绝每个token，严格保证最终分布 = q*

        Args:
            prompts: π_θ的输入prompts
            prompts_t: π_t的输入prompts（可选，支持双prompt）
            max_new_tokens: 最大生成token数
            k: 每次预测的draft token数量（推荐4-8）
            temperature: 采样温度
            top_p: nucleus sampling
            top_k: top-k sampling
            return_diagnostics: 是否返回诊断信息
            skip_decode: 是否跳过decode
            return_logits: 是否返回logits
            return_q_star_probs: 是否返回q*概率
            stopping_criteria: 停止条件

        Returns:
            SamplingOutput对象（与generate方法相同）

        Note:
            - 数学上严格保证：采样分布 = q*（通过speculative sampling验证）
            - 预期加速：1.5-3x（取决于接受率和k值）
            - 接受率通常在40-70%（π_θ和π_t越接近，接受率越高）
        """
        batch_size = len(prompts)

        # ========================================
        # 准备输入
        # ========================================
        use_different_prompts = (prompts_t is not None) and (prompts_t != prompts)

        if use_different_prompts:
            if len(prompts_t) != batch_size:
                raise ValueError(f"prompts_t长度({len(prompts_t)})必须与prompts长度({batch_size})相同")

        # Tokenize
        inputs_theta = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        input_ids_theta = inputs_theta["input_ids"].to(self.model_theta.device)
        attention_mask_theta = inputs_theta["attention_mask"].to(self.model_theta.device)

        if use_different_prompts:
            if self.same_model:
                inputs_t = self.tokenizer(prompts_t, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            else:
                inputs_t = self.tokenizer_t(prompts_t, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            input_ids_t = inputs_t["input_ids"].to(self.model_t.device)
            attention_mask_t = inputs_t["attention_mask"].to(self.model_t.device)
        else:
            input_ids_t = input_ids_theta
            attention_mask_t = attention_mask_theta

        # ========================================
        # 准备存储
        # ========================================
        all_generated_ids = []
        all_alpha_values = []
        all_ess_ratios = []
        all_diagnostics = []

        all_logits_theta = [] if return_logits else None
        all_logits_t = [] if return_logits else None
        all_q_star_probs = [] if return_q_star_probs else None

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.model_theta.device)
        eos_token_id = self.tokenizer_theta.eos_token_id

        # ========================================
        # Prefill阶段：初始化KV cache
        # ========================================
        outputs_theta = self.model_theta(
            input_ids=input_ids_theta,
            attention_mask=attention_mask_theta,
            use_cache=True
        )
        past_key_values_theta = outputs_theta.past_key_values

        if not self.same_model:
            outputs_t = self.model_t(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                use_cache=True
            )
            past_key_values_t = outputs_t.past_key_values
        else:
            past_key_values_t = None

        # ========================================
        # Decode阶段：Speculative Decoding循环
        # ========================================
        total_generated = 0
        total_draft = 0
        total_accepted = 0

        while total_generated < max_new_tokens:
            # 计算这一轮要生成多少个draft tokens
            remaining = max_new_tokens - total_generated
            k_this_round = min(k, remaining)

            # ========================================
            # Step 1: Draft阶段 - 用π_θ快速生成k个候选
            # ========================================
            draft_tokens, draft_probs, _ = self._draft_tokens_with_theta(
                input_ids=torch.cat([input_ids_theta] + all_generated_ids, dim=-1) if all_generated_ids else input_ids_theta,
                attention_mask=attention_mask_theta,
                past_key_values_theta=past_key_values_theta,
                k=k_this_round,
                temperature=temperature
            )

            total_draft += k_this_round

            # ========================================
            # Step 2: Verify阶段 - 批量验证并接受/拒绝
            # ========================================
            (
                accepted_tokens,
                alpha_values,
                diagnostics_list,
                past_key_values_theta,
                past_key_values_t,
                attention_mask_theta,
                attention_mask_t
            ) = self._verify_and_accept_batch(
                input_ids_theta=torch.cat([input_ids_theta] + all_generated_ids, dim=-1) if all_generated_ids else input_ids_theta,
                input_ids_t=torch.cat([input_ids_t] + all_generated_ids, dim=-1) if all_generated_ids else input_ids_t,
                draft_tokens=draft_tokens,
                draft_probs=draft_probs,
                attention_mask_theta=attention_mask_theta,
                attention_mask_t=attention_mask_t,
                past_key_values_theta=past_key_values_theta,
                past_key_values_t=past_key_values_t,
                temperature=temperature,
                use_different_prompts=use_different_prompts
            )

            num_accepted = len(accepted_tokens)
            total_accepted += num_accepted

            # ========================================
            # Step 3: 保存结果
            # ========================================
            for i in range(num_accepted):
                token = accepted_tokens[i]

                # 检查EOS
                if eos_token_id is not None:
                    finished = finished | (token == eos_token_id)

                # 对已完成的样本使用pad token
                if self.tokenizer_theta.pad_token_id is not None:
                    token = torch.where(finished,
                                       torch.tensor(self.tokenizer_theta.pad_token_id, device=token.device),
                                       token)

                all_generated_ids.append(token.unsqueeze(-1))

                if return_diagnostics:
                    all_alpha_values.append(alpha_values[i].cpu())
                    all_ess_ratios.append(diagnostics_list[i]["ess_ratio"].cpu())
                    all_diagnostics.append({k: v.cpu() for k, v in diagnostics_list[i].items()})

            total_generated += num_accepted

            # 检查停止条件
            if stopping_criteria is not None:
                generated_so_far = torch.cat(all_generated_ids, dim=-1)
                current_input_ids = torch.cat([input_ids_theta, generated_so_far], dim=-1)
                if stopping_criteria(current_input_ids, None):
                    break

            if finished.all():
                break

        # ========================================
        # 组装结果
        # ========================================
        generated_ids = torch.cat(all_generated_ids, dim=-1) if all_generated_ids else torch.empty((batch_size, 0), dtype=torch.long)

        if skip_decode:
            generated_texts = []
        else:
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        diagnostics = {}
        if return_diagnostics and all_diagnostics:
            alpha_values = torch.stack(all_alpha_values, dim=1)
            ess_ratios = torch.stack(all_ess_ratios, dim=1)

            for key in all_diagnostics[0].keys():
                values = torch.stack([d[key] for d in all_diagnostics], dim=1)
                diagnostics[key] = values

            # 添加Speculative Decoding统计
            diagnostics['total_draft_tokens'] = total_draft
            diagnostics['total_accepted_tokens'] = total_accepted
            diagnostics['acceptance_rate'] = total_accepted / total_draft if total_draft > 0 else 0.0
            diagnostics['average_accepted_per_round'] = total_accepted / (total_generated // k + 1) if total_generated > 0 else 0.0
        else:
            alpha_values = torch.zeros((batch_size, 0))
            ess_ratios = torch.zeros((batch_size, 0))

        logits = None
        q_star_probs = None

        # 打印统计信息
        if return_diagnostics:
            accept_rate = diagnostics['acceptance_rate']
            print(f"\n📊 Speculative Decoding Stats:")
            print(f"   - Draft tokens: {total_draft}")
            print(f"   - Accepted tokens: {total_accepted}")
            print(f"   - Acceptance rate: {accept_rate:.2%}")
            print(f"   - Estimated speedup: {1 + (k-1) * accept_rate:.2f}x")

        return SamplingOutput(
            generated_texts=generated_texts,
            generated_ids=generated_ids,
            alpha_values=alpha_values,
            ess_ratios=ess_ratios,
            diagnostics=diagnostics,
            logits=logits,
            q_star_probs=q_star_probs
        )


# ============================================
# 便捷函数
# ============================================

def create_optimal_sampling_model(
    model_theta: str,
    model_t: Optional[str] = None,
    alpha_method: str = "kl_symmetry",
    constraint_to_target: bool = True,
    target_top_k: int = 50,
    target_top_p: float = 1.0,
    force_target_for_special_tokens: bool = True,
    force_target_for_first_token: bool = True,
    **kwargs
) -> OptimalSamplingModel:
    """
    便捷的模型创建函数（仅支持transformers backend）

    Args:
        model_theta: π_θ模型路径 (Base model, 如Llama-2-7b)
        model_t: π_t模型路径（Teacher/Instruct model, 如Llama-2-7b-chat）
        alpha_method: Alpha计算方法（α表示Teacher权重）
        constraint_to_target: ✨ 是否限制在π_t的support上（推荐True）
        target_top_k: ✨ π_t的top-k限制
        target_top_p: ✨ π_t的top-p限制
        force_target_for_special_tokens: ✨ 对special tokens直接使用π_t（推荐True）
        force_target_for_first_token: ✨ 强制第一个token使用π_t（推荐True）
        **kwargs: 传递给transformers的额外参数（如load_in_4bit, attn_implementation等）

    Note:
        α的含义: α表示**Teacher模型 (π_t)** 的权重
        - α = 0 → 完全使用Base模型
        - α = 1 → 完全使用Teacher模型
        - α > 0.5 → 更接近Teacher（通常期望）
        - 混合公式: q*(x) = π_θ(x)^(1-α) × π_t(x)^α

    Examples:
        >>> # 基础使用（同一个模型）
        >>> model = create_optimal_sampling_model(
        ...     model_theta="meta-llama/Llama-2-7b-hf",
        ...     alpha_method="fixed",
        ...     fixed_alpha=0.5  # α=0.5表示均匀混合
        ... )

        >>> # 使用不同的模型
        >>> model = create_optimal_sampling_model(
        ...     model_theta="meta-llama/Llama-2-7b-hf",        # Base
        ...     model_t="meta-llama/Llama-2-7b-chat-hf",       # Teacher/Instruct
        ...     alpha_method="kl_symmetry"
        ... )
        >>> # 期望: α > 0.5（更接近高质量的Teacher模型）

        >>> # ✨ 推荐：使用support约束（提升数值稳定性）
        >>> model = create_optimal_sampling_model(
        ...     model_theta="Qwen/Qwen3-8B-Base",
        ...     model_t="Qwen/Qwen3-8B",
        ...     alpha_method="kl_symmetry",
        ...     constraint_to_target=True,    # 限制在π_t的support上
        ...     target_top_k=100,              # 只在π_t的top-100 token上混合
        ...     target_top_p=0.95              # 或使用top-p
        ... )

        >>> # ✨✨ 最佳实践：同时使用support约束和special token处理
        >>> model = create_optimal_sampling_model(
        ...     model_theta="Qwen/Qwen3-8B-Base",
        ...     model_t="Qwen/Qwen3-8B",
        ...     alpha_method="kl_symmetry",
        ...     constraint_to_target=True,              # Support约束
        ...     target_top_k=100,
        ...     force_target_for_special_tokens=True,   # 对EOS等特殊token使用π_t
        ...     force_target_for_first_token=True       # 第一个token使用π_t
        ... )

        >>> # 🔥 大模型加速：使用quantization + Flash Attention 2
        >>> model = create_optimal_sampling_model(
        ...     model_theta="meta-llama/Llama-2-70b-hf",
        ...     model_t="meta-llama/Llama-2-70b-chat-hf",
        ...     alpha_method="kl_symmetry",
        ...     # Flash Attention 2 (2-4x加速)
        ...     attn_implementation="flash_attention_2",
        ...     # INT4量化 (75%显存减少)
        ...     load_in_4bit=True,
        ...     bnb_4bit_compute_dtype=torch.bfloat16,
        ...     bnb_4bit_use_double_quant=True,
        ...     # 多GPU
        ...     device_map="auto",
        ...     max_memory={i: "38GB" for i in range(4)}
        ... )
    """
    return OptimalSamplingModel(
        model_theta_path=model_theta,
        model_t_path=model_t,
        alpha_method=alpha_method,
        constraint_to_target=constraint_to_target,
        target_top_k=target_top_k,
        target_top_p=target_top_p,
        force_target_for_special_tokens=force_target_for_special_tokens,
        force_target_for_first_token=force_target_for_first_token,
        **kwargs
    )


def _simple_template_fallback(messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """简单的模板fallback（当没有chat template或jinja2不可用时）"""
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            prompt += f"Question: {content}\n\n"
        elif role == "assistant":
            prompt += f"Answer: {content}\n\n"
        elif role == "system":
            prompt += f"{content}\n\n"
    if add_generation_prompt:
        prompt += "Answer: "
    return prompt


# ============================================
# Chat Template 常量
# ============================================

NATURAL_LANGUAGE_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ 'Question: ' + message['content'] + '\\n\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'Answer: ' + message['content'] + '\\n\\n' }}"
    "{% elif message['role'] == 'system' %}"
    "{{ message['content'] + '\\n\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'Answer: ' }}"
    "{% endif %}"
)
"""自然语言模板：使用简单的 Question/Answer 格式"""


def create_dual_prompts(
    messages_list: List[List[Dict[str, str]]],
    tokenizer_theta,
    tokenizer_t,
    force_nlt_in_theta: bool = True,
    base_template: Optional[str] = NATURAL_LANGUAGE_TEMPLATE,
    add_generation_prompt: bool = True,
    add_think_token: bool = False,
) -> tuple[List[str], List[str]]:
    """
    为Base和Teacher模型创建不同的prompts

    Args:
        messages_list: 对话消息列表
        tokenizer_theta: Base模型的tokenizer
        tokenizer_t: Teacher模型的tokenizer
        force_nlt_in_theta: 是否强制Base模型使用自然语言模板（默认True）
        base_template: Base模型使用的自定义模板（默认使用自然语言模板）
            仅在 force_nlt_in_theta=True 时生效
        add_generation_prompt: 是否添加生成提示

    Returns:
        (prompts_theta, prompts_t): 两个模型各自的prompts

    Examples:
        >>> messages_list = [
        ...     [{"role": "user", "content": "What is machine learning?"}]
        ... ]
        >>> prompts_theta, prompts_t = create_dual_prompts(
        ...     messages_list,
        ...     model.tokenizer_theta,
        ...     model.tokenizer_t
        ... )
        >>> # prompts_theta: "Question: What is machine learning?\\n\\nAnswer: "
        >>> # prompts_t: "<|im_start|>user\\nWhat is machine learning?<|im_end|>\\n<|im_start|>assistant\\n"
    """
    prompts_theta = []
    prompts_t = []

    for messages in messages_list:
        # ========================================
        # 处理 Base 模型的 prompt
        # ========================================
        if force_nlt_in_theta:
            # 强制使用自然语言模板
            if base_template is not None:
                try:
                    from jinja2 import Template
                    template = Template(base_template)
                    prompt_theta = template.render(
                        messages=messages,
                        add_generation_prompt=add_generation_prompt
                    )
                except ImportError:
                    # Fallback: 如果没有 jinja2
                    prompt_theta = _simple_template_fallback(messages, add_generation_prompt)
            else:
                prompt_theta = _simple_template_fallback(messages, add_generation_prompt)
        else:
            # 使用 tokenizer 的 chat template（如果有）
            if hasattr(tokenizer_theta, 'chat_template') and tokenizer_theta.chat_template is not None:
                prompt_theta = tokenizer_theta.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt
                )
            else:
                # Fallback: tokenizer 没有 chat template
                prompt_theta = _simple_template_fallback(messages, add_generation_prompt)

        # ========================================
        # 处理 Teacher 模型的 prompt
        # ========================================
        if hasattr(tokenizer_t, 'chat_template') and tokenizer_t.chat_template is not None:
            prompt_t = tokenizer_t.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        else:
            # Fallback: tokenizer 没有 chat template
            prompt_t = _simple_template_fallback(messages, add_generation_prompt)

        if add_think_token:
            prompt_theta += "<think>"
            prompt_t += "<think>"

        prompts_theta.append(prompt_theta)
        prompts_t.append(prompt_t)

    return prompts_theta, prompts_t


if __name__ == "__main__":
    # 测试示例：展示双 prompt 功能（Base 和 Teacher 使用不同的 chat template）
    print("Testing OptimalSamplingModel with Dual Prompts...")
    print("=" * 80)

    # 注意: 需要替换为实际的模型路径
    model = create_optimal_sampling_model(
        model_theta="Qwen/Qwen3-4B-Base",           # Base 模型
        model_t="Qwen/Qwen3-14B",      # Teacher/Instruct 模型
        alpha_method="kl_symmetry",
        constraint_to_target=True,
        target_top_k=32,
        force_target_for_first_token=True,
        force_target_for_special_tokens=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    messages_list = [
        [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        [
            {"role": "user", "content": "What is 2+2? Explain your reasoning."}
        ],
        [
            {"role": "user",
             "content": "The operation $\\otimes$ is defined for all nonzero numbers by $a \\otimes b = \\frac{a^{2}}{b}$. Determine $[(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)]$."}
        ],
    ]

    print("\n" + "=" * 80)
    print("✨ Using Dual Prompt Mode")
    print("=" * 80)
    print("- Base Model (π_θ): Using Natural Language Template 'Question: ... Answer: ...'")
    print("- Teacher Model (π_t): Using Std Chat Template (e.g., <|im_start|>...)")
    print()

    # ✨ 创建双 prompt（Base 用自然语言，Teacher 用标准 template）
    prompts_theta, prompts_t = create_dual_prompts(
        messages_list,
        model.tokenizer_theta,
        model.tokenizer_t,
        base_template=NATURAL_LANGUAGE_TEMPLATE  # 自然语言模板
    )

    # 逐个处理每个问题
    for i in range(len(messages_list)):
        print(f"\n\n{'=' * 80}")
        print(f"Question: {i+1}/{len(messages_list)}")
        print("=" * 80)
        # print(f"问题: {messages_list[i][0]['content'][:80]}...")
        print("[PRMPT T] ", prompts_t[i])
        print("[PRMPT θ] ", prompts_theta[i])

        # ✨ 使用双 prompt 生成（可选：返回 logits）
        outputs = model.generate(
            prompts=[prompts_theta[i]],    # Base 看自然语言
            prompts_t=[prompts_t[i]],      # Teacher 看 chat template
            max_new_tokens=4096,
            return_logits=False,            # 可设为 True 返回 logits
            return_q_star_probs=False,      # 可设为 True 返回 q*
            # temperature=0.7,
            # top_p=0.9,
        )

        print("\n" + "-" * 80)
        print("Response:")
        print("-" * 80)
        print(outputs.generated_texts[0])

        print("\n" + "-" * 80)
        print("Stats Info:")
        print("-" * 80)
        print(f"  Generated tokens: {outputs.generated_ids.shape[1]}")
        print(f"  Alpha mean: {outputs.alpha_values.mean():.3f}")
        print(f"  Alpha std: {outputs.alpha_values.std():.3f}")
        print(f"  Alpha min/max: {outputs.alpha_values.min():.3f} / {outputs.alpha_values.max():.3f}")
        print(f"  ESS ratio mean: {outputs.ess_ratios.mean():.3f}")
        print(f"  ESS ratio std: {outputs.ess_ratios.std():.3f}")

        # if outputs.diagnostics:
        #     kl_theta = outputs.diagnostics.get("kl_theta")
        #     kl_t = outputs.diagnostics.get("kl_t")
        #     if kl_theta is not None and kl_t is not None:
        #         print(f"  KL(q||π_θ) mean: {kl_theta.mean():.3f}")
        #         print(f"  KL(q||π_t) mean: {kl_t.mean():.3f}")

    print("\n\n" + "=" * 80)
    print("✅ Test Success!")
    print("=" * 80)
    # print("\n提示:")
    # print("  - Alpha > 0.5 表示更接近 Teacher 模型")
    # print("  - ESS ratio ≈ 1.0 表示两个模型的采样效率平衡")
    # print("  - 可以设置 return_logits=True 来获取每步的 logits")
    # print("  - 可以设置 skip_decode=True 在外部进行 decode")
