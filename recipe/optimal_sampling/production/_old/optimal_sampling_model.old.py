"""
OptimalSamplingModel: 最优采样分布的实现

支持特性:
- 两种backend: transformers 和 VLLM
- 多种alpha计算方法: fixed, kl_symmetry, entropy
- q* 分布计算和采样
- 完整的诊断信息 (ESS, KL散度等)

使用示例:
    model = OptimalSamplingModel(
        model_theta_path="meta-llama/Llama-2-7b-hf",
        model_t_path="meta-llama/Llama-2-7b-chat-hf",
        backend="transformers",
        alpha_method="kl_symmetry"
    )

    outputs = model.generate(
        prompts=["Hello, how are you?"],
        max_new_tokens=100,
        temperature=1.0
    )
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
    generated_texts: List[str]
    generated_ids: torch.Tensor
    alpha_values: torch.Tensor  # [batch, seq_len]
    ess_ratios: torch.Tensor    # [batch, seq_len]
    diagnostics: Dict[str, any]


class AlphaComputer:
    """Alpha参数计算器（增强数值稳定性版本）"""

    def __init__(self, method: str = "kl_symmetry", fixed_alpha: float = 0.5,
                 tol: float = 1e-6, max_iter: int = 20,
                 constraint_to_target: bool = False,
                 target_top_k: int = -1,
                 target_top_p: float = 1.0):
        """
        Args:
            method: alpha计算方法 ["fixed", "kl_symmetry", "reverse_kl_symmetry", "ess_balance", "entropy"]
            fixed_alpha: 当method="fixed"时使用的固定值
            tol: KL对称/ESS平衡求解的容差
            max_iter: 最大迭代次数
            constraint_to_target: 是否限制在π_t的support上（推荐True）
            target_top_k: π_t的top-k限制（-1表示不限制）
            target_top_p: π_t的top-p限制（1.0表示不限制）
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
        计算alpha（带support约束）

        Args:
            probs_theta: [batch, vocab_size] 当前策略概率
            probs_t: [batch, vocab_size] 目标策略概率

        Returns:
            alpha: [batch] alpha值
        """
        # ✅ 新增：Support约束（限制在π_t的support上）
        if self.constraint_to_target and (self.target_top_k > 0 or self.target_top_p < 1.0):
            probs_theta, probs_t = self._apply_support_constraint(probs_theta, probs_t)

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

        注意: 这是ESS平衡条件的一阶近似
        """
        batch_size = probs_theta.shape[0]
        device = probs_theta.device

        # ✅ 数值稳定性保护1：Clamp输入概率
        probs_theta = torch.clamp(probs_theta, min=self.eps, max=1.0)
        probs_t = torch.clamp(probs_t, min=self.eps, max=1.0)

        # ✅ 数值稳定性保护2：检测极端情况
        # 如果两个分布几乎相同，直接返回0.5
        # max_ratio = (probs_theta / probs_t).max(dim=-1)[0]
        # min_ratio = (probs_theta / probs_t).min(dim=-1)[0]
        # nearly_identical = (max_ratio < 1.1) & (min_ratio > 0.9)
        #
        # if nearly_identical.any():
        #     alpha_result = torch.full((batch_size,), 0.5, device=device, dtype=probs_theta.dtype)
        #     if nearly_identical.all():
        #         return alpha_result

        # 二分搜索
        alpha_low = torch.full((batch_size,), 0.0, device=device)
        alpha_high = torch.full((batch_size,), 1.0, device=device)

        # ✅ 预计算log概率（避免重复计算）
        log_probs_theta = torch.log(probs_theta)
        log_probs_t = torch.log(probs_t)
        log_ratio = log_probs_t - log_probs_theta  # log(π_t/π_θ)

        for iteration in range(self.max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2

            # ✅ 在log-space计算 q_alpha
            # log q = α log π_θ + (1-α) log π_t
            alpha_expanded = alpha_mid.unsqueeze(-1)
            log_q_unnormalized = alpha_expanded * log_probs_theta + (1 - alpha_expanded) * log_probs_t

            # ✅ 使用logsumexp归一化（数值稳定）
            log_q = log_q_unnormalized - torch.logsumexp(log_q_unnormalized, dim=-1, keepdim=True)
            q_alpha = torch.exp(log_q)

            # 计算 Δ(α) = E_q[log(π_t/π_θ)]
            delta = (q_alpha * log_ratio).sum(dim=-1)

            # ✅ 数值稳定性保护3：检查delta是否有效
            invalid = torch.isnan(delta) | torch.isinf(delta)
            if invalid.any():
                # 对无效的样本，回退到0.5
                alpha_mid = torch.where(invalid, torch.tensor(0.5, device=device), alpha_mid)
                delta = torch.where(invalid, torch.tensor(0.0, device=device), delta)

            # 更新区间
            mask = delta > 0
            alpha_low = torch.where(mask, alpha_mid, alpha_low)
            alpha_high = torch.where(mask, alpha_high, alpha_mid)

            # 检查收敛
            if (alpha_high - alpha_low).max() < self.tol:
                break

        alpha_result = (alpha_low + alpha_high) / 2

        # ✅ 数值稳定性保护4：最终clamp
        alpha_result = torch.clamp(alpha_result, min=0.0, max=1.0)

        # 处理之前检测到的nearly_identical情况
        # if nearly_identical.any():
        #     alpha_result = torch.where(nearly_identical, torch.tensor(0.5, device=device), alpha_result)

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
            alpha_expanded = alpha_mid.unsqueeze(-1)
            log_q_unnormalized = alpha_expanded * log_probs_theta + (1 - alpha_expanded) * log_probs_t

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

            # 更新区间
            # 当 delta > 0 时，说明 E_{π_θ}[log q] > E_{π_t}[log q]
            # 意味着q在π_θ认为可能的区域给予了更高的概率
            # 需要增大π_t的权重，即减小α
            mask = delta > 0
            alpha_high = torch.where(mask, alpha_mid, alpha_high)
            alpha_low = torch.where(mask, alpha_low, alpha_mid)

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
        """
        batch_size = probs_theta.shape[0]
        device = probs_theta.device

        # ========================================
        # 第1步：检查边界条件
        # ========================================
        # 计算 α=0 和 α=1 时的 ESS 和 sum_sq
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
        """熵公式快速近似"""
        h_theta = -(probs_theta * torch.log(probs_theta + self.eps)).sum(dim=-1)
        h_t = -(probs_t * torch.log(probs_t + self.eps)).sum(dim=-1)
        alpha = h_theta / (h_theta + h_t + self.eps)
        return torch.clamp(alpha, 0.0, 1.0)

    def _geometric_mean(self, p1: torch.Tensor, p2: torch.Tensor,
                       alpha: torch.Tensor) -> torch.Tensor:
        """计算几何平均"""
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        log_q = alpha * torch.log(p1 + self.eps) + (1 - alpha) * torch.log(p2 + self.eps)
        return F.softmax(log_q, dim=-1)


class DiagnosticComputer:
    """诊断信息计算器"""

    def __init__(self):
        self.eps = 1e-10

    def compute(self, probs_theta: torch.Tensor, probs_t: torch.Tensor,
                q_star: torch.Tensor, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算诊断信息

        Returns:
            dict with keys: ess_theta, ess_t, ess_ratio, kl_theta, kl_t, kl_diff
        """
        # ESS (Effective Sample Size)
        ess_theta = 1.0 / ((probs_theta ** 2) / (q_star + self.eps)).sum(dim=-1)
        ess_t = 1.0 / ((probs_t ** 2) / (q_star + self.eps)).sum(dim=-1)
        ess_ratio = ess_theta / (ess_t + self.eps)

        # KL散度
        kl_theta = (q_star * torch.log((q_star + self.eps) / (probs_theta + self.eps))).sum(dim=-1)
        kl_t = (q_star * torch.log((q_star + self.eps) / (probs_t + self.eps))).sum(dim=-1)

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
        backend: Literal["transformers", "vllm"] = "transformers",
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
            backend: 使用的backend ["transformers", "vllm"]
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
            **kwargs: 传递给backend的额外参数
        """
        self.backend = backend
        self.device = device
        self.dtype = dtype

        # 存储special token处理参数
        self.force_target_for_special_tokens = force_target_for_special_tokens
        self.force_target_for_first_token = force_target_for_first_token

        # 初始化模型
        if backend == "transformers":
            self._init_transformers(model_theta_path, model_t_path, **kwargs)
        elif backend == "vllm":
            self._init_vllm(model_theta_path, model_t_path, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

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
        print(f"  Backend: {backend}")
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

    def _init_vllm(self, model_theta_path: str, model_t_path: Optional[str], **kwargs):
        """初始化VLLM backend"""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Please install vllm: pip install vllm")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_theta_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # VLLM参数
        vllm_kwargs = {
            "dtype": str(self.dtype).split(".")[-1],  # torch.float16 -> "float16"
            "trust_remote_code": True,
            **kwargs
        }

        # 加载 π_θ
        print(f"Loading π_θ with VLLM from {model_theta_path}...")
        self.model_theta = LLM(model=model_theta_path, **vllm_kwargs)

        # 加载 π_t
        if model_t_path is None or model_t_path == model_theta_path:
            print(f"Using π_θ as π_t (same model)")
            self.model_t = self.model_theta
            self.same_model = True
        else:
            print(f"Loading π_t with VLLM from {model_t_path}...")
            self.model_t = LLM(model=model_t_path, **vllm_kwargs)
            self.same_model = False

        self.SamplingParams = SamplingParams

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
        """
        if self.backend == "transformers":
            return self._generate_transformers(
                prompts, prompts_t, max_new_tokens, temperature, top_p, top_k,
                return_diagnostics, **kwargs
            )
        elif self.backend == "vllm":
            return self._generate_vllm(
                prompts, prompts_t, max_new_tokens, temperature, top_p, top_k,
                return_diagnostics, **kwargs
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
        use_kv_cache: bool = True,  # ✅ 新增参数
        stopping_criteria: Optional[any] = None,  # ✅ 支持自定义停止条件
        **kwargs
    ) -> SamplingOutput:
        """
        Transformers backend的生成（支持双prompt）

        Args:
            prompts: π_θ的prompts
            prompts_t: π_t的prompts（可选）
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
            for step in range(max_new_tokens):
                # ✅ 对齐logits并计算概率（处理不同tokenizer）
                probs_theta, probs_t = self._align_logits(logits_theta, logits_t, temperature)

                # ✅ 强制第一个token使用π_t
                if step == 0 and self.force_target_for_first_token:
                    # 第一个token直接使用π_t，不进行混合
                    q_star = probs_t
                    alpha = torch.zeros(batch_size, device=probs_theta.device)  # α=0 表示完全使用π_t
                else:
                    # 后续token正常计算alpha和q*
                    # 计算 α* 和 q*
                    alpha = 1 - self.alpha_computer.compute(probs_theta, probs_t)
                    q_star = self._compute_q_star(probs_theta, probs_t, alpha)

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
                    alpha = torch.zeros(batch_size, device=probs_theta.device)  # α=0 表示完全使用π_t
                else:
                    # 后续token正常计算alpha和q*
                    # 计算 α*
                    alpha = self.alpha_computer.compute(probs_theta, probs_t)
                    # 计算 q*
                    q_star = self._compute_q_star(probs_theta, probs_t, alpha)

                # 应用 top-p / top-k
                if top_p < 1.0 or top_k > 0:
                    q_star = self._apply_sampling_filters(q_star, top_p, top_k)

                # ✅ 采样前安全检查
                if torch.isnan(q_star).any() or torch.isinf(q_star).any() or (q_star < 0).any():
                    print(f"⚠️  Warning: Invalid q_star at step {step}, using uniform fallback")
                    q_star = torch.ones_like(q_star) / q_star.size(-1)

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

        return SamplingOutput(
            generated_texts=generated_texts,
            generated_ids=generated_ids,
            alpha_values=alpha_values,
            ess_ratios=ess_ratios,
            diagnostics=diagnostics
        )

    def _generate_vllm(
        self,
        prompts: List[str],
        prompts_t: Optional[List[str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        return_diagnostics: bool,
        **kwargs
    ) -> SamplingOutput:
        """
        VLLM backend的生成

        注意: VLLM不支持逐token的q*采样,这里提供两种模式:
        1. 近似模式: 先用π_θ生成,然后用π_t重新评分
        2. 回退到transformers模式 (慢但精确)
        """
        raise NotImplementedError(
            "VLLM backend暂不支持完整的q*采样 (需要逐token控制)。"
            "请使用backend='transformers'或实现基于VLLM的近似方法。"
            "可选方案: 1) 使用VLLM生成候选,然后rejection sampling "
            "2) 使用transformers backend进行精确采样"
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

        核心改进：
        对于从tokenizer config检测到的special tokens（如EOS、chat template等），
        这些token在Base model中可能没见过或处理不好。
        对这些token，我们直接使用π_t的概率，不进行几何平均。

        数学上等价于：对special token设置α=0（完全使用π_t）

        注意：probs_theta和probs_t应该已经通过_align_logits对齐到相同vocabulary
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)  # [batch, 1]

        eps = 1e-10

        # 计算log概率
        log_probs_theta = torch.log(torch.clamp(probs_theta, min=eps))
        log_probs_t = torch.log(torch.clamp(probs_t, min=eps))

        if self.force_target_for_special_tokens and self.special_token_mask is not None:
            # 使用special token mask
            # special_token_mask: [vocab_size] bool tensor
            # True表示special token，应该使用π_t

            # 将mask移到正确的device并扩展到batch
            device = probs_theta.device
            special_mask = self.special_token_mask.to(device)  # [vocab_size]
            special_mask = special_mask.unsqueeze(0)  # [1, vocab_size]

            # 计算两种情况的log q
            # 正常token: log q = α log π_θ + (1-α) log π_t
            # Special token: log q = log π_t （α=0）

            log_q_normal = alpha * log_probs_theta + (1 - alpha) * log_probs_t
            log_q_special = log_probs_t  # 直接使用π_t

            # 使用mask选择
            log_q = torch.where(special_mask, log_q_special, log_q_normal)
        else:
            # 原始方法：统一的alpha
            log_q = alpha * log_probs_theta + (1 - alpha) * log_probs_t

        # 归一化
        q_star = F.softmax(log_q, dim=-1)
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
# 便捷函数
# ============================================

def create_optimal_sampling_model(
    model_theta: str,
    model_t: Optional[str] = None,
    backend: str = "transformers",
    alpha_method: str = "kl_symmetry",
    constraint_to_target: bool = False,
    target_top_k: int = -1,
    target_top_p: float = 1.0,
    force_target_for_special_tokens: bool = True,
    force_target_for_first_token: bool = True,
    **kwargs
) -> OptimalSamplingModel:
    """
    便捷的模型创建函数

    Args:
        model_theta: π_θ模型路径
        model_t: π_t模型路径（可选）
        backend: Backend类型
        alpha_method: Alpha计算方法
        constraint_to_target: ✨ 是否限制在π_t的support上（推荐True）
        target_top_k: ✨ π_t的top-k限制
        target_top_p: ✨ π_t的top-p限制
        force_target_for_special_tokens: ✨ 对special tokens直接使用π_t（推荐True）
        force_target_for_first_token: ✨ 强制第一个token使用π_t（推荐True）
        **kwargs: 其他参数

    Examples:
        >>> # 基础使用（同一个模型）
        >>> model = create_optimal_sampling_model(
        ...     model_theta="meta-llama/Llama-2-7b-hf",
        ...     alpha_method="fixed",
        ...     fixed_alpha=0.5
        ... )

        >>> # 使用不同的模型
        >>> model = create_optimal_sampling_model(
        ...     model_theta="meta-llama/Llama-2-7b-hf",
        ...     model_t="meta-llama/Llama-2-7b-chat-hf",
        ...     alpha_method="kl_symmetry"
        ... )

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
    """
    return OptimalSamplingModel(
        model_theta_path=model_theta,
        model_t_path=model_t,
        backend=backend,
        alpha_method=alpha_method,
        constraint_to_target=constraint_to_target,
        target_top_k=target_top_k,
        target_top_p=target_top_p,
        force_target_for_special_tokens=force_target_for_special_tokens,
        force_target_for_first_token=force_target_for_first_token,
        **kwargs
    )


def apply_chat_template_to_prompts(
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    add_generation_prompt: bool = True
) -> List[str]:
    """
    对一批对话消息应用chat template

    Args:
        tokenizer: HuggingFace tokenizer
        messages_list: 对话消息列表，每个元素是一个对话
            例如: [[{"role": "user", "content": "Hello"}], ...]
        add_generation_prompt: 是否添加生成提示（通常为True）

    Returns:
        格式化后的prompt列表

    Examples:
        >>> messages = [
        ...     [{"role": "user", "content": "Hello, how are you?"}],
        ...     [{"role": "user", "content": "What is AI?"}]
        ... ]
        >>> prompts = apply_chat_template_to_prompts(tokenizer, messages)
    """
    prompts = []
    for messages in messages_list:
        if hasattr(tokenizer, 'apply_chat_template'):
            # 使用tokenizer的chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        else:
            # 回退到简单拼接（对于没有chat template的模型）
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            if add_generation_prompt:
                prompt += "Assistant: "

        prompts.append(prompt)

    return prompts


if __name__ == "__main__":
    # 测试示例
    print("Testing OptimalSamplingModel...")

    # 注意: 需要替换为实际的模型路径
    model = create_optimal_sampling_model(
        model_theta="Qwen/Qwen3-4B-Base",
        model_t="Qwen/Qwen3-14B",
        alpha_method="ess_balance",
        constraint_to_target=True,
        target_top_k=50,
        force_target_for_first_token=True,
        force_target_for_special_tokens=True,
        # target_top_p=0.95,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    messages_list = [
        [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        [
            {"role": "user",
             "content": "The operation $\\otimes$ is defined for all nonzero numbers by $a \otimes b = \\frac{a^{2}}{b}$. Determine $[(1 \otimes 2) \otimes 3] - [1 \otimes (2 \otimes 3)]$."}
        ],
        [
            {"role": "user",
             "content": "Doug constructs a square window using $8$ equal-size panes of glass. The ratio of the height to width for each pane is $5 : 2$, and the borders around and between the panes are $2$ inches wide. In inches, what is the side length of the square window?"}
        ],
        [
            {"role": "user",
             "content": "Let $P(x)$ be a polynomial of degree $3n$ such that \\begin{align*} P(0) = P(3) = \dots = P(3n) &= 2, \\ P(1) = P(4) = \dots = P(3n+1-2) &= 1, \\ P(2) = P(5) = \dots = P(3n+2-2) &= 0. \end{align*} Also, $P(3n+1) = 730$. Determine $n$."}
        ],
        [
            {"role": "user",
             "content": "Let $f$ be the function defined by $f(x)=ax^2-\sqrt{2}$ for some positive $a$. If $f(f(\sqrt{2}))=-\sqrt{2}$ then $a=$"}
        ]
    ]

    # 应用chat template
    prompts = apply_chat_template_to_prompts(model.tokenizer, messages_list)

    print("\n" + "="*60)
    print("Formatted prompts:")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt}...")  # 只显示前100个字符

        # 生成
        outputs = model.generate(
            prompts=[prompt],
            max_new_tokens=512,
            # temperature=0.7,
            # top_p=0.9,
            # top_k=50,
        )

        print("\n" + "="*60)
        print("Generated texts:")
        print(f"{i+1}. {outputs.generated_texts[0]}\n\n\n")

        print("\n" + "="*60)
        print("Diagnostics:")
        print(f"Alpha values shape: {outputs.alpha_values.shape}")
        print(f"Alpha mean: {outputs.alpha_values.mean():.3f}")
        print(f"ESS ratio mean: {outputs.ess_ratios.mean():.3f}")
        print(f"ESS ratio std: {outputs.ess_ratios.std():.3f}")
