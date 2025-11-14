"""
OptimalSamplingModel v2: 简化版本，易于debug

核心功能：
- 支持两个不同的模型（π_θ 和 π_t）
- 支持不同的输入prompts
- 支持多种alpha计算方法
- 支持constraint_to_target（限制在π_t的support上）

简化设计：
- 只支持transformers backend
- 默认tokenizer相同
- 移除unseen_token_handling等复杂功能
- 清晰的代码结构，易于调试
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class SamplingOutput:
    """采样输出"""
    generated_texts: List[str]
    generated_ids: torch.Tensor
    alpha_values: torch.Tensor  # [batch, seq_len]
    diagnostics: Dict[str, torch.Tensor]


class AlphaComputer:
    """Alpha参数计算器"""

    def __init__(self, method: str = "kl_symmetry", fixed_alpha: float = 0.5,
                 tol: float = 1e-6, max_iter: int = 20,
                 constraint_to_target: bool = False,
                 target_top_k: int = -1,
                 target_top_p: float = 1.0):
        """
        Args:
            method: alpha计算方法 ["fixed", "kl_symmetry", "ess_balance", "entropy"]
            fixed_alpha: 固定alpha值
            tol: 求解容差
            max_iter: 最大迭代次数
            constraint_to_target: 是否限制在π_t的support上
            target_top_k: π_t的top-k限制
            target_top_p: π_t的top-p限制
        """
        self.method = method
        self.fixed_alpha = fixed_alpha
        self.tol = tol
        self.max_iter = max_iter
        self.eps = 1e-10

        self.constraint_to_target = constraint_to_target
        self.target_top_k = target_top_k
        self.target_top_p = target_top_p

    def compute(self, probs_theta: torch.Tensor, probs_t: torch.Tensor) -> torch.Tensor:
        """
        计算alpha

        Args:
            probs_theta: [batch, vocab_size]
            probs_t: [batch, vocab_size]

        Returns:
            alpha: [batch]
        """
        # 应用support约束
        if self.constraint_to_target and (self.target_top_k > 0 or self.target_top_p < 1.0):
            probs_theta, probs_t = self._apply_support_constraint(probs_theta, probs_t)

        # 计算alpha
        if self.method == "fixed":
            return self._fixed(probs_theta)
        elif self.method == "kl_symmetry":
            return self._kl_symmetry(probs_theta, probs_t)
        elif self.method == "ess_balance":
            return self._ess_balance(probs_theta, probs_t)
        elif self.method == "entropy":
            return self._entropy(probs_theta, probs_t)
        else:
            raise ValueError(f"Unknown alpha method: {self.method}")

    def _apply_support_constraint(self, probs_theta: torch.Tensor,
                                   probs_t: torch.Tensor) -> tuple:
        """限制在π_t的support上"""
        batch_size, vocab_size = probs_t.shape
        mask = torch.ones_like(probs_t, dtype=torch.bool)

        # Top-k约束
        if self.target_top_k > 0:
            k = min(self.target_top_k, vocab_size)
            _, top_k_indices = torch.topk(probs_t, k=k, dim=-1)
            mask_k = torch.zeros_like(probs_t, dtype=torch.bool)
            mask_k.scatter_(-1, top_k_indices, True)
            mask = mask & mask_k

        # Top-p约束
        if self.target_top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs_t, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            indices_to_keep = cumsum_probs <= self.target_top_p
            indices_to_keep[..., 0] = True
            mask_p = torch.zeros_like(probs_t, dtype=torch.bool)
            mask_p.scatter_(-1, sorted_indices, indices_to_keep)
            mask = mask & mask_p

        # 应用mask并归一化
        probs_theta_masked = probs_theta * mask.float()
        probs_t_masked = probs_t * mask.float()
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
        KL对称条件: D_KL(q||π_θ) = D_KL(q||π_t)
        等价于: E_q[log(π_t/π_θ)] = 0
        """
        batch_size = probs_theta.shape[0]
        device = probs_theta.device

        # Clamp概率
        probs_theta = torch.clamp(probs_theta, min=self.eps, max=1.0)
        probs_t = torch.clamp(probs_t, min=self.eps, max=1.0)

        # # 检测相同分布
        # max_ratio = (probs_theta / probs_t).max(dim=-1)[0]
        # min_ratio = (probs_theta / probs_t).min(dim=-1)[0]
        # nearly_identical = (max_ratio < 1.1) & (min_ratio > 0.9)
        #
        # if nearly_identical.all():
        #     return torch.full((batch_size,), 0.5, device=device, dtype=probs_theta.dtype)

        # 二分搜索
        alpha_low = torch.full((batch_size,), 0.0, device=device)
        alpha_high = torch.full((batch_size,), 1.0, device=device)

        log_probs_theta = torch.log(probs_theta)
        log_probs_t = torch.log(probs_t)
        log_ratio = log_probs_t - log_probs_theta

        for _ in range(self.max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2
            alpha_expanded = alpha_mid.unsqueeze(-1)

            # 计算q_alpha (log-space)
            log_q_unnormalized = alpha_expanded * log_probs_theta + (1 - alpha_expanded) * log_probs_t
            log_q = log_q_unnormalized - torch.logsumexp(log_q_unnormalized, dim=-1, keepdim=True)
            q_alpha = torch.exp(log_q)

            # 计算delta
            delta = (q_alpha * log_ratio).sum(dim=-1)

            # 处理NaN/Inf
            invalid = torch.isnan(delta) | torch.isinf(delta)
            if invalid.any():
                delta = torch.where(invalid, torch.tensor(0.0, device=device), delta)

            # 更新区间
            mask = delta > 0
            alpha_low = torch.where(mask, alpha_mid, alpha_low)
            alpha_high = torch.where(mask, alpha_high, alpha_mid)

            if (alpha_high - alpha_low).max() < self.tol:
                break

        alpha_result = (alpha_low + alpha_high) / 2
        # alpha_result = alpha_result # torch.clamp(alpha_result, min=0.1, max=0.9)
        # alpha_result = torch.where(nearly_identical, torch.tensor(0.5, device=device), alpha_result)

        return alpha_result

    def _ess_balance(self, probs_theta: torch.Tensor, probs_t: torch.Tensor) -> torch.Tensor:
        """
        ESS平衡条件: ESS_θ(q) = ESS_t(q)
        等价于: Σ(π_θ²/q) = Σ(π_t²/q)
        """
        batch_size = probs_theta.shape[0]
        device = probs_theta.device

        # 检查边界
        alpha_zero = torch.zeros(batch_size, device=device)
        alpha_one = torch.ones(batch_size, device=device)

        q_0 = self._geometric_mean(probs_theta, probs_t, alpha_zero)
        sum_theta_sq_0 = ((probs_theta ** 2) / (q_0 + self.eps)).sum(dim=-1)
        sum_t_sq_0 = ((probs_t ** 2) / (q_0 + self.eps)).sum(dim=-1)
        delta_0 = sum_t_sq_0 - sum_theta_sq_0

        q_1 = self._geometric_mean(probs_theta, probs_t, alpha_one)
        sum_theta_sq_1 = ((probs_theta ** 2) / (q_1 + self.eps)).sum(dim=-1)
        sum_t_sq_1 = ((probs_t ** 2) / (q_1 + self.eps)).sum(dim=-1)
        delta_1 = sum_t_sq_1 - sum_theta_sq_1

        # 检测问题
        need_fallback = (delta_0 * delta_1 > 0) | (sum_theta_sq_0 > 100) | (sum_t_sq_0 > 100)

        # 搜索范围
        alpha_low = torch.where(need_fallback,
                               torch.full((batch_size,), 0.2, device=device),
                               torch.zeros(batch_size, device=device))
        alpha_high = torch.where(need_fallback,
                                torch.full((batch_size,), 0.8, device=device),
                                torch.ones(batch_size, device=device))

        # 二分搜索
        for _ in range(self.max_iter):
            alpha_mid = (alpha_low + alpha_high) / 2
            q_alpha = self._geometric_mean(probs_theta, probs_t, alpha_mid)

            sum_theta_sq = ((probs_theta ** 2) / (q_alpha + self.eps)).sum(dim=-1)
            sum_t_sq = ((probs_t ** 2) / (q_alpha + self.eps)).sum(dim=-1)
            delta = sum_t_sq - sum_theta_sq

            mask = delta < 0
            alpha_low = torch.where(mask, alpha_mid, alpha_low)
            alpha_high = torch.where(mask, alpha_high, alpha_mid)

            if (alpha_high - alpha_low).max() < self.tol:
                break

        alpha_result = (alpha_low + alpha_high) / 2

        # Fallback到KL对称
        if need_fallback.any():
            alpha_fallback = self._kl_symmetry(probs_theta, probs_t)
            alpha_result = torch.where(need_fallback, alpha_fallback, alpha_result)

        return alpha_result  # torch.clamp(alpha_result, 0.1, 0.9)

    def _entropy(self, probs_theta: torch.Tensor, probs_t: torch.Tensor) -> torch.Tensor:
        """熵公式"""
        h_theta = -(probs_theta * torch.log(probs_theta + self.eps)).sum(dim=-1)
        h_t = -(probs_t * torch.log(probs_t + self.eps)).sum(dim=-1)
        alpha = h_theta / (h_theta + h_t + self.eps)
        return torch.clamp(alpha, 0.0, 1.0)

    def _geometric_mean(self, p1: torch.Tensor, p2: torch.Tensor,
                       alpha: torch.Tensor) -> torch.Tensor:
        """几何平均"""
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        log_q = alpha * torch.log(p1 + self.eps) + (1 - alpha) * torch.log(p2 + self.eps)
        return F.softmax(log_q, dim=-1)


class OptimalSamplingModel:
    """最优采样模型（简化版）"""

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
        dtype: torch.dtype = torch.float16
    ):
        """
        Args:
            model_theta_path: π_θ模型路径
            model_t_path: π_t模型路径 (None则使用同一模型)
            alpha_method: alpha计算方法
            fixed_alpha: 固定alpha值
            alpha_tol: 求解容差
            constraint_to_target: 是否限制在π_t的support上
            target_top_k: π_t的top-k限制
            target_top_p: π_t的top-p限制
            force_target_for_special_tokens: 对special tokens强制使用π_t（推荐True）
            force_target_for_first_token: 强制第一个token由π_t解码（推荐True）
            device: 设备
            dtype: 数据类型
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.dtype = dtype
        self.force_target_for_special_tokens = force_target_for_special_tokens
        self.force_target_for_first_token = force_target_for_first_token

        # 加载tokenizer
        print(f"Loading tokenizer from {model_theta_path}...")
        self.tokenizer_theta = AutoTokenizer.from_pretrained(model_theta_path)
        if self.tokenizer_theta.pad_token is None:
            self.tokenizer_theta.pad_token = self.tokenizer_theta.eos_token

        # 保持向后兼容
        self.tokenizer = self.tokenizer_theta

        # 加载π_θ
        print(f"Loading π_θ from {model_theta_path}...")
        self.model_theta = AutoModelForCausalLM.from_pretrained(
            model_theta_path,
            torch_dtype=dtype,
            device_map=device if device != "cuda" else "auto"
        )
        self.model_theta.eval()

        # 加载π_t
        if model_t_path is None or model_t_path == model_theta_path:
            print("Using π_θ as π_t (same model)")
            self.model_t = self.model_theta
            self.tokenizer_t = self.tokenizer_theta
            self.same_model = True
        else:
            print(f"Loading π_t from {model_t_path}...")
            self.tokenizer_t = AutoTokenizer.from_pretrained(model_t_path)
            if self.tokenizer_t.pad_token is None:
                self.tokenizer_t.pad_token = self.tokenizer_t.eos_token

            self.model_t = AutoModelForCausalLM.from_pretrained(
                model_t_path,
                torch_dtype=dtype,
                device_map=device if device != "cuda" else "auto"
            )
            self.model_t.eval()
            self.same_model = False

        # 检测special tokens（π_t有但π_θ可能没见过的）
        if force_target_for_special_tokens and not self.same_model:
            self._detect_special_tokens()
        else:
            self.special_token_mask = None

        # 初始化alpha计算器
        self.alpha_computer = AlphaComputer(
            method=alpha_method,
            fixed_alpha=fixed_alpha,
            tol=alpha_tol,
            constraint_to_target=constraint_to_target,
            target_top_k=target_top_k,
            target_top_p=target_top_p
        )

        print(f"\n✓ OptimalSamplingModel initialized")
        print(f"  Alpha method: {alpha_method}")
        print(f"  Constraint to target: {constraint_to_target}")
        if constraint_to_target:
            if target_top_k > 0:
                print(f"  - Target top-k: {target_top_k}")
            if target_top_p < 1.0:
                print(f"  - Target top-p: {target_top_p}")
        if force_target_for_special_tokens and self.special_token_mask is not None:
            print(f"  Force target for special tokens: ENABLED")
            print(f"  - {self.special_token_mask.sum().item()} special tokens detected")
        if force_target_for_first_token:
            print(f"  Force target for first token: ENABLED")
            print(f"  - First token will use π_t directly")

    def _detect_special_tokens(self):
        """
        检测special tokens（从tokenizer config中获取）

        策略：
        1. 获取π_t的所有special tokens（从tokenizer.all_special_tokens）
        2. 获取π_θ的所有special tokens
        3. 找到π_t特有的special tokens（可能是chat template等）
        4. 创建mask标记这些token ID

        这些token在Base model中可能没见过（如<|im_start|>等），
        应该直接使用π_t的概率。
        """
        print("\n" + "="*60)
        print("Detecting special tokens...")
        print("="*60)

        # ✅ 使用模型的实际vocab size，而不是tokenizer的
        # 模型的vocab size可能比tokenizer大（padding等）
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

        # 方案1：保守策略 - 所有special tokens都强制使用π_t
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
        prompts_t: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        return_diagnostics: bool = True
    ) -> SamplingOutput:
        """
        生成文本

        Args:
            prompts: π_θ的输入prompts
            prompts_t: π_t的输入prompts（可选，None则使用prompts）
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: nucleus sampling
            top_k: top-k sampling
            return_diagnostics: 是否返回诊断信息

        Returns:
            SamplingOutput
        """
        batch_size = len(prompts)

        # 判断是否使用不同的prompts
        use_different_prompts = (prompts_t is not None) and (prompts_t != prompts)
        if use_different_prompts:
            if len(prompts_t) != batch_size:
                raise ValueError(f"prompts_t长度({len(prompts_t)})必须与prompts长度({batch_size})相同")
            print("Using different prompts for π_θ and π_t")

        # Tokenize
        inputs_theta = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids_theta = inputs_theta["input_ids"].to(self.model_theta.device)
        attention_mask_theta = inputs_theta["attention_mask"].to(self.model_theta.device)

        if use_different_prompts:
            inputs_t = self.tokenizer(prompts_t, return_tensors="pt", padding=True, truncation=True)
            input_ids_t = inputs_t["input_ids"].to(self.model_t.device)
            attention_mask_t = inputs_t["attention_mask"].to(self.model_t.device)
        else:
            input_ids_t = input_ids_theta
            attention_mask_t = attention_mask_theta

        # 存储结果
        all_generated_ids = []
        all_alpha_values = []
        all_diagnostics = []

        # EOS检测
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.model_theta.device)
        eos_token_id = self.tokenizer.eos_token_id

        # Prefill: 处理prompt，初始化KV cache
        outputs_theta = self.model_theta(
            input_ids=input_ids_theta,
            attention_mask=attention_mask_theta,
            use_cache=True
        )
        past_key_values_theta = outputs_theta.past_key_values
        logits_theta = outputs_theta.logits[:, -1, :]

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

        # Decode: 逐token生成
        for step in range(max_new_tokens):
            # 计算概率
            probs_theta = F.softmax(logits_theta / temperature, dim=-1)
            probs_t = F.softmax(logits_t / temperature, dim=-1)

            # ✅ 强制第一个token使用π_t
            if step == 0 and self.force_target_for_first_token:
                # 第一个token直接使用π_t，不进行混合
                q_star = probs_t
                alpha = torch.zeros(batch_size, device=probs_theta.device)  # α=0 表示完全使用π_t
            else:
                # 后续token正常计算alpha和q*
                alpha = self.alpha_computer.compute(probs_theta, probs_t)
                q_star = self._compute_q_star(probs_theta, probs_t, alpha)

            # 应用top-p/top-k
            if top_p < 1.0 or top_k > 0:
                q_star = self._apply_sampling_filters(q_star, top_p, top_k)

            # 安全检查
            if torch.isnan(q_star).any() or torch.isinf(q_star).any():
                print(f"⚠️  Warning: Invalid q_star at step {step}, using π_t")
                q_star = probs_t

            # 采样
            try:
                next_tokens = torch.multinomial(q_star, num_samples=1).squeeze(-1)
            except RuntimeError as e:
                print(f"⚠️  Sampling failed at step {step}: {e}, using argmax")
                next_tokens = q_star.argmax(dim=-1)

            # 检查EOS
            if eos_token_id is not None:
                finished = finished | (next_tokens == eos_token_id)

            # 对已完成的样本使用pad token
            if self.tokenizer.pad_token_id is not None:
                next_tokens = torch.where(
                    finished,
                    torch.tensor(self.tokenizer.pad_token_id, device=next_tokens.device),
                    next_tokens
                )

            # 保存
            all_generated_ids.append(next_tokens.unsqueeze(-1))
            if return_diagnostics:
                diag = self._compute_diagnostics(probs_theta, probs_t, q_star, alpha)
                all_alpha_values.append(alpha.cpu())
                all_diagnostics.append({k: v.cpu() for k, v in diag.items()})

            # 所有样本完成则停止
            if finished.all():
                break

            # 更新attention mask
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

            # Forward新token
            outputs_theta = self.model_theta(
                input_ids=next_tokens.unsqueeze(-1),
                attention_mask=attention_mask_theta,
                past_key_values=past_key_values_theta,
                use_cache=True
            )
            past_key_values_theta = outputs_theta.past_key_values
            logits_theta = outputs_theta.logits[:, -1, :]

            if not self.same_model:
                outputs_t = self.model_t(
                    input_ids=next_tokens.unsqueeze(-1),
                    attention_mask=attention_mask_t,
                    past_key_values=past_key_values_t,
                    use_cache=True
                )
                past_key_values_t = outputs_t.past_key_values
                logits_t = outputs_t.logits[:, -1, :]
            else:
                logits_t = logits_theta

        # 组装结果
        generated_ids = torch.cat(all_generated_ids, dim=-1)
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 组装诊断信息
        diagnostics = {}
        if return_diagnostics and all_diagnostics:
            alpha_values = torch.stack(all_alpha_values, dim=1)
            for key in all_diagnostics[0].keys():
                values = torch.stack([d[key] for d in all_diagnostics], dim=1)
                diagnostics[key] = values
        else:
            alpha_values = torch.zeros((batch_size, 0))

        return SamplingOutput(
            generated_texts=generated_texts,
            generated_ids=generated_ids,
            alpha_values=alpha_values,
            diagnostics=diagnostics
        )

    def _compute_q_star(self, probs_theta: torch.Tensor, probs_t: torch.Tensor,
                        alpha: torch.Tensor) -> torch.Tensor:
        """
        计算q*分布（几何平均，支持special token强制使用π_t）

        对于special tokens（从tokenizer config检测到的），直接使用π_t的概率，
        不进行几何平均混合（等价于α=0）。
        """
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)

        eps = 1e-10

        # 计算log概率
        log_probs_theta = torch.log(probs_theta + eps)
        log_probs_t = torch.log(probs_t + eps)

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

    def _apply_sampling_filters(self, probs: torch.Tensor, top_p: float,
                                top_k: int) -> torch.Tensor:
        """应用top-p/top-k过滤"""
        # Top-k
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, top_k_indices, top_k_probs)

        # Top-p
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)

        # 归一化
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        return probs

    def _compute_diagnostics(self, probs_theta: torch.Tensor, probs_t: torch.Tensor,
                            q_star: torch.Tensor, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算诊断信息"""
        eps = 1e-10

        # ESS
        ess_theta = 1.0 / ((probs_theta ** 2) / (q_star + eps)).sum(dim=-1)
        ess_t = 1.0 / ((probs_t ** 2) / (q_star + eps)).sum(dim=-1)
        ess_ratio = ess_theta / (ess_t + eps)

        # KL散度
        kl_theta = (q_star * torch.log((q_star + eps) / (probs_theta + eps))).sum(dim=-1)
        kl_t = (q_star * torch.log((q_star + eps) / (probs_t + eps))).sum(dim=-1)

        return {
            "alpha": alpha,
            "ess_theta": ess_theta,
            "ess_t": ess_t,
            "ess_ratio": ess_ratio,
            "kl_theta": kl_theta,
            "kl_t": kl_t,
            "kl_diff": (kl_theta - kl_t).abs()
        }


def create_optimal_sampling_model(
    model_theta: str,
    model_t: Optional[str] = None,
    alpha_method: str = "kl_symmetry",
    fixed_alpha: float = 0.5,
    constraint_to_target: bool = False,
    target_top_k: int = -1,
    target_top_p: float = 1.0,
    force_target_for_special_tokens: bool = True,
    force_target_for_first_token: bool = True,
    device: str = "cuda",
    **kwargs
) -> OptimalSamplingModel:
    """
    便捷的模型创建函数

    Args:
        model_theta: π_θ模型路径
        model_t: π_t模型路径（None则使用同一模型）
        alpha_method: alpha计算方法
        fixed_alpha: 固定alpha值
        constraint_to_target: 是否限制在π_t的support上
        target_top_k: π_t的top-k限制
        target_top_p: π_t的top-p限制
        force_target_for_special_tokens: 对special tokens强制使用π_t（推荐True）
        force_target_for_first_token: 强制第一个token由π_t解码（推荐True）
        device: 设备
        **kwargs: 其他参数

    Examples:
        >>> # 同一个模型
        >>> model = create_optimal_sampling_model(
        ...     model_theta="Qwen/Qwen2.5-0.5B",
        ...     alpha_method="kl_symmetry"
        ... )

        >>> # 不同模型（推荐配置）
        >>> model = create_optimal_sampling_model(
        ...     model_theta="Qwen/Qwen2.5-0.5B",
        ...     model_t="Qwen/Qwen2.5-0.5B-Instruct",
        ...     alpha_method="kl_symmetry",
        ...     constraint_to_target=True,
        ...     target_top_k=100,
        ...     force_target_for_special_tokens=True,  # Special tokens使用π_t
        ...     force_target_for_first_token=True      # 第一个token使用π_t
        ... )

        >>> # 不同输入
        >>> outputs = model.generate(
        ...     prompts=["Hello"],
        ...     prompts_t=["<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"],
        ...     max_new_tokens=50
        ... )
    """
    return OptimalSamplingModel(
        model_theta_path=model_theta,
        model_t_path=model_t,
        alpha_method=alpha_method,
        fixed_alpha=fixed_alpha,
        constraint_to_target=constraint_to_target,
        target_top_k=target_top_k,
        target_top_p=target_top_p,
        force_target_for_special_tokens=force_target_for_special_tokens,
        force_target_for_first_token=force_target_for_first_token,
        device=device,
        **kwargs
    )


if __name__ == "__main__":
    # 测试示例
    print("Testing OptimalSamplingModel v2...")

    model = create_optimal_sampling_model(
        model_theta="Qwen/Qwen3-4B",
        model_t="Qwen/Qwen3-8B",
        # alpha_method="ess_balance",
        # alpha_method="fixed",
        constraint_to_target=False,
        # target_top_k=25,
        force_target_for_special_tokens=True,  # ✅ 启用special token强制使用π_t
        force_target_for_first_token=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # prompts_theta = ["Hello"]
    # prompts_t = ["<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"]

    messages_list = [
        [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        [
            {"role": "user", "content": "The operation $\\otimes$ is defined for all nonzero numbers by $a \otimes b = \\frac{a^{2}}{b}$. Determine $[(1 \otimes 2) \otimes 3] - [1 \otimes (2 \otimes 3)]$."}
        ],
        [
            {"role": "user", "content": "Doug constructs a square window using $8$ equal-size panes of glass. The ratio of the height to width for each pane is $5 : 2$, and the borders around and between the panes are $2$ inches wide. In inches, what is the side length of the square window?"}
        ],
        [
            {"role": "user", "content": "Let $P(x)$ be a polynomial of degree $3n$ such that \\begin{align*} P(0) = P(3) = \dots = P(3n) &= 2, \\ P(1) = P(4) = \dots = P(3n+1-2) &= 1, \\ P(2) = P(5) = \dots = P(3n+2-2) &= 0. \end{align*} Also, $P(3n+1) = 730$. Determine $n$."}
        ],
        [
            {"role": "user", "content": "Let $f$ be the function defined by $f(x)=ax^2-\sqrt{2}$ for some positive $a$. If $f(f(\sqrt{2}))=-\sqrt{2}$ then $a=$"}
        ]
    ]

    prompts = []
    for messages in messages_list:
        if hasattr(model.tokenizer, 'apply_chat_template'):
            prompt = model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # prompt += "<think><\\think>"
        else:
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
                prompt += "Assistant: "

        prompts.append(prompt)

    for ids, prompt in enumerate(prompts):
        print(f"Prompt: #{ids+1}")
        print(prompt)

        outputs = model.generate(
            prompts=[prompt],
            # prompts_t=prompts,
            max_new_tokens=256,
            temperature=1.
        )

        print("[Response]", outputs.generated_texts[0])

        print("-" * 40)
        print("Diagnostics:")
        print(f"Alpha values shape: {outputs.alpha_values.shape}")
        print(f"Alpha mean: {outputs.alpha_values.mean():.3f}")

        print("\n" + "="*60)
