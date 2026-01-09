#!/usr/bin/env python3
"""
脚本生成器：根据不同的实验配置生成启动脚本
生成19组不同的gamma_is实验配置
"""

import os
from pathlib import Path

# 实验配置表
EXPERIMENT_CONFIGS = [
    # Group, Strategy, k_pos, lam_pos, peak, k_neg, lam_neg, 说明
    ("G1", "Mimic", 1.0, 0.8, 1.25, 3.0, 1.5, 16, "模拟 PPO 线性+宽容"),
    ("G2", "Mimic", 1.2, 1.0, 1.20, 5.0, 2.5, 16, "模拟 PPO + 强 Lower Clip"),
    ("G3", "Mimic", 1.5, 1.2, 1.25, 3.0, 3.0, 16, "模拟 PPO + 强 Upper Clip"),
    ("G4", "Legacy", 2.0, 1.6, 1.25, 4.0, 4.0, 16, "前冠军复刻 + 强防"),
    ("G5", "Math", 2.0, 2.0, 1.00, 4.0, 2.5, 16, "标准推荐 (Bell)"),
    ("G6", "Math", 1.5, 1.5, 1.00, 4.0, 2.5, 16, "宽钟形"),
    ("G7", "Math", 3.0, 3.0, 1.00, 4.0, 2.5, 16, "尖钟形"),
    ("G8", "Math", 4.0, 4.0, 1.00, 4.0, 2.5, 16, "针尖 (高精度)"),
    ("G9", "Drift", 2.0, 2.2, 0.90, 4.0, 3.0, 16, "微左偏"),
    ("G10", "Drift", 2.0, 2.5, 0.80, 5.0, 4.0, 16, "强左偏纠偏"),
    ("G11", "Drift", 1.0, 1.25, 0.80, 4.0, 2.5, 16, "线性左偏"),
    ("G12", "Drift", 1.5, 2.5, 0.60, 4.0, 2.5, 16, "极端深锚"),
    ("G13", "NegDef", 2.0, 2.0, 1.00, 3.0, 1.5, 16, "负向极宽容 (PPO [0.8,3] 模拟)"),
    ("G14", "NegDef", 2.0, 2.0, 1.00, 6.0, 3.0, 16, "负向只杀小 w"),
    ("G15", "NegDef", 2.0, 2.0, 1.00, 6.0, 6.0, 16, "负向全杀 (铁穹)"),
    ("G16", "Wild", 2.0, 1.8, 1.11, 4.0, 2.5, 16, "微右偏激进"),
    ("G17", "Wild", 1.5, 1.0, 1.50, 5.0, 5.0, 16, "赌徒模式 (High Risk)"),
    ("G18", "Wild", 3.0, 3.0, 1.00, 3.0, 3.0, 16, "完全对称"),
    ("G19", "Wild", 4.0, 3.2, 1.25, 4.0, 4.0, 16, "模拟 PPO 的'平顶'感"),

    # V2
    ("G20", "Mimic_V2", 2.0, 2.0, 1, 8.0, 4.0, 16, ""),
    ("G21", "Legacy_Fix", 2.0, 2.0, 1, 6.0, 4.0, 16, ""),
    ("G22", "Dome_Fix", 2.0, 2.0, 1.00, 4.0, 2.0, 16, ""),
    ("G23", "Gamble_Safe", 2.0, 2.0, 1.0, 3.0, 2.0, 16, ""),
    ("G24", "Asym_Prec", 3.0, 2.0, 1.5, 4.0, 2.5, 16, ""),
    ("G25", "Box_Shape", 4.0, 2.0, 2.0, 8.0, 4.0, 16, ""),

    # V3
    ("G26", "Stable_Tight", 2.7, 3.0, 0.90, 6.0, 3.0, 16, "G14改良: Pos微左移收紧 + Neg标准铁壁"),
    ("G27", "Broad_Punish", 2.0, 2.5, 0.80, 5.0, 2.0, 16, "宽域惩罚: Pos左移(0.8) + Neg极宽(Peak 2.5)"),
    ("G28", "Wide_Net", 2.5, 2.5, 1.00, 5.0, 2.0, 16, "广撒网: Neg覆盖极宽范围防止后期逃逸"),
    ("G29", "Precise_Wall", 3.0, 3.0, 1.00, 6.0, 2.4, 16, "高精铁壁: Pos高精(k=3) + Neg高阶宽域"),
    ("G30", "Left_Anchor", 2.5, 3.2, 0.78, 6.0, 3.0, 16, "强锚定: Pos强力左移限制探索"),
    ("G31", "Conservative", 2.0, 3, 0.67, 8.0, 3.2, 16, "极保守: Pos深锚 + Neg究极左侧截断"),
    ("G32", "High_Order_L", 4.0, 5.0, 0.80, 8.0, 3.0, 16, "高阶左偏: 模拟非对称盒式约束"),
    ("G33", "Balanced_Fix", 3.0, 4.0, 0.75, 5.0, 2.0, 16, "综合修正: 高精左移Pos + 宽域Neg"),

    # V4
    ("G34", "Flat_Punisher", 2.5, 3.2, 0.78, 2.0, 0.4, 16, ""),
    ("G35", "Deep_Guard", 3.0, 4.0, 0.75, 4.0, 0.8, 16, ""),
    ("G36", "Ramp_Wall", 2.0, 2.5, 0.80, 1.5, 0.2, 16, ""),
    ("G37", "Heavy_Anchor", 4.0, 5.0, 0.80, 3.0, 0.5, 16, ""),
    ("G38", "Titan_Guard", 3.0, 3.75, 0.80, 3.0, 0.5, 16, ""),
    ("G39", "High_Order_Wall", 2.5, 3.125, 0.80, 6.0, 0.75, 16, ""),
    ("G40", "Smooth_Fortress", 2.0, 2.5, 0.80, 4.0, 0.5, 16, ""),

    # V5
    ("G41", "G14_Evo", 3.0, 4.0, 0.75, 6.0, 2.72, 16, ""),  # Neg: k=6, P=2.2 -> lambda=2.72
    ("G42", "Urgent_Teach", 3.0, 4.0, 0.75, 4.5, 2.5, 16, ""), # Neg: k=4.5, P=1.8 -> lambda=2.5
    ("G43", "Soft_Explore", 2.0, 2.66, 0.75, 5.0, 2.0, 16, ""), # Pos: k=2, P=0.75; Neg: G33原版
    ("G44", "Anchor_Test", 3.0, 3.33, 0.90, 5.0, 2.0, 16, ""), # Pos: P=0.90; Neg: G33原版
    ("G45", "Wide_Net", 3.0, 4.0, 0.75, 2.5, 0.68, 16, ""),  # Neg: k=2.5, P=2.2 -> lambda=0.68
    ("G46", "Strict_Anchor", 5.0, 5.0, 0.80, 5.0, 2.0, 16, ""),  # Pos: k=5, P=0.8; Neg: 同 G33
    ("G47", "G2_Revival", 2.0, 0.91, 1.10, 5.0, 2.0, 16, ""),  # Pos: k=2, P=1.1; Neg: 同 G33
    ("G48", "The_Sniper", 3.0, 4.0, 0.75, 8.0, 3.5, 16, ""),  # Neg: k=8, P=2.0 -> lambda=3.5

    # V6: ppo_epoch = 4
    ("G33_4", "PPO_4", 3.0, 4.0, 0.75, 5.0, 2.0, 4, ""),
    ("G41_4", "PPO_4", 3.0, 4.0, 0.75, 6.0, 2.72, 4, ""),  # Neg: k=6, P=2.2 -> lambda=2.72

    # V7
    ("G27_4", "Broad_Punish", 2.0, 2.5, 0.80, 5.0, 2.0, 4, ""),
    ("G47_4", "G2_Revival", 2.0, 0.91, 2.2, 5.0, 2.0, 4, ""),
    ("G49", "G2_Revival", 2.0, 0.8, 2.0, 5.0, 2.0, 16, ""),
    ("G50", "G2_Revival", 2.0, 1.0, 2.0, 5.0, 2.0, 16, ""),
    ("G49_4", "G2_Revival", 2.0, 0.8, 2.0, 5.0, 2.0, 4, ""),
    ("G50_4", "G2_Revival", 2.0, 1.0, 2.0, 5.0, 2.0, 4, ""),
    ("G27_64", "Broad_Punish", 2.0, 2.5, 0.80, 5.0, 2.0, 64, ""),

    (" ", " ", 0, 0, 0, 0, 0, 0, ""),
    # X1
    ("X1_4", "Broad_Punish", 2.0, 2.5, 0.80, 5.0, 2.0, 4, ""),
    ("X2_4", "Broad_Punish", 2.0, 2.5, 0.80, 4.0, 2.0, 4, ""),
    ("X3_4", "Broad_Punish", 2.0, 2.5, 0.80, 6.0, 2.0, 4, ""),
    ("X4_4", "Broad_Punish", 2.0, 2.5, 0.80, 5.0, 2.5, 4, ""),
    ("Y1_4", "G2_Revival", 2.0, 0.91, 2.2, 5.0, 2.0, 4, ""),
    ("Y2_4", "G2_Revival", 2.0, 1.2, 2.2, 5.0, 2.0, 4, ""),
    ("Y3_4", "G2_Revival", 2.0, 0.81, 2.2, 5.0, 2.0, 4, ""),
    ("Y4_4", "G2_Revival", 1.5, 0.91, 2.2, 5.0, 2.0, 4, ""),

    ("X1_16", "Broad_Punish", 2.0, 2.5, 0.80, 5.0, 2.0, 16, ""),
    ("X2_16", "Broad_Punish", 2.0, 2.5, 0.80, 4.0, 2.0, 16, ""),
    ("X3_16", "Broad_Punish", 2.0, 2.5, 0.80, 6.0, 2.0, 16, ""),
    ("X4_16", "Broad_Punish", 2.0, 2.5, 0.80, 5.0, 2.5, 16, ""),
    ("Y1_16", "G2_Revival", 2.0, 0.91, 2.2, 5.0, 2.0, 16, ""),
    ("Y2_16", "G2_Revival", 2.0, 1.2, 2.2, 5.0, 2.0, 16, ""),
    ("Y3_16", "G2_Revival", 2.0, 0.81, 2.2, 5.0, 2.0, 16, ""),
    ("Y4_16", "G2_Revival", 1.5, 0.91, 2.2, 5.0, 2.0, 16, ""),

    ("X1_64", "Broad_Punish", 2.0, 2.5, 0.80, 5.0, 2.0, 64, ""),
    ("Y1_64", "G2_Revival", 2.0, 0.91, 2.2, 5.0, 2.0, 64, ""),

    # X2  10 configs
    #
    # DAPO_4, DAPO_16. Adaptive_4, Adaptive_16, 6xY
    ("Y5_4", "G2_Revival", 2.0, 1.5, 1.3, 5.0, 2.0, 4, ""),
    ("Y6_4", "G2_Revival", 2.0, 1.0, 2.0, 6.0, 2.0, 4, ""),

    ("Y5_16", "G2_Revival", 2.0, 1.5, 1.3, 5.0, 2.0, 16, ""),
    ("Y6_16", "G2_Revival", 2.0, 0.91, 2.2, 6.0, 2.0, 16, ""),
    ("Y7_16", "G2_Revival", 2.0, 0.91, 2.2, 8.0, 2.0, 16, ""),
    ("Y8_16", "G2_Revival", 2.0, 1.5, 2.0, 6.0, 2.0, 16, ""),

    ("Z1_16", "G2_Revival", 1.5, 0.91, 1.64, 4.0, 1.0, 16, ""),
]


# 脚本模板
SCRIPT_TEMPLATE = """#!/bin/bash
set -x

# ============================================================================
# Cauchy IS: IS-Reshape v11 with MSE-derived Arctan Framework
#
# Key insight: Arctan is NOT a choice, it's a NECESSITY from:
#   1. MSE framework: L(φ) = α(A)·(w-φ)² + β(A)·(wφ)²
#   2. IS variance property: Var ∝ w²
#
# Solving ∂L/∂φ = 0 gives:
#   φ*(w, A) = w / (1 + λ(A)·w²)
#
# where λ(A) = β(A)/α(A) = Risk_Cost / Opportunity_Cost
#
# Objective function (integrate back):
#   f(w) = (1/√λ) · arctan(√λ · w)
#
# With exponential utility:
#   α(A) = α₀·e^{{scale·A}}  (Learning urgency)
#   β(A) = β₀              (Risk aversion)
#   → λ(A) = (β₀/α₀)·e^{{-scale·A}}
#
# Physical meaning:
#   - A > 0: High urgency → λ small → φ ≈ w (IS-like)
#   - A < 0: High risk fear → λ large → φ truncated
# ============================================================================
#
# Experiment Configuration: {group} - {strategy}
# Description: {description}
# k_pos={k_pos}, lambda_pos={lam_pos}, peak={peak}, k_neg={k_neg}, lambda_neg={lam_neg}
# ============================================================================

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

ALGO="gamma_is"
OBJECT="math"
DATASET="dapo_math"
MODEL_FULL="/mnt/tidal-alsh-hilab/usr/shenguobin/modelscope/Qwen/Qwen3-4B-Base"
MODEL="qwen3-4b-base"
BATCH_SIZE=256
MINI_BATCH=256
PPO_EPOCHS={ppo_epoch}

K_POS={k_pos}
LAMBDA_POS={lam_pos}
K_NEG={k_neg}
LAMBDA_NEG={lam_neg}

PROJECT_NAME="is_shape_experiments"
EXP_NAME="${{ALGO}}_${{OBJECT}}_${{DATASET}}_${{MODEL}}_ppo_epochs_${{PPO_EPOCHS}}_${{K_POS}}_${{LAMBDA_POS}}_${{K_NEG}}_${{LAMBDA_NEG}}_{group}"

# Data Configuration
DATA_ROOT="/mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/is_shape/data"
train_files="${{DATA_ROOT}}/${{DATASET}}/train.parquet"
test_files="[${{DATA_ROOT}}/amc23/test.parquet,${{DATA_ROOT}}/aime25/test.parquet,${{DATA_ROOT}}/aime_2024/test.parquet]"

# Sequence Length Configuration
max_prompt_length=1024
max_response_length=15360

# Launch Training
python3 -u -m recipe.is_shape.code.main_ppo \\
    algorithm.adv_estimator=grpo \\
    data.train_files="$train_files" \\
    data.val_files="$test_files" \\
    data.train_batch_size=$BATCH_SIZE \\
    data.max_prompt_length=$max_prompt_length \\
    data.max_response_length=$max_response_length \\
    +data.stale_iteration=0 \\
    data.filter_overlong_prompts=True \\
    data.truncation='error' \\
    +data.seed=42 \\
    actor_rollout_ref.model.path=$MODEL_FULL \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \\
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \\
    actor_rollout_ref.actor.use_dynamic_bsz=True \\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \\
    +actor_rollout_ref.actor.global_batch_info.dp_size=1 \\
    actor_rollout_ref.actor.policy_loss.loss_mode=gamma_is \\
    +actor_rollout_ref.actor.policy_loss.gamma_is.k_pos=$K_POS \\
    +actor_rollout_ref.actor.policy_loss.gamma_is.lambda_pos=$LAMBDA_POS \\
    +actor_rollout_ref.actor.policy_loss.gamma_is.k_neg=$K_NEG \\
    +actor_rollout_ref.actor.policy_loss.gamma_is.lambda_neg=$LAMBDA_NEG \\
    actor_rollout_ref.actor.use_kl_loss=False \\
    actor_rollout_ref.actor.kl_loss_coef=0.00 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.actor.entropy_coeff=0 \\
    actor_rollout_ref.actor.calculate_entropy=true \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.rollout.enforce_eager=True \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \\
    actor_rollout_ref.rollout.n=8 \\
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \\
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    actor_rollout_ref.rollout.val_kwargs.n=8 \\
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \\
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \\
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \\
    algorithm.use_kl_in_reward=False \\
    custom_reward_function.path=recipe/is_shape/code/reward_function.py \\
    custom_reward_function.name=$OBJECT \\
    reward_model.reward_manager=prime \\
    reward_model.launch_reward_fn_async=True \\
    trainer.critic_warmup=0 \\
    trainer.logger=['console','wandb'] \\
    trainer.project_name=$PROJECT_NAME \\
    trainer.experiment_name=$EXP_NAME \\
    trainer.n_gpus_per_node=8 \\
    trainer.nnodes=1 \\
    trainer.val_before_train=True \\
    trainer.default_local_dir=/mnt/tidal-alsh-hilab/usr/shenguobin/verl/checkpoints/$PROJECT_NAME/$EXP_NAME \\
    trainer.validation_data_dir=/mnt/tidal-alsh-hilab/usr/shenguobin/verl/validation_outputs/$PROJECT_NAME/$EXP_NAME/$TIMESTAMP \\
    trainer.save_freq=256 \\
    trainer.test_freq=16 \\
    trainer.total_epochs=1 \\
    $@
"""


def parse_existing_markdown(markdown_path):
    """解析现有的 markdown 文件，提取用户填写的机器、状态和说明信息"""
    user_data = {}  # {group: {"machine": "...", "status": "...", "description": "..."}}

    if not markdown_path.exists():
        return user_data

    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 查找表格数据（跳过标题和分隔线）
        in_table = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '|' not in line:
                continue

            # 跳过表头和分隔线
            if 'Group' in line or '---' in line or ':-' in line:
                in_table = True
                continue

            if in_table and '|' in line:
                # 解析表格行
                parts = [p.strip() for p in line.split('|')]
                # 过滤空字符串
                parts = [p for p in parts if p]

                if len(parts) >= 10:  # 确保有足够的列（10列）
                    group = parts[0]        # 第1列：Group
                    # parts[1]: ppo epoch
                    # parts[2]: Strategy
                    # parts[3]: k_pos
                    # parts[4]: λ_pos
                    # parts[5]: Peak
                    # parts[6]: k_neg
                    # parts[7]: λ_neg
                    machine = parts[8]      # 第9列：运行机器
                    status = parts[9]       # 第10列：运行状态
                    description = parts[10] if len(parts) > 10 else ''  # 第11列：说明（如果存在）

                    # 只保存用户填写过的数据（非空且不是占位符）
                    if machine and machine not in ['-', '待填写', '']:
                        if group not in user_data:
                            user_data[group] = {}
                        user_data[group]['machine'] = machine

                    if status and status not in ['-', '待运行', '']:
                        if group not in user_data:
                            user_data[group] = {}
                        user_data[group]['status'] = status

                    if description and description not in ['-', '']:
                        if group not in user_data:
                            user_data[group] = {}
                        user_data[group]['description'] = description
    except Exception as e:
        print(f"警告: 解析现有 markdown 文件时出错: {e}")

    return user_data


def generate_markdown_table(output_path, user_data):
    """生成实验配置的 markdown 表格，保留用户填写的数据"""
    markdown_file = output_path / "experiment_tracking.md"

    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write("# 实验配置追踪表\n\n")
        f.write("本文件用于追踪所有实验配置的参数和运行状态。\n\n")
        f.write("> **注意**: 请只修改「运行机器」、「运行状态」和「说明」列，其他列会在重新生成时自动更新。\n\n")

        # 写入表头
        f.write("| Group | ppo epoch  | Strategy | k_pos | λ_pos | Peak | k_neg | λ_neg | 运行机器 | 运行状态 | 说明 |\n")
        f.write("|:------|:------|:---------|------:|------:|-----:|------:|------:|:---------|:---------|:-----|\n")

        # 写入数据行
        for group, strategy, k_pos, lam_pos, peak, k_neg, lam_neg, ppo_epoch, desc in EXPERIMENT_CONFIGS:
            # 获取用户之前填写的数据
            machine = user_data.get(group, {}).get('machine', '-')
            status = user_data.get(group, {}).get('status', '-')
            # 如果用户有填写说明就用用户的，否则使用原始配置的说明
            user_description = user_data.get(group, {}).get('description', '')
            description = user_description if user_description and user_description != '-' else desc

            f.write(f"| {group} | {ppo_epoch} | {strategy} | {k_pos} | {lam_pos} | {peak} | {k_neg} | {lam_neg} | {machine} | {status} | {description} |\n")

        f.write("\n---\n\n")
        f.write("## 运行状态说明\n\n")
        f.write("- `-`: 未设置\n")
        f.write("- `待运行`: 等待执行\n")
        f.write("- `运行中`: 正在执行\n")
        f.write("- `已完成`: 执行完成\n")
        f.write("- `失败`: 执行失败\n")
        f.write("- `暂停`: 暂时停止\n\n")

        f.write("## 配置参考（仅供参考）\n\n")
        f.write("以下是各组配置的原始说明，供参考使用：\n\n")
        f.write("| Group | 原始说明 |\n")
        f.write("|:------|:---------|\n")
        for group, strategy, k_pos, lam_pos, peak, k_neg, lam_neg, ppo_epoch, desc in EXPERIMENT_CONFIGS:
            f.write(f"| {group} | {desc} |\n")

        f.write("\n## 快速命令参考\n\n")
        f.write("```bash\n")
        for group, strategy, k_pos, lam_pos, peak, k_neg, lam_neg, ppo_epoch, desc in EXPERIMENT_CONFIGS:
            filename = f"gamma_is_ppo_epochs_{ppo_epoch}_{k_pos}_{lam_pos}_{k_neg}_{lam_neg}_{group}.sh"
            f.write(f"# {group} - {desc}\n")
            f.write(f"bash recipe/is_shape/run/qwen3-4b-base/gamma/{filename}\n\n")
        f.write("```\n")

    return markdown_file


def generate_scripts(output_dir="."):
    """生成所有实验配置脚本"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 解析现有的 markdown 文件，保留用户填写的数据
    markdown_path = output_path / "experiment_tracking.md"
    user_data = parse_existing_markdown(markdown_path)
    if user_data:
        print(f"✓ 已读取现有追踪数据，保留 {len(user_data)} 个实验的用户填写信息")

    generated_files = []

    print("开始生成实验脚本...")
    print("=" * 80)

    for group, strategy, k_pos, lam_pos, peak, k_neg, lam_neg, ppo_epoch, desc in EXPERIMENT_CONFIGS:
        # 生成文件名
        filename = f"gamma_is_ppo_epochs_{ppo_epoch}_{k_pos}_{lam_pos}_{k_neg}_{lam_neg}_{group}.sh"
        filepath = output_path / filename

        # 填充模板
        script_content = SCRIPT_TEMPLATE.format(
            group=group,
            strategy=strategy,
            description=desc,
            k_pos=k_pos,
            lam_pos=lam_pos,
            peak=peak,
            k_neg=k_neg,
            lam_neg=lam_neg,
            ppo_epoch=ppo_epoch
        )

        # 写入文件
        with open(filepath, 'w') as f:
            f.write(script_content)

        # 添加执行权限
        os.chmod(filepath, 0o755)

        generated_files.append(filename)
        print(f"✓ 生成: {filename}")
        print(f"  [{group}] {strategy}: {desc}")
        print(f"  参数: k_pos={k_pos}, lam_pos={lam_pos}, peak={peak}, k_neg={k_neg}, lam_neg={lam_neg}")
        print("-" * 80)

    print("=" * 80)
    print(f"完成！共生成 {len(generated_files)} 个脚本文件")
    print(f"输出目录: {output_path.absolute()}")

    # 生成批量运行脚本
    batch_script = output_path / "run_all_experiments.sh"
    with open(batch_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# 批量运行所有实验配置\n")
        f.write("# 可以直接复制命令到终端执行\n\n")
        for filename in generated_files:
            f.write(f"# {filename}\n")
            f.write(f"bash recipe/is_shape/run/qwen3-4b-base/gamma/{filename}\n\n")

    os.chmod(batch_script, 0o755)
    print(f"\n批量运行脚本: {batch_script}")
    print("(默认注释掉所有命令，请根据需要取消注释)")

    # 生成配置总结
    summary_file = output_path / "experiment_configs_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("实验配置总结\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'组别':<8} {'策略':<10} {'k_pos':<8} {'lam_pos':<10} {'peak':<8} {'k_neg':<8} {'lam_neg':<10} {'说明':<30}\n")
        f.write("-" * 100 + "\n")
        for group, strategy, k_pos, lam_pos, peak, k_neg, lam_neg, ppo_epoch, desc in EXPERIMENT_CONFIGS:
            if group:
                f.write(f"{group:<8} {strategy:<10} {k_pos:<8} {lam_pos:<10.2f} {peak:<8} {k_neg:<8} {lam_neg:<10.2f} {desc:<30}\n")

    print(f"配置总结文件: {summary_file}")

    # 生成实验追踪 markdown 表格
    markdown_file = generate_markdown_table(output_path, user_data)
    print(f"实验追踪文件: {markdown_file}")
    print("  ✓ 用户填写的「运行机器」、「运行状态」和「说明」已保留")

    return generated_files


if __name__ == "__main__":
    import sys

    # 可以通过命令行参数指定输出目录
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print("=" * 80)
    print("实验脚本生成器")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"配置数量: {len(EXPERIMENT_CONFIGS)}")
    print("=" * 80)
    print()

    generated_files = generate_scripts(output_dir)

    print("\n" + "=" * 80)
    print("使用说明:")
    print("=" * 80)
    print("1. 直接运行单个脚本:")
    print("   bash gamma_is_ppo_epochs_16_2.0_2.0_4.0_2.5_G5.sh")
    print()
    print("2. 或编辑 run_all_experiments.sh 批量运行")
    print()
    print("3. 查看配置总结: cat experiment_configs_summary.txt")
    print("=" * 80)
