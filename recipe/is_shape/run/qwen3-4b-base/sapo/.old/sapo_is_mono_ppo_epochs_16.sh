#!/bin/bash
set -x

# ============================================================================
# SAPO-IS-Mono: SAPO with Monotonically Increasing IS-Reshape Gates
#
# Fixes the monotonicity issue in sapo_is while maintaining:
#   - Concave behavior for A > 0 (amplify new good, reduce known good)
#   - Convex behavior for A < 0 (reduce avoided bad, amplify un-avoided bad)
#
# Gate functions:
#   A > 0: g_pos(w) = (1 + τ) * w / (τ + w)     [Michaelis-Menten, concave]
#   A < 0: g_neg(w) = (1 - exp(-τw²/2)) / norm  [Gaussian CDF-like, convex]
#
# Properties:
#   - Both monotonically increasing ✓ (valid IS)
#   - Both bounded ✓ (stable training)
#   - Both pass through (1, 1) ✓ (normalized)
# ============================================================================

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

ALGO="sapo_is_mono"
OBJECT="math"
DATASET="deepscaler"
MODEL_FULL="/mnt/tidal-alsh-hilab/usr/shenguobin/modelscope/Qwen/Qwen3-4B-Base"
MODEL="qwen3-4b-base"
BATCH_SIZE=256
MINI_BATCH=256
PPO_EPOCHS=16

PROJECT_NAME="is_shape_experiments"
EXP_NAME="${ALGO}_${OBJECT}_${DATASET}_${MODEL}_ppo_epochs_${PPO_EPOCHS}"

# Data Configuration
DATA_ROOT="/mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/is_shape/data"
train_files="${DATA_ROOT}/deepscaler/train.parquet"
test_files="[${DATA_ROOT}/amc23/test.parquet,${DATA_ROOT}/aime25/test.parquet,${DATA_ROOT}/aime_2024/test.parquet]"

# Sequence Length Configuration
max_prompt_length=1536
max_response_length=14848

# Launch Training
python3 -u -m recipe.is_shape.code.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    +data.stale_iteration=0 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.seed=42 \
    actor_rollout_ref.model.path=$MODEL_FULL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    +actor_rollout_ref.actor.global_batch_info.dp_size=1 \
    actor_rollout_ref.actor.policy_loss.loss_mode=sapo_is_mono \
    +actor_rollout_ref.actor.policy_loss.sapo_is_mono.tau_pos=1.0 \
    +actor_rollout_ref.actor.policy_loss.sapo_is_mono.tau_neg=1.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=recipe/is_shape/code/reward_function.py \
    custom_reward_function.name=$OBJECT \
    reward_model.reward_manager=prime \
    reward_model.launch_reward_fn_async=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.default_local_dir=/mnt/tidal-alsh-hilab/usr/shenguobin/verl/checkpoints/$PROJECT_NAME/$EXP_NAME \
    trainer.validation_data_dir=/mnt/tidal-alsh-hilab/usr/shenguobin/verl/validation_outputs/$PROJECT_NAME/$EXP_NAME/$TIMESTAMP \
    trainer.save_freq=256 \
    trainer.test_freq=16 \
    trainer.total_epochs=3 \
    $@
