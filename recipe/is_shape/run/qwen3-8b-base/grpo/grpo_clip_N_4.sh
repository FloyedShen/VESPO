#!/bin/bash
set -x
ENGINE=${1:-vllm}

export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
# ============================================================================
# GRPO with Clip: batch_size=256, ppo_epochs=16
# Data Reuse: Each batch is trained 16 times
# ============================================================================

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

ALGO="dapo_clip_new"
OBJECT="math"
DATASET="dapo_math"
MODEL_FULL="/mnt/tidal-alsh-hilab/usr/shenguobin/modelscope/Qwen/Qwen3-8B-Base"
MODEL="qwen3-8b-base"
BATCH_SIZE=1024
MINI_BATCH=256

N=$((BATCH_SIZE / MINI_BATCH))

PROJECT_NAME="is_shape_experiments"
EXP_NAME="${ALGO}_${OBJECT}_${DATASET}_${MODEL}_N_${N}"

# Data Configuration
DATA_ROOT="/mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/is_shape/data"
train_files="${DATA_ROOT}/${DATASET}/train.parquet"
test_files="[${DATA_ROOT}/amc23/test.parquet,${DATA_ROOT}/aime25/test.parquet,${DATA_ROOT}/aime_2024/test.parquet]"

# Sequence Length Configuration
max_prompt_length=1024
max_response_length=15360

# Launch Training
# Using recipe.is_shape.code.main_ppo

# ============ Ray Configuration ============
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/recipe/is_shape/run/runtime_env.yaml"}
NNODES=${NNODES:-4}  # 4 nodes


ray job submit --no-wait \
    --runtime-env="${RUNTIME_ENV}" \
    --address "${RAY_ADDRESS}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -u -m recipe.is_shape.code.main_ppo \
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
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
    +actor_rollout_ref.actor.global_batch_info.dp_size=1 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
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
    trainer.nnodes=$NNODES \
    trainer.val_before_train=False \
    trainer.default_local_dir=/mnt/tidal-alsh-hilab/usr/shenguobin/verl/checkpoints/$PROJECT_NAME/$EXP_NAME \
    trainer.validation_data_dir=/mnt/tidal-alsh-hilab/usr/shenguobin/verl/validation_outputs/$PROJECT_NAME/$EXP_NAME/$TIMESTAMP \
    trainer.save_freq=32 \
    trainer.test_freq=8 \
    trainer.total_epochs=1 \
    $@
