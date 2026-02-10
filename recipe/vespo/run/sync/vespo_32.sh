#!/bin/bash
set -x
ENGINE=${1:-vllm}

#export WANDB_RUN_ID=""
#export WANDB_RESUME="must"  # or "must" or "never"

export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1

# ============================================================================
# IS Reshape with Gamma-IS Loss for Qwen3-30B-A3B (Megatron Backend)
#
# This script runs IS Reshape training with Megatron for MoE models like
# Qwen3-30B-A3B that require expert parallelism.
#
# Key differences from FSDP:
# - Uses Megatron for training (actor.strategy=megatron)
# - Supports expert parallelism (EP=8 for Qwen3-30B-A3B)
# - Uses mbridge for HF->Megatron config conversion
# - Uses ISReshapeMegatronActorRolloutRefWorker for per-update metrics
#
# Experiment Configuration: Gamma-IS with k_pos, lambda_pos, k_neg, lambda_neg
# ============================================================================

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

ALGO="vespo"
OBJECT="math"
DATASET="dapo_math"


MODEL_FULL="path/to/model/Qwen/Qwen3-30B-A3B-Base"
MODEL="qwen3-30b-a3b"

# Data Configuration
DATA_ROOT="path/to/data"
TRAIN_DATA_PATH="${DATA_ROOT}/${DATASET}/train.parquet"
TEST_DATA_PATH="[${DATA_ROOT}/amc23/test.parquet,${DATA_ROOT}/aime25/test.parquet,${DATA_ROOT}/aime_2024/test.parquet]"

# Parallelism Configuration
PP=1
VPP=None
TP=2
EP=8
ETP=1
VLLM_INFER_TP=2

# Memory Configuration
offload=True
gpu_memory_utilization=0.75

# Batch Size Configuration
BATCH_SIZE=8192
MINI_BATCH=256
use_dynamic_bsz=True
micro_bs=2

N=$((BATCH_SIZE / MINI_BATCH))

# Sequence Length Configuration
max_prompt_length=1024
max_response_length=15360
actor_ppo_max_token_len=16384
infer_ppo_max_token_len=16384

# R2: enable routing replay
# R3: enable rollout routing replay
# If enabling R3, please set actor_rollout_ref.rollout.enable_rollout_routing_replay=True
# R3 example is based on vllm related pr https://github.com/vllm-project/vllm/pull/5322
ROUTING_REPLAY_MODE="disabled"

# VESPO parameters
K_POS=${K_POS:-2.0}
LAMBDA_POS=${LAMBDA_POS:-3.0}
K_NEG=${K_NEG:-3.0}
LAMBDA_NEG=${LAMBDA_NEG:-2.0}

PROJECT_NAME="vespo_experiments"
EXP_NAME="${ALGO}_${ROUTING_REPLAY_MODE}_${OBJECT}_${DATASET}_${MODEL}_${K_POS}_${LAMBDA_POS}_${K_NEG}_${LAMBDA_NEG}_N_${N}"


# ============ Ray Configuration ============
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/recipe/vespo/run/runtime_env.yaml"}
NNODES=${NNODES:-8}  # 4 nodes (32 GPUs total)
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ============ Megatron Parallelism Configuration ============
# For Qwen3-30B-A3B MoE model:
# - Expert Parallelism (EP) = 8 (matches the number of experts)
# - Tensor Parallelism for training = 1 (can increase for larger batch)
# - Tensor Parallelism for inference = 4 (vLLM rollout)

# ============ Dynamic Runtime Env for WANDB Resume ============
# Create a temporary runtime_env file to dynamically set WANDB_RUN_ID
TEMP_RUNTIME_ENV=$(mktemp /tmp/runtime_env_XXXXXX.yaml)
cp "${RUNTIME_ENV}" "${TEMP_RUNTIME_ENV}"

# Define a helper function to set or update key-value pairs under env_vars
set_env_var() {
    local file=$1
    local key=$2
    local value=$3

    # Check if the key exists (with two-space indentation)
    if grep -q "^  ${key}:" "${file}"; then
        # If it exists, replace the value
        sed -i "s/^  ${key}:.*/  ${key}: \"${value}\"/" "${file}"
    else
        # If it does not exist, append after env_vars:
        # First check if env_vars: exists in the file
        if grep -q "^env_vars:" "${file}"; then
            # Insert the new key-value pair after the env_vars: line
            sed -i "/^env_vars:/a\  ${key}: \"${value}\"" "${file}"
        else
            # If env_vars: does not exist either, add it first then the key-value pair
            echo "env_vars:" >> "${file}"
            echo "  ${key}: \"${value}\"" >> "${file}"
        fi
    fi
}

if [ -n "${WANDB_RUN_ID}" ]; then
    echo "[INFO] Resuming WANDB run: ${WANDB_RUN_ID} (resume=${WANDB_RESUME})"
    set_env_var "${TEMP_RUNTIME_ENV}" "WANDB_RUN_ID" "${WANDB_RUN_ID}"
    set_env_var "${TEMP_RUNTIME_ENV}" "WANDB_RESUME" "${WANDB_RESUME}"
else
    echo "[INFO] Starting new WANDB run"
    # Do nothing, or explicitly remove these keys
fi

# Use the modified temporary file
RUNTIME_ENV="${TEMP_RUNTIME_ENV}"


# Launch Training
# Using recipe.vespo.code.main_ppo with Megatron strategy
ray job submit --no-wait \
    --runtime-env="${RUNTIME_ENV}" \
    --address "${RAY_ADDRESS}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.vespo.code.main_ppo --config-path=../../../verl/trainer/config \
    --config-name=ppo_megatron_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$TEST_DATA_PATH \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    +data.stale_iteration=0 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.seed=42 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.path=$MODEL_FULL \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.router_replay.mode=${ROUTING_REPLAY_MODE} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_bs \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    +actor_rollout_ref.actor.global_batch_info.dp_size=1 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vespo \
    +actor_rollout_ref.actor.policy_loss.gamma_is.k_pos=$K_POS \
    +actor_rollout_ref.actor.policy_loss.gamma_is.lambda_pos=$LAMBDA_POS \
    +actor_rollout_ref.actor.policy_loss.gamma_is.k_neg=$K_NEG \
    +actor_rollout_ref.actor.policy_loss.gamma_is.lambda_neg=$LAMBDA_NEG \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_bs \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_INFER_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_bs \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=recipe/vespo/code/reward_function.py \
    custom_reward_function.name=$OBJECT \
    reward_model.reward_manager=prime_loop \
    reward_model.num_workers=16 \
    reward_model.launch_reward_fn_async=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name="$EXP_NAME" \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.default_local_dir=/path/to/checkpoints/$PROJECT_NAME/$EXP_NAME \
    trainer.save_freq=1 \
    trainer.test_freq=4 \
    trainer.total_epochs=1 \
    trainer.balance_batch=False 2>&1