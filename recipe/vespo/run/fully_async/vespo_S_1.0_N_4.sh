#!/usr/bin/env bash
set -xeuo pipefail

#export WANDB_RUN_ID=""
#export WANDB_RESUME="must"  # or "must" or "never"

PROJECT_NAME="vespo_experiments"
DATASET="dapo_math"
MODEL_NAME="qwen3-30b-a3b-base"
ALGO="vespo"
LOSS_MODE="vespo"

#RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
#MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-30B-A3B-Base"}
#CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
#TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
#TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

DATA_ROOT="/path/to/data"
TRAIN_FILE="${DATA_ROOT}/${DATASET}/train.parquet"
TEST_FILE="[${DATA_ROOT}/aime25/test.parquet,${DATA_ROOT}/aime_2024/test.parquet]"

MODEL_PATH="/path/to/model/Qwen/Qwen3-30B-A3B-Base"


rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi
# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# IS Reshape Gamma-IS parameters (for vespo loss mode)
# Uncomment to use vespo instead of default PPO clip loss
K_POS=${K_POS:-2.0}
LAMBDA_POS=${LAMBDA_POS:-3.0}
K_NEG=${K_NEG:-3.0}
LAMBDA_NEG=${LAMBDA_NEG:-2.0}
# Set LOSS_MODE to "vespo" to use Gamma-IS loss

# Response length parameters
max_prompt_length=$((1024))
max_response_length=$((1024 * 15))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=0.0

loss_agg_mode="token-mean"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
offload=True
train_ppo_micro_batch_size_per_gpu=2
infer_ppo_micro_batch_size_per_gpu=2

optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.}

COMMON_PP=${COMMON_PP:-1}
COMMON_VPP=${COMMON_VPP:-null}
COMMON_CP=${COMMON_CP:-1}
COMMON_TP=${COMMON_TP:-4}
COMMON_EP=${COMMON_EP:-8}
COMMON_ETP=${COMMON_ETP:-1}

TRAIN_TP=${TRAIN_TP:-$COMMON_TP}
INFER_TP=${INFER_TP:-4}

ACTOR_PP=${ACTOR_PP:-$COMMON_PP}
ACTOR_VPP=${ACTOR_VPP:-$COMMON_VPP}
ACTOR_CP=${ACTOR_CP:-$COMMON_CP}
ACTOR_TP=${ACTOR_TP:-$TRAIN_TP}
ACTOR_EP=${ACTOR_EP:-$COMMON_EP}
ACTOR_ETP=${ACTOR_ETP:-$COMMON_ETP}
ROLLOUT_TP=${ROLLOUT_TP:-$INFER_TP}
REF_PP=${REF_PP:-$COMMON_PP}
REF_VPP=${REF_VPP:-$COMMON_VPP}
REF_CP=${REF_CP:-$COMMON_CP}
REF_TP=${REF_TP:-$TRAIN_TP}
REF_EP=${REF_EP:-$COMMON_EP}
REF_ETP=${REF_ETP:-$COMMON_ETP}
CRITIC_PP=${CRITIC_PP:-$COMMON_PP}
CRITIC_VPP=${CRITIC_VPP:-$COMMON_VPP}
CRITIC_CP=${CRITIC_CP:-$COMMON_CP}
CRITIC_TP=${CRITIC_TP:-$TRAIN_TP}
CRITIC_EP=${CRITIC_EP:-$COMMON_EP}
CRITIC_ETP=${CRITIC_ETP:-$COMMON_ETP}
RM_PP=${RM_PP:-$COMMON_PP}
RM_VPP=${RM_VPP:-$COMMON_VPP}
RM_CP=${RM_CP:-$COMMON_CP}
RM_TP=${RM_TP:-$TRAIN_TP}
RM_EP=${RM_EP:-$COMMON_EP}
RM_ETP=${RM_ETP:-$COMMON_ETP}

# install mbridge
# pip3 install git+https://github.com/ISEEKYAN/mbridge
USE_MBRIDGE=True
USE_DIST_CKPT=False

# Fully async specific parameters
NNODES_ROLLOUT=${NNODES_ROLLOUT:-6}
NNODES_TRAIN=${NNODES_TRAIN:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

NGPUS_ROLLOUT=$((NNODES_ROLLOUT * NGPUS_PER_NODE))
NGPUS_TRAIN=$((NNODES_TRAIN * NGPUS_PER_NODE))

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=8
train_prompt_mini_bsz=256
total_rollout_steps=$(((55936 * 256)))
test_freq=20
staleness_threshold=1.0
trigger_parameter_sync_step=4
require_batches=1
partial_rollout=True

exp_name="${ALGO}_${MODEL_NAME}_stl-${staleness_threshold}_sync_step-${trigger_parameter_sync_step}-fully-async_${NGPUS_ROLLOUT}-${NGPUS_PER_NODE}"


# ============ Ray Configuration ============
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/recipe/vespo/run/runtime_env.yaml"}

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

if [ -n "${WANDB_RUN_ID:-}" ]; then
    echo "[INFO] Resuming WANDB run: ${WANDB_RUN_ID} (resume=${WANDB_RESUME:-allow})"
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
    -- python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml'\
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    +actor_rollout_ref.actor.policy_loss.vespo.k_pos=${K_POS} \
    +actor_rollout_ref.actor.policy_loss.vespo.lambda_pos=${LAMBDA_POS} \
    +actor_rollout_ref.actor.policy_loss.vespo.k_neg=${K_NEG} \
    +actor_rollout_ref.actor.policy_loss.vespo.lambda_neg=${LAMBDA_NEG} \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_decay_steps=1000000 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.lr_warmup_init=1e-6 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="flex" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${REF_ETP} \
    custom_reward_function.path=recipe/vespo/code/reward_function.py \
    custom_reward_function.name="math" \
    reward_model.reward_manager=prime_loop \
    reward_model.num_workers=32 \
    reward_model.launch_reward_fn_async=False \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.default_local_dir=/path/to/checkpoints/$PROJECT_NAME/$exp_name \
    trainer.save_freq="${test_freq}" \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_epochs=1 \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_rollout_log_probs=True \
