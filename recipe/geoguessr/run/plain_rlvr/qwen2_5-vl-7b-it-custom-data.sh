#!/bin/bash
set -x
ENGINE=${1:-vllm}

# directory
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
DATA_DIR=${GEOGUESSR_DIR}/verl_data/plain_rlvr

# Configuration for custom dataset
# Set USE_CUSTOM_DATASET=true to enable custom dataset with configurable prompts
USE_CUSTOM_DATASET=${USE_CUSTOM_DATASET:-true}

# Custom prompts (only used when USE_CUSTOM_DATASET=true)
# These can be overridden by environment variables
CUSTOM_SYSTEM_PROMPT=${CUSTOM_SYSTEM_PROMPT:-"You are a helpful assistant. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags."}
CUSTOM_USER_PROMPT_TEMPLATE=${CUSTOM_USER_PROMPT_TEMPLATE:-"{content}"}

ALGO="plain_rlvr_grpo"
OBJECT="geoguessr_reward_official"
DATASET="mix_3x5"

MODEL_FULL="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL="qwen2_5_vl_7b-it-tk-custom_rm"

PROJECT_NAME="verl_grpo_geoguessr"
EXP_NAME="${ALGO}_${OBJECT}_${DATASET}_${MODEL}_${ENGINE}"

max_prompt_length=3072
max_response_length=4096

# Expand wildcards and create Hydra list format [file1,file2,...]
TRAIN_FILES="[$(ls $DATA_DIR/osv5m_train_chunk_*.parquet $DATA_DIR/geochain_test_chunk_*.parquet $DATA_DIR/gaea_train_chunk_*.parquet 2>/dev/null | tr '\n' ',' | sed 's/,$//')]"
#TRAIN_FILES="[$(ls $DATA_DIR/osv5m_train_chunk_000{0..5}.parquet $DATA_DIR/geochain_test_chunk_000{0..5}.parquet $DATA_DIR/gaea_train_chunk_000{0..5}.parquet 2>/dev/null | tr '\n' ',' | sed 's/,$//')]"
VAL_FILES="[$DATA_DIR/geochain_mini_test_chunk_0000.parquet,$DATA_DIR/gaea_bench_chunk_0000.parquet,$DATA_DIR/yfcc4k_train_chunk_0000.parquet,$DATA_DIR/im2gps3k_train_chunk_0000.parquet]"
# ,$DATA_DIR/yfcc4k_train_chunk_0000.parquet,$DATA_DIR/im2gps3k_train_chunk_0000.parquet

# Build custom dataset arguments as array
CUSTOM_DATASET_ARGS=()
if [ "$USE_CUSTOM_DATASET" = "true" ]; then
    CUSTOM_DATASET_ARGS+=("data.custom_cls.name=GeoguessrRLHFDataset")
    CUSTOM_DATASET_ARGS+=("data.custom_cls.path=\"$PROJECT_DIR/recipe/geoguessr/geoguessr_dataset.py\"")
    # Escape the prompts properly for Hydra by wrapping in quotes
    CUSTOM_DATASET_ARGS+=("+data.custom_system_prompt=\"$CUSTOM_SYSTEM_PROMPT\"")
    CUSTOM_DATASET_ARGS+=("+data.custom_user_prompt_template=\"$CUSTOM_USER_PROMPT_TEMPLATE\"")
    echo "Using custom dataset with:"
    echo "  System prompt: $CUSTOM_SYSTEM_PROMPT"
    echo "  User prompt template: $CUSTOM_USER_PROMPT_TEMPLATE"
else
    echo "Using default RLHFDataset"
fi

# Note: reward_manager=geoguessr is automatically registered when custom_reward_function is loaded

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=recipe/geoguessr/reward_function.py \
    custom_reward_function.name=$OBJECT \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=1024 \
    data.val_batch_size=4096 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.path=$MODEL_FULL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.launch_reward_fn_async=True \
    reward_model.reward_manager=geoguessr \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard", "swanlab"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    "${CUSTOM_DATASET_ARGS[@]}" \
    $@
