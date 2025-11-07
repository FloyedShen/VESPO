#!/bin/bash
set -x
ENGINE=${1:-vllm}
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

# VLLM version >= 0.11.0 for qwen3-vl support
# pip install -U git+https://github.com/ISEEKYAN/mbridge.git # for latest mbridge
# pip install -U transformers # for qwen3-vl support
# pip install --no-deps --no-cache-dir git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1

export VLLM_ALLREDUCE_USE_SYMM_MEM=0 # for vllm0.11.0 with TP

# Data directory
DATA_DIR=${GEOGUESSR_DIR}/verl_data/plain_rlvr

# Configuration
ALGO="plain_rlvr_grpo"
OBJECT="geoguessr_reward_official"
DATASET="mix_3x5"

MODEL_FULL="Qwen/Qwen3-VL-8B-Instruct"
MODEL="qwen3_vl_8b-it-megatron"

PROJECT_NAME="verl_grpo_geoguessr"
EXP_NAME="${ALGO}_${OBJECT}_${DATASET}_${MODEL}_megatron_${ENGINE}"

# Prepare data files
TRAIN_FILES="[$(ls $DATA_DIR/osv5m_train_chunk_000{0..5}.parquet $DATA_DIR/geochain_test_chunk_000{0..5}.parquet $DATA_DIR/gaea_train_chunk_000{0..5}.parquet 2>/dev/null | tr '\n' ',' | sed 's/,$//')]"
VAL_FILES="[$DATA_DIR/geochain_mini_test_chunk_0000.parquet,$DATA_DIR/gaea_bench_chunk_0000.parquet,$DATA_DIR/yfcc4k_train_chunk_0000.parquet,$DATA_DIR/im2gps3k_train_chunk_0000.parquet]"

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=recipe/geoguessr/reward_function.py \
    custom_reward_function.name=$OBJECT \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$MODEL_FULL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard","swanlab"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@