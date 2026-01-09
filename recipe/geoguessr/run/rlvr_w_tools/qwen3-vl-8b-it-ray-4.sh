#!/bin/bash
set -x
ENGINE=${1:-vllm}

export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=COLL

# Generate a valid 21-character SwanLab run_id (lowercase letters and digits only)
# Or comment out these lines to let SwanLab auto-generate a new run_id each time
#RANDOM_SUFFIX=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 13 | head -n 1)
#export SWANLAB_RUN_ID="toolqw3${RANDOM_SUFFIX}"
# export SWANLAB_RESUME=must  # Uncomment to resume from existing run

source ./recipe/geoguessr/.env

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# directory
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
DATA_DIR=${GEOGUESSR_DIR}/verl_data/plain_rlvr

# Algorithm and experiment configuration
ALGO="plain_rlvr_grpo_tools"
OBJECT="geoguessr_reward_official"
DATASET="full"

MODEL_FULL="/mnt/tidal-alsh-hilab/usr/shenguobin/modelscope/Qwen/Qwen3-VL-8B-Instruct"
MODEL="qwen3_vl_8b-it-tools-all"

PROJECT_NAME="verl_grpo_geoguessr"
EXP_NAME="${ALGO}_${OBJECT}_${DATASET}_${MODEL}_${ENGINE}"

max_prompt_length=4096
max_response_length=12288  # 12k = 12 * 1024 = 12288

# Expand wildcards and create Hydra list format [file1,file2,...]
TRAIN_FILES="[$(ls $DATA_DIR/osv5m_train_chunk_*.parquet $DATA_DIR/geochain_test_chunk_*.parquet $DATA_DIR/gaea_train_chunk_*.parquet 2>/dev/null | tr '\n' ',' | sed 's/,$//')]"
#TRAIN_FILES="[$(ls $DATA_DIR/osv5m_train_chunk_000{0..5}.parquet $DATA_DIR/geochain_test_chunk_000{0..5}.parquet $DATA_DIR/gaea_train_chunk_000{0..5}.parquet 2>/dev/null | tr '\n' ',' | sed 's/,$//')]"
VAL_FILES="[$DATA_DIR/geochain_mini_test_chunk_0000.parquet,$DATA_DIR/gaea_bench_chunk_0000.parquet]"

# System prompt for tool-enabled training
CUSTOM_SYSTEM_PROMPT=${CUSTOM_SYSTEM_PROMPT:-"Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools.\n\nCRITICAL: Before taking ANY action (calling a tool or providing a final answer), you MUST first show your thinking process inside <think></think> tags. This is mandatory for every step.\n\nFollow this structured thinking process:\n\nStart an iterative loop for each question:\n\n1. **Think first (REQUIRED):** Inside <think></think> tags, analyze what you see in the image, what the user is asking, what you already know, and what you need to research. Be explicit about your reasoning.\n\n2. **Then act:** Based on your thinking, either:\n   - Use a research tool to find information you identified as needed\n   - Provide your final synthesized answer if research is complete\n\n3. **After each tool use:** Return to step 1 - think about what the tool revealed, analyze the findings, and decide your next action.\n\nContinue this loop until your research is complete.\n\nExample structure:\n<think>\n[Analyze the image in detail relative to the question]\n[Identify what you know vs. what you need to look up]\n[Plan your next action]\n</think>\n\n[Then either call a tool OR provide final answer]\n\nTo finish, after your final thinking step, provide a clear, synthesized answer that fully responds to the user's question.\n\nRemember: NEVER call a tool or provide an answer without first showing your thinking in <think> tags."}


# ============ Ray Configuration ============
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/recipe/geoguessr/run/runtime_env.yaml"}
NNODES=${NNODES:-4}  # 4 nodes


echo "=========================================="
echo "Tool-Enabled Training Configuration"
echo "=========================================="
echo "Model: $MODEL_FULL"
echo "Algorithm: $ALGO"
#echo "Tool Config: recipe/geoguessr/config/image_zoom_tool_config.yaml"
echo "System Prompt: $CUSTOM_SYSTEM_PROMPT"
echo "Engine: $ENGINE"
echo "=========================================="

# Note: Using GeoguessrToolDataset for tool support
# Note: reward_manager=geoguessr is automatically registered when custom_reward_function is loaded
# Note: custom_chat_template is loaded from config/geoguessr_ppo_with_tools.yaml


ray job submit --no-wait \
    --runtime-env="${RUNTIME_ENV}" \
    --address "${RAY_ADDRESS}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    --config-path "${WORKING_DIR}/recipe/geoguessr/config" \
    --config-name geoguessr_ppo_with_tools \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=recipe/geoguessr/reward_function.py \
    custom_reward_function.name=$OBJECT \
    data.custom_cls.name=GeoguessrToolDataset \
    "data.custom_system_prompt=\"$CUSTOM_SYSTEM_PROMPT\"" \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=512 \
    data.val_batch_size=2048 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.prompt_length=$max_prompt_length \
    actor_rollout_ref.rollout.response_length=$max_response_length \
    actor_rollout_ref.model.path=$MODEL_FULL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=30 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=8 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/geoguessr/config/geoguessr_all_tools_config.yaml \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    algorithm.use_kl_in_reward=False \
    reward_model.launch_reward_fn_async=True \
    reward_model.reward_manager=geoguessr \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.default_local_dir=/mnt/tidal-alsh-hilab/usr/shenguobin/verl/checkpoints/$PROJECT_NAME/$EXP_NAME \
    trainer.validation_data_dir=/mnt/tidal-alsh-hilab/usr/shenguobin/verl/validation_outputs/$PROJECT_NAME/$EXP_NAME/$TIMESTAMP \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    $@
