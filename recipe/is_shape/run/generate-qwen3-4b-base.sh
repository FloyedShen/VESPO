#!/bin/bash

# Script to generate all experiment scripts
#
# Experiment Design:
# - train_batch_size = ppo_mini_batch_size = 512 (fixed)
# - ppo_epochs = 1 / 4 / 16 (controls data reuse)
# - This allows fair comparison of how well each algorithm "squeezes" the same data

OUTPUT_DIR="qwen3-4b-base"
mkdir -p "$OUTPUT_DIR"

# Common configuration
MODEL_FULL="/mnt/tidal-alsh-hilab/usr/shenguobin/modelscope/Qwen/Qwen3-4B-Base"
MODEL="qwen3-4b-base"
DATA_ROOT="/mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/is_shape/data"
TRAIN_FILES="\${DATA_ROOT}/deepscaler/train.parquet"
TEST_FILES="[\${DATA_ROOT}/amc23/test.parquet,\${DATA_ROOT}/aime25/test.parquet,\${DATA_ROOT}/aime_2024/test.parquet]"

# Experimental groups
# - All algorithms use the same batch structure
# - ppo_epochs controls how many times we reuse the same data
declare -a ALGOS=("grpo_clip" "grpo_noclip" "is_reshape" "is_reshape_renyi")
declare -a PPO_EPOCHS=(4 16 64)
BATCH_SIZE=256
MINI_BATCH=$BATCH_SIZE  # Same as batch_size for fair comparison

for ALGO in "${ALGOS[@]}"; do
    for PPO_EPOCH in "${PPO_EPOCHS[@]}"; do

        SCRIPT_NAME="${ALGO}_ppo_epochs_${PPO_EPOCH}.sh"
        SCRIPT_PATH="${OUTPUT_DIR}/${SCRIPT_NAME}"

        echo "Generating: $SCRIPT_NAME (batch_size=${BATCH_SIZE}, mini_batch=${MINI_BATCH}, ppo_epochs=${PPO_EPOCH})"

        cat > "$SCRIPT_PATH" << 'EOFSCRIPT'
#!/bin/bash
set -x

# ============================================================================
# ALGO_NAME: batch_size=BATCH_SIZE_VALUE, ppo_epochs=PPO_EPOCHS_VALUE
# Data Reuse: Each batch is trained PPO_EPOCHS_VALUE times
# ============================================================================

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

ALGO="ALGO_TAG"
OBJECT="math"
DATASET="deepscaler"
MODEL_FULL="MODEL_FULL_NAME"
MODEL="MODEL_NAME"
BATCH_SIZE=BATCH_SIZE_VALUE
MINI_BATCH=MINI_BATCH_VALUE
PPO_EPOCHS=PPO_EPOCHS_VALUE

PROJECT_NAME="is_shape_experiments"
EXP_NAME="${ALGO}_${OBJECT}_${DATASET}_${MODEL}_ppo_epochs_${PPO_EPOCHS}"

# Data Configuration
DATA_ROOT="DATA_ROOT_PATH"
train_files="TRAIN_FILES_PATH"
test_files="TEST_FILES_PATH"

# Sequence Length Configuration
max_prompt_length=1536
max_response_length=14848

# Launch Training
MAIN_MODULE_VAR

python3 -u -m MAIN_MODULE_EXEC \
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
CLIP_CONFIG
POLICY_LOSS_CONFIG
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
EOFSCRIPT

        # Replace placeholders
        sed -i "s|ALGO_TAG|${ALGO}|g" "$SCRIPT_PATH"
        sed -i "s|BATCH_SIZE_VALUE|${BATCH_SIZE}|g" "$SCRIPT_PATH"
        sed -i "s|MINI_BATCH_VALUE|${MINI_BATCH}|g" "$SCRIPT_PATH"
        sed -i "s|PPO_EPOCHS_VALUE|${PPO_EPOCH}|g" "$SCRIPT_PATH"
        sed -i "s|MODEL_FULL_NAME|${MODEL_FULL}|g" "$SCRIPT_PATH"
        sed -i "s|MODEL_NAME|${MODEL}|g" "$SCRIPT_PATH"
        sed -i "s|DATA_ROOT_PATH|${DATA_ROOT}|g" "$SCRIPT_PATH"
        sed -i "s|TRAIN_FILES_PATH|${TRAIN_FILES}|g" "$SCRIPT_PATH"
        sed -i "s|TEST_FILES_PATH|${TEST_FILES}|g" "$SCRIPT_PATH"

        # Configure algorithm-specific settings
        case "$ALGO" in
            "grpo_clip")
                sed -i "s|ALGO_NAME|GRPO with Clip|g" "$SCRIPT_PATH"
                # Use custom main_ppo for consistent per-epoch logging across all algorithms
                sed -i "s|MAIN_MODULE_VAR|# Using recipe.is_shape.code.main_ppo|g" "$SCRIPT_PATH"
                sed -i "s|MAIN_MODULE_EXEC|recipe.is_shape.code.main_ppo|g" "$SCRIPT_PATH"
                sed -i "s|CLIP_CONFIG|    actor_rollout_ref.actor.clip_ratio_low=0.2 \\\\\n    actor_rollout_ref.actor.clip_ratio_high=0.2 \\\\\n    actor_rollout_ref.actor.clip_ratio_c=100000 \\\\|g" "$SCRIPT_PATH"
                sed -i "/POLICY_LOSS_CONFIG/d" "$SCRIPT_PATH"
                ;;
            "grpo_noclip")
                sed -i "s|ALGO_NAME|GRPO w/o Clip|g" "$SCRIPT_PATH"
                # Use custom main_ppo for consistent per-epoch logging across all algorithms
                sed -i "s|MAIN_MODULE_VAR|# Using recipe.is_shape.code.main_ppo|g" "$SCRIPT_PATH"
                sed -i "s|MAIN_MODULE_EXEC|recipe.is_shape.code.main_ppo|g" "$SCRIPT_PATH"
                sed -i "s|CLIP_CONFIG|    actor_rollout_ref.actor.clip_ratio=500000000 \\\\\n    actor_rollout_ref.actor.clip_ratio_low=500000000 \\\\\n    actor_rollout_ref.actor.clip_ratio_high=500000000 \\\\\n    actor_rollout_ref.actor.clip_ratio_c=500000000 \\\\|g" "$SCRIPT_PATH"
                sed -i "/POLICY_LOSS_CONFIG/d" "$SCRIPT_PATH"
                ;;
            "is_reshape")
                sed -i "s|ALGO_NAME|IS Reshape|g" "$SCRIPT_PATH"
                # Use custom main_ppo for IS Reshape to track per-epoch metrics
                sed -i "s|MAIN_MODULE_VAR|# Using recipe.is_shape.code.main_ppo|g" "$SCRIPT_PATH"
                sed -i "s|MAIN_MODULE_EXEC|recipe.is_shape.code.main_ppo|g" "$SCRIPT_PATH"
                sed -i "/CLIP_CONFIG/d" "$SCRIPT_PATH"
                sed -i "s|POLICY_LOSS_CONFIG|    actor_rollout_ref.actor.policy_loss.loss_mode=is_reshape \\\\\n    +actor_rollout_ref.actor.policy_loss.is_reshape.rho_min=0.3 \\\\\n    +actor_rollout_ref.actor.policy_loss.is_reshape.gamma_min=0.05 \\\\\n    +actor_rollout_ref.actor.policy_loss.is_reshape.gamma_max=1.0 \\\\\n    +actor_rollout_ref.actor.policy_loss.is_reshape.gamma=null \\\\\n    +actor_rollout_ref.actor.policy_loss.is_reshape.clip_weight=true \\\\\n    +actor_rollout_ref.actor.policy_loss.is_reshape.clip_threshold=10.0 \\\\|g" "$SCRIPT_PATH"
                ;;
            "is_reshape_renyi")
                sed -i "s|ALGO_NAME|IS Reshape Rényi|g" "$SCRIPT_PATH"
                # Use custom main_ppo for IS Reshape Rényi to track per-epoch metrics
                sed -i "s|MAIN_MODULE_VAR|# Using recipe.is_shape.code.main_ppo|g" "$SCRIPT_PATH"
                sed -i "s|MAIN_MODULE_EXEC|recipe.is_shape.code.main_ppo|g" "$SCRIPT_PATH"
                sed -i "/CLIP_CONFIG/d" "$SCRIPT_PATH"
                sed -i "s|POLICY_LOSS_CONFIG|    actor_rollout_ref.actor.policy_loss.loss_mode=is_reshape_renyi \\\\\n    +actor_rollout_ref.actor.policy_loss.is_reshape_renyi.rho_min=0.3 \\\\\n    +actor_rollout_ref.actor.policy_loss.is_reshape_renyi.tau=1.0 \\\\\n    +actor_rollout_ref.actor.policy_loss.is_reshape_renyi.T=5.0 \\\\\n    +actor_rollout_ref.actor.policy_loss.is_reshape_renyi.clip_weight=true \\\\\n    +actor_rollout_ref.actor.policy_loss.is_reshape_renyi.clip_threshold=10.0 \\\\|g" "$SCRIPT_PATH"
                ;;
        esac

        chmod +x "$SCRIPT_PATH"
    done
done

echo ""
echo "============================================================================"
echo "Generated 20 experiment scripts in $OUTPUT_DIR/"
echo "============================================================================"
echo ""
echo "Experiment Design:"
echo "  - train_batch_size = ppo_mini_batch_size = ${BATCH_SIZE} (fixed)"
echo "  - ppo_epochs = ${PPO_EPOCHS[@]} (controls data reuse)"
echo ""
echo "This design ensures:"
echo "  - Same data per global_step across all experiments"
echo "  - ppo_epochs controls how many times each batch is trained"
echo "  - Fair comparison of data efficiency between algorithms"
echo ""
echo "Scripts:"
ls -1 "$OUTPUT_DIR"/*.sh | grep -E "(grpo_clip|grpo_noclip|is_reshape)_ppo_epochs"
