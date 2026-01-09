# GeoGuessr Tool-Enabled Training Scripts

This directory contains training scripts for GeoGuessr models with tool support (specifically image zoom-in capability).

## Scripts

### qwen3-vl-8b-tk.sh

Training script for Qwen3-VL-8B-Thinking model with tool support.

**Features**:
- Uses `GeoguessrToolDataset` with `ImageZoomInTool` support
- Loads config from `config/geoguessr_ppo_with_tools.yaml`
- Multi-turn dialogue with tool calling enabled
- GRPO algorithm with GeoGuessr reward function

**Usage**:

```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr

# Basic usage (uses default data from $GEOGUESSR_DIR/verl_data/plain_rlvr)
./run/rlvr_w_tools/qwen3-vl-8b-tk.sh

# With vLLM engine (default)
./run/rlvr_w_tools/qwen3-vl-8b-tk.sh vllm

# With SGLang engine
./run/rlvr_w_tools/qwen3-vl-8b-tk.sh sglang

# With custom parameters
./run/rlvr_w_tools/qwen3-vl-8b-tk.sh vllm \
    trainer.total_epochs=5 \
    data.train_batch_size=128 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3
```

**Key Differences from Non-Tool Training**:

1. **Dataset**: Uses `GeoguessrToolDataset` instead of `GeoguessrRLHFDataset`
2. **Config File**: Loads `geoguessr_ppo_with_tools.yaml` which includes:
   - `custom_chat_template` for tool injection
   - `hybrid_engine: True`
   - `return_raw_chat: True`
   - Multi-turn configuration
   - Tool config path
3. **Batch Size**: Reduced to 256 (from 512) due to additional memory for multi-turn dialogue
4. **GPU Memory**: Set to 0.5 utilization (from 0.6) to accommodate tool execution
5. **Micro Batch Size**: Reduced to 16 (from 32) per GPU

## Configuration

### Data Directory

The script uses data from `$GEOGUESSR_DIR/verl_data/plain_rlvr`:
- Training: `osv5m_train_chunk_*.parquet`, `geochain_test_chunk_*.parquet`, `gaea_train_chunk_*.parquet`
- Validation: `geochain_mini_test_chunk_0000.parquet`, `gaea_bench_chunk_0000.parquet`

### Model

Default model: `/mnt/tidal-alsh-hilab/usr/shenguobin/modelscope/Qwen/Qwen3-VL-8B-Thinking`

### Tool Configuration

Tool config is loaded from: `recipe/geoguessr/config/image_zoom_tool_config.yaml`

Multi-turn settings (from config file):
- `max_assistant_turns`: 5
- `max_user_turns`: 10
- `max_parallel_calls`: 1
- `max_tool_response_length`: 1024

### System Prompt

Default system prompt encourages tool usage:
```
You are a GeoGuessr expert. Use the image_zoom_in_tool to examine details
like road signs, buildings, and landscapes before making your prediction.
Provide your final answer in \boxed{latitude, longitude} format.
```

## Monitoring

Training metrics are logged to:
- Console
- TensorBoard
- SwanLab (project: `verl_grpo_geoguessr_tools`)

Check for tool usage indicators:
- Average number of turns per trajectory
- Tool call frequency
- Tool execution time

## Troubleshooting

### OOM Errors

If you encounter out-of-memory errors:

1. Reduce batch size:
   ```bash
   ./run/rlvr_w_tools/qwen3-vl-8b-tk.sh vllm data.train_batch_size=128
   ```

2. Reduce max turns:
   ```bash
   ./run/rlvr_w_tools/qwen3-vl-8b-tk.sh vllm \
       actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3
   ```

3. Further reduce GPU memory utilization:
   ```bash
   ./run/rlvr_w_tools/qwen3-vl-8b-tk.sh vllm \
       actor_rollout_ref.rollout.gpu_memory_utilization=0.4
   ```

### Tool Not Being Called

If the model doesn't use tools:

1. Check that config file is properly loaded (see script output)
2. Verify tool config path is correct
3. Check logs for "Initialized tools" message
4. Tool usage may be low initially and increase during training

### Performance Issues

Tool execution adds overhead. To optimize:

1. Adjust number of tool workers in `config/image_zoom_tool_config.yaml`:
   ```yaml
   num_workers: 256  # Increase for more parallelism
   rate_limit: 256
   ```

2. Use faster inference engine (SGLang typically faster than vLLM for multi-turn)

## Documentation

For more details, see:
- Complete guide: `recipe/geoguessr/TOOL_TRAINING_README.md`
- Quick reference: `recipe/geoguessr/QUICK_REFERENCE.md`
- Implementation summary: `recipe/geoguessr/IMPLEMENTATION_SUMMARY.md`

## Examples

### Standard Training Run
```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/geoguessr
export GEOGUESSR_DIR=/mnt/tidal-alsh-hilab/usr/shenguobin/geogussr
./run/rlvr_w_tools/qwen3-vl-8b-tk.sh vllm
```

### Quick Test (Small Dataset)
```bash
# Edit script to uncomment the small dataset line (line 28)
# Then run:
./run/rlvr_w_tools/qwen3-vl-8b-tk.sh vllm trainer.total_epochs=1
```

### Multi-Node Training
```bash
./run/rlvr_w_tools/qwen3-vl-8b-tk.sh vllm \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8
```
