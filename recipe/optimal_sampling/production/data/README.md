### Generate Data via hf-model

1. generate data from mixed distribution:

```bash 
python generate_data_parallel.py \
      --model_theta Qwen/Qwen3-4B-Base \
      --model_t Qwen/Qwen3-14B \
      --dataset agentica-org/DeepScaleR-Preview-Dataset \
      --output ./data/deepscaler_q14b_q4bb_new/kl_symmetry.parquet \
      --output_format parquet \
      --strategy data_parallel \
      --num_gpus 8 \
      --batch_size_per_gpu 32 \
      --max_new_tokens 8192 \
      --alpha_method kl_symmetry \
      --save_diagnostics \
      --dtype bfloat16 \
      --print_sample
```

2. generate data from pure teacher model 

```bash
cd /mnt/tidal-alsh-hilab/usr/shenguobin/verl && source .venv/bin/activate
vllm serve Qwen/Qwen3-14B --port 8123  --enforce-eager  -tp 8

python sample_from_vllm.py \
  --dataset agentica-org/DeepScaleR-Preview-Dataset  \
  --base_url  http://localhost:8123/v1  \
  --model_name Qwen/Qwen3-14B  \
  --max_tokens 8192 \
  --max_concurrent 512 \
  --output data/deepscaler_q14b.jsonl
```

3. combine data of mixed & pure data

```bash
python3 production/filter_and_prepare_sft_data.py     \
  --input production/data/deepscaler_q14b_q4bb_new/kl_symmetry.parquet     \
  --output production/data/deepscaler_q14b_q4bb_new/filtered/  \
  --verify-sampled  \
  --num-workers 128  \
  --teacher-response production/data/deepscaler_q14b.jsonl
```


4. sft from scratch


```bash
# mixed
#/diancpfs/user/guobin/
#/mnt/tidal-alsh-hilab/usr/shenguobin/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
    --num_processes=8 \
    --multi_gpu \
    --main_process_port 29501 \
    trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --dataset_name /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/optimal_sampling/production/data/deepscaler_q14b_q4bb_new/filtered/kl_symmetry_sft_sampled \
    --chat_template_path /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/optimal_sampling/production/data/simple_chat_template.j2 \
    --fsdp "full_shard auto_wrap" \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_length 8192 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --save_steps 10 \
    --use_liger_kernel \
    --output_dir checkpoints/qwen3-4b-base-DeepScaleR-osn-q14b-sft \
    --report_to mlflow \
    --packing False
    
    
# teacher

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
    --num_processes=8 \
    --multi_gpu \
    --main_process_port 29501 \
    trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --dataset_name /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/optimal_sampling/production/data/deepscaler_q14b_q4bb_new/filtered/kl_symmetry_sft_teacher \
    --chat_template_path /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/optimal_sampling/production/data/simple_chat_template.j2 \
    --fsdp "full_shard auto_wrap" \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_length 8192 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --save_steps 10 \
    --use_liger_kernel \
    --output_dir checkpoints/qwen3-4b-base-DeepScaleR-pure-n-q14b-sft \
    --report_to mlflow \
    --packing False
    
# oracle
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
    --num_processes=8 \
    --multi_gpu \
    --main_process_port 29501 \
    trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --dataset_name /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/optimal_sampling/production/data/deepscaler_q14b_q4bb_new/filtered/kl_symmetry_sft_original \
    --chat_template_path /mnt/tidal-alsh-hilab/usr/shenguobin/verl/recipe/optimal_sampling/production/data/simple_chat_template.j2 \
    --fsdp "full_shard auto_wrap" \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_length 8192 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --save_steps 10 \
    --use_liger_kernel \
    --output_dir checkpoints/qwen3-4b-base-DeepScaleR-oracle-sft \
    --report_to mlflow \
    --packing False


```



5. eval

   1. math-ai/amc23 (question answer)
   2. HuggingFaceH4/aime_2024HuggingFaceH4/aime_2024 (problem answer)
   3. math-ai/aime25 (problem answer)
   4. HuggingFaceH4/MATH-500 (problem answer)

```bash
python production/eval_checkpoints.py \
    --checkpoint-dir /mnt/tidal-alsh-hilab/usr/shenguobin/trl/checkpoints \
    --output-dir ./eval_results \
    --tensor-parallel-size 8 \
    --batch-size 500 \
    --resume \
    --extra-models Qwen/Qwen3-1.7B \
    --verbose
    
#    
```