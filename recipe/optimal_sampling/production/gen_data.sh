#!/bin/bash

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

python generate_data_parallel.py \
      --model_theta Qwen/Qwen3-0.6B-Base \
      --model_t Qwen/Qwen3-4B-Instruct-2507 \
      --dataset agentica-org/DeepScaleR-Preview-Dataset \
      --output ./data/deepscaler_q4b-it_q06bb/kl_symmetry.parquet \
      --output_format parquet \
      --strategy data_parallel \
      --num_gpus 8 \
      --batch_size_per_gpu 128 \
      --max_new_tokens 4096 \
      --alpha_method kl_symmetry \
      --save_diagnostics \
      --dtype bfloat16 \
      --print_sample

python generate_data_parallel.py \
      --model_theta Qwen/Qwen3-4B-Base \
      --model_t Qwen/Qwen3-14B \
      --dataset agentica-org/DeepScaleR-Preview-Dataset \
      --output ./data/deepscaler_q14b_q4bb_new/ess_balance.parquet \
      --output_format parquet \
      --strategy data_parallel \
      --num_gpus 8 \
      --batch_size_per_gpu 32 \
      --max_new_tokens 8192 \
      --alpha_method ess_balance \
      --save_diagnostics \
      --dtype bfloat16 \
      --print_sample


python generate_data_parallel.py \
      --model_theta Qwen/Qwen3-4B-Base \
      --model_t Qwen/Qwen3-14B \
      --dataset agentica-org/DeepScaleR-Preview-Dataset \
      --output ./data/deepscaler_q14b_q4bb_new/fixed_50.parquet \
      --output_format parquet \
      --strategy data_parallel \
      --num_gpus 8 \
      --batch_size_per_gpu 32 \
      --max_new_tokens 8192 \
      --alpha_method fixed \
      --fixed_alpha 0.5 \
      --save_diagnostics \
      --dtype bfloat16 \
      --print_sample


python generate_data_parallel.py \
      --model_theta Qwen/Qwen3-4B-Base \
      --model_t Qwen/Qwen3-14B \
      --dataset agentica-org/DeepScaleR-Preview-Dataset \
      --output ./data/deepscaler_q14b_q4bb_new/fixed_25.parquet \
      --output_format parquet \
      --strategy data_parallel \
      --num_gpus 8 \
      --batch_size_per_gpu 32 \
      --max_new_tokens 8192 \
      --alpha_method fixed \
      --fixed_alpha 0.25 \
      --save_diagnostics \
      --dtype bfloat16 \
      --print_sample



python generate_data_parallel.py \
      --model_theta Qwen/Qwen3-4B-Base \
      --model_t Qwen/Qwen3-14B \
      --dataset agentica-org/DeepScaleR-Preview-Dataset \
      --output ./data/deepscaler_q14b_q4bb_new/fixed_75.parquet \
      --output_format parquet \
      --strategy data_parallel \
      --num_gpus 8 \
      --batch_size_per_gpu 32 \
      --max_new_tokens 8192 \
      --alpha_method fixed \
      --fixed_alpha 0.75 \
      --save_diagnostics \
      --dtype bfloat16 \
      --print_sample


python generate_data_parallel.py \
      --model_theta Qwen/Qwen3-4B-Base \
      --model_t Qwen/Qwen3-14B \
      --dataset agentica-org/DeepScaleR-Preview-Dataset \
      --output ./data/deepscaler_q14b_q4bb_new/fixed_00.parquet \
      --output_format parquet \
      --strategy data_parallel \
      --num_gpus 8 \
      --batch_size_per_gpu 32 \
      --max_new_tokens 8192 \
      --alpha_method fixed \
      --fixed_alpha 0.0 \
      --save_diagnostics \
      --dtype bfloat16 \
      --print_sample



python generate_data_parallel.py \
      --model_theta Qwen/Qwen3-4B-Base \
      --model_t Qwen/Qwen3-14B \
      --dataset agentica-org/DeepScaleR-Preview-Dataset \
      --output ./data/deepscaler_q14b_q4bb_new/fixed_100.parquet \
      --output_format parquet \
      --strategy data_parallel \
      --num_gpus 8 \
      --batch_size_per_gpu 32 \
      --max_new_tokens 8192 \
      --alpha_method fixed \
      --fixed_alpha 1.0 \
      --save_diagnostics \
      --dtype bfloat16 \
      --print_sample


#python generate_data_parallel.py \
#      --model_theta Qwen/Qwen3-4B-Base \
#      --model_t Qwen/Qwen3-14B \
#      --dataset agentica-org/DeepScaleR-Preview-Dataset \
#      --output ./data/deepscaler_q14b_q4bb/ess_balance.parquet \
#      --output_format parquet \
#      --strategy data_parallel \
#      --num_gpus 1 \
#      --batch_size_per_gpu 2 \
#      --max_new_tokens 16384 \
#      --alpha_method ess_balance \
#      --save_diagnostics \
#      --dtype bfloat16 \
#      --print_sample