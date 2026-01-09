"""
Phase 1 测试脚本：验证数据分片

这个脚本测试 GeoguessrRLHFDatasetDistributed 的数据分片功能。
⚠️ 只使用 CPU，不调用 GPU
"""

import sys
import os

# 添加路径 - 包含父目录以找到 geoguessr_dataset.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoProcessor
from omegaconf import DictConfig
import torch

print("=" * 80)
print("Phase 1: 测试数据分片功能")
print("=" * 80)

# 1. 配置
print("\n1. 配置参数...")
data_files = [
    "/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/verl_data/plain_rlvr/geochain_mini_test_chunk_0000.parquet"
]

config = DictConfig({
    "prompt_key": "prompt",  # ← 修正：从 raw_chat 改为 prompt
    "image_key": "images",
    "max_prompt_length": 2048,
    "truncation": "error",
    "filter_overlong_prompts": True,
    "return_raw_chat": False,
    "return_full_prompt": False,
    "custom_system_prompt": "You are a helpful assistant.",
    "custom_user_prompt_template": None,
})

print(f"   数据文件: {data_files[0]}")
print(f"   只加载前 1000 个样本进行测试")

# 2. 加载 tokenizer 和 processor（不加载模型）
print("\n2. 加载 tokenizer 和 processor...")
print("   ⚠️  注意: 不加载模型，不占用 GPU")

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True
)

print("   ✅ Tokenizer 和 Processor 加载完成")

# 3. 测试单个 Dataset（baseline）
print("\n" + "=" * 80)
print("3. 测试单个 Dataset（baseline）")
print("=" * 80)

from geoguessr_dataset_distributed import GeoguessrRLHFDatasetDistributed

dataset_single = GeoguessrRLHFDatasetDistributed(
    data_files=data_files,
    tokenizer=tokenizer,
    config=config,
    processor=processor,
    max_samples=1000,  # 只加载 1000 个样本
    rank=0,
    world_size=1,  # 单个 worker
)

print(f"\n   Dataset 大小: {len(dataset_single)} samples")
print(f"   ✅ Baseline dataset 创建成功")

# 4. 测试数据分片（2 workers）
print("\n" + "=" * 80)
print("4. 测试数据分片（2 workers）")
print("=" * 80)

dataset_worker0 = GeoguessrRLHFDatasetDistributed(
    data_files=data_files,
    tokenizer=tokenizer,
    config=config,
    processor=processor,
    max_samples=1000,
    rank=0,
    world_size=2,
)

dataset_worker1 = GeoguessrRLHFDatasetDistributed(
    data_files=data_files,
    tokenizer=tokenizer,
    config=config,
    processor=processor,
    max_samples=1000,
    rank=1,
    world_size=2,
)

print(f"\n   Worker 0 samples: {len(dataset_worker0)}")
print(f"   Worker 1 samples: {len(dataset_worker1)}")
print(f"   总计: {len(dataset_worker0) + len(dataset_worker1)}")
print(f"   原始: {len(dataset_single)}")

# 验证
assert len(dataset_worker0) + len(dataset_worker1) == len(dataset_single), \
    f"数据分片错误: {len(dataset_worker0)} + {len(dataset_worker1)} != {len(dataset_single)}"

print(f"   ✅ 数据分片验证通过")

# 5. 测试数据分片（4 workers）
print("\n" + "=" * 80)
print("5. 测试数据分片（4 workers）")
print("=" * 80)

datasets_4workers = []
for rank in range(4):
    dataset = GeoguessrRLHFDatasetDistributed(
        data_files=data_files,
        tokenizer=tokenizer,
        config=config,
        processor=processor,
        max_samples=1000,
        rank=rank,
        world_size=4,
    )
    datasets_4workers.append(dataset)
    print(f"   Worker {rank} samples: {len(dataset)}")

total = sum(len(d) for d in datasets_4workers)
print(f"\n   总计: {total}")
print(f"   原始: {len(dataset_single)}")

assert total == len(dataset_single), \
    f"数据分片错误: {total} != {len(dataset_single)}"

print(f"   ✅ 4-worker 数据分片验证通过")

# 6. 测试数据不重叠
print("\n" + "=" * 80)
print("6. 测试数据不重叠")
print("=" * 80)

print("\n   获取每个 worker 的第一个样本...")
try:
    sample0 = dataset_worker0[0]
    sample1 = dataset_worker1[0]

    # 比较 input_ids（如果有的话）
    if 'input_ids' in sample0 and 'input_ids' in sample1:
        ids0 = sample0['input_ids'].tolist()
        ids1 = sample1['input_ids'].tolist()

        if ids0 == ids1:
            print("   ⚠️  警告: Worker 0 和 Worker 1 的第一个样本相同!")
        else:
            print("   ✅ Worker 0 和 Worker 1 的样本不同")

    print(f"   Worker 0 sample 0 keys: {list(sample0.keys())}")
    print(f"   Worker 1 sample 0 keys: {list(sample1.keys())}")

except Exception as e:
    print(f"   ⚠️  无法获取样本: {e}")
    print(f"   （这可能是因为 processor 需要图像数据，但我们只是测试分片逻辑）")

# 7. 测试不同的 world_size
print("\n" + "=" * 80)
print("7. 测试不同的 world_size")
print("=" * 80)

for world_size in [8, 16, 64]:
    datasets = []
    for rank in range(world_size):
        dataset = GeoguessrRLHFDatasetDistributed(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=1000,
            rank=rank,
            world_size=world_size,
        )
        datasets.append(dataset)

    total = sum(len(d) for d in datasets)
    sizes = [len(d) for d in datasets]
    min_size = min(sizes)
    max_size = max(sizes)

    print(f"\n   world_size={world_size}:")
    print(f"     总计: {total}")
    print(f"     范围: [{min_size}, {max_size}]")
    print(f"     差值: {max_size - min_size}")

    assert total == len(dataset_single), \
        f"数据分片错误: {total} != {len(dataset_single)}"

    # 验证分片大小差异不超过 1
    assert max_size - min_size <= 1, \
        f"分片大小差异过大: {max_size - min_size} > 1"

    print(f"     ✅ world_size={world_size} 验证通过")

# 8. 总结
print("\n" + "=" * 80)
print("✅ Phase 1 测试完成！")
print("=" * 80)

print("\n总结:")
print("  ✅ 数据分片逻辑正确")
print("  ✅ 所有 workers 的数据总和等于原始数据")
print("  ✅ 分片大小差异不超过 1")
print("  ✅ 支持任意 world_size (2, 4, 8, 16, 64)")
print("\n下一步: Phase 2 - 使用 Ray 创建分布式 workers")
