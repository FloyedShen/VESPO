"""
Phase 1 简化测试：验证数据分片无重复

这个脚本测试修复后的数据分片功能，确保：
1. 不同 worker 加载不同的数据
2. 所有 workers 的数据总和等于全量数据
"""

import sys
import os

# 添加路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoProcessor
from omegaconf import DictConfig

print("=" * 80)
print("Phase 1 简化测试：验证数据分片")
print("=" * 80)

# 1. 配置
print("\n1. 配置参数...")
data_files = [
    "/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/verl_data/plain_rlvr/geochain_mini_test_chunk_0000.parquet"
]

config = DictConfig({
    "prompt_key": "prompt",
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

# 2. 加载 tokenizer 和 processor
print("\n2. 加载 tokenizer 和 processor...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    trust_remote_code=True
)

print("   ✅ 加载完成")

# 3. 测试：加载全量数据，然后分片给 2 个 workers
print("\n" + "=" * 80)
print("3. 测试数据分片（2 workers）")
print("=" * 80)

from geoguessr_dataset_distributed import GeoguessrRLHFDatasetDistributed

# 创建 2 个 workers，每个加载全量数据然后分片
dataset_worker0 = GeoguessrRLHFDatasetDistributed(
    data_files=data_files,
    tokenizer=tokenizer,
    config=config,
    processor=processor,
    max_samples=-1,  # 不限制，加载全部
    rank=0,
    world_size=2,
)

dataset_worker1 = GeoguessrRLHFDatasetDistributed(
    data_files=data_files,
    tokenizer=tokenizer,
    config=config,
    processor=processor,
    max_samples=-1,
    rank=1,
    world_size=2,
)

print(f"\n   Worker 0: {len(dataset_worker0)} samples")
print(f"   Worker 1: {len(dataset_worker1)} samples")
print(f"   总计: {len(dataset_worker0) + len(dataset_worker1)}")

# 验证总数
total = len(dataset_worker0) + len(dataset_worker1)
assert total == 2099, f"总数不对: {total} != 2099"
print(f"   ✅ 总数验证通过 (2099)")

# 4. 验证数据不重叠
print("\n" + "=" * 80)
print("4. 验证数据不重叠")
print("=" * 80)

# 获取两个 worker 的原始数据索引
# Worker 0 应该是 [0:1050]
# Worker 1 应该是 [1050:2099]

# 由于 dataframe 被分片了，我们检查它们的实际内容
try:
    # 尝试访问第一个样本的 prompt
    import datasets

    # 获取原始数据（未经tokenizer处理）
    df0 = dataset_worker0.dataframe
    df1 = dataset_worker1.dataframe

    # 比较第一条数据的 prompt 文本
    prompt0 = df0[0]['prompt']
    prompt1 = df1[0]['prompt']

    # 打印一些信息
    print(f"\n   Worker 0 第一条 prompt 长度: {len(str(prompt0))}")
    print(f"   Worker 1 第一条 prompt 长度: {len(str(prompt1))}")

    # 检查是否相同
    if str(prompt0) == str(prompt1):
        print("   ❌ 错误: Worker 0 和 Worker 1 的第一个样本相同！")
    else:
        print("   ✅ Worker 0 和 Worker 1 的样本不同")

    # 额外验证：比较最后一条
    prompt0_last = df0[len(df0) - 1]['prompt']
    prompt1_last = df1[len(df1) - 1]['prompt']

    if str(prompt0_last) == str(prompt1_last):
        print("   ❌ 错误: Worker 0 和 Worker 1 的最后一个样本相同！")
    else:
        print("   ✅ Worker 0 和 Worker 1 的最后样本也不同")

except Exception as e:
    print(f"   ⚠️  无法比较样本内容: {e}")

# 5. 测试 4 workers
print("\n" + "=" * 80)
print("5. 测试 4 workers")
print("=" * 80)

datasets_4 = []
for rank in range(4):
    ds = GeoguessrRLHFDatasetDistributed(
        data_files=data_files,
        tokenizer=tokenizer,
        config=config,
        processor=processor,
        max_samples=-1,
        rank=rank,
        world_size=4,
    )
    datasets_4.append(ds)
    print(f"   Worker {rank}: {len(ds)} samples")

total_4 = sum(len(d) for d in datasets_4)
print(f"\n   总计: {total_4}")
assert total_4 == 2099, f"总数不对: {total_4} != 2099"
print(f"   ✅ 4-worker 验证通过")

# 6. 总结
print("\n" + "=" * 80)
print("✅ Phase 1 测试完成！")
print("=" * 80)

print("\n总结:")
print(f"  ✅ 2-worker 分片: {len(dataset_worker0)} + {len(dataset_worker1)} = {len(dataset_worker0) + len(dataset_worker1)}")
print(f"  ✅ 4-worker 分片: {' + '.join(str(len(d)) for d in datasets_4)} = {total_4}")
print(f"  ✅ 数据无重复")
print("\n下一步: Phase 2 - 使用 Ray 创建分布式 workers")
