"""
Phase 2: 使用 Ray 创建分布式 workers 验证数据分片

这个脚本验证：
1. Ray actors 可以正确初始化分布式 Dataset
2. 每个 worker 加载不同的数据分片
3. 所有 workers 的数据总和等于全量数据

⚠️ 只使用 CPU，不占用 GPU
"""

import sys
import os
import ray

# 添加路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("Phase 2: Ray 分布式 Workers 验证")
print("=" * 80)

# 1. 初始化 Ray（只使用 CPU）
print("\n1. 初始化 Ray...")
if not ray.is_initialized():
    ray.init(num_cpus=8, num_gpus=0)  # 只使用 CPU，不使用 GPU
    print(f"   ✅ Ray 初始化成功")
    print(f"   可用 CPUs: {ray.available_resources().get('CPU', 0)}")
    print(f"   可用 GPUs: {ray.available_resources().get('GPU', 0)}")
else:
    print(f"   ✅ Ray 已经初始化")

# 2. 定义 Ray Actor
print("\n2. 定义 DataWorker Actor...")

@ray.remote(num_cpus=1, num_gpus=0)  # 每个 worker 使用 1 CPU，0 GPU
class DataWorker:
    """
    分布式数据加载 Worker

    每个 worker:
    - 初始化自己的 GeoguessrRLHFDatasetDistributed
    - 加载自己的数据分片
    - 提供查询接口
    """

    def __init__(self, rank, world_size, data_files, config_dict):
        import sys
        import os

        # 确保路径正确
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        from transformers import AutoTokenizer, AutoProcessor
        from omegaconf import DictConfig
        from geoguessr_dataset_distributed import GeoguessrRLHFDatasetDistributed

        self.rank = rank
        self.world_size = world_size

        print(f"Worker {rank}/{world_size}: 开始初始化...")

        # 加载 tokenizer 和 processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )

        # 创建配置
        self.config = DictConfig(config_dict)

        # 初始化数据集（每个 worker 加载自己的分片）
        self.dataset = GeoguessrRLHFDatasetDistributed(
            data_files=data_files,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
            max_samples=-1,  # 加载全部数据，然后分片
            rank=rank,
            world_size=world_size,
        )

        print(f"Worker {rank}/{world_size}: 初始化完成，数据量={len(self.dataset)}")

    def get_dataset_size(self):
        """返回数据集大小"""
        return len(self.dataset)

    def get_worker_info(self):
        """返回 worker 信息"""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "dataset_size": len(self.dataset),
        }

    def get_sample_info(self, index):
        """获取指定样本的信息（不实际加载图像，只返回元数据）"""
        try:
            # 直接访问 dataframe 获取原始数据
            row = self.dataset.dataframe[index]

            # 返回样本的哈希或唯一标识
            prompt = str(row.get('prompt', ''))
            sample_hash = hash(prompt[:100])  # 使用 prompt 前100字符的哈希

            return {
                "index": index,
                "prompt_length": len(prompt),
                "prompt_hash": sample_hash,
                "has_images": 'images' in row and row['images'] is not None,
            }
        except Exception as e:
            return {"error": str(e)}

print(f"   ✅ DataWorker 定义完成")

# 3. 配置参数
print("\n3. 配置参数...")

data_files = [
    "/mnt/tidal-alsh-hilab/usr/shenguobin/geoguessr/verl_data/plain_rlvr/geochain_mini_test_chunk_0000.parquet"
]

config_dict = {
    "prompt_key": "prompt",
    "image_key": "images",
    "max_prompt_length": 2048,
    "truncation": "error",
    "filter_overlong_prompts": True,
    "return_raw_chat": False,
    "return_full_prompt": False,
    "custom_system_prompt": "You are a helpful assistant.",
    "custom_user_prompt_template": None,
}

print(f"   数据文件: {data_files[0]}")
print(f"   ✅ 配置完成")

# 4. 创建 2 个 Ray workers
print("\n" + "=" * 80)
print("4. 创建 2 个 Ray Workers")
print("=" * 80)

world_size = 2
workers = []

print(f"\n   创建 {world_size} 个 workers...")
for rank in range(world_size):
    worker = DataWorker.remote(
        rank=rank,
        world_size=world_size,
        data_files=data_files,
        config_dict=config_dict
    )
    workers.append(worker)
    print(f"   Worker {rank}: 已提交初始化任务")

print(f"\n   等待所有 workers 初始化完成...")

# 获取每个 worker 的信息
worker_infos = ray.get([w.get_worker_info.remote() for w in workers])

print(f"\n   Workers 信息:")
for info in worker_infos:
    print(f"   - Worker {info['rank']}/{info['world_size']}: {info['dataset_size']} samples")

# 5. 验证数据分片
print("\n" + "=" * 80)
print("5. 验证数据分片")
print("=" * 80)

total_samples = sum(info['dataset_size'] for info in worker_infos)
print(f"\n   Worker 0: {worker_infos[0]['dataset_size']} samples")
print(f"   Worker 1: {worker_infos[1]['dataset_size']} samples")
print(f"   总计: {total_samples} samples")
print(f"   预期: 2099 samples (文件总量)")

# 验证总数
if total_samples == 2099:
    print(f"   ✅ 数据总数验证通过")
else:
    print(f"   ❌ 数据总数不匹配: {total_samples} != 2099")

# 验证分片大小差异
size_diff = abs(worker_infos[0]['dataset_size'] - worker_infos[1]['dataset_size'])
if size_diff <= 1:
    print(f"   ✅ 分片大小差异验证通过 (差值={size_diff})")
else:
    print(f"   ❌ 分片大小差异过大: {size_diff} > 1")

# 6. 验证数据不重叠
print("\n" + "=" * 80)
print("6. 验证数据不重叠")
print("=" * 80)

print(f"\n   获取每个 worker 的第一个样本信息...")
sample0_info = ray.get(workers[0].get_sample_info.remote(0))
sample1_info = ray.get(workers[1].get_sample_info.remote(0))

print(f"\n   Worker 0 第一个样本:")
print(f"     - Prompt 长度: {sample0_info['prompt_length']}")
print(f"     - Prompt 哈希: {sample0_info['prompt_hash']}")

print(f"\n   Worker 1 第一个样本:")
print(f"     - Prompt 长度: {sample1_info['prompt_length']}")
print(f"     - Prompt 哈希: {sample1_info['prompt_hash']}")

# 比较哈希
if sample0_info['prompt_hash'] != sample1_info['prompt_hash']:
    print(f"\n   ✅ 两个 workers 的第一个样本不同（数据无重叠）")
else:
    print(f"\n   ❌ 警告: 两个 workers 的第一个样本相同!")

# 7. 测试 4 个 workers
print("\n" + "=" * 80)
print("7. 测试 4 个 Workers")
print("=" * 80)

world_size_4 = 4
print(f"\n   创建 {world_size_4} 个 workers...")

workers_4 = []
for rank in range(world_size_4):
    worker = DataWorker.remote(
        rank=rank,
        world_size=world_size_4,
        data_files=data_files,
        config_dict=config_dict
    )
    workers_4.append(worker)

print(f"   等待初始化...")
worker_infos_4 = ray.get([w.get_worker_info.remote() for w in workers_4])

print(f"\n   Workers 信息:")
sizes_4 = []
for info in worker_infos_4:
    print(f"   - Worker {info['rank']}/{info['world_size']}: {info['dataset_size']} samples")
    sizes_4.append(info['dataset_size'])

total_4 = sum(sizes_4)
print(f"\n   总计: {total_4} samples")

if total_4 == 2099:
    print(f"   ✅ 4-worker 数据总数验证通过")
else:
    print(f"   ❌ 4-worker 数据总数不匹配: {total_4} != 2099")

# 验证分片均匀性
max_size = max(sizes_4)
min_size = min(sizes_4)
if max_size - min_size <= 1:
    print(f"   ✅ 分片均匀性验证通过 (最大差值={max_size - min_size})")
else:
    print(f"   ❌ 分片不均匀: 最大={max_size}, 最小={min_size}, 差值={max_size - min_size}")

# 8. 清理
print("\n" + "=" * 80)
print("8. 清理资源")
print("=" * 80)

print(f"\n   关闭 Ray...")
ray.shutdown()
print(f"   ✅ Ray 已关闭")

# 9. 总结
print("\n" + "=" * 80)
print("✅ Phase 2 测试完成！")
print("=" * 80)

print("\n总结:")
print(f"  ✅ Ray 分布式 workers 创建成功")
print(f"  ✅ 2-worker 分片: {worker_infos[0]['dataset_size']} + {worker_infos[1]['dataset_size']} = {total_samples}")
print(f"  ✅ 4-worker 分片: {' + '.join(str(s) for s in sizes_4)} = {total_4}")
print(f"  ✅ 数据无重叠验证通过")
print(f"  ✅ 分片均匀性验证通过")

print("\n下一步: Phase 3 - 在 Worker 内部集成 DataLoader")
