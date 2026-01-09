# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
分布式 Dataset 版本：支持数据分片

使用方法:
    # Worker 0 (总共 64 个 workers)
    dataset = GeoguessrRLHFDatasetDistributed(
        ...,
        rank=0,
        world_size=64
    )

    # Worker 1
    dataset = GeoguessrRLHFDatasetDistributed(
        ...,
        rank=1,
        world_size=64
    )

每个 worker 只加载自己的数据分片。
"""

import logging
from typing import Optional

from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import DictConfig

from geoguessr_dataset import GeoguessrRLHFDataset

logger = logging.getLogger(__name__)


class GeoguessrRLHFDatasetDistributed(GeoguessrRLHFDataset):
    """
    分布式版本的 GeoguessrRLHFDataset，支持数据分片。

    新增参数:
        rank (int): 当前 worker 的 rank (0 到 world_size-1)
        world_size (int): 总 worker 数量

    工作原理:
        1. 加载完整数据集
        2. 根据 rank 和 world_size 分片
        3. 每个 worker 只保留自己的数据
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        初始化分布式数据集

        Args:
            rank: 当前 worker 的 rank (0-based)
            world_size: 总 worker 数量
            其他参数同 GeoguessrRLHFDataset
        """
        self.rank = rank
        self.world_size = world_size

        # 验证参数
        assert 0 <= rank < world_size, f"rank must be in [0, {world_size}), got {rank}"

        # 重要：在分布式模式下，先加载全部数据再分片
        # 而不是先随机采样再分片（会导致重复）
        self._original_max_samples = max_samples
        if world_size > 1:
            # 暂时禁用 max_samples，让父类加载全部数据
            max_samples = -1

        # 调用父类初始化
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

        # 在父类初始化后，对数据进行分片
        if world_size > 1:
            self._shard_data()

        logger.info(
            f"Initialized GeoguessrRLHFDatasetDistributed: "
            f"rank={self.rank}/{self.world_size}, "
            f"samples={len(self)}"
        )

    def _shard_data(self):
        """
        对数据进行分片，每个 worker 只保留自己的部分

        分片策略:
            - 连续分片: 将数据平均分配给各个 worker
            - 例如: 1000 samples, world_size=3
                worker 0: [0:334]     (334 samples)
                worker 1: [334:667]   (333 samples)
                worker 2: [667:1000]  (333 samples)
        """
        total_samples = len(self.dataframe)

        # 计算每个 worker 的样本数
        samples_per_worker = total_samples // self.world_size
        remainder = total_samples % self.world_size

        # 计算当前 worker 的起始和结束索引
        # 前 remainder 个 workers 多拿一个样本
        if self.rank < remainder:
            start_idx = self.rank * (samples_per_worker + 1)
            end_idx = start_idx + samples_per_worker + 1
        else:
            start_idx = remainder * (samples_per_worker + 1) + (self.rank - remainder) * samples_per_worker
            end_idx = start_idx + samples_per_worker

        # 分片数据
        indices = list(range(start_idx, end_idx))
        self.dataframe = self.dataframe.select(indices)

        logger.info(
            f"Data sharding completed: "
            f"rank={self.rank}/{self.world_size}, "
            f"total_samples={total_samples}, "
            f"shard_samples={len(self.dataframe)}, "
            f"indices=[{start_idx}:{end_idx}]"
        )

        # 如果指定了 max_samples，在分片后再限制
        if self._original_max_samples > 0:
            current_size = len(self.dataframe)
            if self._original_max_samples < current_size:
                self.dataframe = self.dataframe.select(list(range(self._original_max_samples)))
                logger.info(
                    f"Applied max_samples limit: {current_size} → {self._original_max_samples}"
                )

        # 验证数据分片
        self._verify_sharding(total_samples)

    def _verify_sharding(self, total_samples: int):
        """
        验证数据分片是否正确
        """
        expected_min = total_samples // self.world_size
        expected_max = expected_min + 1

        actual = len(self.dataframe)

        if not (expected_min <= actual <= expected_max):
            logger.warning(
                f"Shard size mismatch: expected [{expected_min}, {expected_max}], "
                f"got {actual}"
            )
        else:
            logger.info(f"✅ Shard size verified: {actual} samples")
