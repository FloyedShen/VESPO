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
Enhanced collate function for Qwen2-VL models with distributed image preprocessing support.

This module provides a collate function that:
1. Processes raw multi_modal_data into multi_modal_inputs for training (distributed preprocessing)
2. Efficiently generates position_ids in batch using get_rope_index

This enables lazy datasets to defer image preprocessing to the collate stage,
achieving true distributed preprocessing across all workers while maintaining
consistency between rollout (inference) and training phases.
"""

import torch
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn


def create_qwen_vl_collate_fn(processor):
    """
    Create a collate function that batch-generates position_ids for Qwen2-VL models.

    Args:
        processor: The Qwen2-VL processor instance

    Returns:
        collate_fn: Function that takes a list of samples and returns a batched dict
    """

    def qwen_vl_collate_fn(batch):
        """
        Collate function that:
        1. Processes multi_modal_data into multi_modal_inputs (for lazy datasets)
        2. Calls default collate_fn to batch samples
        3. Batch-regenerates position_ids using get_rope_index

        Note: Qwen2-VL ALWAYS requires 3D position_ids (4, batch_size, seq_length),
        even for pure text samples without images. So we always call get_rope_index.

        Args:
            batch: List of sample dicts from dataset

        Returns:
            Batched dict with corrected position_ids for Qwen2-VL
        """
        # Dataset now provides multi_modal_inputs, no need to generate here
        # First, use default collate function
        batched = default_collate_fn(batch)

        # For Qwen2-VL, we ALWAYS need to regenerate position_ids as 3D tensors
        # Even pure text samples need (4, seq_len) shape
        if len(batch) > 0:
            # Determine which get_rope_index to use
            if "Qwen3VLProcessor" in processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            # Batch regenerate position_ids
            new_position_ids = []

            for i, sample in enumerate(batch):
                # Get grid info from original sample (may be None for text-only samples)
                image_grid_thw = sample.get("image_grid_thw", None)
                video_grid_thw = sample.get("video_grid_thw", None)
                second_per_grid_ts = sample.get("second_per_grid_ts", None)

                # Get padded input_ids and attention_mask from batched dict
                # This ensures all samples have the same seq_len after padding
                input_ids_i = batched["input_ids"][i]  # (seq_len,)
                attention_mask_i = batched["attention_mask"][i]  # (seq_len,)

                # Generate correct position_ids
                # get_rope_index returns (4, seq_len) for a single sample
                # Even without images, it returns the correct 3D shape
                position_ids = get_rope_index(
                    processor,
                    input_ids=input_ids_i,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask_i,
                )

                new_position_ids.append(position_ids)

            # Stack position_ids and update batched dict
            # new_position_ids is list of (4, seq_len) tensors with same seq_len after padding
            # Stack on dim=0 to get (batch_size, 4, seq_len)
            # Actor's dp_actor.py will transpose it to (4, batch_size, seq_len) later
            stacked_position_ids = torch.stack(new_position_ids, dim=0)
            batched["position_ids"] = stacked_position_ids

        return batched

    return qwen_vl_collate_fn
