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

import logging
import re
from typing import Optional

from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import DictConfig

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class GeoguessrRLHFDataset(RLHFDataset):
    """
    Custom RLHF Dataset for Geoguessr task with configurable system and user prompts.

    This dataset extends RLHFDataset to allow custom prompt templates for system and user messages.
    Key principle: Only modify text content, preserve <image>/<video> tags in their original positions.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # Extract custom prompt configurations before calling parent __init__
        self.custom_system_prompt = config.get("custom_system_prompt", None)
        self.custom_user_prompt_template = config.get("custom_user_prompt_template", None)

        # Call parent constructor
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

        logger.info(f"Initialized GeoguessrRLHFDataset with custom prompts:")
        logger.info(f"  System prompt: {self.custom_system_prompt if self.custom_system_prompt else 'Using default'}")
        logger.info(f"  User prompt template: {self.custom_user_prompt_template if self.custom_user_prompt_template else 'Using default'}")

    def _build_messages(self, example: dict):
        """
        Override parent _build_messages to inject custom prompts before processing multimodal tags.

        This method:
        1. Applies custom system/user prompts
        2. Preserves <image>/<video> tags in their original positions
        3. Calls parent's multimodal processing logic
        """
        # Get messages from example
        messages: list = example.pop(self.prompt_key)

        # Apply custom prompts if configured
        if self.custom_system_prompt is not None or self.custom_user_prompt_template is not None:
            modified_messages = []

            # Handle system message
            system_msg_found = False
            for msg in messages:
                if msg.get("role") == "system":
                    system_msg_found = True
                    if self.custom_system_prompt is not None:
                        # Replace with custom system prompt
                        modified_messages.append({
                            "role": "system",
                            "content": self.custom_system_prompt
                        })
                    else:
                        # Keep original system message
                        modified_messages.append(msg)
                    break

            # Add custom system message if not found and configured
            if not system_msg_found and self.custom_system_prompt is not None:
                modified_messages.insert(0, {
                    "role": "system",
                    "content": self.custom_system_prompt
                })

            # Handle user messages
            for msg in messages:
                if msg.get("role") == "user":
                    original_content = msg["content"]

                    if self.custom_user_prompt_template is not None:
                        # Extract text parts, preserving <image>/<video> tags
                        # Split by multimodal tags but keep them
                        parts = re.split(r'(<image>|<video>)', original_content)

                        # Get text-only content (excluding tags)
                        text_only = ''.join([p for p in parts if p not in ['<image>', '<video>']])

                        # Apply template to text content
                        new_text = self.custom_user_prompt_template.replace("{content}", text_only)
                        new_text = new_text.replace("{question}", text_only)

                        # Reconstruct content with tags in original positions
                        # Strategy: Replace text parts while keeping tags
                        new_content = original_content
                        for part in parts:
                            if part not in ['<image>', '<video>', '']:
                                # Replace first occurrence of text part with new text
                                new_content = new_content.replace(part, new_text, 1)
                                break  # Only replace once

                        modified_messages.append({
                            "role": "user",
                            "content": new_content
                        })
                    else:
                        # Keep original user message
                        modified_messages.append(msg)
                elif msg.get("role") not in ["system"]:
                    # Keep other messages (assistant, etc.) as-is
                    modified_messages.append(msg)

            messages = modified_messages

        # Now process multimodal tags using parent's logic
        # This handles <image>/<video> tag splitting and conversion
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Get item using parent's implementation, which now uses our overridden _build_messages.
        """
        # Increase PIL limits to handle large ICC profiles
        from PIL import PngImagePlugin
        PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024 ** 2)  # 10MB
        PngImagePlugin.MAX_TEXT_MEMORY = 10 * (1024 ** 2)  # 10MB

        # Call parent's __getitem__ which will use our _build_messages
        result = super().__getitem__(item)

        # ✅ Crop watermark for geochain data source
        # Check if data_source is 'geochain' and crop bottom 32 pixels
        data_source = result.get('data_source', None)
        if 'geochain' in data_source:
            # Process pixel_values if present
            if 'pixel_values' in result and result['pixel_values'] is not None:
                try:
                    import torch
                    pixel_values = result['pixel_values']

                    # Handle both single image and batched images
                    if isinstance(pixel_values, torch.Tensor):
                        # pixel_values shape: [num_images, channels, height, width] or [channels, height, width]
                        if pixel_values.dim() == 4:
                            # Batch of images: crop each image
                            result['pixel_values'] = pixel_values[:, :, :-32, :]
                            logger.debug(f"Cropped geochain watermark (batch): {pixel_values.shape} -> {result['pixel_values'].shape}")
                        elif pixel_values.dim() == 3:
                            # Single image: crop bottom 32 pixels
                            result['pixel_values'] = pixel_values[:, :-32, :]
                            logger.debug(f"Cropped geochain watermark: {pixel_values.shape} -> {result['pixel_values'].shape}")
                except Exception as e:
                    logger.warning(f"Failed to crop geochain watermark: {e}")

        elif 'data_source' not in result:
            logger.error(f"data_source key not found in result: {result}")
            exit(1)


        return result


class GeoguessrToolDataset(RLHFDataset):
    """
    GeoGuessr Dataset with Tool Support for multi-turn agent interaction.

    This dataset extends RLHFDataset to support tool calling functionality,
    following the DeepEyes pattern for visual tool integration.

    Key features:
    - Returns raw_prompt (message list) instead of pre-tokenized prompt
    - Provides tools_kwargs for tool initialization (e.g., image data)
    - Sets agent_name="tool_agent" to use ToolAgentLoop
    - Supports custom system prompts for tool usage instructions
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # Extract custom prompt configurations
        self.custom_system_prompt = config.get("custom_system_prompt", None)
        self.custom_user_prompt_template = config.get("custom_user_prompt_template", None)

        # Call parent constructor
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

        # Cache tool schemas for validation logging (initialized once)
        self._cached_tool_schemas = None
        if processor is not None:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config
                # Get tool_config_path from config instead of hardcoding
                import os
                tool_config_path = os.getenv("TOOL_CONFIG_PATH", None)
                if tool_config_path is None:
                    # Try to get from config
                    if hasattr(config, "actor_rollout_ref"):
                        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
                    else:
                        # Fallback to default service tools config
                        tool_config_path = "recipe/geoguessr/config/geoguessr_service_tools_config.yaml"

                tool_list = initialize_tools_from_config(tool_config_path)
                self._cached_tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                    for tool in tool_list
                ]
                logger.info(f"  Cached {len(self._cached_tool_schemas)} tool schemas from {tool_config_path} for logging")
            except Exception as e:
                logger.warning(f"  Failed to load tool schemas for logging: {e}")
                self._cached_tool_schemas = None

        logger.info(f"Initialized GeoguessrToolDataset with tool support")
        logger.info(f"  System prompt: {self.custom_system_prompt if self.custom_system_prompt else 'Using default'}")
        logger.info(f"  User prompt template: {self.custom_user_prompt_template if self.custom_user_prompt_template else 'Using default'}")

    def _build_messages(self, example: dict):
        """
        Build messages list for tool-enabled training.

        This creates a simple message list that will be processed by ToolAgentLoop.
        Tool definitions will be automatically injected by custom_chat_template.
        """
        # Get raw messages from example
        messages: list = example.pop(self.prompt_key)

        # Apply custom prompts if configured
        if self.custom_system_prompt is not None or self.custom_user_prompt_template is not None:
            modified_messages = []

            # Handle system message
            system_msg_found = False
            for msg in messages:
                if msg.get("role") == "system":
                    system_msg_found = True
                    if self.custom_system_prompt is not None:
                        modified_messages.append({
                            "role": "system",
                            "content": self.custom_system_prompt
                        })
                    else:
                        modified_messages.append(msg)
                    break

            # Add custom system message if not found
            if not system_msg_found and self.custom_system_prompt is not None:
                modified_messages.insert(0, {
                    "role": "system",
                    "content": self.custom_system_prompt
                })

            # Handle user messages
            for msg in messages:
                if msg.get("role") == "user":
                    original_content = msg["content"]

                    if self.custom_user_prompt_template is not None:
                        # Extract text parts, preserving <image>/<video> tags
                        parts = re.split(r'(<image>|<video>)', original_content)
                        text_only = ''.join([p for p in parts if p not in ['<image>', '<video>']])

                        # Apply template
                        new_text = self.custom_user_prompt_template.replace("{content}", text_only)
                        new_text = new_text.replace("{question}", text_only)

                        # Reconstruct content with tags
                        new_content = original_content
                        for part in parts:
                            if part not in ['<image>', '<video>', '']:
                                new_content = new_content.replace(part, new_text, 1)
                                break

                        modified_messages.append({
                            "role": "user",
                            "content": new_content
                        })
                    else:
                        modified_messages.append(msg)
                elif msg.get("role") not in ["system"]:
                    modified_messages.append(msg)

            messages = modified_messages

        # Process multimodal tags
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Get dataset item with tool support.

        Returns a dictionary compatible with ToolAgentLoop:
        - raw_prompt: List of messages (for apply_chat_template with tools)
        - tools_kwargs: Tool initialization parameters
        - agent_name: "tool_agent" to use ToolAgentLoop
        - multi_modal_data: Image/video data
        - Other standard fields (input_ids, attention_mask, etc.)
        """
        # Increase PIL limits
        from PIL import PngImagePlugin
        PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024 ** 2)
        PngImagePlugin.MAX_TEXT_MEMORY = 10 * (1024 ** 2)

        # Get row data
        import io
        from PIL import Image
        import verl.utils.torch_functional as verl_F

        row_dict: dict = self.dataframe[item]

        # Build messages (with custom prompts if configured)
        row_dict[self.prompt_key] = self._prepare_messages(row_dict)
        messages = self._build_messages(row_dict)

        # Process images
        images = None
        row_dict_images = row_dict.pop(self.image_key, None)
        if row_dict_images:
            from verl.utils.dataset.vision_utils import process_image
            images = [process_image(image, image_patch_size=self.image_patch_size) for image in row_dict_images]

        # Tokenization
        model_inputs = {}
        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            # Remove unused fields
            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # Prepare multi_modal_data
            multi_modal_data = {}
            if images:
                multi_modal_data["image"] = images

            row_dict["multi_modal_data"] = multi_modal_data

            # Add multi_modal_inputs if needed
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        # Post-process data
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Compute position IDs
        if self.processor is not None and "Qwen2VL" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            from verl.utils.model import compute_position_id_with_mask
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # Prepare raw_prompt_ids
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        # Return raw_chat if needed
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # Return full_prompts if needed
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        # ============================================================================
        # Tool-specific fields (KEY PART!)
        # ============================================================================

        # Get index from extra_info
        index = row_dict.get("extra_info", {}).get("index", 0)

        # Prepare tools_kwargs for tool initialization
        # Each tool needs its create_kwargs specified
        tools_kwargs = {
            "image_zoom_in_tool": {
                "create_kwargs": {
                    "image": images[0] if images else None
                },
            }
        }

        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"  # Critical: tells VERL to use ToolAgentLoop

        # For logging: generate prompt with tools for validation output
        # Use cached tool schemas to avoid repeated initialization
        if self.processor is not None and self._cached_tool_schemas is not None:
            try:
                # Generate prompt text with tools included
                raw_prompt_with_tools = self.processor.apply_chat_template(
                    messages,
                    tools=self._cached_tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False
                )
                row_dict["raw_prompt_text_with_tools"] = raw_prompt_with_tools
            except Exception as e:
                # If generation fails, use regular prompt
                row_dict["raw_prompt_text_with_tools"] = raw_prompt
        else:
            # If no processor or no cached schemas, use regular prompt
            row_dict["raw_prompt_text_with_tools"] = raw_prompt

        # ✅ Crop watermark for geochain data source
        # Check if data_source is 'geochain' and crop bottom 32 pixels
        data_source = row_dict.get('data_source', None)
        if 'geochain' in data_source:
            # Process pixel_values if present
            if 'pixel_values' in model_inputs and model_inputs.get('pixel_values') is not None:
                try:
                    import torch
                    pixel_values = model_inputs['pixel_values']

                    # Handle both single image and batched images
                    if isinstance(pixel_values, torch.Tensor):
                        # pixel_values shape: [num_images, channels, height, width] or [channels, height, width]
                        if pixel_values.dim() == 4:
                            # Batch of images: crop each image
                            model_inputs['pixel_values'] = pixel_values[:, :, :-32, :]
                            logger.debug(f"Cropped geochain watermark (batch): {pixel_values.shape} -> {model_inputs['pixel_values'].shape}")
                        elif pixel_values.dim() == 3:
                            # Single image: crop bottom 32 pixels
                            model_inputs['pixel_values'] = pixel_values[:, :-32, :]
                            logger.debug(f"Cropped geochain watermark: {pixel_values.shape} -> {model_inputs['pixel_values'].shape}")

                    # Also update multi_modal_inputs if present
                    if 'multi_modal_inputs' in row_dict and 'pixel_values' in row_dict['multi_modal_inputs']:
                        row_dict['multi_modal_inputs']['pixel_values'] = model_inputs['pixel_values']

                except Exception as e:
                    logger.warning(f"Failed to crop geochain watermark: {e}")

        elif 'data_source' not in row_dict:
            logger.error(f"data_source key not found in result: {row_dict}")
            exit(1)


        return row_dict

    def _prepare_messages(self, row_dict: dict) -> list:
        """
        Prepare initial messages from raw data.

        This extracts the prompt_key field from row_dict and returns it.
        If no custom prompt is configured, it uses the default from the data.
        """
        # Get messages from the data
        if self.prompt_key in row_dict:
            messages = row_dict[self.prompt_key]
            # Ensure it's a list of dicts
            if isinstance(messages, list):
                return messages

        # Fallback: create default messages structure
        # This assumes the data has a simple text prompt
        prompt_text = row_dict.get(self.prompt_key, "")
        if isinstance(prompt_text, str):
            # If it's just a string, wrap it in a message structure
            return [
                {
                    "role": "system",
                    "content": "You are a GeoGuessr expert. Analyze the image and identify the location coordinates."
                },
                {
                    "role": "user",
                    "content": f"<image>\n{prompt_text}"
                }
            ]

        return []
