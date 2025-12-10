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
        return super().__getitem__(item)
