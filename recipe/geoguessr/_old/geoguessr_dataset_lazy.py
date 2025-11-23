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

import io
import logging
from typing import Optional

from PIL import Image
from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


class GeoguessrRLHFDatasetLazy(RLHFDataset):
    """
    Lazy preprocessing version of GeoguessrRLHFDataset.

    Key difference: Image preprocessing is deferred to rollout workers,
    enabling distributed preprocessing across all GPU nodes instead of
    centralizing it on the TaskRunner node.

    Benefits:
    - Reduces memory pressure on TaskRunner node
    - Distributes image preprocessing across 64 GPUs
    - Enables larger batch sizes without OOM
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

        logger.info(f"Initialized GeoguessrRLHFDatasetLazy with LAZY preprocessing")
        # print(f"  System prompt: {self.custom_system_prompt if self.custom_system_prompt else 'Using default'}")
        # print(f"  User prompt template: {self.custom_user_prompt_template if self.custom_user_prompt_template else 'Using default'}")
        logger.info(f"  System prompt: {self.custom_system_prompt if self.custom_system_prompt else 'Using default'}")
        logger.info(f"  User prompt template: {self.custom_user_prompt_template if self.custom_user_prompt_template else 'Using default'}")

    def __getitem__(self, item):
        """
        Override to skip image preprocessing - only load raw images.
        Preprocessing will be done on rollout workers.
        """
        row_dict: dict = self.dataframe[item]

        # Get the original messages
        original_messages = row_dict.pop(self.prompt_key)

        from PIL import PngImagePlugin

        # Increase PIL limits to handle large ICC profiles in PNG images
        PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024 ** 2)  # 10MB
        PngImagePlugin.MAX_TEXT_MEMORY = 10 * (1024 ** 2)  # 10MB

        # If custom prompts are configured, replace them
        if self.custom_system_prompt is not None or self.custom_user_prompt_template is not None:
            # Extract original user content
            original_user_content = None
            if len(original_messages) >= 2 and original_messages[1].get("role") == "user":
                original_user_content = original_messages[1]["content"]
            elif len(original_messages) >= 1 and original_messages[0].get("role") == "user":
                original_user_content = original_messages[0]["content"]

            # Build new messages with custom prompts
            new_messages = []

            # Add system message
            if self.custom_system_prompt is not None:
                new_messages.append({
                    "role": "system",
                    "content": self.custom_system_prompt
                })
            elif len(original_messages) >= 1 and original_messages[0].get("role") == "system":
                new_messages.append(original_messages[0])

            # Add user message
            if "<image>" not in original_user_content:
                original_user_content = "<image>\n\n" + original_user_content
            if self.custom_user_prompt_template is not None and original_user_content is not None:
                user_content = self.custom_user_prompt_template.replace("{content}", original_user_content)
                user_content = user_content.replace("{question}", original_user_content)
                new_messages.append({
                    "role": "user",
                    "content": user_content
                })
            elif original_user_content is not None:
                new_messages.append({
                    "role": "user",
                    "content": original_user_content
                })

            row_dict[self.prompt_key] = new_messages
        else:
            row_dict[self.prompt_key] = original_messages

        # Build messages
        messages = self._build_messages(row_dict)
        # print(messages)

        if self.processor is not None:
            # Generate text prompt
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )

        # ============ KEY CHANGE: Only load raw images, don't preprocess ============
        multi_modal_data = {}
        row_dict_images = row_dict.pop(self.image_key, None)
        if row_dict_images:
            # Keep raw image data (dict with "bytes" or PIL Images)
            # DON'T preprocess here - let distributed workers do it
            multi_modal_data["image"] = row_dict_images

        # Store raw multi_modal_data for distributed preprocessing
        row_dict["multi_modal_data"] = multi_modal_data

        # For Qwen2-VL models, we need image_grid_thw for position_ids generation
        # But we defer get_rope_index call to collate stage for better performance
        image_grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        # For vision-language models, use processor to tokenize (includes image tokens)
        # For text-only models, use tokenizer
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # Process images if they exist
            if row_dict_images:
                from verl.utils.dataset.vision_utils import process_image

                # Convert image bytes/dict to PIL Images
                pil_images = []
                for img_data in row_dict_images:
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        pil_images.append(Image.open(io.BytesIO(img_data["bytes"])))
                    elif isinstance(img_data, Image.Image):
                        pil_images.append(img_data)

                # Process images
                processed_images = [process_image(img, image_patch_size=self.image_patch_size) for img in pil_images]

                # Use processor to tokenize text + images (this includes image tokens)
                temp_inputs = self.processor(
                    text=[raw_prompt],
                    images=processed_images,
                    return_tensors="pt"
                )

                # Use processor's input_ids and attention_mask (includes image tokens)
                input_ids = temp_inputs.pop("input_ids")
                attention_mask = temp_inputs.pop("attention_mask")

                # Save grid info for collate stage
                image_grid_thw = temp_inputs.get("image_grid_thw")
                video_grid_thw = temp_inputs.get("video_grid_thw")
                second_per_grid_ts = temp_inputs.get("second_per_grid_ts")

                # Save multi_modal_inputs for training (similar to original dataset)
                # Remove second_per_grid_ts as it's only for mrope, not training
                if "second_per_grid_ts" in temp_inputs:
                    temp_inputs.pop("second_per_grid_ts")

                # Store multi_modal_inputs as dict (not BatchFeature)
                if len(temp_inputs) > 0:
                    row_dict["multi_modal_inputs"] = dict(temp_inputs)
            else:
                # No images, use tokenizer for text-only
                model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = model_inputs.pop("input_ids")
                attention_mask = model_inputs.pop("attention_mask")
        else:
            # Non-VL model or text-only, use tokenizer
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        # Pad/truncate to max_prompt_length
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Use simple position_ids for now - will be regenerated in collate_fn if needed
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # Save grid info for collate stage to regenerate position_ids
        if image_grid_thw is not None:
            row_dict["image_grid_thw"] = image_grid_thw
        if video_grid_thw is not None:
            row_dict["video_grid_thw"] = video_grid_thw
        if second_per_grid_ts is not None:
            row_dict["second_per_grid_ts"] = second_per_grid_ts

        # Raw prompt IDs
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        # Add index
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
