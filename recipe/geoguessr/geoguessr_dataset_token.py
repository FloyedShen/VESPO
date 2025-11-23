"""
GeoGuessr TokenizedRLHFDataset - loads pre-tokenized data with dynamic prompt support

This dataset loads data preprocessed by convert_plain_rlvr2token.py, which includes:
1. Pre-processed vision features (pixel_values, image_grid_thw, etc.)
2. Original messages structure
3. Pre-computed default tokenization

Supports dynamic prompt modification at training time while avoiding expensive image preprocessing.
"""

import pickle
import logging
from typing import Optional
import torch
import numpy as np

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def deserialize_tensor(data: bytes) -> Optional[torch.Tensor]:
    """Deserialize bytes to tensor"""
    if data is None:
        return None

    obj = pickle.loads(data)
    if obj is None:
        return None

    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    elif isinstance(obj, torch.Tensor):
        return obj
    else:
        return obj


class GeoguessrTokenizedRLHFDataset(RLHFDataset):
    """
    Tokenized RLHF Dataset for Geoguessr with pre-processed vision features

    Loads pre-tokenized data with vision preprocessing already done.
    Supports dynamic prompt modification at training time via config parameters.

    Key features:
    - Fast loading: vision preprocessing already done
    - Dynamic prompts: can override system/user prompts at training time
    - Efficient: only re-tokenizes text if custom prompts are provided
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

        logger.info(f"Initialized GeoguessrTokenizedRLHFDataset")
        logger.info(f"  Custom system prompt: {self.custom_system_prompt if self.custom_system_prompt else 'Using default'}")
        logger.info(f"  Custom user prompt template: {self.custom_user_prompt_template if self.custom_user_prompt_template else 'Using default'}")
        logger.info(f"  Fast loading: vision features pre-processed ✓")

    def _needs_retokenization(self) -> bool:
        """Check if we need to re-tokenize (i.e., custom prompts are provided)"""
        return self.custom_system_prompt is not None or self.custom_user_prompt_template is not None

    def __getitem__(self, item):
        """
        Load pre-processed sample and optionally apply custom prompts

        If custom prompts are provided, re-tokenize text. Otherwise use cached tokenization.
        Vision features are always pre-processed (no runtime cost).
        """
        row_dict: dict = self.dataframe[item]

        # Get original messages (without any custom prompts applied)
        # Use get() instead of pop() to avoid modifying the cached dataframe
        original_messages = row_dict.get(self.prompt_key)

        # Deserialize vision data (already preprocessed!)
        vision_data = row_dict.get("vision_data", {})
        multi_modal_inputs = {}

        for key in ["pixel_values", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"]:
            if key in vision_data:
                tensor = deserialize_tensor(vision_data[key])
                if tensor is not None:
                    # Fix dimension for grid_thw fields: should be (1, 3) not (3,)
                    if key in ["image_grid_thw", "video_grid_thw"] and tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0)
                    multi_modal_inputs[key] = tensor

        # Check if we need to apply custom prompts and re-tokenize
        if self._needs_retokenization():
            # Apply custom prompts
            messages = self._apply_custom_prompts(original_messages)

            # Re-tokenize with custom prompts
            if self.processor is not None:
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )

                # Tokenize text only (vision already processed)
                model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = model_inputs["input_ids"]
                attention_mask = model_inputs["attention_mask"]
            else:
                raw_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
                model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = model_inputs["input_ids"]
                attention_mask = model_inputs["attention_mask"]

        else:
            # Use pre-computed tokenization (fastest path!)
            default_tokenization = row_dict.get("default_tokenization", {})

            input_ids = deserialize_tensor(default_tokenization.get("input_ids"))
            attention_mask = deserialize_tensor(default_tokenization.get("attention_mask"))

            if input_ids is not None and attention_mask is not None:
                # Unsqueeze to add batch dimension
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            else:
                # Fallback: re-tokenize with default prompts
                logger.warning(f"Sample {item}: Missing default tokenization, re-tokenizing")
                messages = original_messages

                if self.processor is not None:
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
                else:
                    raw_prompt = self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)

                input_ids = model_inputs["input_ids"]
                attention_mask = model_inputs["attention_mask"]

        # Postprocess (padding/truncation)
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Generate position_ids
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # Qwen2-VL m-rope position IDs
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=multi_modal_inputs.get("image_grid_thw"),
                video_grid_thw=multi_modal_inputs.get("video_grid_thw"),
                second_per_grid_ts=multi_modal_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)

            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)

        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            # GLM4-V position IDs
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=multi_modal_inputs.get("image_grid_thw"),
                video_grid_thw=multi_modal_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)

            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)

        else:
            # Standard position IDs
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # Add multi_modal_inputs (vision features)
        if self.return_multi_modal_inputs and multi_modal_inputs:
            row_dict["multi_modal_inputs"] = multi_modal_inputs

        # Raw prompt IDs (for reference)
        if self._needs_retokenization():
            messages = self._apply_custom_prompts(original_messages)
        else:
            messages = original_messages

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )

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
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} > {self.max_prompt_length}")

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        # Return raw chat and full prompts if requested
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages if self._needs_retokenization() else original_messages

        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        # Extra info
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()

        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)

        if need_tools_kwargs and not tools_kwargs:
            logger.warning(f"tools_kwargs is empty for index {index}")

        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        return row_dict

    def _apply_custom_prompts(self, original_messages: list) -> list:
        """
        Apply custom system/user prompts to original messages

        Args:
            original_messages: Original message list (may contain default prompts)

        Returns:
            Modified messages with custom prompts applied
        """
        # Extract original user content (typically the second message)
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
        if original_user_content and "<image>" not in original_user_content:
            original_user_content = "<image>\n\n" + original_user_content

        if self.custom_user_prompt_template is not None and original_user_content is not None:
            # Replace placeholder in template
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

        return new_messages
