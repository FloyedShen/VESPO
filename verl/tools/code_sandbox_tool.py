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
Code sandbox tool adapter for verl.

Calls dots-tools' _code_sandbox_execute_async() and returns JSON format.
Supports image inputs similar to image_zoom_in_tool.
"""
from typing import Any

from dots_tools.tools.code_sandbox import _code_sandbox_execute_async

from .dots_tools_adapter import DotsToolAdapter
from .schemas import OpenAIFunctionToolSchema


class CodeSandboxTool(DotsToolAdapter):
    """Code sandbox tool - executes Python code with optional image inputs."""

    @staticmethod
    def get_default_schema() -> OpenAIFunctionToolSchema:
        """
        Get default OpenAI tool schema for code sandbox.

        Description and parameters adapted from distil_pipe.
        """
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "code_sandbox",
                "description": (
                    "Execute Python code in a sandbox environment. "
                    "Supports calculations, data processing, plotting, and image editing (PIL/OpenCV). "
                    "All images from the conversation are automatically injected as global variables (image_0, image_1, etc.). "
                    "These are PIL Image objects, ready to use directly. "
                    "Use plt.show() to return images. DO NOT use file I/O operations (.save(), .savefig(), Image.open())."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Python code to execute. "
                                "Images are available as image_0 (original), image_1, image_2, etc. (tool results). "
                                "Use PIL, OpenCV, matplotlib, numpy for image processing."
                            )
                        },
                        "session_id": {
                            "type": "string",
                            "default": "default",
                            "description": "Session ID for state persistence. Use same ID to share variables across calls."
                        },
                        "timeout": {
                            "type": "integer",
                            "default": 300,
                            "minimum": 5,
                            "maximum": 300,
                            "description": "Execution timeout in seconds (5-300). Default is 300."
                        }
                    },
                    "required": ["code"]
                }
            }
        })

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        if tool_schema is None:
            tool_schema = self.get_default_schema()
        super().__init__(config, tool_schema)

    def _prepare_instance_data(self, create_kwargs: dict) -> dict:
        """
        Store images for code_sandbox.

        Images are passed through create_kwargs and stored in instance_dict.
        They will be passed to _code_sandbox_execute_async().
        """
        return {
            "images": create_kwargs.get("images", None)
        }

    async def _call_dots_tool(self, parameters: dict[str, Any], instance_data: dict):
        """Call dots-tools code sandbox."""
        code = parameters.get("code", "")
        session_id = parameters.get("session_id", "default")
        timeout = max(5, min(parameters.get("timeout", 300), 300))

        # Get images from instance_data (stored during create)
        images = instance_data.get("images", None)

        return await _code_sandbox_execute_async(
            code=code,
            session_id=session_id,
            images=images,
            timeout=timeout
        )

    def _result_to_json(self, result) -> dict:
        """
        Convert CodeSandboxResult to JSON.

        Returns:
        {
            "stdout": "...",
            "stderr": "...",
            "return_value": "...",
            "execution_time": 0.123,
            "has_images": true
        }
        """
        return {
            "stdout": result.stdout if hasattr(result, 'stdout') else "",
            "stderr": result.stderr if hasattr(result, 'stderr') else "",
            "return_value": result.return_value if hasattr(result, 'return_value') else None,
            "execution_time": result.execution_time if hasattr(result, 'execution_time') else 0.0,
            "has_images": bool(hasattr(result, 'images') and result.images)
        }
