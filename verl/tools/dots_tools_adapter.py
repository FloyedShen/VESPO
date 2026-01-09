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
Simplified adapter for dots-tools integration.

Key principles:
1. Only wrap the interface, don't implement logic
2. Call dots-tools' _xxx_async() functions directly
3. Return JSON format (not flatten text)
4. Handle images similar to image_zoom_in_tool
5. Retry on HTTP 500 errors (max 3 attempts)
"""
import asyncio
import json
from abc import abstractmethod
from typing import Any, Optional
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse


class DotsToolAdapter(BaseTool):
    """
    Simplified adapter base class for dots-tools.

    Subclasses only need to:
    1. Define tool schema
    2. Implement _call_dots_tool() to call the specific _xxx_async() function
    3. Implement _result_to_json() to convert BaseToolResult to JSON dict
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}  # Store per-instance data (e.g., images)

    @abstractmethod
    async def _call_dots_tool(self, parameters: dict[str, Any], instance_data: dict) -> Any:
        """
        Call the underlying dots-tools async function.

        Args:
            parameters: Tool parameters from model
            instance_data: Instance-specific data (e.g., images from create)

        Returns:
            BaseToolResult from dots-tools
        """
        pass

    @abstractmethod
    def _result_to_json(self, result: Any) -> dict:
        """
        Convert dots-tools result to JSON dict.

        Args:
            result: BaseToolResult from dots-tools

        Returns:
            JSON-serializable dict (will be converted to JSON string)
        """
        pass

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Create tool instance and store any necessary data (e.g., images).

        Args:
            instance_id: Optional instance ID
            **kwargs: Should contain 'create_kwargs' with tool-specific data

        Returns:
            (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Extract create_kwargs
        create_kwargs = kwargs.get("create_kwargs", {})

        # Store instance data (subclasses can override to store specific data)
        self._instance_dict[instance_id] = self._prepare_instance_data(create_kwargs)

        return instance_id, ToolResponse()

    def _prepare_instance_data(self, create_kwargs: dict) -> dict:
        """
        Prepare instance data from create_kwargs.

        Override this in subclasses if needed (e.g., for code_sandbox to store images).

        Args:
            create_kwargs: Data passed during create()

        Returns:
            Dict to store in _instance_dict
        """
        return {}

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute tool by calling dots-tools and returning JSON.
        Retries up to 3 times on HTTP 500 errors with exponential backoff.

        Args:
            instance_id: Tool instance ID
            parameters: Tool parameters from model
            **kwargs: Additional context (usually not needed)

        Returns:
            (ToolResponse, reward, metrics)
        """
        max_retries = 3
        retry_delays = [0.5, 1.0, 2.0]  # Exponential backoff

        for attempt in range(max_retries):
            try:
                # Get instance data
                instance_data = self._instance_dict.get(instance_id, {})

                # Step 1: Call dots-tools async function
                result = await self._call_dots_tool(parameters, instance_data)

                # Step 2: Check if successful
                if not result.success:
                    error_msg = result.error or "Unknown error"

                    # Check if it's an HTTP 500 error
                    is_http_500 = (
                        "HTTP 500" in error_msg or
                        "500 Internal Server Error" in error_msg or
                        "Internal Server Error" in error_msg
                    )

                    # Retry on HTTP 500 errors
                    if is_http_500 and attempt < max_retries - 1:
                        await asyncio.sleep(retry_delays[attempt])
                        continue  # Retry

                    # Return error if not retryable or max retries reached
                    return (
                        ToolResponse(text=json.dumps({"error": error_msg}, ensure_ascii=False)),
                        0.0,
                        {"success": False, "error": error_msg, "attempts": attempt + 1}
                    )

                # Step 3: Convert result to JSON dict
                result_dict = self._result_to_json(result)

                # Step 4: Convert to JSON string
                result_json = json.dumps(result_dict, ensure_ascii=False, indent=2)

                # Step 5: Handle images if result has them (for code_sandbox)
                response_kwargs = {"text": result_json}
                if hasattr(result, 'images') and result.images:
                    response_kwargs["image"] = result.images

                return (
                    ToolResponse(**response_kwargs),
                    0.0,
                    {"success": True, "tool_name": self.name, "attempts": attempt + 1}
                )

            except Exception as e:
                error_msg = str(e)

                # Check if it's an HTTP 500 related exception
                is_http_500 = (
                    "500" in error_msg or
                    "Internal Server Error" in error_msg
                )

                # Retry on HTTP 500 errors
                if is_http_500 and attempt < max_retries - 1:
                    await asyncio.sleep(retry_delays[attempt])
                    continue  # Retry

                # Return error if not retryable or max retries reached
                error_json = json.dumps({"error": error_msg}, ensure_ascii=False)
                return (
                    ToolResponse(text=error_json),
                    0.0,
                    {"success": False, "error": error_msg, "attempts": attempt + 1}
                )

        # Should never reach here, but just in case
        error_json = json.dumps({"error": "Max retries exceeded"}, ensure_ascii=False)
        return (
            ToolResponse(text=error_json),
            0.0,
            {"success": False, "error": "Max retries exceeded", "attempts": max_retries}
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release tool instance and clean up data."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
