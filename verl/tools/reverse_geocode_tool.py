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
Reverse geocode tool adapter for verl.

Calls dots-tools' _reverse_geocode_nominatim_async() and returns JSON format.
"""
from typing import Any

from dots_tools.tools.geocode_nominatim import (
    _reverse_geocode_nominatim_async as _reverse_geocode_async,
    NominatimReverseResult,
)

from .dots_tools_adapter import DotsToolAdapter
from .schemas import OpenAIFunctionToolSchema


class ReverseGeocodeTool(DotsToolAdapter):
    """Reverse geocode tool - converts coordinates to place names."""

    @staticmethod
    def get_default_schema() -> OpenAIFunctionToolSchema:
        """
        Get default OpenAI tool schema for reverse geocoding.

        Description and parameters adapted from distil_pipe.
        """
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "reverse_geocode",
                "description": "Convert geographic coordinates into a human-readable address using reverse geocoding. Takes latitude and longitude coordinates and returns detailed location information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "minimum": -90,
                            "maximum": 90,
                            "description": "Latitude coordinate in decimal degrees (range: -90 to 90). Positive values are North, negative values are South."
                        },
                        "longitude": {
                            "type": "number",
                            "minimum": -180,
                            "maximum": 180,
                            "description": "Longitude coordinate in decimal degrees (range: -180 to 180). Positive values are East, negative values are West."
                        },
                        "zoom": {
                            "type": "integer",
                            "default": 18,
                            "minimum": 0,
                            "maximum": 18,
                            "description": "Detail level (0-18). Higher values provide more detailed results. Default is 18."
                        }
                    },
                    "required": ["latitude", "longitude"]
                }
            }
        })

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        if tool_schema is None:
            tool_schema = self.get_default_schema()
        super().__init__(config, tool_schema)

    async def _call_dots_tool(self, parameters: dict[str, Any], instance_data: dict):
        """Call dots-tools reverse geocode."""
        latitude = parameters.get("latitude")
        longitude = parameters.get("longitude")
        zoom = max(0, min(parameters.get("zoom", 18), 18))

        # Validate required parameters
        if latitude is None or longitude is None:
            return NominatimReverseResult(
                success=False,
                error="Missing required parameters: latitude and longitude",
                latitude=latitude or 0.0,
                longitude=longitude or 0.0,
                zoom=zoom
            )

        return await _reverse_geocode_async(latitude=latitude, longitude=longitude, zoom=zoom)

    def _result_to_json(self, result) -> dict:
        """
        Convert NominatimReverseResult to JSON.

        Returns:
        {
            "latitude": 48.8584,
            "longitude": 2.2945,
            "zoom": 18,
            "result": {
                "name": "Tour Eiffel",
                "display_name": "...",
                "type": "attraction",
                "class": "tourism",
                "address": {...},
                "boundingbox": [...]
            }
        }

        Note: Filters out useless fields (licence, place_id, osm_type, osm_id, importance)
        to reduce response size and avoid truncation issues.
        """
        # Fields to keep from Nominatim response
        useful_fields = {"name", "display_name", "type", "class", "address", "boundingbox"}

        # Filter the result to only include useful fields
        filtered_result = None
        if result.result:
            filtered_result = {k: v for k, v in result.result.items() if k in useful_fields}

        return {
            "latitude": result.latitude,
            "longitude": result.longitude,
            "zoom": result.zoom,
            "result": filtered_result
        }
