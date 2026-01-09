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
Geocode tool adapter for verl.

Calls dots-tools' _search_nominatim_async() and returns JSON format.
"""
from typing import Any

from dots_tools.tools.geocode_nominatim import _search_nominatim_async as _geocode_async

from .dots_tools_adapter import DotsToolAdapter
from .schemas import OpenAIFunctionToolSchema


class GeocodeTool(DotsToolAdapter):
    """Geocode tool - converts place names to coordinates."""

    @staticmethod
    def get_default_schema() -> OpenAIFunctionToolSchema:
        """
        Get default OpenAI tool schema for geocoding.

        Description and parameters adapted from distil_pipe.
        """
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "geocode",
                "description": "Search for places by name or address using geocoding service. Converts place names, addresses, or landmarks into geographic coordinates and detailed location information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query - can be a place name, address, or landmark. Be specific and include context when possible. Examples: 'Eiffel Tower', 'Tokyo Tower', 'Central Park'"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Maximum number of results to return (1-10). Default is 5."
                        }
                    },
                    "required": ["query"]
                }
            }
        })

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        if tool_schema is None:
            tool_schema = self.get_default_schema()
        super().__init__(config, tool_schema)

    async def _call_dots_tool(self, parameters: dict[str, Any], instance_data: dict):
        """Call dots-tools geocode."""
        query = parameters.get("query", "")
        limit = max(1, min(parameters.get("limit", 5), 10))

        return await _geocode_async(query=query, limit=limit)

    def _result_to_json(self, result) -> dict:
        """
        Convert NominatimSearchResult to JSON.

        Returns:
        {
            "query": "Eiffel Tower, Paris",
            "results": [
                {
                    "name": "Tour Eiffel",
                    "display_name": "...",
                    "lat": "48.8584",
                    "lon": "2.2945",
                    "type": "attraction",
                    "class": "tourism",
                    "address": {...},
                    "boundingbox": [...]
                },
                ...
            ],
            "total_count": 3
        }

        Note: Filters out useless fields (licence, place_id, osm_type, osm_id, importance)
        to reduce response size and avoid truncation issues.
        """
        # Fields to keep from Nominatim response
        useful_fields = {"name", "display_name", "lat", "lon", "type", "class", "address", "boundingbox"}

        # Filter each result to only include useful fields
        filtered_results = []
        for raw_result in result.results:
            filtered = {k: v for k, v in raw_result.items() if k in useful_fields}
            filtered_results.append(filtered)

        return {
            "query": result.query,
            "results": filtered_results,
            "total_count": result.total_count
        }
