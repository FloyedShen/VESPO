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
Wikipedia search tool adapter for verl.

Calls dots-tools' _search_wikipedia_async() and returns JSON format.
"""
from typing import Any

from dots_tools.tools.search_wikipedia import _search_wikipedia_async

from .dots_tools_adapter import DotsToolAdapter
from .schemas import OpenAIFunctionToolSchema


class SearchWikipediaTool(DotsToolAdapter):
    """Wikipedia search tool - calls dots-tools and returns JSON."""

    @staticmethod
    def get_default_schema() -> OpenAIFunctionToolSchema:
        """
        Get default OpenAI tool schema for Wikipedia search.

        Description and parameters adapted from distil_pipe.
        """
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "search_wikipedia",
                "description": "Use this tool to find factual information, historical context, scientific explanations, or general knowledge from Wikipedia articles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query in natural language. Be specific and use keywords. Examples: 'machine learning', 'Eiffel Tower', 'photosynthesis process'"
                        },
                        "k": {
                            "type": "integer",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                            "description": "Number of results to return (1-20). Default is 5. Use fewer results (1-3) for specific queries, more (5-10) for exploration."
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
        """Call dots-tools Wikipedia search."""
        query = parameters.get("query", "")
        k = max(1, min(parameters.get("k", 5), 20))

        return await _search_wikipedia_async(query=query, k=k)

    def _result_to_json(self, result) -> dict:
        """
        Convert WikipediaSearchResult to JSON dict.

        Returns:
        {
            "query": "Paris",
            "results": [
                {"title": "...", "text": "...", "score": 0.95},
                ...
            ],
            "total_count": 2
        }

        Note: Limits each result's text to 800 characters to avoid tool response truncation.
        With k=5 results, total response will be ~4000 chars, well under the 1024 token limit.
        """
        # Limit text length per result to avoid truncation issues
        max_text_length = 800

        filtered_results = []
        for raw_result in result.results:
            filtered = {
                "title": raw_result.get("title", ""),
                "score": raw_result.get("score", 0.0),
            }

            # Truncate text field if too long
            text = raw_result.get("text", "")
            if len(text) > max_text_length:
                filtered["text"] = text[:max_text_length] + "..."
            else:
                filtered["text"] = text

            filtered_results.append(filtered)

        return {
            "query": result.query,
            "results": filtered_results,
            "total_count": result.total_count
        }
