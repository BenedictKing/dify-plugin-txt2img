from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from openai import OpenAI
from yarl import URL


class SeededitTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        credentials = self.runtime.credentials
        openai_api_key = credentials.get("openai_api_key")
        openai_base_url = credentials.get("openai_base_url")
        openai_base_url = str(URL(openai_base_url) / "v1")

        image_url = tool_parameters.get("image_url")
        # 构建model和messages
        model = "gpt-4o-all"
        messages = [
            {
                "role": "user",
                "content": tool_parameters["instruction"],
            }
        ]
        if image_url:
            model = "seededit"
            messages[0]["content"] = f"{image_url} {tool_parameters['instruction']}"

        client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=tool_parameters.get("stream", False),
            )

            if tool_parameters.get("stream", False):
                content = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                        yield self.create_text_message(chunk.choices[0].delta.content)
                yield self.create_json_message({"result": content.strip()})
            else:
                result = response.choices[0].message.content
                yield self.create_json_message({"result": result})

        except Exception as e:
            raise f"API Error: {str(e)}"
