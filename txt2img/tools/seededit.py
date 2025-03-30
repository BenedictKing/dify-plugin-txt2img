from collections.abc import Generator
from typing import Any
import re
import openai
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from openai import OpenAI
from yarl import URL

# mock_result = '{\n  "prompt": "A Tibetan Mastiff walking on a zebra crossing along with a golden retriever and a dachshund.",\n  "referenced_image_ids": ["file-K51ZztW2MdH2z9J6gQzxR7"]\n}\n\n> 进度 **2%**\n\n\n> 进度 **34%**\n\n\n> 进度 **67%**\n\n\n\n![file_000000007a5451f7b4c098cb3d95dace](https://filesystem.site/cdn/20250330/JfAL6qr7iuQsq0UwLSfK8W2XvP1rgv.png)\n[下载⏬](https://filesystem.site/cdn/download/20250330/JfAL6qr7iuQsq0UwLSfK8W2XvP1rgv.png)\n\n'
# mock_failed_result = '{\n  "prompt": "A tiger in the center of the image, replacing the dog, while maintaining the rest of the scene as is.",\n  "size": "1024x1024"\n}I wasn\'t able to generate the image because there\'s a rate limit in place. Please wait for a little while before I can assist with your request again. Feel free to let me know if you\'d like to modify the request or need anything else!'


class SeededitTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        credentials = self.runtime.credentials
        openai_api_key = credentials.get("openai_api_key")
        openai_base_url = credentials.get("openai_base_url")
        openai_base_url = str(URL(openai_base_url) / "v1")

        image_url = tool_parameters.get("image_url")
        model = "gpt-4o-all"
        messages = [
            {
                "role": "user",
                "content": tool_parameters["instruction"],
            }
        ]
        if image_url:
            messages[0]["content"] = f"{image_url} {tool_parameters['instruction']}"
        else:
            sp = tool_parameters["instruction"].split(" ")
            if sp[0].startswith("http"):
                messages[0]["content"] = tool_parameters["instruction"]

        client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
            )

            # if tool_parameters.get("stream", False):
            #     content = ""
            #     for chunk in response:
            #         if chunk.choices and chunk.choices[0].delta.content:
            #             content += chunk.choices[0].delta.content
            #             yield self.create_text_message(chunk.choices[0].delta.content)
            #     yield self.create_json_message({"result": content.strip()})
            # else:
            result = response.choices[0].message.content
            # result = mock_failed_result
            # Extract image URLs from markdown format
            image_urls = re.findall(r"!\[.*?\]\((https?://[^\s)]+)", result)
            if image_urls:
                yield self.create_image_message(image_urls[0])
                return
            yield self.create_text_message("当前流量限制，请稍后再试")

        except openai.error.OpenAIError as e:
            # 通过异常对象获取状态码
            # print(f"Error Status Code: {e.status_code}")
            # print(f"Error Message: {e}")
            raise BaseException(f"API Error: {str(e)}")
        except Exception as e:
            raise BaseException(f"API Error: {str(e)}")
