import json
import re
from collections.abc import Generator
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from yarl import URL

# mock_result = '{\n  "prompt": "A Tibetan Mastiff walking on a zebra crossing along with a golden retriever and a dachshund.",\n  "referenced_image_ids": ["file-K51ZztW2MdH2z9J6gQzxR7"]\n}\n\n> 进度 **2%**\n\n\n> 进度 **34%**\n\n\n> 进度 **67%**\n\n\n\n![file_000000007a5451f7b4c098cb3d95dace](https://filesystem.site/cdn/20250330/JfAL6qr7iuQsq0UwLSfK8W2XvP1rgv.png)\n[下载⏬](https://filesystem.site/cdn/download/20250330/JfAL6qr7iuQsq0UwLSfK8W2XvP1rgv.png)\n\n'
# mock_failed_result = '{\n  "prompt": "A tiger in the center of the image, replacing the dog, while maintaining the rest of the scene as is.",\n  "size": "1024x1024"\n}I wasn\'t able to generate the image because there\'s a rate limit in place. Please wait for a little while before I can assist with your request again. Feel free to let me know if you\'d like to modify the request or need anything else!'


class SeededitTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        credentials = self.runtime.credentials
        openai_api_key = credentials.get("openai_api_key")
        openai_base_url = credentials.get("openai_base_url")
        openai_url = str(URL(openai_base_url) / "v1/chat/completions")

        image_url = tool_parameters.get("image_url")
        model = tool_parameters.get("model", "gpt-4o-all")
        stream = tool_parameters.get("stream", False)
        messages = [
            {
                "role": "user",
                "content": tool_parameters["instruction"],
            },
        ]
        if image_url:
            messages[0]["content"] = f"{image_url} {tool_parameters['instruction']}"
        print(f"base: {openai_base_url}, model: {model}, stream: {stream} , messages: {messages}")

        try:
            openai_payload = {"model": model, "messages": messages}

            # 发送API请求
            headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
            response = requests.post(openai_url, headers=headers, json=openai_payload, timeout=300, stream=stream)
            response.raise_for_status()

            if stream:
                # 如果是流式响应，需要拼接内容
                content = ""
                for line in response.iter_lines():
                    if line:
                        # 移除"data: "前缀并解析JSON
                        line_text = line.decode("utf-8")
                        if line_text.startswith("data: "):
                            json_str = line_text[6:]
                            if json_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(json_str)
                                if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                                    chunk_content = chunk["choices"][0]["delta"]["content"]
                                    content += chunk_content
                                    print(chunk_content)
                            except json.JSONDecodeError:
                                print(f"无法解析JSON: {json_str}")
                answer = content
            else:
                # 非流式响应直接获取内容
                answer = response.json()["choices"][0]["message"]["content"]

            print(answer)
            image_urls = re.findall(r"!\[.*?\]\((https?://[^\s)]+)", answer)
            if image_urls:
                yield self.create_image_message(image_urls[-1])
                return
            yield self.create_text_message("当前流量限制，请稍后再试")

        except Exception as e:
            raise BaseException(f"API Error: {str(e)}")
