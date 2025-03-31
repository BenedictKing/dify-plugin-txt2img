import json
import logging
import re
from collections.abc import Generator
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from yarl import URL


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeededitTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        credentials = self.runtime.credentials
        openai_api_key = credentials.get("openai_api_key")
        openai_base_url = credentials.get("openai_base_url")
        openai_url = str(URL(openai_base_url) / "v1/chat/completions")

        image_url = tool_parameters.get("image_url")
        model = tool_parameters.get("model", "gpt-4o-all")
        stream = bool(tool_parameters.get("stream", False) in ["True", "true", "TRUE", True])
        messages = [
            {
                "role": "user",
                "content": tool_parameters["instruction"],
            },
        ]
        if image_url:
            messages[0]["content"] = f"{image_url} {tool_parameters['instruction']}"

        try:
            # 发送API请求
            openai_payload = {"model": model, "messages": messages, "stream": stream}
            headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
            logger.info({"url": openai_url, "headers": headers, "payload": openai_payload})
            response = requests.post(openai_url, headers=headers, json=openai_payload, timeout=60, stream=stream)
            response.raise_for_status()

            content = ""
            if stream:
                # 如果是流式响应，需要拼接内容
                for line in response.iter_lines():
                    logger.info(line)
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
                            except json.JSONDecodeError:
                                logger.warning(f"无法解析JSON: {json_str}")

            else:
                # 非流式响应直接获取内容
                content = response.json()["choices"][0]["message"]["content"]

            logger.info(content)
            image_urls = re.findall(r"!\[.*?\]\((https?://[^\s)]+)", content)
            if image_urls:
                # 如果image_urls[-1]中没有.png或者.jpg的文件名后缀，需要下载下来转换成base64格式给到下面的函数中 AI!
                yield self.create_image_message(image_urls[-1])
                return
            yield self.create_text_message("当前流量限制，请稍后再试")

        except Exception as e:
            raise BaseException(f"API Error: {str(e)}")
