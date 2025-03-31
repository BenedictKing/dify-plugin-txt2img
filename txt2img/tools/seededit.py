import base64
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
                last_url = image_urls[-1]

                if not any(last_url.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg")):
                    try:
                        # Download image and convert to base64
                        response = requests.get(last_url, timeout=30)
                        response.raise_for_status()
                        content_type = response.headers.get("Content-Type", "image/png")
                        base64_image = base64.b64encode(response.content).decode("utf-8")
                        data_uri = f"data:{content_type};base64,{base64_image}"
                        yield self.create_image_message(data_uri)
                    except Exception as e:
                        logger.error(f"Failed to process image URL: {e}")
                        yield self.create_text_message("图片处理失败，请尝试重新生成")
                else:
                    yield self.create_image_message(last_url)

                return
            yield self.create_text_message("当前流量限制，请稍后再试")

        except Exception as e:
            raise BaseException(f"API Error: {str(e)}")
