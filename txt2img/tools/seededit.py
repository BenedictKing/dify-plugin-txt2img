import json
import logging
import random
import re
import time
from collections.abc import Generator
from typing import Any

import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from yarl import URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeededitTool(Tool):
    @staticmethod
    def _is_image_url(url: str) -> bool:
        """使用正则验证URL路径中包含图片扩展名"""
        pattern = r"\.(png|jpe?g)(?=[^/]*$)"  # 匹配最后一个路径段的图片扩展名
        return bool(re.search(pattern, url, re.IGNORECASE))

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        credentials = self.runtime.credentials
        openai_api_key = credentials.get("openai_api_key")
        openai_base_url = credentials.get("openai_base_url")
        openai_url = str(URL(openai_base_url) / "v1/chat/completions")

        image_files = tool_parameters.get("image_files")
        images = [i for i in image_files if i.type == "image"]

        model = tool_parameters.get("model", "gpt-4o-all")
        stream = bool(tool_parameters.get("stream", False) in ["True", "true", "TRUE", True])
        messages = [
            {
                "role": "user",
                "content": tool_parameters["instruction"],
            },
        ]
        if images:
            image_format = tool_parameters.get("image_format", "text")  # Default to text format

            if image_format == "vision":
                messages[0]["content"] = [{"type": "text", "text": tool_parameters["instruction"]}]
                for image in images:
                    messages[0]["content"].append({"type": "image_url", "image_url": {"url": image.url}})
            else:  # Default text format
                # Concatenate all image URLs with instruction
                image_urls = " ".join([image.url for image in images])
                messages[0]["content"] = f"{image_urls} {tool_parameters['instruction']}"
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

                if not self._is_image_url(last_url):
                    logger.info(f"unknown image url: {last_url}")
                    try:
                        # 随机选择一个User-Agent，模拟不同浏览器
                        user_agents = [
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
                            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
                        ]
                        selected_ua = random.choice(user_agents)

                        # 构建更真实的请求头
                        headers = {
                            "User-Agent": selected_ua,
                            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                            "Accept-Encoding": "gzip, deflate, br",
                            "Connection": "keep-alive",
                            "Sec-Fetch-Dest": "image",
                            "Sec-Fetch-Mode": "no-cors",
                            "Sec-Fetch-Site": "cross-site",
                            "Referer": "https://www.google.com/",
                        }
                        # 添加必要的Cookie参数，减少被检测的可能性
                        cookies = {
                            "appmsglist_action_3941382959": "card",  # 一些随机的Cookie值
                            "appmsglist_action_3941382968": "card",
                            "pac_uid": f"{int(time.time())}_f{random.randint(10000, 99999)}",
                            "rewardsn": "",
                            "wxtokenkey": f"{random.randint(100000, 999999)}",
                        }
                        # Download image and convert to base64
                        response = requests.get(last_url, headers=headers, cookies=cookies, timeout=30)
                        response.raise_for_status()
                        content_type = response.headers.get("Content-Type", "image/png")

                        yield self.create_blob_message(blob=response.content, meta={"mime_type": content_type})
                    except Exception as e:
                        logger.error(f"Failed to process image URL: {e}")
                        yield self.create_text_message("图片处理失败，请尝试重新生成")
                else:
                    yield self.create_image_message(last_url)

                return
            yield self.create_text_message("当前流量限制，请稍后再试")

        except Exception as e:
            raise BaseException(f"API Error: {str(e)}")
