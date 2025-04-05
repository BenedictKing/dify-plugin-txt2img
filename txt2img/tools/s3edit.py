import hashlib
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
from tos.exceptions import TosServerError
from yarl import URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3editTool(Tool):
    @staticmethod
    def _is_image_url(url: str) -> bool:
        """使用正则验证URL路径中包含图片扩展名"""
        pattern = r"\.(png|jpe?g)(?=[^/]*$)"  # 匹配最后一个路径段的图片扩展名
        return bool(re.search(pattern, url, re.IGNORECASE))

    @staticmethod
    def _generate_headers() -> tuple[dict, dict]:
        """Generate browser-like headers and cookies for requests"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
        ]

        headers = {
            "User-Agent": random.choice(user_agents),
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "cross-site",
            "Referer": "https://www.google.com/",
        }

        cookies = {
            "appmsglist_action_3941382959": "card",
            "appmsglist_action_3941382968": "card",
            "pac_uid": f"{int(time.time())}_f{random.randint(10000, 99999)}",
            "rewardsn": "",
            "wxtokenkey": f"{random.randint(100000, 999999)}",
        }

        return headers, cookies

    def save_tos(self, credentials: dict, original_url: str) -> str:
        """Upload external resource to TOS and return new URL"""
        import tos

        # Download original resource
        headers, cookies = self._generate_headers()
        try:
            response = requests.get(original_url, headers=headers, cookies=cookies, timeout=60)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to download resource from {original_url}: {e}")
            return original_url  # Return original URL on failure

        # Generate content-hashed object key
        content_type = response.headers.get("Content-Type", "image/png").split("/")[-1]
        extension_map = {"jpeg": "jpg", "png": "png", "webp": "webp", "gif": "gif"}
        file_ext = extension_map.get(content_type.split(";")[0].lower(), "png")  # Handle charset params
        content_hash = hashlib.sha256(response.content).hexdigest()
        object_key = f"s3edit/{content_hash}.{file_ext}"

        bucket_name = credentials["VOLCENGINE_TOS_BUCKET_NAME"]

        # Initialize TOS client
        client = tos.TosClientV2(
            ak=credentials["VOLCENGINE_TOS_ACCESS_KEY"],
            sk=credentials["VOLCENGINE_TOS_SECRET_KEY"],
            endpoint=credentials["VOLCENGINE_TOS_ENDPOINT"],
            region=credentials["VOLCENGINE_TOS_REGION"],
        )

        # Check if object exists before uploading
        try:
            client.head_object(bucket=bucket_name, key=object_key)
            logger.info(f"File {content_hash} already exists in TOS bucket")
        except TosServerError as e:
            if e.status_code != 404:
                logger.error(f"TOS check error: {e}")
            # Upload if not exists
            client.put_object(bucket=bucket_name, key=object_key, content=response.content)
        except Exception as e:
            logger.error(f"Failed to save tos: {e}")
            return original_url  # Return original URL on failure

        # Return new TOS URL
        return f"https://{bucket_name}.{str(URL(credentials['VOLCENGINE_TOS_ENDPOINT']).host)}/{object_key}"

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
        image_format = tool_parameters.get("image_format", "text")  # Default to text format

        # Extract URLs from instruction text
        instruction_text = tool_parameters["instruction"]
        instruction_urls = re.findall(r"(https?://\S+)", instruction_text)
        # image_urls_from_instruction = [url for url in instruction_urls if self._is_image_url(url)]

        # Combine all image sources (uploaded files + URLs in instruction)
        uploaded_image_urls = [image.url for image in images] if images else []
        all_image_urls = list(dict.fromkeys(uploaded_image_urls + instruction_urls))

        # Process all URLs - convert non-image URLs to TOS
        processed_urls = []
        for url in all_image_urls:
            if self._is_image_url(url):
                processed_urls.append(url)
            else:
                try:
                    # Save to TOS and get new URL
                    new_url = self.save_tos(credentials=credentials, original_url=url)
                    processed_urls.append(new_url)
                except Exception as e:
                    logger.error(f"Failed to process URL {url}: {e}")
                    continue

        # Update messages based on processed URLs
        cleaned_instruction = re.sub(r"https?://\S+", "", instruction_text).strip()
        if image_format == "vision":
            content = [{"type": "text", "text": cleaned_instruction}]
            for url in processed_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})
            messages[0]["content"] = content
        else:
            url_prefix = " ".join(processed_urls)
            messages[0]["content"] = f"{url_prefix} {instruction_text}".strip()
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
                                    yield self.create_text_message(chunk_content)
                            except json.JSONDecodeError:
                                logger.warning(f"无法解析JSON: {json_str}")

            else:
                # 非流式响应直接获取内容
                content = response.json()["choices"][0]["message"]["content"]
                yield self.create_text_message(content)

            logger.info(content)
            image_urls = re.findall(r"!\[.*?\]\((https?://[^\s)]+)", content)
            if image_urls:
                last_url = image_urls[-1]

                if not self._is_image_url(last_url):
                    logger.info(f"unknown image url: {last_url}")
                    try:
                        headers, cookies = self._generate_headers()
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
