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

logging.basicConfig(level=logging.DEBUG)
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

    @staticmethod
    def _is_valid_image(content: bytes) -> tuple[bool, str]:
        """Validate image content using magic numbers, returns (is_valid, detected_type)"""
        try:
            if len(content) < 12:
                return False, ""

            if content.startswith(b"\xff\xd8\xff"):
                return True, "image/jpeg"
            if content.startswith(b"\x89PNG\r\n\x1a\n"):
                return True, "image/png"
            if content[0:4] == b"RIFF" and content[8:12] == b"WEBP":
                return True, "image/webp"
            if content[0:3] == b"GIF":
                return True, "image/gif"
            if len(content) > 8 and content[1:4] == b"PNG":
                return True, "image/png"

        except Exception as e:
            logger.error(f"Image validation error: {e}")

        return False, ""

    FILE_TYPE_MAP = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif"}

    def save_tos(self, credentials: dict, original_url: str) -> str:
        """Upload external resource to TOS and return new URL"""
        import tos

        # Download original resource
        headers, cookies = self._generate_headers()
        try:
            logger.info(f"Downloading external resource from {original_url}")
            response = requests.get(original_url, headers=headers, cookies=cookies, timeout=60)
            logger.info(f"Download completed [status={response.status_code}, content_length={len(response.content)}]")
            response.raise_for_status()

            # Add image validation check
            if not response.content:
                logger.error(f"Empty content from {original_url}")
                return original_url

            # Perform validation
            is_valid, detected_type = self._is_valid_image(response.content)
            if not is_valid:
                logger.error(f"Invalid image content from {original_url} (Detected type: {detected_type})")
                return original_url

        except Exception as e:
            logger.error(f"Failed to download resource from {original_url}: {e}")
            return original_url  # Return original URL on failure

        # Generate content-hashed object key using detected type
        file_ext = self.FILE_TYPE_MAP.get(detected_type, "png")
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
            logger.info("Checking TOS object existence [bucket=%s, key=%s]", bucket_name, object_key)
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
        conversation_id = tool_parameters.get("conversation_id")
        dialogue_count = tool_parameters.get("dialogue_count", 0)
        image_files = tool_parameters.get("image_files")
        images = [i for i in image_files if i.type == "image"]

        processed_urls = []
        instruction_to_use = tool_parameters["instruction"]  # Default to new instruction
        is_retry = False
        storage_key = f"s3edit_history_{conversation_id}"
        # Check for existing history entry for this specific dialogue_count at the beginning

        history = []
        try:
            existing_data = self.session.storage.get(storage_key)
            if existing_data:
                logger.info("Loading conversation history [conversation_id=%s, exists=True]", conversation_id)
                history = json.loads(existing_data.decode())
                # Only keep entries with dialogue_count <= current count
                lens = len(history)
                history = [entry for entry in history if entry.get("dialogue_count", 0) <= dialogue_count]
                if len(history) < lens:
                    self.session.storage.set(storage_key, json.dumps(history).encode())
                logger.info("Loaded conversation history [conversation_id=%s]\n%s", conversation_id, json.dumps(history, indent=2, ensure_ascii=False))

                # Check only the last entry after truncation
                if history and history[-1].get("dialogue_count") == dialogue_count:
                    logger.info(f"Retry detected for dialogue_count {dialogue_count}. Using historical data.")
                    is_retry = True
                    processed_urls = history[-1].get("image_urls", [])
                    instruction_to_use = history[-1].get("instruction", tool_parameters["instruction"])
            else:
                logger.info("No existing history found [conversation_id=%s]", conversation_id)
                history = []
        except Exception as e:
            logger.error(f"History initialization failed: {e}")
            history = []

        # Only process URLs and save initial history if it's NOT a retry
        if not is_retry:
            logger.info(f"Processing new request or request without history for dialogue_count {dialogue_count}.")
            # 1. Process URLs
            instruction_text = tool_parameters["instruction"]  # Use the current instruction
            instruction_urls = re.findall(r"(https?://\S+)", instruction_text)
            uploaded_image_urls = [image.url for image in images] if images else []
            all_image_urls = list(dict.fromkeys(uploaded_image_urls + instruction_urls))

            logger.info("Processing image URLs [total=%d, uploaded=%d, from_instruction=%d]", len(all_image_urls), len(uploaded_image_urls), len(instruction_urls))
            logger.info("Full URL list: %s", [str(URL(url).host) for url in all_image_urls])

            current_processed_urls = []  # Use a temporary list for new processing
            for url in all_image_urls:
                if self._is_image_url(url):
                    current_processed_urls.append(url)
                else:
                    try:
                        new_url = self.save_tos(credentials=credentials, original_url=url)
                        current_processed_urls.append(new_url)
                    except Exception as e:
                        logger.error(f"Failed to process URL {url}: {e}")
                        continue
            processed_urls = current_processed_urls  # Assign to the main variable
            instruction_to_use = instruction_text  # Ensure instruction_to_use reflects the current input

            # Handle case where no new URLs were processed (user modifying previous request)
            if dialogue_count == 0:  # 新增判断条件
                instruction_to_use = instruction_text  # 直接使用原始指令
            else:
                try:
                    analysis_prompt = f"""
任务描述: 
根据用户的聊天历史记录, 分析用户当前的绘画任务是创建新图还是修改之前的结果。
如果是创建新图, 则输出用户的原始请求内容和空的图片URL数组;
如果是修改, 则输出最多小幅调整的请求内容和相关图片URL数组, 
注意如果用户同时提供了新的图片, 则输出的图片URL数组首先要包含历史记录中的最符合用户要求的图片URL, 再附加上用户新提供的图片URL;

分析步骤: 

1. 目标图片URL的确定: 
- 明确指定的图片: 优先使用用户明确指定的图片URL。
- 默认使用最近生成的图片: 如果用户没有明确指定, 则使用历史记录中的最后一个图片URL作为默认值。
- 无可用图片: 如果完全没有可用的图片URL, 则返回空数组。

2. 优化后的提示词: 
- 保留原始意图: 确保优化后的提示词清晰明确, 并保留用户原始的绘画意图。
- 小幅调整: 如果是修改任务, 可以根据用户的修改请求进行小幅调整, 让用户的修改意图更为明确。

输入内容: 
- 当前请求内容:  {instruction_text}
- 当前用户提供的图片URL: {processed_urls}
- 相关历史记录:  {json.dumps([entry for entry in history if entry.get("dialogue_count", 0) < dialogue_count], ensure_ascii=False)}

输出格式: 
```json
{{
"target_image_urls": ["图片URL"], 
"revised_instruction": "优化后的提示词"
}}
```"""
                    logger.info(f"LLM analysis_prompt: {analysis_prompt}")

                    # 3. Call LLM for analysis
                    analysis_response = requests.post(
                        openai_url,
                        headers={"Authorization": f"Bearer {openai_api_key}"},
                        json={"model": "deepseek-v3", "messages": [{"role": "user", "content": analysis_prompt}], "temperature": 0.2},
                    ).json()

                    # 4. Parse and apply results
                    response_content = analysis_response["choices"][0]["message"]["content"]
                    logger.info(f"原始分析响应内容:\n{response_content}")

                    try:
                        # 尝试提取被```json包裹的JSON内容
                        json_match = re.search(r"```json\s*({.*?})\s*```", response_content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            analysis = json.loads(json_str)
                        else:
                            # 如果没有```json标记, 尝试直接解析整个内容
                            analysis = json.loads(response_content)

                        logger.info(f"解析后的分析结果: {json.dumps(analysis, ensure_ascii=False)}")

                        # 验证必要字段
                        required_keys = ["target_image_urls", "revised_instruction"]
                        if not all(key in analysis for key in required_keys):
                            missing = [key for key in required_keys if key not in analysis]
                            logger.error(f"分析结果缺少必要字段: {missing}")
                            yield self.create_text_message("分析失败: 返回结果字段缺失")
                            return

                        # 确保target_image_urls是列表且不为空
                        if not isinstance(analysis.get("target_image_urls"), list):
                            analysis["target_image_urls"] = []

                        # 直接使用LLM提取的URLs
                        processed_urls = analysis["target_image_urls"]
                        instruction_to_use = analysis["revised_instruction"]
                        logger.info(f"Updated processed_urls from analysis: {processed_urls}")
                        logger.info(f"Updated instruction: {instruction_to_use}")

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析失败: {e}\n原始内容: {response_content}")
                        yield self.create_text_message("分析失败: 服务返回格式异常")
                        return

                except Exception as e:
                    logger.error(f"History analysis failed: {e}")
                    yield self.create_text_message("无法定位历史图片, 请明确指定需要修改的图片")
                    return

            # 2. Store initial conversation history (only if not a retry)
            logger.info(f"Preparing to save history - current processed_urls: {processed_urls}")
            history_entry = {
                "dialogue_count": dialogue_count,
                "instruction": instruction_to_use,  # 总是使用更新后的instruction
                "image_urls": processed_urls,
            }
            logger.info(f"History entry to save: {history_entry}")
            try:
                existing_data = self.session.storage.get(storage_key)
                logger.info("Loading conversation history [conversation_id=%s, exists=%s]", conversation_id, existing_data is not None)
                try:
                    history = json.loads(existing_data.decode()) if existing_data else []
                except json.JSONDecodeError:
                    logger.error("Invalid JSON data in storage, initializing new history")
                    history = []

                logger.info("History entries loaded [conversation_id=%s, count=%d]", conversation_id, len(history))

                # 只需要判断最后一条history
                if history and history[-1]["dialogue_count"] == dialogue_count:
                    logger.info(f"Updating existing entry for dialogue_count {dialogue_count}")
                    history[-1] = history_entry
                else:
                    logger.info(f"Appending new entry for dialogue_count {dialogue_count}")
                    history.append(history_entry)

                self.session.storage.set(storage_key, json.dumps(history).encode())
                logger.info("Updated conversation history [conversation_id=%s]\n%s", conversation_id, json.dumps(history, indent=2, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Failed to save initial conversation history: {e}")

        # Prepare messages for API call using instruction_to_use and processed_urls
        logger.info("Final values before API call:")
        logger.info(f"instruction_to_use: {instruction_to_use}")
        logger.info(f"processed_urls: {processed_urls}")
        logger.info(f"dialogue_count: {dialogue_count}")
        # This part runs for both new requests and retries
        model = tool_parameters.get("model", "gpt-4o-all")
        stream = bool(tool_parameters.get("stream", False) in ["True", "true", "TRUE", True])
        image_format = tool_parameters.get("image_format", "text")
        messages = [{"role": "user", "content": ""}]  # Placeholder

        cleaned_instruction = re.sub(r"https?://\S+", "", instruction_to_use).strip()

        if image_format == "vision":
            content_list = [{"type": "text", "text": cleaned_instruction}]
            for url in processed_urls:
                content_list.append({"type": "image_url", "image_url": {"url": url}})
            messages[0]["content"] = content_list
        else:  # text format
            url_prefix = " ".join(processed_urls)
            messages[0]["content"] = f"{url_prefix} {cleaned_instruction}".strip() if url_prefix else cleaned_instruction

        # --- API Call Section ---
        try:
            # 发送API请求
            openai_payload = {"model": model, "messages": messages, "stream": stream}
            headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
            logger.info({"url": openai_url, "headers": headers, "payload": openai_payload, "processed_urls": processed_urls})
            response = requests.post(openai_url, headers=headers, json=openai_payload, timeout=60, stream=stream)
            response.raise_for_status()

            content = ""
            if stream:
                logger.info("Starting to process streaming response")
                # 如果是流式响应, 需要拼接内容
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode("utf-8")
                        logger.info("Received streaming chunk: %s", line_text[:100])  # 记录前100字符
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
                logger.info("Processing non-streaming response")
                content = response.json()["choices"][0]["message"]["content"]
                yield self.create_text_message(content)

            logger.info("Full response content: %s", content)

            # Reconstruct the entry to include the response
            history_entry_with_response = {
                "dialogue_count": dialogue_count,
                "instruction": instruction_to_use,
                "image_urls": processed_urls.copy(),  # 使用副本避免后续修改
                "response_content": content,
            }
            logger.info(f"Final history entry with response: {history_entry_with_response}")
            try:
                # 只需要判断最后一条history
                if history and history[-1]["dialogue_count"] == dialogue_count:
                    logger.info(f"Updating existing entry for dialogue_count {dialogue_count} with response")
                    history[-1] = history_entry_with_response
                else:
                    logger.info(f"Appending new entry for dialogue_count {dialogue_count} with response")
                    history.append(history_entry_with_response)

                self.session.storage.set(storage_key, json.dumps(history).encode())
                logger.info("Updated conversation history with response [conversation_id=%s]\n%s", conversation_id, json.dumps(history, indent=2, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Failed to update conversation history with response: {e}")

            image_urls = re.findall(r"!\[.*?\]\((https?://[^\s)]+)", content)
            if image_urls:
                logger.info("Detected %d image URLs in response", len(image_urls))
                last_url = image_urls[-1]
                logger.info("Processing last image URL: %s", str(URL(last_url).host))
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
                        yield self.create_text_message("\n\n图片处理失败, 请尝试重新生成")
                else:
                    yield self.create_image_message(last_url)
                return

            yield self.create_text_message("\n\n当前流量限制, 请稍后再试")

        except Exception as e:
            raise BaseException(f"API Error: {str(e)}")
