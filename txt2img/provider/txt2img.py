import logging
from typing import Any

import tos
from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from openai import OpenAI
from yarl import URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Txt2imgProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        logger.info(f"verify credentials: {credentials}")
        openai_api_key = credentials.get("openai_api_key")
        openai_base_url = credentials.get("openai_base_url")

        if not openai_api_key:
            raise ToolProviderCredentialValidationError("OpenAI API key is required")

        if not openai_base_url:
            raise ToolProviderCredentialValidationError("OpenAI base URL is required")
        elif URL(openai_base_url).path.endswith("/v1"):
            raise ToolProviderCredentialValidationError("OpenAI base URL should not end with '/v1'")
        else:
            openai_base_url = str(URL(openai_base_url) / "v1")

        client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

        try:
            # Validate credentials by listing models
            client.models.list()
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))

        tos_client = tos.TosClientV2(
            ak=credentials["VOLCENGINE_TOS_ACCESS_KEY"],
            sk=credentials["VOLCENGINE_TOS_SECRET_KEY"],
            endpoint=credentials["VOLCENGINE_TOS_ENDPOINT"],
            region=credentials["VOLCENGINE_TOS_REGION"],
        )
        try:
            tos_client.head_bucket(bucket=credentials["VOLCENGINE_TOS_BUCKET_NAME"])
        except Exception as e:
            raise ToolProviderCredentialValidationError(f"tos密钥验证失败: {str(e)}")
