from typing import Any
from openai import OpenAI
from yarl import URL
from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class Txt2imgProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        openai_api_key = credentials.get("openai_api_key")
        openai_base_url = credentials.get("openai_base_url")

        if not openai_api_key:
            raise ToolProviderCredentialValidationError("OpenAI API key is required")

        if not openai_base_url:
            raise ToolProviderCredentialValidationError("OpenAI base URL is required")
        elif URL(openai_base_url).path.endswith("/v1"):
            raise ToolProviderCredentialValidationError(
                "OpenAI base URL should not end with '/v1'"
            )
        else:
            openai_base_url = str(URL(openai_base_url) / "v1")

        client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

        try:
            # Validate credentials by listing models
            client.models.list()
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
