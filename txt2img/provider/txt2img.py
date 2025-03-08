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
        
        client = OpenAI(
            api_key=openai_api_key,
            base_url=self._get_base_url(openai_base_url)
        )
        
        try:
            # Validate credentials by listing models
            client.models.list()
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
            
    def _get_base_url(self, base_url: str) -> str:
        """处理 base_url，如果已经以 /v1 结尾则不再添加"""
        if not base_url:
            return None
        
        url = URL(base_url)
        if url.path.endswith('/v1'):
            return str(url)
        return str(url / "v1")
