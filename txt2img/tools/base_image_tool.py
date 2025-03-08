import base64
from collections.abc import Generator
from openai import OpenAI
from yarl import URL

class ImageGenerator:
    """Helper class that encapsulates image generation logic"""
    
    def __init__(self, credentials: dict):
        self.credentials = credentials

    def get_client(self):
        """Create OpenAI client with credentials"""
        openai_base_url = self.credentials.get("openai_base_url")
        base_url = self._get_base_url(openai_base_url)
        
        return OpenAI(
            api_key=self.credentials["openai_api_key"],
            base_url=base_url,
        )
        
    def _get_base_url(self, base_url: str) -> str:
        """处理 base_url，如果已经以 /v1 结尾则不再添加"""
        if not base_url:
            return None
        
        url = URL(base_url)
        if url.path.endswith('/v1'):
            return str(url)
        return str(url / "v1")

    def generate_image(self, prompt: str, model: str, size: str, n: int = 1):
        """Generate image using OpenAI API"""
        client = self.get_client()
        return client.images.generate(
            prompt=prompt,
            model=model,
            size=size,
            n=n,
            response_format="b64_json",
        )

    @staticmethod
    def process_response(response) -> Generator[tuple[str, bytes], None, None]:
        """Process API response and return mime_type and blob data"""
        # 处理字典格式的响应
        if isinstance(response, dict) and "data" in response:
            data = response["data"]
        else:
            # 保持原有逻辑，处理对象格式的响应
            data = response.data
            
        for image in data:
            # 检查是否有 b64_json 字段
            b64_json = getattr(image, "b64_json", None) if not isinstance(image, dict) else image.get("b64_json")
            if not b64_json:
                continue
            mime_type, blob = ImageGenerator.decode_image(b64_json)
            yield mime_type, blob

    @staticmethod
    def decode_image(base64_image: str) -> tuple[str, bytes]:
        """Decode base64 image with MIME type detection"""
        if not base64_image.startswith("data:image"):
            return ("image/png", base64.b64decode(base64_image))
        
        mime_type = base64_image.split(";")[0].split(":")[1]
        image_data = base64.b64decode(base64_image.split(",")[1])
        return (mime_type, image_data)
