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
        openai_base_url = self.credentials.get("openai_base_url", None)
        openai_base_url = str(URL(openai_base_url) / "v1")

        return OpenAI(
            api_key=self.credentials["openai_api_key"],
            base_url=openai_base_url,
        )

    def find_closest_size(self, requested_size: str, supported_sizes: list) -> str:
        """Find the closest supported size to the requested size

        Args:
            requested_size: The size requested by the user (e.g. '1024x1024')
            supported_sizes: List of sizes supported by the model (e.g. ['1024x1024', '512x512'])
                             Must be provided and not empty

        Returns:
            The closest supported size
        """
        if not supported_sizes:
            raise ValueError("supported_sizes must be provided and not empty")

        # If the requested size is in the supported sizes, return it
        if requested_size in supported_sizes:
            return requested_size

        # Parse the requested size
        try:
            req_width, req_height = map(int, requested_size.split("x"))
        except (ValueError, AttributeError):
            # If parsing fails, return the default size
            return supported_sizes[0]

        # Find the closest size by comparing aspect ratios and total pixels
        closest_size = supported_sizes[0]
        closest_diff = float("inf")

        for size in supported_sizes:
            try:
                width, height = map(int, size.split("x"))

                # Calculate aspect ratio difference
                req_ratio = req_width / req_height
                size_ratio = width / height
                ratio_diff = abs(req_ratio - size_ratio)

                # Calculate total pixels difference
                req_pixels = req_width * req_height
                size_pixels = width * height
                pixel_diff = abs(req_pixels - size_pixels)

                # Combined difference (weighted)
                diff = (ratio_diff * 100) + (pixel_diff / 10000)

                if diff < closest_diff:
                    closest_diff = diff
                    closest_size = size
            except (ValueError, ZeroDivisionError):
                continue

        return closest_size

    def generate_image(
        self, prompt: str, model: str, size: str, supported_sizes: list, n: int = 1
    ):
        """Generate image using OpenAI API"""
        client = self.get_client()

        # Find the closest supported size
        actual_size = self.find_closest_size(size, supported_sizes)

        return client.images.generate(
            prompt=prompt,
            model=model,
            size=actual_size,
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
            b64_json = (
                getattr(image, "b64_json", None)
                if not isinstance(image, dict)
                else image.get("b64_json")
            )
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
