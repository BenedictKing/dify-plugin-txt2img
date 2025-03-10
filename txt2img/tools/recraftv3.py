from collections.abc import Generator
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin import Tool
from .base_image_tool import ImageGenerator


class RecraftV3Tool(Tool):
    SUPPORTED_SIZES = [
        "1024x512",
        "1024x1024",
        "1024x576",
        "1024x768",
        "512x1024",
        "512x768",
        "1280x960",
        "960x1280",
        "768x1366",
        "768x512",
        "1366x768",
        "1344x576",
    ]

    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        prompt = tool_parameters.get("prompt", "")
        if not prompt:
            yield self.create_text_message("Please input prompt")
            return

        generator = ImageGenerator(self.runtime.credentials)
        response = generator.generate_image(
            prompt=prompt,
            model=tool_parameters.get("model", "recraftv3"),
            size=tool_parameters.get("size", "1024x1024"),
            supported_sizes=self.SUPPORTED_SIZES,
        )

        for mime_type, blob in generator.process_response(response):
            yield self.create_blob_message(blob=blob, meta={"mime_type": mime_type})
