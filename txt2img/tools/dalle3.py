from collections.abc import Generator
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin import Tool
from .base_image_tool import ImageGenerator


class DallE3Tool(Tool):
    SUPPORTED_SIZES = ["1024x1024", "1024x1792", "1792x1024"]

    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        prompt = tool_parameters.get("prompt", "")
        if not prompt:
            yield self.create_text_message("Please input prompt")
            return

        quality = tool_parameters.get("quality", "standard")
        if quality not in {"standard", "hd"}:
            quality = "standard"
        style = tool_parameters.get("style", "vivid")
        if style not in {"natural", "vivid"}:
            style = "vivid"

        generator = ImageGenerator(self.runtime.credentials)
        response = generator.generate_image(
            prompt=prompt,
            model="dall-e-3",
            size=tool_parameters.get("size", "1024x1024"),
            style=style,
            quality=quality,
            supported_sizes=self.SUPPORTED_SIZES,
        )

        for mime_type, blob in generator.process_response(response):
            yield self.create_blob_message(blob=blob, meta={"mime_type": mime_type})
