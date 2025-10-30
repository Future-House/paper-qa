import base64
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


def encode_image_as_base64(
    image: "Image", format: str = "PNG"  # noqa: A002
) -> tuple[str, str]:
    """Encode a PIL Image to base64 string.

    Args:
        image: PIL Image to encode.
        format: Image format (PNG, JPEG, etc.).

    Returns:
        Tuple of (base64_string, mime_type).
    """
    # TODO: remove in favor of aviary.utils.encode_image_to_base64
    # after the release of https://github.com/Future-House/aviary/pull/313
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii"), f"image/{format.lower()}"
