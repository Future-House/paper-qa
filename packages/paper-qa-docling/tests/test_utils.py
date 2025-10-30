import base64

from PIL import Image
from pytest_subtests import SubTests

from paperqa_docling.utils import encode_image_as_base64


def test_encode_image_as_base64(subtests: SubTests) -> None:
    img = Image.new("RGB", (100, 100), color="red")

    with subtests.test(msg="png"):
        b64_str, mime = encode_image_as_base64(img, format="PNG")
        assert isinstance(b64_str, str)
        assert mime == "image/png"
        decoded = base64.b64decode(b64_str)
        assert len(decoded) > 0, "Expected valid base64"

    with subtests.test(msg="jpeg"):
        b64_str, mime = encode_image_as_base64(img, format="JPEG")
        assert mime == "image/jpeg"
        assert isinstance(b64_str, str)

    with subtests.test(msg="default"):
        b64_str, mime = encode_image_as_base64(img)
        assert mime == "image/png"
        assert isinstance(b64_str, str)
