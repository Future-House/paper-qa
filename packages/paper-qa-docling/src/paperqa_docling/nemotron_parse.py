"""
Docling PDF backend for Nvidia's nemotron-parse model via API.

SEE:
- https://build.nvidia.com/nvidia/nemotron-parse
- https://docs.nvidia.com/nim/vision-language-models/1.2.0/examples/retriever/overview.html
"""

import logging
import os
from collections.abc import Iterable
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Literal, TypeAlias, assert_never, cast, overload

import litellm
from aviary.core import Message, ToolCall
from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.utils.locks import pypdfium2_lock
from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from pydantic import BaseModel, Field, TypeAdapter
from pypdfium2 import PdfDocument

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image
    from pypdfium2 import PdfPage

logger = logging.getLogger(__name__)

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NemotronParseToolName: TypeAlias = Literal[
    "markdown_bbox", "markdown_no_bbox", "detection_only"
]


class NvidiaBBox(BaseModel):
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def to_docling(self) -> BoundingBox:
        return BoundingBox(
            l=self.xmin,
            r=self.xmax,
            t=self.ymin,
            b=self.ymax,
            coord_origin=CoordOrigin.TOPLEFT,
        )


@unique
class NvidiaClassification(StrEnum):
    """
    Classes that can come from nemotron-parse.

    SEE: https://docs.nvidia.com/nim/vision-language-models/1.2.0/examples/retriever/overview.html
    """

    BIBLIOGRAPHY = "Bibliography"
    CAPTION = "Caption"  # Table or picture
    FOOTNOTE = "Footnote"
    FORMULA = "Formula"
    LIST_ITEM = "List-item"  # Numbered, alphanumeric, or bullet point
    PAGE_FOOTER = "Page-footer"
    PAGE_HEADER = "Page-header"
    PICTURE = "Picture"
    SECTION_HEADER = "Section-header"
    TABLE = "Table"
    TABLE_OF_CONTENTS = "TOC"
    TEXT = "Text"  # Regular paragraph text
    TITLE = "Title"


class NvidiaAnnotatedBBox(BaseModel):
    """Payload for 'detection_only' tool."""

    bbox: NvidiaBBox
    type: NvidiaClassification = Field(
        description="Possible classifications of the bbox."
    )


class NvidiaMarkdown(BaseModel):
    """Payload for 'markdown_no_bbox' tool."""

    text: str = Field(description="Markdown text.")


class NvidiaMarkdownBBox(NvidiaAnnotatedBBox, NvidiaMarkdown):
    """Payload for 'markdown_bbox' tool."""


BulkNvidiaAnnotatedBBox = TypeAdapter(list[NvidiaAnnotatedBBox])
BulkNvidiaMarkdown = TypeAdapter(list[NvidiaMarkdown])
BulkNvidiaMarkdownBBox = TypeAdapter(list[NvidiaMarkdownBBox])


def _parse_response_json(  # noqa: PLR0911
    args_json: str, tool_name: NemotronParseToolName
) -> list[NvidiaMarkdownBBox] | list[NvidiaMarkdown] | list[NvidiaAnnotatedBBox]:
    """Parse the JSON response based on tool type.

    Args:
        args_json: JSON string from API response.
        tool_name: Name of the nemotron-parse tool.

    Returns:
        Parsed list of objects.
    """
    # Select the appropriate type adapter based on tool type
    if tool_name == "markdown_bbox":
        try:
            bulk_response = BulkNvidiaMarkdownBBox.validate_json(args_json)
            # If it's a list of lists, unwrap it
            if bulk_response and isinstance(bulk_response[0], list):  # type: ignore[unreachable]
                return cast(list[NvidiaMarkdownBBox], bulk_response[0])  # type: ignore[unreachable]
            return bulk_response  # noqa: TRY300
        except Exception:
            # Try as nested list
            nested = TypeAdapter(list[list[NvidiaMarkdownBBox]]).validate_json(
                args_json
            )
            return nested[0] if nested else []
    elif tool_name == "markdown_no_bbox":
        try:
            bulk_response = BulkNvidiaMarkdown.validate_json(args_json)  # type: ignore[assignment]
            if bulk_response and isinstance(bulk_response[0], list):  # type: ignore[unreachable]
                return cast(list[NvidiaMarkdown], bulk_response[0])  # type: ignore[unreachable]
            return bulk_response  # noqa: TRY300
        except Exception:
            nested = TypeAdapter(list[list[NvidiaMarkdown]]).validate_json(args_json)  # type: ignore[assignment]
            return nested[0] if nested else []
    elif tool_name == "detection_only":
        try:
            bulk_response = BulkNvidiaAnnotatedBBox.validate_json(args_json)  # type: ignore[assignment]
            if bulk_response and isinstance(bulk_response[0], list):  # type: ignore[unreachable]
                return cast(list[NvidiaAnnotatedBBox], bulk_response[0])  # type: ignore[unreachable]
            return bulk_response  # noqa: TRY300
        except Exception:
            nested = TypeAdapter(list[list[NvidiaAnnotatedBBox]]).validate_json(  # type: ignore[assignment]
                args_json
            )
            return nested[0] if nested else []
    else:
        assert_never(tool_name)


@overload
def _call_nemotron_parse_api(
    image: "np.ndarray",
    tool_name: Literal["markdown_bbox"],
    api_key: str | None = None,
    **completion_kwargs,
) -> list[NvidiaMarkdownBBox]: ...


@overload
def _call_nemotron_parse_api(
    image: "np.ndarray",
    tool_name: Literal["markdown_no_bbox"],
    api_key: str | None = None,
    **completion_kwargs,
) -> list[NvidiaMarkdown]: ...


@overload
def _call_nemotron_parse_api(
    image: "np.ndarray",
    tool_name: Literal["detection_only"],
    api_key: str | None = None,
    **completion_kwargs,
) -> list[NvidiaAnnotatedBBox]: ...


def _call_nemotron_parse_api(
    image: "np.ndarray",
    tool_name: NemotronParseToolName,
    api_key: str | None = None,
    **completion_kwargs,
) -> list[NvidiaMarkdownBBox] | list[NvidiaMarkdown] | list[NvidiaAnnotatedBBox]:
    """Call the nemotron-parse API with an image.

    Args:
        image: PIL Image to parse.
        tool_name: Name of the nemotron-parse tool.
        api_key: Optional API key for Nvidia, default uses the NVIDIA_API_KEY env var.
        completion_kwargs: Keyword arguments to pass to the completion.

    Returns:
        JSON response from the API.
    """
    if api_key is None:
        api_key = os.environ["NVIDIA_API_KEY"]
    image_data = Message.create_message(images=[image]).model_dump()["content"][0][
        "image_url"
    ]["url"]
    tool_spec = ToolCall.from_name(tool_name).model_dump(
        exclude={"id": True, "function": {"arguments"}}
    )
    response = litellm.completion(
        model="nvidia/nemotron-parse",
        messages=[{"role": "user", "content": f'<img src="{image_data}" />'}],
        tools=[tool_spec],
        tool_choice=tool_spec,
        api_key=api_key,
        api_base="https://integrate.api.nvidia.com/v1",
        **completion_kwargs,
    )
    if (
        not isinstance(response, litellm.ModelResponse)
        or len(response.choices) != 1
        or not isinstance(response.choices[0], litellm.Choices)
        or response.choices[0].finish_reason != "tool_calls"
        or response.choices[0].message.tool_calls is None
        or len(response.choices[0].message.tool_calls) != 1
    ):
        raise NotImplementedError(
            f"Didn't yet handle shape of model response {response}."
        )
    # Parse the response based on tool type
    args_json = response.choices[0].message.tool_calls[0]["function"]["arguments"]
    return _parse_response_json(args_json, tool_name)


class NemotronParsePageBackend(PdfPageBackend):
    """Page backend using nemotron-parse for parsing."""

    def __init__(self, *, ppage: "PdfPage", tool_name: NemotronParseToolName):
        """Initialize.

        Args:
            ppage: PyPDFium2 page object.
            tool_name: Name of the nemotron-parse tool.
        """
        self._ppage = ppage
        self._tool_name = tool_name
        self._dpage: SegmentedPdfPage | None = None
        self._unloaded = False
        self.valid = self._ppage is not None

    def _ensure_parsed(self) -> None:
        """Parse the page using nemotron-parse API if not already parsed."""
        if self._dpage is not None:
            return

        with pypdfium2_lock:
            page_image = self._ppage.render(scale=1.5, rotation=0).to_numpy()

        response = _call_nemotron_parse_api(page_image, tool_name=self._tool_name)
        # Convert API response to SegmentedPdfPage based on tool type
        if self._tool_name == "markdown_bbox":
            self._dpage = self._parse_markdown_bbox_response(response)  # type: ignore[arg-type]
        elif self._tool_name == "markdown_no_bbox":
            self._dpage = self._parse_markdown_no_bbox_response(response)  # type: ignore[arg-type]
        elif self._tool_name == "detection_only":
            self._dpage = self._parse_detection_only_response(response)  # type: ignore[arg-type]
        else:
            assert_never(self._tool_name)

    @property
    def _page_size(self) -> Size:
        """Get page size from the PDF page."""
        with pypdfium2_lock:
            return Size(width=self._ppage.get_width(), height=self._ppage.get_height())

    def _create_page_geometry(self) -> PdfPageGeometry:
        """Create standard page geometry for the current page."""
        page_size = self._page_size
        media_bbox = BoundingBox(
            l=0,
            r=page_size.width,
            t=0,
            b=page_size.height,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        return PdfPageGeometry(
            angle=0,
            rect=BoundingRectangle.from_bounding_box(media_bbox),
            boundary_type=PdfPageBoundaryType.MEDIA_BOX,
            art_bbox=media_bbox,
            bleed_bbox=media_bbox,
            crop_bbox=media_bbox,
            media_bbox=media_bbox,
            trim_bbox=media_bbox,
        )

    def _parse_markdown_bbox_response(
        self, response: list[NvidiaMarkdownBBox]
    ) -> SegmentedPdfPage:
        """Parse markdown_bbox response into SegmentedPdfPage.

        Args:
            response: List of NvidiaMarkdownBBox objects with text, bbox, and type.

        Returns:
            SegmentedPdfPage.
        """
        return SegmentedPdfPage(
            dimension=self._create_page_geometry(),
            textline_cells=[
                TextCell(
                    text=item.text,
                    rect=BoundingRectangle.from_bounding_box(item.bbox.to_docling()),
                    orig="",
                    from_ocr=False,
                )
                for item in response
            ],
            word_cells=[],
            char_cells=[],
            bitmap_resources=[],
        )

    def _parse_markdown_no_bbox_response(
        self, response: list[NvidiaMarkdown]
    ) -> SegmentedPdfPage:
        """Parse markdown_no_bbox response into SegmentedPdfPage.

        Since we don't have bboxes, distribute text evenly across the page.

        Args:
            response: List of NvidiaMarkdown objects with text only.

        Returns:
            SegmentedPdfPage.
        """
        textlines = []
        page_size = self._page_size

        # Calculate line height for even distribution
        num_items = len(response)
        if num_items == 0:
            return SegmentedPdfPage(
                dimension=self._create_page_geometry(),
                textline_cells=[],
                word_cells=[],
                char_cells=[],
                bitmap_resources=[],
            )

        line_height = page_size.height / num_items
        y_position = 0.0

        for item in response:
            if item.text.strip():  # Skip empty text
                bbox = BoundingBox(
                    l=0,
                    r=page_size.width,
                    t=y_position,
                    b=y_position + line_height,
                    coord_origin=CoordOrigin.TOPLEFT,
                )

                text_cell = TextCell(
                    text=item.text,
                    rect=BoundingRectangle.from_bounding_box(bbox),
                    orig="",
                    from_ocr=False,
                )
                textlines.append(text_cell)

            y_position += line_height

        return SegmentedPdfPage(
            dimension=self._create_page_geometry(),
            textline_cells=textlines,
            word_cells=[],
            char_cells=[],
            bitmap_resources=[],
        )

    def _parse_detection_only_response(
        self, response: list[NvidiaAnnotatedBBox]
    ) -> SegmentedPdfPage:
        """Parse detection_only response into SegmentedPdfPage.

        This mode only returns bboxes with classifications, no text.

        Args:
            response: List of NvidiaAnnotatedBBox objects with bbox and type.

        Returns:
            SegmentedPdfPage with empty text cells but bbox info preserved.
        """
        return SegmentedPdfPage(
            dimension=self._create_page_geometry(),
            textline_cells=[
                TextCell(
                    text=f"[{item.type}]",
                    rect=BoundingRectangle.from_bounding_box(item.bbox.to_docling()),
                    orig="",
                    from_ocr=False,
                )
                for item in response
            ],
            word_cells=[],
            char_cells=[],
            bitmap_resources=[],
        )

    def is_valid(self) -> bool:
        return self.valid

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        self._ensure_parsed()
        assert self._dpage is not None  # noqa: S101

        # Find intersecting cells on the page
        text_piece = ""
        page_size = self.get_size()

        scale = 1
        OVERLAP_THRESHOLD = 0.5

        for cell in self._dpage.textline_cells:
            cell_bbox = (
                cell.rect.to_bounding_box()
                .to_top_left_origin(page_height=page_size.height)
                .scaled(scale)
            )

            overlap_frac = cell_bbox.intersection_over_self(bbox)

            if overlap_frac > OVERLAP_THRESHOLD:
                if text_piece:
                    text_piece += " "
                text_piece += cell.text

        return text_piece

    def get_segmented_page(self) -> SegmentedPdfPage | None:
        self._ensure_parsed()
        return self._dpage

    def get_text_cells(self) -> Iterable[TextCell]:
        self._ensure_parsed()
        assert self._dpage is not None  # noqa: S101
        return self._dpage.textline_cells

    def get_bitmap_rects(self, scale: int = 1) -> Iterable[BoundingBox]:
        self._ensure_parsed()
        assert self._dpage is not None  # noqa: S101

        AREA_THRESHOLD = 0

        images = self._dpage.bitmap_resources

        for img in images:
            cropbox = img.rect.to_bounding_box().to_top_left_origin(
                self.get_size().height
            )

            if cropbox.area() > AREA_THRESHOLD:
                cropbox = cropbox.scaled(scale=scale)
                yield cropbox

    def get_page_image(
        self, scale: float = 1, cropbox: BoundingBox | None = None
    ) -> "Image":
        page_size = self.get_size()

        if not cropbox:
            cropbox = BoundingBox(
                l=0,
                r=page_size.width,
                t=0,
                b=page_size.height,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            padbox = BoundingBox(
                l=0, r=0, t=0, b=0, coord_origin=CoordOrigin.BOTTOMLEFT
            )
        else:
            padbox = cropbox.to_bottom_left_origin(page_size.height).model_copy()
            padbox.r = page_size.width - padbox.r
            padbox.t = page_size.height - padbox.t

        with pypdfium2_lock:
            return (
                self._ppage.render(
                    scale=scale * 1.5,
                    rotation=0,
                    crop=padbox.as_tuple(),
                )
                .to_pil()
                .resize(
                    size=(round(cropbox.width * scale), round(cropbox.height * scale))
                )
            )

    def get_size(self) -> Size:
        return self._page_size

    def unload(self) -> None:
        if not self._unloaded:
            self._unloaded = True
        self._ppage = None
        self._dpage = None


class NemotronParseDocumentBackend(PdfDocumentBackend):
    """Document backend using nemotron-parse for PDF parsing."""

    def __init__(self, *args, tool_name: NemotronParseToolName, **kwargs):
        """Initialize.

        Args:
            args: Positional arguments for the parent class.
            tool_name: Name of the nemotron-parse tool.
            kwargs: Keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)
        password = (
            self.options.password.get_secret_value() if self.options.password else None
        )
        with pypdfium2_lock:
            self._pdoc = PdfDocument(self.path_or_stream, password=password)
        if self._pdoc is None:
            raise RuntimeError(
                f"nemotron-parse backend could not load document {self.document_hash}."
            )
        self._tool_name = tool_name

    def page_count(self) -> int:
        with pypdfium2_lock:
            return len(self._pdoc)

    def load_page(self, page_no: int) -> NemotronParsePageBackend:
        """Load a page backend.

        Args:
            page_no: Zero-indexed page number.
        """
        with pypdfium2_lock:
            ppage = self._pdoc[page_no]

        return NemotronParsePageBackend(ppage=ppage, tool_name=self._tool_name)

    def is_valid(self) -> bool:
        return self.page_count() > 0

    def unload(self) -> None:
        super().unload()
        if self._pdoc is not None:
            with pypdfium2_lock:
                from contextlib import suppress

                with suppress(Exception):
                    self._pdoc.close()
            self._pdoc = None


class NemotronParseDocumentMarkdownBBoxBackend(NemotronParseDocumentBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, tool_name="markdown_bbox", **kwargs)


class NemotronParseDocumentMarkdownBackend(NemotronParseDocumentBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, tool_name="markdown_no_bbox", **kwargs)


class NemotronParseDocumentDetectionOnlyBackend(NemotronParseDocumentBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, tool_name="detection_only", **kwargs)
