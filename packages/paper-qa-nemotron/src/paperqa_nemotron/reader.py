"""Reader for PaperQA using Nvidia's nemotron-parse VLM."""

import asyncio
import io
import json
import logging
import os
from contextlib import closing
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, assert_never, cast, overload

import litellm
import pypdfium2 as pdfium
from aviary.core import Message, ToolCall
from lmi.rate_limiter import GLOBAL_LIMITER
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError
from pydantic import BaseModel, Field, TypeAdapter

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

NEMOTRON_PARSE_RATE_LIMIT = "40 per 1 minute"  # Default rate for nemotron-parse API


NemotronParseToolName: TypeAlias = Literal[
    "markdown_bbox", "markdown_no_bbox", "detection_only"
]


class NvidiaBBox(BaseModel):
    """
    Bounding box, values target the range [0, 1], origin is upper left corner.

    In practice, Nemotron Parse commonly gives values outside of [0, 1].
    """

    xmin: float = Field(
        description="Lower bound, when looking right across (horizontally) the page.",
        examples=[0.33, -0.67],
    )
    xmax: float = Field(
        description="Upper bound, when looking right across (horizontally) the page.",
        examples=[0.65, 1.07],
    )
    ymin: float = Field(
        description="Lower bound, when looking down (vertically) the page.",
        examples=[0.26],
    )
    ymax: float = Field(
        description="Upper bound, when looking down (vertically) the page.",
        examples=[0.34, 1.20],
    )

    def to_page_coordinates(
        self, height: float, width: float
    ) -> tuple[float, float, float, float]:
        return (
            self.xmin * width,
            self.ymin * height,
            self.xmax * width,
            self.ymax * height,
        )


@unique
class NvidiaClassification(StrEnum):
    """
    Classes that can come from nemotron-parse, values match the API.

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


# These classifications will lead to parsing a media in addition to text
CLASSIFICATIONS_WITH_MEDIA = {
    NvidiaClassification.FORMULA,
    NvidiaClassification.PICTURE,
    NvidiaClassification.TABLE,
}


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


VectorNvidiaAnnotatedBBox = TypeAdapter(list[NvidiaAnnotatedBBox])
MatrixNvidiaAnnotatedBBox = TypeAdapter(list[list[NvidiaAnnotatedBBox]])
VectorNvidiaMarkdown = TypeAdapter(list[NvidiaMarkdown])
MatrixNvidiaMarkdown = TypeAdapter(list[list[NvidiaMarkdown]])
VectorNvidiaMarkdownBBox = TypeAdapter(list[NvidiaMarkdownBBox])
MatrixNvidiaMarkdownBBox = TypeAdapter(list[list[NvidiaMarkdownBBox]])


@overload
async def _call_nemotron_parse_api(
    image: "np.ndarray",
    tool_name: Literal["markdown_bbox"],
    api_key: str | None = None,
    **completion_kwargs,
) -> list[NvidiaMarkdownBBox]: ...


@overload
async def _call_nemotron_parse_api(
    image: "np.ndarray",
    tool_name: Literal["markdown_no_bbox"],
    api_key: str | None = None,
    **completion_kwargs,
) -> list[NvidiaMarkdown]: ...


@overload
async def _call_nemotron_parse_api(
    image: "np.ndarray",
    tool_name: Literal["detection_only"],
    api_key: str | None = None,
    **completion_kwargs,
) -> list[NvidiaAnnotatedBBox]: ...


async def _call_nemotron_parse_api(
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
    await GLOBAL_LIMITER.try_acquire(
        ("client|request", "nvidia/nemotron-parse"),
        rate_limit=NEMOTRON_PARSE_RATE_LIMIT,
    )

    if api_key is None:
        api_key = os.environ["NVIDIA_API_KEY"]
    image_data = Message.create_message(images=[image]).model_dump()["content"][0][
        "image_url"
    ]["url"]
    tool_spec = ToolCall.from_name(tool_name).model_dump(
        exclude={"id": True, "function": {"arguments"}}
    )
    response = await litellm.acompletion(
        model="nvidia/nemotron-parse",
        messages=[{"role": "user", "content": f'<img src="{image_data}" />'}],
        tools=[tool_spec],
        tool_choice=tool_spec,
        api_key=api_key,
        api_base="https://integrate.api.nvidia.com/v1",
        base_url="https://integrate.api.nvidia.com/v1",  # Duplicate so LiteLLM can infer LLM provider
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

    args_json = response.choices[0].message.tool_calls[0]["function"]["arguments"]
    if tool_name == "markdown_bbox":
        (response,) = MatrixNvidiaMarkdownBBox.validate_json(args_json)
    elif tool_name == "markdown_no_bbox":
        response = VectorNvidiaMarkdown.validate_json(args_json)
    elif tool_name == "detection_only":
        (response,) = MatrixNvidiaAnnotatedBBox.validate_json(args_json)
    else:
        assert_never(tool_name)
    return response


async def parse_pdf_to_pages(
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    page_range: int | tuple[int, int] | None = None,
    full_page: bool = False,
    dpi: int | None = None,
    **_: Any,
) -> ParsedText:
    """Parse a PDF using Nvidia's nemotron-parse VLM.

    Args:
        path: Path to the PDF file to parse.
        page_size_limit: Sensible character limit one page's text,
            used to catch bad PDF reads.
        page_range: Optional start_page or two-tuple of inclusive (start_page, end_page)
            to parse only specific pages, where pages are one-indexed.
            Leaving as the default of None will parse all pages.
        full_page: Set True to screenshot the entire page as one image,
            instead of parsing individual images or tables.
        dpi: Optional DPI (dots per inch) for image resolution.
        **_: Thrown away kwargs.

    Returns:
        ParsedText with parsed content and metadata.
    """
    try:
        pdf_doc = pdfium.PdfDocument(path)
    except pdfium.PdfiumError as exc:
        raise ImpossibleParsingError(
            f"PDF reading via {pdfium.__name__} failed on the PDF at path {path!r},"
            " likely this PDF file is corrupt."
        ) from exc

    with closing(pdf_doc):
        page_count = len(pdf_doc)
        # Determine page range (convert from 1-indexed to 0-indexed)
        if page_range is None:
            start_page, end_page = 0, page_count
        elif isinstance(page_range, int):
            start_page, end_page = page_range - 1, page_range
        else:
            start_page, end_page = page_range[0] - 1, page_range[1]

        async def process_page(
            i: int,
        ) -> tuple[str, tuple[str, list[ParsedMedia]]]:
            """Process a page and return its one-indexed page number and content."""
            try:
                page = pdf_doc[i]
            except pdfium.PdfiumError as pdfium_exc:
                if not 0 <= i < len(pdf_doc):
                    raise ValueError(
                        f"Page range {page_range}'s value {i} is outside"
                        f" the size of document {path!r}."
                    ) from pdfium_exc
                raise
            render_kwargs: dict[str, Any] = {}
            if dpi is not None:
                render_kwargs["scale"] = dpi / 72

            rendered_page = page.render(**render_kwargs)
            response = await _call_nemotron_parse_api(
                image=rendered_page.to_numpy(), tool_name="markdown_bbox"
            )
            text = "\n\n".join(item.text for item in response)
            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The text in page {i} of {page_count} was {len(text)} chars"
                    f" long, which exceeds the {page_size_limit} char limit for the PDF"
                    f" at path {path}."
                )

            media: list[ParsedMedia] = []
            if full_page:
                pil_image = rendered_page.to_pil()
                img_bytes = io.BytesIO()
                pil_image.save(img_bytes, format="PNG")

                media_metadata = render_kwargs | {
                    "type": "screenshot",
                    "width": pil_image.width,
                    "height": pil_image.height,
                }
                media_metadata["info_hashable"] = json.dumps(
                    media_metadata, sort_keys=True
                )
                # Add page number after info_hashable so differing pages
                # don't break the cache key
                media_metadata["page_num"] = i + 1
                media.append(
                    ParsedMedia(index=0, data=img_bytes.getvalue(), info=media_metadata)
                )
            else:
                counters = dict.fromkeys(CLASSIFICATIONS_WITH_MEDIA, 0)
                for item in (
                    item for item in response if item.type in CLASSIFICATIONS_WITH_MEDIA
                ):
                    bbox = item.bbox.to_page_coordinates(
                        rendered_page.height, rendered_page.width
                    )

                    pil_image = rendered_page.to_pil()
                    try:
                        region_pix = pil_image.crop(bbox)
                    except ValueError:
                        logger.warning(
                            f"Nemotron Parse {item} has invalid bounding box."
                        )
                    else:
                        img_bytes = io.BytesIO()
                        region_pix.save(img_bytes, format="PNG")
                        media_metadata = render_kwargs | {
                            "bbox": bbox,
                            "type": item.type.name.lower(),
                            "width": region_pix.width,
                            "height": region_pix.height,
                        }
                        media_metadata["info_hashable"] = json.dumps(
                            {
                                k: (
                                    v
                                    if k != "bbox"
                                    # Enables bbox deduplication based on whole pixels,
                                    # since <1-px differences are just noise
                                    else tuple(round(x) for x in cast(tuple, v))
                                )
                                for k, v in media_metadata.items()
                            },
                            sort_keys=True,
                        )
                        # Add page number after info_hashable so differing pages
                        # don't break the cache key
                        media_metadata["page_num"] = i + 1
                        media.append(
                            ParsedMedia(
                                index=counters[item.type],
                                data=img_bytes.getvalue(),
                                text=(
                                    item.text
                                    if item.type == NvidiaClassification.TABLE
                                    else None
                                ),
                                info=media_metadata,
                            )
                        )
                    counters[item.type] += 1

            return str(i + 1), (text, media)

        content: dict[str, tuple[str, list[ParsedMedia]]] = {}
        total_length = count_media = 0
        for page_num, (page_text, page_media) in await asyncio.gather(
            *(process_page(i) for i in range(start_page, end_page))
        ):
            content[page_num] = page_text, page_media
            total_length += len(page_text)
            count_media += len(page_media)

    metadata = ParsedMetadata(
        parsing_libraries=[
            f"{pdfium.__name__} ({pdfium.version.PYPDFIUM_INFO})",
            "nvidia/nemotron-parse",
        ],
        total_parsed_text_length=total_length,
        count_parsed_media=count_media,
        name=(
            f"pdf|page_range={str(page_range).replace(' ', '')}"
            f"|dpi={dpi}|mode={'full-page' if full_page else 'individual'}"
        ),
    )
    return ParsedText(content=content, metadata=metadata)
