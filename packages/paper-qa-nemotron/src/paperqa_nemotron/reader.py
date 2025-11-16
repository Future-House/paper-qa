"""
Reader for PaperQA using Nvidia's nemotron-parse VLM.

For more info on `nemotron-parse`, check out:
- Technical blog: https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/
- Hugging Face weights: https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1
- Model card: https://build.nvidia.com/nvidia/nemotron-parse/modelcard
- API docs: https://docs.nvidia.com/nim/vision-language-models/1.5.0/examples/nemotron-parse/overview.html#nemotron-parse-overview
- Cookbook: https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-Parse-v1.1/build_general_usage_cookbook.ipynb
"""

import asyncio
import contextlib
import io
import json
import logging
import os
from collections.abc import Mapping
from contextlib import closing
from enum import StrEnum, unique
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeAlias,
    assert_never,
    cast,
    overload,
)

import litellm
import pypdfium2 as pdfium
from aviary.core import Message, ToolCall
from lmi.rate_limiter import GLOBAL_LIMITER, GLOBAL_RATE_LIMITER_TIMEOUT
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
)
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

NEMOTRON_PARSE_RATE_LIMIT = "40 per 1 minute"  # Default rate for nemotron-parse API


class NemotronLengthError(ValueError):
    r"""
    Error for nemotron-parse rejecting an input with 'length' finish reason.

    This can happen if nemotron-parse starts babbling '\\n' a bunch
    at lower DPI, but a re-request can skirt this error.
    """


class NemotronBBoxError(ValueError):
    """
    Error for nemotron-parse returning an invalid bounding box.

    Examples include values outside of [0, 1] or a non-positive gap between min and max.
    """


NemotronParseToolName: TypeAlias = Literal[
    "markdown_bbox", "markdown_no_bbox", "detection_only"
]


class NemotronParseBBox(BaseModel):
    """
    Bounding box, values target the range [0, 1], origin is upper left corner.

    In practice, nemotron-parse commonly gives values outside of [0, 1]
    at temperature of 1, but a re-request can get values inside [0, 1].
    """

    @staticmethod
    def validate_min_less_than_max(v: float, info: ValidationInfo) -> float:
        if info.field_name in {"xmax", "ymax"}:
            min_field_name = info.field_name.replace("max", "min")
            with contextlib.suppress(KeyError):
                if info.data[min_field_name] >= v:
                    raise ValueError(
                        f"{min_field_name} must be less than {info.field_name}."
                    )
        return v

    xmin: float = Field(
        description="Lower bound, when looking right across (horizontally) the page.",
        examples=[0.33],
        ge=0,
        le=1,
    )
    xmax: Annotated[
        float,
        Field(
            description="Upper bound, when looking right across (horizontally) the page.",
            examples=[0.65],
            ge=0,
            le=1,
        ),
        AfterValidator(validate_min_less_than_max),
    ]
    ymin: float = Field(
        description="Lower bound, when looking down (vertically) the page.",
        examples=[0.26],
        ge=0,
        le=1,
    )
    ymax: Annotated[
        float,
        Field(
            description="Upper bound, when looking down (vertically) the page.",
            examples=[0.34],
            ge=0,
            le=1,
        ),
        AfterValidator(validate_min_less_than_max),
    ]

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
class NemotronParseClassification(StrEnum):
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
    NemotronParseClassification.FORMULA,
    NemotronParseClassification.PICTURE,
    NemotronParseClassification.TABLE,
}


class NemotronParseAnnotatedBBox(BaseModel):
    """Payload for 'detection_only' tool."""

    bbox: NemotronParseBBox
    type: NemotronParseClassification = Field(
        description="Possible classifications of the bbox."
    )


class NemotronParseMarkdown(BaseModel):
    """Payload for 'markdown_no_bbox' tool."""

    text: str = Field(description="Markdown text.")


class NemotronParseMarkdownBBox(NemotronParseAnnotatedBBox, NemotronParseMarkdown):
    """Payload for 'markdown_bbox' tool."""


MatrixNemotronParseAnnotatedBBox = TypeAdapter(list[list[NemotronParseAnnotatedBBox]])
VectorNemotronParseMarkdown = TypeAdapter(list[NemotronParseMarkdown])
MatrixNemotronParseMarkdownBBox = TypeAdapter(list[list[NemotronParseMarkdownBBox]])


@overload
async def _call_nemotron_parse_api(
    image: "np.ndarray",
    tool_name: Literal["markdown_bbox"],
    api_key: str | None = None,
    api_base: str = ...,
    model_name: str = ...,
    **completion_kwargs,
) -> list[NemotronParseMarkdownBBox]: ...


@overload
async def _call_nemotron_parse_api(
    image: "np.ndarray",
    tool_name: Literal["markdown_no_bbox"],
    api_key: str | None = None,
    api_base: str = ...,
    model_name: str = ...,
    **completion_kwargs,
) -> list[NemotronParseMarkdown]: ...


@overload
async def _call_nemotron_parse_api(
    image: "np.ndarray",
    tool_name: Literal["detection_only"],
    api_key: str | None = None,
    api_base: str = ...,
    model_name: str = ...,
    **completion_kwargs,
) -> list[NemotronParseAnnotatedBBox]: ...


@retry(
    retry=retry_if_exception_type((NemotronLengthError, NemotronBBoxError)),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
@retry(
    retry=retry_if_exception_type(TimeoutError),  # Hitting rate limits
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=GLOBAL_RATE_LIMITER_TIMEOUT),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _call_nemotron_parse_api(
    image: "np.ndarray",
    tool_name: NemotronParseToolName,
    api_key: str | None = None,
    api_base: str = "https://integrate.api.nvidia.com/v1",
    model_name: str = "nvidia/nemotron-parse",
    **completion_kwargs,
) -> (
    list[NemotronParseMarkdownBBox]
    | list[NemotronParseMarkdown]
    | list[NemotronParseAnnotatedBBox]
):
    """Call the nemotron-parse API with an image.

    Args:
        image: Image to parse.
        tool_name: Name of the nemotron-parse tool.
        api_key: Optional API key for Nvidia, default uses the NVIDIA_API_KEY env var.
        api_base: API base URL to pass to the completion,
            default uses nemotron-parse API's expected base URL.
        model_name: Model name to pass to the completion,
            default uses nemotron-parse API's expected model name.
        completion_kwargs: Keyword arguments to pass to the completion.

    Returns:
        Parsed response from the API.
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
        model=model_name,
        messages=[{"role": "user", "content": f'<img src="{image_data}" />'}],
        tools=[tool_spec],
        tool_choice=tool_spec,
        api_key=api_key,
        api_base=api_base,
        base_url=api_base,  # Duplicate so LiteLLM can infer LLM provider
        **completion_kwargs,
    )
    if (
        not isinstance(response, litellm.ModelResponse)
        or len(response.choices) != 1
        or not isinstance(response.choices[0], litellm.Choices)
    ):
        raise NotImplementedError(
            f"Didn't yet handle choices shape of model response {response}."
        )
    if response.choices[0].finish_reason == "length":
        raise NemotronLengthError(
            f"Model response {response} indicates the input"
            f" image of shape {image.shape} is too large or the model started babbling."
        )
    if (
        response.choices[0].finish_reason != "tool_calls"
        or response.choices[0].message.tool_calls is None
        or len(response.choices[0].message.tool_calls) != 1
    ):
        raise NotImplementedError(
            f"Didn't yet handle choice shape of model response {response}."
        )

    args_json = response.choices[0].message.tool_calls[0]["function"]["arguments"]
    try:
        if tool_name == "markdown_bbox":
            (response,) = MatrixNemotronParseMarkdownBBox.validate_json(args_json)
        elif tool_name == "markdown_no_bbox":
            response = VectorNemotronParseMarkdown.validate_json(args_json)
        elif tool_name == "detection_only":
            (response,) = MatrixNemotronParseAnnotatedBBox.validate_json(args_json)
        else:
            assert_never(tool_name)
    except ValidationError as exc:
        raise NemotronBBoxError(
            f"nemotron-parse response {args_json} has invalid bounding box."
        ) from exc
    return response


async def parse_pdf_to_pages(
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    page_range: int | tuple[int, int] | None = None,
    parse_media: bool = True,
    full_page: bool = False,
    dpi: int | None = 300,
    api_params: Mapping[str, Any] | None = None,
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
        parse_media: Flag to also parse media (e.g. images, tables).
        full_page: Set True to screenshot the entire page as one image,
            instead of parsing individual images or tables.
        dpi: Optional DPI (dots per inch) for image resolution,
            if set as None then pypdfium2's default 1 scale will be employed.
        api_params: Optional parameters to pass to the nemotron-parse API.
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
    api_params = api_params or {}

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
        ) -> tuple[str, str | tuple[str, list[ParsedMedia]]]:
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
            try:
                response = await _call_nemotron_parse_api(
                    image=rendered_page.to_numpy(),
                    tool_name="markdown_bbox",
                    **api_params,
                )
            except RetryError as model_err:
                if isinstance(
                    model_err.last_attempt._exception,
                    NemotronLengthError | NemotronBBoxError,
                ):
                    # Nice-ify nemotron-parse failures to speed debugging
                    raise RuntimeError(  # noqa: TRY004
                        f"Failed to attain a valid response for page {i}"
                        f" of PDF at path {path!r}"
                        f" due to {type(model_err.last_attempt._exception).__name__}."
                        " Perhaps try tweaking parameters such as"
                        f" increasing DPI {dpi} or increasing API parameter's"
                        f" temperature {api_params.get('temperature')}."
                    ) from model_err
                raise
            # Per https://docs.nvidia.com/nim/vision-language-models/1.5.0/examples/nemotron-parse/overview.html#nemotron-parse-overview
            # > It outputs text in reading order.
            # So according to that, we can just strictly join here.
            # In practice, this hasn't been strictly true at temperature T=1,
            # sometimes the model will get the ordering wrong. Unfortunately,
            # corrections such as sorting by bounding box are hard because of edge cases
            # such as two-column PDFs (where 'vertical then horizontal' ordering
            # is not a valid sorting heuristic)
            text = "\n\n".join(item.text for item in response)
            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The text in page {i} of {page_count} was {len(text)} chars"
                    f" long, which exceeds the {page_size_limit} char limit for the PDF"
                    f" at path {path}."
                )

            media: list[ParsedMedia] = []
            if parse_media and full_page:
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
            elif parse_media:
                counters = dict.fromkeys(CLASSIFICATIONS_WITH_MEDIA, 0)
                for item in (
                    item for item in response if item.type in CLASSIFICATIONS_WITH_MEDIA
                ):
                    bbox = item.bbox.to_page_coordinates(
                        rendered_page.height, rendered_page.width
                    )
                    region_pix = rendered_page.to_pil().crop(bbox)
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
                                if item.type == NemotronParseClassification.TABLE
                                else None
                            ),
                            info=media_metadata,
                        )
                    )
                    counters[item.type] += 1

            return str(i + 1), text if not parse_media else (text, media)

        content: dict[str, str | tuple[str, list[ParsedMedia]]] = {}
        total_length = count_media = 0
        for page_num, page_content in await asyncio.gather(
            *(process_page(i) for i in range(start_page, end_page))
        ):
            content[page_num] = page_content
            if parse_media:
                page_text, page_media = page_content  # type: ignore[misc]
                total_length += len(page_text)
                count_media += len(page_media)
            else:
                total_length += len(page_content)

    # No need to reflect api_params such as api_base or temperature here
    multimodal_string = (
        f"|multimodal|dpi={dpi}|mode={'full-page' if full_page else 'individual'}"
    )
    metadata = ParsedMetadata(
        parsing_libraries=[
            f"{pdfium.__name__} ({pdfium.version.PYPDFIUM_INFO})",
            "nvidia/nemotron-parse",
        ],
        total_parsed_text_length=total_length,
        count_parsed_media=count_media,
        name=(
            f"pdf|page_range={str(page_range).replace(' ', '')}"
            f"{multimodal_string if parse_media else ''}"
        ),
    )
    return ParsedText(content=content, metadata=metadata)
