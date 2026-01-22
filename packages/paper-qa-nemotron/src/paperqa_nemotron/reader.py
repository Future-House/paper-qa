"""Reader for PaperQA using Nvidia's nemotron-parse VLM."""

import asyncio
import io
import json
import logging
import os
from collections.abc import Awaitable, Mapping
from contextlib import closing
from typing import Any, Literal, cast

import litellm
import numpy as np
import pypdfium2 as pdfium
from lmi.utils import gather_with_concurrency
from paperqa.readers import PDFParserFn
from paperqa.settings import ParsingSettings
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError
from PIL import Image
from tenacity import RetryError

from paperqa_nemotron.api import (
    CLASSIFICATIONS_WITH_MEDIA,
    NemotronBBoxError,
    NemotronLengthError,
    NemotronParseAnnotatedBBox,
    NemotronParseClassification,
    NemotronParseMarkdownBBox,
    _call_nvidia_api,
    _call_sagemaker_api,
)

logger = logging.getLogger(__name__)

WHITE_RGB = (255, 255, 255)
# On DOI 10.1016/j.neuron.2011.12.023, 36-px was an insufficient border,
# then on DOI 10.1111/jnc.13398, 42-px was an insufficient border,
# then on DOI 10.1016/j.neuron.2011.12.023 (again), 56-px was an insufficient border,
# all with temperature of 0 and DPI 300
DEFAULT_BORDER_SIZE = 60  # pixels

# Nemotron-parse's native input dimensions, see:
# https://docs.nvidia.com/nim/vision-language-models/latest/examples/nemotron-parse/overview.html
# https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1/blob/main/preprocessor_config.json
NEMOTRON_PARSE_TARGET_WIDTH = 1648
NEMOTRON_PARSE_TARGET_HEIGHT = 2048


def pad_image_with_border(
    image: "Image.Image",
    border: int | tuple[int, int] = DEFAULT_BORDER_SIZE,
    pad_color: float | tuple[float, ...] | str = WHITE_RGB,
) -> "tuple[Image.Image, int, int]":
    """Pad image with colored borders.

    Padding the border can improve nemotron-parse performance
    because now there's room to bound PDF artwork that extends to the edge of the PDF.
    Without a border margin, the bounding box can extend beyond [0, 1].

    Args:
        image: Image to pad.
        border: Border size (pixels) to add on all sides.
            If a two-tuple it's the x border and y border,
            otherwise both x and y borders are symmetric.
        pad_color: Color to use for padding, default is white.

    Returns:
        Three-tuple of padded image and x + y image offset (pixels), where offsets
            indicate where the original image starts in the padded image.
    """
    border_x, border_y = border if isinstance(border, tuple) else (border, border)

    # Create canvas with border on all sides, while not manipulating the original image
    orig_w, orig_h = image.size
    canvas = Image.new(
        image.mode, (orig_w + 2 * border_x, orig_h + 2 * border_y), pad_color  # type: ignore[arg-type]
    )
    # Paste original image onto canvas with border offset
    canvas.paste(image, (border_x, border_y))
    return canvas, border_x, border_y


def fit_image_to_target_aspect_ratio(
    image: "Image.Image",
    target_w: int = NEMOTRON_PARSE_TARGET_WIDTH,
    target_h: int = NEMOTRON_PARSE_TARGET_HEIGHT,
    pad_color: float | tuple[float, ...] | str = WHITE_RGB,
    resample: "Image.Resampling" = Image.Resampling.LANCZOS,
) -> "tuple[Image.Image, float, int, int]":
    """Scale up and center the image onto the smallest possible canvas with nemotron-parse's aspect ratio.

    Began from preprocess_image_like_eclair:
    https://github.com/xinyu-dev/nemotron-parse-prod-hf/blob/9535a08f560e90c11f700d08c2870c40defa8aab/Step_2_Extract.ipynb

    Args:
        image: Input image.
        target_w: Minimum output width, and width component of output aspect ratio.
        target_h: Minimum output height, and height component of output aspect ratio.
        pad_color: Color to use for canvas padding.
        resample: Resampling filter to use when scaling up.

    Returns:
        Four-tuple of:
        - Image on canvas with target aspect ratio
        - Scale factor applied (1.0 if no scaling)
        - X offset (px) where scaled image starts on canvas.
        - Y offset (px) where scaled image starts on canvas.
    """
    original_width, original_height = image.size
    target_aspect_ratio = target_w / target_h

    # Scale up small images to at least target dimensions
    scale = max(target_w / original_width, target_h / original_height, 1.0)
    scaled_image = (
        image.resize(
            (int(original_width * scale), int(original_height * scale)), resample
        )
        if scale > 1.0
        else image
    )
    scaled_width, scaled_height = scaled_image.size

    # Calculate smallest canvas with target aspect ratio that fits the scaled image
    if scaled_width / scaled_height > target_aspect_ratio:
        # Image is wider than target ratio: match width, extend height
        canvas_width = scaled_width
        canvas_height = int(scaled_width / target_aspect_ratio)
    else:
        # Image is taller than target ratio: match height, extend width
        canvas_height = scaled_height
        canvas_width = int(scaled_height * target_aspect_ratio)

    # Create canvas and center the scaled image
    canvas = Image.new(
        scaled_image.mode, (canvas_width, canvas_height), pad_color  # type: ignore[arg-type]
    )
    center_offset_x = (canvas_width - scaled_width) // 2
    center_offset_y = (canvas_height - scaled_height) // 2
    canvas.paste(scaled_image, (center_offset_x, center_offset_y))
    return canvas, scale, center_offset_x, center_offset_y


async def parse_pdf_to_pages(
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    page_range: int | tuple[int, int] | None = None,
    parse_media: bool = True,
    full_page: bool = False,
    dpi: int | None = 300,
    api_params: Mapping[str, Any] | None = None,
    concurrency: int | asyncio.Semaphore | None = 128,
    border: int | tuple[int, int] = DEFAULT_BORDER_SIZE,
    optimize_aspect_ratio: bool = True,
    failover_parser: str | PDFParserFn | None = None,
    **kwargs: Any,
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
        concurrency: Optional concurrency semaphore on concurrent processing of pages,
            use to put a ceiling on memory usage. Default is 128 to prioritize reader
            speed over memory, but not get obliterated by huge 1000-page PDFs.
            Set as None to disable concurrency limits, processing all pages at once.
        border: Border size (pixels) to add on all sides.
            If a two-tuple it's the x border and y border,
            otherwise both x and y borders are symmetric.
        optimize_aspect_ratio: Flag (default is enabled) to preprocess images to the
            aspect ratio used when training nemotron-parse before sending to the API.
        failover_parser: Optional PDF parser to use when nemotron-parse fails on a
            given page. Can be a callable or an importable fully qualified name.
            Any metadata from the failover reader is not used (as of now).
        **kwargs: Keyword arguments passed to the failover parser, if specified.
            Otherwise they are thrown away.

    Returns:
        ParsedText with parsed content and metadata.
    """
    if failover_parser is not None:
        failover_parser = ParsingSettings._resolve_parse_pdf(failover_parser)

    try:
        pdf_doc = pdfium.PdfDocument(path)
    except pdfium.PdfiumError as exc:
        raise ImpossibleParsingError(
            f"PDF reading via {pdfium.__name__} failed on the PDF at path {path!r},"
            " likely this PDF file is corrupt."
        ) from exc
    api_params = {"model_name": "nvidia/nemotron-parse"} | dict(api_params or {})
    if api_params["model_name"].startswith("sagemaker/"):
        api_params["model_name"] = api_params["model_name"].removeprefix("sagemaker/")
        call_fn = _call_sagemaker_api
    else:
        call_fn = _call_nvidia_api  # type: ignore[assignment]

    with closing(pdf_doc):
        page_count = len(pdf_doc)
        # Determine page range (convert from 1-indexed to 0-indexed)
        if page_range is None:
            start_page, end_page = 0, page_count
        elif isinstance(page_range, int):
            start_page, end_page = page_range - 1, page_range
        else:
            start_page, end_page = page_range[0] - 1, page_range[1]

        async def call_failover(
            page_num: int, cause_exc: BaseException
        ) -> tuple[str, str | tuple[str, list[ParsedMedia]]]:
            logger.warning(
                f"Falling back to failover parser {failover_parser} for page {page_num}"
                f" of {path!r} due to {type(cause_exc).__name__}."
            )
            fallback_parsed_text = cast(PDFParserFn, failover_parser)(
                path,
                page_size_limit=page_size_limit,
                page_range=page_num,
                parse_media=parse_media,
                full_page=full_page,
                dpi=dpi,
                api_params=api_params,
                concurrency=concurrency,
                border=border,
                **kwargs,
            )
            if isinstance(fallback_parsed_text, Awaitable):
                fallback_parsed_text = await fallback_parsed_text
            if not isinstance(fallback_parsed_text.content, dict):
                raise NotImplementedError(
                    f"Didn't yet handle the fallback parser {failover_parser}"
                    " not giving dictionary content, got"
                    f" {type(fallback_parsed_text.content).__name__}."
                ) from cause_exc
            return str(page_num), fallback_parsed_text.content[str(page_num)]

        async def process_page(  # noqa: PLR0912
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
            rendered_page_pil = rendered_page.to_pil()

            # Initialize transformation tracking variables
            aspect_scale = 1.0
            aspect_offset_x = aspect_offset_y = 0
            border_offset_x = border_offset_y = 0
            if parse_media and not full_page:  # If we need bounding boxes
                if optimize_aspect_ratio:
                    aspect_image, aspect_scale, aspect_offset_x, aspect_offset_y = (
                        fit_image_to_target_aspect_ratio(rendered_page_pil)
                    )
                else:
                    aspect_image = rendered_page_pil
                # Apply white border padding to increase bounding box reliability
                rendered_page_padded_pil, border_offset_x, border_offset_y = (
                    pad_image_with_border(aspect_image, border)
                )
                image_for_api = np.array(rendered_page_padded_pil)
                tool_name: Literal["markdown_bbox", "markdown_no_bbox"] = (
                    "markdown_bbox"
                )
            else:
                if optimize_aspect_ratio:
                    aspect_image, aspect_scale, aspect_offset_x, aspect_offset_y = (
                        fit_image_to_target_aspect_ratio(rendered_page_pil)
                    )
                    image_for_api = np.array(aspect_image)
                else:
                    image_for_api = rendered_page.to_numpy()
                tool_name = "markdown_no_bbox"
            del rendered_page  # Free pdfium bitmap memory

            try:
                try:
                    response = await call_fn(
                        image=image_for_api, tool_name=tool_name, **api_params
                    )
                except NemotronLengthError:
                    if tool_name != "markdown_bbox":
                        raise
                    # Fallback to detection_only + markdown_no_bbox to reinvent
                    # markdown_bbox, bypassing its length error
                    detection_results = await call_fn(
                        image=image_for_api, tool_name="detection_only", **api_params
                    )

                    async def extract_text(
                        detection: NemotronParseAnnotatedBBox,
                    ) -> NemotronParseMarkdownBBox:
                        # Convert bbox from normalized [0, 1] to padded image pixel
                        # coordinates, then convert to original image coordinates by
                        # removing offsets and scaling
                        pad_xmin, pad_ymin, pad_xmax, pad_ymax = (
                            detection.bbox.to_page_coordinates(
                                rendered_page_padded_pil.height,
                                rendered_page_padded_pil.width,
                            )
                        )
                        original_bbox = (
                            max(
                                0,
                                (pad_xmin - border_offset_x - aspect_offset_x)
                                / aspect_scale,
                            ),
                            max(
                                0,
                                (pad_ymin - border_offset_y - aspect_offset_y)
                                / aspect_scale,
                            ),
                            min(
                                rendered_page_pil.width,
                                (pad_xmax - border_offset_x - aspect_offset_x)
                                / aspect_scale,
                            ),
                            min(
                                rendered_page_pil.height,
                                (pad_ymax - border_offset_y - aspect_offset_y)
                                / aspect_scale,
                            ),
                        )
                        # Crop original image at bbox (without border or aspect ratio padding)
                        region_pil = rendered_page_pil.crop(original_bbox)
                        # Use markdown_no_bbox to get text for this region
                        # abandoning the text if we're still hitting a length error
                        try:
                            text: str | None = "\n\n".join(
                                item.text
                                for item in await call_fn(
                                    image=np.array(region_pil),
                                    tool_name="markdown_no_bbox",
                                    **api_params,
                                )
                            )
                        except NemotronLengthError:
                            logger.warning(
                                "Suppressed NemotronLengthError during markdown_no_bbox"
                                f" fallback for {detection.type} bbox on page {i + 1} of {path!r}.",
                            )
                            text = None
                        return NemotronParseMarkdownBBox(
                            bbox=detection.bbox, type=detection.type, text=text
                        )

                    response = await asyncio.gather(
                        *(extract_text(d) for d in detection_results)
                    )
            except NemotronLengthError as length_err:
                # This length failure is from the detection_only tool
                if failover_parser is not None:
                    return await call_failover(page_num=i + 1, cause_exc=length_err)
                raise RuntimeError(
                    f"Failed to attain a valid response for page {i}"
                    f" of PDF at path {path!r}"
                    f" due to NemotronLengthError."
                    " Perhaps try tweaking parameters such as"
                    f" increasing DPI {dpi} or increasing API parameter's"
                    f" temperature {api_params.get('temperature')}."
                ) from length_err
            except RetryError as model_err:
                inner_exc = model_err.last_attempt._exception
                if (
                    isinstance(
                        inner_exc, (NemotronBBoxError, TimeoutError, litellm.Timeout)
                    )
                    and failover_parser is not None
                ):
                    return await call_failover(page_num=i + 1, cause_exc=inner_exc)
                if isinstance(inner_exc, NemotronBBoxError):
                    # Nice-ify nemotron-parse failures to speed debugging
                    raise RuntimeError(  # noqa: TRY004
                        f"Failed to attain a valid response for page {i}"
                        f" of PDF at path {path!r}"
                        f" due to {type(inner_exc).__name__}."
                        " Perhaps try tweaking parameters such as"
                        f" increasing DPI {dpi} or increasing API parameter's"
                        f" temperature {api_params.get('temperature')}."
                    ) from model_err
                raise
            del image_for_api  # Free up memory as API call is done

            # Per https://docs.nvidia.com/nim/vision-language-models/1.5.0/examples/nemotron-parse/overview.html#nemotron-parse-overview
            # > It outputs text in reading order.
            # So according to that, we can just strictly join here.
            # In practice, this hasn't been strictly true at temperature T=1,
            # sometimes the model will get the ordering wrong. Unfortunately,
            # corrections such as sorting by bounding box are hard because of edge cases
            # such as two-column PDFs (where 'vertical then horizontal' ordering
            # is not a valid sorting heuristic)
            text = "\n\n".join(item.text or "" for item in response)  # noqa: FURB143
            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The text in page {i} of {page_count} was {len(text)} chars"
                    f" long, which exceeds the {page_size_limit} char limit for the PDF"
                    f" at path {path}."
                )

            media: list[ParsedMedia] = []
            if parse_media and full_page:
                img_bytes = io.BytesIO()
                rendered_page_pil.save(img_bytes, format="PNG")

                media_metadata = render_kwargs | {
                    "type": "screenshot",
                    "width": rendered_page_pil.width,
                    "height": rendered_page_pil.height,
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
                    item
                    for item in cast(list[NemotronParseMarkdownBBox], response)
                    if item.type in CLASSIFICATIONS_WITH_MEDIA
                ):
                    # Convert bbox from normalized [0, 1] to padded image pixel
                    # coordinates, then convert to original image coordinates by
                    # removing offsets and scaling
                    pad_xmin, pad_ymin, pad_xmax, pad_ymax = (
                        item.bbox.to_page_coordinates(
                            rendered_page_padded_pil.height,
                            rendered_page_padded_pil.width,
                        )
                    )
                    original_bbox = (
                        max(
                            0,
                            (pad_xmin - border_offset_x - aspect_offset_x)
                            / aspect_scale,
                        ),
                        max(
                            0,
                            (pad_ymin - border_offset_y - aspect_offset_y)
                            / aspect_scale,
                        ),
                        min(
                            rendered_page_pil.width,
                            (pad_xmax - border_offset_x - aspect_offset_x)
                            / aspect_scale,
                        ),
                        min(
                            rendered_page_pil.height,
                            (pad_ymax - border_offset_y - aspect_offset_y)
                            / aspect_scale,
                        ),
                    )
                    region_pix = rendered_page_pil.crop(original_bbox)
                    img_bytes = io.BytesIO()
                    region_pix.save(img_bytes, format="PNG")
                    media_metadata = render_kwargs | {
                        "bbox": original_bbox,
                        "type": item.type.name.lower(),
                        "width": region_pix.width,
                        "height": region_pix.height,
                    }
                    del region_pix  # Free cropped image memory
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

        gather = (
            asyncio.gather(*(process_page(i) for i in range(start_page, end_page)))
            if concurrency is None
            else gather_with_concurrency(
                concurrency, (process_page(i) for i in range(start_page, end_page))
            )
        )
        for page_num, page_content in await gather:
            content[page_num] = page_content
            if parse_media:
                page_text, page_media = page_content  # type: ignore[misc]
                total_length += len(page_text)
                count_media += len(page_media)
            else:
                total_length += len(page_content)

    # No need to reflect border or api_params such as api_base or temperature here
    multimodal_string = (
        f"|multimodal|dpi={dpi}|mode={'full-page' if full_page else 'individual'}"
    )
    metadata = ParsedMetadata(
        parsing_libraries=[
            f"{pdfium.__name__} ({pdfium.version.PYPDFIUM_INFO})",
            api_params["model_name"],
        ],
        total_parsed_text_length=total_length,
        count_parsed_media=count_media,
        name=(
            f"pdf|page_range={str(page_range).replace(' ', '')}"
            f"{multimodal_string if parse_media else ''}"
        ),
    )
    return ParsedText(content=content, metadata=metadata)
