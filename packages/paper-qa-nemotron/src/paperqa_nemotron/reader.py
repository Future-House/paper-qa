"""Reader for PaperQA using Nvidia's nemotron-parse VLM."""

import asyncio
import io
import json
import logging
import os
from collections.abc import Awaitable, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from typing import Any, Literal, cast

import litellm
import numpy as np
import pypdfium2 as pdfium
from aviary.core import encode_image_to_base64
from lmi.utils import gather_with_concurrency
from paperqa.readers import PDFParserFn, resolve_page_range
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


def _render_page(
    path: str,
    page_num: int,
    dpi: int | None = 300,
    border: int | tuple[int, int] = DEFAULT_BORDER_SIZE,
    needs_bbox: bool = True,
    page_range: int | tuple[int, int] | None = None,
) -> tuple[int, str, "Image.Image", int, int, int, int]:
    """Render a single PDF page and pre-encode the API image as a base64 data URI.

    NOTE: keep this top-level for pickling support.

    Returns:
        Seven-tuple of (page_num, image_data_uri, rendered_page_pil,
            padded_height, padded_width, offset_x, offset_y).
    """
    pdf_doc = pdfium.PdfDocument(path)
    with closing(pdf_doc):
        try:
            page = pdf_doc[page_num]
        except pdfium.PdfiumError as pdfium_exc:
            if not 0 <= page_num < len(pdf_doc):
                raise ValueError(
                    f"Page range {page_range}'s value {page_num} is outside"
                    f" the size of document {path!r}."
                ) from pdfium_exc
            raise

        render_kwargs: dict[str, Any] = {}
        if dpi is not None:
            render_kwargs["scale"] = dpi / 72

        rendered_page = page.render(**render_kwargs)
        rendered_page_pil = rendered_page.to_pil()
        if needs_bbox:
            # Apply white border padding to increase bounding box reliability
            padded_pil, offset_x, offset_y = pad_image_with_border(
                rendered_page_pil, border
            )
            image_data_uri = encode_image_to_base64(padded_pil, format="PNG")
            padded_height, padded_width = padded_pil.height, padded_pil.width
        else:
            image_data_uri = encode_image_to_base64(rendered_page_pil, format="PNG")
            offset_x = offset_y = padded_height = padded_width = 0

    return (
        page_num,
        image_data_uri,
        rendered_page_pil,
        padded_height,
        padded_width,
        offset_x,
        offset_y,
    )


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
    failover_parser: str | PDFParserFn | None = None,
    num_workers: int = min(os.cpu_count() or 1, 4),
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
        failover_parser: Optional PDF parser to use when nemotron-parse fails on a
            given page. Can be a callable or an importable fully qualified name.
            Any metadata from the failover reader is not used (as of now).
        num_workers: Number of worker processes for parallel page rendering,
            default targets 4 processes.
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

    # Pre-render and send to model API in a pipeline
    needs_bbox = parse_media and not full_page
    path_str = str(path)
    render_args = [
        (path_str, i, dpi, border, needs_bbox, page_range)
        for i in resolve_page_range(page_range, page_count)
    ]
    render_kwargs: dict[str, Any] = {}
    if dpi is not None:
        render_kwargs["scale"] = dpi / 72

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

    async def process_page(
        i: int,
        image_data_uri: str,
        rendered_page_pil: "Image.Image",
        padded_height: int,
        padded_width: int,
        offset_x: int,
        offset_y: int,
    ) -> tuple[str, str | tuple[str, list[ParsedMedia]]]:
        """Process a pre-rendered page via the API and return its content."""
        tool_name: Literal["markdown_bbox", "markdown_no_bbox"] = (
            "markdown_bbox" if needs_bbox else "markdown_no_bbox"
        )

        try:
            try:
                response = await call_fn(
                    image=image_data_uri, tool_name=tool_name, **api_params
                )
            except NemotronLengthError:
                if tool_name != "markdown_bbox":
                    raise
                # Fallback to detection_only + markdown_no_bbox to reinvent
                # markdown_bbox, bypassing its length error
                detection_results = await call_fn(
                    image=image_data_uri, tool_name="detection_only", **api_params
                )

                async def extract_text(
                    detection: NemotronParseAnnotatedBBox,
                ) -> NemotronParseMarkdownBBox:
                    # Convert bbox from normalized [0, 1] to padded image pixel coordinates
                    padded_bbox = detection.bbox.to_page_coordinates(
                        padded_height, padded_width
                    )
                    original_bbox = (
                        # xmin, ymin
                        max(0, padded_bbox[0] - offset_x),
                        max(0, padded_bbox[1] - offset_y),
                        # xmax, ymax
                        min(rendered_page_pil.width, padded_bbox[2] - offset_x),
                        min(rendered_page_pil.height, padded_bbox[3] - offset_y),
                    )
                    # Crop original image at bbox (without border)
                    region_pil = rendered_page_pil.crop(original_bbox)  # type: ignore[arg-type]
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
        del image_data_uri  # Free up memory as API call is done

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
            media_metadata["info_hashable"] = json.dumps(media_metadata, sort_keys=True)
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
                # Convert bbox from normalized [0, 1] to padded image pixel coordinates
                padded_bbox = item.bbox.to_page_coordinates(padded_height, padded_width)
                # Adjust bbox to account for padding offsets
                # Also if the bbox had extended into the padding zone,
                # clamp it here as we're ditching the padding
                original_bbox = (
                    # xmin, ymin
                    max(0, padded_bbox[0] - offset_x),
                    max(0, padded_bbox[1] - offset_y),
                    # xmax, ymax
                    min(rendered_page_pil.width, padded_bbox[2] - offset_x),
                    min(rendered_page_pil.height, padded_bbox[3] - offset_y),
                )
                region_pix = rendered_page_pil.crop(original_bbox)  # type: ignore[arg-type]
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

    # Pipeline rendering with API calls: each page is rendered then immediately
    # sent to the API, overlapping compute-bound rendering with network-bound waits
    loop = asyncio.get_running_loop()
    executor = (
        ProcessPoolExecutor(max_workers=num_workers)
        if num_workers > 1 and len(render_args) > 1
        else None
    )

    async def render_and_process(
        page_args: tuple,
    ) -> tuple[str, str | tuple[str, list[ParsedMedia]]]:
        if executor is not None:
            rendered = await loop.run_in_executor(executor, _render_page, *page_args)
        else:
            rendered = _render_page(*page_args)
        return await process_page(*rendered)

    content: dict[str, str | tuple[str, list[ParsedMedia]]] = {}
    total_length = count_media = 0
    try:
        gather = (
            asyncio.gather(*(render_and_process(args) for args in render_args))
            if concurrency is None
            else gather_with_concurrency(
                concurrency, (render_and_process(args) for args in render_args)
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
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

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
