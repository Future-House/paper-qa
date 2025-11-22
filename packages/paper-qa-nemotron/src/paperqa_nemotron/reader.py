"""Reader for PaperQA using Nvidia's nemotron-parse VLM."""

import asyncio
import io
import json
import os
from collections.abc import Mapping
from contextlib import closing
from typing import Any, cast

import pypdfium2 as pdfium
from lmi.utils import gather_with_concurrency
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError
from tenacity import RetryError

from paperqa_nemotron.api import (
    CLASSIFICATIONS_WITH_MEDIA,
    NemotronBBoxError,
    NemotronLengthError,
    NemotronParseClassification,
    _call_nvidia_api,
    _call_sagemaker_api,
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
        concurrency: Optional concurrency semaphore on concurrent processing of pages,
            use to put a ceiling on memory usage. Default is 128 to prioritize reader
            speed over memory, but not get obliterated by huge 1000-page PDFs.
            Set as None to disable concurrency limits, processing all pages at once.
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
                response = await call_fn(
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

    # No need to reflect api_params such as api_base or temperature here
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
