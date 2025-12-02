import collections
import io
import json
import os
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import docling
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.exceptions import ConversionError
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import (
    DescriptionAnnotation,
    DocItem,
    FormulaItem,
    PictureItem,
    TableItem,
    TextItem,
)
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError

if TYPE_CHECKING:
    from docling.backend.abstract_backend import AbstractDocumentBackend

DOCLING_VERSION = version(docling.__name__)
DOCLING_IMAGES_SCALE_PER_DPI = (
    72  # SEE: https://github.com/docling-project/docling/issues/2405
)


def parse_pdf_to_pages(  # noqa: PLR0912
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    page_range: int | tuple[int, int] | None = None,
    parse_media: bool = True,
    pipeline_cls: type = StandardPdfPipeline,
    dpi: int | None = None,
    custom_pipeline_options: Mapping[str, Any] | None = None,
    backend: "type[AbstractDocumentBackend]" = DoclingParseV4DocumentBackend,
    **_,
) -> ParsedText:
    """Parse a PDF.

    Args:
        path: Path to the PDF file to parse.
        page_size_limit: Sensible character limit one page's text,
            used to catch bad PDF reads.
        parse_media: Flag to also parse media (e.g. images, tables).
        pipeline_cls: Optional custom pipeline class for document conversion.
            Default is Docling's standard PDF pipeline.
        dpi: Optional DPI (dots per inch) for image resolution,
            if left unspecified Docling's default 1.0 scale will be employed.
        custom_pipeline_options: Optional keyword arguments to use to construct the
            PDF pipeline's options.
        page_range: Optional start_page or two-tuple of inclusive (start_page, end_page)
            to parse only specific pages, where pages are one-indexed.
            Leaving as the default of None will parse all pages.
        backend: PDF backend class to use for parsing, defaults to docling-parse v4.
        **_: Thrown away kwargs.
    """
    path = Path(path)

    if parse_media:
        pipeline_options = PdfPipelineOptions(
            generate_picture_images=True,
            generate_table_images=True,
            images_scale=1.0 if dpi is None else dpi / DOCLING_IMAGES_SCALE_PER_DPI,
            **(custom_pipeline_options or {}),
        )
    else:
        pipeline_options = PdfPipelineOptions(**(custom_pipeline_options or {}))

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=pipeline_cls,
                backend=backend,
            )
        }
    )
    try:
        # NOTE: this conversion is synchronous, because many backends only support sync
        # https://github.com/docling-project/docling/issues/2229#issuecomment-3269019929
        result = converter.convert(
            path,
            page_range=(
                (page_range, page_range)
                if isinstance(page_range, int)
                else (page_range or DEFAULT_PAGE_RANGE)
            ),
        )
    except ConversionError as exc:
        raise ImpossibleParsingError(
            f"PDF reading via {docling.__name__} failed on the PDF at path {path!r},"
            " likely this PDF file is corrupt."
        ) from exc
    if result.status != ConversionStatus.SUCCESS:
        raise ImpossibleParsingError(
            f"Docling conversion failed with status {result.status.value!r}"
            f" for the PDF at path {path!r}."
        )

    doc = result.document

    # NOTE: the list value here is a two-item list of page text, page media.
    # It's mutable so we can append text and media as found
    content: dict[str, list] = collections.defaultdict(lambda: ["", []])
    total_length = count_media = 0

    for item, __ in doc.iterate_items():
        if not isinstance(item, DocItem) or not item.prov:
            raise NotImplementedError(
                f"Didn't yet handle the shape of node item {item}."
            )

        # NOTE: docling pages are 1-indexed
        page_nums = [prov.page_no for prov in item.prov]

        if isinstance(item, TextItem | FormulaItem):  # Handle items with text
            item_text = item.text
            if not item_text and isinstance(item, FormulaItem) and item.orig:
                # Sometimes the sanitization of formula text fails, so use the original
                item_text = item.orig
            for page_num in page_nums:
                new_text = (
                    item_text if not content[str(page_num)][0] else "\n\n" + item_text
                )
                total_length += len(new_text)
                if page_size_limit and total_length > page_size_limit:
                    raise ImpossibleParsingError(
                        f"The text in page {page_num} was {total_length} chars long,"
                        f" which exceeds the {page_size_limit} char limit"
                        f" for the PDF at path {path}."
                    )
                content[str(page_num)][0] += new_text

        if parse_media and isinstance(  # Handle images and formulae
            item, PictureItem | FormulaItem
        ):
            image_data = item.get_image(doc)
            if image_data:
                try:
                    (page_num,) = page_nums
                except ValueError as exc:
                    raise NotImplementedError(
                        f"Picture item spanning multiple pages {page_nums}"
                        " is not yet handled."
                    ) from exc

                # Convert PIL Image to bytes (PNG format)
                img_bytes = io.BytesIO()
                image_data.save(img_bytes, format="PNG")
                img_bytes.seek(0)  # Reset pointer before read to avoid empty data

                media_metadata = {
                    "type": "formula" if isinstance(item, FormulaItem) else "picture",
                    "width": image_data.width,
                    "height": image_data.height,
                    "bbox": item.prov[0].bbox.as_tuple(),
                    "images_scale": pipeline_options.images_scale,
                }
                annotations = [
                    x
                    for x in getattr(item, "annotations", [])
                    if isinstance(x, DescriptionAnnotation)
                ]
                if len(annotations) == 1:
                    # We don't set this text in ParsedMedia.text because it's
                    # a synthetic description, not actually text in the PDF,
                    # and we don't want citations going to synthetic text
                    media_metadata.update(
                        {
                            "description_text": annotations[0].text,
                            "description_provenance": annotations[0].provenance,
                        }
                    )
                elif len(annotations) > 1:
                    raise NotImplementedError(
                        f"Didn't yet handle 2+ picture description annotations {annotations}."
                    )

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
                media_metadata["page_num"] = page_num
                content[str(page_num)][1].append(
                    ParsedMedia(
                        index=len(content[str(page_num)][1]),
                        data=img_bytes.read(),
                        info=media_metadata,
                    )
                )
                count_media += 1

        elif parse_media and isinstance(item, TableItem):  # Handle tables
            table_image_data = item.get_image(doc)
            if table_image_data:
                try:
                    (page_num,) = page_nums
                except ValueError as exc:
                    raise NotImplementedError(
                        f"Table item spanning multiple pages {page_nums}"
                        " is not yet handled."
                    ) from exc

                img_bytes = io.BytesIO()
                table_image_data.save(img_bytes, format="PNG")
                img_bytes.seek(0)  # Reset pointer before read to avoid empty data

                media_metadata = {
                    "type": "table",
                    "width": table_image_data.width,
                    "height": table_image_data.height,
                    "bbox": item.prov[0].bbox.as_tuple(),
                    "images_scale": pipeline_options.images_scale,
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
                media_metadata["page_num"] = page_num
                content[str(page_num)][1].append(
                    ParsedMedia(
                        index=len(content[str(page_num)][1]),
                        data=img_bytes.read(),
                        text=item.export_to_markdown(doc),
                        info=media_metadata,
                    )
                )
                count_media += 1

    multimodal_string = f"|multimodal|images_scale={pipeline_options.images_scale}" + (
        "" if not custom_pipeline_options else f"|options={custom_pipeline_options}"
    )
    metadata = ParsedMetadata(
        parsing_libraries=[f"{docling.__name__} ({DOCLING_VERSION})"],
        total_parsed_text_length=total_length,
        count_parsed_media=count_media,
        name=(
            f"pdf|pipeline={pipeline_cls.__name__}"
            f"|page_range={str(page_range).replace(' ', '')}"  # Remove space in tuple
            f"|backend={backend.__name__}"
            f"{multimodal_string if parse_media else ''}"
        ),
    )
    return ParsedText(
        # Convert content from list to 2-tuple for return
        content={
            pgn: text if not parse_media else (text, images)
            for pgn, (text, images) in sorted(content.items(), key=lambda x: int(x[0]))
        },
        metadata=metadata,
    )
