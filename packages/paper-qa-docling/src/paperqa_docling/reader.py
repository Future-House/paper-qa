import collections
import io
import os
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import docling
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import DocItem, PictureItem, TableItem, TextItem
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError

if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import PipelineOptions

DOCLING_VERSION = version(docling.__name__)


def parse_pdf_to_pages(  # noqa: PLR0912
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    parse_media: bool = True,
    pipeline_cls: type = StandardPdfPipeline,
    document_timeout: float | None = None,
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
        document_timeout: Optional timeout (sec) for document processing.
        **_: Thrown away kwargs.
    """
    path = Path(path)

    if parse_media:
        pipeline_options: "PipelineOptions | None" = PdfPipelineOptions(  # noqa: UP037
            generate_picture_images=True,
            generate_table_images=True,
            document_timeout=document_timeout,
        )
    else:
        pipeline_options = PdfPipelineOptions(document_timeout=document_timeout)

    result = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, pipeline_cls=pipeline_cls
            )
        }
    ).convert(path)
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

        if isinstance(item, TextItem):
            for page_num in page_nums:
                new_text = (
                    item.text if not content[str(page_num)][0] else "\n\n" + item.text
                )
                total_length += len(new_text)
                if page_size_limit and total_length > page_size_limit:
                    raise ImpossibleParsingError(
                        f"The text in page {page_num} was {total_length} chars long,"
                        f" which exceeds the {page_size_limit} char limit"
                        f" for the PDF at path {path}."
                    )
                content[str(page_num)][0] += new_text

        if parse_media and isinstance(item, PictureItem):  # Handle images
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

                content[str(page_num)][1].append(
                    ParsedMedia(
                        index=len(content[str(page_num)][1]),
                        data=img_bytes.read(),
                        info={
                            "type": "picture",
                            "width": image_data.width,
                            "height": image_data.height,
                            "bbox": item.prov[0].bbox.as_tuple(),
                        },
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

                content[str(page_num)][1].append(
                    ParsedMedia(
                        index=len(content[str(page_num)][1]),
                        data=img_bytes.read(),
                        text=item.export_to_markdown(doc),
                        info={
                            "type": "table",
                            "width": table_image_data.width,
                            "height": table_image_data.height,
                            "bbox": item.prov[0].bbox.as_tuple(),
                        },
                    )
                )
                count_media += 1

    metadata = ParsedMetadata(
        parsing_libraries=[f"{docling.__name__} ({DOCLING_VERSION})"],
        total_parsed_text_length=total_length,
        count_parsed_media=count_media,
        parse_type="pdf",
    )
    return ParsedText(
        # Convert content from list to 2-tuple for return
        content={
            pgn: text if not parse_media else (text, images)
            for pgn, (text, images) in sorted(content.items(), key=lambda x: int(x[0]))
        },
        metadata=metadata,
    )
