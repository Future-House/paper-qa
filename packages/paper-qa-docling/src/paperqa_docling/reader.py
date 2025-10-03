import collections
import io
import os
from importlib.metadata import version
from pathlib import Path

import docling
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.exceptions import ConversionError
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import DocItem, PictureItem, TableItem, TextItem
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError

DOCLING_VERSION = version(docling.__name__)
DOCLING_IMAGES_SCALE_PER_DPI = (
    72  # SEE: https://github.com/docling-project/docling/issues/2405
)


def parse_pdf_to_pages(  # noqa: PLR0912
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    parse_media: bool = True,
    pipeline_cls: type = StandardPdfPipeline,
    document_timeout: float | None = None,
    dpi: int | None = None,
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
        dpi: Optional DPI (dots per inch) for image resolution.
            Default PDF resolution is 72 DPI, so dpi of 144 would render at 2x scale.
        **_: Thrown away kwargs.
    """
    path = Path(path)

    if parse_media:
        pipeline_options = PdfPipelineOptions(
            generate_picture_images=True,
            generate_table_images=True,
            images_scale=1.0 if dpi is None else dpi / DOCLING_IMAGES_SCALE_PER_DPI,
            document_timeout=document_timeout,
        )
    else:
        pipeline_options = PdfPipelineOptions(document_timeout=document_timeout)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, pipeline_cls=pipeline_cls
            )
        }
    )
    try:
        # NOTE: this conversion is synchronous, because many backends only support sync
        # https://github.com/docling-project/docling/issues/2229#issuecomment-3269019929
        result = converter.convert(path)
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
                            "images_scale": pipeline_options.images_scale,
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
                            "images_scale": pipeline_options.images_scale,
                        },
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
        name=f"pdf|pipeline={pipeline_cls.__name__}{multimodal_string if parse_media else ''}",
    )
    return ParsedText(
        # Convert content from list to 2-tuple for return
        content={
            pgn: text if not parse_media else (text, images)
            for pgn, (text, images) in sorted(content.items(), key=lambda x: int(x[0]))
        },
        metadata=metadata,
    )
