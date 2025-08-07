import io
import os
from typing import Any

import pypdf
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError
from paperqa.version import __version__ as pqa_version

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

# Attributes of pdfium.PdfBitmap that contain useful metadata
PDFIUM_BITMAP_ATTRS = {"width", "height", "stride", "n_channels", "mode"}


def parse_pdf_to_pages(
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    parse_media: bool = True,
    full_page: bool = False,
    **_: Any,
) -> ParsedText:
    """Parse a PDF.

    Args:
        path: Path to the PDF file to parse.
        page_size_limit: Sensible character limit one page's text,
            used to catch bad PDF reads.
        parse_media: Flag to also parse media (e.g. images, tables).
        full_page: Set True to screenshot the entire page as one image,
            instead of parsing individual images or tables.
        **_: Thrown away kwargs.
    """
    with open(path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)

        pages: dict[str, str | tuple[str, list[ParsedMedia]]] = {}
        total_length = count_media = 0
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The text in page {i} of {len(pdf_reader.pages)} was {len(text)} chars"
                    f" long, which exceeds the {page_size_limit} char limit for the PDF"
                    f" at path {path}."
                )

            if parse_media:
                if not full_page:
                    raise NotImplementedError(
                        "Media support without just a screenshot of the full page"
                        " is not yet implemented."
                    )
                # Render the whole page to a PIL image.
                try:
                    pdf_doc = pdfium.PdfDocument(str(path))
                except AttributeError as exc:
                    raise ImportError(
                        "Media parsing requires 'pypdfium2' to be installed for rasterization support."
                        " Please install it via `pip install paper-qa-pypdf[media]`."
                    ) from exc

                pdfium_page: pdfium.PdfPage = pdf_doc[i]
                pdfium_rendered_page: pdfium.PdfBitmap = pdfium_page.render(scale=1)
                buf = io.BytesIO()
                pdfium_rendered_page.to_pil().save(buf, format="PNG")
                pages[str(i + 1)] = text, [
                    ParsedMedia(
                        index=0,
                        data=buf.getvalue(),
                        info={
                            "type": "screenshot",
                            "page_width": pdfium_page.get_width(),
                            "page_height": pdfium_page.get_height(),
                        }
                        | {
                            f"bitmap_{a}": getattr(pdfium_rendered_page, a)
                            for a in PDFIUM_BITMAP_ATTRS
                        },
                    )
                ]
                count_media += 1
            else:
                pages[str(i + 1)] = text
            total_length += len(text)

    pypdf_version_str = f"{pypdf.__name__} ({pypdf.__version__})"
    metadata = ParsedMetadata(
        parsing_libraries=[
            (
                f"{pypdf_version_str}, {pdfium.__name__} ({pdfium.PYPDFIUM_INFO})"
                if parse_media
                else pypdf_version_str
            )
        ],
        paperqa_version=pqa_version,
        total_parsed_text_length=total_length,
        count_parsed_media=count_media,
        parse_type="pdf",
    )
    return ParsedText(content=pages, metadata=metadata)
