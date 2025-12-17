import io
import json
import os
from contextlib import AbstractContextManager, closing, nullcontext
from typing import TYPE_CHECKING, Any, cast

import pypdf
import pypdf.errors
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

if TYPE_CHECKING:
    from PIL import Image

# Attributes of pdfium.PdfBitmap that contain useful metadata
PDFIUM_BITMAP_ATTRS = {"width", "height", "stride", "n_channels", "mode"}


def parse_pdf_to_pages(  # noqa: PLR0912
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    page_range: int | tuple[int, int] | None = None,
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
        page_range: Optional start_page or two-tuple of inclusive (start_page, end_page)
            to parse only specific pages, where pages are one-indexed.
            Leaving as the default of None will parse all pages.
        **_: Thrown away kwargs.
    """
    with open(path, "rb") as file:  # noqa: PLR1702
        try:
            pdf_reader = pypdf.PdfReader(file)
        except pypdf.errors.PdfReadError as exc:
            raise ImpossibleParsingError(
                f"PDF reading via {pypdf.__name__} failed on the PDF at path {path!r},"
                " likely this PDF file is corrupt."
            ) from exc

        pages: dict[str, str | tuple[str, list[ParsedMedia]]] = {}
        total_length = count_media = 0

        # Determine page range (convert from 1-indexed to 0-indexed)
        if page_range is None:
            start_page, end_page = 0, len(pdf_reader.pages)
        elif isinstance(page_range, int):
            start_page, end_page = page_range - 1, page_range
        else:
            start_page, end_page = page_range[0] - 1, page_range[1]

        if parse_media and full_page:
            try:
                pdf_doc = pdfium.PdfDocument(str(path))
            except AttributeError as exc:
                raise ImportError(
                    "Media parsing requires 'pypdfium2' to be installed for rasterization support."
                    " Please install it via `pip install paper-qa-pypdf[media]`."
                ) from exc
            pdf_context: AbstractContextManager = closing(pdf_doc)
        else:
            pdf_context = nullcontext()

        with pdf_context:
            for i, page in enumerate(
                pdf_reader.pages[start_page:end_page], start=start_page
            ):
                text = page.extract_text()
                if page_size_limit and len(text) > page_size_limit:
                    raise ImpossibleParsingError(
                        f"The text in page {i} of {len(pdf_reader.pages)} was {len(text)} chars"
                        f" long, which exceeds the {page_size_limit} char limit for the PDF"
                        f" at path {path}."
                    )

                if parse_media:
                    if full_page:
                        pdfium_page: pdfium.PdfPage = pdf_doc[i]
                        pdfium_rendered_page: pdfium.PdfBitmap = pdfium_page.render(
                            scale=1
                        )
                        buf = io.BytesIO()
                        try:
                            pdfium_rendered_page.to_pil().save(buf, format="PNG")
                        except AttributeError as exc:
                            # Nice-ify pypdfium2's bad error message
                            raise ImportError(
                                "Full page media rendering requires 'Pillow' to be installed."
                                " Please install it via `pip install paper-qa-pypdf[media]`."
                            ) from exc
                        media_metadata = {
                            "type": "screenshot",
                            "page_width": pdfium_page.get_width(),
                            "page_height": pdfium_page.get_height(),
                        } | {
                            f"bitmap_{a}": getattr(pdfium_rendered_page, a)
                            for a in PDFIUM_BITMAP_ATTRS
                        }
                        media_metadata["info_hashable"] = json.dumps(
                            media_metadata, sort_keys=True
                        )
                        # Add page number after info_hashable so differing pages
                        # don't break the cache key
                        media_metadata["page_num"] = i + 1
                        pages[str(i + 1)] = text, [
                            ParsedMedia(
                                index=0, data=buf.getvalue(), info=media_metadata
                            )
                        ]
                        count_media += 1
                    else:
                        media_list: list[ParsedMedia] = []
                        # NOTE: if Pillow is not installed,
                        # PyPDF will blow up here with a nice message
                        for img_idx, img_obj in enumerate(page.images):
                            width, height = cast("Image.Image", img_obj.image).size
                            media_metadata = {
                                "type": "picture",
                                "width": width,
                                "height": height,
                            }
                            media_metadata["info_hashable"] = json.dumps(
                                media_metadata, sort_keys=True
                            )
                            # Add page number after info_hashable so differing pages
                            # don't break the cache key
                            media_metadata["page_num"] = i + 1
                            media_list.append(
                                ParsedMedia(
                                    index=img_idx,
                                    data=img_obj.data,
                                    info=media_metadata,
                                )
                            )
                        pages[str(i + 1)] = text, media_list
                        count_media += len(media_list)
                else:
                    pages[str(i + 1)] = text
                total_length += len(text)

    pypdf_version_str = f"{pypdf.__name__} ({pypdf.__version__})"
    multimodal_string = f"|multimodal|mode={'full-page' if full_page else 'individual'}"
    if parse_media and full_page:
        parsing_libs = [
            f"{pypdf_version_str}, {pdfium.__name__} ({pdfium.PYPDFIUM_INFO})"
        ]
    else:
        parsing_libs = [pypdf_version_str]
    metadata = ParsedMetadata(
        parsing_libraries=parsing_libs,
        total_parsed_text_length=total_length,
        count_parsed_media=count_media,
        name=(
            f"pdf|page_range={str(page_range).replace(' ', '')}"
            f"{multimodal_string if parse_media else ''}"
        ),
    )
    return ParsedText(content=pages, metadata=metadata)
