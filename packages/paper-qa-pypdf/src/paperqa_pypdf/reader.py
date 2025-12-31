import io
import json
import os
from contextlib import AbstractContextManager, closing, nullcontext
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any, cast

import pypdf
import pypdf.errors
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError, clean_invalid_unicode

from .utils import cluster_bboxes

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from PIL import Image


@unique
class MediaMode(StrEnum):
    """Mode for media extraction from PDFs."""

    NONE = ""  # No media extraction
    FULL_PAGE = "full-page"  # Screenshot entire page
    INDIVIDUAL_CLUSTERING = (  # Extract individual images then cluster
        "individual-clustering"
    )
    INDIVIDUAL = "individual"  # Extract individual images

    def __str__(self) -> str:
        return self.metadata_value

    @property
    def metadata_value(self) -> str:
        return self.value.removesuffix("-clustering")


# Attributes of pdfium.PdfBitmap that contain useful metadata
PDFIUM_BITMAP_ATTRS = {"width", "height", "stride", "n_channels", "mode"}


SCALE_TO_DPI = 72


def parse_pdf_to_pages(  # noqa: PLR0912
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    page_range: int | tuple[int, int] | None = None,
    parse_media: bool = True,
    full_page: bool = False,
    image_cluster_tolerance: float = 50,
    image_cluster_padding: float = 10,
    dpi: float | None = None,
    **_: Any,
) -> ParsedText:
    """Parse a PDF.

    Args:
        path: Path to the PDF file to parse.
        page_size_limit: Sensible character limit one page's text,
            used to catch bad PDF reads.
        parse_media: Flag to also parse media (e.g. images, tables).
        full_page: Set True to screenshot the entire page as one image,
            instead of parsing individual images or tables. When False and
            pdfplumber is available, nearby images will be clustered into
            figure regions.
        page_range: Optional start_page or two-tuple of inclusive (start_page, end_page)
            to parse only specific pages, where pages are one-indexed.
            Leaving as the default of None will parse all pages.
        image_cluster_tolerance: Maximum distance (pixels) between images
            to consider them part of the same cluster (inclusive).
            Only used when not screenshotting pages and pdfplumber is available.
        image_cluster_padding: Padding (pixels) to add around clustered
            image regions when rendering.
            Only used when not screenshotting pages and pdfplumber is available.
        dpi: Optional DPI (dots per inch) for image resolution,
            if left unspecified pypdfium2's default 1.0 scale will be employed.
        **_: Thrown away kwargs.
    """
    render_kwargs = {}
    if dpi is not None:
        render_kwargs["scale"] = dpi / SCALE_TO_DPI
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

        match (parse_media, full_page, pdfplumber is not None):
            case (False, _, _):
                media_mode = MediaMode.NONE
            case (True, True, _):
                media_mode = MediaMode.FULL_PAGE
            case (True, False, True):
                media_mode = MediaMode.INDIVIDUAL_CLUSTERING
            case (True, False, False):
                media_mode = MediaMode.INDIVIDUAL

        if media_mode in {MediaMode.FULL_PAGE, MediaMode.INDIVIDUAL_CLUSTERING}:
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

        if media_mode == MediaMode.INDIVIDUAL_CLUSTERING:
            plumber_pdf = pdfplumber.open(str(path))
            plumber_context: AbstractContextManager = plumber_pdf
        else:
            plumber_context = nullcontext()

        with pdf_context, plumber_context:
            for i, page in enumerate(
                pdf_reader.pages[start_page:end_page], start=start_page
            ):
                # On 12/30/2025 with pypdf==6.4.2, a `PageObject.extract_text` call on
                # https://arxiv.org/pdf/1711.07566's page 3's Figure 2a's rasterization
                # example outputs an orphaned low surrogate (U+DC63), which is
                # interpreted as an incomplete UTF-16 surrogate pair downstream and causes:
                # > UnicodeEncodeError: 'utf-8' codec can't encode character '\udc63'
                # > in position 17404: surrogates not allowed
                # Thus, the extracted text is cleaned
                text = clean_invalid_unicode(page.extract_text())
                if page_size_limit and len(text) > page_size_limit:
                    raise ImpossibleParsingError(
                        f"The text in page {i} of {len(pdf_reader.pages)} was {len(text)} chars"
                        f" long, which exceeds the {page_size_limit} char limit for the PDF"
                        f" at path {path}."
                    )

                if media_mode == MediaMode.FULL_PAGE:
                    pdfium_page: pdfium.PdfPage = pdf_doc[i]
                    pdfium_rendered_page: pdfium.PdfBitmap = pdfium_page.render(
                        **render_kwargs
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
                        ParsedMedia(index=0, data=buf.getvalue(), info=media_metadata)
                    ]
                    count_media += 1
                elif media_mode == MediaMode.INDIVIDUAL_CLUSTERING:
                    media_list: list[ParsedMedia] = []
                    plumber_page = plumber_pdf.pages[i]
                    page_width = plumber_page.width
                    page_height = plumber_page.height

                    # Cluster images into figure regions
                    pdfium_page = pdf_doc[i]
                    for cluster_idx, (x0, y0, x1, y1) in enumerate(
                        cluster_bboxes(
                            [
                                (img["x0"], img["top"], img["x1"], img["bottom"])
                                for img in plumber_page.images
                            ],
                            tolerance=image_cluster_tolerance,
                        )
                    ):
                        # Add padding around the figure region
                        x0 = max(0, x0 - image_cluster_padding)
                        y0 = max(0, y0 - image_cluster_padding)
                        x1 = min(page_width, x1 + image_cluster_padding)
                        y1 = min(page_height, y1 + image_cluster_padding)

                        # Calculate and render the cropped region
                        pix = pdfium_page.render(
                            crop=(
                                x0,
                                page_height - y1,
                                page_width - x1,
                                page_height - (page_height - y0),
                            ),
                            **render_kwargs,
                        )
                        buf = io.BytesIO()
                        try:
                            pix.to_pil().save(buf, format="PNG")
                        except AttributeError as exc:
                            raise ImportError(
                                "Figure rendering requires 'Pillow' to be installed."
                                " Please install it via `pip install paper-qa-pypdf[media]`."
                            ) from exc

                        media_metadata = {
                            "type": "picture",
                            "bbox": (x0, y0, x1, y1),
                            "width": pix.width,
                            "height": pix.height,
                        }
                        media_metadata["info_hashable"] = json.dumps(
                            {
                                k: (
                                    v
                                    if k != "bbox"
                                    else tuple(round(x) for x in cast(tuple, v))
                                )
                                for k, v in media_metadata.items()
                            },
                            sort_keys=True,
                        )
                        # Add page number after info_hashable so differing pages
                        # don't break the cache key
                        media_metadata["page_num"] = i + 1
                        media_list.append(
                            ParsedMedia(
                                index=cluster_idx,
                                data=buf.getvalue(),
                                info=media_metadata,
                            )
                        )

                    # Extract tables
                    for table in plumber_page.find_tables():
                        x0, y0, x1, y1 = table.bbox
                        # Add padding around the table region
                        x0 = max(0, x0 - image_cluster_padding)
                        y0 = max(0, y0 - image_cluster_padding)
                        x1 = min(page_width, x1 + image_cluster_padding)
                        y1 = min(page_height, y1 + image_cluster_padding)

                        # Render the table region as an image
                        pix = pdfium_page.render(
                            crop=(
                                x0,
                                page_height - y1,
                                page_width - x1,
                                page_height - (page_height - y0),
                            ),
                            **render_kwargs,
                        )
                        buf = io.BytesIO()
                        pix.to_pil().save(buf, format="PNG")

                        table_metadata: dict[str, Any] = {
                            "type": "table",
                            "bbox": (x0, y0, x1, y1),
                            "width": pix.width,
                            "height": pix.height,
                        }
                        table_metadata["info_hashable"] = json.dumps(
                            {
                                k: (
                                    v
                                    if k != "bbox"
                                    else tuple(round(x) for x in cast(tuple, v))
                                )
                                for k, v in table_metadata.items()
                            },
                            sort_keys=True,
                        )
                        # Add page number after info_hashable so differing pages
                        # don't break the cache key
                        table_metadata["page_num"] = i + 1
                        media_list.append(
                            ParsedMedia(
                                index=len(media_list),
                                data=buf.getvalue(),
                                info=table_metadata,
                            )
                        )

                    pages[str(i + 1)] = text, media_list
                    count_media += len(media_list)
                elif media_mode == MediaMode.INDIVIDUAL:
                    # NOTE: if Pillow is not installed,
                    # PyPDF will blow up here with a nice message
                    media_list = []
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

    # Determine mode string and parsing libraries based on actual mode used
    lib_parts = [f"{pypdf.__name__} ({pypdf.__version__})"]
    if media_mode == MediaMode.FULL_PAGE:
        lib_parts.append(f"{pdfium.__name__} ({pdfium.PYPDFIUM_INFO})")
    elif media_mode == MediaMode.INDIVIDUAL_CLUSTERING:
        lib_parts.extend(
            [
                f"{pdfium.__name__} ({pdfium.PYPDFIUM_INFO})",
                f"pdfplumber ({pdfplumber.__version__})",
            ]
        )
    multimodal_string = f"|multimodal|dpi={dpi}|mode={media_mode.metadata_value}"
    metadata = ParsedMetadata(
        parsing_libraries=[", ".join(lib_parts)],
        total_parsed_text_length=total_length,
        count_parsed_media=count_media,
        name=(
            f"pdf|page_range={str(page_range).replace(' ', '')}"
            f"{multimodal_string if media_mode != MediaMode.NONE else ''}"
        ),
    )
    return ParsedText(content=pages, metadata=metadata)
