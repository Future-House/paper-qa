import json
import os
from itertools import starmap
from multiprocessing import Pool

import pymupdf
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError, clean_invalid_unicode
from pydantic import JsonValue


def setup_pymupdf_python_logging() -> None:
    """
    Configure PyMuPDF to use Python logging.

    SEE: https://pymupdf.readthedocs.io/en/latest/app3.html#diagnostics
    """
    pymupdf.set_messages(pylogging=True)


BLOCK_TEXT_INDEX = 4
# Attributes of pymupdf.Pixmap that contain useful metadata
PYMUPDF_PIXMAP_ATTRS = {
    "alpha",
    # YAGNI on "digest" because it's not JSON serializable
    "height",
    "irect",
    "is_monochrome",
    "is_unicolor",
    "n",
    "size",
    "stride",
    "width",
    "x",
    "xres",
    "y",
    "yres",
}


def resolve_page_range(
    page_range: int | tuple[int, int] | None, page_count: int
) -> range:
    """Convert a 1-indexed page_range into a 0-indexed range object."""
    if page_range is None:
        return range(page_count)
    if isinstance(page_range, int):
        return range(page_range - 1, page_range)
    return range(page_range[0] - 1, page_range[1])


def _extract_page_text(
    file: pymupdf.Document,
    page_num: int,
    path: str | os.PathLike,
    use_block_parsing: bool,
    page_size_limit: int | None = None,
) -> tuple[pymupdf.Page, str]:
    """Load a PDF page and extract its text.

    Args:
        file: An open (assumed) PyMuPDF document.
        page_num: Zero-indexed page number to load.
        path: Path to the PDF file (used in error messages).
        use_block_parsing: If True, extract text block-wise,
            preserving the order of text blocks as they appear in the PDF.
        page_size_limit: Optional character limit for a single page's text.

    Returns:
        A two-tuple of loaded PyMuPDF page and extracted text.

    Raises:
        ImpossibleParsingError: If the page cannot be loaded or its text
            exceeds page_size_limit.
    """
    try:
        page = file.load_page(page_num)
    except pymupdf.mupdf.FzErrorFormat as exc:
        raise ImpossibleParsingError(
            f"Page loading via {pymupdf.__name__} failed on page {page_num} of"
            f" {file.page_count} for the PDF at path {path}, likely this PDF"
            " file is corrupt."
        ) from exc

    if use_block_parsing:
        # NOTE: this block-based parsing appears to be better, but until
        # fully validated on 1+ benchmarks, it's considered experimental

        # Extract text blocks from the page
        # Note: sort=False is important to preserve the order of text blocks
        # as they appear in the PDF
        blocks = page.get_text("blocks", sort=False)

        # Concatenate text blocks into a single string
        text = "\n".join(
            block[BLOCK_TEXT_INDEX] for block in blocks if len(block) > BLOCK_TEXT_INDEX
        )
    else:
        text = page.get_text("text", sort=True)

    if page_size_limit and len(text) > page_size_limit:
        raise ImpossibleParsingError(
            f"The text in page {page_num} of {file.page_count} was {len(text)}"
            f" chars long, which exceeds the {page_size_limit} char limit for"
            f" the PDF at path {path}."
        )
    return page, text


def _parse_single_page_screenshot(
    path: str,
    page_num: int,
    dpi: float | None,
    page_size_limit: int | None,
    use_block_parsing: bool,
) -> tuple[int, str, list[ParsedMedia]]:
    """Worker function for parallel full-page screenshot parsing.

    NOTE: must be top-level for pickling.
    """
    with pymupdf.open(path) as file:
        page, text = _extract_page_text(
            file, page_num, path, use_block_parsing, page_size_limit
        )
        pix = page.get_pixmap(dpi=dpi)
        media_metadata: dict[str, JsonValue] = {"type": "screenshot"} | {
            a: getattr(pix, a) for a in PYMUPDF_PIXMAP_ATTRS
        }
        media_metadata["info_hashable"] = json.dumps(media_metadata, sort_keys=True)
        # Add page number after info_hashable so differing pages
        # don't break the cache key
        media_metadata["page_num"] = page_num + 1
        media = [ParsedMedia(index=0, data=pix.tobytes(), info=media_metadata)]
    return page_num, text, media


def parse_pdf_to_pages(
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    page_range: int | tuple[int, int] | None = None,
    use_block_parsing: bool = False,
    parse_media: bool = True,
    full_page: bool = False,
    image_cluster_tolerance: float | tuple[float, float] = 25,
    dpi: float | None = None,
    num_workers: int = min(os.cpu_count() or 1, 4),
    **_,
) -> ParsedText:
    """Parse a PDF.

    Args:
        path: Path to the PDF file to parse.
        page_size_limit: Sensible character limit one page's text,
            used to catch bad PDF reads.
        use_block_parsing: Opt-in flag to parse text block-wise.
        parse_media: Flag to also parse media (e.g. images, tables).
        full_page: Set True to screenshot the entire page as one image,
            instead of parsing individual images or tables.
        image_cluster_tolerance: Tolerance (points) passed to `Page.cluster_drawings`.
            Can be a single value to apply to both X and Y directions,
            or a two-tuple to specify X and Y directions separately.
            The default was chosen to perform well on image extraction from LitQA2 PDFs.
        dpi: Optional DPI (dots per inch) for image resolution,
            if left unspecified PyMuPDF's default resolution from
            pymupdf.Page.get_pixmap will be applied.
        page_range: Optional start_page or two-tuple of inclusive (start_page, end_page)
            to parse only specific pages, where pages are one-indexed.
            Leaving as the default of None will parse all pages.
        num_workers: Number of worker processes for parallel full-page screenshots,
            default targets 4 processes.
        **_: Thrown away kwargs.
    """
    x_tol, y_tol = (
        image_cluster_tolerance
        if isinstance(image_cluster_tolerance, tuple)
        else (image_cluster_tolerance, image_cluster_tolerance)
    )

    content: dict[str, str | tuple[str, list[ParsedMedia]]] = {}
    total_length = count_media = 0

    if full_page and parse_media:  # Capture the entire page as one image
        with pymupdf.open(path) as file:
            page_iter = resolve_page_range(page_range, file.page_count)
        path_str = str(path)
        args = [
            (path_str, i, dpi, page_size_limit, use_block_parsing) for i in page_iter
        ]
        if num_workers > 1:
            with Pool(num_workers) as pool:
                results = pool.starmap(_parse_single_page_screenshot, args)
        else:  # Avoid multiprocessing overhead when using just one process
            results = list(starmap(_parse_single_page_screenshot, args))
        for page_num, text, media in results:
            content[str(page_num + 1)] = text, media
            total_length += len(text)
            count_media += len(media)
    else:
        with pymupdf.open(path) as file:
            for i in resolve_page_range(page_range, file.page_count):
                page, text = _extract_page_text(
                    file, i, path, use_block_parsing, page_size_limit
                )
                media = []
                if parse_media:
                    # Capture drawings/figures
                    for box_i, box in enumerate(
                        page.cluster_drawings(
                            drawings=page.get_drawings(),
                            x_tolerance=x_tol,
                            y_tolerance=y_tol,
                        )
                    ):
                        pix = page.get_pixmap(clip=box, dpi=dpi)
                        media_metadata = {"bbox": tuple(box), "type": "drawing"} | {
                            a: getattr(pix, a) for a in PYMUPDF_PIXMAP_ATTRS
                        }
                        media_metadata["info_hashable"] = json.dumps(
                            media_metadata, sort_keys=True
                        )
                        # Add page number after info_hashable so differing pages
                        # don't break the cache key
                        media_metadata["page_num"] = i + 1
                        media.append(
                            ParsedMedia(
                                index=box_i, data=pix.tobytes(), info=media_metadata
                            )
                        )

                    # Capture tables
                    for table_i, table in enumerate(t for t in page.find_tables()):
                        pix = page.get_pixmap(clip=table.bbox, dpi=dpi)
                        media_metadata = {
                            "bbox": tuple(table.bbox),
                            "type": "table",
                        } | {a: getattr(pix, a) for a in PYMUPDF_PIXMAP_ATTRS}
                        media_metadata["info_hashable"] = json.dumps(
                            media_metadata, sort_keys=True
                        )
                        # Add page number after info_hashable so differing pages
                        # don't break the cache key
                        media_metadata["page_num"] = i + 1
                        media.append(
                            ParsedMedia(
                                index=table_i,
                                data=pix.tobytes(),
                                # On 9/14/2025, a `pymupdf.table.Table.to_markdown` stripped call returned:
                                # '|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|\n|---|---|---|---|---|---|---|---|\n||\x02\x03<br>|\x04\x05\x06\x07\x08<br> <br>|\x07\x08\x08<br>\n\x08<br>\x0e\x0f<br>\x17\x18\x18\x08<br>|\x02<br>\x0c\x10<br>\x11<br>\x19\r\x02\x1a\x00\x01\x02\x03<br>|\x11<br>\x12\x06\x05<br>\x0e\x13\x14\x15<br>\x04\x05\x06\x07<br>|\x05\x08<br>\x0c\x10<br>\x12\x06\x05<br>\x0e\x16\x13<br>|\x05\x08<br>\x0c\x10<br>\x12\x06\x05<br>\x0e\x16\x13<br>|'  # noqa: E501, W505
                                # This garbage led to `asyncpg==0.30.0` with a PostgreSQL 15 DB throwing:
                                # > asyncpg.exceptions.CharacterNotInRepertoireError:
                                # > invalid byte sequence for encoding "UTF8": 0x00
                                # On 12/30/2025 with pymupdf==1.26.7, a `pymupdf.table.Table.to_markdown` call on
                                # https://arxiv.org/pdf/1711.07566's page 3's Figure 2a's mesh and pixels example
                                # outputs an orphaned low surrogate (U+DC3C), which is interpreted as an
                                # incomplete UTF-16 surrogate pair downstream and causes:
                                # > UnicodeEncodeError: 'utf-8' codec can't encode character '\udc3c'
                                # > in position 46888: surrogates not allowed
                                # Thus, the extracted markdown is cleaned
                                text=(
                                    clean_invalid_unicode(table.to_markdown().strip())
                                ),
                                info=media_metadata,
                            )
                        )
                    content[str(i + 1)] = text, media
                else:
                    content[str(i + 1)] = text
                total_length += len(text)
                count_media += len(media)

    multimodal_string = f"|multimodal|dpi={dpi}" + (
        "|mode=full-page"
        if full_page
        else f"|mode=individual|x-tol={x_tol}|y-tol={y_tol}"
    )
    metadata = ParsedMetadata(
        parsing_libraries=[f"{pymupdf.__name__} ({pymupdf.__version__})"],
        total_parsed_text_length=total_length,
        count_parsed_media=count_media,
        name=(
            f"pdf|page_range={str(page_range).replace(' ', '')}"
            f"|block={use_block_parsing}{multimodal_string if parse_media else ''}"
        ),
    )
    return ParsedText(content=content, metadata=metadata)
