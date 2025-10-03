import os
import re

import pymupdf
from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError


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

# On 9/14/2025, a `pymupdf.table.Table.to_markdown` stripped call returned:
# '|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|\n|---|---|---|---|---|---|---|---|\n||\x02\x03<br>|\x04\x05\x06\x07\x08<br> <br>|\x07\x08\x08<br>\n\x08<br>\x0e\x0f<br>\x17\x18\x18\x08<br>|\x02<br>\x0c\x10<br>\x11<br>\x19\r\x02\x1a\x00\x01\x02\x03<br>|\x11<br>\x12\x06\x05<br>\x0e\x13\x14\x15<br>\x04\x05\x06\x07<br>|\x05\x08<br>\x0c\x10<br>\x12\x06\x05<br>\x0e\x16\x13<br>|\x05\x08<br>\x0c\x10<br>\x12\x06\x05<br>\x0e\x16\x13<br>|'  # noqa: E501, W505
# This garbage led to `asyncpg==0.30.0` with a PostgreSQL 15 DB throwing:
# > asyncpg.exceptions.CharacterNotInRepertoireError:
# > invalid byte sequence for encoding "UTF8": 0x00
# Thus, this regex exists to deny table markdown exports containing invalid chars
_INVALID_MD_CHARS = re.compile(r"\x00")


def parse_pdf_to_pages(
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    use_block_parsing: bool = False,
    parse_media: bool = True,
    full_page: bool = False,
    image_cluster_tolerance: float | tuple[float, float] = 25,
    image_dpi: float | None = 150,
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
        image_dpi: Dots per inch for images captured from the PDF.
        **_: Thrown away kwargs.
    """
    x_tol, y_tol = (
        image_cluster_tolerance
        if isinstance(image_cluster_tolerance, tuple)
        else (image_cluster_tolerance, image_cluster_tolerance)
    )

    with pymupdf.open(path) as file:
        content: dict[str, str | tuple[str, list[ParsedMedia]]] = {}
        total_length = count_media = 0

        for i in range(file.page_count):
            try:
                page = file.load_page(i)
            except pymupdf.mupdf.FzErrorFormat as exc:
                raise ImpossibleParsingError(
                    f"Page loading via {pymupdf.__name__} failed on page {i} of"
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
                    block[BLOCK_TEXT_INDEX]
                    for block in blocks
                    if len(block) > BLOCK_TEXT_INDEX
                )
            else:
                text = page.get_text("text", sort=True)

            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The text in page {i} of {file.page_count} was {len(text)} chars"
                    f" long, which exceeds the {page_size_limit} char limit for the PDF"
                    f" at path {path}."
                )
            media: list[ParsedMedia] = []
            if parse_media:
                if full_page:  # Capture the entire page as one image
                    pix = page.get_pixmap(dpi=image_dpi)
                    media.append(
                        ParsedMedia(
                            index=0,
                            data=pix.tobytes(),
                            info={"type": "screenshot"}
                            | {a: getattr(pix, a) for a in PYMUPDF_PIXMAP_ATTRS},
                        )
                    )
                else:
                    # Capture drawings/figures
                    for box_i, box in enumerate(
                        page.cluster_drawings(
                            drawings=page.get_drawings(),
                            x_tolerance=x_tol,
                            y_tolerance=y_tol,
                        )
                    ):
                        pix = page.get_pixmap(clip=box, dpi=image_dpi)
                        media.append(
                            ParsedMedia(
                                index=box_i,
                                data=pix.tobytes(),
                                info={"bbox": tuple(box), "type": "drawing"}
                                | {a: getattr(pix, a) for a in PYMUPDF_PIXMAP_ATTRS},
                            )
                        )

                    # Capture tables
                    for table_i, table in enumerate(t for t in page.find_tables()):
                        pix = page.get_pixmap(clip=table.bbox, dpi=image_dpi)
                        raw_md = table.to_markdown().strip()
                        media.append(
                            ParsedMedia(
                                index=table_i,
                                data=pix.tobytes(),
                                # If the markdown contains invalid control characters
                                # that'd trigger encoding errors later, drop the markdown
                                text=(
                                    None if _INVALID_MD_CHARS.search(raw_md) else raw_md
                                ),
                                info={"bbox": tuple(table.bbox), "type": "table"}
                                | {a: getattr(pix, a) for a in PYMUPDF_PIXMAP_ATTRS},
                            )
                        )
                content[str(i + 1)] = text, media
            else:
                content[str(i + 1)] = text
            total_length += len(text)
            count_media += len(media)

    metadata = ParsedMetadata(
        parsing_libraries=[f"{pymupdf.__name__} ({pymupdf.__version__})"],
        total_parsed_text_length=total_length,
        count_parsed_media=count_media,
        parse_type="pdf",
    )
    return ParsedText(content=content, metadata=metadata)
