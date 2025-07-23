import os

import pymupdf
from paperqa.types import ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError
from paperqa.version import __version__ as pqa_version


def setup_pymupdf_python_logging() -> None:
    """
    Configure PyMuPDF to use Python logging.

    SEE: https://pymupdf.readthedocs.io/en/latest/app3.html#diagnostics
    """
    pymupdf.set_messages(pylogging=True)


BLOCK_TEXT_INDEX = 4


def parse_pdf_to_pages(
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    use_block_parsing: bool = False,
    **_,
) -> ParsedText:

    with pymupdf.open(path) as file:
        pages: dict[str, str] = {}
        total_length = 0

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
            pages[str(i + 1)] = text
            total_length += len(text)

    metadata = ParsedMetadata(
        parsing_libraries=[f"pymupdf ({pymupdf.__version__})"],
        paperqa_version=pqa_version,
        total_parsed_text_length=total_length,
        parse_type="pdf",
    )
    return ParsedText(content=pages, metadata=metadata)
