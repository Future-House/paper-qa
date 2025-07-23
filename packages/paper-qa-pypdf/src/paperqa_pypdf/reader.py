import os

import pypdf
from paperqa.types import ParsedMetadata, ParsedText
from paperqa.utils import ImpossibleParsingError
from paperqa.version import __version__ as pqa_version


def parse_pdf_to_pages(
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    **_,
) -> ParsedText:
    with open(path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
        pages: dict[str, str] = {}
        total_length = 0

        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The text in page {i} of {len(pdf_reader.pages)} was {len(text)} chars"
                    f" long, which exceeds the {page_size_limit} char limit for the PDF"
                    f" at path {path}."
                )

            pages[str(i + 1)] = text
            total_length += len(text)

    metadata = ParsedMetadata(
        parsing_libraries=[f"pypdf ({pypdf.__version__})"],
        paperqa_version=pqa_version,
        total_parsed_text_length=total_length,
        parse_type="pdf",
    )
    return ParsedText(content=pages, metadata=metadata)
