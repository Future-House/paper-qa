import re
from pathlib import Path

import pypdf
import pytest
from paperqa.readers import PDFParserFn
from paperqa.utils import ImpossibleParsingError

from paperqa_pypdf import parse_pdf_to_pages

REPO_ROOT = Path(__file__).parents[3]
STUB_DATA_DIR = REPO_ROOT / "tests" / "stub_data"


def test_parse_pdf_to_pages() -> None:
    assert isinstance(parse_pdf_to_pages, PDFParserFn)

    filepath = STUB_DATA_DIR / "pasa.pdf"
    parsed_text = parse_pdf_to_pages(filepath)
    assert isinstance(parsed_text.content, dict)
    assert "1" in parsed_text.content, "Parsed text should contain page 1"
    matches = re.findall(
        r"Abstract\nWe introduce PaSa, an advanced Paper ?Search"
        r"\nagent powered by large language models.",
        parsed_text.content["1"],
    )
    assert len(matches) == 1, "Parsing failed to handle abstract"

    # Check Figure 1
    p2_text = parsed_text.content["2"]
    assert "Figure 1" in p2_text, "Expected Figure 1 title"
    assert "Crawler" in p2_text, "Expected Figure 1 contents"

    # Check metadata
    (parsing_library,) = parsed_text.metadata.parsing_libraries
    assert pypdf.__name__ in parsing_library
    assert parsed_text.metadata.parse_type == "pdf"


def test_page_size_limit_denial() -> None:
    with pytest.raises(ImpossibleParsingError, match="char limit"):
        parse_pdf_to_pages(STUB_DATA_DIR / "paper.pdf", page_size_limit=10)  # chars
