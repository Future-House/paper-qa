import base64
import json
import re
from pathlib import Path
from typing import cast

import pypdf
import pytest
from paperqa import Doc, Docs
from paperqa.readers import PDFParserFn, chunk_pdf
from paperqa.utils import ImpossibleParsingError, bytes_to_string

from paperqa_pypdf import parse_pdf_to_pages

REPO_ROOT = Path(__file__).parents[3]
STUB_DATA_DIR = REPO_ROOT / "tests" / "stub_data"


@pytest.mark.asyncio
async def test_parse_pdf_to_pages() -> None:
    assert isinstance(parse_pdf_to_pages, PDFParserFn)

    filepath = STUB_DATA_DIR / "pasa.pdf"
    parsed_text_full_page = parse_pdf_to_pages(filepath, full_page=True)
    assert isinstance(parsed_text_full_page.content, dict)
    assert "1" in parsed_text_full_page.content, "Parsed text should contain page 1"
    assert isinstance(parsed_text_full_page.content["1"], tuple)
    matches = re.findall(
        r"Abstract\nWe introduce PaSa, an advanced Paper ?Search"
        r"\nagent powered by large language models.",
        parsed_text_full_page.content["1"][0],
    )
    assert len(matches) == 1, "Parsing failed to handle abstract"

    # Check the images in Figure 1
    assert not isinstance(parsed_text_full_page.content["2"], str)
    p2_text, p2_media = parsed_text_full_page.content["2"]
    assert "Figure 1" in p2_text, "Expected Figure 1 title"
    assert "Crawler" in p2_text, "Expected Figure 1 contents"
    (p2_image,) = p2_media
    assert p2_image.index == 0
    assert isinstance(p2_image.data, bytes)

    # Check the image is valid base64
    base64_data = bytes_to_string(p2_image.data)
    assert base64_data
    assert base64.b64decode(base64_data, validate=True) == p2_image.data

    # Check we can round-trip serialize the image
    serde_p2_image = type(p2_image).model_validate_json(p2_image.model_dump_json())
    assert serde_p2_image == p2_image

    # Check useful attributes are present and are JSON serializable
    json.dumps(p2_image.info)
    for attr in ("page_width", "page_height"):
        dim = serde_p2_image.info[attr]
        assert isinstance(dim, int | float)
        assert dim > 0, "Edge length should be positive"

    # Check Figure 1 can be used to answer questions
    doc = Doc(
        docname="He2025",
        dockey="stub",
        citation=(
            'He, Yichen, et al. "PaSa: An LLM Agent for Comprehensive Academic Paper'
            ' Search." *arXiv*, 2025, arXiv:2501.10120v1. Accessed 2025.'
        ),
    )
    texts = chunk_pdf(parsed_text_full_page, doc=doc, chunk_chars=3000, overlap=100)
    # pylint: disable=duplicate-code
    fig_1_text = texts[1]
    assert (
        "Figure 1: Architecture of PaSa" in fig_1_text.text
    ), "Expecting Figure 1 for the test to work"
    assert fig_1_text.media, "Expecting media to test multimodality"
    fig_1_text.text = "stub"  # Replace text to confirm multimodality works
    docs = Docs()
    assert await docs.aadd_texts(texts=[fig_1_text], doc=doc)
    for query, substrings_min_counts in [
        ("What actions can the Crawler take?", [(("search", "expand", "stop"), 2)]),
        ("What actions can the Selector take?", [(("select", "drop"), 2)]),
        (
            "How many User Query are there, and what do they do?",
            [(("two", "2"), 2), (("crawler", "selector"), 2)],
        ),
    ]:
        session = await docs.aquery(query=query)
        assert session.contexts, "Expected contexts to be generated"
        assert all(
            c.text.text == fig_1_text.text and c.text.media == fig_1_text.media
            for c in session.contexts
        ), "Expected context to reuse Figure 1's text and media"
        for substrings, min_count in cast(
            list[tuple[tuple[str, ...], int]], substrings_min_counts
        ):
            assert (
                sum(x in session.answer.lower() for x in substrings) >= min_count
            ), f"Expected {session.answer=} to have at {substrings} present"

    # Check the no-media behavior
    parsed_text_no_media = parse_pdf_to_pages(filepath, parse_media=False)
    assert isinstance(parsed_text_no_media.content, dict)
    assert all(isinstance(c, str) for c in parsed_text_no_media.content.values())

    # Check metadata
    for pt in (parsed_text_full_page, parsed_text_no_media):
        (parsing_library,) = pt.metadata.parsing_libraries
        assert pypdf.__name__ in parsing_library
        assert pt.metadata.parse_type == "pdf"

    # Check commonalities across all modes
    assert len(parsed_text_full_page.content) == len(
        parsed_text_no_media.content
    ), "All modes should parse the same number of pages"


def test_page_size_limit_denial() -> None:
    with pytest.raises(ImpossibleParsingError, match="char limit"):
        parse_pdf_to_pages(STUB_DATA_DIR / "paper.pdf", page_size_limit=10)  # chars
