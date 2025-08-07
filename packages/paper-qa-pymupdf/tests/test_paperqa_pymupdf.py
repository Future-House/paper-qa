import base64
import json
from pathlib import Path
from typing import cast

import pymupdf
import pytest
from paperqa import Doc, Docs
from paperqa.readers import PDFParserFn, chunk_pdf
from paperqa.utils import ImpossibleParsingError, bytes_to_string

from paperqa_pymupdf import parse_pdf_to_pages

REPO_ROOT = Path(__file__).parents[3]
STUB_DATA_DIR = REPO_ROOT / "tests" / "stub_data"


@pytest.mark.asyncio
async def test_parse_pdf_to_pages() -> None:
    assert isinstance(parse_pdf_to_pages, PDFParserFn)

    filepath = STUB_DATA_DIR / "pasa.pdf"
    parsed_text = parse_pdf_to_pages(filepath, use_block_parsing=True)
    assert isinstance(parsed_text.content, dict)
    assert "1" in parsed_text.content, "Parsed text should contain page 1"
    assert (
        "Abstract\n\nWe introduce PaSa, an advanced Paper Search"
        "\nagent powered by large language models."
    ) in parsed_text.content["1"][0], "Block parsing failed to handle abstract"

    # Check the images in Figure 1
    assert not isinstance(parsed_text.content["2"], str)
    p2_text, p2_media = parsed_text.content["2"]
    assert "Figure 1" in p2_text, "Expected Figure 1 title"
    assert "Crawler" in p2_text, "Expected Figure 1 contents"
    (p2_image,) = [m for m in p2_media if m.info["type"] == "drawing"]
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
    for attr in ("width", "height"):
        dim = p2_image.info[attr]
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
    texts = chunk_pdf(parsed_text, doc=doc, chunk_chars=3000, overlap=100)
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

    # Let's check the full page parsing behavior
    parsed_text_full_page = parse_pdf_to_pages(filepath, full_page=True)
    assert isinstance(parsed_text_full_page.content, dict)
    assert "1" in parsed_text_full_page.content, "Parsed text should contain page 1"
    assert "2" in parsed_text_full_page.content, "Parsed text should contain page 2"
    for page_num in ("1", "2"):
        page_content = parsed_text_full_page.content[page_num]
        assert not isinstance(page_content, str), f"Page {page_num} should have images"
        # Check each page has exactly one image
        page_text, (full_page_image,) = page_content
        assert page_text
        assert full_page_image.index == 0, "Full page image should have index 0"
        assert isinstance(full_page_image.data, bytes)
        assert len(full_page_image.data) > 0, "Full page image should have data"
        # Check useful attributes are present and are JSON serializable
        json.dumps(p2_image.info)
        for attr in ("width", "height"):
            dim = full_page_image.info[attr]
            assert isinstance(dim, int | float)
            assert dim > 0, "Edge length should be positive"

    # Check the no-media behavior
    parsed_text_no_media = parse_pdf_to_pages(filepath, parse_media=False)
    assert isinstance(parsed_text_no_media.content, dict)
    assert all(isinstance(c, str) for c in parsed_text_no_media.content.values())

    # Check metadata
    for pt in (parsed_text, parsed_text_full_page, parsed_text_no_media):
        (parsing_library,) = pt.metadata.parsing_libraries
        assert pymupdf.__name__ in parsing_library
        assert pt.metadata.parse_type == "pdf"

    # Check commonalities across all modes
    assert (
        len(parsed_text.content)
        == len(parsed_text_full_page.content)
        == len(parsed_text_no_media.content)
    ), "All modes should parse the same number of pages"


def test_page_size_limit_denial() -> None:
    with pytest.raises(ImpossibleParsingError, match="char limit"):
        parse_pdf_to_pages(STUB_DATA_DIR / "paper.pdf", page_size_limit=10)  # chars


def test_table_parsing() -> None:
    filepath = STUB_DATA_DIR / "influence.pdf"
    parsed_text = parse_pdf_to_pages(filepath)
    assert isinstance(parsed_text.content, dict)
    assert all(
        t and t[0] != "\n" and t[-1] != "\n" for t in parsed_text.content.values()
    ), "Expected no leading/trailing newlines in parsed text"
    assert "1" in parsed_text.content, "Parsed text should contain page 1"
    all_tables = {
        i: [m for m in pagenum_media[1] if m.info["type"] == "table"]
        for i, pagenum_media in parsed_text.content.items()
        if isinstance(pagenum_media, tuple)
    }
    assert (
        sum(len(tables) for tables in all_tables.values()) >= 2
    ), "Expected a few tables to be parsed"
