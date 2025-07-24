import base64
from pathlib import Path

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

    # Now let's check the images in Figure 1
    assert not isinstance(parsed_text.content["2"], str)
    p2_text, p2_media = parsed_text.content["2"]
    assert "Figure 1" in p2_text, "Expected Figure 1 title"
    assert "Crawler" in p2_text, "Expected Figure 1 contents"
    p2_images = [m for m in p2_media if m.info["type"] == "drawing"]
    assert (
        1 <= len(p2_images) <= 3
    ), "Expected Figure 1 to be present within a reasonable number of media"
    for i, image in enumerate(p2_images):
        assert image.index == i
        assert isinstance(image.data, bytes)

        # Check the image is valid base64
        base64_data = bytes_to_string(image.data)
        assert base64_data
        assert base64.b64decode(base64_data, validate=True) == image.data

        # Check we can round-trip serialize the image
        serde_image = type(image).model_validate_json(image.model_dump_json())
        assert serde_image == image

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
        for attr in ("width", "height"):
            dim = full_page_image.info[attr]
            assert isinstance(dim, int | float)
            assert dim > 0, "Edge length should be positive"

    # Compare individual mode with full page mode
    assert len(parsed_text_full_page.content) == len(
        parsed_text.content
    ), "Both modes should parse the same number of pages"

    # Check metadata
    (parsing_library,) = parsed_text.metadata.parsing_libraries
    assert pymupdf.__name__ in parsing_library
    assert parsed_text.metadata.parse_type == "pdf"

    doc = Doc(
        docname="He2025",
        dockey="stub",
        citation=(
            'He, Yichen, et al. "PaSa: An LLM Agent for Comprehensive Academic Paper'
            ' Search." *arXiv*, 2025, arXiv:2501.10120v1. Accessed 2025.'
        ),
    )
    texts = chunk_pdf(parsed_text, doc=doc, chunk_chars=3000, overlap=100)
    fig_1_text = texts[1]
    assert (
        "Figure 1: Architecture of PaSa" in fig_1_text.text
    ), "Expecting Figure 1 for the test to work"
    assert fig_1_text.media, "Expecting media to test multimodality"
    fig_1_text.text = "stub"  # Replace text to confirm multimodality works
    docs = Docs()
    assert await docs.aadd_texts(texts=[fig_1_text], doc=doc)
    # Now let's gather evidence, where the answer is in one of the images
    session = await docs.aquery(query="What actions can the Crawler take?")
    assert session.contexts, "Expected contexts to be generated"
    assert all(
        c.text.text == fig_1_text.text and c.text.media == fig_1_text.media
        for c in session.contexts
    )
    assert (
        sum(x in session.answer.lower() for x in ("search", "expand", "stop")) >= 2
    ), "Expected answer to have at least two of the three actions available"


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
