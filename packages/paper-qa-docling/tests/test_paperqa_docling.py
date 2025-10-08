import base64
import json
import os
import re
import time
from pathlib import Path
from typing import cast

import pytest
from lmi.utils import bytes_to_string
from paperqa import Doc, Docs
from paperqa.readers import PDFParserFn, chunk_pdf
from paperqa.utils import ImpossibleParsingError

from paperqa_docling import parse_pdf_to_pages

REPO_ROOT = Path(__file__).parents[3]
STUB_DATA_DIR = REPO_ROOT / "tests" / "stub_data"


@pytest.mark.asyncio
async def test_parse_pdf_to_pages() -> None:
    assert isinstance(parse_pdf_to_pages, PDFParserFn)

    filepath = STUB_DATA_DIR / "pasa.pdf"
    parsed_text = parse_pdf_to_pages(filepath)
    assert isinstance(parsed_text.content, dict)
    assert "1" in parsed_text.content, "Parsed text should contain page 1"
    assert isinstance(parsed_text.content["1"], tuple)
    # Weird spaces are because 'Pa S a' is bolded in the original PDF
    matches = re.findall(
        r"Abstract\n+We introduce PaSa, an advanced Pa ?per S ?e ?a ?rch"
        r" agent powered by large language models.",
        parsed_text.content["1"][0],
    )
    assert len(matches) == 1, "Parsing failed to handle abstract"

    # Check the images in Figure 1
    assert not isinstance(parsed_text.content["2"], str)
    p2_text, p2_media = parsed_text.content["2"]
    assert "Figure 1" in p2_text, "Expected Figure 1 title"
    assert "Crawler" in p2_text, "Expected Figure 1 contents"
    # pylint: disable=duplicate-code
    (p2_image,) = [m for m in p2_media if m.info["type"] == "picture"]
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
        assert (
            p2_image.info[attr] == serde_p2_image.info[attr]
        ), "Expected serialization to match original"
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
    fig_1_text = texts[1]
    assert (
        "Figure 1: Architecture of PaSa" in fig_1_text.text
    ), "Expecting Figure 1 for the test to work"
    assert fig_1_text.media, "Expecting media to test multimodality"
    fig_1_text.text = "stub"  # Replace text to confirm multimodality works
    docs = Docs()
    assert await docs.aadd_texts(texts=[fig_1_text], doc=doc)
    for query, substrings_min_counts in (
        ("What actions can the Crawler take?", [(("search", "expand", "stop"), 2)]),
        ("What actions can the Selector take?", [(("select", "drop"), 2)]),
        (
            "How many User Query are there, and what do they do?",
            [(("two", "2"), 2), (("crawler", "selector"), 2)],
        ),
    ):
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

    # Check our ability to get a high DPI
    parsed_text_high_dpi = parse_pdf_to_pages(filepath, dpi=144)
    assert isinstance(parsed_text_high_dpi.content, dict)
    assert not isinstance(parsed_text_high_dpi.content["2"], str)
    p2_text_high_dpi, p2_media_high_dpi = parsed_text_high_dpi.content["2"]
    assert "Figure 1" in p2_text_high_dpi, "Expected Figure 1 title"
    assert "Crawler" in p2_text_high_dpi, "Expected Figure 1 contents"
    (p2_image_high_dpi,) = [m for m in p2_media_high_dpi if m.info["type"] == "picture"]
    assert p2_image_high_dpi.info["images_scale"] == 2
    assert p2_image_high_dpi.info["height"] / p2_image.info["height"] == pytest.approx(  # type: ignore[operator]
        2, abs=0.01
    )
    assert p2_image_high_dpi.info["width"] / p2_image.info["width"] == pytest.approx(  # type: ignore[operator]
        2, abs=0.01
    )

    # Check metadata
    for pt in (parsed_text, parsed_text_no_media, parsed_text_high_dpi):
        (parsing_library,) = pt.metadata.parsing_libraries
        assert "docling" in parsing_library
        assert pt.metadata.name
        assert "pdf" in pt.metadata.name

    # Check commonalities across all modes
    assert (
        len(parsed_text.content)
        == len(parsed_text_no_media.content)
        == len(parsed_text_high_dpi.content)
    ), "All modes should parse the same number of pages"


def test_page_size_limit_denial() -> None:
    with pytest.raises(ImpossibleParsingError, match="char limit"):
        parse_pdf_to_pages(STUB_DATA_DIR / "paper.pdf", page_size_limit=10)  # chars


def test_invalid_pdf_is_denied(tmp_path) -> None:
    # pylint: disable=duplicate-code
    # This PDF content (actually it's a 404 HTML page) was seen with open access
    # in June 2025, so let's make sure it's denied
    bad_pdf_content = """<html>
<head><title>404 Not Found</title></head>
<body>
<center><h1>404 Not Found</h1></center>
<hr><center>nginx</center>
</body>
</html>
<!-- a padding to disable MSIE and Chrome friendly error page -->"""

    bad_pdf_path = tmp_path / "bad.pdf"
    bad_pdf_path.write_text(bad_pdf_content)

    with pytest.raises(ImpossibleParsingError, match="corrupt"):
        parse_pdf_to_pages(bad_pdf_path)


def test_table_parsing() -> None:
    # pylint: disable=duplicate-code
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
    all_tables = {k: v for k, v in all_tables.items() if v}
    assert (
        sum(len(tables) for tables in all_tables.values()) >= 2
    ), "Expected a few tables to be parsed for assertions to work"


IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"


def test_document_timeout_denial() -> None:
    tic = time.perf_counter()
    with pytest.raises(ImpossibleParsingError, match="partial"):
        parse_pdf_to_pages(
            STUB_DATA_DIR / "pasa.pdf", custom_pipeline_options={"document_timeout": 1}
        )
    if not IN_GITHUB_ACTIONS:  # GitHub Actions runners are too noisy in timing
        # On 10/3/2025 on a MacBook M3 Pro with 36-GB RAM, reading PaSa took 18.7-sec
        assert (
            time.perf_counter() - tic < 10
        ), "Expected document timeout to have taken much less time than a normal read"
