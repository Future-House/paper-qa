import base64
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from lmi.utils import bytes_to_string
from paperqa import Doc, Docs
from paperqa.readers import PDFParserFn, chunk_pdf
from paperqa.utils import REPLACEMENT_CHAR, ImpossibleParsingError, get_citation_ids

from paperqa_pypdf import parse_pdf_to_pages
from paperqa_pypdf.reader import MediaMode

REPO_ROOT = Path(__file__).parents[3]
STUB_DATA_DIR = REPO_ROOT / "tests" / "stub_data"


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
@pytest.mark.asyncio
async def test_parse_pdf_to_pages() -> None:
    assert isinstance(parse_pdf_to_pages, PDFParserFn)

    filepath = STUB_DATA_DIR / "pasa.pdf"
    parsed_text = parse_pdf_to_pages(filepath)
    assert isinstance(parsed_text.content, dict)
    assert len(parsed_text.content) == 15, "Expected all pages to be parsed"
    assert "1" in parsed_text.content, "Parsed text should contain page 1"
    assert isinstance(parsed_text.content["1"], tuple)
    p1_text = parsed_text.content["1"][0]
    matches = re.findall(
        r"Abstract\nWe introduce PaSa, an advanced Paper Search"
        r"\s?agent powered by large language models\.",
        p1_text,
    )
    assert len(matches) == 1, f"Parsing failed to handle abstract in {p1_text}."
    assert (
        p1_text.count("outperforms existing") == 1
    ), "Test expects one match of this substring"
    col_1_bottom_idx = p1_text.index("outperforms existing")
    assert (
        p1_text.count("address fine-grained") == 1
    ), "Test expects one match of this substring"
    col_2_top_idx = p1_text.index("address fine-grained")
    assert col_1_bottom_idx < col_2_top_idx, "Expected column ordering to be correct"

    # Check the images in Figure 1
    assert not isinstance(parsed_text.content["2"], str)
    p2_text, p2_media = parsed_text.content["2"]
    assert "Figure 1" in p2_text, "Expected Figure 1 title"
    assert "Crawler" in p2_text, "Expected Figure 1 contents"
    (p2_image,) = [m for m in p2_media if m.info["type"] == "picture"]
    assert p2_image.index == 0
    assert p2_image.info["page_num"] == 2
    assert p2_image.info["height"] == pytest.approx(135, rel=0.1)
    assert p2_image.info["width"] == pytest.approx(440, rel=0.1)
    p2_bbox = p2_image.info["bbox"]
    assert isinstance(p2_bbox, tuple)
    for i, value in enumerate((91.4, 65.6, 531.8, 201.7)):
        assert p2_bbox[i] == pytest.approx(value, rel=0.1)
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
            "How many User Query blue boxes are there, and what are they connected to?",
            [(("two", "2"), 1), (("crawler", "selector"), 2)],
        ),
    ):
        session = await docs.aquery(query=query)
        assert session.contexts, "Expected contexts to be generated"
        assert all(
            c.text.text == fig_1_text.text and c.text.media == fig_1_text.media
            for c in session.contexts
        ), "Expected context to reuse Figure 1's text and media"
        # Remove citations so numeric assertions don't have false positives
        raw_answer_no_citations = session.raw_answer
        for key in get_citation_ids(session.raw_answer):
            raw_answer_no_citations = raw_answer_no_citations.replace(f"({key})", "")
        for substrings, min_count in cast(
            list[tuple[tuple[str, ...], int]], substrings_min_counts
        ):
            assert (
                sum(x in raw_answer_no_citations.lower() for x in substrings)
                >= min_count
            ), f"Expected {raw_answer_no_citations=} to have {substrings} present"

    # Check the full page parsing behavior
    parsed_text_full_page = parse_pdf_to_pages(filepath, full_page=True)
    assert isinstance(parsed_text_full_page.content, dict)
    assert len(parsed_text_full_page.content) == 15, "Expected all pages to be parsed"
    assert "1" in parsed_text_full_page.content, "Parsed text should contain page 1"
    assert "2" in parsed_text_full_page.content, "Parsed text should contain page 2"
    for page_num in ("1", "2"):
        page_content = parsed_text_full_page.content[page_num]
        assert not isinstance(page_content, str), f"Page {page_num} should have images"
        # Check each page has exactly one image
        page_text, (full_page_image,) = page_content
        assert page_text
        assert full_page_image.index == 0, "Full page image should have index 0"
        assert full_page_image.info["type"] == "screenshot"
        assert full_page_image.info["page_num"] == int(page_num)
        assert full_page_image.info["page_height"] == pytest.approx(842, rel=0.01)
        assert full_page_image.info["page_width"] == pytest.approx(596, rel=0.01)
        assert isinstance(full_page_image.data, bytes)
        assert full_page_image.data, "Full page image should have data"
        # Check useful attributes are present and are JSON serializable
        json.dumps(full_page_image.info)
        for attr in ("page_width", "page_height"):
            dim = full_page_image.info[attr]
            assert isinstance(dim, int | float)
            assert dim > 0, "Edge length should be positive"

    # Check the no-media behavior
    parsed_text_no_media = parse_pdf_to_pages(filepath, parse_media=False)
    assert isinstance(parsed_text_no_media.content, dict)
    assert all(isinstance(c, str) for c in parsed_text_no_media.content.values())
    assert len(parsed_text_no_media.content) == 15, "Expected all pages to be parsed"

    # Check metadata
    for pt in (parsed_text, parsed_text_full_page, parsed_text_no_media):
        (parsing_library,) = pt.metadata.parsing_libraries
        assert "pypdf" in parsing_library
        assert pt.metadata.name
        assert "pdf" in pt.metadata.name
        assert "page_range=None" in pt.metadata.name

    # Check commonalities across all modes
    assert (
        len(parsed_text.content)
        == len(parsed_text_full_page.content)
        == len(parsed_text_no_media.content)
    ), "All modes should parse the same number of pages"


def test_page_range() -> None:
    filepath = STUB_DATA_DIR / "pasa.pdf"

    parsed_text_p1 = parse_pdf_to_pages(filepath, page_range=1)
    assert isinstance(parsed_text_p1.content, dict)
    assert list(parsed_text_p1.content) == ["1"]
    assert parsed_text_p1.metadata.name
    assert "page_range=1" in parsed_text_p1.metadata.name

    parsed_text_p12 = parse_pdf_to_pages(filepath, page_range=(1, 2))
    assert isinstance(parsed_text_p12.content, dict)
    assert list(parsed_text_p12.content) == ["1", "2"]
    assert parsed_text_p12.metadata.name
    assert "page_range=(1,2)" in parsed_text_p12.metadata.name

    # NOTE: exceeds 15-page PDF length
    parsed_text_p1_20 = parse_pdf_to_pages(filepath, page_range=(1, 20))
    assert isinstance(parsed_text_p1_20.content, dict)
    assert list(parsed_text_p1_20.content) == [
        str(i) for i in range(1, 15 + 1)
    ], "Expected pages to be truncated to 15 or us to get blown up"
    assert parsed_text_p1_20.metadata.name
    assert "page_range=(1,20)" in parsed_text_p1_20.metadata.name


def test_page_size_limit_denial() -> None:
    with pytest.raises(ImpossibleParsingError, match="char limit"):
        parse_pdf_to_pages(STUB_DATA_DIR / "paper.pdf", page_size_limit=10)  # chars


def test_invalid_pdf_is_denied(tmp_path) -> None:
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


def test_nonexistent_file_failure() -> None:
    filename = "/nonexistent/path/file.pdf"
    with pytest.raises(FileNotFoundError, match=filename):
        parse_pdf_to_pages(filename)


def test_table_parsing() -> None:
    filepath = STUB_DATA_DIR / "influence.pdf"
    parsed_text = parse_pdf_to_pages(filepath)
    assert isinstance(parsed_text.content, dict)
    assert all(
        t and t[0] != "\n" and t[-1] != "\n" for t in parsed_text.content.values()
    ), "Expected no leading/trailing newlines in parsed text"
    all_tables = {
        i: [m for m in pagenum_media[1] if m.info["type"] == "table"]
        for i, pagenum_media in parsed_text.content.items()
        if isinstance(pagenum_media, tuple)
    }
    all_tables = {k: v for k, v in all_tables.items() if v}
    assert (
        sum(len(tables) for tables in all_tables.values()) >= 2
    ), "Expected a few tables to be parsed"


def test_table_parsing_orphaned_surrogate() -> None:
    # Simulate orphaned low surrogate (U+DC63) in extracted text
    surrogate_char = chr(0xDC63)
    text_with_surrogate = f"Normal text {surrogate_char} more text"

    filepath = STUB_DATA_DIR / "paper.pdf"
    with patch("pypdf.PageObject.extract_text", return_value=text_with_surrogate):
        # Reading just page one without media speeds the test up
        parsed_text = parse_pdf_to_pages(filepath, page_range=1, parse_media=False)

    assert isinstance(parsed_text.content, dict)
    assert "1" in parsed_text.content
    page_text = parsed_text.content["1"]
    assert isinstance(page_text, str)
    assert surrogate_char not in page_text, "Expected no surrogate chars"
    assert REPLACEMENT_CHAR in page_text, "Expected replacement char(s)"
    assert "Normal text" in page_text, "Expected other text to be preserved"


def test_media_deduplication() -> None:
    parsed_text = parse_pdf_to_pages(STUB_DATA_DIR / "duplicate_media.pdf")
    assert isinstance(parsed_text.content, dict)
    assert len(parsed_text.content) == 5, "Expected full PDF read"
    all_media = [m for _, media in parsed_text.content.values() for m in media]  # type: ignore[misc]

    all_images = [m for m in all_media if m.info.get("type") == "picture"]
    assert (
        len(all_images) == 2 * 5
    ), "Expected each image (one/page) and equation (one/page) to be read"
    assert (
        len({m for m in all_images if cast(int, m.info["page_num"]) > 1}) <= 3
    ), "Expected images/equations on all pages beyond 1 to be deduplicated"


def test_equation_parsing() -> None:
    parsed_text = parse_pdf_to_pages(STUB_DATA_DIR / "duplicate_media.pdf")
    assert isinstance(parsed_text.content, dict)
    assert isinstance(parsed_text.content["1"], tuple)
    p1_text, p1_media = parsed_text.content["1"]
    # SEE: https://regex101.com/r/pyOHLq/1
    assert re.search(
        r"[_*]*E[_*]* ?= ?[_*]*mc[_*]*(?:<sup>)?[ ^]?[2²] ?(?:<\/sup>)?", p1_text
    ), "Expected inline equation in page 1 text"
    assert re.search(r"n ?\+ ?a", p1_text), "Expected block equation in page 1 text"
    assert p1_media


def test_clustering() -> None:
    filepath = STUB_DATA_DIR / "pasa.pdf"

    # With very small tolerance, Figure 1 images shouldn't cluster together
    parsed_text_small_cluster = parse_pdf_to_pages(
        filepath, page_range=2, image_cluster_tolerance=1
    )
    assert isinstance(parsed_text_small_cluster.content, dict)
    assert "2" in parsed_text_small_cluster.content, "Parsed text should contain page 2"
    assert isinstance(parsed_text_small_cluster.content["2"], tuple)
    p2_small_cluster = parsed_text_small_cluster.content["2"]
    assert "Figure 1" in p2_small_cluster[0], "Expected Figure 1"
    for m in p2_small_cluster[1]:
        assert "bbox" in m.info
        assert isinstance(m.info["bbox"], Sequence)
        assert len(m.info["bbox"]) == 4, "bbox should have 4 coordinates"
        for coord in m.info["bbox"]:
            assert isinstance(coord, int | float)
            assert coord >= 0, "bbox coordinates should be non-negative"

    # With normal tolerance, images should cluster into one figure
    parsed_text_default_cluster = parse_pdf_to_pages(filepath, page_range=2)
    assert isinstance(parsed_text_default_cluster.content, dict)
    assert (
        "2" in parsed_text_default_cluster.content
    ), "Parsed text should contain page 2"
    assert isinstance(parsed_text_default_cluster.content["2"], tuple)
    p2_default_cluster = parsed_text_default_cluster.content["2"]
    assert "Figure 1" in p2_default_cluster[0], "Expected Figure 1"
    for m in p2_default_cluster[1]:
        assert "bbox" in m.info
        assert isinstance(m.info["bbox"], Sequence)
        assert len(m.info["bbox"]) == 4, "bbox should have 4 coordinates"
        for coord in m.info["bbox"]:
            assert isinstance(coord, int | float)
            assert coord >= 0, "bbox coordinates should be non-negative"

    assert len(p2_small_cluster[1]) >= len(
        p2_default_cluster[1]
    ), "Small tolerance should cluster less aggressively"


class TestMediaMode:

    def test_same_member_is_equal(self) -> None:
        assert MediaMode.INDIVIDUAL == MediaMode.INDIVIDUAL
        assert MediaMode.INDIVIDUAL_CLUSTERING == MediaMode.INDIVIDUAL_CLUSTERING
        assert MediaMode.FULL_PAGE == MediaMode.FULL_PAGE
        assert MediaMode.NONE == MediaMode.NONE

    def test_individual_with_and_without_clustering(self) -> None:
        assert MediaMode.INDIVIDUAL is not MediaMode.INDIVIDUAL_CLUSTERING
        assert MediaMode.INDIVIDUAL != MediaMode.INDIVIDUAL_CLUSTERING
        assert MediaMode.INDIVIDUAL_CLUSTERING != MediaMode.INDIVIDUAL

        assert (
            MediaMode.INDIVIDUAL.metadata_value
            == MediaMode.INDIVIDUAL_CLUSTERING.metadata_value
            == "individual"
        )
        assert (
            str(MediaMode.INDIVIDUAL)  # noqa: FURB123
            == str(MediaMode.INDIVIDUAL_CLUSTERING)  # noqa: FURB123
            == "individual"
        )

    def test_different_members_different_values_are_not_equal(self) -> None:
        assert MediaMode.NONE != MediaMode.FULL_PAGE
        assert MediaMode.FULL_PAGE != MediaMode.INDIVIDUAL
        assert MediaMode.NONE != MediaMode.INDIVIDUAL_CLUSTERING
