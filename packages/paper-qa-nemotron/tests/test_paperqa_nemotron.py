import base64
import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from lmi.utils import bytes_to_string
from paperqa import Doc, Docs, Settings
from paperqa.readers import PDFParserFn, chunk_pdf
from paperqa.utils import ImpossibleParsingError, get_citation_ids
from PIL import Image

from paperqa_nemotron import parse_pdf_to_pages
from paperqa_nemotron.reader import pad_image_with_border

REPO_ROOT = Path(__file__).parents[3]
STUB_DATA_DIR = REPO_ROOT / "tests" / "stub_data"


@pytest.mark.flaky(reruns=2, only_rerun=["RuntimeError", "AssertionError"])
@pytest.mark.parametrize(
    "api_params_base",
    [
        pytest.param({}, id="nvidia"),
        # Uncomment to test with AWS SageMaker
        # pytest.param({"model_name": "sagemaker/nvidia/nemotron-parse"}, id="sagemaker"),
    ],
)
@pytest.mark.asyncio
async def test_parse_pdf_to_pages(api_params_base: dict[str, Any]) -> None:
    assert isinstance(parse_pdf_to_pages, PDFParserFn)
    filepath = STUB_DATA_DIR / "pasa.pdf"
    # Lower temperature a bit so shape assertions are more reliable,
    # and use null DPI here as later 'high DPI' assertions compare
    # against the 'default' DPI from pypdfium2
    parsed_text = await parse_pdf_to_pages(
        filepath, dpi=None, api_params={"temperature": 0.5} | api_params_base
    )
    assert isinstance(parsed_text.content, dict)
    assert len(parsed_text.content) == 15, "Expected all pages to be parsed"
    assert "1" in parsed_text.content, "Parsed text should contain page 1"
    assert isinstance(parsed_text.content["1"], tuple)
    p1_text = parsed_text.content["1"][0]
    # Don't match Abstract as sometimes nemotron-parse places authors or organizations
    # between Abstract and Introduction
    matches = re.findall(
        r"(?:###? 1 Introduction[\n]+)?We introduce Pa ?S[as],"
        r" an advanced Paper Search agent powered by large language models\.",
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
    assert p2_image.info["height"] == pytest.approx(130, rel=0.25)
    assert p2_image.info["width"] == pytest.approx(452, rel=0.25)
    p2_bbox = p2_image.info["bbox"]
    assert isinstance(p2_bbox, tuple)
    for i, value in enumerate((71, 71.40, 522, 213.00)):
        assert p2_bbox[i] == pytest.approx(value, rel=0.275 if value < 100 else 0.15)
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
    fig_1_text = texts[2]
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

    # Let's check the full page parsing behavior
    parsed_text_full_page = await parse_pdf_to_pages(
        filepath, dpi=None, full_page=True, api_params=api_params_base
    )
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
        assert full_page_image.info["height"] == pytest.approx(842, rel=0.01)
        assert full_page_image.info["width"] == pytest.approx(596, rel=0.01)
        assert isinstance(full_page_image.data, bytes)
        assert full_page_image.data, "Full page image should have data"
        # Check useful attributes are present and are JSON serializable
        json.dumps(p2_image.info)
        for attr in ("width", "height"):
            dim = full_page_image.info[attr]
            assert isinstance(dim, int | float)
            assert dim > 0, "Edge length should be positive"

    # Check our ability to get a high DPI, and lower temperature a bit
    # so shape assertions are more reliable
    parsed_text_high_dpi = await parse_pdf_to_pages(
        filepath, dpi=216, api_params={"temperature": 0.5} | api_params_base
    )
    assert isinstance(parsed_text_high_dpi.content, dict)
    assert len(parsed_text_high_dpi.content) == 15, "Expected all pages to be parsed"
    assert not isinstance(parsed_text_high_dpi.content["2"], str)
    p2_text_high_dpi, p2_media_high_dpi = parsed_text_high_dpi.content["2"]
    assert "Figure 1" in p2_text_high_dpi, "Expected Figure 1 title"
    assert "Crawler" in p2_text_high_dpi, "Expected Figure 1 contents"
    (p2_image_high_dpi,) = [m for m in p2_media_high_dpi if m.info["type"] == "picture"]
    assert p2_image_high_dpi.info["scale"] == 3
    assert p2_image_high_dpi.info["height"] / p2_image.info["height"] == pytest.approx(  # type: ignore[operator]
        3, rel=0.2
    )
    assert p2_image_high_dpi.info["width"] / p2_image.info["width"] == pytest.approx(  # type: ignore[operator]
        3, rel=0.2
    )

    # Check the no-media behavior
    parsed_text_no_media = await parse_pdf_to_pages(
        filepath, parse_media=False, api_params=api_params_base
    )
    assert isinstance(parsed_text_no_media.content, dict)
    assert all(isinstance(c, str) for c in parsed_text_no_media.content.values())
    assert len(parsed_text_no_media.content) == 15, "Expected all pages to be parsed"

    # Check metadata
    for pt in (
        parsed_text,
        parsed_text_full_page,
        parsed_text_high_dpi,
        parsed_text_no_media,
    ):
        assert "pypdfium2" in pt.metadata.parsing_libraries[0]
        assert "nemotron-parse" in pt.metadata.parsing_libraries[1]
        assert pt.metadata.name
        assert "pdf" in pt.metadata.name
        assert "page_range=None" in pt.metadata.name

    # Check commonalities across all modes
    assert (
        len(parsed_text.content)
        == len(parsed_text_full_page.content)
        == len(parsed_text_high_dpi.content)
        == len(parsed_text_no_media.content)
    ), "All modes should parse the same number of pages"


@pytest.mark.asyncio
async def test_page_range() -> None:
    filepath = STUB_DATA_DIR / "pasa.pdf"

    parsed_text_p1 = await parse_pdf_to_pages(filepath, page_range=1)
    assert isinstance(parsed_text_p1.content, dict)
    assert list(parsed_text_p1.content) == ["1"]
    assert parsed_text_p1.metadata.name
    assert "page_range=1" in parsed_text_p1.metadata.name

    parsed_text_p1_2 = await parse_pdf_to_pages(filepath, page_range=(1, 2))
    assert isinstance(parsed_text_p1_2.content, dict)
    assert list(parsed_text_p1_2.content) == ["1", "2"]
    assert parsed_text_p1_2.metadata.name
    assert "page_range=(1,2)" in parsed_text_p1_2.metadata.name

    # NOTE: exceeds 15-page PDF length
    with pytest.raises(ValueError, match="value 15 is outside the size"):
        await parse_pdf_to_pages(filepath, page_range=(1, 20))


@pytest.mark.skip(reason="Nemotron Parse cannot handle duplicate_media.pdf reliably")
@pytest.mark.asyncio
async def test_media_deduplication() -> None:
    parsed_text = await parse_pdf_to_pages(
        STUB_DATA_DIR / "duplicate_media.pdf", api_params={"temperature": 0}
    )
    assert isinstance(parsed_text.content, dict)
    assert len(parsed_text.content) == 5, "Expected full PDF read"
    all_media = [m for _, media in parsed_text.content.values() for m in media]  # type: ignore[misc]

    all_images = [m for m in all_media if m.info.get("type") == "picture"]
    assert len(all_images) == 5, "Expected each image (one/page) to be read"
    assert (
        len({m for m in all_images if cast(int, m.info["page_num"]) > 1}) <= 2
    ), "Expected images on all pages beyond 1 to be deduplicated"

    all_equations = [m for m in all_media if m.info.get("type") == "formula"]
    assert len(all_equations) == 5, "Expected each equation (one/page) to be read"
    assert (
        len({m for m in all_equations if cast(int, m.info["page_num"]) > 1}) <= 2
    ), "Expected equations on all pages beyond 1 to be deduplicated"

    all_tables = [m for m in all_media if m.info.get("type") == "table"]
    assert len(all_tables) == 5, "Expected each table (one/page) to be read"
    assert (
        len({m for m in all_tables if cast(int, m.info["page_num"]) > 1}) <= 2
    ), "Expected tables on all pages beyond 1 to be deduplicated"


@pytest.mark.asyncio
async def test_page_size_limit_denial() -> None:
    with pytest.raises(ImpossibleParsingError, match="char limit"):
        await parse_pdf_to_pages(
            STUB_DATA_DIR / "paper.pdf", page_size_limit=10  # chars
        )


@pytest.mark.asyncio
async def test_invalid_pdf_is_denied(tmp_path) -> None:
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
        await parse_pdf_to_pages(bad_pdf_path)


@pytest.mark.asyncio
async def test_nonexistent_file_failure() -> None:
    filename = "/nonexistent/path/file.pdf"
    with pytest.raises(FileNotFoundError, match=filename):
        await parse_pdf_to_pages(filename)


@pytest.mark.asyncio
async def test_table_parsing() -> None:
    parsed_text = await parse_pdf_to_pages(STUB_DATA_DIR / "influence.pdf")
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


@pytest.mark.asyncio
async def test_equation_parsing() -> None:
    parsed_text = await parse_pdf_to_pages(STUB_DATA_DIR / "duplicate_media.pdf")
    assert isinstance(parsed_text.content, dict)
    assert isinstance(parsed_text.content["1"], tuple)
    p1_text, p1_media = parsed_text.content["1"]
    # SEE: https://regex101.com/r/pyOHLq/1
    assert re.search(
        r"[_*]*E[_*]* ?= ?[_*]*mc[_*]*(?:<sup>)?[ ^]?[2²] ?(?:<\/sup>)?", p1_text
    ), "Expected inline equation in page 1 text"
    assert re.search(r"n ?\+ ?a", p1_text), "Expected block equation in page 1 text"
    assert p1_media


def test_pad_image_with_border(subtests: pytest.Subtests) -> None:
    stub_gray_image = Image.new("RGB", (1000, 1500), (128, 128, 128))

    with subtests.test(msg="default-size"):
        padded, offset_x, offset_y = pad_image_with_border(stub_gray_image)
        assert padded.width == stub_gray_image.width + 60 * 2
        assert padded.height == stub_gray_image.height + 60 * 2
        assert offset_x == offset_y == 60

    with subtests.test(msg="custom-size"):
        padded, offset_x, offset_y = pad_image_with_border(stub_gray_image, border=30)
        assert padded.width == stub_gray_image.width + 60
        assert padded.height == stub_gray_image.height + 60
        assert offset_x == offset_y == 30

    with subtests.test(msg="tuple-size"):
        padded, offset_x, offset_y = pad_image_with_border(
            stub_gray_image, border=(20, 40)
        )
        assert padded.width == stub_gray_image.width + 40
        assert padded.height == stub_gray_image.height + 80
        assert offset_x == 20
        assert offset_y == 40

    with subtests.test(msg="rgba-mode"):
        rgba_image = Image.new("RGBA", (500, 800), (100, 100, 100, 255))
        padded, offset_x, offset_y = pad_image_with_border(rgba_image)
        assert padded.mode == "RGBA"
        assert padded.width == rgba_image.width + 60 * 2
        assert padded.height == rgba_image.height + 60 * 2

    with subtests.test(msg="grayscale-mode"):
        grayscale_image = Image.new("L", (600, 900), 128)
        padded, offset_x, offset_y = pad_image_with_border(
            grayscale_image, pad_color=255
        )
        assert padded.mode == "L"
        assert padded.width == grayscale_image.width + 60 * 2
        assert padded.height == grayscale_image.height + 60 * 2


@pytest.mark.asyncio
async def test_media_enrichment_filters_irrelevant() -> None:
    parsed_text = await parse_pdf_to_pages(
        STUB_DATA_DIR / "duplicate_media.pdf", api_params={"temperature": 0}
    )
    assert isinstance(parsed_text.content, dict)

    # Get media before enrichment to track their types
    media_before = [
        m
        for page_contents in parsed_text.content.values()
        if isinstance(page_contents, tuple)
        for m in page_contents[1]
    ]
    assert media_before, "Expected some media to be parsed from the PDF"

    # Create settings and run enrichment
    enricher = Settings().make_media_enricher()
    enrichment_summary = await enricher(parsed_text)
    assert "filtered=" in enrichment_summary, "Test extractions require this substring"

    # Game the in-place change of enrichment to get Wikimedia logos,
    # so we can later confirm they are filtered out
    wikimedia_media_before = [
        m
        for m in media_before
        if isinstance(m.info["enriched_description"], str)
        and "wikimedia" in m.info["enriched_description"].lower()
    ]
    assert (
        len(wikimedia_media_before) > 1
    ), "Test expects several Wikimedia logos to be parsed"

    # Get media after enrichment to track filtration
    media_after = [
        m
        for page_contents in parsed_text.content.values()
        if isinstance(page_contents, tuple)
        for m in page_contents[1]
    ]
    assert media_after, "Expected some media to remain after enrichment's filtration"
    for media in media_after:
        assert not media.info[
            "is_irrelevant"
        ], "Expected remaining media to be marked as relevant"
        assert media.info["enriched_description"], "Expected enriched description"
    filtered_count = int(
        enrichment_summary.split("filtered=", maxsplit=1)[1].split("|")[0]
    )
    assert (
        len(media_before) - len(media_after) == filtered_count
    ), "Filtered summary mismatches actual filtration"
    assert filtered_count > 0, "Expected some filtration to take place"

    wikimedia_media_after = [
        m
        for m in media_after
        if isinstance(m.info["enriched_description"], str)
        and "wikimedia" in m.info["enriched_description"].lower()
    ]
    assert len(wikimedia_media_after) <= 1, "Expected most Wikimedia logos to be gone"
