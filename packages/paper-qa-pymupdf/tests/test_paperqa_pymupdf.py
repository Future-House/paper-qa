import base64
import json
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from lmi.utils import bytes_to_string
from paperqa import Doc, Docs, Settings
from paperqa.readers import PDFParserFn, chunk_pdf
from paperqa.utils import ImpossibleParsingError

from paperqa_pymupdf import parse_pdf_to_pages

REPO_ROOT = Path(__file__).parents[3]
STUB_DATA_DIR = REPO_ROOT / "tests" / "stub_data"


@pytest.mark.asyncio
async def test_parse_pdf_to_pages() -> None:
    assert isinstance(parse_pdf_to_pages, PDFParserFn)

    filepath = STUB_DATA_DIR / "pasa.pdf"
    parsed_text = parse_pdf_to_pages(filepath, use_block_parsing=True)
    assert isinstance(parsed_text.content, dict)
    assert len(parsed_text.content) == 15, "Expected all pages to be parsed"
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
    # pylint: disable=duplicate-code
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

    # Let's check the full page parsing behavior
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
        assert isinstance(full_page_image.data, bytes)
        assert full_page_image.data, "Full page image should have data"
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
    assert len(parsed_text_no_media.content) == 15, "Expected all pages to be parsed"

    # Check metadata
    for pt in (parsed_text, parsed_text_full_page, parsed_text_no_media):
        (parsing_library,) = pt.metadata.parsing_libraries
        assert pymupdf.__name__ in parsing_library
        assert pt.metadata.name
        assert "pdf" in pt.metadata.name

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

    parsed_text_p1_2 = parse_pdf_to_pages(filepath, page_range=(1, 2))
    assert isinstance(parsed_text_p1_2.content, dict)
    assert list(parsed_text_p1_2.content) == ["1", "2"]
    assert parsed_text_p1_2.metadata.name
    assert "page_range=(1,2)" in parsed_text_p1_2.metadata.name

    # NOTE: exceeds 15-page PDF length
    with pytest.raises(ValueError, match="page not in document"):
        parse_pdf_to_pages(filepath, page_range=(1, 20))


def test_page_size_limit_denial() -> None:
    with pytest.raises(ImpossibleParsingError, match="char limit"):
        parse_pdf_to_pages(STUB_DATA_DIR / "paper.pdf", page_size_limit=10)  # chars


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

    docs = Docs()
    with pytest.raises(ValueError, match="does not look"):
        await docs.aadd(
            bad_pdf_path,
            citation="Citation 1",  # Skip citation inference
            title="Title",  # Skip title inference
            settings=Settings(
                parsing={"parse_pdf": parse_pdf_to_pages, "use_doc_details": False}
            ),
        )


def test_table_parsing() -> None:
    spy_to_markdown = MagicMock(side_effect=pymupdf.table.Table.to_markdown)
    zeroth_raw_table_text = ""

    def custom_to_markdown(self, clean=False, fill_empty=True) -> str:
        md = spy_to_markdown(self, clean=clean, fill_empty=fill_empty)
        if spy_to_markdown.call_count == 1:
            nonlocal zeroth_raw_table_text
            zeroth_raw_table_text = md
            return (  # NOTE: this text has a null byte, which we want to filter
                "|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|"
                "\n|---|---|---|---|---|---|---|---|"
                "\n||\x02\x03<br>|\x04\x05\x06\x07\x08<br>"
                " <br>|\x07\x08\x08<br>\n\x08<br>\x0e\x0f<br>\x17\x18\x18\x08<br>|\x02<br>\x0c\x10<br>\x11<br>\x19\r\x02\x1a\x00\x01\x02\x03<br>|\x11<br>\x12\x06\x05<br>\x0e\x13\x14\x15<br>\x04\x05\x06\x07<br>|\x05\x08<br>\x0c\x10<br>\x12\x06\x05<br>\x0e\x16\x13<br>|\x05\x08<br>\x0c\x10<br>\x12\x06\x05<br>\x0e\x16\x13<br>|"  # noqa: E501
            )
        return md

    filepath = STUB_DATA_DIR / "influence.pdf"
    with patch.object(pymupdf.table.Table, "to_markdown", custom_to_markdown):
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
    zeroth_media, *_ = next(iter(all_tables.values()))
    assert zeroth_media.text is None, "Expected null byte to be filtered"
    assert zeroth_raw_table_text == (
        "|Gap Size (mm)|Ununited|Uncertain|United|"
        "\n|---|---|---|---|"
        "\n|**1.0**|1/5 (20%)|1/5 (20%)|3/5 (60%)|"
        "\n|**1.5**|3/7  (43%)|2/7  (29%)|2/7 (29%)|"
        "\n|**2.0** <br>|3/6 (50%)|2/6 (33%)|1/6 (17%)|"
        "\n\n"  # NOTE: this is before strip, so there can be trailing whitespace
    )
