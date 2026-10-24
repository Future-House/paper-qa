import json
from pathlib import Path

import pytest
from paperqa.readers import PDFParserFn

from paperqa_mineru import parse_pdf_to_pages

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-bytes"


def _write_mineru_dir(root: Path, blocks: list[dict]) -> Path:
    """Create `<root>/paper.pdf` (placeholder) and the sibling MinerU directory."""
    pdf_path = root / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-placeholder")
    mineru_dir = root / "paper"
    (mineru_dir / "images").mkdir(parents=True)
    (mineru_dir / "images" / "fig1.png").write_bytes(PNG_BYTES)
    (mineru_dir / "content_list.json").write_text(json.dumps(blocks), encoding="utf-8")
    return pdf_path


def test_is_pdf_parser_fn() -> None:
    assert isinstance(parse_pdf_to_pages, PDFParserFn)


def test_pages_are_one_based_and_drop_running_headers(tmp_path: Path) -> None:
    pdf_path = _write_mineru_dir(
        tmp_path,
        [
            {"type": "header", "text": "Articles", "page_idx": 0},
            {"type": "text", "text": "Body on page one.", "page_idx": 0},
            {"type": "footer", "text": "www.journal.example", "page_idx": 0},
            {"type": "page_number", "text": "1", "page_idx": 0},
            {"type": "text", "text": "Body on page two.", "page_idx": 1},
        ],
    )

    parsed = parse_pdf_to_pages(pdf_path)

    assert set(parsed.content) == {"1", "2"}
    page1_text, page1_media = parsed.content["1"]
    assert page1_text == "Articles\nBody on page one."
    assert page1_media == []
    assert parsed.content["2"][0] == "Body on page two."


def test_table_body_and_image_media(tmp_path: Path) -> None:
    pdf_path = _write_mineru_dir(
        tmp_path,
        [
            {
                "type": "table",
                "table_body": "<table><tr><td>EL1</td><td>100-105</td></tr></table>",
                "img_path": "images/fig1.png",
                "page_idx": 0,
                "bbox": [1, 2, 3, 4],
            },
            {"type": "image", "img_path": "images/fig1.png", "page_idx": 1},
        ],
    )

    parsed = parse_pdf_to_pages(pdf_path)

    table_text, table_media = parsed.content["1"]
    assert "EL1" in table_text
    (table_item,) = table_media
    assert table_item.index == 0
    assert table_item.data == PNG_BYTES
    assert table_item.text == "<table><tr><td>EL1</td><td>100-105</td></tr></table>"
    assert table_item.info == {"mineru_type": "table", "bbox": [1, 2, 3, 4]}

    image_text, image_media = parsed.content["2"]
    assert not image_text
    (image_item,) = image_media
    assert image_item.text is None
    assert image_item.info["mineru_type"] == "image"


def test_missing_content_list_raises(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-placeholder")

    with pytest.raises(FileNotFoundError):
        parse_pdf_to_pages(pdf_path)


def test_missing_media_file_raises(tmp_path: Path) -> None:
    pdf_path = _write_mineru_dir(
        tmp_path,
        [{"type": "image", "img_path": "images/absent.png", "page_idx": 0}],
    )

    with pytest.raises(FileNotFoundError):
        parse_pdf_to_pages(pdf_path)
