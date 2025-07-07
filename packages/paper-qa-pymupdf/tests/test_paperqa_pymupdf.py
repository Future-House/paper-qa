from pathlib import Path

from paperqa_pymupdf import parse_pdf_to_pages

REPO_ROOT = Path(__file__).parents[3]
STUB_DATA_DIR = REPO_ROOT / "tests" / "stub_data"


def test_parse_pdf_to_pages() -> None:
    filepath = STUB_DATA_DIR / "pasa.pdf"
    parsed_text = parse_pdf_to_pages(filepath, use_block_parsing=True)
    assert isinstance(parsed_text.content, dict)
    assert "1" in parsed_text.content, "Parsed text should contain page 1"
    assert (
        "Abstract\n\nWe introduce PaSa, an advanced Paper Search"
        "\nagent powered by large language models."
    ) in parsed_text.content["1"]
