"""Read PDFs from MinerU output directories without re-running MinerU."""

import json
from pathlib import Path

from paperqa.types import ParsedMedia, ParsedText

# Page header/footer/page-number blocks are excluded from retrieval context
TEXT_BLOCK_TYPES = {"text", "header", "aside_text", "page_footnote", "ref_text"}
MEDIA_BLOCK_TYPES = {"image", "chart", "table"}


def _block_text(block: dict) -> str:
    if block["type"] == "table":
        return block.get("table_body", "")
    return block.get("text", "")


def parse_pdf_to_pages(path: str | Path, page_size_limit: int | None = None, **_) -> ParsedText:
    """Parse a PDF by reading the sibling MinerU directory `<stem>/content_list.json`.

    The MinerU directory is expected next to the PDF under the same stem, so
    `papers/G7ZCDWQ4.pdf` maps to `papers/G7ZCDWQ4/content_list.json`.
    `page_size_limit` is accepted to satisfy `PDFParserFn` and is not applied,
    because MinerU already splits content by page.
    """
    del page_size_limit
    pdf_path = Path(path)
    mineru_dir = pdf_path.with_suffix("")
    content_list = mineru_dir / "content_list.json"
    blocks = json.loads(content_list.read_text(encoding="utf-8"))

    pages: dict[str, tuple[str, list[ParsedMedia]]] = {}
    for block in blocks:
        page = str(int(block["page_idx"]) + 1)
        text, media = pages.get(page, ("", []))
        block_type = block["type"]
        if block_type in TEXT_BLOCK_TYPES or block_type == "table":
            text = f"{text}\n{_block_text(block)}".strip()
        if block_type in MEDIA_BLOCK_TYPES:
            media.append(
                ParsedMedia(
                    index=len(media),
                    data=(mineru_dir / block["img_path"]).read_bytes(),
                    text=block.get("table_body") if block_type == "table" else None,
                    info={"mineru_type": block_type, "bbox": block.get("bbox", "")},
                )
            )
        pages[page] = (text, media)

    return ParsedText(
        content=pages,
        metadata={
            "parsing_libraries": ["mineru (content_list.json)"],
            "paperqa_version": "paper-qa-mineru",
            "total_parsed_text_length": sum(len(t) for t, _ in pages.values()),
            "count_parsed_media": sum(len(m) for _, m in pages.values()),
            "name": f"mineru|{mineru_dir}",
            "chunk_metadata": None,
        },
    )
