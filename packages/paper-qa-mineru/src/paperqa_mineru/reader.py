"""Read PDFs from MinerU output directories without re-running MinerU."""

import json
from importlib.metadata import version
from pathlib import Path

from paperqa.readers import resolve_page_range
from paperqa.types import ParsedMedia, ParsedText

# Running headers, footers, and page numbers stay out of retrieval context
TEXT_BLOCK_TYPES = {"text", "aside_text", "page_footnote", "ref_text"}
MEDIA_BLOCK_TYPES = {"image", "chart", "table"}
# Caption and footnote fields attached to figures and tables
CAPTION_FIELDS = (
    "image_caption",
    "image_footnote",
    "chart_caption",
    "chart_footnote",
    "table_caption",
    "table_footnote",
)


def _block_text(block: dict) -> str:
    parts = (
        [block.get("table_body", "")] if block["type"] == "table" else [block.get("text", "")]
    )
    for field in CAPTION_FIELDS:
        parts.extend(block.get(field) or [])
    return "\n".join(part for part in parts if part)


def parse_pdf_to_pages(
    path: str | Path,
    page_size_limit: int | None = None,
    page_range: int | tuple[int, int] | None = None,
    parse_media: bool = True,
    **_,
) -> ParsedText:
    """Parse a PDF by reading the sibling MinerU directory `<stem>/content_list.json`.

    The MinerU directory is expected next to the PDF under the same stem, so
    `papers/G7ZCDWQ4.pdf` maps to `papers/G7ZCDWQ4/content_list.json`.
    `page_size_limit` is accepted to satisfy `PDFParserFn` and is not applied,
    because MinerU already splits content by page. `page_range` keeps only the
    selected 1-based pages. `parse_media=False` skips media files entirely.
    """
    del page_size_limit
    mineru_dir = Path(path).with_suffix("")
    blocks = json.loads((mineru_dir / "content_list.json").read_text(encoding="utf-8"))
    page_count = max(block["page_idx"] for block in blocks) + 1
    selected = resolve_page_range(page_range, page_count=page_count)

    pages: dict[str, tuple[str, list[ParsedMedia]]] = {}
    for block in blocks:
        page_idx = block["page_idx"]
        if page_idx not in selected:
            continue
        page = str(page_idx + 1)
        text, media = pages.get(page, ("", []))
        block_type = block["type"]
        if block_type in TEXT_BLOCK_TYPES or block_type in MEDIA_BLOCK_TYPES:
            text = f"{text}\n{_block_text(block)}".strip()
        if block_type in MEDIA_BLOCK_TYPES and parse_media:
            img_path = block["img_path"]
            media.append(
                ParsedMedia(
                    index=len(media),
                    data=(mineru_dir / img_path).read_bytes(),
                    text=block.get("table_body") if block_type == "table" else None,
                    info={
                        "mineru_type": block_type,
                        "bbox": block.get("bbox", ""),
                        "suffix": Path(img_path).suffix,
                    },
                )
            )
        pages[page] = (text, media)

    return ParsedText(
        content=pages,
        metadata={
            "parsing_libraries": ["mineru (content_list.json)"],
            "paperqa_version": version("paper-qa"),
            "total_parsed_text_length": sum(len(t) for t, _ in pages.values()),
            "count_parsed_media": sum(len(m) for _, m in pages.values()),
            "name": f"mineru|{mineru_dir}",
            "chunk_metadata": None,
        },
    )
