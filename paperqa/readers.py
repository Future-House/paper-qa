from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Literal, overload

import html2text
import tiktoken

from .types import ChunkMetadata, Doc, ParsedMetadata, ParsedText, Text
from .version import __version__ as pqa_version


def parse_pdf_fitz_to_pages(path: Path) -> ParsedText:
    import fitz

    with fitz.open(path) as file:
        pages: dict[str, str] = {}
        total_length = 0

        for i in range(file.page_count):
            page = file.load_page(i)
            pages[str(i + 1)] = page.get_text("text", sort=True)
            total_length += len(pages[str(i + 1)])

    metadata = ParsedMetadata(
        parsing_libraries=[f"fitz ({fitz.__doc__})"],
        paperqa_version=str(pqa_version),
        total_parsed_text_length=total_length,
        parse_type="pdf",
    )
    return ParsedText(content=pages, metadata=metadata)


def parse_pdf_to_pages(path: Path) -> ParsedText:
    import pypdf

    with open(path, "rb") as pdfFileObj:
        pdfReader = pypdf.PdfReader(pdfFileObj)
        pages: dict[str, str] = {}
        total_length = 0

        for i, page in enumerate(pdfReader.pages):
            pages[str(i + 1)] = page.extract_text()
            total_length += len(pages[str(i + 1)])

    metadata = ParsedMetadata(
        parsing_libraries=[f"pypdf ({pypdf.__version__})"],
        paperqa_version=str(pqa_version),
        total_parsed_text_length=total_length,
        parse_type="pdf",
    )
    return ParsedText(content=pages, metadata=metadata)


def chunk_pdf(
    parsed_text: ParsedText, doc: Doc, chunk_chars: int, overlap: int
) -> list[Text]:
    pages: list[str] = []
    texts: list[Text] = []
    split: str = ""

    if not isinstance(parsed_text.content, dict):
        raise NotImplementedError(
            f"ParsedText.content must be a `dict`, not {type(parsed_text.content)}."
        )

    for page_num, page_text in parsed_text.content.items():
        split += page_text
        pages.append(page_num)
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(
                Text(
                    text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [page_num]

    if len(split) > overlap or len(texts) == 0:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    return texts


def parse_text(
    path: Path, html: bool = False, split_lines=False, use_tiktoken=True
) -> ParsedText:
    """Simple text splitter, can optionally use tiktoken, parse html, or split into newlines.

    Args:
        path: path to file
        html: flag to use html2text library for parsing
        split_lines: flag to split lines into a list
        use_tiktoken: flag to use tiktoken library to encode text

    """
    try:
        with open(path) as f:
            text = [str(line) for line in f] if split_lines else f.read()
    except UnicodeDecodeError:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()

    if html:
        text = html2text.html2text(text)

    metadata = {
        "parsing_libraries": ["tiktoken (cl100k_base)"] if use_tiktoken else [],
        "paperqa_version": str(pqa_version),
        "total_parsed_text_length": (
            len(text) if isinstance(text, str) else sum([len(t) for t in text])
        ),
        "parse_type": "txt" if not html else "html",
    }
    if html:
        metadata["parsing_libraries"].append(f"html2text ({html2text.__version__})")  # type: ignore[attr-defined]

    return ParsedText(content=text, metadata=ParsedMetadata(**metadata))


def chunk_text(
    parsed_text: ParsedText, doc: Doc, chunk_chars: int, overlap: int, use_tiktoken=True
) -> list[Text]:
    """Parse a document into chunks, based on tiktoken encoding.

    NOTE: We get some byte continuation errors.
    Currently ignored, but should explore more to make sure we
    don't miss anything.
    """
    texts: list[Text] = []
    enc = tiktoken.get_encoding("cl100k_base")
    split = []

    if not isinstance(parsed_text.content, str):
        raise NotImplementedError(
            f"ParsedText.content must be a `str`, not {type(parsed_text.content)}."
        )

    content = parsed_text.content if not use_tiktoken else parsed_text.encode_content()

    # convert from characters to chunks
    char_count = parsed_text.metadata.total_parsed_text_length  # e.g., 25,000
    token_count = len(content)  # e.g., 4,500
    chars_per_token = char_count / token_count  # e.g., 5.5
    chunk_tokens = chunk_chars / chars_per_token  # e.g., 3000 / 5.5 = 545
    overlap_tokens = overlap / chars_per_token  # e.g., 100 / 5.5 = 18
    chunk_count = ceil(token_count / chunk_tokens)  # e.g., 4500 / 545 = 9

    for i in range(chunk_count):
        split = content[
            max(int(i * chunk_tokens - overlap_tokens), 0) : int(
                (i + 1) * chunk_tokens + overlap_tokens
            )
        ]
        texts.append(
            Text(
                text=enc.decode(split) if use_tiktoken else split,
                name=f"{doc.docname} chunk {i + 1}",
                doc=doc,
            )
        )
    return texts


def chunk_code_text(
    parsed_text: ParsedText, doc: Doc, chunk_chars: int, overlap: int
) -> list[Text]:
    """Parse a document into chunks, based on line numbers (for code)."""
    split = ""
    texts: list[Text] = []
    last_line = 0

    if not isinstance(parsed_text.content, list):
        raise NotImplementedError(
            f"ParsedText.content must be a `list`, not {type(parsed_text.content)}."
        )

    for i, line in enumerate(parsed_text.content):
        split += line
        while len(split) > chunk_chars:
            texts.append(
                Text(
                    text=split[:chunk_chars],
                    name=f"{doc.docname} lines {last_line}-{i}",
                    doc=doc,
                )
            )
            split = split[chunk_chars - overlap :]
            last_line = i
    if len(split) > overlap or len(texts) == 0:
        texts.append(
            Text(
                text=split[:chunk_chars],
                name=f"{doc.docname} lines {last_line}-{i}",
                doc=doc,
            )
        )
    return texts


@overload
def read_doc(
    path: Path,
    doc: Doc,
    parsed_text_only: Literal[False],
    include_metadata: Literal[False],
    chunk_chars: int = ...,
    overlap: int = ...,
    force_pypdf: bool = ...,
) -> list[Text]: ...


@overload
def read_doc(
    path: Path,
    doc: Doc,
    parsed_text_only: Literal[False] = ...,
    include_metadata: Literal[False] = ...,
    chunk_chars: int = ...,
    overlap: int = ...,
    force_pypdf: bool = ...,
) -> list[Text]: ...


@overload
def read_doc(
    path: Path,
    doc: Doc,
    parsed_text_only: Literal[True],
    include_metadata: bool = ...,
    chunk_chars: int = ...,
    overlap: int = ...,
    force_pypdf: bool = ...,
) -> ParsedText: ...


@overload
def read_doc(
    path: Path,
    doc: Doc,
    parsed_text_only: Literal[False],
    include_metadata: Literal[True],
    chunk_chars: int = ...,
    overlap: int = ...,
    force_pypdf: bool = ...,
) -> tuple[list[Text], ParsedMetadata]: ...


def read_doc(  # noqa: PLR0912
    path: Path,
    doc: Doc,
    parsed_text_only: bool = False,
    include_metadata: bool = False,
    chunk_chars: int = 3000,
    overlap: int = 100,
    force_pypdf: bool = False,
) -> list[Text] | ParsedText | tuple[list[Text], ParsedMetadata]:
    """Parse a document and split into chunks.

    Optionally can include just the parsing as well as metadata about the parsing/chunking

    Args:
        path: local document path
        doc: object with document metadata
        chunk_chars: size of chunks
        overlap: size of overlap between chunks
        force_pypdf: flag to force use of pypdf in parsing
        parsed_text_only: return parsed text without chunking
        include_metadata: return a tuple
    """
    str_path = str(path)
    parsed_text = None

    # start with parsing -- users may want to store this separately
    if str_path.endswith(".pdf"):
        if force_pypdf:
            parsed_text = parse_pdf_to_pages(path)
        else:
            try:
                parsed_text = parse_pdf_fitz_to_pages(path)
            except ImportError:
                parsed_text = parse_pdf_to_pages(path)

    elif str_path.endswith(".txt"):
        parsed_text = parse_text(path, html=False, split_lines=False, use_tiktoken=True)
    elif str_path.endswith(".html"):
        parsed_text = parse_text(path, html=True, split_lines=False, use_tiktoken=True)
    else:
        parsed_text = parse_text(path, html=False, split_lines=True, use_tiktoken=False)

    if parsed_text_only:
        return parsed_text

    # next chunk the parsed text
    if str_path.endswith(".pdf"):
        chunked_text = chunk_pdf(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars, overlap=overlap, chunk_type="overlap_pdf_by_page"
        )
    elif str_path.endswith((".txt", ".html")):
        chunked_text = chunk_text(
            parsed_text,
            doc,
            chunk_chars=chunk_chars,
            overlap=overlap,
            use_tiktoken=True,
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars, overlap=overlap, chunk_type="overlap"
        )
    else:
        chunked_text = chunk_code_text(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars, overlap=overlap, chunk_type="overlap_code_by_line"
        )

    if include_metadata:
        parsed_text.metadata.chunk_metadata = chunk_metadata
        return chunked_text, parsed_text.metadata

    return chunked_text
