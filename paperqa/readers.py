from __future__ import annotations

import asyncio
import os
from math import ceil
from pathlib import Path
from typing import Literal, Protocol, cast, overload, runtime_checkable

import pymupdf
import tiktoken
from html2text import __version__ as html2text_version
from html2text import html2text

from paperqa.types import (
    ChunkMetadata,
    Doc,
    ParsedMetadata,
    ParsedText,
    Text,
)
from paperqa.utils import ImpossibleParsingError
from paperqa.version import __version__ as pqa_version


def setup_pymupdf_python_logging() -> None:
    """
    Configure PyMuPDF to use Python logging.

    SEE: https://pymupdf.readthedocs.io/en/latest/app3.html#diagnostics
    """
    pymupdf.set_messages(pylogging=True)


@runtime_checkable
class PDFParserFn(Protocol):
    """Protocol for parsing a PDF."""

    def __call__(
        self, path: str | os.PathLike, page_size_limit: int | None = None, **kwargs
    ) -> ParsedText: ...


BLOCK_TEXT_INDEX = 4


def parse_pdf_to_pages(
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    use_block_parsing: bool = False,
    **_,
) -> ParsedText:

    with pymupdf.open(path) as file:
        pages: dict[str, str] = {}
        total_length = 0

        for i in range(file.page_count):
            try:
                page = file.load_page(i)
            except pymupdf.mupdf.FzErrorFormat as exc:
                raise ImpossibleParsingError(
                    f"Page loading via {pymupdf.__name__} failed on page {i} of"
                    f" {file.page_count} for the PDF at path {path}, likely this PDF"
                    " file is corrupt."
                ) from exc

            if use_block_parsing:
                # NOTE: this block-based parsing appears to be better, but until
                # fully validated on 1+ benchmarks, it's considered experimental

                # Extract text blocks from the page
                # Note: sort=False is important to preserve the order of text blocks
                # as they appear in the PDF
                blocks = page.get_text("blocks", sort=False)

                # Concatenate text blocks into a single string
                text = "\n".join(
                    block[BLOCK_TEXT_INDEX]
                    for block in blocks
                    if len(block) > BLOCK_TEXT_INDEX
                )
            else:
                text = page.get_text("text", sort=True)

            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The text in page {i} of {file.page_count} was {len(text)} chars"
                    f" long, which exceeds the {page_size_limit} char limit for the PDF"
                    f" at path {path}."
                )
            pages[str(i + 1)] = text
            total_length += len(text)

    metadata = ParsedMetadata(
        parsing_libraries=[f"pymupdf ({pymupdf.__version__})"],
        paperqa_version=pqa_version,
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

    if not parsed_text.content:
        raise ImpossibleParsingError(
            f"No text was parsed from the document named {doc.docname!r} with ID"
            f" {doc.dockey}, either empty or corrupted."
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

    if len(split) > overlap or not texts:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    return texts


def parse_text(
    path: str | os.PathLike,
    html: bool = False,
    split_lines: bool = False,
    use_tiktoken: bool = True,
    page_size_limit: int | None = None,
    **_,
) -> ParsedText:
    """Simple text splitter, can optionally use tiktoken, parse html, or split into newlines.

    Args:
        path: path to file.
        html: flag to use html2text library for parsing.
        split_lines: flag to split lines into a list.
        use_tiktoken: flag to use tiktoken library to encode text.
        page_size_limit: optional limit on the number of characters per page. Only
            relevant when split_lines is True.
    """
    path = Path(path)
    try:
        with path.open() as f:
            text = list(f) if split_lines else f.read()
    except UnicodeDecodeError:
        with path.open(encoding="utf-8", errors="ignore") as f:
            text = f.read()

    parsing_libraries: list[str] = ["tiktoken (cl100k_base)"] if use_tiktoken else []
    if html:
        if not isinstance(text, str):
            raise NotImplementedError(
                "HTML parsing is not yet set up to work with split_lines."
            )
        parse_type: str = "html"
        text = html2text(text)
        parsing_libraries.append(f"html2text ({html2text_version})")
    else:
        parse_type = "txt"
    if isinstance(text, str):
        total_length: int = len(text)
    else:
        total_length = sum(len(t) for t in text)
        for i, t in enumerate(text):
            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The {parse_type} on page {i} of {len(text)} was {len(t)} chars"
                    f" long, which exceeds the {page_size_limit} char limit at path"
                    f" {path}."
                )
    return ParsedText(
        content=text,
        metadata=ParsedMetadata(
            parsing_libraries=parsing_libraries,
            paperqa_version=pqa_version,
            total_parsed_text_length=total_length,
            parse_type=parse_type,
        ),
    )


def chunk_text(
    parsed_text: ParsedText,
    doc: Doc,
    chunk_chars: int,
    overlap: int,
    use_tiktoken: bool = True,
) -> list[Text]:
    """Parse a document into chunks, based on tiktoken encoding.

    NOTE: We get some byte continuation errors.
    Currently ignored, but should explore more to make sure we don't miss anything.
    """
    texts: list[Text] = []
    enc = tiktoken.get_encoding("cl100k_base")

    if not isinstance(parsed_text.content, str):
        raise NotImplementedError(
            f"ParsedText.content must be a `str`, not {type(parsed_text.content)}."
        )

    content: str | list[int] = (
        parsed_text.content if not use_tiktoken else parsed_text.encode_content()
    )
    if not content:  # Avoid div0 in token calculations
        raise ImpossibleParsingError(
            f"No text was parsed from the document named {doc.docname!r} with ID"
            f" {doc.dockey}, either empty or corrupted."
        )

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
                text=(
                    enc.decode(cast("list[int]", split))
                    if use_tiktoken
                    else cast("str", split)
                ),
                name=f"{doc.docname} chunk {i + 1}",
                doc=doc,
            )
        )
    return texts


def chunk_code_text(
    parsed_text: ParsedText, doc: Doc, chunk_chars: int, overlap: int
) -> list[Text]:
    """Parse a document into chunks, based on line numbers (for code)."""
    text_buffer = ""
    texts: list[Text] = []
    line_i = last_line_i = 0

    if not isinstance(parsed_text.content, str | list):
        raise NotImplementedError(
            f"Didn't yet handle ParsedText.content of type {type(parsed_text.content)}."
        )

    for line_i, line in enumerate(
        [parsed_text.content]
        if isinstance(parsed_text.content, str)
        else parsed_text.content
    ):
        text_buffer += line
        while len(text_buffer) > chunk_chars:
            texts.append(
                Text(
                    text=text_buffer[:chunk_chars],
                    name=f"{doc.docname} lines {last_line_i}-{line_i}",
                    doc=doc,
                )
            )
            text_buffer = text_buffer[chunk_chars - overlap :]
            last_line_i = line_i
    if (
        len(text_buffer) > overlap  # Save meaningful amount of content as a final text
        or not texts  # Contents were smaller than one chunk, save it anyways
    ):
        texts.append(
            Text(
                text=text_buffer[:chunk_chars],
                name=f"{doc.docname} lines {last_line_i}-{line_i}",
                doc=doc,
            )
        )
    return texts


@overload
async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[True],
    include_metadata: Literal[True],
    chunk_chars: int = ...,
    overlap: int = ...,
    parse_pdf: PDFParserFn | None = ...,
    **parser_kwargs,
) -> ParsedText: ...
@overload
async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[True],
    include_metadata: Literal[False] = ...,
    chunk_chars: int = ...,
    overlap: int = ...,
    parse_pdf: PDFParserFn | None = ...,
    **parser_kwargs,
) -> ParsedText: ...
@overload
async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[False],
    include_metadata: Literal[True],
    chunk_chars: int = ...,
    overlap: int = ...,
    parse_pdf: PDFParserFn | None = ...,
    **parser_kwargs,
) -> tuple[list[Text], ParsedMetadata]: ...
@overload
async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[False] = ...,
    include_metadata: Literal[False] = ...,
    chunk_chars: int = ...,
    overlap: int = ...,
    parse_pdf: PDFParserFn | None = ...,
    **parser_kwargs,
) -> list[Text]: ...
@overload
async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    *,
    include_metadata: Literal[True],
    chunk_chars: int = ...,
    overlap: int = ...,
    parse_pdf: PDFParserFn | None = ...,
    **parser_kwargs,
) -> tuple[list[Text], ParsedMetadata]: ...
async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: bool = False,
    include_metadata: bool = False,
    chunk_chars: int = 3000,
    overlap: int = 100,
    parse_pdf: PDFParserFn | None = None,
    **parser_kwargs,
) -> list[Text] | ParsedText | tuple[list[Text], ParsedMetadata]:
    """Parse a document and split into chunks.

    Optionally can include just the parsing as well as metadata about the parsing/chunking
    Args:
        path: local document path
        doc: object with document metadata
        parsed_text_only: return parsed text without chunking
        include_metadata: return a tuple
        chunk_chars: size of chunks
        overlap: size of overlap between chunks
        parse_pdf: Optional function to parse PDF files (if you're parsing a PDF).
        parser_kwargs: Keyword arguments to pass to the used parsing function.
    """
    str_path = str(path)

    # start with parsing -- users may want to store this separately
    if str_path.endswith(".pdf"):
        if parse_pdf is None:
            raise ValueError("When parsing a PDF, a parsing function must be provided.")
        # Some PDF parsers are not thread-safe,
        # so can't use multithreading via `asyncio.to_thread` here
        parsed_text: ParsedText = parse_pdf(path, **parser_kwargs)
    elif str_path.endswith(".txt"):
        # TODO: Make parse_text async
        parsed_text = await asyncio.to_thread(parse_text, path, **parser_kwargs)
    elif str_path.endswith(".html"):
        parsed_text = await asyncio.to_thread(
            parse_text, path, html=True, **parser_kwargs
        )
    else:
        parsed_text = await asyncio.to_thread(
            parse_text, path, split_lines=True, use_tiktoken=False, **parser_kwargs
        )

    if parsed_text_only:
        return parsed_text

    # next chunk the parsed text

    # check if chunk is 0 (no chunking)
    if chunk_chars == 0:
        chunked_text = [
            Text(text=parsed_text.reduce_content(), name=doc.docname, doc=doc)
        ]
        chunk_metadata = ChunkMetadata(chunk_chars=0, overlap=0, chunk_type="no_chunk")
    elif str_path.endswith(".pdf"):
        chunked_text = chunk_pdf(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars,
            overlap=overlap,
            chunk_type="overlap_pdf_by_page",
        )
    elif str_path.endswith((".txt", ".html")):
        chunked_text = chunk_text(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars, overlap=overlap, chunk_type="overlap"
        )
    else:
        chunked_text = chunk_code_text(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            chunk_chars=chunk_chars,
            overlap=overlap,
            chunk_type="overlap_code_by_line",
        )

    if include_metadata:
        parsed_text.metadata.chunk_metadata = chunk_metadata
        return chunked_text, parsed_text.metadata

    return chunked_text
