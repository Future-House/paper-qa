from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from importlib.metadata import version
from math import ceil
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast, overload, runtime_checkable

import anyio
import tiktoken
from aviary.core import is_coroutine_callable
from html2text import __version__ as html2text_version
from html2text import html2text

from paperqa.types import (
    ChunkMetadata,
    Doc,
    ParsedMedia,
    ParsedMetadata,
    ParsedText,
    Text,
)
from paperqa.utils import ImpossibleParsingError
from paperqa.version import __version__ as pqa_version


@runtime_checkable
class SyncPDFParserFn(Protocol):
    """Protocol for synchronously parsing a PDF."""

    def __call__(
        self,
        path: str | os.PathLike,
        page_size_limit: int | None = None,
        page_range: int | tuple[int, int] | None = None,
        **kwargs,
    ) -> ParsedText: ...


@runtime_checkable
class AsyncPDFParserFn(Protocol):
    """Protocol for asynchronously parsing a PDF."""

    async def __call__(
        self,
        path: str | os.PathLike,
        page_size_limit: int | None = None,
        page_range: int | tuple[int, int] | None = None,
        **kwargs,
    ) -> ParsedText: ...


PDFParserFn: TypeAlias = SyncPDFParserFn | AsyncPDFParserFn


async def parse_image(
    path: str | os.PathLike, validator: Callable[[bytes], Awaitable] | None = None, **_
) -> ParsedText:
    apath = anyio.Path(path)
    image_data = await anyio.Path(path).read_bytes()
    if validator:
        try:
            await validator(image_data)
        except Exception as exc:
            raise ImpossibleParsingError(
                f"Image validation failed for the image at path {path}."
            ) from exc
    parsed_media = ParsedMedia(index=0, data=image_data, info={"suffix": apath.suffix})
    metadata = ParsedMetadata(
        parsing_libraries=[],
        paperqa_version=pqa_version,
        total_parsed_text_length=0,  # No text, just an image
        count_parsed_media=1,
        name="image",
    )
    return ParsedText(content={"1": ("", [parsed_media])}, metadata=metadata)


def _make_chunk(
    parsed_text: ParsedText, doc: Doc, text: str, lower_page: str, upper_page: str
) -> Text:
    media: list[ParsedMedia] = []
    for pg_num in range(int(lower_page), int(upper_page) + 1):
        pg_contents = cast(dict, parsed_text.content).get(str(pg_num))
        if isinstance(pg_contents, tuple):
            media.extend(pg_contents[1])
    # pretty formatting of pages (e.g. 1-3, 4, 5-7)
    name = "-".join([lower_page, upper_page])
    return Text(text=text, name=f"{doc.docname} pages {name}", media=media, doc=doc)


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

    for page_num, page_contents in parsed_text.content.items():
        page_text = (
            page_contents if isinstance(page_contents, str) else page_contents[0]
        )
        split += page_text
        pages.append(page_num)
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            texts.append(
                _make_chunk(parsed_text, doc, split[:chunk_chars], pages[0], pages[-1])
            )
            split = split[chunk_chars - overlap :]
            pages = [page_num]

    if len(split) > overlap or not texts:
        texts.append(
            _make_chunk(parsed_text, doc, split[:chunk_chars], pages[0], pages[-1])
        )
    return texts


def parse_text(
    path: str | os.PathLike,
    html: bool = False,
    split_lines: bool = False,
    page_size_limit: int | None = None,
    **_,
) -> ParsedText:
    """Simple text splitter, can parse html or split into newlines.

    Args:
        path: path to file.
        html: flag to use html2text library for parsing.
        split_lines: flag to split lines into a list.
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

    parsing_libraries: list[str] = []
    if html:
        if not isinstance(text, str):
            raise NotImplementedError(
                "HTML parsing is not yet set up to work with split_lines."
            )
        parse_summary: str = "html"
        text = html2text(text)
        parsing_libraries.append(f"html2text ({html2text_version})")
    else:
        parse_summary = "txt"
    if isinstance(text, str):
        total_length: int = len(text)
    else:
        total_length = sum(len(t) for t in text)
        for i, t in enumerate(text):
            if page_size_limit and len(text) > page_size_limit:
                raise ImpossibleParsingError(
                    f"The {parse_summary} on page {i} of {len(text)} was {len(t)} chars"
                    f" long, which exceeds the {page_size_limit} char limit at path"
                    f" {path}."
                )
    return ParsedText(
        content=text,
        metadata=ParsedMetadata(
            parsing_libraries=parsing_libraries,
            paperqa_version=pqa_version,
            total_parsed_text_length=total_length,
            name=f"{parse_summary}|split-lines={split_lines}",
        ),
    )


def parse_office_doc(
    path: str | os.PathLike,
    **kwargs,
) -> ParsedText:
    """Parse office documents (.docx, .xlsx, .pptx) using unstructured, extracting text and images."""
    try:
        import unstructured
        from unstructured.documents.elements import Image, Table
        from unstructured.partition.auto import partition
    except ImportError as exc:
        raise ImportError(
            "Could not import `unstructured` dependencies. "
            "Please install with `pip install paper-qa[office]`."
        ) from exc
    UNSTRUCTURED_VERSION = version(unstructured.__name__)
    elements = partition(str(path), **kwargs)

    content_dict = {}
    media_list: list[ParsedMedia] = []
    current_text = ""
    media_index = 0

    for el in elements:
        if isinstance(el, Image):
            image_data = el.metadata.image_data
            # Create a ParsedMedia object
            parsed_media = ParsedMedia(
                index=media_index,
                data=image_data,
                info={"suffix": el.metadata.image_mime_type},
            )
            media_list.append(parsed_media)
            media_index += 1
        elif isinstance(el, Table):
            # For tables, we could get the HTML representation for better structure
            if el.metadata.text_as_html:
                current_text += el.metadata.text_as_html + "\n\n"
        else:
            current_text += str(el) + "\n\n"

    # For office docs, we can treat the whole document as a single "page"
    content_dict["1"] = (current_text, media_list)

    return ParsedText(
        content=content_dict,
        metadata=ParsedMetadata(
            parsing_libraries=[f"{unstructured.__name__} ({UNSTRUCTURED_VERSION})"],
            paperqa_version=pqa_version,
            total_parsed_text_length=len(current_text),
            count_parsed_media=len(media_list),
            name=f"office_doc|path={path}",
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
        parsed_text.content
        if not use_tiktoken
        # we tokenize using tiktoken so cuts are in reasonable places
        else cast(list[int], parsed_text.encode_content(enc))
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


IMAGE_EXTENSIONS = tuple({".png", ".jpg", ".jpeg"})
# When HTML reader supports images, add here
ENRICHMENT_EXTENSIONS = tuple({".pdf", ".docx", ".xlsx", ".pptx", *IMAGE_EXTENSIONS})


@overload
async def read_doc(
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: Literal[True],
    include_metadata: Literal[True],
    chunk_chars: int = ...,
    overlap: int = ...,
    multimodal_enricher: Callable[[ParsedText], Awaitable] | None = ...,
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
    multimodal_enricher: Callable[[ParsedText], Awaitable] | None = ...,
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
    multimodal_enricher: Callable[[ParsedText], Awaitable] | None = ...,
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
    multimodal_enricher: Callable[[ParsedText], Awaitable] | None = ...,
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
    multimodal_enricher: Callable[[ParsedText], Awaitable] | None = ...,
    parse_pdf: PDFParserFn | None = ...,
    **parser_kwargs,
) -> tuple[list[Text], ParsedMetadata]: ...
async def read_doc(  # noqa: PLR0912
    path: str | os.PathLike,
    doc: Doc,
    parsed_text_only: bool = False,
    include_metadata: bool = False,
    chunk_chars: int = 5000,
    overlap: int = 250,
    multimodal_enricher: Callable[[ParsedText], Awaitable[str]] | None = None,
    parse_pdf: PDFParserFn | None = None,
    **parser_kwargs,
) -> list[Text] | ParsedText | tuple[list[Text], ParsedMetadata]:
    """Parse a document and split into chunks.

    Args:
        path: local document path
        doc: object with document metadata
        parsed_text_only: return parsed text without chunking
        include_metadata: Opt-in flag to include metadata about the chunking algorithm.
        chunk_chars: size of chunks
        overlap: size of overlap between chunks
        multimodal_enricher: Optional function to enrich the parsed text
            and return a hashable string summary before chunking.
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
        if is_coroutine_callable(parse_pdf):
            parsed_text: ParsedText = await cast(AsyncPDFParserFn, parse_pdf)(
                path, **parser_kwargs
            )
        else:
            parsed_text = cast(SyncPDFParserFn, parse_pdf)(path, **parser_kwargs)
    elif str_path.endswith(".txt"):
        # TODO: Make parse_text async
        parsed_text = await asyncio.to_thread(parse_text, path, **parser_kwargs)
    elif str_path.endswith(".html"):
        parsed_text = await asyncio.to_thread(
            parse_text, path, html=True, **parser_kwargs
        )
    elif str_path.endswith(IMAGE_EXTENSIONS):
        parsed_text = await parse_image(path, **parser_kwargs)
    elif str_path.endswith((".docx", ".xlsx", ".pptx")):
        # TODO: Make parse_office_doc async
        parsed_text = await asyncio.to_thread(parse_office_doc, path, **parser_kwargs)
    else:
        parsed_text = await asyncio.to_thread(
            parse_text, path, split_lines=True, **parser_kwargs
        )

    if parsed_text_only:
        return parsed_text

    # Enrich upon full parsed text before chunking, since enrichment
    # may view adjacent pages (and not getting cut off on chunk boundaries)
    if str_path.endswith(ENRICHMENT_EXTENSIONS) and multimodal_enricher:
        enrichment_summary: str = f"|{await multimodal_enricher(parsed_text)}"
    else:
        enrichment_summary = ""

    # next chunk the parsed text

    if chunk_chars == 0:
        chunked_text = [
            Text(text=parsed_text.reduce_content(), name=doc.docname, doc=doc)
        ]
        chunk_metadata = ChunkMetadata(
            size=0,
            overlap=0,
            name=(
                f"paper-qa={pqa_version}|algorithm=none"
                f"|reduction=cl100k_base{enrichment_summary}"
            ),
        )
    elif str_path.endswith((".pdf", ".docx", ".xlsx", ".pptx")):
        chunked_text = chunk_pdf(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            size=chunk_chars,
            overlap=overlap,
            name=(
                f"paper-qa={pqa_version}|algorithm=overlap-document"
                f"|size={chunk_chars}|overlap={overlap}{enrichment_summary}"
            ),
        )
    elif str_path.endswith(IMAGE_EXTENSIONS):
        chunked_text = chunk_pdf(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            size=0,
            overlap=0,
            name=f"paper-qa={pqa_version}|algorithm=none{enrichment_summary}",
        )
    elif str_path.endswith((".txt", ".html")):
        chunked_text = chunk_text(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            size=chunk_chars,
            overlap=overlap,
            name=(
                f"paper-qa={pqa_version}|algorithm=overlap-text|reduction=cl100k_base"
                f"|size={chunk_chars}|overlap={overlap}{enrichment_summary}"
            ),
        )

    else:
        chunked_text = chunk_code_text(
            parsed_text, doc, chunk_chars=chunk_chars, overlap=overlap
        )
        chunk_metadata = ChunkMetadata(
            size=chunk_chars,
            overlap=overlap,
            name=(
                f"paper-qa={pqa_version}|algorithm=overlap-code|reduction=cl100k_base"
                f"|size={chunk_chars}|overlap={overlap}{enrichment_summary}"
            ),
        )

    if include_metadata:
        parsed_text.metadata.chunk_metadata = chunk_metadata
        return chunked_text, parsed_text.metadata

    return chunked_text
