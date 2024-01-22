from math import ceil
from pathlib import Path
from typing import List

import tiktoken
from html2text import html2text

from .types import Doc, Text


def parse_pdf_fitz(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    import fitz

    file = fitz.open(path)
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i in range(file.page_count):
        page = file.load_page(i)
        split += page.get_text("text", sort=True)
        pages.append(str(i + 1))
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
            pages = [str(i + 1)]
    if len(split) > overlap or len(texts) == 0:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    file.close()
    return texts


def parse_pdf(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    import pypdf

    pdfFileObj = open(path, "rb")
    pdfReader = pypdf.PdfReader(pdfFileObj)
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        pages.append(str(i + 1))
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
            pages = [str(i + 1)]
    if len(split) > overlap or len(texts) == 0:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        )
    pdfFileObj.close()
    return texts


def parse_txt(
    path: Path, doc: Doc, chunk_chars: int, overlap: int, html: bool = False
) -> List[Text]:
    """Parse a document into chunks, based on tiktoken encoding.

    NOTE: We get some byte continuation errors.
    Currnetly ignored, but should explore more to make sure we
    don't miss anything.
    """
    try:
        with open(path) as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    if html:
        text = html2text(text)
    texts: list[Text] = []
    # we tokenize using tiktoken so cuts are in reasonable places
    # See https://github.com/openai/tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    encoded = enc.encode_ordinary(text)
    split = []
    # convert from characters to chunks
    char_count = len(text)  # e.g., 25,000
    token_count = len(encoded)  # e.g., 4,500
    chars_per_token = char_count / token_count  # e.g., 5.5
    chunk_tokens = chunk_chars / chars_per_token  # e.g., 3000 / 5.5 = 545
    overlap_tokens = overlap / chars_per_token  # e.g., 100 / 5.5 = 18
    chunk_count = ceil(token_count / chunk_tokens)  # e.g., 4500 / 545 = 9
    for i in range(chunk_count):
        split = encoded[
            max(int(i * chunk_tokens - overlap_tokens), 0) : int(
                (i + 1) * chunk_tokens + overlap_tokens
            )
        ]
        texts.append(
            Text(
                text=enc.decode(split),
                name=f"{doc.docname} chunk {i + 1}",
                doc=doc,
            )
        )
    return texts


def parse_code_txt(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    """Parse a document into chunks, based on line numbers (for code)."""

    split = ""
    texts: List[Text] = []
    last_line = 0

    with open(path) as f:
        for i, line in enumerate(f):
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


def read_doc(
    path: Path,
    doc: Doc,
    chunk_chars: int = 3000,
    overlap: int = 100,
    force_pypdf: bool = False,
) -> List[Text]:
    """Parse a document into chunks."""
    str_path = str(path)
    if str_path.endswith(".pdf"):
        if force_pypdf:
            return parse_pdf(path, doc, chunk_chars, overlap)
        try:
            return parse_pdf_fitz(path, doc, chunk_chars, overlap)
        except ImportError:
            return parse_pdf(path, doc, chunk_chars, overlap)
    elif str_path.endswith(".txt"):
        return parse_txt(path, doc, chunk_chars, overlap)
    elif str_path.endswith(".html"):
        return parse_txt(path, doc, chunk_chars, overlap, html=True)
    else:
        return parse_code_txt(path, doc, chunk_chars, overlap)
