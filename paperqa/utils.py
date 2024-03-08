from __future__ import annotations

import asyncio
import inspect
import math
import re
import string
from pathlib import Path
from typing import Any, BinaryIO, Coroutine, Iterator, Union

import pypdf

StrPath = Union[str, Path]


def name_in_text(name: str, text: str) -> bool:
    sname = name.strip()
    pattern = rf"\b({re.escape(sname)})\b(?!\w)"
    if re.search(pattern, text):
        return True
    return False


def maybe_is_text(s: str, thresh: float = 2.5) -> bool:
    if len(s) == 0:
        return False
    # Calculate the entropy of the string
    entropy = 0.0
    for c in string.printable:
        p = s.count(c) / len(s)
        if p > 0:
            entropy += -p * math.log2(p)

    # Check if the entropy is within a reasonable range for text
    if entropy > thresh:
        return True
    return False


def maybe_is_pdf(file: BinaryIO) -> bool:
    magic_number = file.read(4)
    file.seek(0)
    return magic_number == b"%PDF"


def maybe_is_html(file: BinaryIO) -> bool:
    magic_number = file.read(4)
    file.seek(0)
    return magic_number in (b"<htm", b"<!DO", b"<xsl", b"<!X")


def strings_similarity(s1: str, s2: str) -> float:
    if len(s1) == 0 or len(s2) == 0:
        return 0
    # break the strings into words
    ss1 = set(s1.split())
    ss2 = set(s2.split())
    # return the similarity ratio
    return len(ss1.intersection(ss2)) / len(ss1.union(ss2))


def count_pdf_pages(file_path: StrPath) -> int:
    with open(file_path, "rb") as pdf_file:
        try:  # try fitz by default
            import fitz

            doc = fitz.open(file_path)
            num_pages = len(doc)
        except ModuleNotFoundError:  # pypdf instead
            pdf_reader = pypdf.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
    return num_pages


def md5sum(file_path: StrPath) -> str:
    import hashlib

    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()  # noqa: S324


async def gather_with_concurrency(n: int, coros: list[Coroutine]) -> list[Any]:
    # https://stackoverflow.com/a/61478547/2392535
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


def guess_is_4xx(msg: str) -> bool:
    if re.search(r"4\d\d", msg):
        return True
    return False


def strip_citations(text: str) -> str:
    # Combined regex for identifying citations (see unit tests for examples)
    citation_regex = r"\b[\w\-]+\set\sal\.\s\([0-9]{4}\)|\((?:[^\)]*?[a-zA-Z][^\)]*?[0-9]{4}[^\)]*?)\)"
    # Remove the citations from the text
    return re.sub(citation_regex, "", text, flags=re.MULTILINE)


def get_citenames(text: str) -> set[str]:
    # Combined regex for identifying citations (see unit tests for examples)
    citation_regex = r"\b[\w\-]+\set\sal\.\s\([0-9]{4}\)|\((?:[^\)]*?[a-zA-Z][^\)]*?[0-9]{4}[^\)]*?)\)"
    results = re.findall(citation_regex, text, flags=re.MULTILINE)
    # now find None patterns
    none_citation_regex = r"(\(None[a-f]{0,1} pages [0-9]{1,10}-[0-9]{1,10}\))"
    none_results = re.findall(none_citation_regex, text, flags=re.MULTILINE)
    results.extend(none_results)
    values = []
    for citation in results:
        citation = citation.strip("() ")  # noqa: PLW2901
        for c in re.split(",|;", citation):
            if c == "Extra background information":
                continue
            # remove leading/trailing spaces
            c = c.strip()  # noqa: PLW2901
            values.append(c)
    return set(values)


def extract_doi(reference: str) -> str:
    """
    Extracts DOI from the reference string using regex.

    :param reference: A string containing the reference.
    :return: A string containing the DOI link or a message if DOI is not found.
    """
    # DOI regex pattern
    doi_pattern = r"10.\d{4,9}/[-._;()/:A-Z0-9]+"
    doi_match = re.search(doi_pattern, reference, re.IGNORECASE)

    # If DOI is found in the reference, return the DOI link
    if doi_match:
        return "https://doi.org/" + doi_match.group()
    else:  # noqa: RET505
        return ""


def batch_iter(iterable: list, n: int = 1) -> Iterator[list]:
    """
    Batch an iterable into chunks of size n.

    :param iterable: The iterable to batch
    :param n: The size of the batches
    :return: A list of batches
    """
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


def flatten(iteratble: list) -> list:
    """
    Flatten a list of lists.

    :param l: The list of lists to flatten
    :return: A flattened list
    """
    return [item for sublist in iteratble for item in sublist]


def get_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def is_coroutine_callable(obj):
    if inspect.isfunction(obj):
        return inspect.iscoroutinefunction(obj)
    elif callable(obj):  # noqa: RET505
        return inspect.iscoroutinefunction(obj.__call__)
    return False
