import asyncio
import math
import re
import string
from pathlib import Path
from typing import BinaryIO, List, Union

import pypdf
from langchain.base_language import BaseLanguageModel

StrPath = Union[str, Path]


def name_in_text(name: str, text: str) -> bool:
    sname = name.strip()
    pattern = r"\b({0})\b(?!\w)".format(re.escape(sname))
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
    return (
        magic_number == b"<htm"
        or magic_number == b"<!DO"
        or magic_number == b"<xsl"
        or magic_number == b"<!X"
    )


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
        pdf_reader = pypdf.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
    return num_pages


def md5sum(file_path: StrPath) -> str:
    import hashlib

    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


async def gather_with_concurrency(n: int, *coros: List) -> List:
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


def get_llm_name(llm: BaseLanguageModel) -> str:
    try:
        return llm.model_name  # type: ignore
    except AttributeError:
        return llm.model  # type: ignore


def strip_citations(text: str) -> str:
    # Combined regex for identifying citations (see unit tests for examples)
    citation_regex = r"\b[\w\-]+\set\sal\.\s\([0-9]{4}\)|\((?:[^\)]*?[a-zA-Z][^\)]*?[0-9]{4}[^\)]*?)\)"
    # Remove the citations from the text
    text = re.sub(citation_regex, "", text, flags=re.MULTILINE)
    return text


def iter_citations(text: str) -> List[str]:
    # Combined regex for identifying citations (see unit tests for examples)
    citation_regex = r"\b[\w\-]+\set\sal\.\s\([0-9]{4}\)|\((?:[^\)]*?[a-zA-Z][^\)]*?[0-9]{4}[^\)]*?)\)"
    result = re.findall(citation_regex, text, flags=re.MULTILINE)
    return result


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
    else:
        return ""
