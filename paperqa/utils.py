from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import re
import string
from collections.abc import Iterable
from datetime import datetime
from functools import reduce
from http import HTTPStatus
from pathlib import Path
from typing import Any, BinaryIO, Collection, Coroutine, Iterator, Union
from uuid import UUID

import aiohttp
import httpx
import pypdf
from pybtex.database import Person, parse_string
from pybtex.database.input.bibtex import Parser
from pybtex.style.formatting import unsrtalpha
from pybtex.style.template import FieldIsMissing
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_incrementing,
)

logger = logging.getLogger(__name__)


StrPath = Union[str, Path]


def name_in_text(name: str, text: str) -> bool:
    sname = name.strip()
    pattern = rf"\b({re.escape(sname)})\b(?!\w)"
    return bool(re.search(pattern, text))


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
    return entropy > thresh


def maybe_is_pdf(file: BinaryIO) -> bool:
    magic_number = file.read(4)
    file.seek(0)
    return magic_number == b"%PDF"


def maybe_is_html(file: BinaryIO) -> bool:
    magic_number = file.read(4)
    file.seek(0)
    return magic_number in {b"<htm", b"<!DO", b"<xsl", b"<!X"}


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


def hexdigest(data: str | bytes) -> str:
    if isinstance(data, str):
        return hashlib.md5(data.encode("utf-8")).hexdigest()  # noqa: S324
    return hashlib.md5(data).hexdigest()  # noqa: S324


def md5sum(file_path: StrPath) -> str:
    with open(file_path, "rb") as f:
        return hexdigest(f.read())


async def gather_with_concurrency(n: int, coros: list[Coroutine]) -> list[Any]:
    # https://stackoverflow.com/a/61478547/2392535
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


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


def llm_read_json(text: str) -> dict:
    """Read LLM output and extract JSON data from it."""
    # fetch from markdown ```json if present
    text = text.strip().split("```json")[-1].split("```")[0]
    # split anything before the first {
    text = "{" + text.split("{", 1)[-1]
    # split anything after the last }
    text = text.rsplit("}", 1)[0] + "}"

    # escape new lines within strings
    def replace_newlines(match: re.Match) -> str:
        return match.group(0).replace("\n", "\\n")

    # Match anything between double quotes
    # including escaped quotes and other escaped characters.
    # https://regex101.com/r/VFcDmB/1
    pattern = r'"(?:[^"\\]|\\.)*"'
    text = re.sub(pattern, replace_newlines, text)

    return json.loads(text)


def encode_id(value: str | bytes | UUID, maxsize: int | None = 16) -> str:
    """Encode a value (e.g. a DOI) optionally with a max length."""
    if isinstance(value, UUID):
        value = str(value)
    if isinstance(value, str):
        value = value.lower().encode()
    return hashlib.md5(value).hexdigest()[:maxsize]  # noqa: S324


def get_year(ts: datetime | None = None) -> str:
    """Get the year from the input datetime, otherwise using the current datetime."""
    if ts is None:
        ts = datetime.now()
    return ts.strftime("%Y")


class CitationConversionError(Exception):
    """Exception to throw when we can't process a citation from a BibTeX."""


def clean_upbibtex(bibtex: str) -> str:

    if not bibtex:
        return bibtex

    mapping = {
        "None": "article",
        "Article": "article",
        "JournalArticle": "article",
        "Review": "article",
        "Book": "book",
        "BookSection": "inbook",
        "ConferencePaper": "inproceedings",
        "Conference": "inproceedings",
        "Dataset": "misc",
        "Dissertation": "phdthesis",
        "Journal": "article",
        "Patent": "patent",
        "Preprint": "article",
        "Report": "techreport",
        "Thesis": "phdthesis",
        "WebPage": "misc",
        "Plain": "article",
    }
    if "@None" in bibtex:
        return bibtex.replace("@None", "@article")
    match = re.findall(r"@\['(.*)'\]", bibtex)
    if not match:
        match = re.findall(r"@(\w+)\{", bibtex)
        bib_type = match[0]
        current = f"@{match[0]}"
    else:
        bib_type = match[0]
        current = f"@['{bib_type}']"
    for k, v in mapping.items():
        # can have multiple
        if k in bib_type:
            bibtex = bibtex.replace(current, f"@{v}")
            break
    return bibtex


def format_bibtex(  # noqa: C901
    bibtex: str,
    key: str | None = None,
    clean: bool = True,
    missing_replacements: dict[str, str] | None = None,
) -> str:
    """Transform bibtex entry into a citation, potentially adding missing fields."""
    if missing_replacements is None:
        missing_replacements = {}
    if key is None:
        key = bibtex.split("{")[1].split(",")[0]
    style = unsrtalpha.Style()
    try:
        bd = parse_string(clean_upbibtex(bibtex) if clean else bibtex, "bibtex")
    except Exception:
        return "Ref " + key
    try:
        entry = bd.entries[key]
    except KeyError as exc:  # Let's check if key is a non-empty prefix
        try:
            entry = next(
                iter(v for k, v in bd.entries.items() if k.startswith(key) and key)
            )
        except StopIteration:
            raise CitationConversionError(
                f"Failed to process{' and clean up' if clean else ''} bibtex {bibtex}"
                f" due to failed lookup of key {key}."
            ) from exc
    try:
        # see if we can insert missing fields
        for field, replacement_value in missing_replacements.items():
            # Deal with special case for author, since it needs to be parsed
            # into Person objects. This reorganizes the names automatically.
            if field == "author" and "author" not in entry.persons:
                tmp_author_bibtex = f"@misc{{tmpkey, author={{{replacement_value}}}}}"
                authors: list[Person] = (
                    Parser()
                    .parse_string(tmp_author_bibtex)
                    .entries["tmpkey"]
                    .persons["author"]
                )
                for a in authors:
                    entry.add_person(a, "author")
            elif field not in entry.fields:
                entry.fields.update({field: replacement_value})
        entry = style.format_entry(label="1", entry=entry)
        return entry.text.render_as("text")
    except (FieldIsMissing, UnicodeDecodeError):
        try:
            return entry.fields["title"]
        except KeyError as exc:
            raise CitationConversionError(
                f"Failed to process{' and clean up' if clean else ''} bibtex {bibtex}"
                " due to missing a 'title' field."
            ) from exc


def remove_substrings(target: str, substr_removal_list: Collection[str]) -> str:
    """Remove substrings from a target string."""
    if all(len(w) == 1 for w in substr_removal_list):
        return target.translate(str.maketrans("", "", "".join(substr_removal_list)))

    for substr in substr_removal_list:
        target = target.replace(substr, "")
    return target


def bibtex_field_extract(
    bibtex: str, field: str, missing_replacements: dict[str, str] | None = None
) -> str:
    """Get a field from a bibtex entry.

    Args:
        bibtex: bibtex entry
        field: field to extract
        missing_replacements: replacement extract for field if not present in the bibtex string
    """
    if missing_replacements is None:
        missing_replacements = {}
    try:
        pattern = rf"{field}\s*=\s*{{(.*?)}},"
        # note: we intentionally have an attribute error if no match
        return re.search(pattern, bibtex, re.IGNORECASE).group(1).strip()  # type: ignore[union-attr]
    except AttributeError:
        return missing_replacements.get(field, "")


def create_bibtex_key(author: list[str], year: str, title: str) -> str:
    FORBIDDEN_KEY_CHARACTERS = {"_", " ", "-", "/", "'", "`", ":", ",", "\n"}
    author_rep = (
        author[0].split()[-1].casefold()
        if "Unknown" not in author[0]
        else "unknownauthors"
    )
    # we don't want a bibtex-parsing induced line break in the key
    # so we cap it to 100+50+4 = 154 characters max
    # 50 for the author, 100 for the first three title words, 4 for the year
    # the first three title words are just emulating the s2 convention
    key = f"{author_rep[:50]}{year}{''.join([t.casefold() for t in title.split()[:3]])[:100]}"
    return remove_substrings(key, FORBIDDEN_KEY_CHARACTERS)


@retry(
    retry=retry_if_exception(
        lambda x: isinstance(x, aiohttp.ServerDisconnectedError)
        or isinstance(x, aiohttp.ClientResponseError)
        and x.status
        in {
            httpx.codes.INTERNAL_SERVER_ERROR.value,
            httpx.codes.GATEWAY_TIMEOUT.value,
        }
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    stop=stop_after_attempt(5),
    wait=wait_incrementing(0.1, 0.1),
)
async def _get_with_retrying(
    url: str,
    params: dict[str, Any],
    session: aiohttp.ClientSession,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,  # noqa: ASYNC109
    http_exception_mappings: dict[HTTPStatus | int, Exception] | None = None,
) -> dict[str, Any]:
    """Get from a URL with retrying protection."""
    try:
        async with session.get(
            url,
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(timeout),
        ) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientResponseError as e:
        if http_exception_mappings and e.status in http_exception_mappings:
            raise http_exception_mappings[e.status] from e
        raise


def union_collections_to_ordered_list(collections: Iterable) -> list:
    return sorted(reduce(lambda x, y: set(x) | set(y), collections))


def pqa_directory(name: str) -> Path:
    if pqa_home := os.environ.get("PQA_HOME"):
        directory = Path(pqa_home) / ".pqa" / name
    else:
        directory = Path.home() / ".pqa" / name

    directory.mkdir(parents=True, exist_ok=True)
    return directory
