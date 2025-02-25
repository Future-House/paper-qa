from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import logging.config
import math
import os
import re
import string
import unicodedata
from collections.abc import Collection, Iterable, Iterator
from datetime import datetime
from functools import reduce
from http import HTTPStatus
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, TypeVar
from uuid import UUID

import aiohttp
import httpx
import pymupdf
from lmi import configure_llm_logs
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

T = TypeVar("T")


class ImpossibleParsingError(Exception):
    """Error to throw when a parsing is impossible."""

    LOG_METHOD_NAME: ClassVar[str] = "warning"


def name_in_text(name: str, text: str) -> bool:
    sname = name.strip()
    pattern = rf"\b({re.escape(sname)})\b(?!\w)"
    return bool(re.search(pattern, text))


def maybe_is_text(s: str, thresh: float = 2.5) -> bool:
    """
    Calculate the entropy of the string to discard files with excessively repeated symbols.

    PDF parsing sometimes represents horizontal distances between words on title pages
    and in tables with spaces, which should therefore not be included in this calculation.
    """
    if not s:
        return False

    entropy = 0.0
    s_wo_spaces = s.replace(" ", "")
    for c in string.printable:
        p = s_wo_spaces.count(c) / len(s_wo_spaces)
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


def strings_similarity(s1: str, s2: str, case_insensitive: bool = True) -> float:
    if not s1 or not s2:
        return 0

    # break the strings into words
    ss1 = set(s1.lower().split()) if case_insensitive else set(s1.split())
    ss2 = set(s2.lower().split()) if case_insensitive else set(s2.split())

    # return the similarity ratio
    return len(ss1.intersection(ss2)) / len(ss1.union(ss2))


def count_pdf_pages(file_path: str | os.PathLike) -> int:
    with pymupdf.open(file_path) as doc:
        return len(doc)


def hexdigest(data: str | bytes) -> str:
    if isinstance(data, str):
        return hashlib.md5(data.encode("utf-8")).hexdigest()  # noqa: S324
    return hashlib.md5(data).hexdigest()  # noqa: S324


def md5sum(file_path: str | os.PathLike) -> str:
    return hexdigest(Path(file_path).read_bytes())


def strip_citations(text: str) -> str:
    # Combined regex for identifying citations (see unit tests for examples)
    citation_regex = r"\b[\w\-]+\set\sal\.\s\([0-9]{4}\)|\((?:[^\)]*?[a-zA-Z][^\)]*?[0-9]{4}[^\)]*?)\)"
    # Remove the citations from the text
    return re.sub(citation_regex, "", text, flags=re.MULTILINE)


def extract_score(text: str) -> int:
    """
    Extract an integer score from the text in 0 to 10.

    Note: score is 1-10, and we use 0 as a sentinel for not applicable.
    """
    # Check for N/A, not applicable, not relevant.
    # Don't check for NA, as there can be genes containing "NA"
    last_line = text.split("\n")[-1]
    if (
        "n/a" in last_line.lower()
        or "not applicable" in text.lower()
        or "not relevant" in text.lower()
    ):
        return 0

    score = re.search(r"[sS]core[:is\s]+([0-9]+)", text)
    if not score:
        score = re.search(r"\(([0-9])\w*\/", text)
    if not score:
        score = re.search(r"([0-9]+)\w*\/", text)
    if score:
        s = int(score.group(1))
        if s > 10:  # noqa: PLR2004
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    last_few = text[-15:]
    scores = re.findall(r"([0-9]+)", last_few)
    if scores:
        s = int(scores[-1])
        if s > 10:  # noqa: PLR2004
            s = int(s / 10)  # sometimes becomes out of 100
        return s
    if len(text) < 100:  # noqa: PLR2004
        return 1
    return 5


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
        citation = citation.strip("() ")
        for c in re.split(r",|;", citation):
            if c == "Extra background information":
                continue
            # remove leading/trailing spaces
            c = c.strip()
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


def get_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


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


def format_bibtex(
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


def mutate_acute_accents(text: str, replace: bool = False) -> str:
    """
    Replaces or removes acute accents in a string based on the boolean flag.

    Args:
        text: The input string.
        replace: A flag to determine whether to replace (True) or remove (False) acute accents.

            If 'replace' is True, acute accents on vowels are replaced with an apostrophe (e.g., "á" becomes "'a").

            If 'replace' is False, all acute accents are removed from the string.

    Returns:
        The modified string with acute accents either replaced or removed.
    """
    if replace:

        def replace_acute(match):
            return f"'{match.group(1)}"

        nfd = unicodedata.normalize("NFD", text)
        converted = re.sub(r"([aeiouAEIOU])\u0301", replace_acute, nfd)
        return unicodedata.normalize("NFC", converted)
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


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


UNKNOWN_AUTHOR_KEY: str = "unknownauthors"


def create_bibtex_key(author: list[str], year: str, title: str) -> str:
    FORBIDDEN_KEY_CHARACTERS = {"_", " ", "-", "/", "'", "`", ":", ",", "\n"}
    try:
        author_rep = (
            # casefold will not remove accutes
            mutate_acute_accents(text=author[0].split()[-1].casefold())
            if "Unknown" not in author[0]
            else UNKNOWN_AUTHOR_KEY
        )
    except IndexError:
        author_rep = UNKNOWN_AUTHOR_KEY
    # we don't want a bibtex-parsing induced line break in the key
    # so we cap it to 100+50+4 = 154 characters max
    # 50 for the author, 100 for the first three title words, 4 for the year
    # the first three title words are just emulating the s2 convention
    key = f"{author_rep[:50]}{year}{''.join([t.casefold() for t in title.split()[:3]])[:100]}"
    return remove_substrings(key, FORBIDDEN_KEY_CHARACTERS)


def is_retryable(exc: BaseException) -> bool:
    """Check if an exception is known to be a retryable HTTP issue."""
    if isinstance(
        exc, aiohttp.ServerDisconnectedError | aiohttp.ClientConnectionResetError
    ):
        # Seen with Semantic Scholar:
        # > aiohttp.client_exceptions.ClientConnectionResetError:
        # > Cannot write to closing transport
        return True
    return isinstance(exc, aiohttp.ClientResponseError) and exc.status in {
        httpx.codes.INTERNAL_SERVER_ERROR.value,
        httpx.codes.GATEWAY_TIMEOUT.value,
    }


@retry(
    retry=retry_if_exception(is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    stop=stop_after_attempt(5),
    wait=wait_incrementing(0.1, 0.1),
)
async def _get_with_retrying(
    url: str,
    session: aiohttp.ClientSession,
    http_exception_mappings: dict[HTTPStatus | int, Exception] | None = None,
    **get_kwargs,
) -> dict[str, Any]:
    """Get from a URL with retrying protection."""
    try:
        async with session.get(url, **get_kwargs) as response:
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


def setup_default_logs() -> None:
    """Configure logs to reasonable defaults."""
    # Trigger PyMuPDF to use Python logging
    # SEE: https://pymupdf.readthedocs.io/en/latest/app3.html#diagnostics
    pymupdf.set_messages(pylogging=True)
    configure_llm_logs()


def extract_thought(content: str | None) -> str:
    """Extract an Anthropic thought from a message's content."""
    # SEE: https://regex101.com/r/bpJt05/1
    return re.sub(r"<\/?thinking>", "", content or "")


BIBTEX_MAPPING: dict[str, str] = {
    """Maps client bibtex types to pybtex types""" "journal-article": "article",
    "journal-issue": "misc",  # No direct equivalent, so 'misc' is used
    "journal-volume": "misc",  # No direct equivalent, so 'misc' is used
    "journal": "misc",  # No direct equivalent, so 'misc' is used
    "proceedings-article": "inproceedings",
    "proceedings": "proceedings",
    "dataset": "misc",  # No direct equivalent, so 'misc' is used
    "component": "misc",  # No direct equivalent, so 'misc' is used
    "report": "techreport",
    "report-series": (  # 'series' implies multiple tech reports, but each is still a 'techreport'
        "techreport"
    ),
    "standard": "misc",  # No direct equivalent, so 'misc' is used
    "standard-series": "misc",  # No direct equivalent, so 'misc' is used
    "edited-book": "book",  # Edited books are considered books in BibTeX
    "monograph": "book",  # Monographs are considered books in BibTeX
    "reference-book": "book",  # Reference books are considered books in BibTeX
    "book": "book",
    "book-series": "book",  # Series of books can be considered as 'book' in BibTeX
    "book-set": "book",  # Set of books can be considered as 'book' in BibTeX
    "book-chapter": "inbook",
    "book-section": "inbook",  # Sections in books can be considered as 'inbook'
    "book-part": "inbook",  # Parts of books can be considered as 'inbook'
    "book-track": "inbook",  # Tracks in books can be considered as 'inbook'
    "reference-entry": (  # Entries in reference books can be considered as 'inbook'
        "inbook"
    ),
    "dissertation": "phdthesis",  # Dissertations are usually PhD thesis
    "posted-content": "misc",  # No direct equivalent, so 'misc' is used
    "peer-review": "misc",  # No direct equivalent, so 'misc' is used
    "other": "article",  # Assume an article if we don't know the type
}


@contextlib.contextmanager
def logging_filters(
    loggers: Collection[str], filters: Collection[type[logging.Filter]]
):
    """Temporarily add a filter to each specified logger."""
    filters_added: dict[str, list[logging.Filter]] = {}
    try:
        for logger_name in loggers:
            log_to_filter = logging.getLogger(logger_name)
            for log_filter in filters:
                _filter = log_filter()
                log_to_filter.addFilter(_filter)
                if logger_name not in filters_added:
                    filters_added[logger_name] = [_filter]
                else:
                    filters_added[logger_name] += [_filter]
        yield
    finally:
        for logger_name, log_filters_to_remove in filters_added.items():
            log_with_filter = logging.getLogger(logger_name)
            for log_filter_to_remove in log_filters_to_remove:
                log_with_filter.removeFilter(log_filter_to_remove)


def citation_to_docname(citation: str) -> str:
    """Create a docname that follows MLA parenthetical in-text citation."""
    # get first name and year from citation
    match = re.search(r"([A-Z][a-z]+)", citation)
    if match is not None:
        author = match.group(1)
    else:
        # panicking - no word??
        raise ValueError(
            f"Could not parse docname from citation {citation}. "
            "Consider just passing key explicitly - e.g. docs.py "
            "(path, citation, key='mykey')"
        )
    year = ""
    match = re.search(r"(\d{4})", citation)
    if match is not None:
        year = match.group(1)
    return f"{author}{year}"


def maybe_get_date(date: str | datetime | None) -> datetime | None:
    if not date:
        return None
    if isinstance(date, str):
        # Try common date formats in sequence
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",  # ISO with timezone: 2023-01-31T14:30:00+0000
            "%Y-%m-%d %H:%M:%S",  # ISO with time: 2023-01-31 14:30:00
            "%B %d, %Y",  # Full month day, year: January 31, 2023
            "%b %d, %Y",  # Month day, year: Jan 31, 2023
            "%Y-%m-%d",  # ISO format: 2023-01-31
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date, fmt)
            except ValueError:
                continue
        return None
    return date
