import re
from typing import Any

import aiohttp
import httpx
import logging
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt, wait_incrementing

from paperqa.clients.exceptions import CitationConversionError


logger = logging.getLogger(__name__)

def clean_upbibtex(bibtex: str) -> str:
    # WTF Semantic Scholar?
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
    # new format check
    match = re.findall(r"@\['(.*)'\]", bibtex)
    if not match:
        match = re.findall(r"@(.*)\{", bibtex)
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
    """Clean bibtex entry, potentially adding missing fields."""
    # WOWOW This is hard to use
    from pybtex.database import Person, parse_string  # noqa: PLC0415
    from pybtex.database.input.bibtex import Parser  # noqa: PLC0415
    from pybtex.style.formatting import unsrtalpha  # noqa: PLC0415
    from pybtex.style.template import FieldIsMissing  # noqa: PLC0415

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
    url: str, params: dict[str, Any], session: aiohttp.ClientSession, headers: dict[str, str] | None = None, timeout: int = 10
) -> dict[str, Any]:
    """Get from a Semantic Scholar API, with retrying protection."""
    async with session.get(
        url,
        params=params,
        headers=headers,
        timeout=aiohttp.ClientTimeout(timeout),
    ) as response:
        response.raise_for_status()
        return await response.json()