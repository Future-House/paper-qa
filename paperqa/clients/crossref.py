from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
from collections.abc import Collection
from datetime import datetime
from typing import Any
from urllib.parse import quote

import aiohttp
from anyio import open_file
from tenacity import retry, stop_after_attempt, wait_exponential

from paperqa.types import CITATION_FALLBACK_DATA, DocDetails
from paperqa.utils import (
    bibtex_field_extract,
    create_bibtex_key,
    remove_substrings,
    strings_similarity,
    union_collections_to_ordered_list,
)

from .client_models import DOIOrTitleBasedProvider, DOIQuery, TitleAuthorQuery
from .exceptions import DOINotFoundError

logger = logging.getLogger(__name__)

CROSSREF_BASE_URL = "https://api.crossref.org"
CROSSREF_HEADER_KEY = "Crossref-Plus-API-Token"
CROSSREF_API_REQUEST_TIMEOUT = 5.0
CROSSREF_API_MAPPING: dict[str, Collection[str]] = {
    "title": {"title"},
    "doi": {"DOI"},
    "authors": {"author"},
    "publication_date": {"published"},
    "year": {"published"},
    "volume": {"volume"},
    "issue": {"issue"},
    "publisher": {"publisher"},
    "issn": {"ISSN"},
    "pages": {"page"},
    "journal": {"container-title"},
    "doi_url": {"URL"},
    "url": {"URL"},
    "bibtex": {"bibtex", "type"},
    "citation_count": {"is-referenced-by-count"},
    "bibtex_type": {"type"},
    "citation": {
        "title",
        "DOI",
        "published",
        "volume",
        "issue",
        "publisher",
        "ISSN",
        "page",
        "container-title",
        "is-referenced-by-count",
        "type",
    },
    "source_quality": {"container-title"},
    "doc_id": {"DOI"},
}
CROSSREF_CONTENT_TYPE_TO_BIBTEX_MAPPING: dict[str, str] = {
    "journal-article": "article",
    "journal-issue": "misc",  # No direct equivalent, so 'misc' is used
    "journal-volume": "misc",  # No direct equivalent, so 'misc' is used
    "journal": "misc",  # No direct equivalent, so 'misc' is used
    "proceedings-article": "inproceedings",
    "proceedings": "proceedings",
    "dataset": "misc",  # No direct equivalent, so 'misc' is used
    "component": "misc",  # No direct equivalent, so 'misc' is used
    "report": "techreport",
    "report-series": "techreport",  # 'series' implies multiple tech reports, but each is still a 'techreport'
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
    "reference-entry": "inbook",  # Entries in reference books can be considered as 'inbook'
    "dissertation": "phdthesis",  # Dissertations are usually PhD thesis
    "posted-content": "misc",  # No direct equivalent, so 'misc' is used
    "peer-review": "misc",  # No direct equivalent, so 'misc' is used
    "other": "article",  # Assume an article if we don't know the type
}


def crossref_headers() -> dict[str, str]:
    """Crossref API key if available, otherwise nothing."""
    if api_key := os.environ.get("CROSSREF_API_KEY"):
        return {CROSSREF_HEADER_KEY: f"Bearer {api_key}"}
    logger.warning(
        "CROSSREF_API_KEY environment variable not set. Crossref API rate limits may"
        " apply."
    )
    return {}


def get_crossref_mailto() -> str:
    """Crossref mailto if available, otherwise a default."""
    MAILTO = os.getenv("CROSSREF_MAILTO")

    if not MAILTO:
        logger.warning(
            "CROSSREF_MAILTO environment variable not set. Crossref API rate limits may"
            " apply."
        )
        return "example@papercrow.ai"

    return MAILTO


async def doi_to_bibtex(
    doi: str,
    session: aiohttp.ClientSession,
    missing_replacements: dict[str, str] | None = None,
) -> str:
    """Get a bibtex entry from a DOI via Crossref, replacing the key if possible.

    `missing_replacements` can optionally be used to fill missing fields in the bibtex key.
        these fields are NOT replaced or inserted into the bibtex string otherwise.

    """
    if missing_replacements is None:
        missing_replacements = {}
    FORBIDDEN_KEY_CHARACTERS = {"_", " ", "-", "/"}
    # get DOI via crossref
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}/transform/application/x-bibtex"
    async with session.get(url, headers=crossref_headers()) as r:
        if not r.ok:
            raise DOINotFoundError(
                f"Per HTTP status code {r.status}, could not resolve DOI {doi}."
            )
        data = await r.text()
    # must make new key
    key = data.split("{")[1].split(",")[0]
    new_key = remove_substrings(key, FORBIDDEN_KEY_CHARACTERS)
    substrings_to_remove_per_field = {"author": [" and ", ","]}
    fragments = [
        remove_substrings(
            bibtex_field_extract(
                data, field, missing_replacements=missing_replacements
            ),
            substrings_to_remove_per_field.get(field, []),
        )
        for field in ("author", "year", "title")
    ]
    # replace the key if all the fragments are present
    if all(fragments):
        new_key = create_bibtex_key(
            author=fragments[0].split(), year=fragments[1], title=fragments[2]
        )
    # we use the count parameter below to ensure only the 1st entry is replaced
    return data.replace(key, new_key, 1)


async def parse_crossref_to_doc_details(
    message: dict[str, Any],
    session: aiohttp.ClientSession,
    query_bibtex: bool = True,
) -> DocDetails:

    bibtex_source = "self_generated"
    bibtex = None

    with contextlib.suppress(DOINotFoundError):
        # get the title from the message, if it exists
        # rare circumstance, but bibtex may not have a title
        fallback_data = copy.copy(CITATION_FALLBACK_DATA)
        if title := (
            None if not message.get("title") else message.get("title", [None])[0]
        ):
            fallback_data["title"] = title

        # TODO: we keep this for robustness, but likely not needed anymore,
        # since we now create the bibtex from scratch
        if query_bibtex:
            bibtex = await doi_to_bibtex(
                message["DOI"], session, missing_replacements=fallback_data  # type: ignore[arg-type]
            )
            # track the origin of the bibtex entry for debugging
            bibtex_source = "crossref"

    authors = [
        f"{author.get('given', '')} {author.get('family', '')}".strip()
        for author in message.get("author", [])
    ]

    publication_date = None
    if "published" in message and "date-parts" in message["published"]:
        date_parts = message["published"]["date-parts"][0]
        if len(date_parts) >= 3:  # noqa: PLR2004
            publication_date = datetime(date_parts[0], date_parts[1], date_parts[2])
        elif len(date_parts) == 2:  # noqa: PLR2004
            publication_date = datetime(date_parts[0], date_parts[1], 1)
        elif len(date_parts) == 1:
            publication_date = datetime(date_parts[0], 1, 1)

    doc_details = DocDetails(  # type: ignore[call-arg]
        key=None if not bibtex else bibtex.split("{")[1].split(",")[0],
        bibtex_type=CROSSREF_CONTENT_TYPE_TO_BIBTEX_MAPPING.get(
            message.get("type", "other"), "misc"
        ),
        bibtex=bibtex,
        authors=authors,
        publication_date=publication_date,
        year=message.get("published", {}).get("date-parts", [[None]])[0][0],
        volume=message.get("volume"),
        issue=message.get("issue"),
        publisher=message.get("publisher"),
        issn=message.get("ISSN", [None])[0],
        pages=message.get("page"),
        journal=(
            None
            if not message.get("container-title")
            else message["container-title"][0]
        ),
        url=message.get("URL"),
        title=None if not message.get("title") else message.get("title", [None])[0],
        citation_count=message.get("is-referenced-by-count"),
        doi=message.get("DOI"),
        other={},  # Initialize empty dict for other fields
    )

    # Add any additional fields to the 'other' dict
    for key, value in (
        message | {"client_source": ["crossref"], "bibtex_source": [bibtex_source]}
    ).items():
        if key not in doc_details.model_fields:
            if key in doc_details.other:
                doc_details.other[key] = [doc_details.other[key], value]
            else:
                doc_details.other[key] = value

    return doc_details


async def get_doc_details_from_crossref(  # noqa: PLR0912
    session: aiohttp.ClientSession,
    doi: str | None = None,
    authors: list[str] | None = None,
    title: str | None = None,
    title_similarity_threshold: float = 0.75,
    fields: Collection[str] | None = None,
) -> DocDetails | None:
    """
    Get paper details from Crossref given a DOI or paper title.

    SEE: https://api.crossref.org/swagger-ui/index.html#/Works
    """
    if authors is None:
        authors = []
    if doi is title is None:
        raise ValueError("Either a DOI or title must be provided.")
    if doi is not None and title is not None:
        title = None  # Prefer DOI over title

    inputs_msg = f"DOI {doi}" if doi is not None else f"title {title}"

    CROSSREF_MAILTO = get_crossref_mailto()
    quoted_doi = f"/{quote(doi, safe='')}" if doi else ""
    url = f"{CROSSREF_BASE_URL}/works{quoted_doi}"
    params = {"mailto": CROSSREF_MAILTO}
    if title:
        params.update({"query.title": title, "rows": "1"})

    if authors:
        params.update(
            {"query.author": " ".join([a.strip() for a in authors if len(a) > 1])}
        )

    query_bibtex = True

    # note we only do field selection if querying on title
    if fields and title:
        # crossref has a special endpoint for bibtex, so we don't need to request it here
        if "bibtex" not in fields:
            query_bibtex = False
        params.update(
            {
                "select": ",".join(
                    union_collections_to_ordered_list(
                        [
                            CROSSREF_API_MAPPING[field]
                            for field in fields
                            if field in CROSSREF_API_MAPPING and field != "bibtex"
                        ]
                    )
                )
            }
        )

    async with session.get(
        url,
        params=params,
        headers=crossref_headers(),
        timeout=aiohttp.ClientTimeout(CROSSREF_API_REQUEST_TIMEOUT),
    ) as response:
        try:
            response.raise_for_status()
        except aiohttp.ClientResponseError as exc:
            raise DOINotFoundError(f"Could not find paper given {inputs_msg}.") from exc
        try:
            response_data = await response.json()
        except json.JSONDecodeError as exc:
            # JSONDecodeError: Crossref didn't answer with JSON, perhaps HTML
            raise DOINotFoundError(  # Use DOINotFoundError so we fall back to Google Scholar
                f"Crossref API did not return JSON for {inputs_msg}, instead it"
                f" responded with text: {await response.text()}"
            ) from exc
    if response_data["status"] == "failed":
        raise DOINotFoundError(
            f"Crossref API returned a failed status for {inputs_msg}."
        )
    message: dict[str, Any] = response_data["message"]
    # restructure data if it comes back as a list result
    # it'll also be a list if we searched by title and it's empty
    if "items" in message:
        try:
            message = message["items"][0]
        except IndexError as e:
            raise DOINotFoundError(
                f"Crossref API did not return any items for {inputs_msg}."
            ) from e
    # since score is not consistent between queries, we need to rely on our own criteria
    # title similarity must be > title_similarity_threshold
    if (
        doi is None
        and title
        and strings_similarity(message["title"][0], title) < title_similarity_threshold
    ):
        raise DOINotFoundError(f"Crossref results did not match for title {title!r}.")
    if doi is not None and message["DOI"] != doi:
        raise DOINotFoundError(f"DOI ({inputs_msg}) not found in Crossref")

    return await parse_crossref_to_doc_details(message, session, query_bibtex)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5),
    reraise=True,
)
async def download_retracted_dataset(
    retraction_data_path: os.PathLike | str,
) -> None:
    """
    Download the retraction dataset from Crossref.

    Saves the retraction dataset to `retraction_data_path`.
    """
    url = f"https://api.labs.crossref.org/data/retractionwatch?{get_crossref_mailto()}"

    async with (
        aiohttp.ClientSession() as session,
        session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as response,
    ):
        response.raise_for_status()

        logger.info(
            f"Retraction data was not cashed. Downloading retraction data from {url}..."
        )

        async with await open_file(str(retraction_data_path), "wb") as f:
            while True:
                chunk = await response.content.read(1024)
                if not chunk:
                    break
                await f.write(chunk)

        if os.path.getsize(str(retraction_data_path)) == 0:
            raise RuntimeError("Retraction data is empty")


class CrossrefProvider(DOIOrTitleBasedProvider):
    async def _query(self, query: TitleAuthorQuery | DOIQuery) -> DocDetails | None:
        if isinstance(query, DOIQuery):
            return await get_doc_details_from_crossref(
                doi=query.doi, session=query.session, fields=query.fields
            )
        return await get_doc_details_from_crossref(
            title=query.title,
            authors=query.authors,
            session=query.session,
            title_similarity_threshold=query.title_similarity_threshold,
            fields=query.fields,
        )
