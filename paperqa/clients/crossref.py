from datetime import datetime
import json
import os
import logging
from typing import Any, Collection
import aiohttp

from paperqa.clients.exceptions import DOINotFoundError
from paperqa.clients.utils import format_bibtex
from paperqa.types import PaperDetails
from paperqa.utils import strings_similarity, encode_id, get_year

logger = logging.getLogger(__name__)

TITLE_SET_SIMILARITY_THRESHOLD = 0.75
CROSSREF_BASE_URL = "https://api.crossref.org"
CROSSREF_API_REQUEST_TIMEOUT = 5.0


def crossref_headers():
    """Crossref API key if available, otherwise nothing."""
    if api_key := os.environ.get("CROSSREF_API_KEY"):
        return {"Crossref-Plus-API-Token": f"Bearer {api_key}"}
    return {}


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
        return bibtex.split(field + "={")[1].split("}")[0].split()[0].strip()
    except IndexError:
        return missing_replacements.get(field, "")

async def doi_to_bibtex(
    doi: str, session: aiohttp.ClientSession, missing_replacements: dict[str, str] | None = None
) -> str:
    """Get a bibtex entry from a DOI via Crossref, replacing the key if possible.

    `missing_replacements` can optionally be used to fill missing fields in the bibtex key.
        these fields are NOT replaced or inserted into the bibtex string otherwise.

    """
    if missing_replacements is None:
        missing_replacements = {}
    FORBIDDEN_KEY_CHARACTERS = {"_", " ", "-", "/"}
    # get DOI via crossref
    url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
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
        new_key = remove_substrings(("".join(fragments)), FORBIDDEN_KEY_CHARACTERS)
    # we use the count parameter below to ensure only the 1st entry is replaced
    return data.replace(key, new_key, 1)


async def parse_crossref_to_paper_details(
    message: dict[str, Any],
    session: aiohttp.ClientSession,
) -> PaperDetails:
    bibtex = await doi_to_bibtex(message["DOI"], session)

    authors = [f"{author.get('given', '')} {author.get('family', '')}".strip() for author in message.get('author', [])]
    
    publication_date = None
    if 'published' in message and 'date-parts' in message['published']:
        date_parts = message['published']['date-parts'][0]
        if len(date_parts) >= 3:
            publication_date = datetime(date_parts[0], date_parts[1], date_parts[2])
        elif len(date_parts) == 2:
            publication_date = datetime(date_parts[0], date_parts[1], 1)
        elif len(date_parts) == 1:
            publication_date = datetime(date_parts[0], 1, 1)

    paper_details = PaperDetails(
        citation=format_bibtex(bibtex),  # Not provided in the JSON
        key=bibtex.split("{")[1].split(",")[0],
        bibtex=bibtex,  # Not provided in the JSON
        authors=authors,
        publication_date=publication_date,
        year=message.get('published', {}).get('date-parts', [[None]])[0][0],
        volume=message.get('volume'),
        issue=message.get('issue'),
        pages=None,  # Not directly provided in the JSON
        journal=message.get('container-title', [None])[0],
        url=message.get('URL'),
        title=message.get('title', [None])[0],
        citation_count=message.get('is-referenced-by-count'),
        doi=message.get('DOI'),
        other={}  # Initialize empty dict for other fields
    )

    # Add any additional fields to the 'other' dict
    for key, value in message.items():
        if key not in paper_details.model_fields:
            paper_details.other[key] = value

    return paper_details


async def get_paper_details_from_crossref(  # noqa: C901
    doi: str | None = None,
    title: str | None = None,
    session: aiohttp.ClientSession | None = None,
    title_similarity_threshold: float = TITLE_SET_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """
    Get paper details from Crossref given a DOI or paper title.

    SEE: https://api.crossref.org/swagger-ui/index.html#/Works
    """
    if doi is title is None:
        raise ValueError("Either a DOI or title must be provided.")
    if doi is not None and title is not None:
        title = None  # Prefer DOI over title
    inputs_msg = f"DOI {doi}" if doi is not None else f"title {title}"

    if not (CROSSREF_MAILTO := os.getenv("CROSSREF_MAILTO")):
        logger.warning(
            "CROSSREF_MAILTO environment variable not set. Crossref API rate limits may apply.")


    url = f"{CROSSREF_BASE_URL}/works{f'/{doi}' if doi else ''}"
    params = {"mailto": CROSSREF_MAILTO}
    if title:
        params.update({"query.title": title, "rows": "1"})
    async with session.get(
        url,
        params=params,
        headers=crossref_headers(),
        timeout=aiohttp.ClientTimeout(CROSSREF_API_REQUEST_TIMEOUT),
    ) as response:
        try:
            response.raise_for_status()
        except aiohttp.ClientResponseError as exc:
            raise DOINotFoundError(
                f"Could not find paper given {inputs_msg}."
            ) from exc
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
    if "items" in message:
        message = message["items"][0]

    # since score is not consistent between queries, we need to rely on our own criteria
    # title similarity must be > title_similarity_threshold
    if (
        doi is None
        and title
        and strings_similarity(message['title'][0], title) < title_similarity_threshold 
    ):
        raise DOINotFoundError(
            f"Crossref results did not match for title {title!r}."
        )
    if doi is not None and message["DOI"] != doi:
        raise DOINotFoundError(f"DOI ({inputs_msg}) not found in Crossref")

    return await parse_crossref_to_paper_details(message, session)
