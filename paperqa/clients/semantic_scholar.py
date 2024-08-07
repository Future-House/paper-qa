import contextlib
from datetime import datetime
from enum import IntEnum, auto
from itertools import starmap
import logging
import os
from typing import Any
from wsgiref import headers
import aiohttp

from paperqa.clients.crossref import doi_to_bibtex
from paperqa.clients.exceptions import DOINotFoundError
from paperqa.clients.utils import _get_with_retrying, clean_upbibtex, format_bibtex
from paperqa.types import PaperDetails

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API_FIELDS: str = ",".join([
    "citationStyles",
    "externalIds",
    "url",
    "openAccessPdf",
    "year",
    "isOpenAccess",
    "influentialCitationCount",
    "citationCount",
    "publicationDate",
    "journal",
    "publicationTypes"
    "title",
    "authors",
    "venue"
])
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org"


class SematicScholarSearchType(IntEnum):
    DEFAULT = auto()
    PAPER = auto()
    PAPER_RECOMMENDATIONS = auto()
    DOI = auto()
    FUTURE_CITATIONS = auto()
    PAST_REFERENCES = auto()
    GOOGLE = auto()
    MATCH = auto()

    def make_url_params(  # noqa: PLR0911
        self,
        params: dict[str, Any],
        query: str = "",
        offset: int = 0,
        limit: int = 1,
        include_base_url: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        """
        Make the target URL and in-place update the input URL parameters.

        Args:
            params: URL parameters to in-place update.
            query: Either a search query or a Semantic Scholar paper ID.
            offset: Offset to place in the URL parameters for the default search type.
            limit: Limit to place in the URL parameters for some search types.
            include_base_url: Set True (default) to include the base URL.

        Returns:
            Two-tuple of URL and URL parameters.
        """
        base = SEMANTIC_SCHOLAR_BASE_URL if include_base_url else ""
        if self == SematicScholarSearchType.DEFAULT:
            params["query"] = query.replace("-", " ")
            params["offset"] = offset
            params["limit"] = limit
            return f"{base}/graph/v1/paper/search", params
        if self == SematicScholarSearchType.PAPER:
            return f"{base}/graph/v1/paper/{query}", params
        if self == SematicScholarSearchType.PAPER_RECOMMENDATIONS:
            return f"{base}/recommendations/v1/papers/forpaper/{query}", params
        if self == SematicScholarSearchType.DOI:
            return f"{base}/graph/v1/paper/DOI:{query}", params
        if self == SematicScholarSearchType.FUTURE_CITATIONS:
            params["limit"] = limit
            return f"{base}/graph/v1/paper/{query}/citations", params
        if self == SematicScholarSearchType.PAST_REFERENCES:
            params["limit"] = limit
            return f"{base}/graph/v1/paper/{query}/references", params
        if self == SematicScholarSearchType.GOOGLE:
            params["limit"] = 1
            return f"{base}/graph/v1/paper/search", params
        if self == SematicScholarSearchType.MATCH:
            return f"{base}/graph/v1/paper/search/match", params
        raise NotImplementedError


def s2_authors_match(authors: list[str], data: dict):
    """Check if the authors in the data match the authors in the paper."""
    s2_authors_noinit = [
        " ".join([w for w in a["name"].split() if len(w) > 2])
        for a in data["authors"]
    ]
    authors_noinit = [
        " ".join([w for w in a.split() if len(w) > 2]) for a in authors
    ]
    # Note: we expect the number of authors to be possibly different
    return any(
        starmap(
            lambda x, y: x in y or y in x,
            zip(s2_authors_noinit, authors_noinit, strict=False),
        )
    )


async def parse_s2_to_paper_details(data: dict, session: aiohttp.ClientSession) -> PaperDetails:
    
    paper_data = data['data'][0]

    if "ArXiv" in paper_data['externalIds']:
        doi = "10.48550/arXiv." + paper_data['externalIds']["ArXiv"]
    elif "DOI" in paper_data['externalIds']:
        doi = paper_data['externalIds']['DOI']
    else:
        raise DOINotFoundError(f"Could not find DOI for {data}.")

    bibtex = await doi_to_bibtex(doi, session)

    # Parse publication date
    publication_date = None
    if paper_data.get('publicationDate'):
        publication_date = datetime.strptime(paper_data['publicationDate'], '%Y-%m-%d')

    # Create PaperDetails object
    paper_details = PaperDetails(
        citation=format_bibtex(bibtex),
        key=bibtex.split("{")[1].split(",")[0],
        bibtex=bibtex,
        authors=[author['name'] for author in paper_data.get('authors', [])],
        publication_date=publication_date,
        year=paper_data.get('year'),
        volume=paper_data.get('journal', {}).get('volume'),
        issue=None,  # Not provided in this JSON format
        pages=paper_data.get('journal', {}).get('pages'),
        journal=paper_data.get('journal', {}).get('name'),
        url=None,  # URL is not provided in this JSON format
        title=paper_data.get('title'),
        citation_count=paper_data.get('citationCount'),
        doi=paper_data.get('externalIds', {}).get('DOI'),
        other={}  # Initialize empty dict for other fields
    )

    # Add any additional fields to the 'other' dict
    for key, value in paper_data.items():
        if key not in paper_details.model_fields:
            paper_details.other[key] = value

    return paper_details

async def s2_title_search(
    title: str, authors: list[str], session: aiohttp.ClientSession
) -> str:
    """Reconcile DOI from Semantic Scholar - which only checks title. So we manually check authors."""
    endpoint, params = SematicScholarSearchType.MATCH.make_url_params(
        params={"query": title, "fields": SEMANTIC_SCHOLAR_API_FIELDS}
    )
    header = {}
    try:
        header["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    except KeyError:
        logger.warning(
            "SEMANTIC_SCHOLAR_API_KEY environment variable not set. Semantic Scholar API rate limits may apply."
        )
    async with session.get(url=endpoint, params=params, headers=header) as response:
        # check for 404 ( = no results)
        if response.status == 404:
            raise DOINotFoundError(f"Could not find DOI for {title}.")
        data = await response.json()

        if authors is not None and not s2_authors_match(authors, data["data"][0]):
            raise DOINotFoundError(
                f"Could not find DOI for {title} - author disagreement."
            )
        return parse_s2_to_paper_details(data, session)


async def get_paper_details_from_s2(
    title: str | None = None,
    doi: str | None = None,
    session: aiohttp.ClientSession | None = None,
) -> PaperDetails:
    """Get paper details from Semantic Scholar given a Semantic Scholar ID or a DOI."""
    if title is doi is None:
        raise ValueError("Either title or DOI must be provided.")
    if doi:
        end = f"DOI:{doi}"

    async def get_parse(session_: aiohttp.ClientSession) -> PaperDetails:
        details = await _get_with_retrying(
            url=f"{SEMANTIC_SCHOLAR_BASE_URL}/graph/v1/paper/{end}",
            params={"fields": SEMANTIC_SCHOLAR_API_FIELDS},
            session=session_,
            headers={'x-api-key': os.environ["SEMANTIC_SCHOLAR_API_KEY"]}
        )
        return parse_s2_to_paper_details(details, session_)

    if doi:
        return await get_parse(session)
    
    return await s2_title_search(title, session)