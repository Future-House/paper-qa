from __future__ import annotations

import logging
import os
from collections.abc import Collection
from datetime import datetime
from enum import IntEnum, auto
from http import HTTPStatus
from itertools import starmap
from typing import Any

import aiohttp
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt

from paperqa.types import DocDetails
from paperqa.utils import (
    _get_with_retrying,
    clean_upbibtex,
    strings_similarity,
    union_collections_to_ordered_list,
)

from .client_models import DOIOrTitleBasedProvider, DOIQuery, TitleAuthorQuery
from .crossref import doi_to_bibtex
from .exceptions import DOINotFoundError, make_flaky_ssl_error_predicate

logger = logging.getLogger(__name__)

# map from S2 fields to those in the DocDetails model
# allows users to specify which fields to include in the response
SEMANTIC_SCHOLAR_API_MAPPING: dict[str, Collection[str]] = {
    "title": {"title"},
    "doi": {"externalIds"},
    "authors": {"authors"},
    "publication_date": {"publicationDate"},
    "year": {"year"},
    "volume": {"journal"},
    "pages": {"journal"},
    "journal": {"journal"},
    "url": {"url", "openAccessPdf"},
    "bibtex": {"citationStyles"},
    "doi_url": {"url"},
    "other": {"isOpenAccess", "influentialCitationCount", "publicationTypes", "venue"},
    "citation_count": {"citationCount"},
    "source_quality": {"journal"},
}
SEMANTIC_SCHOLAR_API_REQUEST_TIMEOUT = 10.0
SEMANTIC_SCHOLAR_API_FIELDS: str = ",".join(
    union_collections_to_ordered_list(SEMANTIC_SCHOLAR_API_MAPPING.values())
)
SEMANTIC_SCHOLAR_HOST = "api.semanticscholar.org"
SEMANTIC_SCHOLAR_BASE_URL = f"https://{SEMANTIC_SCHOLAR_HOST}"
SEMANTIC_SCHOLAR_HEADER_KEY = "x-api-key"


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


@retry(
    retry=retry_if_exception(make_flaky_ssl_error_predicate(SEMANTIC_SCHOLAR_HOST)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    stop=stop_after_attempt(5),
)
async def _s2_get_with_retrying(url: str, **get_kwargs) -> dict[str, Any]:
    return await _get_with_retrying(
        url=url,
        headers=get_kwargs.get("headers") or semantic_scholar_headers(),
        timeout=(
            get_kwargs.get("timeout")
            or aiohttp.ClientTimeout(SEMANTIC_SCHOLAR_API_REQUEST_TIMEOUT)
        ),
        **get_kwargs,
    )


def s2_authors_match(authors: list[str], data: dict) -> bool:
    """Check if the authors in the data match the authors in the paper."""
    AUTHOR_NAME_MIN_LENGTH = 2
    s2_authors_noinit = [
        " ".join([w for w in a["name"].split() if len(w) > AUTHOR_NAME_MIN_LENGTH])
        for a in data["authors"]
    ]
    authors_noinit = [
        " ".join([w for w in a.split() if len(w) > AUTHOR_NAME_MIN_LENGTH])
        for a in authors
    ]
    # Note: we expect the number of authors to be possibly different
    return any(
        starmap(
            lambda x, y: x in y or y in x,
            zip(s2_authors_noinit, authors_noinit, strict=False),
        )
    )


async def parse_s2_to_doc_details(
    paper_data: dict[str, Any], session: aiohttp.ClientSession
) -> DocDetails:

    bibtex_source = "self_generated"

    if "data" in paper_data:
        paper_data = paper_data["data"][0]

    # ArXiV check goes 1st to override another DOI
    if "ArXiv" in paper_data["externalIds"]:
        doi = "10.48550/arXiv." + paper_data["externalIds"]["ArXiv"]
    elif "DOI" in paper_data["externalIds"]:
        doi = paper_data["externalIds"]["DOI"]
    else:
        raise DOINotFoundError(f"Could not find DOI for {paper_data}.")

    # Should we give preference to auto-generation?
    if not (
        bibtex := clean_upbibtex(paper_data.get("citationStyles", {}).get("bibtex", ""))
    ):
        try:
            bibtex = await doi_to_bibtex(doi, session)
            bibtex_source = "crossref"
        except DOINotFoundError:
            bibtex = None
    else:
        bibtex_source = "semantic_scholar"

    publication_date = None
    if paper_data.get("publicationDate"):
        publication_date = datetime.strptime(paper_data["publicationDate"], "%Y-%m-%d")

    journal_data = paper_data.get("journal") or {}

    doc_details = DocDetails(  # type: ignore[call-arg]
        key=None if not bibtex else bibtex.split("{")[1].split(",")[0],
        bibtex_type="article",  # s2 should be basically all articles
        bibtex=bibtex,
        authors=[author["name"] for author in paper_data.get("authors", [])],
        publication_date=publication_date,
        year=paper_data.get("year"),
        volume=journal_data.get("volume"),
        pages=journal_data.get("pages"),
        journal=journal_data.get("name"),
        url=(paper_data.get("openAccessPdf") or {}).get("url"),
        title=paper_data.get("title"),
        citation_count=paper_data.get("citationCount"),
        doi=doi,
        other={},  # Initialize empty dict for other fields
    )

    # Add any additional fields to the 'other' dict
    for key, value in (
        paper_data
        | {"client_source": ["semantic_scholar"], "bibtex_source": [bibtex_source]}
    ).items():
        if key not in doc_details.model_fields:
            doc_details.other[key] = value

    return doc_details


def semantic_scholar_headers() -> dict[str, str]:
    """Semantic Scholar API key if available, otherwise nothing."""
    if api_key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        return {SEMANTIC_SCHOLAR_HEADER_KEY: api_key}
    logger.warning(
        "SEMANTIC_SCHOLAR_API_KEY environment variable not set. Semantic Scholar API"
        " rate limits may apply."
    )
    return {}


async def s2_title_search(
    title: str,
    session: aiohttp.ClientSession,
    authors: list[str] | None = None,
    title_similarity_threshold: float = 0.75,
    fields: str = SEMANTIC_SCHOLAR_API_FIELDS,
) -> DocDetails:
    """Reconcile DOI from Semantic Scholar - which only checks title. So we manually check authors."""
    if authors is None:
        authors = []
    endpoint, params = SematicScholarSearchType.MATCH.make_url_params(
        params={"query": title, "fields": fields}
    )

    data = await _s2_get_with_retrying(
        url=endpoint,
        params=params,
        session=session,
        http_exception_mappings={
            HTTPStatus.NOT_FOUND: DOINotFoundError(f"Could not find DOI for {title}.")
        },
    )
    try:
        if authors and not s2_authors_match(authors, data=data["data"][0]):
            raise DOINotFoundError(
                f"Could not find DOI for {title} - author disagreement."
            )
    except KeyError as exc:  # Very rare, but "data" may not be in data
        raise DOINotFoundError(
            f"Unexpected Semantic Scholar search/match endpoint shape for {title}"
            f" given data {data}."
        ) from exc
    # need to check if nested under a 'data' key or not (depends on filtering)
    if (
        strings_similarity(
            data.get("title", "") if "data" not in data else data["data"][0]["title"],
            title,
        )
        < title_similarity_threshold
    ):
        raise DOINotFoundError(
            f"Semantic scholar results did not match for title {title!r}."
        )
    return await parse_s2_to_doc_details(data, session)


@retry(
    retry=retry_if_exception(make_flaky_ssl_error_predicate(SEMANTIC_SCHOLAR_HOST)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    stop=stop_after_attempt(5),
)
async def get_s2_doc_details_from_doi(
    doi: str | None,
    session: aiohttp.ClientSession,
    fields: Collection[str] | None = None,
) -> DocDetails:
    """Get paper details from Semantic Scholar given a DOI."""
    # should always be string, runtime error catch
    if doi is None:
        raise ValueError("Valid DOI must be provided.")

    if fields:
        s2_fields = ",".join(
            union_collections_to_ordered_list(
                SEMANTIC_SCHOLAR_API_MAPPING[f]
                for f in fields
                if f in SEMANTIC_SCHOLAR_API_MAPPING
            )
        )
    else:
        s2_fields = SEMANTIC_SCHOLAR_API_FIELDS

    return await parse_s2_to_doc_details(
        paper_data=await _s2_get_with_retrying(
            url=f"{SEMANTIC_SCHOLAR_BASE_URL}/graph/v1/paper/DOI:{doi}",
            params={"fields": s2_fields},
            session=session,
            http_exception_mappings={
                HTTPStatus.NOT_FOUND: DOINotFoundError(f"Could not find DOI for {doi}.")
            },
        ),
        session=session,
    )


async def get_s2_doc_details_from_title(
    title: str | None,
    session: aiohttp.ClientSession,
    authors: list[str] | None = None,
    fields: Collection[str] | None = None,
    title_similarity_threshold: float = 0.75,
) -> DocDetails:
    """Get paper details from Semantic Scholar given a title.

    Optionally match against authors if provided.
    """
    if title is None:
        raise ValueError("Valid title must be provided.")
    if authors is None:
        authors = []
    if fields:
        s2_fields = ",".join(
            union_collections_to_ordered_list(
                SEMANTIC_SCHOLAR_API_MAPPING[f]
                for f in fields
                if f in SEMANTIC_SCHOLAR_API_MAPPING
            )
        )
    else:
        s2_fields = SEMANTIC_SCHOLAR_API_FIELDS

    return await s2_title_search(
        title,
        authors=authors,
        session=session,
        title_similarity_threshold=title_similarity_threshold,
        fields=s2_fields,
    )


class SemanticScholarProvider(DOIOrTitleBasedProvider):
    async def _query(self, query: TitleAuthorQuery | DOIQuery) -> DocDetails | None:
        if isinstance(query, DOIQuery):
            return await get_s2_doc_details_from_doi(
                doi=query.doi, session=query.session, fields=query.fields
            )
        return await get_s2_doc_details_from_title(
            title=query.title,
            authors=query.authors,
            session=query.session,
            title_similarity_threshold=query.title_similarity_threshold,
            fields=query.fields,
        )
