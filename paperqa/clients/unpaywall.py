from __future__ import annotations

import os
from datetime import datetime
from http import HTTPStatus
from urllib.parse import quote

import aiohttp
from pydantic import BaseModel, ConfigDict, ValidationError

from paperqa.types import DocDetails
from paperqa.utils import _get_with_retrying, strings_similarity

from .client_models import DOIOrTitleBasedProvider, DOIQuery, TitleAuthorQuery
from .exceptions import DOINotFoundError

UNPAYWALL_BASE_URL = "https://api.unpaywall.org/v2/"
UNPAYWALL_TIMEOUT = float(os.environ.get("UNPAYWALL_TIMEOUT", "10.0"))  # seconds


class Author(BaseModel):
    family: str | None = None
    given: str | None = None
    sequence: str | None = None
    affiliation: list[dict[str, str]] | None = None
    model_config = ConfigDict(extra="allow")


class BestOaLocation(BaseModel):
    updated: datetime | None = None
    url: str | None = None
    url_for_pdf: str | None = None
    url_for_landing_page: str | None = None
    evidence: str | None = None
    license: str | None = None
    version: str | None = None
    host_type: str | None = None
    is_best: bool | None = None
    pmh_id: str | None = None
    endpoint_id: str | None = None
    repository_institution: str | None = None
    oa_date: str | None = None
    model_config = ConfigDict(extra="allow")


class UnpaywallResponse(BaseModel):
    doi: str
    doi_url: str | None = None
    title: str | None = None
    genre: str | None = None
    is_paratext: bool | None = None
    published_date: str | None = None
    year: int | None = None
    journal_name: str | None = None
    journal_issns: str | None = None
    journal_issn_l: str | None = None
    journal_is_oa: bool | None = None
    journal_is_in_doaj: bool | None = None
    publisher: str | None = None
    is_oa: bool
    oa_status: str | None = None
    has_repository_copy: bool | None = None
    best_oa_location: BestOaLocation | None = None
    updated: datetime | None = None
    z_authors: list[Author] | None = None


class SearchResponse(BaseModel):
    response: UnpaywallResponse
    score: float
    snippet: str


class SearchResults(BaseModel):
    results: list[SearchResponse]
    elapsed_seconds: float


class UnpaywallProvider(DOIOrTitleBasedProvider):

    async def get_doc_details(
        self, doi: str, session: aiohttp.ClientSession
    ) -> DocDetails:

        try:
            results = UnpaywallResponse(
                **(
                    await _get_with_retrying(
                        url=(
                            f"{UNPAYWALL_BASE_URL}{doi}"
                            f"?email={os.environ.get('UNPAYWALL_EMAIL', 'example@papercrow.ai')}"
                        ),
                        params={},
                        session=session,
                        timeout=UNPAYWALL_TIMEOUT,
                        http_exception_mappings={
                            HTTPStatus.NOT_FOUND: DOINotFoundError(
                                f"Unpaywall not find DOI for {doi}."
                            )
                        },
                    )
                )
            )
        except ValidationError as e:
            raise DOINotFoundError(
                f"Unpaywall results returned with a bad schema for DOI {doi!r}."
            ) from e

        return self._create_doc_details(results)

    async def search_by_title(
        self,
        query: str,
        session: aiohttp.ClientSession,
        title_similarity_threshold: float = 0.75,
    ) -> DocDetails:
        try:
            results = SearchResults(
                **(
                    await _get_with_retrying(
                        url=(
                            f"{UNPAYWALL_BASE_URL}search?query={quote(query)}"
                            f"&email={os.environ.get('UNPAYWALL_EMAIL', 'example@papercrow.ai')}"
                        ),
                        params={},
                        session=session,
                        timeout=UNPAYWALL_TIMEOUT,
                        http_exception_mappings={
                            HTTPStatus.NOT_FOUND: DOINotFoundError(
                                f"Could not find DOI for {query}."
                            )
                        },
                    )
                )
            ).results
        except ValidationError as e:
            raise DOINotFoundError(
                f"Unpaywall results returned with a bad schema for title {query!r}."
            ) from e

        if not results:
            raise DOINotFoundError(
                f"Unpaywall results did not match for title {query!r}."
            )

        details = self._create_doc_details(results[0].response)

        if (
            strings_similarity(
                details.title or "",
                query,
            )
            < title_similarity_threshold
        ):
            raise DOINotFoundError(
                f"Unpaywall results did not match for title {query!r}."
            )
        return details

    def _create_doc_details(self, data: UnpaywallResponse) -> DocDetails:
        return DocDetails(  # type: ignore[call-arg]
            authors=[
                f"{author.given} {author.family}" for author in (data.z_authors or [])
            ],
            publication_date=(
                None
                if not data.published_date
                else datetime.strptime(data.published_date, "%Y-%m-%d")
            ),
            year=data.year,
            journal=data.journal_name,
            publisher=data.publisher,
            url=None if not data.best_oa_location else data.best_oa_location.url,
            title=data.title,
            doi=data.doi,
            doi_url=data.doi_url,
            other={
                "genre": data.genre,
                "is_paratext": data.is_paratext,
                "journal_issns": data.journal_issns,
                "journal_issn_l": data.journal_issn_l,
                "journal_is_oa": data.journal_is_oa,
                "journal_is_in_doaj": data.journal_is_in_doaj,
                "is_oa": data.is_oa,
                "oa_status": data.oa_status,
                "has_repository_copy": data.has_repository_copy,
                "best_oa_location": (
                    None
                    if not data.best_oa_location
                    else data.best_oa_location.model_dump()
                ),
            },
        )

    async def _query(self, query: TitleAuthorQuery | DOIQuery) -> DocDetails | None:
        if isinstance(query, DOIQuery):
            return await self.get_doc_details(doi=query.doi, session=query.session)
        return await self.search_by_title(
            query=query.title,
            session=query.session,
            title_similarity_threshold=query.title_similarity_threshold,
        )
