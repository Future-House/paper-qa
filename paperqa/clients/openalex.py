from __future__ import annotations

import json
import logging
import os
from collections.abc import Collection
from datetime import datetime
from typing import Any
from urllib.parse import quote

import aiohttp

from paperqa.types import DocDetails
from paperqa.utils import BIBTEX_MAPPING, mutate_acute_accents, strings_similarity

from .client_models import DOIOrTitleBasedProvider, DOIQuery, TitleAuthorQuery
from .exceptions import DOINotFoundError

OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_API_REQUEST_TIMEOUT = 5.0

logger = logging.getLogger(__name__)


# author_name will be FamilyName, GivenName Middle initial. (if available)
# there is no standalone "FamilyName" or "GivenName" fields
# this manually constructs the name into the format the other clients use
def reformat_name(name: str) -> str:
    if "," not in name:
        return name
    family, given_names = (x.strip() for x in name.split(",", maxsplit=1))
    # Return the reformatted name
    return f"{given_names} {family}"


def get_openalex_mailto() -> str | None:
    """Get the OpenAlex mailto address.

    Returns:
        The OpenAlex mailto address if available.
    """
    mailto_address = os.environ.get("OPENALEX_MAILTO")
    if mailto_address is None:
        logger.warning(
            "OPENALEX_MAILTO environment variable not set."
            " your request may be deprioritized by OpenAlex."
        )
    return os.environ.get("OPENALEX_MAILTO")


async def get_doc_details_from_openalex(
    session: aiohttp.ClientSession,
    doi: str | None = None,
    title: str | None = None,
    fields: Collection[str] | None = None,
    title_similarity_threshold: float = 0.75,
) -> DocDetails | None:
    """Get paper details from OpenAlex given a DOI or paper title.

    Args:
        session: The active session of the request.
        doi: The DOI of the paper.
        title: The title of the paper.
        fields: Specific fields to include in the request.
        title_similarity_threshold: The threshold for title similarity.

    Returns:
        The details of the document if found, otherwise None.

    Raises:
        ValueError: If neither DOI nor title is provided.
        DOINotFoundError: If the paper cannot be found.
    """
    mailto = get_openalex_mailto()
    params = {"mailto": mailto} if mailto else {}

    if doi is title is None:
        raise ValueError("Either a DOI or title must be provided.")

    url = f"{OPENALEX_BASE_URL}/works"
    if doi:
        # this looks wrong but it's now
        # will compile to a relative url similar to:
        # https://api.openalex.org/works/https://doi.org/10.7717/peerj.4375
        url += f"/https://doi.org/{quote(doi, safe='')}"
    elif title:
        params["filter"] = f"title.search:{title}"

    if fields:
        params["select"] = ",".join(fields)

    async with session.get(
        url,
        params=params,
        timeout=aiohttp.ClientTimeout(OPENALEX_API_REQUEST_TIMEOUT),
    ) as response:
        try:
            response.raise_for_status()
            response_data = await response.json()
        except (aiohttp.ClientResponseError, json.JSONDecodeError) as exc:
            raise DOINotFoundError("Could not find paper given DOI/title.") from exc

        if response_data.get("status") == "failed":
            raise DOINotFoundError(
                "OpenAlex API returned a failed status for the query."
            )

        results_data = response_data
        if params.get("filter") is not None:
            results_data = results_data["results"]
            if len(results_data) == 0:
                raise DOINotFoundError(
                    "OpenAlex API did not return any items for the query."
                )
            results_data = results_data[0]

        if (
            doi is None
            and title
            and strings_similarity(results_data.get("title", ""), title)
            < title_similarity_threshold
        ):
            raise DOINotFoundError(
                f"OpenAlex results did not match for title {title!r}."
            )
        if doi and results_data.get("doi") != doi:
            raise DOINotFoundError(f"DOI {doi!r} not found in OpenAlex.")

        return parse_openalex_to_doc_details(results_data)


def parse_openalex_to_doc_details(message: dict[str, Any]) -> DocDetails:
    """Parse OpenAlex API response to DocDetails.

    Args:
        message: The OpenAlex API response message.

    Returns:
        Parsed document details.
    """
    raw_author_names = [
        authorship.get("raw_author_name", "")
        for authorship in (message.get("authorships") or [])  # Handle None authorships
        if authorship
    ]
    sanitized_authors = [
        mutate_acute_accents(text=reformat_name(author), replace=True)
        for author in raw_author_names
    ]

    primary_location = message.get("primary_location") or {}
    source = primary_location.get("source") or {}

    publisher = source.get("host_organization_name", None)
    journal = source.get("display_name", None)
    issn = source.get("issn_l", None)
    volume = message.get("biblio", {}).get("volume", None)
    issue = message.get("biblio", {}).get("issue", None)
    pages = message.get("biblio", {}).get("last_page", None)
    doi = message.get("doi")
    title = message.get("title")
    citation_count = message.get("cited_by_count")
    publication_year = message.get("publication_year")

    best_oa_location = message.get("best_oa_location") or {}
    pdf_url = best_oa_location.get("pdf_url", None)
    oa_license = best_oa_location.get("license", None)

    publication_date_str = message.get("publication_date", "")
    try:
        publication_date = (
            datetime.fromisoformat(publication_date_str)
            if publication_date_str
            else None
        )
    except ValueError:
        publication_date = None

    bibtex_type = BIBTEX_MAPPING.get(message.get("type") or "other", "misc")

    return DocDetails(
        key=None,
        bibtex_type=bibtex_type,
        bibtex=None,
        authors=sanitized_authors,
        publication_date=publication_date,
        year=publication_year,
        volume=volume,
        issue=issue,
        publisher=publisher,
        issn=issn,
        pages=pages,
        journal=journal,
        url=doi,
        title=title,
        citation_count=citation_count,
        doi=doi,
        license=oa_license,
        pdf_url=pdf_url,
        other=message,
    )


class OpenAlexProvider(DOIOrTitleBasedProvider):
    """An open source provider of scholarly documents.

    Includes information on work, researchers, institutions, journals,
    and research topics.
    """

    async def get_doc_details(
        self, doi: str, session: aiohttp.ClientSession
    ) -> DocDetails | None:
        """Get document details by DOI.

        Args:
            doi: The DOI of the document.
            session: The active session of the request.

        Returns:
            The document details if found, otherwise None.
        """
        return await get_doc_details_from_openalex(doi=doi, session=session)

    async def search_by_title(
        self,
        query: str,
        session: aiohttp.ClientSession,
        title_similarity_threshold: float = 0.75,
    ) -> DocDetails | None:
        """Search for document details by title.

        Args:
            query: The title query for the document.
            session: The active session of the request.
            title_similarity_threshold: Threshold for title similarity.

        Returns:
            The document details if found, otherwise None.
        """
        return await get_doc_details_from_openalex(
            title=query,
            session=session,
            title_similarity_threshold=title_similarity_threshold,
        )

    async def _query(self, query: TitleAuthorQuery | DOIQuery) -> DocDetails | None:
        """Query the OpenAlex API via the provided DOI or title.

        Args:
            query: The query containing either a DOI or title.
                    DOI is prioritized over title.

        Returns:
            The document details if found, otherwise None.
        """
        if isinstance(query, DOIQuery):
            return await self.get_doc_details(doi=query.doi, session=query.session)
        return await self.search_by_title(
            query=query.title,
            session=query.session,
            title_similarity_threshold=query.title_similarity_threshold,
        )
