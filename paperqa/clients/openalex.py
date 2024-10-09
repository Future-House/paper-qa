from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Collection
from datetime import datetime
from typing import Any
from urllib.parse import quote

import aiohttp

from paperqa.types import DocDetails
from paperqa.utils import convert_acutes, strings_similarity

from .client_models import DOIOrTitleBasedProvider, DOIQuery, TitleAuthorQuery
from .exceptions import DOINotFoundError
from .shared_dicts import BIBTEX_MAPPING

OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_API_REQUEST_TIMEOUT = 5.0
logger = logging.getLogger(__name__)


async def get_openalex_mailto() -> str | None:
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
    mailto = await get_openalex_mailto()
    params = {"mailto": mailto} if mailto else {}

    if doi is None and title is None:
        raise ValueError("Either a DOI or title must be provided.")

    url = f"{OPENALEX_BASE_URL}/works"
    if doi:
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
            raise DOINotFoundError("DOI not found in OpenAlex")

        return await parse_openalex_to_doc_details(results_data)


async def parse_openalex_to_doc_details(message: dict[str, Any]) -> DocDetails:
    """Parse OpenAlex API response to DocDetails.

    Args:
        message: The OpenAlex API response message.

    Returns:
        Parsed document details.
    """

    # author_name will be FamilyName, GivenName Middle initial. (if available)
    # there is no standalone "FamilyName" or "GivenName" fields
    # this manually constructs the name into the format the other clients use
    def reformat_name(name: str) -> str:
        # https://regex101.com/r/74vR57/1
        pattern = r"^([^,]+),\s*(.+?)(?:\s+(\w+\.?))?$"
        match = re.match(pattern, name)
        if match:
            family_name, given_name, middle = match.groups()

            family_name = family_name.strip()
            given_name = given_name.strip()

            reformatted = f"{given_name}"
            if middle:
                reformatted += f" {middle.strip()}"
            reformatted += f" {family_name}"
            return reformatted.strip()

        return name

    authors = [
        authorship.get("raw_author_name")
        for authorship in message.get("authorships", [])
    ]
    authors = [reformat_name(author) for author in authors]
    sanitized_authors = [convert_acutes(author) for author in authors]

    publisher = (
        message.get("primary_location", {})
        .get("source", {})
        .get("host_organization_name")
    )
    journal = message.get("primary_location", {}).get("source", {}).get("display_name")

    return DocDetails(  # type: ignore[call-arg]
        key=None,
        bibtex_type=BIBTEX_MAPPING.get(message.get("type", "other"), "misc"),
        bibtex=None,
        authors=sanitized_authors,
        publication_date=datetime.fromisoformat(message.get("publication_date", "")),
        year=message.get("publication_year"),
        volume=message.get("biblio", {}).get("volume"),
        issue=message.get("biblio", {}).get("issue"),
        publisher=publisher,
        issn=message.get("primary_location", {}).get("source", {}).get("issn_l"),
        pages=message.get("biblio", {}).get("last_page"),
        journal=journal,
        url=message.get("doi"),
        title=message.get("title"),
        citation_count=message.get("cited_by_count"),
        doi=message.get("doi"),
        other=message,
    )


class OpenAlexProvider(DOIOrTitleBasedProvider):
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
