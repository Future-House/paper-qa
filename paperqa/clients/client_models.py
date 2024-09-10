from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any, Generic, TypeVar

import aiohttp
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from paperqa.types import DocDetails

from .exceptions import DOINotFoundError

logger = logging.getLogger(__name__)


# ClientQuery is a base class for all queries to the client_models
class ClientQuery(BaseModel):
    session: aiohttp.ClientSession
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TitleAuthorQuery(ClientQuery):
    title: str
    authors: list[str] = []
    title_similarity_threshold: float = 0.75
    fields: Collection[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def ensure_fields_are_present(cls, data: dict[str, Any]) -> dict[str, Any]:
        if fields := data.get("fields"):
            if "doi" not in fields:
                fields.append("doi")
            if "title" not in fields:
                fields.append("title")
            if data.get("authors") is not None and "authors" not in fields:
                fields.append("authors")
            # ensure these are ranked the same for caching purposes
            data["fields"] = sorted(fields)
        return data

    @field_validator("title_similarity_threshold")
    @classmethod
    def zero_and_one(cls, v: float, info: ValidationInfo) -> float:  # noqa: ARG003
        if v < 0.0 or v > 1.0:
            raise ValueError(
                "title_similarity_threshold must be between 0 and 1. (inclusive)"
            )
        return v


class DOIQuery(ClientQuery):
    doi: str
    fields: Collection[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def add_doi_to_fields_and_validate(cls, data: dict[str, Any]) -> dict[str, Any]:

        if (fields := data.get("fields")) and "doi" not in fields:
            fields.append("doi")

        # sometimes the DOI has a URL prefix, remove it
        remove_urls = ["https://doi.org/", "http://dx.doi.org/"]
        for url in remove_urls:
            if data["doi"].startswith(url):
                data["doi"] = data["doi"].replace(url, "")

        return data


class JournalQuery(ClientQuery):
    journal: str


ClientQueryType = TypeVar("ClientQueryType", bound=ClientQuery)


class MetadataProvider(ABC, Generic[ClientQueryType]):
    """Provide metadata from a query by any means necessary."""

    async def query(self, query: dict) -> DocDetails | None:
        return await self._query(self.query_transformer(query))

    @abstractmethod
    async def _query(self, query: ClientQueryType) -> DocDetails | None:
        pass

    @abstractmethod
    def query_transformer(self, query: dict) -> ClientQueryType:
        pass


class DOIOrTitleBasedProvider(MetadataProvider[DOIQuery | TitleAuthorQuery]):

    async def query(self, query: dict) -> DocDetails | None:
        try:
            client_query = self.query_transformer(query)
            return await self._query(client_query)
        # We allow graceful failures, i.e. return "None" for both DOI errors and timeout errors
        # DOINotFoundError means the paper doesn't exist in the source, the timeout is to prevent
        # this service from failing us when it's down or slow.
        except DOINotFoundError:
            logger.warning(
                "Metadata not found for"
                f" {client_query.doi if isinstance(client_query, DOIQuery) else client_query.title} in"
                f" {self.__class__.__name__}."
            )
        except TimeoutError:
            logger.warning(
                f"Request to {self.__class__.__name__} for"
                f" {client_query.doi if isinstance(client_query, DOIQuery) else client_query.title} timed"
                " out."
            )
        return None

    @abstractmethod
    async def _query(self, query: DOIQuery | TitleAuthorQuery) -> DocDetails | None:
        """
        Query the source using either a DOI or title/author search.

        None should be returned if the DOI or title is not a good match.

        Raises:
            DOINotFoundError: This is when the DOI or title is not found in the sources
            TimeoutError: When the request takes too long on the client side
        """

    def query_transformer(self, query: dict) -> DOIQuery | TitleAuthorQuery:
        try:
            if "doi" in query:
                return DOIQuery(**query)
            if "title" in query:
                return TitleAuthorQuery(**query)
        except ValidationError as e:
            raise ValueError(
                f"Query {query} format not supported by {self.__class__.__name__}."
            ) from e

        raise ValueError("Provider query missing 'doi' or 'title' field.")


class MetadataPostProcessor(ABC, Generic[ClientQueryType]):
    """Post-process metadata from a query.

    MetadataPostProcessor should be idempotent and not order-dependent, i.e.
    all MetadataPostProcessor instances should be able to run in parallel.

    """

    async def process(self, doc_details: DocDetails, **kwargs) -> DocDetails:
        if query := self.query_creator(doc_details, **kwargs):
            return await self._process(query, doc_details)
        return doc_details

    @abstractmethod
    async def _process(
        self, query: ClientQueryType, doc_details: DocDetails
    ) -> DocDetails:
        pass

    @abstractmethod
    def query_creator(
        self, doc_details: DocDetails, **kwargs
    ) -> ClientQueryType | None:
        pass
