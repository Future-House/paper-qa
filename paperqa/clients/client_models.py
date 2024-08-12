from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Collection, Generic, TypeVar

import aiohttp
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ..types import DocDetails


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
    def ensure_fields_are_present(cls, data: dict[str, Any]) -> dict[str, Any]:
        if (fields := data.get("fields")) and "doi" not in fields:
            fields.append("doi")
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

    @abstractmethod
    async def _query(self, query: DOIQuery | TitleAuthorQuery) -> DocDetails | None:
        pass

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
    """Post-process metadata from a query."""

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
