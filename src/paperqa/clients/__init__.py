from __future__ import annotations

import copy
import logging
from collections.abc import Awaitable, Collection, Coroutine, Sequence
from typing import Any, cast

import httpx
from lmi.utils import gather_with_concurrency
from pydantic import BaseModel, ConfigDict, Field

from paperqa.types import Doc, DocDetails

from .client_models import MetadataPostProcessor, MetadataProvider
from .crossref import CrossrefProvider
from .journal_quality import JournalQualityPostProcessor
from .openalex import OpenAlexProvider
from .retractions import RetractionDataPostProcessor
from .semantic_scholar import SemanticScholarProvider
from .unpaywall import UnpaywallProvider

logger = logging.getLogger(__name__)

# NOTE: we use tuple here so ordering is constant for unit testing's HTTP caching
DEFAULT_CLIENTS: Collection[type[MetadataPostProcessor | MetadataProvider]] = (
    CrossrefProvider,
    SemanticScholarProvider,
    JournalQualityPostProcessor,
)
ALL_CLIENTS: Collection[type[MetadataPostProcessor | MetadataProvider]] = (
    *DEFAULT_CLIENTS,
    OpenAlexProvider,
    UnpaywallProvider,
    RetractionDataPostProcessor,
)


class DocMetadataTask(BaseModel):
    """Simple container pairing metadata providers with processors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    providers: Collection[MetadataProvider] = Field(
        description=(
            "Metadata providers allotted to this task."
            " An example would be providers for Crossref and Semantic Scholar."
        )
    )
    processors: Collection[MetadataPostProcessor] = Field(
        description=(
            "Metadata post-processors allotted to this task."
            " An example would be a journal quality filter."
        )
    )

    def provider_queries(
        self, query: dict
    ) -> list[Coroutine[Any, Any, DocDetails | None]]:
        """Set up query coroutines for each contained metadata provider."""
        return [p.query(query) for p in self.providers]

    def processor_queries(
        self, doc_details: DocDetails, client: httpx.AsyncClient
    ) -> list[Coroutine[Any, Any, DocDetails]]:
        """Set up process coroutines for each contained metadata post-processor."""
        return [
            p.process(copy.copy(doc_details), client=client) for p in self.processors
        ]

    def __repr__(self) -> str:
        return (
            f"DocMetadataTask(providers={self.providers}, processors={self.processors})"
        )


class DocMetadataClient:
    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        metadata_clients: (
            Collection[type[MetadataPostProcessor | MetadataProvider]]
            | Sequence[Collection[type[MetadataPostProcessor | MetadataProvider]]]
        ) = DEFAULT_CLIENTS,
    ) -> None:
        """Metadata client for querying multiple metadata providers and processors.

        Args:
            http_client: Async HTTP client to allow for connection pooling.
            metadata_clients: list of MetadataProvider and MetadataPostProcessor classes to query;
                if nested, will query in order looking for termination criteria after each.
                Will terminate early if either DocDetails.is_hydration_needed is False OR if
                all requested fields are present in the DocDetails object.
        """
        self._http_client = http_client
        self.tasks: list[DocMetadataTask] = []

        # first see if we are nested; i.e. we want order
        if isinstance(metadata_clients, Sequence) and all(
            isinstance(sub_clients, Collection) for sub_clients in metadata_clients
        ):
            for sub_clients in metadata_clients:
                self.tasks.append(
                    DocMetadataTask(
                        providers=[
                            c if isinstance(c, MetadataProvider) else c()
                            for c in sub_clients
                            if (isinstance(c, type) and issubclass(c, MetadataProvider))
                            or isinstance(c, MetadataProvider)
                        ],
                        processors=[
                            c if isinstance(c, MetadataPostProcessor) else c()
                            for c in sub_clients
                            if (
                                isinstance(c, type)
                                and issubclass(c, MetadataPostProcessor)
                            )
                            or isinstance(c, MetadataPostProcessor)
                        ],
                    )
                )
        # otherwise, we are a flat collection
        if not self.tasks and all(
            not isinstance(c, Collection) for c in metadata_clients
        ):
            self.tasks.append(
                DocMetadataTask(
                    providers=[
                        c if isinstance(c, MetadataProvider) else c()  # type: ignore[redundant-expr]
                        for c in metadata_clients
                        if (isinstance(c, type) and issubclass(c, MetadataProvider))
                        or isinstance(c, MetadataProvider)
                    ],
                    processors=[
                        c if isinstance(c, MetadataPostProcessor) else c()  # type: ignore[redundant-expr]
                        for c in metadata_clients
                        if (
                            isinstance(c, type) and issubclass(c, MetadataPostProcessor)
                        )
                        or isinstance(c, MetadataPostProcessor)
                    ],
                )
            )

        if not self.tasks or (self.tasks and not self.tasks[0].providers):
            raise ValueError("At least one MetadataProvider must be provided.")

    async def query(self, **kwargs) -> DocDetails | None:

        client = (
            httpx.AsyncClient(timeout=10.0)
            if self._http_client is None
            else self._http_client
        )

        query_args = kwargs if "client" in kwargs else kwargs | {"client": client}

        all_doc_details: DocDetails | None = None

        for ti, task in enumerate(self.tasks):

            logger.debug(
                f"Attempting to populate metadata query: {query_args} via {task}"
            )

            # first query all client_models and aggregate the results
            doc_details = (
                sum(
                    p
                    for p in (
                        await gather_with_concurrency(
                            len(task.providers), task.provider_queries(query_args)
                        )
                    )
                    if p
                )
                or None
            )
            # then process and re-aggregate the results
            if doc_details and task.processors:
                doc_details = (
                    sum(
                        await gather_with_concurrency(
                            len(task.processors),
                            task.processor_queries(doc_details, client),
                        )
                    )
                    or None
                )

            if doc_details:
                # abuse int handling in __add__ for empty all_doc_details, None types won't work
                all_doc_details = doc_details + (all_doc_details or 0)

                if not all_doc_details.is_hydration_needed(
                    inclusion=kwargs.get("fields", [])
                ):
                    logger.debug(
                        "All requested fields are present in the DocDetails "
                        f"object{', stopping early.' if ti != len(self.tasks) - 1 else '.'}"
                    )
                    break

        if self._http_client is None:
            await client.aclose()

        return all_doc_details

    async def bulk_query(
        self, queries: Collection[dict[str, Any]], concurrency: int = 10
    ) -> list[DocDetails]:
        return await gather_with_concurrency(
            concurrency,
            [cast("Awaitable[DocDetails]", self.query(**kwargs)) for kwargs in queries],
        )

    async def upgrade_doc_to_doc_details(self, doc: Doc, **kwargs) -> DocDetails:

        # note we have some extra fields which may have come from reading the doc text,
        # but aren't in the doc object, we add them here too.
        extra_fields = {
            k: v for k, v in kwargs.items() if k in set(DocDetails.model_fields)
        }
        # abuse our doc_details object to be an int if it's empty
        # our __add__ operation supports int by doing nothing
        extra_doc: int | DocDetails = (
            0 if not extra_fields else DocDetails(**extra_fields)
        )

        if doc_details := await self.query(**kwargs):

            # hard overwrite the details from the prior object
            if "dockey" in doc.fields_to_overwrite_from_metadata:
                doc_details.dockey = doc.dockey
            if "doc_id" in doc.fields_to_overwrite_from_metadata:
                doc_details.doc_id = doc.dockey
            if "docname" in doc.fields_to_overwrite_from_metadata:
                doc_details.docname = doc.docname
            if "key" in doc.fields_to_overwrite_from_metadata:
                doc_details.key = doc.docname
            if "citation" in doc.fields_to_overwrite_from_metadata:
                doc_details.citation = doc.citation
            return extra_doc + doc_details

        # if we can't get metadata, just return the doc, but don't overwrite any fields
        prior_doc = doc.model_dump()
        prior_doc["fields_to_overwrite_from_metadata"] = set()
        return DocDetails(**(prior_doc | extra_fields))
