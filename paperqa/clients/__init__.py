from __future__ import annotations

import copy
import logging
from collections.abc import Collection, Coroutine, Sequence
from typing import Any

import aiohttp
from pydantic import BaseModel, ConfigDict

from paperqa.types import Doc, DocDetails
from paperqa.utils import gather_with_concurrency

from .client_models import MetadataPostProcessor, MetadataProvider
from .crossref import CrossrefProvider
from .journal_quality import JournalQualityPostProcessor
from .retractions import RetractionDataPostProcessor
from .semantic_scholar import SemanticScholarProvider
from .unpaywall import UnpaywallProvider

logger = logging.getLogger(__name__)

DEFAULT_CLIENTS: Collection[type[MetadataPostProcessor | MetadataProvider]] = {
    CrossrefProvider,
    SemanticScholarProvider,
    JournalQualityPostProcessor,
}

ALL_CLIENTS: Collection[type[MetadataPostProcessor | MetadataProvider]] = {
    *DEFAULT_CLIENTS,
    UnpaywallProvider,
    RetractionDataPostProcessor,
}


class DocMetadataTask(BaseModel):
    """Holder for provider and processor tasks."""

    providers: Collection[MetadataProvider]
    processors: Collection[MetadataPostProcessor]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def provider_queries(
        self, query: dict
    ) -> list[Coroutine[Any, Any, DocDetails | None]]:
        return [p.query(query) for p in self.providers]

    def processor_queries(
        self, doc_details: DocDetails, session: aiohttp.ClientSession
    ) -> list[Coroutine[Any, Any, DocDetails]]:
        return [
            p.process(copy.copy(doc_details), session=session) for p in self.processors
        ]

    def __repr__(self) -> str:
        return (
            f"DocMetadataTask(providers={self.providers}, processors={self.processors})"
        )


class DocMetadataClient:
    def __init__(  # pylint: disable=dangerous-default-value
        self,
        session: aiohttp.ClientSession | None = None,
        clients: (
            Collection[type[MetadataPostProcessor | MetadataProvider]]
            | Sequence[Collection[type[MetadataPostProcessor | MetadataProvider]]]
        ) = DEFAULT_CLIENTS,
    ) -> None:
        """Metadata client for querying multiple metadata providers and processors.

        Args:
            session: outer scope aiohttp session to allow for connection pooling
            clients: list of MetadataProvider and MetadataPostProcessor classes to query;
                if nested, will query in order looking for termination criteria after each.
                Will terminate early if either DocDetails.is_hydration_needed is False OR if
                all requested fields are present in the DocDetails object.

        """
        self._session = session
        self.tasks: list[DocMetadataTask] = []

        # first see if we are nested; i.e. we want order
        if isinstance(clients, Sequence) and all(
            isinstance(sub_clients, Collection) for sub_clients in clients
        ):
            for sub_clients in clients:
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
        if not self.tasks and all(not isinstance(c, Collection) for c in clients):
            self.tasks.append(
                DocMetadataTask(
                    providers=[
                        c if isinstance(c, MetadataProvider) else c()  # type: ignore[redundant-expr]
                        for c in clients
                        if (isinstance(c, type) and issubclass(c, MetadataProvider))
                        or isinstance(c, MetadataProvider)
                    ],
                    processors=[
                        c if isinstance(c, MetadataPostProcessor) else c()  # type: ignore[redundant-expr]
                        for c in clients
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

        session = aiohttp.ClientSession() if self._session is None else self._session

        query_args = kwargs if "session" in kwargs else kwargs | {"session": session}

        doc_details: DocDetails | None = None

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
                doc_details = sum(
                    await gather_with_concurrency(
                        len(task.processors),
                        task.processor_queries(doc_details, session),
                    )
                )

            if doc_details and not doc_details.is_hydration_needed(
                inclusion=kwargs.get("fields", [])
            ):
                logger.debug(
                    "All requested fields are present in the DocDetails "
                    f"object{', stopping early.' if ti != len(self.tasks) - 1 else '.'}"
                )
                break

        if self._session is None:
            await session.close()

        return doc_details

    async def bulk_query(
        self, queries: Collection[dict[str, Any]], concurrency: int = 10
    ) -> list[DocDetails]:
        return await gather_with_concurrency(
            concurrency, [self.query(**kwargs) for kwargs in queries]
        )

    async def upgrade_doc_to_doc_details(self, doc: Doc, **kwargs) -> DocDetails:

        # note we have some extra fields which may have come from reading the doc text,
        # but aren't in the doc object, we add them here too.
        extra_fields = {
            k: v for k, v in kwargs.items() if k in {"title", "authors", "doi"}
        }
        # abuse our doc_details object to be an int if it's empty
        # our __add__ operation supports int by doing nothing
        extra_doc: int | DocDetails = (
            0 if not extra_fields else DocDetails(**extra_fields)
        )

        if doc_details := await self.query(**kwargs):
            if doc.overwrite_fields_from_metadata:
                return extra_doc + doc_details

            # hard overwrite the details from the prior object
            doc_details.dockey = doc.dockey
            doc_details.doc_id = doc.dockey
            doc_details.docname = doc.docname
            doc_details.key = doc.docname
            doc_details.citation = doc.citation
            return extra_doc + doc_details

        # if we can't get metadata, just return the doc, but don't overwrite any fields
        prior_doc = doc.model_dump()
        prior_doc["overwrite_fields_from_metadata"] = False
        return DocDetails(**(prior_doc | extra_fields))
