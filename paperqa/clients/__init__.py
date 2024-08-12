from __future__ import annotations

import copy
import logging
from typing import Any, Collection, Coroutine

import aiohttp

from ..types import Doc, DocDetails
from ..utils import gather_with_concurrency
from .client_models import MetadataPostProcessor, MetadataProvider
from .crossref import CrossrefProvider
from .journal_quality import JournalQualityPostProcessor
from .semantic_scholar import SemanticScholarProvider

logger = logging.getLogger(__name__)

ALL_CLIENTS: Collection[type[MetadataPostProcessor | MetadataProvider]] = {
    CrossrefProvider,
    SemanticScholarProvider,
    JournalQualityPostProcessor,
}


class DocMetadataClient:
    def __init__(
        self,
        session: aiohttp.ClientSession | None,
        clients: Collection[
            type[MetadataPostProcessor | MetadataProvider]
        ] = ALL_CLIENTS,
    ) -> None:
        self.session = aiohttp.ClientSession() if session is None else session
        self.providers = [c() for c in clients if issubclass(c, MetadataProvider)]
        self.processors = [c() for c in clients if issubclass(c, MetadataPostProcessor)]
        if not self.providers:
            raise ValueError("At least one MetadataProvider must be provided.")

    def provider_queries(
        self, query: dict
    ) -> list[Coroutine[Any, Any, DocDetails | None]]:
        return [p.query(query) for p in self.providers]

    def processor_queries(
        self, doc_details: DocDetails
    ) -> list[Coroutine[Any, Any, DocDetails]]:
        return [
            p.process(copy.copy(doc_details), session=self.session)
            for p in self.processors
        ]

    async def query(self, **kwargs) -> DocDetails | None:

        query_args = (
            kwargs if "session" in kwargs else kwargs | {"session": self.session}
        )

        # first query all client_models and aggregate the results
        doc_details: DocDetails | None = (
            sum(
                p
                for p in (
                    await gather_with_concurrency(
                        len(self.providers), self.provider_queries(query_args)
                    )
                )
                if p
            )
            or None
        )

        # then process and re-aggregate the results
        if doc_details and self.processors:
            doc_details = sum(
                await gather_with_concurrency(
                    len(self.processors), self.processor_queries(doc_details)
                )
            )

        return doc_details

    async def bulk_query(
        self, queries: Collection[dict[str, Any]], concurrency: int = 10
    ) -> list[DocDetails]:
        return await gather_with_concurrency(
            concurrency, [self.query(**kwargs) for kwargs in queries]
        )

    async def upgrade_doc_to_doc_details(self, doc: Doc, **kwargs) -> DocDetails:
        if doc_details := await self.query(**kwargs):
            if doc.overwrite_fields_from_metadata:
                return doc_details
            # hard overwrite the details from the prior object
            doc_details.dockey = doc.dockey
            doc_details.doc_id = doc.dockey
            doc_details.docname = doc.docname
            doc_details.key = doc.docname
            doc_details.citation = doc.citation
            return doc_details

        # if we can't get metadata, just return the doc, but don't overwrite any fields
        prior_doc = doc.model_dump()
        prior_doc["overwrite_fields_from_metadata"] = False
        return DocDetails(**prior_doc)
