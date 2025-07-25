from __future__ import annotations

import csv
import logging
import os
from typing import Any, ClassVar

from pydantic import ValidationError

from paperqa.types import DocDetails

from .client_models import JournalQuery, MetadataPostProcessor

logger = logging.getLogger(__name__)


# TODO: refresh script for journal quality data


class JournalQualityPostProcessor(MetadataPostProcessor[JournalQuery]):

    # these will be deleted from any journal names before querying
    CASEFOLD_PHRASES_TO_REMOVE: ClassVar[list[str]] = ["amp;"]

    def __init__(self, journal_quality_path: os.PathLike | str | None = None) -> None:
        if journal_quality_path is None:
            # Construct the path relative to module
            self.journal_quality_path = str(
                os.path.join(
                    os.path.dirname(__file__), "client_data", "journal_quality.csv"
                )
            )
        else:
            self.journal_quality_path = str(journal_quality_path)
        self.data: dict[str, Any] | None = None

    def load_data(self) -> None:
        self.data = {}
        with open(self.journal_quality_path, newline="", encoding="utf-8") as csvfile:
            for row in csv.DictReader(csvfile):
                self.data[row["clean_name"]] = int(row["quality"])

    async def _process(
        self, query: JournalQuery, doc_details: DocDetails
    ) -> DocDetails:
        if not self.data:
            self.load_data()

        # TODO: not super scalable, but unless we need more than this we can just grugbrain
        journal_query = query.journal.casefold()
        for phrase in self.CASEFOLD_PHRASES_TO_REMOVE:
            journal_query = journal_query.replace(phrase, "")

        # docname can be blank since the validation will add it
        # remember, if both have docnames (i.e. key) they are
        # wiped and re-generated with resultant data
        return doc_details + DocDetails(
            doc_id=doc_details.doc_id,  # ensure doc_id is preserved
            dockey=doc_details.dockey,  # ensure dockey is preserved
            source_quality=max(
                self.data.get(journal_query, DocDetails.UNDEFINED_JOURNAL_QUALITY),  # type: ignore[union-attr]
                self.data.get("the " + journal_query, DocDetails.UNDEFINED_JOURNAL_QUALITY),  # type: ignore[union-attr]
                self.data.get(journal_query.replace("&", "and"), DocDetails.UNDEFINED_JOURNAL_QUALITY),  # type: ignore[union-attr]
            ),
        )

    def query_creator(self, doc_details: DocDetails, **kwargs) -> JournalQuery | None:
        try:
            return JournalQuery(journal=doc_details.journal, **kwargs)
        except ValidationError:
            logger.debug(
                "Must have a valid journal name to query journal quality data."
            )
            return None
