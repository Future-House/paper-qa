from __future__ import annotations

import csv
import logging
import os
from typing import Any

from pydantic import ValidationError

from paperqa.types import DocDetails

from .client_models import JournalQuery, MetadataPostProcessor

logger = logging.getLogger(__name__)


# TODO: refresh script for journal quality data


class JournalQualityPostProcessor(MetadataPostProcessor[JournalQuery]):
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
        # docname can be blank since the validation will add it
        # remember, if both have docnames (i.e. key) they are
        # wiped and re-generated with resultant data
        return doc_details + DocDetails(  # type: ignore[call-arg]
            source_quality=max(
                [
                    self.data.get(query.journal.casefold(), DocDetails.UNDEFINED_JOURNAL_QUALITY),  # type: ignore[union-attr]
                    self.data.get("the " + query.journal.casefold(), DocDetails.UNDEFINED_JOURNAL_QUALITY),  # type: ignore[union-attr]
                ]
            )
        )

    def query_creator(self, doc_details: DocDetails, **kwargs) -> JournalQuery | None:
        try:
            return JournalQuery(journal=doc_details.journal, **kwargs)
        except ValidationError:
            logger.debug(
                "Must have a valid journal name to query journal quality data."
            )
            return None
