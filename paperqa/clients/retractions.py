from __future__ import annotations

import csv
import logging
import os
from typing import Any

from pydantic import ValidationError

from ..types import DocDetails
from .client_models import DOIQuery, MetadataPostProcessor

import asyncio
import datetime
from io import StringIO
from typing import ClassVar

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class RetrationDataPostProcessor(MetadataPostProcessor[DOIQuery]):
    def __init__(self, retraction_data_path: os.PathLike | str | None = None, 
                 retraction_filter: list[str] | None = None) -> None:

        self.RETRACTION_REASONS: dict[str, Any] = {
        "Retraction",
        "Expression of concern",
        "Reinstatement",
        "Correction",
    }
        if retraction_data_path is None:
            # Construct the path relative to module
            self.retraction_data_path = str(
                os.path.join(
                    os.path.dirname(__file__), "client_data", "retractions.csv"
                )
            )
        else:
            self.retraction_data_path = str(retraction_data_path)

        if retraction_filter is None:
            retraction_filter = ["Retraction"]

        if not all(type_ in self.RETRACTION_REASONS for type_ in retraction_filter):
            raise ValueError(
                f"Invalid retraction types in filter {retraction_filter}."
                f" Options are: {self.RETRACTION_REASONS}."
            )
        self.retraction_filter = retraction_filter
        self.doi_set: set[str] = set()
        self.columns: list[str] = [
            "Title",
            "RetractionDate",
            "RetractionDOI",
            "OriginalPaperDOI",
            "RetractionNature",
            "Reason",
        ]


    def _populate_retracted_dois(self, filtered_data: pd.DataFrame) -> set[str]:
        return set(filtered_data[self.columns[2]].dropna()) | set(
            filtered_data[self.columns[3]].dropna()
        )

    def _filter_dois(
        self, filter_cols: list, retraction_data: pd.DataFrame
    ) -> pd.DataFrame:
        return retraction_data[retraction_data["RetractionNature"].isin(filter_cols)]
    
    async def load_data(self) -> None:
        retraction_data = pd.read_csv(self.retraction_data_path)
        filtered_data = self._filter_dois(self.retraction_filter, retraction_data)
        self.doi_set = self._populate_retracted_dois(filtered_data)
    
    #TODO: download if cache_expired


    async def _process(
        self, query: DOIQuery, doc_details: DocDetails
    ) -> DocDetails:
        if not self.doi_set:
            self.load_data()
        
   
        return doc_details + DocDetails(  # type: ignore[call-arg]
            is_retracted= query.doi in self.doi_set  # type: ignore[union-attr]
        )

    def query_creator(self, doc_details: DocDetails, **kwargs) -> DOIQuery | None:
        try:
            return DOIQuery(doi=doc_details.doi, **kwargs)
        except ValidationError:
            logger.debug(
                "Must have a valid doi to query retraction data."
            )
            return None