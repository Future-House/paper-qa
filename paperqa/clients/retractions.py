from __future__ import annotations

import asyncio
import datetime
import logging
import os
from io import StringIO

import aiohttp
import pandas as pd
from pydantic import ValidationError

from ..types import DocDetails
from .client_models import DOIQuery, MetadataPostProcessor

logger = logging.getLogger(__name__)


class RetrationDataPostProcessor(MetadataPostProcessor[DOIQuery]):
    def __init__(self, retraction_data_path: os.PathLike | str | None = None) -> None:

        if retraction_data_path is None:
            # Construct the path relative to module
            self.retraction_data_path = str(
                os.path.join(
                    os.path.dirname(__file__), "client_data", "retractions.csv"
                )
            )
        else:
            self.retraction_data_path = str(retraction_data_path)

        self.retraction_filter = ["Retraction"]
        self.doi_set: set[str] = set()
        self.columns: list[str] = [
            "Title",
            "RetractionDate",
            "RetractionDOI",
            "OriginalPaperDOI",
            "RetractionNature",
            "Reason",
        ]

    def _has_cache_expired(self) -> bool:
        creation_time = os.path.getctime(self.retraction_data_path)
        file_creation_date = datetime.datetime.fromtimestamp(creation_time).replace(
            tzinfo=datetime.UTC
        )

        current_time = datetime.datetime.now(datetime.UTC)
        time_difference = current_time - file_creation_date

        return time_difference > datetime.timedelta(days=30)

    async def _download_raw_retracted(self) -> pd.DataFrame:
        retries = 3
        delay = 5
        url = "https://api.labs.crossref.org/data/retractionwatch"
        for i in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    response = await session.get(
                        url, timeout=aiohttp.ClientTimeout(total=15)
                    )
                    async with response:
                        response.raise_for_status()
                        content = await response.text(encoding="mac_roman")
                        downloaded_df: pd.DataFrame = pd.read_csv(
                            StringIO(content), usecols=self.columns
                        )
                        return downloaded_df.apply(
                            lambda col: col.apply(
                                lambda x: x.strip() if isinstance(x, str) else x
                            )
                        )

            except (TimeoutError, aiohttp.ClientError) as e:  # noqa: PERF203
                if i < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise RuntimeError(
                        f"Failed to download retracted data after {retries} attempts: {e}"
                    ) from e
        return pd.DataFrame(columns=self.columns)  # NOTE: this is empty

    def _write_csv_to_gcs(self, retraction_dataframe: pd.DataFrame) -> None:
        retraction_dataframe.to_csv(self.retraction_data_path, index=False)

    def _populate_retracted_dois(self, filtered_data: pd.DataFrame) -> set[str]:
        return set(filtered_data[self.columns[2]].dropna()) | set(
            filtered_data[self.columns[3]].dropna()
        )

    def _filter_dois(
        self, filter_cols: list, retraction_data: pd.DataFrame
    ) -> pd.DataFrame:
        return retraction_data[retraction_data["RetractionNature"].isin(filter_cols)]

    async def load_data(self) -> None:
        if os.path.exists(self.retraction_data_path) and not self._has_cache_expired():
            retraction_data = pd.read_csv(self.retraction_data_path)
        else:
            retraction_data = await self._download_raw_retracted()
            if not retraction_data.empty:
                raise RuntimeError("Retraction data was not found.")
            self._write_csv_to_gcs(retraction_data)

        filtered_data = self._filter_dois(self.retraction_filter, retraction_data)
        self.doi_set = self._populate_retracted_dois(filtered_data)

    async def _process(self, query: DOIQuery, doc_details: DocDetails) -> DocDetails:
        if not self.doi_set:
            await self.load_data()

        return doc_details + DocDetails(  # type: ignore[call-arg]
            is_retracted=query.doi in self.doi_set
        )

    def query_creator(self, doc_details: DocDetails, **kwargs) -> DOIQuery | None:
        try:
            return DOIQuery(doi=doc_details.doi, **kwargs)
        except ValidationError:
            logger.debug("Must have a valid doi to query retraction data.")
            return None
