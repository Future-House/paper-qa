from __future__ import annotations

import csv
import datetime
import logging
import os

import aiohttp
from anyio import open_file
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from paperqa.types import DocDetails

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

        self.retraction_filter: str = "Retraction"
        self.doi_set: set[str] = set()
        self.columns: list[str] = [
            "RetractionDOI",
            "OriginalPaperDOI",
            "RetractionNature",
        ]

    def _has_cache_expired(self) -> bool:
        creation_time = os.path.getctime(self.retraction_data_path)
        file_creation_date = datetime.datetime.fromtimestamp(creation_time).replace(
            tzinfo=datetime.UTC
        )

        current_time = datetime.datetime.now(datetime.UTC)
        time_difference = current_time - file_creation_date

        return time_difference > datetime.timedelta(days=30)

    def _is_csv_cached(self) -> bool:
        return os.path.exists(self.retraction_data_path)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=5, min=5),
        reraise=True,
    )
    async def _download_retracted_dataset(self) -> None:

        if not (CROSSREF_MAILTO := os.getenv("CROSSREF_MAILTO")):
            CROSSREF_MAILTO = "test@example.com"
        url = f"https://api.labs.crossref.org/data/retractionwatch?{CROSSREF_MAILTO}"

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response,
        ):
            response.raise_for_status()

            logger.info(
                f"Retraction data was not cashed. Downloading retraction data from {url}..."
            )

            async with await open_file(self.retraction_data_path, "wb") as f:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    await f.write(chunk)

            if os.path.getsize(self.retraction_data_path) == 0:
                raise RuntimeError("Retraction data is empty")

    def _filter_dois(self) -> None:
        with open(self.retraction_data_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row[self.columns[2]] == self.retraction_filter:
                    self.doi_set.add(row[self.columns[0]])
                    self.doi_set.add(row[self.columns[1]])

    async def load_data(self) -> None:
        if not self._is_csv_cached() or self._has_cache_expired():
            await self._download_retracted_dataset()

        self._filter_dois()

        if not self.doi_set:
            raise RuntimeError("Retraction data was not found.")

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
            logger.debug(
                f"Must have a valid DOI to query retraction data:{doc_details.doi} "
            )
            return None
