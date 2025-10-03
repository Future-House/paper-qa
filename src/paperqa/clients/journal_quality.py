from __future__ import annotations

import asyncio
import csv
import logging
import os
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import anyio
import httpx
import httpx_aiohttp
from pydantic import ValidationError
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from paperqa.types import DocDetails

from .client_models import JournalQuery, MetadataPostProcessor

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_QUALITY_CSV_PATH = (
    Path(__file__).parent / "client_data" / "journal_quality.csv"
)

# TODO: refresh script for journal quality data


class JournalQualityPostProcessor(MetadataPostProcessor[JournalQuery]):

    # these will be deleted from any journal names before querying
    CASEFOLD_PHRASES_TO_REMOVE: ClassVar[list[str]] = ["amp;"]

    def __init__(self, journal_quality_path: os.PathLike | str | None = None) -> None:
        if journal_quality_path is None:
            # Construct the path relative to module
            self.journal_quality_path: str | os.PathLike = (
                DEFAULT_JOURNAL_QUALITY_CSV_PATH
            )
        else:
            self.journal_quality_path = journal_quality_path
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


# SEE: https://en.wikipedia.org/wiki/JUFO
JUFO_PORTAL_DOWNLOAD_QUALITY_URL = (
    "https://jfp.csc.fi/jufoportal_base/api/download?query=&isActive=true&col=Jufo_ID"
    "&col=Name&col=Abbreviation&col=Level&col=ISSNL&col=ISSN1&col=ISSN2&col=ISBN"
    "&col=Other_Title&col=Title_details&col=Continues&col=Continued_by&col=Website"
    "&col=Country&col=country_code&col=Publisher&col=Language&col=lang_code3"
    "&col=lang_code2&col=Year_Start&col=Year_End&col=isScientific&col=isProfessional"
    "&col=isGeneral&col=Type_fi&col=Type_sv&col=Type_en&col=Jufo_History"
)


async def download_file(
    dest_path: str | os.PathLike,
    url: str = JUFO_PORTAL_DOWNLOAD_QUALITY_URL,
    client: httpx.AsyncClient | None = None,
) -> Path:
    dest_path = Path(dest_path)

    async def download(client_: httpx.AsyncClient) -> None:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        )

        async with client_.stream("GET", url, timeout=60) as response:
            response.raise_for_status()
            task_id = progress.add_task(
                f"Downloading {dest_path.name}",
                total=int(response.headers.get("Content-Length", 0)) or None,
            )
            with progress:
                async with await anyio.open_file(dest_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=2048):
                        if not chunk:
                            continue
                        await f.write(chunk)
                        progress.update(task_id, advance=len(chunk))

    if client is None:
        async with httpx_aiohttp.HttpxAiohttpClient() as client:  # noqa: PLR1704
            await download(client)
    else:
        await download(client)
    return dest_path


async def process_csv(
    file_path: str | os.PathLike,
    override_allowlist: Sequence[tuple[str, int]] | None = (
        ("annual review of pathology", 2),
        ("annual review of pathology: mechanisms of disease", 2),
        ("biochimica et biophysica acta (bba) - bioenergetics", 1),
        ("biochimica et biophysica acta (bba) - biomembranes", 1),
        ("biochimica et biophysica acta (bba) - gene regulatory mechanisms", 1),
        ("biochimica et biophysica acta (bba) - general subjects", 1),
        (
            "biochimica et biophysica acta (bba) - molecular and cell biology of lipids",
            1,
        ),
        ("biochimica et biophysica acta (bba) - molecular basis of disease", 1),
        ("biochimica et biophysica acta (bba) - molecular cell research", 1),
        ("biochimica et biophysica acta (bba) - proteins and proteomics", 1),
        ("biochimica et biophysica acta (bba) - reviews on cancer", 1),
        ("bmc evolutionary biology", 2),
        ("pnas", 3),
        ("proceedings of the national academy of sciences", 3),
    ),
    override_blocklist: Sequence[tuple[str, int]] | None = (("scientific reports", 0),),
    records_callback: Callable[[Sequence[tuple[str, int]]], Awaitable] | None = None,
) -> list[tuple[str, int]]:
    async with await anyio.open_file(file_path, encoding="utf-8") as f:
        content = await f.read()

    lines = content.splitlines()
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
    )

    task_id = progress.add_task("Processing", total=len(lines) - 1)
    # Keys are case-insensitive, values are case-sensitive
    records: dict[tuple[str, int], tuple[str, int]] = {}
    with progress:
        for row in csv.DictReader(lines):
            data = (
                row["Name"],
                (
                    int(row["Level"])
                    if str(row.get("Level", "")).isdigit()
                    else DocDetails.UNDEFINED_JOURNAL_QUALITY
                ),
            )
            records[data[0].lower(), data[1]] = data
            progress.update(task_id, advance=1)
    for row_override in override_allowlist or []:
        records[row_override[0].lower(), row_override[1]] = row_override
    for row_override in override_blocklist or []:
        records.pop((row_override[0].lower(), row_override[1]), None)
    records_list = [records[key] for key in sorted(records)]

    if records_callback is not None:
        await records_callback(records_list)
    return records_list


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        downloaded_path = await download_file(
            dest_path=Path(tmpdir) / "journal_quality.csv"
        )
        records = await process_csv(downloaded_path)

        with DEFAULT_JOURNAL_QUALITY_CSV_PATH.open("w", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["clean_name", "quality"])
            for name, quality in records:
                writer.writerow([name.lower(), quality])


if __name__ == "__main__":
    asyncio.run(main())
