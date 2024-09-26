from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import pathlib
import pickle
import zlib
from collections.abc import Awaitable, Callable, Collection, Sequence
from enum import StrEnum, auto
from io import StringIO
from typing import Any, ClassVar, cast
from uuid import UUID

import anyio
from pydantic import BaseModel
from rich.progress import Progress
from tantivy import (  # pylint: disable=no-name-in-module
    Document,
    Index,
    Schema,
    SchemaBuilder,
    Searcher,
)
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from paperqa.docs import Docs
from paperqa.settings import MaybeSettings, Settings, get_settings
from paperqa.types import DocDetails
from paperqa.utils import ImpossibleParsingError, hexdigest

from .models import SupportsPickle

logger = logging.getLogger(__name__)


class AsyncRetryError(Exception):
    """Flags a retry for another tenacity attempt."""


class RobustEncoder(json.JSONEncoder):
    """JSON encoder that can handle UUID and set objects."""

    def default(self, o):
        if isinstance(o, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return str(o)
        if isinstance(o, set):
            return list(o)
        if isinstance(o, os.PathLike):
            return str(o)
        return json.JSONEncoder.default(self, o)


class SearchDocumentStorage(StrEnum):
    JSON_MODEL_DUMP = auto()
    PICKLE_COMPRESSED = auto()
    PICKLE_UNCOMPRESSED = auto()

    def extension(self) -> str:
        if self == SearchDocumentStorage.JSON_MODEL_DUMP:
            return "json"
        if self == SearchDocumentStorage.PICKLE_COMPRESSED:
            return "zip"
        return "pkl"

    def write_to_string(self, data: BaseModel | SupportsPickle) -> bytes:
        if self == SearchDocumentStorage.JSON_MODEL_DUMP:
            if isinstance(data, BaseModel):
                return json.dumps(data.model_dump(), cls=RobustEncoder).encode("utf-8")
            raise ValueError("JSON_MODEL_DUMP requires a BaseModel object.")
        if self == SearchDocumentStorage.PICKLE_COMPRESSED:
            return zlib.compress(pickle.dumps(data))
        return pickle.dumps(data)

    def read_from_string(self, data: str | bytes) -> BaseModel | SupportsPickle:
        if self == SearchDocumentStorage.JSON_MODEL_DUMP:
            return json.loads(data)
        if self == SearchDocumentStorage.PICKLE_COMPRESSED:
            return pickle.loads(zlib.decompress(data))  # type: ignore[arg-type] # noqa: S301
        return pickle.loads(data)  # type: ignore[arg-type] # noqa: S301


class SearchIndex:
    REQUIRED_FIELDS: ClassVar[list[str]] = ["file_location", "body"]

    def __init__(
        self,
        fields: Sequence[str] | None = None,
        index_name: str = "pqa_index",
        index_directory: str | os.PathLike | None = None,
        storage: SearchDocumentStorage = SearchDocumentStorage.PICKLE_COMPRESSED,
    ):
        if fields is None:
            fields = self.REQUIRED_FIELDS
        self.fields = fields
        if not all(f in self.fields for f in self.REQUIRED_FIELDS):
            raise ValueError(
                f"{self.REQUIRED_FIELDS} must be included in search index fields."
            )
        if index_directory is None:
            index_directory = Settings.model_fields["index_directory"].default
        self.index_name = index_name
        self._index_directory = index_directory
        self._schema = None
        self._index = None
        self._searcher = None
        self._index_files: dict[str, str] = {}
        self.changed = False
        self.storage = storage

    async def init_directory(self) -> None:
        await anyio.Path(await self.index_directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    async def extend_and_make_directory(base: anyio.Path, *dirs: str) -> anyio.Path:
        directory = base.joinpath(*dirs)
        await directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    async def index_directory(self) -> anyio.Path:
        return await self.extend_and_make_directory(
            anyio.Path(self._index_directory), self.index_name
        )

    @property
    async def index_filename(self) -> anyio.Path:
        return await self.extend_and_make_directory(await self.index_directory, "index")

    @property
    async def docs_index_directory(self) -> anyio.Path:
        return await self.extend_and_make_directory(await self.index_directory, "docs")

    @property
    async def file_index_filename(self) -> anyio.Path:
        return (await self.index_directory) / "files.zip"

    @property
    def schema(self) -> Schema:
        if not self._schema:
            schema_builder = SchemaBuilder()
            for field in self.fields:
                schema_builder.add_text_field(field, stored=True)
            self._schema = schema_builder.build()  # type: ignore[assignment]
        return cast(Schema, self._schema)

    @property
    async def index(self) -> Index:
        if not self._index:
            index_path = await self.index_filename
            if await (index_path / "meta.json").exists():
                self._index = Index.open(str(index_path))  # type: ignore[assignment]
            else:
                self._index = Index(self.schema, str(index_path))  # type: ignore[assignment]
        return cast(Index, self._index)

    @property
    async def searcher(self) -> Searcher:
        if not self._searcher:
            index = await self.index
            index.reload()
            self._searcher = index.searcher()  # type: ignore[assignment]
        return cast(Searcher, self._searcher)

    @property
    async def count(self) -> int:
        return (await self.searcher).num_docs

    @property
    async def index_files(self) -> dict[str, str]:
        if not self._index_files:
            file_index_path = await self.file_index_filename
            if await file_index_path.exists():
                async with await anyio.open_file(file_index_path, "rb") as f:
                    content = await f.read()
                    self._index_files = pickle.loads(  # noqa: S301
                        zlib.decompress(content)
                    )
        return self._index_files

    @staticmethod
    def filehash(body: str) -> str:
        return hexdigest(body)

    async def filecheck(self, filename: str, body: str | None = None):
        filehash = None
        if body:
            filehash = self.filehash(body)
        index_files = await self.index_files
        return bool(
            index_files.get(filename)
            and (filehash is None or index_files[filename] == filehash)
        )

    async def add_document(
        self, index_doc: dict, document: Any | None = None, max_retries: int = 1000
    ) -> None:
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_random_exponential(multiplier=0.25, max=60),
            retry=retry_if_exception_type(AsyncRetryError),
            reraise=True,
        )
        async def _add_document_with_retry() -> None:
            if not await self.filecheck(index_doc["file_location"], index_doc["body"]):
                try:
                    index = await self.index
                    writer = index.writer()
                    writer.add_document(Document.from_dict(index_doc))  # type: ignore[call-arg]
                    writer.commit()

                    filehash = self.filehash(index_doc["body"])
                    (await self.index_files)[index_doc["file_location"]] = filehash

                    if document:
                        docs_index_dir = await self.docs_index_directory
                        async with await anyio.open_file(
                            docs_index_dir / f"{filehash}.{self.storage.extension()}",
                            "wb",
                        ) as f:
                            await f.write(self.storage.write_to_string(document))

                    self.changed = True
                except ValueError as e:
                    if "Failed to acquire Lockfile: LockBusy." in str(e):
                        raise AsyncRetryError("Failed to acquire lock") from e
                    raise

        try:
            await _add_document_with_retry()
        except RetryError:
            logger.exception(
                f"Failed to add document after {max_retries} attempts:"
                f" {index_doc['file_location']}"
            )
            raise

        # Success

    @staticmethod
    @retry(
        stop=stop_after_attempt(1000),
        wait=wait_random_exponential(multiplier=0.25, max=60),
        retry=retry_if_exception_type(AsyncRetryError),
        reraise=True,
    )
    def delete_document(index: Index, file_location: str) -> None:
        try:
            writer = index.writer()
            writer.delete_documents("file_location", file_location)
            writer.commit()
        except ValueError as e:
            if "Failed to acquire Lockfile: LockBusy." in str(e):
                raise AsyncRetryError("Failed to acquire lock") from e
            raise

    async def remove_from_index(self, file_location: str) -> None:
        index_files = await self.index_files
        if index_files.get(file_location):
            index = await self.index
            self.delete_document(index, file_location)
            filehash = index_files.pop(file_location)
            docs_index_dir = await self.docs_index_directory
            # TODO: since the directory is part of the filehash these
            # are always missing. Unsure of how to get around this.
            await (docs_index_dir / f"{filehash}.{self.storage.extension()}").unlink(
                missing_ok=True
            )

            self.changed = True

    async def save_index(self) -> None:
        file_index_path = await self.file_index_filename
        async with await anyio.open_file(file_index_path, "wb") as f:
            await f.write(zlib.compress(pickle.dumps(await self.index_files)))

    async def get_saved_object(
        self, file_location: str, keep_filenames: bool = False
    ) -> Any | None | tuple[Any, str]:
        filehash = (await self.index_files).get(file_location)
        if filehash:
            docs_index_dir = await self.docs_index_directory
            async with await anyio.open_file(
                docs_index_dir / f"{filehash}.{self.storage.extension()}", "rb"
            ) as f:
                content = await f.read()
                if keep_filenames:
                    return self.storage.read_from_string(content), file_location
                return self.storage.read_from_string(content)
        return None

    def clean_query(self, query: str) -> str:
        for replace in ("*", "[", "]", ":", "(", ")", "{", "}", "~", '"'):
            query = query.replace(replace, "")
        return query

    async def query(
        self,
        query: str,
        top_n: int = 10,
        offset: int = 0,
        min_score: float = 0.0,
        keep_filenames: bool = False,
        field_subset: list[str] | None = None,
    ) -> list[Any]:
        query_fields = list(field_subset or self.fields)
        searcher = await self.searcher
        index = await self.index
        addresses = [
            s[1]
            for s in searcher.search(
                index.parse_query(self.clean_query(query), query_fields), top_n
            ).hits
            if s[0] > min_score
        ][offset : offset + top_n]
        search_index_docs = [searcher.doc(address) for address in addresses]
        return [
            result
            for result in [
                await self.get_saved_object(
                    doc["file_location"][0], keep_filenames=keep_filenames  # type: ignore[index]
                )
                for doc in search_index_docs
            ]
            if result is not None
        ]


async def maybe_get_manifest(
    filename: anyio.Path | None = None,
) -> dict[str, DocDetails]:
    if not filename:
        return {}
    if filename.suffix == ".csv":
        try:
            async with await anyio.open_file(filename, mode="r") as file:
                content = await file.read()
            records = [DocDetails(**row) for row in csv.DictReader(StringIO(content))]
            logger.debug(
                f"Found manifest file at {filename} and read {len(records)} records"
                " from it."
            )
            return {str(r.file_location): r for r in records if r.file_location}
        except FileNotFoundError:
            logging.warning(f"Manifest file at {filename} could not be found.")
        except Exception:
            logging.exception(f"Error reading manifest file {filename}")
    else:
        logging.error(f"Invalid manifest file type: {filename.suffix}")

    return {}


FAILED_DOCUMENT_ADD_ID = "ERROR"


async def process_file(
    file_path: anyio.Path,
    search_index: SearchIndex,
    metadata: dict[str, Any],
    semaphore: anyio.Semaphore,
    settings: Settings,
    progress_bar_update: Callable[[], Awaitable] | None = None,
) -> None:
    async with semaphore:
        file_name = file_path.name
        if not await search_index.filecheck(str(file_path)):
            logger.info(f"New file to index: {file_name}...")

            doi, title = None, None
            if file_name in metadata:
                doi, title = metadata[file_name].doi, metadata[file_name].title

            tmp_docs = Docs()
            try:
                await tmp_docs.aadd(
                    path=pathlib.Path(file_path),
                    title=title,
                    doi=doi,
                    fields=["title", "author", "journal", "year"],
                    settings=settings,
                )
            except (ValueError, ImpossibleParsingError):
                logger.exception(
                    f"Error parsing {file_name}, skipping index for this file."
                )
                (await search_index.index_files)[
                    str(file_path)
                ] = FAILED_DOCUMENT_ADD_ID
                await search_index.save_index()
                if progress_bar_update:
                    await progress_bar_update()
                return

            this_doc = next(iter(tmp_docs.docs.values()))

            if isinstance(this_doc, DocDetails):
                title = this_doc.title or file_name
                year = this_doc.year or "Unknown year"
            else:
                title, year = file_name, "Unknown year"

            await search_index.add_document(
                {
                    "title": title,
                    "year": year,
                    "file_location": str(file_path),
                    "body": "".join([t.text for t in tmp_docs.texts]),
                },
                document=tmp_docs,
            )
            await search_index.save_index()
            logger.info(f"Complete ({title}).")

        # Update progress bar for either a new or previously indexed file
        if progress_bar_update:
            await progress_bar_update()


WARN_IF_INDEXING_MORE_THAN = 999
ENV_VAR_MATCH: Collection[str] = {"1", "true"}


def _make_progress_bar_update(
    sync_index_w_directory: bool, total: int
) -> tuple[contextlib.AbstractContextManager, Callable[[], Awaitable] | None]:
    # Disable should override enable
    env_var_disable = (
        os.environ.get("PQA_INDEX_DISABLE_PROGRESS_BAR", "").lower() in ENV_VAR_MATCH
    )
    env_var_enable = (
        os.environ.get("PQA_INDEX_ENABLE_PROGRESS_BAR", "").lower() in ENV_VAR_MATCH
    )
    try:
        is_cli = is_running_under_cli()  # pylint: disable=used-before-assignment
    except NameError:  # Work around circular import
        from . import is_running_under_cli

        is_cli = is_running_under_cli()

    if sync_index_w_directory and not env_var_disable and (is_cli or env_var_enable):
        progress = Progress()
        task_id = progress.add_task("Indexing...", total=total)

        async def progress_bar_update() -> None:
            progress.update(task_id, advance=1)

        return progress, progress_bar_update
    return contextlib.nullcontext(), None


async def get_directory_index(
    index_name: str | None = None,
    sync_index_w_directory: bool = True,
    settings: MaybeSettings = None,
) -> SearchIndex:
    """
    Create a Tantivy index by reading from a directory of text files.

    This function only reads from the source directory, not edits or writes to it.

    Args:
        index_name: Override on the name of the index. If unspecified, the default
            behavior is to generate the name from the input settings.
        sync_index_w_directory: Sync the index (add or delete index files) with the
            source paper directory.
        settings: Application settings.
    """
    _settings = get_settings(settings)

    semaphore = anyio.Semaphore(_settings.agent.index_concurrency)
    directory = anyio.Path(_settings.paper_directory)

    if _settings.index_absolute_directory:
        directory = await directory.absolute()

    search_index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
        index_name=index_name or _settings.get_index_name(),
        index_directory=_settings.index_directory,
    )

    manifest_file = (
        anyio.Path(_settings.manifest_file) if _settings.manifest_file else None
    )
    if manifest_file and not await manifest_file.exists():
        # If the manifest file was specified but doesn't exist,
        # perhaps it was specified as a relative path from the paper_directory
        manifest_file = directory / manifest_file

    metadata = await maybe_get_manifest(manifest_file)
    valid_paper_dir_files = [
        file
        async for file in (
            directory.rglob("*") if _settings.index_recursively else directory.iterdir()
        )
        if file.suffix in {".txt", ".pdf", ".html"}
    ]
    if len(valid_paper_dir_files) > WARN_IF_INDEXING_MORE_THAN:
        logger.warning(
            f"Indexing {len(valid_paper_dir_files)} files. This may take a few minutes."
        )
    # NOTE: if the index was not previously built, this will be empty.
    # Otherwise, it will not be empty
    index_unique_file_paths: set[str] = set((await search_index.index_files).keys())

    if extra_index_files := (
        index_unique_file_paths - {str(f) for f in valid_paper_dir_files}
    ):
        if sync_index_w_directory:
            for extra_file in extra_index_files:
                logger.warning(
                    f"[bold red]Removing {extra_file} from index.[/bold red]"
                )
                await search_index.remove_from_index(extra_file)
            logger.warning("[bold red]Files removed![/bold red]")
        else:
            logger.warning(
                f"[bold red]Indexed files {extra_index_files} are missing from paper"
                f" folder ({directory}).[/bold red]"
            )

    progress_bar, progress_bar_update_fn = _make_progress_bar_update(
        sync_index_w_directory, total=len(valid_paper_dir_files)
    )
    with progress_bar:
        async with anyio.create_task_group() as tg:
            for file_path in valid_paper_dir_files:
                if sync_index_w_directory:
                    tg.start_soon(
                        process_file,
                        file_path,
                        search_index,
                        metadata,
                        semaphore,
                        _settings,
                        progress_bar_update_fn,
                    )
                else:
                    logger.debug(f"File {file_path.name} found in paper directory.")

    if search_index.changed:
        await search_index.save_index()
    else:
        logger.debug("No changes to index.")

    return search_index
