from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import pathlib
import pickle
import re
import warnings
import zlib
from collections import Counter
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import datetime
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID

import anyio
from pydantic import BaseModel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
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
from paperqa.settings import IndexSettings, get_settings
from paperqa.types import VAR_MATCH_LOOKUP, DocDetails
from paperqa.utils import ImpossibleParsingError, hexdigest

from .models import SupportsPickle

if TYPE_CHECKING:
    from tantivy import IndexWriter

    from paperqa.settings import MaybeSettings, Settings

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
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


class SearchDocumentStorage(StrEnum):
    """Method to serialize a document."""

    JSON_MODEL_DUMP = auto()  # utf-8 JSON dump
    PICKLE_COMPRESSED = auto()  # pickle + zlib compression
    PICKLE_UNCOMPRESSED = auto()  # pickle

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


# Cache keys are a two-tuple of index name and absolute index directory
# Cache values are a two-tuple of an opened Index instance and the count
# of SearchIndex instances currently referencing that Index
_OPENED_INDEX_CACHE: dict[tuple[str, str], tuple[Index, int]] = {}
DONT_USE_OPENED_INDEX_CACHE = (
    os.environ.get("PQA_INDEX_DONT_CACHE_INDEXES", "").lower() in VAR_MATCH_LOOKUP
)


def reap_opened_index_cache() -> None:
    """Delete any unreferenced Index instances from the Index cache."""
    for index_name, (index, count) in _OPENED_INDEX_CACHE.items():
        if count == 0:
            _OPENED_INDEX_CACHE.pop(index_name)
            del index


class SearchIndex:
    """Wrapper around a tantivy.Index exposing higher-level behaviors for documents."""

    REQUIRED_FIELDS: ClassVar[list[str]] = ["file_location", "body"]

    def __init__(
        self,
        fields: Sequence[str] | None = None,
        index_name: str = "pqa_index",
        index_directory: str | os.PathLike = IndexSettings.model_fields[
            "index_directory"
        ].default,
        storage: SearchDocumentStorage = SearchDocumentStorage.PICKLE_COMPRESSED,
    ):
        if fields is None:
            fields = self.REQUIRED_FIELDS
        self.fields = fields
        if not all(f in self.fields for f in self.REQUIRED_FIELDS):
            raise ValueError(
                f"{self.REQUIRED_FIELDS} must be included in search index fields."
            )
        self.index_name = index_name
        self._index_directory = index_directory
        self._schema: Schema | None = None
        self._index: Index | None = None
        self._searcher: Searcher | None = None
        self._writer: IndexWriter | None = None
        self._index_files: dict[str, str] = {}
        self.changed = False
        self.storage = storage

    @property
    async def index_directory(  # TODO: rename to index_root_directory
        self,
    ) -> anyio.Path:
        directory = anyio.Path(self._index_directory).joinpath(self.index_name)
        await directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    async def index_filename(  # TODO: rename to index_meta_directory
        self,
    ) -> anyio.Path:
        """Directory to store files used to house index internals."""
        index_dir = (await self.index_directory) / "index"
        await index_dir.mkdir(exist_ok=True)
        return index_dir

    @property
    async def docs_index_directory(self) -> anyio.Path:
        """Directory to store documents (e.g. chunked PDFs) given the storage type."""
        docs_dir = (await self.index_directory) / "docs"
        await docs_dir.mkdir(exist_ok=True)
        return docs_dir

    @property
    async def file_index_filename(self) -> anyio.Path:
        """File containing a zlib-compressed pickle of the index_files."""
        return (await self.index_directory) / "files.zip"

    @property
    def schema(self) -> Schema:
        if not self._schema:
            schema_builder = SchemaBuilder()
            for field in self.fields:
                schema_builder.add_text_field(field, stored=True)
            self._schema = schema_builder.build()
        return self._schema

    @property
    async def index(self) -> Index:
        if not self._index:
            index_meta_directory = await self.index_filename
            if await (index_meta_directory / "meta.json").exists():
                if DONT_USE_OPENED_INDEX_CACHE:
                    self._index = Index.open(path=str(index_meta_directory))
                else:
                    key = self.index_name, str(await index_meta_directory.absolute())
                    # NOTE: now we know we're using the cache and have created the cache
                    # key. And we know we're in asyncio.gather race condition risk land.
                    # All of the following operations are *synchronous* so we are not
                    # giving the opportunity for an await to switch to another parallel
                    # version of this code. Otherwise, we risk counts being incorrect
                    # due to race conditions
                    if key not in _OPENED_INDEX_CACHE:  # open a new Index
                        self._index = Index.open(path=str(index_meta_directory))
                        prev_count: int = 0
                    else:  # reuse Index
                        self._index, prev_count = _OPENED_INDEX_CACHE[key]
                    _OPENED_INDEX_CACHE[key] = self._index, prev_count + 1
            else:
                # NOTE: this creates the above meta.json file
                self._index = Index(self.schema, path=str(index_meta_directory))
        return self._index

    def __del__(self) -> None:
        index_meta_directory = (
            pathlib.Path(self._index_directory) / self.index_name / "index"
        )
        key = self.index_name, str(index_meta_directory.absolute())
        if key in _OPENED_INDEX_CACHE:
            index, count = _OPENED_INDEX_CACHE[key]
            _OPENED_INDEX_CACHE[key] = index, count - 1

    @property
    async def searcher(self) -> Searcher:
        if not self._searcher:
            index = await self.index
            index.reload()
            self._searcher = index.searcher()
        return self._searcher

    @contextlib.asynccontextmanager
    async def writer(self, reset: bool = False) -> AsyncIterator[IndexWriter]:
        if not self._writer:
            index = await self.index
            self._writer = index.writer()
        yield self._writer
        if reset:
            self._writer = None

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

    async def filecheck(self, filename: str, body: str | None = None) -> bool:
        """Check if this index contains the filename and if the body's filehash matches."""
        filehash: str | None = self.filehash(body) if body else None
        index_files = await self.index_files
        return bool(
            index_files.get(filename)
            and (filehash is None or index_files[filename] == filehash)
        )

    async def mark_failed_document(self, path: str | os.PathLike) -> None:
        (await self.index_files)[str(path)] = FAILED_DOCUMENT_ADD_ID
        self.changed = True

    async def add_document(
        self,
        index_doc: dict[str, Any],  # TODO: rename to something more intuitive
        document: Any | None = None,
        lock_acquisition_max_retries: int = 1000,
    ) -> None:
        """
        Add the input document to this index.

        Args:
            index_doc: "Document" (thinking types.Doc) of metadata such as 'title' to
                use in the index.
            document: Document to store according to the specified storage method.
            lock_acquisition_max_retries: Amount of retries to acquire a file lock. A
                large default of 1000 is used because lock acquisition can take a while.
        """

        @retry(
            stop=stop_after_attempt(lock_acquisition_max_retries),
            wait=wait_random_exponential(multiplier=0.25, max=60),
            retry=retry_if_exception_type(AsyncRetryError),
        )
        async def _add_document() -> None:
            if not await self.filecheck(index_doc["file_location"], index_doc["body"]):
                try:
                    async with self.writer() as writer:
                        # Let caller handle commit to allow for batching
                        writer.add_document(Document.from_dict(index_doc))  # type: ignore[call-arg]

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
                        raise AsyncRetryError("Failed to acquire lock.") from e
                    raise

        try:
            await _add_document()  # If this runs, we succeeded
        except RetryError:
            logger.exception(
                f"Failed to add document to {index_doc['file_location']}"
                f" within {lock_acquisition_max_retries} attempts."
            )
            raise

    @retry(
        stop=stop_after_attempt(1000),
        wait=wait_random_exponential(multiplier=0.25, max=60),
        retry=retry_if_exception_type(AsyncRetryError),
        reraise=True,
    )
    async def delete_document(self, file_location: str) -> None:
        try:
            async with self.writer() as writer:
                writer.delete_documents("file_location", file_location)
            await self.save_index()
        except ValueError as e:
            if "Failed to acquire Lockfile: LockBusy." in str(e):
                raise AsyncRetryError("Failed to acquire lock") from e
            raise

    async def remove_from_index(self, file_location: str) -> None:
        index_files = await self.index_files
        if index_files.get(file_location):
            await self.delete_document(file_location)
            filehash = index_files.pop(file_location)
            docs_index_dir = await self.docs_index_directory
            # TODO: since the directory is part of the filehash these
            # are always missing. Unsure of how to get around this.
            await (docs_index_dir / f"{filehash}.{self.storage.extension()}").unlink(
                missing_ok=True
            )

            self.changed = True

    @retry(
        stop=stop_after_attempt(1000),
        wait=wait_random_exponential(multiplier=0.25, max=60),
        retry=retry_if_exception_type(AsyncRetryError),
        reraise=True,
    )
    async def save_index(self) -> None:
        try:
            async with self.writer(reset=True) as writer:
                writer.commit()
                writer.wait_merging_threads()
        except ValueError as e:
            if "Failed to acquire Lockfile: LockBusy." in str(e):
                raise AsyncRetryError("Failed to acquire lock") from e
            raise
        file_index_path = await self.file_index_filename
        async with await anyio.open_file(file_index_path, "wb") as f:
            await f.write(zlib.compress(pickle.dumps(await self.index_files)))
        self.changed = False

    async def get_saved_object(
        self, file_location: str, keep_filenames: bool = False
    ) -> Any | tuple[Any, str] | None:
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
        # SEE: https://regex101.com/r/DoLMoa/3
        return re.sub(r'[*\[\]:(){}~^><+"\\]', "", query)

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
                index.parse_query(self.clean_query(query), query_fields),
                top_n,
                offset=offset,
            ).hits
            if s[0] > min_score
        ]
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


def fetch_kwargs_from_manifest(
    file_location: str, manifest: dict[str, Any], manifest_fallback_location: str
) -> dict[str, Any]:
    manifest_entry: dict[str, Any] | None = manifest.get(file_location) or manifest.get(
        manifest_fallback_location
    )
    if manifest_entry:
        return DocDetails(**manifest_entry).model_dump()
    return {}


async def maybe_get_manifest(
    filename: anyio.Path | None = None,
) -> dict[str, dict[str, Any]]:
    if not filename:
        return {}
    if filename.suffix == ".csv":
        try:
            async with await anyio.open_file(filename, mode="r") as file:
                content = await file.read()
            file_loc_to_records = {
                str(r.get("file_location")): r
                for r in csv.DictReader(content.splitlines())
                if r.get("file_location")
            }
            if not file_loc_to_records:
                raise ValueError(  # noqa: TRY301
                    "No mapping of file location to details extracted from manifest"
                    f" file {filename}."
                )
            logger.debug(
                f"Found manifest file at {filename}, read"
                f" {len(file_loc_to_records)} records from it, which maps to"
                f" {len(file_loc_to_records)} locations."
            )
        except FileNotFoundError:
            logger.warning(f"Manifest file at {filename} could not be found.")
        except Exception:
            logger.exception(f"Error reading manifest file {filename}.")
        else:
            return file_loc_to_records
    else:
        logger.error(f"Invalid manifest file type: {filename.suffix}")

    return {}


FAILED_DOCUMENT_ADD_ID = "ERROR"


async def process_file(
    rel_file_path: anyio.Path,
    search_index: SearchIndex,
    manifest: dict[str, Any],
    semaphore: anyio.Semaphore,
    settings: Settings,
    processed_counter: Counter[str],
    progress_bar_update: Callable[[], Any] | None = None,
) -> None:

    abs_file_path = (
        pathlib.Path(settings.agent.index.paper_directory).absolute() / rel_file_path
    )
    fallback_title = rel_file_path.name
    if settings.agent.index.use_absolute_paper_directory:
        file_location = str(abs_file_path)
        manifest_fallback_location = str(rel_file_path)
    else:
        file_location = str(rel_file_path)
        manifest_fallback_location = str(abs_file_path)

    async with semaphore:
        if not await search_index.filecheck(filename=file_location):
            logger.info(f"New file to index: {file_location}...")

            kwargs = fetch_kwargs_from_manifest(
                file_location, manifest, manifest_fallback_location
            )

            tmp_docs = Docs()
            try:
                await tmp_docs.aadd(
                    path=abs_file_path,
                    fields=["title", "author", "journal", "year"],
                    settings=settings,
                    **kwargs,
                )
            except Exception as e:
                # We handle any exception here because we want to save_index so we
                # 1. can resume the build without rebuilding this file if a separate
                # process_file invocation leads to a segfault or crash.
                # 2. don't have deadlock issues after.
                logger.exception(
                    f"Error parsing {file_location}, skipping index for this file."
                )
                await search_index.mark_failed_document(file_location)
                await search_index.save_index()
                if progress_bar_update:
                    progress_bar_update()

                if not isinstance(e, ValueError | ImpossibleParsingError):
                    # ImpossibleParsingError: parsing failure, don't retry
                    # ValueError: TODOC
                    raise
                return

            this_doc = next(iter(tmp_docs.docs.values()))

            if isinstance(this_doc, DocDetails):
                title = this_doc.title or fallback_title
                year = this_doc.year or "Unknown year"
            else:
                title, year = fallback_title, "Unknown year"

            await search_index.add_document(
                {
                    "title": title,
                    "year": year,
                    "file_location": file_location,
                    "body": "".join(t.text for t in tmp_docs.texts),
                },
                document=tmp_docs,
            )

            processed_counter["batched_save_counter"] += 1
            if (
                processed_counter["batched_save_counter"]
                == settings.agent.index.batch_size
            ):
                await search_index.save_index()
                processed_counter["batched_save_counter"] = 0

            logger.info(f"Complete ({title}).")

        # Update progress bar for either a new or previously indexed file
        if progress_bar_update:
            progress_bar_update()


WARN_IF_INDEXING_MORE_THAN = 999


def _make_progress_bar_update(
    sync_index_w_directory: bool, total: int
) -> tuple[contextlib.AbstractContextManager, Callable[[], Any] | None]:
    # Disable should override enable
    env_var_disable = (
        os.environ.get("PQA_INDEX_DISABLE_PROGRESS_BAR", "").lower() in VAR_MATCH_LOOKUP
    )
    env_var_enable = (
        os.environ.get("PQA_INDEX_ENABLE_PROGRESS_BAR", "").lower() in VAR_MATCH_LOOKUP
    )
    try:
        is_cli = is_running_under_cli()  # pylint: disable=used-before-assignment
    except NameError:  # Work around circular import
        from . import is_running_under_cli

        is_cli = is_running_under_cli()

    if sync_index_w_directory and not env_var_disable and (is_cli or env_var_enable):
        # Progress.get_default_columns with a few more
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        task_id = progress.add_task("Indexing...", total=total)

        def progress_bar_update() -> None:
            progress.update(task_id, advance=1)

        return progress, progress_bar_update
    return contextlib.nullcontext(), None


async def get_directory_index(  # noqa: PLR0912
    index_name: str | None = None,
    sync_index_w_directory: bool = True,
    settings: MaybeSettings = None,
    build: bool = True,
) -> SearchIndex:
    """
    Create a Tantivy index by reading from a directory of text files.

    This function only reads from the source directory, not edits or writes to it.

    Args:
        index_name: Deprecated override on the name of the index. If unspecified,
            the default behavior is to generate the name from the input settings.
        sync_index_w_directory: Opt-out flag to sync the index (add or delete index
            files) with the source paper directory.
        settings: Application settings.
        build: Opt-out flag (default is True) to read the contents of the source paper
            directory and if sync_index_w_directory is enabled also update the index.
    """
    _settings = get_settings(settings)
    index_settings = _settings.agent.index
    if index_name:
        warnings.warn(
            "The index_name argument has been moved to"
            f" {type(_settings.agent.index).__name__},"
            " this deprecation will conclude in version 6.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        index_settings.name = index_name
    del index_name

    search_index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
        index_name=index_settings.name or _settings.get_index_name(),
        index_directory=index_settings.index_directory,
    )
    # NOTE: if the index was not previously built, its index_files will be empty.
    # Otherwise, the index_files will not be empty
    if not build:
        if not await search_index.index_files:
            raise RuntimeError(
                f"Index {search_index.index_name} was empty, please rebuild it."
            )
        return search_index

    if not sync_index_w_directory:
        warnings.warn(
            "The sync_index_w_directory argument has been moved to"
            f" {type(_settings.agent.index).__name__},"
            " this deprecation will conclude in version 6.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        index_settings.sync_with_paper_directory = sync_index_w_directory
    del sync_index_w_directory

    paper_directory = anyio.Path(index_settings.paper_directory)
    manifest = await maybe_get_manifest(
        filename=await index_settings.finalize_manifest_file()
    )
    valid_papers_rel_file_paths = [
        file.relative_to(paper_directory)
        async for file in (
            paper_directory.rglob("*")
            if index_settings.recurse_subdirectories
            else paper_directory.iterdir()
        )
        if file.suffix in {".txt", ".pdf", ".html", ".md"}
    ]
    if len(valid_papers_rel_file_paths) > WARN_IF_INDEXING_MORE_THAN:
        logger.warning(
            f"Indexing {len(valid_papers_rel_file_paths)} files into the index"
            f" {search_index.index_name}, may take a few minutes."
        )

    index_unique_file_paths: set[str] = set((await search_index.index_files).keys())
    if extra_index_files := (
        index_unique_file_paths - {str(f) for f in valid_papers_rel_file_paths}
    ):
        if index_settings.sync_with_paper_directory:
            for extra_file in extra_index_files:
                logger.warning(
                    f"[bold red]Removing {extra_file} from index.[/bold red]"
                )
                await search_index.remove_from_index(extra_file)
            logger.warning("[bold red]Files removed![/bold red]")
        else:
            logger.warning(
                f"[bold red]Indexed files {extra_index_files} are missing from paper"
                f" folder ({paper_directory}).[/bold red]"
            )

    semaphore = anyio.Semaphore(index_settings.concurrency)
    progress_bar, progress_bar_update_fn = _make_progress_bar_update(
        index_settings.sync_with_paper_directory, total=len(valid_papers_rel_file_paths)
    )
    with progress_bar:
        async with anyio.create_task_group() as tg:
            processed_counter: Counter[str] = Counter()
            for rel_file_path in valid_papers_rel_file_paths:
                if index_settings.sync_with_paper_directory:
                    tg.start_soon(
                        process_file,
                        rel_file_path,
                        search_index,
                        manifest,
                        semaphore,
                        _settings,
                        processed_counter,
                        progress_bar_update_fn,
                    )
                else:
                    logger.debug(
                        f"File {rel_file_path} found in paper directory"
                        f" {paper_directory}."
                    )

    if search_index.changed:
        await search_index.save_index()
    else:
        logger.debug("No changes to index.")

    return search_index
