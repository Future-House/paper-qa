"""Test indexing error policies and reports using local indexes without model calls."""

import asyncio
import sys
from unittest.mock import patch

import anyio
import httpx
import pytest

from paperqa import Docs, Settings
from paperqa.agents import main
from paperqa.agents.search import (
    FAILED_DOCUMENT_ADD_ID,
    SearchIndex,
    get_directory_index,
)
from paperqa.types import Doc, Text
from paperqa.utils import ImpossibleParsingError


@pytest.fixture
def settings(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    return Settings(
        embedding="sparse",
        parsing={"use_doc_details": False, "multimodal": False},
        agent={
            "index": {
                "name": "issue-1275",
                "paper_directory": papers,
                "index_directory": tmp_path / "indexes",
                "concurrency": 2,
            }
        },
    )


def http_error():
    request = httpx.Request("GET", "https://example.invalid/metadata")
    return httpx.HTTPStatusError(
        "429 Too Many Requests",
        request=request,
        response=httpx.Response(429, request=request),
    )


def add_stub_document(docs, path):
    doc = Doc(dockey=path.name, docname="Example2026", citation="Example. 2026.")
    docs.docs[doc.dockey] = doc
    docs.texts.append(
        Text(text="A synthetic research document.", name="Example", doc=doc)
    )


@pytest.mark.asyncio
async def test_real_text_documents_continue_and_remain_searchable(settings):
    settings.agent.index.continue_on_error = True
    papers = settings.agent.index.paper_directory
    (papers / "healthy.txt").write_text(
        "Research on the resilience of document processing systems. " * 20,
        encoding="utf-8",
    )
    (papers / "empty.txt").write_text("", encoding="utf-8")
    manifest = papers / "manifest.csv"
    manifest.write_text(
        "file_location,citation\n"
        "healthy.txt,Example Research 2026\n"
        "empty.txt,Empty Research 2026\n",
        encoding="utf-8",
    )
    settings.agent.index.manifest_file = manifest

    index = await get_directory_index(settings=settings)

    assert await index.count == 1
    assert await index.query("resilience")
    assert index.indexing_report is not None
    assert index.indexing_report.indexed == 1
    assert len(index.indexing_report.failures) == 1
    assert index.indexing_report.failures[0].file_location == "empty.txt"
    assert (await index.index_files)["empty.txt"] == FAILED_DOCUMENT_ADD_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("error_kind", ["http", "timeout", "runtime"])
async def test_non_parsing_failure_cancels_sibling_and_persists_error(
    settings, error_kind
):
    papers = settings.agent.index.paper_directory
    (papers / "failed.txt").write_text("failed", encoding="utf-8")
    (papers / "healthy.txt").write_text("healthy", encoding="utf-8")
    healthy_started = anyio.Event()
    healthy_cancelled = anyio.Event()
    error = {
        "http": http_error(),
        "timeout": TimeoutError("Synthetic timeout"),
        "runtime": RuntimeError("Synthetic reader error"),
    }[error_kind]

    async def aadd(_docs, path, **_kwargs):
        if path.name == "failed.txt":
            await healthy_started.wait()
            raise error
        healthy_started.set()
        try:
            await anyio.sleep_forever()
        except asyncio.CancelledError:
            healthy_cancelled.set()
            raise

    with (
        patch.object(Docs, "aadd", aadd),
        pytest.raises(ExceptionGroup) as caught,
        anyio.fail_after(10),
    ):
        await get_directory_index(settings=settings)
    assert any(isinstance(e, type(error)) for e in caught.value.exceptions)
    assert healthy_cancelled.is_set()
    loaded = await get_directory_index(settings=settings, build=False)
    assert (await loaded.index_files) == {"failed.txt": FAILED_DOCUMENT_ADD_ID}
    assert await loaded.count == 0


@pytest.mark.asyncio
async def test_rerun_reports_previously_failed_file(settings, caplog):
    paper = settings.agent.index.paper_directory / "failed.txt"
    paper.write_text("research", encoding="utf-8")

    async def fail(_docs, **_kwargs):  # noqa: RUF029
        raise http_error()

    with patch.object(Docs, "aadd", fail), pytest.raises(ExceptionGroup):
        await get_directory_index(settings=settings)

    calls = []

    async def recover(docs, path, **_kwargs):  # noqa: RUF029
        calls.append(path.name)
        add_stub_document(docs, path)

    with patch.object(Docs, "aadd", recover):
        index = await get_directory_index(settings=settings)
    assert not calls
    assert (await index.index_files)["failed.txt"] == FAILED_DOCUMENT_ADD_ID
    assert await index.count == 0
    assert await index.filecheck("failed.txt")
    assert index.indexing_report is not None
    assert index.indexing_report.indexed == 0
    assert index.indexing_report.skipped == 0
    assert index.indexing_report.failures[0].previously_failed
    assert index.indexing_report.failures[0].error_message is None
    assert "skipped after a previous failure" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ValueError, ImpossibleParsingError])
async def test_parsing_error_allows_healthy_document_to_be_indexed(
    settings, error_type
):
    papers = settings.agent.index.paper_directory
    (papers / "failed.txt").write_text("failed", encoding="utf-8")
    (papers / "healthy.txt").write_text("healthy", encoding="utf-8")

    async def aadd(docs, path, **_kwargs):  # noqa: RUF029
        if path.name == "failed.txt":
            raise error_type("Synthetic parsing failure")
        add_stub_document(docs, path)

    # Serial indexing avoids making this control test depend on index writer races.
    settings.agent.index.concurrency = 1
    with patch.object(Docs, "aadd", aadd):
        index = await get_directory_index(settings=settings)
    files = await index.index_files
    assert files["failed.txt"] == FAILED_DOCUMENT_ADD_ID
    assert files["healthy.txt"] != FAILED_DOCUMENT_ADD_ID
    assert await index.count == 1


@pytest.mark.asyncio
async def test_index_write_failure_is_outside_document_error_handler(settings):
    settings.agent.index.continue_on_error = True
    paper = settings.agent.index.paper_directory / "healthy.txt"
    paper.write_text("healthy", encoding="utf-8")

    async def aadd(docs, path, **_kwargs):  # noqa: RUF029
        add_stub_document(docs, path)

    with (
        patch.object(Docs, "aadd", aadd),
        patch.object(
            SearchIndex, "add_document", side_effect=OSError("Synthetic disk failure")
        ),
        patch.object(SearchIndex, "mark_failed_document") as mark_failed,
        pytest.raises(ExceptionGroup),
    ):
        await get_directory_index(settings=settings)
    mark_failed.assert_not_called()


@pytest.mark.asyncio
async def test_cancellation_is_not_recorded_as_document_failure(settings):
    settings.agent.index.continue_on_error = True
    paper = settings.agent.index.paper_directory / "healthy.txt"
    paper.write_text("healthy", encoding="utf-8")

    async def aadd(_docs, **_kwargs):  # noqa: RUF029
        raise asyncio.CancelledError

    with (
        patch.object(Docs, "aadd", aadd),
        patch.object(SearchIndex, "mark_failed_document") as mark_failed,
    ):
        index = await get_directory_index(settings=settings)
    assert not await index.index_files
    mark_failed.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_kind", ["http", "timeout", "runtime"])
async def test_continue_on_error_preserves_healthy_task_and_reports_failure(
    settings, error_kind, caplog
):
    settings.agent.index.continue_on_error = True
    papers = settings.agent.index.paper_directory
    (papers / "failed.txt").write_text("failed", encoding="utf-8")
    (papers / "healthy.txt").write_text("healthy", encoding="utf-8")
    healthy_started = anyio.Event()
    failure_saved = anyio.Event()
    error = {
        "http": http_error(),
        "timeout": TimeoutError("Synthetic timeout"),
        "runtime": RuntimeError("Synthetic reader error"),
    }[error_kind]
    original_save = SearchIndex.save_index

    async def save(index):
        await original_save(index)
        failure_saved.set()

    async def aadd(docs, path, **_kwargs):
        if path.name == "failed.txt":
            await healthy_started.wait()
            raise error
        healthy_started.set()
        await failure_saved.wait()
        add_stub_document(docs, path)

    with (
        patch.object(Docs, "aadd", aadd),
        patch.object(SearchIndex, "save_index", save),
        anyio.fail_after(10),
    ):
        index = await get_directory_index(settings=settings)
    assert await index.count == 1
    assert await index.query("synthetic")
    files = await index.index_files
    assert files["failed.txt"] == FAILED_DOCUMENT_ADD_ID
    assert files["healthy.txt"] != FAILED_DOCUMENT_ADD_ID
    report = index.indexing_report
    assert report is not None
    assert report.indexed == 1
    assert report.skipped == 0
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure.file_location == "failed.txt"
    assert failure.error_type == type(error).__name__
    assert failure.error_message == str(error)
    assert not failure.previously_failed
    assert "1 failed document(s): 1 indexed, 0 skipped" in caplog.text
    assert f"Failed document failed.txt: {type(error).__name__}: {error}" in caplog.text


@pytest.mark.asyncio
async def test_all_failures_are_reported_and_no_success_is_claimed(settings, caplog):
    settings.agent.index.continue_on_error = True
    for name in ("one.txt", "two.txt"):
        (settings.agent.index.paper_directory / name).write_text(
            "fail", encoding="utf-8"
        )

    async def aadd(_docs, path, **_kwargs):  # noqa: RUF029
        raise RuntimeError(f"Cannot read {path.name}")

    # The report should aggregate failures, independently of disk-writer concurrency.
    settings.agent.index.concurrency = 1
    with patch.object(Docs, "aadd", aadd):
        index = await get_directory_index(settings=settings)
    assert await index.count == 0
    assert index.indexing_report is not None
    assert index.indexing_report.indexed == 0
    assert {f.file_location for f in index.indexing_report.failures} == {
        "one.txt",
        "two.txt",
    }
    assert "2 failed document(s): 0 indexed, 0 skipped" in caplog.text


@pytest.mark.asyncio
async def test_document_preparation_error_is_also_contained(settings, caplog):
    settings.agent.index.continue_on_error = True
    paper = settings.agent.index.paper_directory / "empty_result.txt"
    paper.write_text("research", encoding="utf-8")

    async def aadd(_docs, **_kwargs):  # noqa: RUF029
        return None

    with patch.object(Docs, "aadd", aadd):
        index = await get_directory_index(settings=settings)
    assert (await index.index_files)[paper.name] == FAILED_DOCUMENT_ADD_ID
    assert index.indexing_report is not None
    assert index.indexing_report.failures[0].error_type == "StopIteration"
    assert "1 failed document(s): 0 indexed, 0 skipped" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("continue_on_error", [False, True])
async def test_manifest_error_preserves_default_boundary(settings, continue_on_error):
    settings.agent.index.continue_on_error = continue_on_error
    paper = settings.agent.index.paper_directory / "invalid_manifest.txt"
    paper.write_text("research", encoding="utf-8")

    with patch(
        "paperqa.agents.search.fetch_kwargs_from_manifest",
        side_effect=ValueError("Invalid manifest entry"),
    ):
        if continue_on_error:
            index = await get_directory_index(settings=settings)
            assert (await index.index_files)[paper.name] == FAILED_DOCUMENT_ADD_ID
            assert index.indexing_report is not None
            assert (
                index.indexing_report.failures[0].error_message
                == "Invalid manifest entry"
            )
        else:
            with pytest.raises(ExceptionGroup) as caught:
                await get_directory_index(settings=settings)
            assert isinstance(caught.value.exceptions[0], ValueError)
            untouched = SearchIndex(
                index_name=settings.agent.index.name,
                index_directory=settings.agent.index.index_directory,
            )
            assert not await untouched.index_files


@pytest.mark.asyncio
async def test_successful_build_and_skip_have_fresh_reports(settings):
    paper = settings.agent.index.paper_directory / "healthy.txt"
    paper.write_text("healthy", encoding="utf-8")

    async def aadd(docs, path, **_kwargs):  # noqa: RUF029
        add_stub_document(docs, path)

    with patch.object(Docs, "aadd", aadd):
        index = await get_directory_index(settings=settings)
    assert index.indexing_report is not None
    assert index.indexing_report.indexed == 1
    assert not index.indexing_report.failures
    with patch.object(Docs, "aadd") as add:
        reused = await get_directory_index(settings=settings)
    add.assert_not_called()
    assert reused.indexing_report is not None
    assert reused.indexing_report.indexed == 0
    assert reused.indexing_report.skipped == 1
    assert not reused.indexing_report.failures
    loaded = await get_directory_index(settings=settings, build=False)
    assert loaded.indexing_report is None


@pytest.mark.asyncio
async def test_parent_cancellation_propagates_with_continue_on_error(settings):
    settings.agent.index.continue_on_error = True
    paper = settings.agent.index.paper_directory / "healthy.txt"
    paper.write_text("healthy", encoding="utf-8")
    started = anyio.Event()
    cancelled = anyio.Event()

    async def aadd(_docs, **_kwargs):
        started.set()
        try:
            await anyio.sleep_forever()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with (
        patch.object(Docs, "aadd", aadd),
        patch.object(SearchIndex, "mark_failed_document") as mark_failed,
        anyio.fail_after(10),
    ):
        async with anyio.create_task_group() as group:
            group.start_soon(get_directory_index, settings)
            await started.wait()
            group.cancel_scope.cancel()
    assert cancelled.is_set()
    mark_failed.assert_not_called()


@pytest.mark.asyncio
async def test_failure_state_save_error_still_propagates(settings):
    settings.agent.index.continue_on_error = True
    paper = settings.agent.index.paper_directory / "failed.txt"
    paper.write_text("failed", encoding="utf-8")

    async def aadd(_docs, **_kwargs):  # noqa: RUF029
        raise http_error()

    with (
        patch.object(Docs, "aadd", aadd),
        patch.object(
            SearchIndex, "save_index", side_effect=OSError("Cannot save index")
        ),
        pytest.raises(ExceptionGroup) as caught,
    ):
        await get_directory_index(settings=settings)
    assert any(isinstance(e, OSError) for e in caught.value.exceptions)


def test_cli_continue_on_error_and_final_summary(settings, monkeypatch, caplog):
    paper = settings.agent.index.paper_directory / "failed.txt"
    paper.write_text("failed", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pqa",
            "--agent.index.continue_on_error",
            "true",
            "--agent.index.index_directory",
            str(settings.agent.index.index_directory),
            "--agent.index.name",
            "cli-issue-1275",
            "index",
            str(settings.agent.index.paper_directory),
        ],
    )

    async def aadd(_docs, **_kwargs):  # noqa: RUF029
        raise http_error()

    with patch.object(Docs, "aadd", aadd):
        main()
    assert "1 failed document(s): 0 indexed, 0 skipped" in caplog.text
    assert "HTTPStatusError: 429 Too Many Requests" in caplog.text
