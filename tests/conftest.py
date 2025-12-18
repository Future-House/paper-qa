from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from collections.abc import AsyncIterator, Coroutine, Iterator
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx_aiohttp
import litellm.llms.custom_httpx.aiohttp_transport
import pytest
import vcr.stubs.aiohttp_stubs
import vcr.stubs.httpcore_stubs
from dotenv import load_dotenv
from lmi.utils import (
    ANTHROPIC_API_KEY_HEADER,
    CROSSREF_KEY_HEADER,
    OPENAI_API_KEY_HEADER,
    SEMANTIC_SCHOLAR_KEY_HEADER,
    update_litellm_max_callbacks,
)

if TYPE_CHECKING:
    from paperqa.settings import Settings
    from paperqa.types import PQASession

TESTS_DIR = Path(__file__).parent
CASSETTES_DIR = TESTS_DIR / "cassettes"

IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(autouse=True, scope="session")
def _load_env() -> None:
    load_dotenv()


@pytest.fixture(autouse=True, scope="session")
def _setup_default_logs() -> None:
    # Lazily import from paperqa so typeguard doesn't throw:
    # > /path/to/.venv/lib/python3.12/site-packages/typeguard/_pytest_plugin.py:93:
    # > InstrumentationWarning: typeguard cannot check these packages because they
    # > are already imported: paperqa
    from paperqa.settings import ParsingSettings
    from paperqa.utils import setup_default_logs

    setup_default_logs()
    ParsingSettings.model_fields["configure_pdf_parser"].default()


@pytest.fixture(autouse=True, scope="session")
def _defeat_litellm_callbacks() -> None:
    update_litellm_max_callbacks()


@pytest.fixture(autouse=True, scope="session")
def _patch_litellm_logging_worker_for_race_condition() -> Iterator[None]:
    """
    Patch litellm's GLOBAL_LOGGING_WORKER for asyncio functionality.

    SEE: https://github.com/BerriAI/litellm/issues/16518
    SEE: https://github.com/BerriAI/litellm/issues/14521
    """
    try:
        from litellm.litellm_core_utils import logging_worker
    except ImportError:
        if tuple(int(x) for x in version(litellm.__name__).split(".")) < (1, 76, 0):
            # Module didn't exist before https://github.com/BerriAI/litellm/pull/13905
            yield
            return
        raise

    class NoOpLoggingWorker:
        """No-op worker that executes callbacks immediately without queuing."""

        def start(self) -> None:
            pass

        def enqueue(self, coroutine: Coroutine) -> None:
            # Execute immediately in current loop instead of queueing,
            # and do nothing if there's no current loop
            with contextlib.suppress(RuntimeError):
                # This logging task is fire-and-forget
                asyncio.create_task(  # type: ignore[unused-awaitable]  # noqa: RUF006
                    coroutine
                )

        def ensure_initialized_and_enqueue(self, async_coroutine: Coroutine) -> None:
            self.enqueue(async_coroutine)

        async def stop(self) -> None:
            pass

        async def flush(self) -> None:
            pass

        async def clear_queue(self) -> None:
            pass

    with patch.object(logging_worker, "GLOBAL_LOGGING_WORKER", NoOpLoggingWorker()):
        yield


@pytest.fixture(scope="session", name="vcr_config")
def fixture_vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [
            CROSSREF_KEY_HEADER,
            SEMANTIC_SCHOLAR_KEY_HEADER,
            OPENAI_API_KEY_HEADER,
            ANTHROPIC_API_KEY_HEADER,
            "cookie",
        ],
        "record_mode": "once" if not IN_GITHUB_ACTIONS else "none",
        "allow_playback_repeats": True,
        "cassette_library_dir": str(CASSETTES_DIR),
        # "drop_unused_requests": True,  # Restore after https://github.com/kevin1024/vcrpy/issues/961
    }


@pytest.fixture(name="tmp_path_cleanup")
def fixture_tmp_path_cleanup(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path
    # Cleanup after the test
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture(name="agent_home_dir")
def fixture_agent_home_dir(
    tmp_path_cleanup: str | os.PathLike,
) -> Iterator[str | os.PathLike]:
    """Set up a unique temporary folder for the agent module."""
    with patch.dict("os.environ", {"PQA_HOME": str(tmp_path_cleanup)}):
        yield tmp_path_cleanup


@pytest.fixture(name="agent_index_dir")
def fixture_agent_index_dir(agent_home_dir: Path) -> Path:
    return agent_home_dir / ".pqa" / "indexes"


@pytest.fixture(scope="session", name="stub_data_dir")
def fixture_stub_data_dir() -> Path:
    return Path(__file__).parent / "stub_data"


@pytest.fixture
def agent_test_settings(agent_index_dir: Path, stub_data_dir: Path) -> Settings:
    # Lazily import from paperqa so typeguard doesn't throw:
    # > /path/to/.venv/lib/python3.12/site-packages/typeguard/_pytest_plugin.py:93:
    # > InstrumentationWarning: typeguard cannot check these packages because they
    # > are already imported: paperqa
    from paperqa.settings import Settings

    # NOTE: originally here we had usage of embedding="sparse", but this was
    # shown to be too crappy of an embedding to get past the Obama article
    settings = Settings()
    settings.agent.index.paper_directory = stub_data_dir
    settings.agent.index.index_directory = agent_index_dir
    settings.agent.search_count = 2
    settings.answer.answer_max_sources = 2
    settings.answer.evidence_k = 10
    return settings


@pytest.fixture
def agent_stub_session() -> PQASession:
    # Lazily import from paperqa so typeguard doesn't throw:
    # > /path/to/.venv/lib/python3.12/site-packages/typeguard/_pytest_plugin.py:93:
    # > InstrumentationWarning: typeguard cannot check these packages because they
    # > are already imported: paperqa
    from paperqa.types import PQASession

    return PQASession(question="What is a self-explanatory model?")


@pytest.fixture
def stub_data_dir_w_near_dupes(stub_data_dir: Path, tmp_path: Path) -> Iterator[Path]:

    # add some near duplicate files then removes them after testing
    for filename in ("bates.txt", "obama.txt"):
        if not (tmp_path / f"{filename}_modified.txt").exists():
            with (stub_data_dir / filename).open() as f:
                content = f.read()
            with (tmp_path / f"{Path(filename).stem}_modified.txt").open("w") as f:
                f.write(content)
                f.write("## MODIFIED FOR HASH")

    yield tmp_path

    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


class PreReadCompatibleAiohttpResponseStream(
    httpx_aiohttp.transport.AiohttpResponseStream
):
    """aiohttp-backed response stream that works if the response was pre-read."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        with httpx_aiohttp.transport.map_aiohttp_exceptions():
            if self._aiohttp_response._body is not None:
                # Happens if some intermediary called `await _aiohttp_response.read()`
                # TODO: take into account chunk size
                yield self._aiohttp_response._body
            else:
                async for chunk in self._aiohttp_response.content.iter_chunked(
                    self.CHUNK_SIZE
                ):
                    yield chunk


async def _vcr_handle_async_request(
    cassette,  # noqa: ARG001
    real_handle_async_request,
    self,
    real_request,
):
    """VCR handler that only sends, not possibly recording or playing back responses."""
    return await real_handle_async_request(self, real_request)


# Permanently patch the original response stream,
# to work around https://github.com/karpetrosyan/httpx-aiohttp/issues/23
# and https://github.com/BerriAI/litellm/issues/11724
httpx_aiohttp.transport.AiohttpResponseStream = (  # type: ignore[misc]
    litellm.llms.custom_httpx.aiohttp_transport.AiohttpResponseStream  # type: ignore[misc]
) = PreReadCompatibleAiohttpResponseStream  # type: ignore[assignment]

# Permanently patch vcrpy's async VCR recording functionality,
# to work around https://github.com/kevin1024/vcrpy/issues/944
vcr.stubs.httpcore_stubs._vcr_handle_async_request = _vcr_handle_async_request

# Permanently patch vcrpy's aiohttp build_response to set raw_headers,
# to work around https://github.com/kevin1024/vcrpy/issues/970
_original_aiohttp_stubs_build_response = vcr.stubs.aiohttp_stubs.build_response


def _build_response_with_raw_headers(vcr_request, vcr_response, history):
    """Patched build_response that also sets _raw_headers on MockClientResponse."""
    response = _original_aiohttp_stubs_build_response(
        vcr_request, vcr_response, history
    )
    if response._raw_headers is None and response._headers is not None:
        response._raw_headers = tuple(
            (k.encode("utf-8"), v.encode("utf-8")) for k, v in response._headers.items()
        )
    return response


vcr.stubs.aiohttp_stubs.build_response = _build_response_with_raw_headers
