import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx_aiohttp
import litellm.llms.custom_httpx.aiohttp_transport
import pytest
from lmi.utils import (
    ANTHROPIC_API_KEY_HEADER,
    CROSSREF_KEY_HEADER,
    OPENAI_API_KEY_HEADER,
    SEMANTIC_SCHOLAR_KEY_HEADER,
)

TESTS_DIR = Path(__file__).parent
CASSETTES_DIR = TESTS_DIR / "cassettes"
OFFLINE_TESTS = os.getenv("PQA_OFFLINE_TESTS") == "true"
if OFFLINE_TESTS:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["NVIDIA_API_KEY"] = "ci-cassette-placeholder"


@pytest.fixture(scope="session", name="vcr_config")
def fixture_vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [
            CROSSREF_KEY_HEADER,
            SEMANTIC_SCHOLAR_KEY_HEADER,
            OPENAI_API_KEY_HEADER,  # Works for Nvidia API key too as it's OpenAI-compatible
            ANTHROPIC_API_KEY_HEADER,
            "cookie",
        ],
        "record_mode": (
            "none" if OFFLINE_TESTS or os.getenv("GITHUB_ACTIONS") == "true" else "once"
        ),
        "cassette_library_dir": str(CASSETTES_DIR),
        # "drop_unused_requests": True,  # Restore after https://github.com/kevin1024/vcrpy/issues/961
    }


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


# Permanently patch the original response stream,
# to work around https://github.com/karpetrosyan/httpx-aiohttp/issues/23
# and https://github.com/BerriAI/litellm/issues/11724
httpx_aiohttp.transport.AiohttpResponseStream = (  # type: ignore[misc]
    litellm.llms.custom_httpx.aiohttp_transport.AiohttpResponseStream  # type: ignore[misc]
) = PreReadCompatibleAiohttpResponseStream  # type: ignore[assignment]
