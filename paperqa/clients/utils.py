from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import reduce
from typing import Any

import aiohttp
import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_incrementing,
)

logger = logging.getLogger(__name__)

TITLE_SET_SIMILARITY_THRESHOLD = 0.75


@retry(
    retry=retry_if_exception(
        lambda x: isinstance(x, aiohttp.ServerDisconnectedError)
        or isinstance(x, aiohttp.ClientResponseError)
        and x.status
        in {
            httpx.codes.INTERNAL_SERVER_ERROR.value,
            httpx.codes.GATEWAY_TIMEOUT.value,
        }
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    stop=stop_after_attempt(5),
    wait=wait_incrementing(0.1, 0.1),
)
async def _get_with_retrying(
    url: str,
    params: dict[str, Any],
    session: aiohttp.ClientSession,
    headers: dict[str, str] | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    """Get from a Semantic Scholar API, with retrying protection."""
    async with session.get(
        url,
        params=params,
        headers=headers,
        timeout=aiohttp.ClientTimeout(timeout),
    ) as response:
        response.raise_for_status()
        return await response.json()


def union_collections_to_ordered_list(collections: Iterable) -> list:
    return sorted(reduce(lambda x, y: set(x) | set(y), collections))
