from __future__ import annotations

from typing import Any

import httpx
import pytest

from paperqa.clients.crossref import get_doc_details_from_crossref
from paperqa.clients.exceptions import DOINotFoundError

pytestmark = pytest.mark.offline

TITLE = "A Perspective on Explanations of Molecular Prediction Models"


def _crossref_item(
    doi: str,
    title: str,
    published: list[int],
    article_type: str = "journal-article",
) -> dict[str, Any]:
    return {
        "DOI": doi,
        "title": [title],
        "published": {"date-parts": [published]},
        "type": article_type,
    }


def _mock_crossref_transport(items: list[dict[str, Any]]) -> httpx.MockTransport:
    response = httpx.Response(200, json={"status": "ok", "message": {"items": items}})

    def handler(_: httpx.Request) -> httpx.Response:
        return response

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_title_search_uses_stable_latest_match() -> None:
    items = [
        _crossref_item(
            "10.26434/chemrxiv-2022-qfv02",
            TITLE,
            [2022, 12, 8],
            article_type="posted-content",
        ),
        _crossref_item(
            "10.1000/newer-partial-match",
            "A Perspective on Molecular Prediction",
            [2025, 1, 1],
        ),
        _crossref_item("10.1021/acs.jctc.2c01235", TITLE, [2023, 3, 27]),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query.title"] == TITLE
        assert request.url.params["rows"] == "3"
        assert set(request.url.params["select"].split(",")) >= {
            "DOI",
            "published",
            "title",
        }
        return httpx.Response(200, json={"status": "ok", "message": {"items": items}})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        details = await get_doc_details_from_crossref(
            client=client,
            title=TITLE,
            fields=["title", "doi", "publication_date"],
        )

    assert details
    assert details.doi == "10.1021/acs.jctc.2c01235"
    assert details.title == TITLE


@pytest.mark.asyncio
async def test_title_search_rejects_unrelated_items() -> None:
    items = [
        _crossref_item(
            f"10.1000/unrelated-{index}",
            f"An unrelated paper about molecular explanations, volume {index}",
            [2020 + index],
        )
        for index in range(3)
    ]

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "message": {"items": items}})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(DOINotFoundError, match="did not match"):
            await get_doc_details_from_crossref(client=client, title=TITLE)


@pytest.mark.asyncio
async def test_title_search_is_independent_of_result_order() -> None:
    items = [
        _crossref_item("10.1000/z", TITLE, [2024, 1, 1]),
        _crossref_item("10.1000/a", TITLE, [2024, 1, 1]),
    ]
    selected_dois = []

    for candidate_items in (items, list(reversed(items))):
        transport = _mock_crossref_transport(candidate_items)
        async with httpx.AsyncClient(transport=transport) as client:
            details = await get_doc_details_from_crossref(
                client=client, title=TITLE, fields=["title", "doi"]
            )
        assert details
        selected_dois.append(details.doi)

    assert selected_dois == ["10.1000/a", "10.1000/a"]
