from __future__ import annotations

from typing import Any, cast

from .llms import VectorStore
from .types import DocKey, Text


async def retrieve(
    query: str,
    texts_index: VectorStore,
    client: Any,
    k: int = 10,  # Number of evidence pieces to retrieve
    include_dockey_filter: set[DocKey] | None = None,  # by dockey
    exclude_dockey_filter: set[DocKey] | None = None,
    include_text_filter: set[str] | None = None,  # by text
    exclude_text_filter: set[str] | None = None,
) -> list[Text]:
    _k = k
    if any(
        [
            include_dockey_filter,
            exclude_dockey_filter,
            include_text_filter,
            exclude_text_filter,
        ]
    ):
        _k = k * 10  # shitty heuristic - get enough so we can downselect
    # retrieve
    matches = cast(
        list[Text],
        (
            await texts_index.max_marginal_relevance_search(
                client, query, k=_k, fetch_k=5 * _k
            )
        )[0],
    )
    # apply filters
    if include_dockey_filter:
        matches = [m for m in matches if m.doc.dockey in include_dockey_filter]
    if exclude_dockey_filter:
        matches = [m for m in matches if m.doc.dockey not in exclude_dockey_filter]
    if include_text_filter:
        matches = [m for m in matches if m.text in include_text_filter]
    if exclude_text_filter:
        matches = [m for m in matches if m.text not in exclude_text_filter]

    return matches[:k]
