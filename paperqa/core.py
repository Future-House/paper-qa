from __future__ import annotations

import json
from typing import Any, Tuple, cast

from .llms import Chain, VectorStore
from .types import Answer, CallbackFactory, Context, DocKey, LLMResult, Text
from .utils import get_score, llm_read_json, strip_citations


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


async def map_fxn_summary_text(
    text: Text,
    answer: Answer,
    chain: Chain,
    callbacks: list[callable[[str, str], None]] | None = None,
) -> tuple[Context, LLMResult]:
    callbacks = callbacks or []
    # needed empties for failures/skips
    llm_result = LLMResult(model="", date="")
    extras: dict[str, Any] = {}
    citation = text.name + ": " + text.doc.citation
    llm_result = await chain(
        {
            "question": answer.question,
            "citation": citation,
            "summary_length": answer.summary_length,
            "text": text.text,
        },
        callbacks,
    )
    llm_result.answer_id = answer.id
    llm_result.name = "evidence:" + text.name
    context = llm_result.text

    # remove citations that collide with our grounded citations (for the answer LLM)
    context = strip_citations(context)
    score = get_score(context)

    c = Context(
        context=context,
        text=Text(
            text=text.text,
            name=text.name,
            doc=text.doc.__class__(**text.doc.model_dump(exclude="embedding")),
        ),
        score=score,
        **extras,
    )
    return c, llm_result


async def map_fxn_summary_json(
    text: Text,
    answer: Answer,
    chain: Chain,
    callbacks: list[callable[[str, str], None]] | None = None,
) -> tuple[Context, LLMResult]:
    callbacks = callbacks or []
    # needed empties for failures/skips
    llm_result = LLMResult(model="", date="")
    extras: dict[str, Any] = {}
    citation = text.name + ": " + text.doc.citation
    llm_result = await chain(
        {
            "question": answer.question,
            "citation": citation,
            "summary_length": answer.summary_length,
            "text": text.text,
        },
        callbacks,
    )
    llm_result.answer_id = answer.id
    llm_result.name = "evidence:" + text.name
    context = llm_result.text
    try:
        result_data = llm_read_json(context)
    except json.decoder.JSONDecodeError:
        # fallback to string
        success = False
    else:
        success = isinstance(result_data, dict)
    if success:
        try:
            context = result_data.pop("summary")
            score = result_data.pop("relevance_score")
            result_data.pop("question", None)
            extras = result_data
        except KeyError:
            success = False
    # remove citations that collide with our grounded citations (for the answer LLM)
    context = strip_citations(context)
    score = get_score(context)

    c = Context(
        context=context,
        text=Text(
            text=text.text,
            name=text.name,
            doc=text.doc.__class__(**text.doc.model_dump(exclude="embedding")),
        ),
        score=score,
        **extras,
    )
    return c, llm_result
