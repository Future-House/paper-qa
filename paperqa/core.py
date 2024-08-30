from __future__ import annotations

import json
import re
from typing import Any, Callable, cast

from .llms import Chain, VectorStore
from .types import Answer, Context, DocKey, LLMResult, Text
from .utils import get_score, strip_citations


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


def llm_parse_json(text: str) -> dict:
    """Read LLM output and extract JSON data from it."""
    # fetch from markdown ```json if present
    text = text.strip().split("```json")[-1].split("```")[0]
    # split anything before the first {
    text = "{" + text.split("{", 1)[-1]
    # split anything after the last }
    text = text.rsplit("}", 1)[0] + "}"

    # escape new lines within strings
    def replace_newlines(match: re.Match) -> str:
        return match.group(0).replace("\n", "\\n")

    # Match anything between double quotes
    # including escaped quotes and other escaped characters.
    # https://regex101.com/r/VFcDmB/1
    pattern = r'"(?:[^"\\]|\\.)*"'
    text = re.sub(pattern, replace_newlines, text)

    return json.loads(text)


async def map_fxn_summary(
    text: Text,
    answer: Answer,
    chain: Chain,
    parser: Callable[[str], dict[str, Any]] | None = None,
    callbacks: list[Callable[[str], None]] | None = None,
) -> tuple[Context, LLMResult]:
    """Parses the given text and returns a context object with the parser and chain.

    The parser should at least return a dict with `summary`. A `relevant_score` will be used and any
    extra fields except `question` will be added to the context object. `question` is stripped
    because it can be incorrectly parsed from LLM outputs.

    Args:
        text: The text to parse.
        answer: Used to fill in some of the chain fields
        chain: The chain to execute - should have question, citation, summary_length, and text fields.
        parser: The parser to use for parsing - return empty dict on Failure to fallback to text parsing.
        callbacks: LLM callbacks to execute in chain

    Returns:
        The context object and LLMResult to get stats/info
    """
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
    result_data = parser(context) if parser else {}
    success = bool(result_data)
    if success:
        try:
            context = result_data.pop("summary")
            score = (
                result_data.pop("relevance_score")
                if "relevance_score" in result_data
                else get_score(context)
            )
            # just in case question was present
            result_data.pop("question", None)
            extras = result_data
        except KeyError:
            success = False
    # remove citations that collide with our grounded citations (for the answer LLM)
    context = strip_citations(context)
    if not success:
        score = get_score(context)

    c = Context(
        context=context,
        text=Text(
            text=text.text,
            name=text.name,
            doc=text.doc.__class__(**text.doc.model_dump(exclude={"embedding"})),
        ),
        score=score,
        **extras,
    )
    return c, llm_result
