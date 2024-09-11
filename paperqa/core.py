from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from paperqa.llms import PromptRunner
from paperqa.types import Context, LLMResult, Text
from paperqa.utils import extract_score, strip_citations


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
    question: str,
    prompt_runner: PromptRunner | None,
    extra_prompt_data: dict[str, str] | None = None,
    parser: Callable[[str], dict[str, Any]] | None = None,
    callbacks: list[Callable[[str], None]] | None = None,
) -> tuple[Context, LLMResult]:
    """Parses the given text and returns a context object with the parser and prompt runner.

    The parser should at least return a dict with `summary`. A `relevant_score` will be used and any
    extra fields except `question` will be added to the context object. `question` is stripped
    because it can be incorrectly parsed from LLM outputs when parsing them as JSON.

    Args:
        text: The text to parse.
        question: The question to use for the chain.
        prompt_runner: The prompt runner to call - should have question, citation,
            summary_length, and text fields.
        extra_prompt_data: Optional extra kwargs to pass to the prompt runner's data.
        parser: The parser to use for parsing - return empty dict on Failure to fallback to text parsing.
        callbacks: LLM callbacks to execute in the prompt runner.

    Returns:
        The context object and LLMResult to get info about the LLM execution.
    """
    # needed empties for failures/skips
    llm_result = LLMResult(model="", date="")
    extras: dict[str, Any] = {}
    citation = text.name + ": " + text.doc.citation
    success = False

    if prompt_runner:
        llm_result = await prompt_runner(
            {"question": question, "citation": citation, "text": text.text}
            | (extra_prompt_data or {}),
            callbacks,
            "evidence:" + text.name,
        )
        context = llm_result.text
        result_data = parser(context) if parser else {}
        success = bool(result_data)
        if success:
            try:
                context = result_data.pop("summary")
                score = (
                    result_data.pop("relevance_score")
                    if "relevance_score" in result_data
                    else extract_score(context)
                )
                # just in case question was present
                result_data.pop("question", None)
                extras = result_data
            except KeyError:
                success = False
    else:
        context = text.text
        # If we don't assign scores, just default to 5.
        # why 5? Because we filter out 0s in another place
        # and 5/10 is the other default I could come up with
        score = 5
        success = True
    # remove citations that collide with our grounded citations (for the answer LLM)
    context = strip_citations(context)
    if not success:
        score = extract_score(context)

    return (
        Context(
            context=context,
            text=Text(
                text=text.text,
                name=text.name,
                doc=text.doc.__class__(**text.doc.model_dump(exclude={"embedding"})),
            ),
            score=score,  # pylint: disable=possibly-used-before-assignment
            **extras,
        ),
        llm_result,
    )
