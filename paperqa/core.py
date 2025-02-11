from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any, cast

from aviary.core import Message
from llmclient import LLMModel

from paperqa.types import Context, LLMResult, Text
from paperqa.utils import extract_score, strip_citations


def llm_parse_json(text: str) -> dict:
    """Read LLM output and extract JSON data from it."""
    # fetch from markdown ```json if present
    ptext = text.strip().split("```json")[-1].split("```")[0]
    # split anything before the first { after the last }
    ptext = ("{" + ptext.split("{", 1)[-1]).rsplit("}", 1)[0] + "}"

    def escape_newlines(match: re.Match) -> str:
        return match.group(0).replace("\n", "\\n")

    # Match anything between double quotes
    # including escaped quotes and other escaped characters.
    # https://regex101.com/r/VFcDmB/1
    pattern = r'"(?:[^"\\]|\\.)*"'
    ptext = re.sub(pattern, escape_newlines, ptext)
    try:
        return json.loads(ptext)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from text {text!r}. Your model may not be capable of"
            " supporting JSON output or our parsing technique could use some work. Try"
            " a different model or specify `Settings(prompts={'use_json': False})`"
        ) from e


async def map_fxn_summary(
    text: Text,
    question: str,
    summary_llm_model: LLMModel | None,
    prompt_templates: tuple[str, str] | None,
    extra_prompt_data: dict[str, str] | None = None,
    parser: Callable[[str], dict[str, Any]] | None = None,
    callbacks: Sequence[Callable[[str], None]] | None = None,
) -> tuple[Context, LLMResult]:
    """Parses the given text and returns a context object with the parser and prompt runner.

    The parser should at least return a dict with `summary`. A `relevant_score` will be used and any
    extra fields except `question` will be added to the context object. `question` is stripped
    because it can be incorrectly parsed from LLM outputs when parsing them as JSON.

    Args:
        text: The text to parse.
        question: The question to use for summarization.
        summary_llm_model: The LLM model to use for generating summaries.
        prompt_templates: Tuple containing templates for the message and system prompts.
        extra_prompt_data: Optional extra data to pass to the prompt template.
        parser: Optional parser function to parse LLM output into structured data.
            Should return dict with at least 'summary' field.
        callbacks: Optional sequence of callback functions to execute during LLM calls.

    Returns:
        The context object and LLMResult to get info about the LLM execution.
    """
    # needed empties for failures/skips
    llm_result = LLMResult(model="", date="")
    extras: dict[str, Any] = {}
    citation = text.name + ": " + text.doc.formatted_citation
    success = False

    if summary_llm_model and prompt_templates:
        data = {
            "question": question,
            "citation": citation,
            "text": text.text,
            **(extra_prompt_data or {}),
        }
        message_prompt = prompt_templates[0]
        system_prompt = prompt_templates[1]
        messages = [
            Message(role="system", content=system_prompt.format(**data)),
            Message(role="user", content=message_prompt.format(**data)),
        ]
        llm_result = await summary_llm_model.call_single(
            messages=messages,
            callbacks=callbacks,
            name="evidence:" + text.name,
        )
        context = cast(str, llm_result.text)
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
                doc=text.doc.model_dump(exclude={"embedding"}),
            ),
            score=score,  # pylint: disable=possibly-used-before-assignment
            **extras,
        ),
        llm_result,
    )
