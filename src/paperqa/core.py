from __future__ import annotations

import contextlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, cast

import litellm
from aviary.core import Message
from lmi import LLMModel

from paperqa.prompts import text_with_tables_prompt_template
from paperqa.types import Context, LLMResult, Text
from paperqa.utils import extract_score, strip_citations

logger = logging.getLogger(__name__)


def llm_parse_json(text: str) -> dict:
    """Read LLM output and extract JSON data from it."""
    # Remove leading/trailing whitespaces
    ptext = text.strip()

    # Removing <think> tags for reasoning models
    ptext = re.sub(r"<think>.*?</think>", "", ptext, flags=re.DOTALL).strip()

    # fetches from markdown ```json if present
    ptext = ptext.split("```json")[-1].split("```")[0]

    # Fix specific case with raw fractions in relevance_score
    ptext = re.sub(
        r'"relevance_score"\s*:\s*(\d+)/(\d+)',
        lambda m: f'"relevance_score": {round(int(m.group(1)) / int(m.group(2)) * 10)}',
        ptext,
    )

    # Wrap non-JSON text in a dictionary
    if "{" not in ptext and "}" not in ptext:
        ptext = json.dumps({"summary": ptext})

    # Remove any introductory/closing text and ensure {} to make it a valid JSON
    ptext = ("{" + ptext.split("{", 1)[-1]).rsplit("}", 1)[0] + "}"

    def escape_newlines(match: re.Match) -> str:
        return match.group(0).replace("\n", "\\n")

    # Match anything between double quotes
    # including escaped quotes and other escaped characters.
    # https://regex101.com/r/VFcDmB/1
    pattern = r'"(?:[^"\\]|\\.)*"'
    ptext = re.sub(pattern, escape_newlines, ptext)

    # Ensure that any backslashes in the string that are not part
    # of a valid escape sequence are properly escaped
    # https://regex101.com/r/IzMDlI/1
    ptext = re.sub(r'\\([^"\\/bfnrtu])', r"\\\\\1", ptext)

    def fraction_replacer(match: re.Match) -> str:
        key = match.group(1)  # The key (unchanged)

        # Case 1: If quoted fraction `"5/10"`
        if match.group(2) and match.group(3):
            numerator = int(match.group(2))
            denominator = int(match.group(3))

        # Case 2: If unquoted fraction `5/10`
        elif match.group(4) and match.group(5):
            numerator = int(match.group(4))
            denominator = int(match.group(5))

        else:
            return match.group(0)  # No change if no fraction is found

        fraction_value = round(numerator / denominator * 10)  # Convert to integer
        return f"{key}{fraction_value}"

    # Replace X/Y scores with integer value from 0-10
    # e.g. "relevance_score": "8/10" -> "relevance_score": 8
    # e.g. "relevance_score": 3/5 -> "relevance_score": 6
    pattern = r'("\s*(?:relevance|score)[\w\s\-]*"\s*:\s*)(?:"(\d+)\s*/\s*(\d+)"|(\d+)\s*/\s*(\d+))'
    ptext = re.sub(pattern, fraction_replacer, ptext)

    # Remove extra commas
    ptext = re.sub(r",\s*,+", ",", ptext)  # Remove multiple consecutive commas
    ptext = re.sub(r",\s*}", "}", ptext)  # Remove trailing commas before closing brace
    ptext = re.sub(r"\{\s*,", "{", ptext)  # Remove leading commas inside object

    # Try to parse the JSON normally first
    try:
        data = json.loads(ptext)
    except json.JSONDecodeError as e:
        # If normal parsing fails, try to handle nested quotes case
        if "summary" in ptext and '"relevance_score"' in ptext:
            with contextlib.suppress(
                Exception  # Continue to the standard error if regex approach fails
            ):
                # Extract summary and relevance_score directly using regex
                summary_match = re.search(
                    r'"summary"\s*:\s*"(.*?)",\s*"relevance_score"', ptext, re.DOTALL
                )
                score_match = re.search(r'"relevance_score"\s*:\s*"?(\d+)"?', ptext)

                if summary_match and score_match:
                    return {
                        "summary": summary_match.group(1).replace(r"\'", "'"),
                        "relevance_score": int(score_match.group(1)),
                    }

        raise ValueError(
            f"Failed to parse JSON from text {text!r}. Your model may not be capable of"
            " supporting JSON output or our parsing technique could use some work. Try"
            " a different model or specify `Settings(prompts={'use_json': False})`"
        ) from e

    # Handling incorrect key names for "relevance_score"
    for key in list(data.keys()):
        if re.search(r"relevance|score", key, re.IGNORECASE):
            data["relevance_score"] = data.pop(key)  # Renaming key

    # Handling float, str values for relevance_score
    if "relevance_score" in data:
        try:
            data["relevance_score"] = round(float(data["relevance_score"]))

        except ValueError:
            data["relevance_score"] = (
                0  # Default if relevance_score is empty/not a number
            )

    return data


async def map_fxn_summary(
    text: Text,
    question: str,
    summary_llm_model: LLMModel | None,
    prompt_templates: tuple[str, str] | None,
    extra_prompt_data: dict[str, str] | None = None,
    parser: Callable[[str], dict[str, Any]] | None = None,
    callbacks: Sequence[Callable[[str], None]] | None = None,
    skip_citation_strip: bool = False,
    evidence_text_only_fallback: bool = False,
) -> tuple[Context, LLMResult]:
    """Parses the given text and returns a context object with the parser and prompt runner.

    The parser should at least return a dict with `summary`. A `relevant_score` will be used and any
    extra fields except `question` will be added to the context object. `question` is stripped
    because it can be incorrectly parsed from LLM outputs when parsing them as JSON.

    Args:
        text: The text to parse.
        question: The question to use for summarization.
        summary_llm_model: The LLM model to use for generating summaries.
        prompt_templates: Optional two-elements tuple containing templates for the user and system prompts.
            prompt_templates = (user_prompt_template, system_prompt_template)
        extra_prompt_data: Optional extra data to pass to the prompt template.
        parser: Optional parser function to parse LLM output into structured data.
            Should return dict with at least 'summary' field.
        callbacks: Optional sequence of callback functions to execute during LLM calls.
        skip_citation_strip: Optional skipping of citation stripping, if you want to keep in the context.
        evidence_text_only_fallback: Opt-in flag to allow retrying context creation
            without media in the completion.

    Returns:
        The context object and LLMResult to get info about the LLM execution.
    """
    # needed empties for failures/skips
    llm_result = LLMResult(model="", date="")
    extras: dict[str, Any] = {}
    citation = text.name + ": " + text.doc.formatted_citation
    success = False
    used_text_only_fallback = False

    # Strip newlines in case chunking led to blank lines,
    # but not spaces, to preserve text alignment
    cleaned_text = text.text.strip("\n")
    if summary_llm_model and prompt_templates:
        media_text: list[str] = [m.text for m in text.media if m.text]
        data = {
            "question": question,
            "citation": citation,
            "text": (
                text_with_tables_prompt_template.format(
                    text=cleaned_text,
                    citation=citation,
                    tables="\n\n----\n\n".join(media_text),
                )
                if media_text
                else cleaned_text
            ),
        } | (extra_prompt_data or {})
        message_prompt, system_prompt = (pt.format(**data) for pt in prompt_templates)
        try:
            llm_result = await summary_llm_model.call_single(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message.create_message(
                        text=message_prompt,
                        images=(
                            [i.to_image_url() for i in text.media]
                            if text.media
                            else None
                        ),
                    ),
                ],
                callbacks=callbacks,
                name="evidence:" + text.name,
            )
        except litellm.BadRequestError as exc:
            if not evidence_text_only_fallback:
                raise
            logger.warning(
                f"LLM call to create a context failed with exception {exc!r}"
                f" on text named {text.name!r}"
                f" with doc name {text.doc.docname!r} and doc key {text.doc.dockey!r}."
                f" Retrying without media."
            )
            llm_result = await summary_llm_model.call_single(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(content=message_prompt),
                ],
                callbacks=callbacks,
                name="evidence:" + text.name,
            )
            used_text_only_fallback = True
        context = cast("str", llm_result.text)
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
        context = cleaned_text
        # If we don't assign scores, just default to 5.
        # why 5? Because we filter out 0s in another place
        # and 5/10 is the other default I could come up with
        score = 5
        success = True
    # remove citations that collide with our grounded citations (for the answer LLM)
    if not skip_citation_strip:
        context = strip_citations(context)

    if not success:
        score = extract_score(context)
    if used_text_only_fallback:
        extras["used_images"] = False

    return (
        Context(
            context=context,
            question=question,
            text=Text(
                # Embeddings enable the retrieval of Texts to make Contexts.
                # Once we already have Contexts, we filter them by score
                # (and not the underlying Text's embeddings),
                # so embeddings can be safely dropped from the deepcopy
                doc=text.doc.model_dump(exclude={"embedding"}),
                **text.model_dump(exclude={"embedding", "doc"}),
            ),
            score=score,  # pylint: disable=possibly-used-before-assignment
            **extras,
        ),
        llm_result,
    )
