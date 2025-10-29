import json
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import litellm
from aviary.core import Message
from lmi import LLMModel, LLMResult
from pydantic import JsonValue

from paperqa.prompts import text_with_tables_prompt_template
from paperqa.types import Context, Text
from paperqa.utils import extract_score, strip_citations

logger = logging.getLogger(__name__)


def llm_parse_json(text: str) -> dict[str, JsonValue]:
    """Read LLM output and extract JSON data from it."""
    # Removing <think> tags for reasoning models
    ptext = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

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
    ptext = re.sub(r'"(?:[^"\\]|\\.)*"', escape_newlines, ptext)

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
    ptext = re.sub(
        r'("\s*(?:relevance|score)[\w\s\-]*"\s*:\s*)(?:"(\d+)\s*/\s*(\d+)"|(\d+)\s*/\s*(\d+))',
        fraction_replacer,
        ptext,
    )

    # Add missing commas after fields where another key follows
    ptext = re.sub(r'(?<=[}\]0-9"])\s*(?="[^"\\]*"\s*:)', ", ", ptext)

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

        raise ValueError(f"Failed to load JSON from text {text!r}.") from e

    # Handling incorrect key names for "relevance_score"
    for key in list(data):  # List is here to copy keys, since we're in-place mutating
        if re.search(r"relevance|score", key, re.IGNORECASE):
            data["relevance_score"] = data.pop(key)  # Renaming key

    # Handling float, str values for relevance_score
    if "relevance_score" in data and not isinstance(data["relevance_score"], int):
        try:
            data["relevance_score"] = round(float(data["relevance_score"]))
        except ValueError as exc:
            raise ValueError(
                f"Failed to extract 'relevance_score' of {data['relevance_score']!r}"
                " to an integer."
            ) from exc

    return data


class LLMContextError(ValueError):
    retryable: ClassVar[bool]
    help_message: ClassVar[str]  # Eventually passed to logger.exception

    def __init__(self, message: str, llm_results: list[LLMResult]) -> None:
        super().__init__(message)
        self.llm_results = llm_results  # House so we can cost track across retries


class LLMBadContextJSONError(LLMContextError):
    """Retryable exception for when the LLM gives back bad JSON."""

    retryable = True
    help_message = (
        "Abandoning this context creation."
        " Your model may not be capable of supporting JSON output"
        " or our parsing technique could use some work. Try"
        " a different model or specify `Settings(prompts={'use_json': False})`."
        " Or, feel free to just ignore this message, as many contexts are"
        " concurrently made and we're not attached to any one given context."
    )


class LLMContextTimeoutError(LLMContextError):
    """Non-retryable exception for when the LLM call times out."""

    retryable = False
    help_message = (
        "Timeout when creating a context, abandoning it."
        " If you see this error frequently, consider increasing the timeout in"
        " Settings(summary_llm_config=...). Or, feel free to just ignore this message,"
        " as many contexts are concurrently made and we're not attached to any one"
        " given context."
    )


class LLMContextRequestFailedError(LLMContextError):
    """Non-retryable exception for when the LLM provider fails to respond.

    Kind of a catch-all for intermittent failures, safety refusals, etc.
    Catches all litellm.BadRequestErrors and litellm.MidStreamFallbackErrors.
    """

    retryable = False
    help_message = (
        "Response error when creating a context, abandoning it."
        " If you see this error frequently, the summary_llm endpoint is either"
        " misconfigured or is having issues."
    )


async def _map_fxn_summary(  # noqa: PLR0912
    text: Text,
    question: str,
    summary_llm_model: LLMModel | None,
    prompt_templates: tuple[str, str] | None,
    extra_prompt_data: dict[str, str] | None = None,
    parser: Callable[[str], dict[str, Any]] | None = None,
    callbacks: Sequence[Callable[[str], None]] | None = None,
    skip_citation_strip: bool = False,
    evidence_text_only_fallback: bool = False,
    _prior_attempt: LLMContextError | None = None,
) -> tuple[Context, list[LLMResult]]:
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
        _prior_attempt: Optional failure from a prior attempt, for LLM result tracking.

    Returns:
        A two-tuple of the made Context, and any LLM results made along the way.
    """
    if _prior_attempt is not None:
        llm_results = _prior_attempt.llm_results
        append_msgs = [
            Message(
                content=(
                    "In a prior attempt, we failed with this failure message:"
                    f" {_prior_attempt!s}."
                )
            )
        ]
    else:
        llm_results, append_msgs = [], []
    extras: dict[str, Any] = {}
    citation = text.name + ": " + text.doc.formatted_citation
    used_text_only_fallback = False

    # Strip newlines in case chunking led to blank lines,
    # but not spaces, to preserve text alignment
    cleaned_text = text.text.strip("\n") or "(no text)"
    if summary_llm_model and prompt_templates:
        unique_media = list(dict.fromkeys(text.media))  # Preserve order
        table_texts: list[str] = [
            m.text for m in unique_media if m.info.get("type") == "table" and m.text
        ]
        data = {
            "question": question,
            "citation": citation,
            "text": (
                text_with_tables_prompt_template.format(
                    text=cleaned_text,
                    citation=citation,
                    tables="\n\n".join(table_texts),
                )
                if table_texts
                else cleaned_text
            ),
        } | (extra_prompt_data or {})
        message_prompt, system_prompt = (pt.format(**data) for pt in prompt_templates)
        try:
            try:
                llm_result = await summary_llm_model.call_single(
                    messages=[
                        Message(role="system", content=system_prompt),
                        Message.create_message(
                            text=message_prompt,
                            images=(
                                [i.to_image_url() for i in unique_media]
                                if unique_media
                                else None
                            ),
                        ),
                        *append_msgs,
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
                        *append_msgs,
                    ],
                    callbacks=callbacks,
                    name="evidence:" + text.name,
                )
                used_text_only_fallback = True
        except litellm.Timeout as exc:
            raise LLMContextTimeoutError(
                f"LLM call to create a context timed out on text named {text.name!r}.",
                llm_results=llm_results,
            ) from exc
        except (
            litellm.exceptions.MidStreamFallbackError,
            litellm.BadRequestError,
        ) as exc:
            # BadRequestError: what is thrown if you directly call an LLM with a bad request
            # MidStreamFallbackError: what litellm throws if there are fallbacks configured
            raise LLMContextRequestFailedError(
                f"LLM call to create a context failed on text named {text.name!r}.",
                llm_results=llm_results,
            ) from exc

        llm_results.append(llm_result)
        context = llm_result.text or ""
        if parser:
            try:
                result_data = parser(context)
            except ValueError as exc:
                raise LLMBadContextJSONError(
                    f"Failed to parse JSON from context {context!r} due to: {exc}",
                    llm_results=llm_results,
                ) from exc
            try:
                context = result_data.pop("summary")
                try:
                    score = (
                        result_data.pop("relevance_score")
                        if "relevance_score" in result_data
                        else extract_score(context)
                    )
                except ValueError as exc:
                    raise LLMBadContextJSONError(
                        f"Successfully parsed JSON and extracted 'summary' key,"
                        f" but then failed to extract score from context {context!r} due to: {exc}",
                        llm_results=llm_results,
                    ) from exc
                # just in case question was present
                result_data.pop("question", None)
                extras = result_data
            except KeyError:  # No summary key, so extract from LLM result
                try:
                    score = extract_score(context)
                except ValueError as exc:
                    raise LLMBadContextJSONError(
                        f"Successfully parsed JSON but it had no 'summary' key."
                        f" Then, the failover to extract score from raw context {context!r}"
                        f" failed due to: {exc}",
                        llm_results=llm_results,
                    ) from exc
        else:
            try:
                score = extract_score(context)
            except ValueError as exc:
                raise LLMBadContextJSONError(
                    f"Extracting score from raw context {context!r} failed due to: {exc}",
                    llm_results=llm_results,
                ) from exc
    else:
        llm_results.append(LLMResult(model="", date=""))
        context = cleaned_text
        # If we don't assign scores, just default to 5.
        # why 5? Because we filter out 0s in another place
        # and 5/10 is the only default I could come up with
        score = 5
    # remove citations that collide with our grounded citations (for the answer LLM)
    if not skip_citation_strip:
        context = strip_citations(context)

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
            score=score,
            **extras,
        ),
        llm_results,
    )


async def map_fxn_summary(**kwargs) -> tuple[Context | None, list[LLMResult]]:
    if "_prior_attempt" in kwargs:
        raise ValueError(
            "_prior_attempt is reserved for internal use only, don't specify it."
        )
    try:
        return await _map_fxn_summary(**kwargs)
    except LLMContextError as exc:
        if not exc.retryable:
            logger.exception(
                "Non-retryable failure creating a context. %s", exc.help_message
            )
            return None, exc.llm_results
        try:
            return await _map_fxn_summary(**kwargs, _prior_attempt=exc)
        except LLMContextError as exc2:
            logger.exception("Failed twice to create a context. %s", exc2.help_message)
            return None, exc2.llm_results
