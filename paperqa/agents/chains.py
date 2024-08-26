from __future__ import annotations

import logging
import re

from openai import AsyncOpenAI

from paperqa import OpenAILLMModel

from .tools import get_year

logger = logging.getLogger(__name__)


async def search_chain(
    question: str,
    count: int,
    template: str | None = None,
    llm: str = "gpt-3.5-turbo",
    temperature: float = 1.0,
) -> list[str]:
    if isinstance(template, str):
        if not (
            "{count}" in template and "{question}" in template and "{date}" in template
        ):
            logger.warning(
                "Template does not contain {count}, {question} and {date} variables. Ignoring template."
            )
            template = None

        else:
            # partial formatting
            search_prompt = template.replace(r"{date}", get_year())

    if template is None:
        search_prompt = (
            "We want to answer the following question: {question} \n"
            "Provide {count} unique keyword searches (one search per line) and year ranges "
            "that will find papers to help answer the question. "
            "Do not use boolean operators. "
            "Make sure not to repeat searches without changing the keywords or year ranges. "
            "Make some searches broad and some narrow. "
            "Use this format: [keyword search], [start year]-[end year]. "
            "where end year is optional. "
            f"The current year is {get_year()}\n\n"
        )

    if "gpt" not in llm:
        raise ValueError(
            f"Invalid llm: {llm}, note a GPT model must be used for the fake agent search."
        )
    client = AsyncOpenAI()
    model = OpenAILLMModel(config={"model": llm, "temperature": temperature})
    chain = model.make_chain(client, prompt=search_prompt, skip_system=True)
    result = await chain({"question": question, "count": count})  # type: ignore[call-arg]
    search_query = result.text
    queries = [s for s in search_query.split("\n") if len(s) > 3]  # noqa: PLR2004
    # remove 2., 3. from queries
    queries = [re.sub(r"^\d+\.\s*", "", q) for q in queries]
    # remove quotes
    return [re.sub(r"\"", "", q) for q in queries]
