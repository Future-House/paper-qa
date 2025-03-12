from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import cast

from aviary.core import Message
from lmi import LiteLLMModel, LLMModel
from rich.table import Table

from paperqa.docs import Docs
from paperqa.types import DocDetails

from .models import AnswerResponse

logger = logging.getLogger(__name__)


def get_year(ts: datetime | None = None) -> str:
    """Get the year from the input datetime, otherwise using the current datetime."""
    if ts is None:
        ts = datetime.now()
    return ts.strftime("%Y")


async def litellm_get_search_query(
    question: str,
    count: int,
    template: str | None = None,
    llm: LLMModel | str = "gpt-4o-mini",
    temperature: float = 1.0,
) -> list[str]:
    search_prompt = ""
    if isinstance(template, str) and all(
        x in template for x in ("{count}", "{question}", "{date}")
    ):
        # partial formatting
        search_prompt = template.replace("{date}", get_year())
    elif isinstance(template, str):
        logger.warning(
            "Template does not contain {count}, {question} and {date} variables."
            " Ignoring template and using default search prompt."
        )
    if not search_prompt:
        # TODO: move to use tools instead of DIY schema in prompt
        search_prompt = (
            "We want to answer the following question: {question}\nProvide"
            " {count} unique keyword searches (one search per line) and year ranges"
            " that will find papers to help answer the question. Do not use boolean"
            " operators. Make sure not to repeat searches without changing the"
            " keywords or year ranges. Make some searches broad and some narrow. Use"
            " this format: [keyword search], [start year]-[end year]. where end year"
            f" is optional. The current year is {get_year()}."
        )

    if isinstance(llm, str):
        model: LLMModel = LiteLLMModel(name=llm)
        model.config["model_list"][0]["litellm_params"].update(
            {"temperature": temperature}
        )
    else:
        model = llm
    messages = [
        Message(content=search_prompt.format(question=question, count=count)),
    ]
    result = await model.call_single(
        messages=messages,
    )
    search_query = cast("str", result.text)
    queries = [s for s in search_query.split("\n") if len(s) > 3]  # noqa: PLR2004
    # remove "2.", "3.", etc. -- https://regex101.com/r/W2f7F1/1
    queries = [re.sub(r"^\d+\.\s*", "", q) for q in queries]
    # remove quotes
    return [re.sub(r'["\[\]]', "", q) for q in queries]


def table_formatter(
    objects: list[tuple[AnswerResponse | Docs, str]], max_chars_per_column: int = 2000
) -> Table:
    example_object, _ = objects[0]
    if isinstance(example_object, AnswerResponse):
        table = Table(title="Prior Answers")
        table.add_column("Question", style="cyan")
        table.add_column("Answer", style="magenta")
        for obj, _ in objects:
            table.add_row(
                cast("AnswerResponse", obj).session.question[:max_chars_per_column],
                cast("AnswerResponse", obj).session.answer[:max_chars_per_column],
            )
        return table
    if isinstance(example_object, Docs):
        table = Table(title="PDF Search")
        table.add_column("Title", style="cyan")
        table.add_column("File", style="magenta")
        for obj, filename in objects:
            docs = cast("Docs", obj)  # Assume homogeneous objects
            doc = docs.texts[0].doc
            if isinstance(doc, DocDetails) and doc.title:
                display_name: str = doc.title  # Prefer title if available
            else:
                display_name = doc.formatted_citation
            table.add_row(display_name[:max_chars_per_column], filename)
        return table
    raise NotImplementedError(
        f"Object type {type(example_object)} can not be converted to table."
    )
