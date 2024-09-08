from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, cast

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from rich.table import Table

from .. import (
    AnthropicLLMModel,
    Docs,
    OpenAILLMModel,
    embedding_model_factory,
    llm_model_factory,
)
from ..llms import LangchainLLMModel
from .models import AnswerResponse, QueryRequest

logger = logging.getLogger(__name__)


def get_year(ts: datetime | None = None) -> str:
    """Get the year from the input datetime, otherwise using the current datetime."""
    if ts is None:
        ts = datetime.now()
    return ts.strftime("%Y")


async def openai_get_search_query(
    question: str,
    count: int,
    template: str | None = None,
    llm: str = "gpt-4o-mini",
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
            search_prompt = template.replace("{date}", get_year())

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
            f"The current year is {get_year()}."
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
    # remove "2.", "3.", etc. -- https://regex101.com/r/W2f7F1/1
    queries = [re.sub(r"^\d+\.\s*", "", q) for q in queries]
    # remove quotes
    return [re.sub(r"\"", "", q) for q in queries]


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
                cast(AnswerResponse, obj).answer.question[:max_chars_per_column],
                cast(AnswerResponse, obj).answer.answer[:max_chars_per_column],
            )
        return table
    if isinstance(example_object, Docs):
        table = Table(title="PDF Search")
        table.add_column("Title", style="cyan")
        table.add_column("File", style="magenta")
        for obj, filename in objects:
            table.add_row(
                cast(Docs, obj).texts[0].doc.title[:max_chars_per_column], filename  # type: ignore[attr-defined]
            )
        return table
    raise NotImplementedError(
        f"Object type {type(example_object)} can not be converted to table."
    )


# Index 0 is for prompt tokens, index 1 is for completion tokens
costs: dict[str, tuple[float, float]] = {
    "claude-2": (11.02 / 10**6, 32.68 / 10**6),
    "claude-instant-1": (1.63 / 10**6, 5.51 / 10**6),
    "claude-3-sonnet-20240229": (3 / 10**6, 15 / 10**6),
    "claude-3-5-sonnet-20240620": (3 / 10**6, 15 / 10**6),
    "claude-3-opus-20240229": (15 / 10**6, 75 / 10**6),
    "babbage-002": (0.0004 / 10**3, 0.0004 / 10**3),
    "gpt-3.5-turbo": (0.0010 / 10**3, 0.0020 / 10**3),
    "gpt-3.5-turbo-1106": (0.0010 / 10**3, 0.0020 / 10**3),
    "gpt-3.5-turbo-0613": (0.0010 / 10**3, 0.0020 / 10**3),
    "gpt-3.5-turbo-0301": (0.0010 / 10**3, 0.0020 / 10**3),
    "gpt-3.5-turbo-0125": (0.0005 / 10**3, 0.0015 / 10**3),
    "gpt-4-1106-preview": (0.010 / 10**3, 0.030 / 10**3),
    "gpt-4-0125-preview": (0.010 / 10**3, 0.030 / 10**3),
    "gpt-4-turbo-2024-04-09": (10 / 10**6, 30 / 10**6),
    "gpt-4-turbo": (10 / 10**6, 30 / 10**6),
    "gpt-4": (0.03 / 10**3, 0.06 / 10**3),
    "gpt-4-0613": (0.03 / 10**3, 0.06 / 10**3),
    "gpt-4-0314": (0.03 / 10**3, 0.06 / 10**3),
    "gpt-4o": (2.5 / 10**6, 10 / 10**6),
    "gpt-4o-2024-05-13": (5 / 10**6, 15 / 10**6),
    "gpt-4o-2024-08-06": (2.5 / 10**6, 10 / 10**6),
    "gpt-4o-mini": (0.15 / 10**6, 0.60 / 10**6),
    "gemini-1.5-flash": (0.35 / 10**6, 0.35 / 10**6),
    "gemini-1.5-pro": (3.5 / 10**6, 10.5 / 10**6),
    # supported Anyscale models per
    # https://docs.anyscale.com/endpoints/text-generation/query-a-model
    "meta-llama/Meta-Llama-3-8B-Instruct": (0.15 / 10**6, 0.15 / 10**6),
    "meta-llama/Meta-Llama-3-70B-Instruct": (1.0 / 10**6, 1.0 / 10**6),
    "mistralai/Mistral-7B-Instruct-v0.1": (0.15 / 10**6, 0.15 / 10**6),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (1.0 / 10**6, 1.0 / 10**6),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": (1.0 / 10**6, 1.0 / 10**6),
}


def compute_model_token_cost(model: str, tokens: int, is_completion: bool) -> float:
    if model in costs:  # Prefer our internal costs model
        model_costs: tuple[float, float] = costs[model]
    else:
        logger.warning(f"Model {model} not found in costs.")
        return 0.0
    return tokens * model_costs[int(is_completion)]


def compute_total_model_token_cost(token_counts: dict[str, list[int]]) -> float:
    """Sum the token counts for each model and return the total cost."""
    cost = 0.0
    for model, tokens in token_counts.items():
        if sum(tokens) > 0:
            cost += compute_model_token_cost(
                model, tokens=tokens[0], is_completion=False
            ) + compute_model_token_cost(model, tokens=tokens[1], is_completion=True)
    return cost


# the defaults here should be (about) the same as in QueryRequest
def update_doc_models(doc: Docs, request: QueryRequest | None = None):
    if request is None:
        request = QueryRequest()
    client: Any = None

    if request.llm.startswith("gemini"):
        doc.llm_model = LangchainLLMModel(name=request.llm)
        doc.summary_llm_model = LangchainLLMModel(name=request.summary_llm)
    else:
        doc.llm_model = llm_model_factory(request.llm)
        doc.summary_llm_model = llm_model_factory(request.summary_llm)

    # set temperatures
    doc.llm_model.config["temperature"] = request.temperature
    doc.summary_llm_model.config["temperature"] = request.temperature

    if isinstance(doc.llm_model, OpenAILLMModel):
        if request.llm.startswith(
            ("meta-llama/Meta-Llama-3-", "mistralai/Mistral-", "mistralai/Mixtral-")
        ):
            client = AsyncOpenAI(
                base_url=os.environ.get("ANYSCALE_BASE_URL"),
                api_key=os.environ.get("ANYSCALE_API_KEY"),
            )
            logger.info(f"Using Anyscale (via OpenAI client) for {request.llm}")
        else:
            client = AsyncOpenAI()
    elif isinstance(doc.llm_model, AnthropicLLMModel):
        client = AsyncAnthropic()
    elif isinstance(doc.llm_model, LangchainLLMModel):
        from langchain_google_vertexai import (
            ChatVertexAI,
            HarmBlockThreshold,
            HarmCategory,
        )

        # we have to convert system to human because system is unsupported
        # Also we do get blocked content, so adjust thresholds
        client = ChatVertexAI(
            model=request.llm,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
            convert_system_message_to_human=True,
        )
    else:
        raise TypeError(f"Unsupported LLM model: {doc.llm_model}")

    doc._client = client  # set client, since could be just unpickled.
    doc._embedding_client = AsyncOpenAI()  # hard coded to OpenAI for now

    doc.texts_index.embedding_model = embedding_model_factory(
        request.embedding, **(request.texts_index_embedding_config or {})
    )
    doc.docs_index.embedding_model = embedding_model_factory(
        request.embedding, **(request.docs_index_embedding_config or {})
    )
    doc.texts_index.mmr_lambda = request.texts_index_mmr_lambda
    doc.docs_index.mmr_lambda = request.docs_index_mmr_lambda
    doc.embedding = request.embedding
    doc.max_concurrent = request.max_concurrent
    doc.prompts = request.prompts
    Docs.make_llm_names_consistent(doc)

    logger.debug(
        f"update_doc_models: {doc.name}"
        f" | {(doc.llm_model.config)} | {(doc.summary_llm_model.config)}"
        f" | {doc.docs_index.__class__}"
    )
