from __future__ import annotations

import logging
import os
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from .. import (
    Answer,
    AnthropicLLMModel,
    Docs,
    OpenAILLMModel,
    embedding_model_factory,
    llm_model_factory,
)
from ..llms import LangchainLLMModel
from .models import (
    QueryRequest,
)

logger = logging.getLogger(__name__)


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


async def stream_filter(
    docs: Docs,
    query: str,
    answer: Answer,
    num_doc_match_retries: int = 1,
    adoc_match_threshold: int = QueryRequest.model_fields[
        "adoc_match_threshold"
    ].default,
) -> Answer:
    """Set the `answer.dockey_filter`."""
    if len(docs.docs) >= adoc_match_threshold:

        for i in range(num_doc_match_retries + 1):
            try:
                result = await docs.adoc_match(
                    query,
                    rerank=True,  # want to set it explicitly
                    answer=answer,
                )
                break
            except Exception as e:
                if i < num_doc_match_retries:
                    logger.warning(
                        f"Failed to filter, possibly due to huge paper citation, full message: {e}."
                    )
                    continue
                raise
    else:
        result = set(docs.docs.keys())
    answer.dockey_filter = result
    return answer


async def stream_evidence(
    docs: Docs,
    request: QueryRequest,
    answer: Answer | None = None,
) -> Answer:
    if answer is None:
        answer = await stream_filter(
            docs=docs,
            query=request.query,
            answer=Answer(question=request.query, id=request.id),
            adoc_match_threshold=request.adoc_match_threshold,
        )

    # clear texts
    docs.jit_texts_index = True
    # ensure length is set correctly
    answer.summary_length = request.summary_length
    answer = await docs.aget_evidence(
        answer=answer,
        max_sources=request.max_sources,
        k=request.consider_sources,
        detailed_citations=True,
    )
    # we uniquify the contexts
    names = set()
    uniq_docs = []
    for c in answer.contexts:
        if c.text.doc.docname not in names:
            uniq_docs.append(c.text.doc)
            names.add(c.text.doc.docname)
    return answer


async def stream_cost(
    answer: Answer,
    extra_values: dict[str, list[int]] | None = None,
) -> Answer:
    """
    Calculate and set Answer cost and send to the input websocket.

    Args:
        answer: Answer to get token counts from and
        websocket: Websocket to send token counts and cost.
        extra_values: Dictionary mapping of model name to two-item list of LLM prompt
            token counts and LLM completion token counts.

    Returns:
        Answer passed in.
    """
    token_counts = answer.token_counts
    if extra_values is not None:
        # merge, but add to existing if already there
        for k, v in extra_values.items():
            if k in token_counts:
                token_counts[k][0] += v[0]
                token_counts[k][1] += v[1]
            else:
                token_counts[k] = v
    answer.cost = compute_total_model_token_cost(token_counts)
    return answer


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
    doc.llm_model.config["temperature"] = request.temperature  # type: ignore[attr-defined]
    doc.summary_llm_model.config["temperature"] = request.temperature  # type: ignore[attr-defined]

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
        f" | {(doc.llm_model.config)} | {(doc.summary_llm_model.config)}"  # type: ignore[attr-defined]
        f" | {doc.docs_index.__class__}"
    )
