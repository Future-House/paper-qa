from __future__ import annotations

import logging
import os
from typing import Any, cast
from unittest.mock import patch

from langchain.agents import AgentExecutor, BaseSingleActionAgent, ZeroShotAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager, Callbacks
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

from ..docs import Docs
from ..types import Answer
from ..utils import pqa_directory
from .helpers import openai_get_search_query, table_formatter, update_doc_models
from .models import (
    AgentCallback,
    AgentStatus,
    AnswerResponse,
    QueryRequest,
    SimpleProfiler,
)
from .search import SearchDocumentStorage, SearchIndex
from .tools import (
    EmptyDocsError,
    GatherEvidenceTool,
    GenerateAnswerTool,
    PaperSearchTool,
    SharedToolState,
    query_to_tools,
    status,
)

logger = logging.getLogger(__name__)
agent_logger = logging.getLogger(__name__ + ".agent_callers")


async def agent_query(
    query: str | QueryRequest,
    docs: Docs | None = None,
    agent_type: str = "OpenAIFunctionsAgent",
    verbosity: int = 0,
    index_directory: str | os.PathLike | None = None,
) -> AnswerResponse:

    if isinstance(query, str):
        query = QueryRequest(query=query)

    if docs is None:
        docs = Docs()

    if index_directory is None:
        index_directory = pqa_directory("indexes")

    # in-place modification of the docs object to match query
    update_doc_models(
        docs,
        query,
    )

    search_index = SearchIndex(
        fields=SearchIndex.REQUIRED_FIELDS | {"question"},
        index_name="answers",
        index_directory=index_directory,
        storage=SearchDocumentStorage.JSON_MODEL_DUMP,
    )

    response = await run_agent(docs, query, agent_type)

    agent_logger.debug(f"agent_response: {response}")
    truncation_chars = 1_000_000 if verbosity > 1 else 1500 * (verbosity + 1)
    agent_logger.info(
        f"[bold blue]Answer: {response.answer.answer[:truncation_chars]}"
        f'{"...(truncated)" if len(response.answer.answer) > truncation_chars else ""}[/bold blue]'
    )

    await search_index.add_document(
        {
            "file_location": str(response.answer.id),
            "body": response.answer.answer or "",
            "question": response.answer.question,
        },
        document=response,
    )

    await search_index.save_index()

    return response


async def run_agent(
    docs: Docs,
    query: QueryRequest,
    agent_type: str = "OpenAIFunctionsAgent",
) -> AnswerResponse:
    """
    Run an agent.

    Args:
        docs: Docs to run upon.
        query: Query to answer.
        websocket: Websocket to send JSON data and receive text.
        agent_type: Agent type to pass to AgentType.get_agent, or "fake" to TODOC.

    Returns:
        Tuple of resultant answer, token counts, and agent status.
    """
    profiler = SimpleProfiler()
    outer_profile_name = f"agent-{agent_type}-{query.agent_llm}"
    profiler.start(outer_profile_name)

    logger.info(
        f"Beginning agent {agent_type!r} run with question {query.query!r} and full"
        f" query {query.model_dump()}."
    )

    if agent_type == "fake":
        answer, agent_status = await run_fake_agent(query, docs)
    else:
        answer, agent_status = await run_langchain_agent(
            query, docs, agent_type, profiler
        )

    if "cannot answer" in answer.answer.lower() and agent_status != AgentStatus.TIMEOUT:
        agent_status = AgentStatus.UNSURE
    # stop after, so overall isn't reported as long-running step.
    logger.info(
        f"Finished agent {agent_type!r} run with question {query.query!r} and status"
        f" {agent_status}."
    )
    return AnswerResponse(
        answer=answer,
        usage=answer.token_counts,
        status=agent_status,
    )


async def run_fake_agent(
    query: QueryRequest,
    docs: Docs,
) -> tuple[Answer, AgentStatus]:
    answer = Answer(question=query.query, dockey_filter=set(), id=query.id)
    tools = query_to_tools(query, state=SharedToolState(docs=docs, answer=answer))
    search_tool = cast(
        PaperSearchTool,
        next(
            filter(
                lambda x: x.name == PaperSearchTool.__fields__["name"].default, tools
            )
        ),
    )
    gather_evidence_tool = cast(
        GatherEvidenceTool,
        next(
            filter(
                lambda x: x.name == GatherEvidenceTool.__fields__["name"].default, tools
            )
        ),
    )

    generate_answer_tool = cast(
        GenerateAnswerTool,
        next(
            filter(
                lambda x: x.name == GenerateAnswerTool.__fields__["name"].default, tools
            )
        ),
    )
    # seed docs with keyword search
    for search in await openai_get_search_query(
        answer.question, llm=query.llm, count=3
    ):
        await search_tool.arun(search)

    await gather_evidence_tool.arun(tool_input=answer.question)

    await generate_answer_tool.arun(tool_input=answer.question)

    return answer, AgentStatus.SUCCESS


LANGCHAIN_AGENT_TYPES: dict[str, type[BaseSingleActionAgent]] = {
    "ReactAgent": ZeroShotAgent,
    "OpenAIFunctionsAgent": OpenAIFunctionsAgent,
}


async def run_langchain_agent(
    query: QueryRequest,
    docs: Docs,
    agent_type: str,
    profiler: SimpleProfiler,
    timeout: float | None = None,  # noqa: ASYNC109
) -> tuple[Answer, AgentStatus]:
    answer = Answer(question=query.query, dockey_filter=set(), id=query.id)
    shared_callbacks: list[BaseCallbackHandler] = [
        AgentCallback(
            profiler, name=f"step-{agent_type}-{query.agent_llm}", answer_id=answer.id
        ),
    ]
    tools = query_to_tools(
        query,
        state=SharedToolState(docs=docs, answer=answer),
        callbacks=shared_callbacks,
    )
    try:
        search_tool = next(
            filter(
                lambda x: x.name == PaperSearchTool.__fields__["name"].default, tools
            )
        )
    except StopIteration:
        search_tool = None
    answer_tool = cast(
        GenerateAnswerTool,
        next(
            filter(
                lambda x: x.name == GenerateAnswerTool.__fields__["name"].default, tools
            )
        ),
    )

    # optionally use the search tool before the agent
    if search_tool is not None and query.agent_tools.should_pre_search:
        logger.debug("Running search tool before agent choice.")
        await search_tool.arun(answer.question)
    else:
        logger.debug("Skipping search tool before agent choice.")

    llm = ChatOpenAI(
        model=query.agent_llm,
        request_timeout=timeout or query.agent_tools.timeout / 2.0,
        temperature=query.temperature,
    )
    agent_status = AgentStatus.SUCCESS
    cost_callback = OpenAICallbackHandler()
    agent_instance = LANGCHAIN_AGENT_TYPES[agent_type].from_llm_and_tools(
        llm,
        tools,
        system_message=(
            SystemMessage(content=query.agent_tools.agent_system_prompt)
            if query.agent_tools.agent_system_prompt
            else None
        ),
    )
    orig_aplan = agent_instance.aplan
    agent_exec_instance = AgentExecutor.from_agent_and_tools(
        tools=tools,
        agent=agent_instance,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_execution_time=query.agent_tools.timeout,
        callbacks=[*shared_callbacks, cost_callback],
        **(query.agent_tools.agent_config or {}),
    )

    async def aplan_with_injected_callbacks(
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs,
    ) -> AgentAction | AgentFinish:
        # Work around https://github.com/langchain-ai/langchain/issues/22703
        for callback in cast(list[BaseCallbackHandler], agent_exec_instance.callbacks):
            cast(BaseCallbackManager, callbacks).add_handler(callback, inherit=False)
        return await orig_aplan(intermediate_steps, callbacks, **kwargs)

    try:
        # Patch at instance (not class) level to avoid concurrency issues, and we have
        # to patch the dict to work around Pydantic's BaseModel.__setattr__'s validations
        with patch.dict(
            agent_instance.__dict__, {"aplan": aplan_with_injected_callbacks}
        ):
            call_response = await agent_exec_instance.ainvoke(
                input={
                    # NOTE: str.format still works even if the prompt doesn't have
                    # template fields like 'status' or 'gen_answer_tool_name'
                    "input": query.agent_tools.agent_prompt.format(
                        question=answer.question,
                        status=await status(docs, answer),
                        gen_answer_tool_name=answer_tool.name,
                    )
                }
            )
    except TimeoutError:
        call_response = {"output": "Agent stopped", "intermediate_steps": []}
    except EmptyDocsError:
        call_response = {
            "output": "Agent failed due to failed search",
            "intermediate_steps": [],
        }
        agent_status = AgentStatus.FAIL

    async with profiler.timer("agent-accounting"):
        # TODO: move agent trace to LangChain callback
        if "Agent stopped" in call_response["output"]:
            # Log that this agent has gone over timeout, and then answer directly
            logger.warning(
                f"Agent timeout after {query.agent_tools.timeout}-sec, just answering."
            )
            await answer_tool.arun(answer.question)
            agent_status = AgentStatus.TIMEOUT

    return answer, agent_status


async def search(
    query: str,
    index_name: str = "answers",
    index_directory: str | os.PathLike | None = None,
) -> list[tuple[AnswerResponse, str] | tuple[Any, str]]:

    search_index = SearchIndex(
        ["file_location", "body", "question"],
        index_name=index_name,
        index_directory=index_directory or pqa_directory("indexes"),
        storage=SearchDocumentStorage.JSON_MODEL_DUMP,
    )

    results = [
        (AnswerResponse(**a[0]) if index_name == "answers" else a[0], a[1])
        for a in await search_index.query(query=query, keep_filenames=True)
    ]

    if results:
        console = Console(record=True)
        # Render the table to a string
        console.print(table_formatter(results))
    else:
        agent_logger.info("No results found.")

    return results
