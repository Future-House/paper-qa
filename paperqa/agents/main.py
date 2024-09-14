import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pydoc import locate
from typing import TYPE_CHECKING, Any, cast

from aviary.message import MalformedMessageError, Message
from aviary.tools import (
    Tool,
    ToolCall,
    ToolRequestMessage,
    ToolSelector,
    ToolSelectorLedger,
)
from pydantic import BaseModel, TypeAdapter
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
)

try:
    from ldp.agent import (
        Agent,
        HTTPAgentClient,
        MemoryAgent,
        ReActAgent,
        SimpleAgent,
        SimpleAgentState,
    )
    from ldp.graph.memory import Memory, UIndexMemoryModel
    from ldp.graph.op_utils import set_training_mode
    from ldp.llms import EmbeddingModel

    _Memories = TypeAdapter(dict[int, Memory] | list[Memory])  # type: ignore[var-annotated]

    HAS_LDP_INSTALLED = True
except ImportError:
    HAS_LDP_INSTALLED = False
from rich.console import Console

from paperqa.docs import Docs
from paperqa.settings import Settings
from paperqa.types import Answer
from paperqa.utils import pqa_directory

from .env import PaperQAEnvironment
from .helpers import litellm_get_search_query, table_formatter
from .models import AgentStatus, AnswerResponse, QueryRequest, SimpleProfiler
from .search import SearchDocumentStorage, SearchIndex
from .tools import EnvironmentState, GatherEvidence, GenerateAnswer, PaperSearch

if TYPE_CHECKING:
    from ldp.graph.ops import OpResult

logger = logging.getLogger(__name__)
agent_logger = logging.getLogger(__name__ + ".agent_callers")

DEFAULT_AGENT_TYPE = ToolSelector


async def agent_query(
    query: str | QueryRequest,
    docs: Docs | None = None,
    agent_type: str | type = DEFAULT_AGENT_TYPE,
    **runner_kwargs,
) -> AnswerResponse:
    if isinstance(query, str):
        query = QueryRequest(query=query)
    if docs is None:
        docs = Docs()

    search_index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "question"],
        index_name="answers",
        index_directory=query.settings.index_directory,
        storage=SearchDocumentStorage.JSON_MODEL_DUMP,
    )

    response = await run_agent(docs, query, agent_type, **runner_kwargs)
    agent_logger.debug(f"agent_response: {response}")

    agent_logger.info(f"[bold blue]Answer: {response.answer.answer}[/bold blue]")

    await search_index.add_document(
        {
            "file_location": str(response.answer.id),
            "body": response.answer.answer,
            "question": response.answer.question,
        },
        document=response,
    )
    await search_index.save_index()
    return response


def to_aviary_tool_selector(
    agent_type: str | type, settings: Settings
) -> ToolSelector | None:
    """Attempt to convert the agent type to an aviary ToolSelector."""
    if agent_type is ToolSelector or (
        isinstance(agent_type, str)
        and (
            agent_type == ToolSelector.__name__
            or (
                agent_type.startswith(ToolSelector.__module__.split(".", maxsplit=1)[0])
                and locate(agent_type) is ToolSelector
            )
        )
    ):
        return ToolSelector(
            model_name=settings.agent.agent_llm,
            acompletion=settings.get_agent_llm().router.acompletion,
            **(settings.agent.agent_config or {}),
        )
    return None


async def to_ldp_agent(
    agent_type: str | type, settings: Settings
) -> "Agent[SimpleAgentState] | None":
    """Attempt to convert the agent type to an ldp Agent."""
    if not isinstance(agent_type, str):  # Convert to fully qualified name
        agent_type = f"{agent_type.__module__}.{agent_type.__name__}"
    if not agent_type.startswith("ldp"):
        return None
    if not HAS_LDP_INSTALLED:
        raise ImportError(
            "ldp agents requires the 'ldp' extra for 'ldp'. Please:"
            " `pip install paper-qa[ldp]`."
        )

    # TODO: support general agents
    agent_cls = cast(type[Agent], locate(agent_type))
    agent_settings = settings.agent
    agent_llm, config = agent_settings.agent_llm, agent_settings.agent_config or {}
    if issubclass(agent_cls, ReActAgent | MemoryAgent):
        if (
            issubclass(agent_cls, MemoryAgent)
            and "memory_model" in config
            and "memories" in config
        ):
            if "embedding_model" in config["memory_model"]:
                # Work around EmbeddingModel not yet supporting deserialization
                config["memory_model"]["embedding_model"] = EmbeddingModel.from_name(
                    embedding=config["memory_model"].pop("embedding_model")["name"]
                )
            config["memory_model"] = UIndexMemoryModel(**config["memory_model"])
            memories = _Memories.validate_python(config.pop("memories"))
            await asyncio.gather(
                *(
                    config["memory_model"].add_memory(memory)
                    for memory in (
                        memories.values() if isinstance(memories, dict) else memories
                    )
                )
            )
        return agent_cls(
            llm_model={"model": agent_llm, "temperature": settings.temperature},
            **config,
        )
    if issubclass(agent_cls, SimpleAgent):
        return agent_cls(
            llm_model={"model": agent_llm, "temperature": settings.temperature},
            sys_prompt=agent_settings.agent_system_prompt,
            **config,
        )
    if issubclass(agent_cls, HTTPAgentClient):
        set_training_mode(False)
        return HTTPAgentClient[SimpleAgentState](
            agent_state_type=SimpleAgentState, **config
        )
    raise NotImplementedError(f"Didn't yet handle agent type {agent_type}.")


async def run_agent(
    docs: Docs,
    query: QueryRequest,
    agent_type: str | type = DEFAULT_AGENT_TYPE,
    **runner_kwargs,
) -> AnswerResponse:
    """
    Run an agent.

    Args:
        docs: Docs to run upon.
        query: Query to answer.
        agent_type: Agent type (or fully qualified name to the type) to pass to
            AgentType.get_agent, or "fake" to TODOC.
        runner_kwargs: Keyword arguments to pass to the runner.

    Returns:
        Tuple of resultant answer, token counts, and agent status.
    """
    profiler = SimpleProfiler()
    outer_profile_name = f"agent-{agent_type}-{query.settings.agent.agent_llm}"
    profiler.start(outer_profile_name)

    logger.info(
        f"Beginning agent {agent_type!r} run with question {query.query!r} and full"
        f" query {query.model_dump()}."
    )

    if agent_type == "fake":
        answer, agent_status = await run_fake_agent(query, docs, **runner_kwargs)
    elif tool_selector_or_none := to_aviary_tool_selector(agent_type, query.settings):
        answer, agent_status = await run_aviary_agent(
            query, docs, tool_selector_or_none, **runner_kwargs
        )
    elif ldp_agent_or_none := await to_ldp_agent(agent_type, query.settings):
        answer, agent_status = await run_ldp_agent(
            query, docs, ldp_agent_or_none, **runner_kwargs
        )
    else:
        raise NotImplementedError(f"Didn't yet handle agent type {agent_type}.")

    if "cannot answer" in answer.answer.lower() and agent_status != AgentStatus.TIMEOUT:
        agent_status = AgentStatus.UNSURE
    # stop after, so overall isn't reported as long-running step.
    logger.info(
        f"Finished agent {agent_type!r} run with question {query.query!r} and status"
        f" {agent_status}."
    )
    return AnswerResponse(answer=answer, status=agent_status)


async def run_fake_agent(
    query: QueryRequest,
    docs: Docs,
    on_env_reset_callback: Callable[[EnvironmentState], Awaitable] | None = None,
    on_env_step_callback: (
        Callable[[list[Message], float, bool, bool], Awaitable] | None
    ) = None,
    **env_kwargs,
) -> tuple[Answer, AgentStatus]:
    env = PaperQAEnvironment(query, docs, **env_kwargs)
    _, tools = await env.reset()
    if on_env_reset_callback:
        await on_env_reset_callback(env.state)

    question = env.state.answer.question
    search_tool = next(filter(lambda x: x.info.name == PaperSearch.TOOL_FN_NAME, tools))
    gather_evidence_tool = next(
        filter(lambda x: x.info.name == GatherEvidence.TOOL_FN_NAME, tools)
    )
    generate_answer_tool = next(
        filter(lambda x: x.info.name == GenerateAnswer.TOOL_FN_NAME, tools)
    )

    async def step(tool: Tool, **call_kwargs) -> None:
        obs, reward, done, truncated = await env.step(
            action=ToolRequestMessage(
                tool_calls=[ToolCall.from_tool(tool, **call_kwargs)]
            )
        )
        if on_env_step_callback:
            await on_env_step_callback(obs, reward, done, truncated)

    # Seed docs with a few keyword searches
    for search in await litellm_get_search_query(
        question, llm=query.settings.llm, count=3
    ):
        await step(search_tool, query=search, min_year=None, max_year=None)
    await step(gather_evidence_tool, question=question)
    await step(generate_answer_tool, question=question)
    return env.state.answer, AgentStatus.SUCCESS


async def run_aviary_agent(
    query: QueryRequest,
    docs: Docs,
    agent: ToolSelector,
    on_env_reset_callback: Callable[[EnvironmentState], Awaitable] | None = None,
    on_agent_action_callback: (
        Callable[[ToolRequestMessage, BaseModel], Awaitable] | None
    ) = None,
    on_env_step_callback: (
        Callable[[list[Message], float, bool, bool], Awaitable] | None
    ) = None,
    **env_kwargs,
) -> tuple[Answer, AgentStatus]:
    env = PaperQAEnvironment(query, docs, **env_kwargs)
    done = False

    try:
        async with asyncio.timeout(query.settings.agent.timeout):
            obs, tools = await env.reset()
            if on_env_reset_callback:
                await on_env_reset_callback(env.state)

            agent_state = ToolSelectorLedger(
                messages=(
                    [
                        Message(
                            role="system",
                            content=query.settings.agent.agent_system_prompt,
                        )
                    ]
                    if query.settings.agent.agent_system_prompt
                    else []
                ),
                tools=tools,
            )

            while not done:
                agent_state.messages += obs
                for attempt in Retrying(
                    stop=stop_after_attempt(5),
                    retry=retry_if_exception_type(MalformedMessageError),
                    before_sleep=before_sleep_log(logger, logging.WARNING),
                    reraise=True,
                ):
                    with attempt:  # Retrying if ToolSelector fails to select a tool
                        action = await agent(agent_state.messages, tools)
                agent_state.messages = [*agent_state.messages, action]
                if on_agent_action_callback:
                    await on_agent_action_callback(action, agent_state)

                obs, reward, done, truncated = await env.step(action)
                if on_env_step_callback:
                    await on_env_step_callback(obs, reward, done, truncated)
            status = AgentStatus.SUCCESS
    except TimeoutError:
        logger.warning(
            f"Agent timeout after {query.settings.agent.timeout}-sec, just answering."
        )
        status = AgentStatus.TIMEOUT
        await tools[-1]._tool_fn(question=query.query, state=env.state)
    except Exception:
        logger.exception(f"Agent {agent} failed.")
        status = AgentStatus.FAIL
    return env.state.answer, status


async def run_ldp_agent(
    query: QueryRequest,
    docs: Docs,
    agent: "Agent[SimpleAgentState]",
    on_env_reset_callback: Callable[[EnvironmentState], Awaitable] | None = None,
    on_agent_action_callback: "Callable[[OpResult[ToolRequestMessage], SimpleAgentState, float], Awaitable] | None" = None,  # noqa: E501
    on_env_step_callback: (
        Callable[[list[Message], float, bool, bool], Awaitable] | None
    ) = None,
    **env_kwargs,
) -> tuple[Answer, AgentStatus]:
    env = PaperQAEnvironment(query, docs, **env_kwargs)
    done = False

    try:
        async with asyncio.timeout(query.settings.agent.timeout):
            obs, tools = await env.reset()
            if on_env_reset_callback:
                await on_env_reset_callback(env.state)

            agent_state = await agent.init_state(tools=tools)

            while not done:
                action, agent_state, value = await agent.get_asv(agent_state, obs)
                if on_agent_action_callback:
                    await on_agent_action_callback(action, agent_state, value)

                obs, reward, done, truncated = await env.step(action.value)
                if on_env_step_callback:
                    await on_env_step_callback(obs, reward, done, truncated)
            status = AgentStatus.SUCCESS
    except TimeoutError:
        logger.warning(
            f"Agent timeout after {query.settings.agent.timeout}-sec, just answering."
        )
        status = AgentStatus.TIMEOUT
        await tools[-1]._tool_fn(question=query.query, state=env.state)
    except Exception:
        logger.exception(f"Agent {agent} failed.")
        status = AgentStatus.FAIL
    return env.state.answer, status


async def index_search(
    query: str,
    index_name: str = "answers",
    index_directory: str | os.PathLike | None = None,
) -> list[tuple[AnswerResponse, str] | tuple[Any, str]]:
    fields = [*SearchIndex.REQUIRED_FIELDS]
    if index_name == "answers":
        fields.append("question")
    search_index = SearchIndex(
        fields=fields,
        index_name=index_name,
        index_directory=index_directory or pqa_directory("indexes"),
        storage=(
            SearchDocumentStorage.JSON_MODEL_DUMP
            if index_name == "answers"
            else SearchDocumentStorage.PICKLE_COMPRESSED
        ),
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
        count = await search_index.count
        agent_logger.info(f"No results found. Searched {count} docs")

    return results
