import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aviary.core import (
    MalformedMessageError,
    Message,
    Tool,
    ToolCall,
    ToolRequestMessage,
    ToolSelector,
    ToolSelectorLedger,
)
from pydantic import BaseModel
from rich.console import Console
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
)

try:
    from ldp.alg import Callback, RolloutManager
except ImportError:

    class Callback:  # type: ignore[no-redef]
        """Placeholder parent class for when ldp isn't installed."""

    RolloutManager = None  # type: ignore[assignment,misc]

from paperqa.docs import Docs
from paperqa.settings import AgentSettings
from paperqa.types import PQASession

from .env import PaperQAEnvironment
from .helpers import litellm_get_search_query, table_formatter
from .models import AgentStatus, AnswerResponse, QueryRequest, SimpleProfiler
from .search import SearchDocumentStorage, SearchIndex, get_directory_index
from .tools import EnvironmentState, GatherEvidence, GenerateAnswer, PaperSearch

if TYPE_CHECKING:
    from aviary.core import Environment
    from ldp.agent import Agent, SimpleAgentState
    from ldp.graph.ops import OpResult

logger = logging.getLogger(__name__)
agent_logger = logging.getLogger(__name__ + ".agent_callers")

DEFAULT_AGENT_TYPE = AgentSettings.model_fields["agent_type"].default


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

    answers_index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "question"],
        index_name="answers",
        index_directory=query.settings.agent.index.index_directory,
        storage=SearchDocumentStorage.JSON_MODEL_DUMP,
    )

    response = await run_agent(docs, query, agent_type, **runner_kwargs)
    agent_logger.debug(f"agent_response: {response}")

    agent_logger.info(f"[bold blue]Answer: {response.session.answer}[/bold blue]")

    await answers_index.add_document(
        {
            "file_location": str(response.session.id),
            "body": response.session.answer,
            "question": response.session.question,
        },
        document=response,
    )
    await answers_index.save_index()
    return response


FAKE_AGENT_TYPE = "fake"  # No agent, just invoke tools in deterministic order


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

    # Build the index once here, and then all tools won't need to rebuild it
    await get_directory_index(settings=query.settings)
    if isinstance(agent_type, str) and agent_type.lower() == FAKE_AGENT_TYPE:
        session, agent_status = await run_fake_agent(query, docs, **runner_kwargs)
    elif tool_selector_or_none := query.settings.make_aviary_tool_selector(agent_type):
        session, agent_status = await run_aviary_agent(
            query, docs, tool_selector_or_none, **runner_kwargs
        )
    elif ldp_agent_or_none := await query.settings.make_ldp_agent(agent_type):
        session, agent_status = await run_ldp_agent(
            query, docs, ldp_agent_or_none, **runner_kwargs
        )
    else:
        raise NotImplementedError(f"Didn't yet handle agent type {agent_type}.")

    if session.could_not_answer and agent_status != AgentStatus.TRUNCATED:
        agent_status = AgentStatus.UNSURE
    # stop after, so overall isn't reported as long-running step.
    logger.info(
        f"Finished agent {agent_type!r} run with question {query.query!r} and status"
        f" {agent_status}."
    )
    return AnswerResponse(session=session, status=agent_status)


async def run_fake_agent(
    query: QueryRequest,
    docs: Docs,
    env_class: type[PaperQAEnvironment] = PaperQAEnvironment,
    on_env_reset_callback: Callable[[EnvironmentState], Awaitable] | None = None,
    on_agent_action_callback: (
        Callable[[ToolRequestMessage, BaseModel], Awaitable] | None
    ) = None,
    on_env_step_callback: (
        Callable[[list[Message], float, bool, bool], Awaitable] | None
    ) = None,
    **env_kwargs,
) -> tuple[PQASession, AgentStatus]:
    if query.settings.agent.max_timesteps is not None:
        logger.warning(
            f"Max timesteps (configured {query.settings.agent.max_timesteps}) is not"
            " applicable with the fake agent, ignoring it."
        )
    env = env_class(query, docs, **env_kwargs)
    _, tools = await env.reset()
    if on_env_reset_callback:
        await on_env_reset_callback(env.state)

    question = env.state.session.question
    search_tool = next(filter(lambda x: x.info.name == PaperSearch.TOOL_FN_NAME, tools))
    gather_evidence_tool = next(
        filter(lambda x: x.info.name == GatherEvidence.TOOL_FN_NAME, tools)
    )
    generate_answer_tool = next(
        filter(lambda x: x.info.name == GenerateAnswer.TOOL_FN_NAME, tools)
    )

    async def step(tool: Tool, **call_kwargs) -> None:
        action = ToolRequestMessage(
            tool_calls=[ToolCall.from_tool(tool, **call_kwargs)]
        )
        if on_agent_action_callback:
            await on_agent_action_callback(action, env.state)
        obs, reward, done, truncated = await env.step(action)
        if on_env_step_callback:
            await on_env_step_callback(obs, reward, done, truncated)

    # Seed docs with a few keyword searches
    for search in await litellm_get_search_query(
        question, llm=query.settings.get_llm(), count=3
    ):
        await step(search_tool, query=search, min_year=None, max_year=None)
    await step(gather_evidence_tool, question=question)
    await step(generate_answer_tool, question=question)
    return env.state.session, AgentStatus.SUCCESS


async def run_aviary_agent(
    query: QueryRequest,
    docs: Docs,
    agent: ToolSelector,
    env_class: type[PaperQAEnvironment] = PaperQAEnvironment,
    on_env_reset_callback: Callable[[EnvironmentState], Awaitable] | None = None,
    on_agent_action_callback: (
        Callable[[ToolRequestMessage, BaseModel], Awaitable] | None
    ) = None,
    on_env_step_callback: (
        Callable[[list[Message], float, bool, bool], Awaitable] | None
    ) = None,
    **env_kwargs,
) -> tuple[PQASession, AgentStatus]:
    env = env_class(query, docs, **env_kwargs)
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

            timestep, max_timesteps = 0, query.settings.agent.max_timesteps
            while not done:
                if max_timesteps is not None and timestep >= max_timesteps:
                    logger.warning(
                        f"Agent didn't finish within {max_timesteps} timesteps, just"
                        " answering."
                    )
                    generate_answer_tool = next(
                        filter(
                            lambda x: x.info.name == GenerateAnswer.TOOL_FN_NAME,
                            env.tools,
                        )
                    )
                    await generate_answer_tool._tool_fn(
                        question=query.query, state=env.state
                    )
                    return env.state.session, AgentStatus.TRUNCATED
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
                timestep += 1
            status = AgentStatus.SUCCESS
    except TimeoutError:
        logger.warning(
            f"Agent timeout after {query.settings.agent.timeout}-sec, just answering."
        )
        status = AgentStatus.TRUNCATED
        generate_answer_tool = next(
            filter(lambda x: x.info.name == GenerateAnswer.TOOL_FN_NAME, env.tools)
        )
        await generate_answer_tool._tool_fn(question=query.query, state=env.state)
    except Exception:
        logger.exception(f"Agent {agent} failed.")
        status = AgentStatus.FAIL
    return env.state.session, status


class LDPRolloutCallback(Callback):
    """Shim connecting ldp RolloutManager Callbacks with paperqa runner callbacks."""

    def __init__(
        self,
        env: "Environment",
        on_env_reset_callback: Callable[[EnvironmentState], Awaitable] | None = None,
        on_agent_action_callback: "Callable[[OpResult[ToolRequestMessage], SimpleAgentState, float], Awaitable] | None" = None,  # noqa: E501
        on_env_step_callback: (
            Callable[[list[Message], float, bool, bool], Awaitable] | None
        ) = None,
    ):
        self.env = env
        self.on_env_reset_callback = on_env_reset_callback
        self.on_agent_action_callback = on_agent_action_callback
        self.on_env_step_callback = on_env_step_callback

    async def after_agent_get_asv(self, traj_id: str, *args) -> None:  # noqa: ARG002
        if self.on_agent_action_callback is not None:
            await self.on_agent_action_callback(*args)

    async def after_env_reset(self, traj_id: str, *_) -> None:  # noqa: ARG002
        if self.on_env_reset_callback is not None:
            await self.on_env_reset_callback(self.env.state)

    async def after_env_step(self, traj_id: str, *args) -> None:  # noqa: ARG002
        if self.on_env_step_callback is not None:
            await self.on_env_step_callback(*args)


async def run_ldp_agent(
    query: QueryRequest,
    docs: Docs,
    agent: "Agent[SimpleAgentState]",
    env_class: type[PaperQAEnvironment] = PaperQAEnvironment,
    on_env_reset_callback: Callable[[EnvironmentState], Awaitable] | None = None,
    on_agent_action_callback: "Callable[[OpResult[ToolRequestMessage], SimpleAgentState, float], Awaitable] | None" = None,  # noqa: E501
    on_env_step_callback: (
        Callable[[list[Message], float, bool, bool], Awaitable] | None
    ) = None,
    ldp_callback_type: type[LDPRolloutCallback] = LDPRolloutCallback,
    **env_kwargs,
) -> tuple[PQASession, AgentStatus]:
    env = env_class(query, docs, **env_kwargs)
    # NOTE: don't worry about ldp import checks, because we know Settings.make_ldp_agent
    # has already taken place, which checks that ldp is installed

    try:
        async with asyncio.timeout(query.settings.agent.timeout):
            rollout_manager = RolloutManager(
                agent,
                callbacks=[
                    ldp_callback_type(
                        env,
                        on_env_reset_callback,
                        on_agent_action_callback,
                        on_env_step_callback,
                    )
                ],
            )
            await rollout_manager.sample_trajectories(
                environments=[env], max_steps=query.settings.agent.max_timesteps
            )
            status = AgentStatus.SUCCESS
    except TimeoutError:
        logger.warning(
            f"Agent timeout after {query.settings.agent.timeout}-sec, just answering."
        )
        status = AgentStatus.TRUNCATED
        generate_answer_tool = next(
            filter(lambda x: x.info.name == GenerateAnswer.TOOL_FN_NAME, env.tools)
        )
        await generate_answer_tool._tool_fn(question=query.query, state=env.state)
    except Exception:
        logger.exception(f"Agent {agent} failed.")
        status = AgentStatus.FAIL
    return env.state.session, status


async def index_search(
    query: str, index_name: str = "answers", **index_kwargs
) -> list[tuple[AnswerResponse, str] | tuple[Any, str]]:
    fields = [*SearchIndex.REQUIRED_FIELDS]
    if index_name == "answers":
        fields.append("question")
    index_to_query = SearchIndex(
        fields=fields,
        index_name=index_name,
        storage=(
            SearchDocumentStorage.JSON_MODEL_DUMP
            if index_name == "answers"
            else SearchDocumentStorage.PICKLE_COMPRESSED
        ),
        **index_kwargs,
    )

    results = [
        (AnswerResponse(**a[0]) if index_name == "answers" else a[0], a[1])
        for a in await index_to_query.query(query=query, keep_filenames=True)
    ]
    if results:
        console = Console(record=True)
        # Render the table to a string
        console.print(table_formatter(results))
    else:
        count = await index_to_query.count
        agent_logger.info(f"No results found. Searched {count} docs.")

    return results
