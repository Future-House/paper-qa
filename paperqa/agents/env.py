import logging
from copy import deepcopy
from typing import Any, ClassVar, Self, cast

from aviary.core import (
    Environment,
    Frame,
    Message,
    Messages,
    Tool,
    ToolRequestMessage,
    ToolResponseMessage,
)

from paperqa.docs import Docs
from paperqa.llms import EmbeddingModel, LiteLLMModel
from paperqa.settings import Settings
from paperqa.types import PQASession
from paperqa.utils import get_year

from .models import QueryRequest
from .tools import (
    AVAILABLE_TOOL_NAME_TO_CLASS,
    EnvironmentState,
    GatherEvidence,
    GenerateAnswer,
    PaperSearch,
)

logger = logging.getLogger(__name__)

POPULATE_FROM_SETTINGS = None


def settings_to_tools(
    settings: Settings,
    llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
    summary_llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
    embedding_model: EmbeddingModel | None = POPULATE_FROM_SETTINGS,
) -> list[Tool]:
    """
    Convert a Settings into tools, confirming the gen_answer tool is present.

    NOTE: the last element of the return will always be GenerateAnswer.
    """
    llm_model = llm_model or settings.get_llm()
    summary_llm_model = summary_llm_model or settings.get_summary_llm()
    embedding_model = embedding_model or settings.get_embedding_model()
    tools: list[Tool] = []
    has_answer_tool = False
    for tool_type in (
        (PaperSearch, GatherEvidence, GenerateAnswer)
        if settings.agent.tool_names is None
        else [
            AVAILABLE_TOOL_NAME_TO_CLASS[name]
            for name in set(settings.agent.tool_names)
        ]
    ):
        if issubclass(tool_type, PaperSearch):
            tool = Tool.from_function(
                PaperSearch(
                    settings=settings, embedding_model=embedding_model
                ).paper_search
            )
            for pname in ("min_year", "max_year"):
                tool.info.parameters.properties[pname]["description"] = cast(
                    str, tool.info.parameters.properties[pname]["description"]
                ).format(current_year=get_year())
        elif issubclass(tool_type, GatherEvidence):
            tool = Tool.from_function(
                GatherEvidence(
                    settings=settings,
                    summary_llm_model=summary_llm_model,
                    embedding_model=embedding_model,
                ).gather_evidence
            )
        elif issubclass(tool_type, GenerateAnswer):
            tool = Tool.from_function(
                GenerateAnswer(
                    settings=settings,
                    llm_model=llm_model,
                    summary_llm_model=summary_llm_model,
                    embedding_model=embedding_model,
                ).gen_answer
            )
        else:
            raise NotImplementedError(f"Didn't handle tool type {tool_type}.")
        if tool.info.name == GenerateAnswer.gen_answer.__name__:
            tools.append(tool)
            has_answer_tool = True
        else:
            tools.insert(0, tool)
    if not has_answer_tool:
        raise ValueError(
            f"{GenerateAnswer.gen_answer.__name__} must be one of the tools."
        )
    return tools


class PaperQAEnvironment(Environment[EnvironmentState]):
    """Environment connecting paper-qa's tools with state."""

    def __init__(
        self,
        query: QueryRequest,
        docs: Docs,
        llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        summary_llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        embedding_model: EmbeddingModel | None = POPULATE_FROM_SETTINGS,
        **env_kwargs,
    ):
        super().__init__(**env_kwargs)
        # Hold onto QueryRequest to create fresh tools and answer during each reset
        self._query = query
        # Hold onto Docs to clear and reuse in state during each reset
        self._docs = docs
        self._llm_model = llm_model
        self._summary_llm_model = summary_llm_model
        self._embedding_model = embedding_model

    def make_tools(self) -> list[Tool]:
        return settings_to_tools(
            settings=self._query.settings,
            llm_model=self._llm_model,
            summary_llm_model=self._summary_llm_model,
            embedding_model=self._embedding_model,
        )

    def make_initial_state(self) -> EnvironmentState:
        return EnvironmentState(
            docs=self._docs,
            answer=PQASession(
                question=self._query.query,
                config_md5=self._query.settings.md5,
                id=self._query.id,
            ),
        )

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        # NOTE: don't build the index here, as sometimes we asyncio.gather over this
        # method, and our current design (as of v5.0.10) could hit race conditions
        # because index building does not use file locks
        self._docs.clear_docs()
        self.state, self.tools = self.make_initial_state(), self.make_tools()
        return (
            [
                Message(
                    content=self._query.settings.agent.agent_prompt.format(
                        question=self.state.session.question,
                        status=self.state.status,
                        gen_answer_tool_name=GenerateAnswer.TOOL_FN_NAME,
                    ),
                )
            ],
            self.tools,
        )

    def export_frame(self) -> Frame:
        return Frame(state=self.state, info={"query": self._query})

    def _has_excess_answer_failures(self) -> bool:
        if self._query.settings.answer.max_answer_attempts is None:
            return False
        return (
            sum(
                tn == GenerateAnswer.gen_answer.__name__
                for s in self.state.tool_history
                for tn in s
            )
            > self._query.settings.answer.max_answer_attempts
        )

    USE_POST_PROCESSED_REWARD: ClassVar[float] = 0.0

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        self.state.record_action(action)
        if not action.tool_calls:
            return (
                # NOTE: don't put:
                # - GenerateAnswer.FAILED_TO_ANSWER here because this wasn't a failure
                # - 'cannot answer' because that information belongs in
                #   PQASession.answer, not in the message history
                # Let's just put a nice message about being done :)
                [Message(content="Agent specified 0 tool calls, which means done.")],
                self.USE_POST_PROCESSED_REWARD,
                True,  # Matching LangChain: https://github.com/langchain-ai/langchain/blob/langchain%3D%3D0.2.17/libs/langchain/langchain/agents/output_parsers/openai_functions.py#L38-L77
                False,  # Let caller determine truncations
            )

        response_messages = cast(
            list[Message],
            await self.exec_tool_calls(action, state=self.state, handle_tool_exc=True),
        )
        return (
            response_messages,
            self.USE_POST_PROCESSED_REWARD,
            any(
                isinstance(msg, ToolResponseMessage)
                and msg.name == GenerateAnswer.gen_answer.__name__
                and GenerateAnswer.did_not_fail_to_answer(msg.content)
                for msg in response_messages
            )
            or self._has_excess_answer_failures(),
            False,  # Let caller determine truncations
        )

    def __deepcopy__(self, memo) -> Self:
        copy_state = deepcopy(self.state, memo)
        # We don't know the side effects of deep copying a litellm.Router,
        # so we force a shallow copy of these LiteLLMModels
        env_model_kwargs: dict[str, Any] = {
            name: model if model is None else type(model)(**model.model_dump())
            for name, model in (
                ("llm_model", self._llm_model),
                ("summary_llm_model", self._summary_llm_model),
                ("embedding_model", self._embedding_model),
            )
        }
        copy_self = type(self)(
            query=deepcopy(self._query, memo),  # deepcopy for _docs_name
            docs=copy_state.docs,
            **env_model_kwargs,
        )
        copy_self.state = copy_state
        # Because we shallow copied the LiteLLMModels, we need to re-make the
        # tool functions within the tools
        copy_self.tools = copy_self.make_tools()
        return copy_self
