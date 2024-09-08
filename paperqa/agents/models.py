from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Collection, assert_never
from uuid import UUID, uuid4

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage, messages_to_dict
from langchain_core.outputs import ChatGeneration, LLMResult
from openai import AsyncOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
)
from typing_extensions import Protocol

from .. import (
    Answer,
    OpenAILLMModel,
    PromptCollection,
    llm_model_factory,
)
from ..utils import hexdigest
from ..version import __version__
from .prompts import STATIC_PROMPTS

logger = logging.getLogger(__name__)


class SupportsPickle(Protocol):
    """Type protocol for typing any object that supports pickling."""

    def __reduce__(self) -> str | tuple[Any, ...]: ...
    def __getstate__(self) -> object: ...
    def __setstate__(self, state: object) -> None: ...


class AgentStatus(str, Enum):
    # FAIL - no answer could be generated
    FAIL = "fail"
    # SUCCESS - answer was generated
    SUCCESS = "success"
    # TIMEOUT - agent took too long, but an answer was generated
    TIMEOUT = "timeout"
    # UNSURE - the agent was unsure, but an answer is present
    UNSURE = "unsure"


class AgentPromptCollection(BaseModel):
    """Configuration for the agent."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    agent_system_prompt: str | None = Field(
        # Matching https://github.com/langchain-ai/langchain/blob/langchain%3D%3D0.2.3/libs/langchain/langchain/agents/openai_functions_agent/base.py#L213-L215
        default="You are a helpful AI assistant.",
        description="Optional system prompt message to precede the below agent_prompt.",
    )

    # TODO: make this prompt more minimalist, instead improving tool descriptions so
    # how to use them together can be intuited, and exposing them for configuration
    agent_prompt: str = (
        "Answer question: {question}"
        "\n\nSearch for papers, gather evidence, collect papers cited in evidence then re-gather evidence, and answer."
        " Gathering evidence will do nothing if you have not done a new search or collected new papers."
        " If you do not have enough evidence to generate a good answer, you can:"
        "\n- Search for more papers (preferred)"
        "\n- Collect papers cited by previous evidence (preferred)"
        "\n- Gather more evidence using a different phrase"
        "\nIf you search for more papers or collect new papers cited by previous evidence,"
        " remember to gather evidence again."
        " Once you have five or more pieces of evidence from multiple sources, or you have tried a few times, "
        "call {gen_answer_tool_name} tool. The {gen_answer_tool_name} tool output is visible to the user, "
        "so you do not need to restate the answer and can simply terminate if the answer looks sufficient. "
        "The current status of evidence/papers/cost is {status}"
    )
    paper_directory: str | os.PathLike = Field(
        default=Path.cwd(),
        description=(
            "Local directory which contains the papers to be indexed and searched."
        ),
    )
    index_directory: str | os.PathLike | None = Field(
        default=None,
        description=(
            "Directory to store the PQA generated search index, configuration, and answer indexes."
        ),
    )
    manifest_file: str | os.PathLike | None = Field(
        default=None,
        description=(
            "Optional manifest CSV, containing columns which are attributes for a DocDetails object. "
            "Only 'file_location','doi', and 'title' will be used when indexing."
        ),
    )
    search_count: int = 8
    wipe_context_on_answer_failure: bool = True
    timeout: float = Field(
        default=500.0,
        description=(
            "Matches LangChain AgentExecutor.max_execution_time (seconds), the timeout"
            " on agent execution."
        ),
    )
    should_pre_search: bool = Field(
        default=False,
        description="If set to true, run the search tool before invoking agent.",
    )
    agent_config: dict[str, Any] | None = Field(
        default=None,
        description="Optional keyword argument configuration for the agent.",
    )
    tool_names: Collection[str] | None = Field(
        default=None,
        description=(
            "Optional override on the tools to provide the agent. Leaving as the"
            " default of None will use a minimal toolset of the paper search, gather"
            " evidence, collect cited papers from evidence, and gen answer. If passing tool"
            " names (non-default route), at least the gen answer tool must be supplied."
        ),
    )

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, v: set[str] | None) -> set[str] | None:
        if v is None:
            return None
        # imported here to avoid circular imports
        from .main import GenerateAnswerTool

        answer_tool_name = GenerateAnswerTool.__fields__["name"].default
        if answer_tool_name not in v:
            raise ValueError(
                f"If using an override, must contain at least the {answer_tool_name}."
            )
        return v


class ParsingOptions(str, Enum):
    PAPERQA_DEFAULT = "paperqa_default"

    def available_for_inference(self) -> list[ParsingOptions]:
        return [self.PAPERQA_DEFAULT]  # type: ignore[list-item]

    def get_parse_type(self, config: ParsingConfiguration) -> str:
        if self == ParsingOptions.PAPERQA_DEFAULT:
            return config.parser_version_string
        assert_never()


class ChunkingOptions(str, Enum):
    SIMPLE_OVERLAP = "simple_overlap"

    @property
    def valid_parsings(self) -> list[ParsingOptions]:
        # Note that SIMPLE_OVERLAP must be valid for all by default
        # TODO: implement for future parsing options
        valid_parsing_dict: dict[str, list[ParsingOptions]] = {}
        return valid_parsing_dict.get(self.value, [])


class ImpossibleParsingError(Exception):
    """Error to throw when a parsing is impossible."""

    LOG_METHOD_NAME: ClassVar[str] = "warning"


class ParsingConfiguration(BaseModel):
    """Holds a superset of params and methods needed for each algorithm."""

    ordered_parser_preferences: list[ParsingOptions] = [
        ParsingOptions.PAPERQA_DEFAULT,
    ]
    chunksize: int = 6000
    overlap: int = 100
    chunking_algorithm: ChunkingOptions = ChunkingOptions.SIMPLE_OVERLAP

    def chunk_type(self, chunking_selection: ChunkingOptions | None = None) -> str:
        """Future chunking implementations (i.e. by section) will get an elif clause here."""
        if chunking_selection is None:
            chunking_selection = self.chunking_algorithm
        if chunking_selection == ChunkingOptions.SIMPLE_OVERLAP:
            return (
                f"{self.parser_version_string}|{chunking_selection.value}"
                f"|tokens={self.chunksize}|overlap={self.overlap}"
            )
        assert_never()

    @property
    def parser_version_string(self) -> str:
        return f"paperqa-{__version__}"

    def is_chunking_valid_for_parsing(self, parsing: str):
        # must map the parsings because they won't include versions by default
        return (
            self.chunking_algorithm == ChunkingOptions.SIMPLE_OVERLAP
            or parsing
            in {  # type: ignore[unreachable]
                p.get_parse_type(self) for p in self.chunking_algorithm.valid_parsings
            }
        )


class MismatchedModelsError(Exception):
    """Error to throw when model clients clash ."""

    LOG_METHOD_NAME: ClassVar[str] = "warning"


class QueryRequest(BaseModel):
    query: str = ""
    id: UUID = Field(
        default_factory=uuid4,
        description="Identifier which will be propagated to the Answer object.",
    )
    llm: str = "gpt-4o-2024-08-06"
    agent_llm: str = Field(
        default="gpt-4o-2024-08-06",
        description="Chat model to use for agent planning",
    )
    summary_llm: str = "gpt-4o-2024-08-06"
    length: str = "about 200 words, but can be longer if necessary"
    summary_length: str = "about 100 words"
    max_sources: int = 10
    consider_sources: int = 16
    named_prompt: str | None = None
    # if you change this to something other than default
    # modify code below in update_prompts
    prompts: PromptCollection = Field(
        default=STATIC_PROMPTS["default"], validate_default=True
    )
    agent_tools: AgentPromptCollection = Field(default_factory=AgentPromptCollection)
    texts_index_mmr_lambda: float = 1.0
    texts_index_embedding_config: dict[str, Any] | None = None
    docs_index_mmr_lambda: float = 1.0
    docs_index_embedding_config: dict[str, Any] | None = None
    parsing_configuration: ParsingConfiguration = ParsingConfiguration()
    embedding: str = "text-embedding-3-small"
    # concurrent number of summary calls to use inside Doc object
    max_concurrent: int = 20
    temperature: float = 0.0
    summary_temperature: float = 0.0
    # at what size should we start using adoc_match?
    adoc_match_threshold: int = 500
    # Should we filter out "Extra Background Information" citations
    # which come from pre-step in paper-qa algorithm
    filter_extra_background: bool = True
    # provides post-hoc linkage of request to a docs object
    # NOTE: this isn't a unique field, on the user to keep straight
    _docs_name: str | None = PrivateAttr(default=None)

    # strict validation for now
    model_config = ConfigDict(extra="forbid")

    @computed_field  # type: ignore[misc]
    @property
    def docs_name(self) -> str | None:
        return self._docs_name

    @model_validator(mode="after")
    def llm_models_must_match(self) -> QueryRequest:
        llm = llm_model_factory(self.llm)
        summary_llm = llm_model_factory(self.summary_llm)
        if type(llm) is not type(summary_llm):
            raise MismatchedModelsError(
                f"Answer LLM and summary LLM types must match: {type(llm)} != {type(summary_llm)}"
            )
        return self

    @field_validator("prompts")
    def update_prompts(
        cls,  # noqa: N805
        v: PromptCollection,
        info: ValidationInfo,
    ) -> PromptCollection:
        values = info.data
        if values["named_prompt"] is not None:
            if values["named_prompt"] not in STATIC_PROMPTS:
                raise ValueError(
                    f"Named prompt {values['named_prompt']} not in {list(STATIC_PROMPTS.keys())}"
                )
            v = STATIC_PROMPTS[values["named_prompt"]]
        if values["summary_llm"] == "none":
            v.skip_summary = True
            # for simplicity (it is not used anywhere)
            # so that Docs doesn't break when we don't have a summary_llm
            values["summary_llm"] = "gpt-4o-mini"
        return v

    def set_docs_name(self, docs_name: str) -> None:
        """Set the internal docs name for tracking."""
        self._docs_name = docs_name

    @staticmethod
    def get_index_name(
        paper_directory: str | os.PathLike,
        embedding: str,
        parsing_configuration: ParsingConfiguration,
    ) -> str:

        # index name should use an absolute path
        # this way two different folders where the
        # user locally uses '.' will make different indexes
        if isinstance(paper_directory, Path):
            paper_directory = str(paper_directory.absolute())

        index_fields = "|".join(
            [
                str(paper_directory),  # cast for typing
                embedding,
                str(parsing_configuration.chunksize),
                str(parsing_configuration.overlap),
                parsing_configuration.chunking_algorithm,
            ]
        )

        return f"pqa_index_{hexdigest(index_fields)}"


class AnswerResponse(BaseModel):
    answer: Answer
    usage: dict[str, list[int]]
    bibtex: dict[str, str] | None = None
    status: AgentStatus
    timing_info: dict[str, dict[str, float]] | None = None
    duration: float = 0.0
    # A placeholder for interesting statistics we can show users
    # about the answer, such as the number of sources used, etc.
    stats: dict[str, str] | None = None

    @field_validator("answer")
    def strip_answer(
        cls, v: Answer, info: ValidationInfo  # noqa: ARG002, N805
    ) -> Answer:
        # This modifies in place, this is fine
        # because when a response is being constructed,
        # we should be done with the Answer object
        v.filter_content_for_user()
        return v

    async def get_summary(self, llm_model="gpt-4-turbo") -> str:
        sys_prompt = (
            "Revise the answer to a question to be a concise SMS message. "
            "Use abbreviations or emojis if necessary."
        )
        model = OpenAILLMModel(config={"model": llm_model, "temperature": 0.1})
        chain = model.make_chain(
            AsyncOpenAI(), prompt="{question}\n\n{answer}", system_prompt=sys_prompt
        )
        result = await chain({"question": self.answer.question, "answer": self.answer.answer})  # type: ignore[call-arg]
        return result.text.strip()


class TimerData(BaseModel):
    start_time: float = Field(default_factory=time.time)  # noqa: FURB111
    durations: list[float] = Field(default_factory=list)


class SimpleProfiler(BaseModel):
    """Basic profiler with start/stop and named timers.

    The format for this logger needs to be strictly followed, as downstream google
    cloud monitoring is based on the following
    # [Profiling] {**name** of timer} | {**elapsed** time of function} | {**__version__** of PaperQA}
    """

    timers: dict[str, list[float]] = {}
    running_timers: dict[str, TimerData] = {}
    uid: UUID = Field(default_factory=uuid4)

    @asynccontextmanager
    async def timer(self, name: str):
        start_time = asyncio.get_running_loop().time()
        try:
            yield
        finally:
            end_time = asyncio.get_running_loop().time()
            elapsed = end_time - start_time
            self.timers.setdefault(name, []).append(elapsed)
            logger.info(
                f"[Profiling] | UUID: {self.uid} | NAME: {name} | TIME: {elapsed:.3f}s | VERSION: {__version__}"
            )

    def start(self, name: str) -> None:
        try:
            self.running_timers[name] = TimerData()
        except RuntimeError:  # No running event loop (not in async)
            self.running_timers[name] = TimerData(start_time=time.time())

    def stop(self, name: str):
        timer_data = self.running_timers.pop(name, None)
        if timer_data:
            try:
                t_stop: float = asyncio.get_running_loop().time()
            except RuntimeError:  # No running event loop (not in async)
                t_stop = time.time()
            elapsed = t_stop - timer_data.start_time
            self.timers.setdefault(name, []).append(elapsed)
            logger.info(
                f"[Profiling] | UUID: {self.uid} | NAME: {name} | TIME: {elapsed:.3f}s | VERSION: {__version__}"
            )
        else:
            logger.warning(f"Timer {name} not running")

    def results(self) -> dict[str, dict[str, float]]:
        result = {}
        for name, durations in self.timers.items():
            mean = sum(durations) / len(durations)
            result[name] = {
                "low": min(durations),
                "mean": mean,
                "max": max(durations),
                "total": sum(durations),
            }
        return result


class AgentCallback(AsyncCallbackHandler):
    """
    Callback handler used to monitor the agent, for debugging.

    Its various capabilities include:
    - Chain start --> error/stop: profile runtime
    - Tool start: count tool invocations
    - LLM start --> error/stop: insert into LLMResultDB

    NOTE: this is not a thread safe implementation since start(s)/end(s) mutate self.
    """

    def __init__(
        self, profiler: SimpleProfiler, name: str, answer_id: UUID, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.profiler = profiler
        self.name = name
        self._tool_starts: list[str] = []
        self._answer_id = answer_id
        # This will be None before/after a completion, and a dict during one
        self._llm_result_db_kwargs: dict[str, Any] | None = None

    @property
    def tool_invocations(self) -> list[str]:
        return self._tool_starts

    async def on_chain_start(self, *args, **kwargs) -> None:
        await super().on_chain_start(*args, **kwargs)
        self.profiler.start(self.name)

    async def on_chain_end(self, *args, **kwargs) -> None:
        await super().on_chain_end(*args, **kwargs)
        self.profiler.stop(self.name)

    async def on_chain_error(self, *args, **kwargs) -> None:
        await super().on_chain_error(*args, **kwargs)
        self.profiler.stop(self.name)

    async def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs
    ) -> None:
        await super().on_tool_start(serialized, input_str, **kwargs)
        self._tool_starts.append(serialized["name"])

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002
        messages: list[list[BaseMessage]],
        **kwargs,
    ) -> None:
        # NOTE: don't call super(), as it changes semantics
        if len(messages) != 1:
            raise NotImplementedError(f"Didn't handle shape of messages {messages}.")
        self._llm_result_db_kwargs = {
            "answer_id": self._answer_id,
            "name": f"tool_selection:{len(messages[0])}",
            "prompt": {
                "messages": messages_to_dict(messages[0]),
                # SEE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-functions
                "functions": kwargs["invocation_params"]["functions"],
                "tool_history": self.tool_invocations,
            },
            "model": kwargs["invocation_params"]["model"],
            "date": datetime.now().isoformat(),
        }

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        await super().on_llm_end(response, **kwargs)
        if (
            len(response.generations) != 1
            or len(response.generations[0]) != 1
            or not isinstance(response.generations[0][0], ChatGeneration)
        ):
            raise NotImplementedError(
                f"Didn't handle shape of generations {response.generations}."
            )
        if self._llm_result_db_kwargs is None:
            raise NotImplementedError(
                "There should have been an LLM result populated here by now."
            )
        if not isinstance(response.llm_output, dict):
            raise NotImplementedError(
                f"Expected llm_output to be a dict, but got {response.llm_output}."
            )
