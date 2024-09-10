from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar
from uuid import UUID, uuid4

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage, messages_to_dict
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    computed_field,
    field_validator,
)
from typing_extensions import Protocol

from paperqa.llms import LiteLLMModel
from paperqa.settings import Settings
from paperqa.types import Answer
from paperqa.version import __version__

logger = logging.getLogger(__name__)


class SupportsPickle(Protocol):
    """Type protocol for typing any object that supports pickling."""

    def __reduce__(self) -> str | tuple[Any, ...]: ...
    def __getstate__(self) -> object: ...
    def __setstate__(self, state: object) -> None: ...


class AgentStatus(StrEnum):
    # FAIL - no answer could be generated
    FAIL = "fail"
    # SUCCESS - answer was generated
    SUCCESS = "success"
    # TIMEOUT - agent took too long, but an answer was generated
    TIMEOUT = "timeout"
    # UNSURE - the agent was unsure, but an answer is present
    UNSURE = "unsure"


class ImpossibleParsingError(Exception):
    """Error to throw when a parsing is impossible."""

    LOG_METHOD_NAME: ClassVar[str] = "warning"


class MismatchedModelsError(Exception):
    """Error to throw when model clients clash ."""

    LOG_METHOD_NAME: ClassVar[str] = "warning"


class QueryRequest(BaseModel):
    query: str = ""
    id: UUID = Field(
        default_factory=uuid4,
        description="Identifier which will be propagated to the Answer object.",
    )
    settings_template: str | None = None
    settings: Settings = Field(default_factory=Settings, validate_default=True)
    # provides post-hoc linkage of request to a docs object
    # NOTE: this isn't a unique field, on the user to keep straight
    _docs_name: str | None = PrivateAttr(default=None)

    model_config = ConfigDict(extra="forbid")

    @field_validator("settings")
    @classmethod
    def apply_settings_template(cls, v: Settings, info: ValidationInfo) -> Settings:
        if info.data["settings_template"] and isinstance(v, Settings):
            base_settings = Settings.from_name(info.data["settings_template"])
            return Settings(**(base_settings.model_dump() | v.model_dump()))
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def docs_name(self) -> str | None:
        return self._docs_name

    def set_docs_name(self, docs_name: str) -> None:
        """Set the internal docs name for tracking."""
        self._docs_name = docs_name


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

    async def get_summary(self, llm_model: str = "gpt-4o") -> str:
        sys_prompt = (
            "Revise the answer to a question to be a concise SMS message. "
            "Use abbreviations or emojis if necessary."
        )
        model = LiteLLMModel(name=llm_model)
        result = await model.run_prompt(
            prompt="{question}\n\n{answer}",
            data={"question": self.answer.question, "answer": self.answer.answer},
            system_prompt=sys_prompt,
        )
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
                f"[Profiling] | UUID: {self.uid} | NAME: {name} | TIME: {elapsed:.3f}s"
                f" | VERSION: {__version__}"
            )

    def start(self, name: str) -> None:
        try:
            self.running_timers[name] = TimerData()
        except RuntimeError:  # No running event loop (not in async)
            self.running_timers[name] = TimerData(start_time=time.time())

    def stop(self, name: str) -> None:
        timer_data = self.running_timers.pop(name, None)
        if timer_data:
            try:
                t_stop: float = asyncio.get_running_loop().time()
            except RuntimeError:  # No running event loop (not in async)
                t_stop = time.time()
            elapsed = t_stop - timer_data.start_time
            self.timers.setdefault(name, []).append(elapsed)
            logger.info(
                f"[Profiling] | UUID: {self.uid} | NAME: {name} | TIME: {elapsed:.3f}s"
                f" | VERSION: {__version__}"
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
