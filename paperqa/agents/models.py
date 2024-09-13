from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Any, ClassVar, Protocol
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    computed_field,
    field_validator,
)

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

    timers: dict[str, list[float]] = Field(default_factory=dict)
    running_timers: dict[str, TimerData] = Field(default_factory=dict)
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
