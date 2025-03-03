import logging
from copy import deepcopy
from typing import Any, ClassVar, Self, cast
from uuid import UUID

from aviary.core import (
    Environment,
    Frame,
    Message,
    Messages,
    Tool,
    ToolRequestMessage,
    ToolResponseMessage,
)
from aviary.utils import MultipleChoiceQuestion
from lmi import EmbeddingModel, LiteLLMModel

from paperqa.docs import Docs
from paperqa.settings import Settings
from paperqa.sources.clinical_trials import (
    CLINICAL_TRIALS_BASE,
    partition_clinical_trials_by_source,
)
from paperqa.types import PQASession
from paperqa.utils import get_year

from .tools import (
    AVAILABLE_TOOL_NAME_TO_CLASS,
    DEFAULT_TOOL_NAMES,
    ClinicalTrialsSearch,
    Complete,
    EnvironmentState,
    GatherEvidence,
    GenerateAnswer,
    PaperSearch,
    Reset,
)

logger = logging.getLogger(__name__)

POPULATE_FROM_SETTINGS = None


def settings_to_tools(  # noqa: PLR0912
    settings: Settings,
    llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
    summary_llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
    embedding_model: EmbeddingModel | None = POPULATE_FROM_SETTINGS,
) -> list[Tool]:
    """
    Convert a Settings into tools, confirming the complete tool is present.

    NOTE: the last element of the return will always be Complete.
    """
    llm_model = llm_model or settings.get_llm()
    summary_llm_model = summary_llm_model or settings.get_summary_llm()
    embedding_model = embedding_model or settings.get_embedding_model()
    tools: list[Tool] = []
    for tool_type in (
        [AVAILABLE_TOOL_NAME_TO_CLASS[name] for name in DEFAULT_TOOL_NAMES]
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
                tool.info.get_properties()[pname]["description"] = cast(
                    "str", tool.info.get_properties()[pname]["description"]
                ).format(current_year=get_year())
        elif issubclass(tool_type, GatherEvidence):
            gather_evidence_tool = GatherEvidence(
                settings=settings,
                summary_llm_model=summary_llm_model,
                embedding_model=embedding_model,
            )

            # if we're using the SearchClinicalTrialsTool,
            # we override this tool's docstring/prompt
            # because the default prompt is unaware of the clinical trials tool

            if ClinicalTrialsSearch.TOOL_FN_NAME in (
                settings.agent.tool_names or DEFAULT_TOOL_NAMES
            ):
                gather_evidence_tool.gather_evidence.__func__.__doc__ = (  # type: ignore[attr-defined]
                    ClinicalTrialsSearch.GATHER_EVIDENCE_TOOL_PROMPT_OVERRIDE
                )
                gather_evidence_tool.partitioning_fn = (
                    partition_clinical_trials_by_source
                )

            tool = Tool.from_function(gather_evidence_tool.gather_evidence)

        elif issubclass(tool_type, GenerateAnswer):
            generate_answer_tool = GenerateAnswer(
                settings=settings,
                llm_model=llm_model,
                summary_llm_model=summary_llm_model,
                embedding_model=embedding_model,
            )

            if ClinicalTrialsSearch.TOOL_FN_NAME in (
                settings.agent.tool_names or DEFAULT_TOOL_NAMES
            ):
                generate_answer_tool.partitioning_fn = (
                    partition_clinical_trials_by_source
                )

            tool = Tool.from_function(generate_answer_tool.gen_answer)

        elif issubclass(tool_type, Reset):
            tool = Tool.from_function(Reset().reset)
        elif issubclass(tool_type, Complete):
            tool = Tool.from_function(Complete().complete)
        elif issubclass(tool_type, ClinicalTrialsSearch):
            tool = Tool.from_function(
                ClinicalTrialsSearch(
                    search_count=settings.agent.search_count,
                    settings=settings,
                ).clinical_trials_search
            )
        else:
            raise NotImplementedError(f"Didn't handle tool type {tool_type}.")
        if tool.info.name == Complete.complete.__name__:
            tools.append(tool)  # Place at the end
        else:
            tools.insert(0, tool)
    return tools


def make_clinical_trial_status(
    total_paper_count: int,
    relevant_paper_count: int,
    total_clinical_trials: int,
    relevant_clinical_trials: int,
    evidence_count: int,
    cost: float,
) -> str:
    return (
        f"Status: Paper Count={total_paper_count}"
        f" | Relevant Papers={relevant_paper_count}"
        f" | Clinical Trial Count={total_clinical_trials}"
        f" | Relevant Clinical Trials={relevant_clinical_trials}"
        f" | Current Evidence={evidence_count}"
        f" | Current Cost=${cost:.4f}"
    )


# SEE: https://regex101.com/r/L0L5MH/1
CLINICAL_STATUS_SEARCH_REGEX_PATTERN: str = (
    r"Status: Paper Count=(\d+) \| Relevant Papers=(\d+)(?:\s\|\sClinical Trial"
    r" Count=(\d+)\s\|\sRelevant Clinical Trials=(\d+))?\s\|\sCurrent Evidence=(\d+)"
)


def clinical_trial_status(state: "EnvironmentState") -> str:
    relevant_contexts = state.get_relevant_contexts()
    return make_clinical_trial_status(
        total_paper_count=len(
            {
                d.dockey
                for d in state.docs.docs.values()
                if CLINICAL_TRIALS_BASE
                not in getattr(d, "other", {}).get("client_source", [])
            }
        ),
        relevant_paper_count=len(
            {
                c.text.doc.dockey
                for c in relevant_contexts
                if CLINICAL_TRIALS_BASE
                not in getattr(c.text.doc, "other", {}).get("client_source", [])
            }
        ),
        total_clinical_trials=len(
            {
                d.dockey
                for d in state.docs.docs.values()
                if CLINICAL_TRIALS_BASE
                in getattr(d, "other", {}).get("client_source", [])
            }
        ),
        relevant_clinical_trials=len(
            {
                c.text.doc.dockey
                for c in relevant_contexts
                if CLINICAL_TRIALS_BASE
                in getattr(c.text.doc, "other", {}).get("client_source", [])
            }
        ),
        evidence_count=len(relevant_contexts),
        cost=state.session.cost,
    )


class PaperQAEnvironment(Environment[EnvironmentState]):
    """Environment connecting paper-qa's tools with state."""

    def __init__(
        self,
        query: str | MultipleChoiceQuestion,
        settings: Settings,
        docs: Docs,
        llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        summary_llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        embedding_model: EmbeddingModel | None = POPULATE_FROM_SETTINGS,
        session_id: UUID | None = None,
        **env_kwargs,
    ):
        super().__init__(**env_kwargs)
        self._query = query
        self._settings = settings
        self._docs = docs
        self._llm_model = llm_model
        self._summary_llm_model = summary_llm_model
        self._embedding_model = embedding_model
        self._session_id = session_id

    def make_tools(self) -> list[Tool]:
        return settings_to_tools(
            settings=self._settings,
            llm_model=self._llm_model,
            summary_llm_model=self._summary_llm_model,
            embedding_model=self._embedding_model,
        )

    def make_initial_state(self) -> EnvironmentState:
        status_fn = None

        if ClinicalTrialsSearch.TOOL_FN_NAME in (
            self._settings.agent.tool_names or DEFAULT_TOOL_NAMES
        ):
            status_fn = clinical_trial_status

        session_kwargs: dict[str, Any] = {}
        if self._session_id:
            session_kwargs["id"] = self._session_id
        return EnvironmentState(
            docs=self._docs,
            session=PQASession(
                question=(
                    self._query
                    if isinstance(self._query, str)
                    else self._query.question_prompt
                ),
                config_md5=self._settings.md5,
                **session_kwargs,
            ),
            status_fn=status_fn,
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
                    content=self._settings.agent.agent_prompt.format(
                        question=self.state.session.question,
                        status=self.state.status,
                        complete_tool_name=Complete.TOOL_FN_NAME,
                    ),
                )
            ],
            self.tools,
        )

    def export_frame(self) -> Frame:
        return Frame(state=self.state, info={"query": self._query})

    def _has_excess_answer_failures(self) -> bool:
        if self._settings.answer.max_answer_attempts is None:
            return False
        return (
            sum(
                tn == GenerateAnswer.gen_answer.__name__
                for s in self.state.session.tool_history
                for tn in s
            )
            > self._settings.answer.max_answer_attempts
        )

    USE_POST_PROCESSED_REWARD: ClassVar[float] = 0.0

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        self.state.record_action(action)

        response_messages = cast(
            "list[Message]",
            await self.exec_tool_calls(
                action,
                concurrency=False,  # PQA tools aren't yet concurrency safe
                state=self.state,
                handle_tool_exc=True,
            ),
        ) or [Message(content=f"No tool calls input in tool request {action}.")]
        done = any(
            isinstance(msg, ToolResponseMessage)
            and msg.name == Complete.complete.__name__
            for msg in response_messages
        )
        if not done and self._has_excess_answer_failures():
            # If the caller set max_answer_attempts, and the agent has tried to answer
            # too many times, we consider this done, but we cannot determine success
            # because we're not calling the complete tool
            self.state.session.has_successful_answer = None
            done = True
        return (
            response_messages,
            self.USE_POST_PROCESSED_REWARD,
            done,
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
            query=self._query,  # No need to copy since we read only
            settings=deepcopy(self._settings, memo),  # Deepcopy just to be safe
            docs=copy_state.docs,
            **env_model_kwargs,
        )
        copy_self.state = copy_state
        # Because we shallow copied the LiteLLMModels, we need to re-make the
        # tool functions within the tools
        copy_self.tools = copy_self.make_tools()
        return copy_self
