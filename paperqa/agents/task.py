__all__ = [
    "ENV_NAME",
    "TASK_DATASET_NAME",
    "GradablePaperQAEnvironment",
    "LitQATaskDataset",
    "LitQAv2TaskDataset",
    "LitQAv2TaskSplit",
]

import logging
import re
from abc import ABC
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self, assert_never

from aviary.core import (
    TASK_DATASET_REGISTRY,
    Frame,
    Messages,
    TaskDataset,
    ToolRequestMessage,
    ToolResponseMessage,
)
from aviary.env import ENV_REGISTRY

from paperqa.types import DocDetails

from .search import SearchIndex, maybe_get_manifest

try:
    from ldp.alg import ComputeTrajectoryMetricsMixin
except ImportError:

    class ComputeTrajectoryMetricsMixin:  # type: ignore[no-redef]
        """Placeholder for when ldp isn't installed."""


from paperqa.docs import Docs
from paperqa.litqa import (
    DEFAULT_EVAL_MODEL_NAME,
    DEFAULT_LABBENCH_HF_HUB_NAME,
    DEFAULT_REWARD_MAPPING,
    LitQAEvaluation,
    read_litqa_v2_from_hub,
)
from paperqa.llms import EmbeddingModel, LiteLLMModel, LLMModel
from paperqa.types import PQASession

from .env import POPULATE_FROM_SETTINGS, PaperQAEnvironment
from .models import QueryRequest
from .tools import Complete

if TYPE_CHECKING:
    from ldp.data_structures import Trajectory

logger = logging.getLogger(__name__)


class GradablePaperQAEnvironment(PaperQAEnvironment):
    """Extended environment that can grade answers."""

    def __init__(
        self,
        query: QueryRequest,
        docs: Docs,
        llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        summary_llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        embedding_model: EmbeddingModel | None = POPULATE_FROM_SETTINGS,
        evaluation_from_answer: (
            Callable[[PQASession | str], Awaitable[LitQAEvaluation]] | None
        ) = None,
        sources: str | list[str] | None = None,
        rewards: Mapping[str, float] = DEFAULT_REWARD_MAPPING,
        evaluation_callback: Callable[[LitQAEvaluation], Awaitable] | None = None,
        **env_kwargs,
    ):
        super().__init__(
            query, docs, llm_model, summary_llm_model, embedding_model, **env_kwargs
        )
        self._evaluation_from_answer = evaluation_from_answer
        # Enables checking an Index has the right DOI(s)
        self.sources: list[str] | None = (
            [sources] if isinstance(sources, str) else sources
        )
        self._evaluation_callback = evaluation_callback
        self._rewards = rewards

    async def validate_sources(
        self, manifest_or_index: dict[str, DocDetails] | SearchIndex | None = None
    ) -> None:
        """Validate the sources can be found in the input manifest or index."""
        if not self.sources:
            return
        if manifest_or_index is None:  # Let's try to load in the manifest
            manifest_or_index = await maybe_get_manifest(
                filename=await self._query.settings.agent.index.finalize_manifest_file()
            )
        if isinstance(manifest_or_index, SearchIndex):
            entity: str = "index"
            file_names: set[str] = {k for k in await manifest_or_index.index_files if k}
            lowercased_dois: set[str] = set()
        else:
            entity = "manifest"
            file_names = {k for k in manifest_or_index if k}
            lowercased_dois = {
                v["doi"].lower() for v in manifest_or_index.values() if v["doi"]
            }
        if not file_names:  # File names being empty means something's wrong
            logger.warning(
                f"Can't validate sources {self.sources} without a correctly specified"
                f" {entity}."
            )
            return
        not_found = [
            s
            for s in self.sources
            if s not in file_names and s.lower() not in lowercased_dois
        ]
        if not_found:
            raise ValueError(
                f"Sources {not_found} of {self.sources} not found in the {entity},"
                f" the corresponding query was {self._query.query!r}."
            )

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        messages, reward, done, truncated = await super().step(action)
        if not done or not self._evaluation_from_answer:
            return messages, reward, done, truncated
        # If the ensuring evaluation fails (e.g. due to OpenAI being down), we can:
        # - Suppress the exception and declare the evaluation as incorrect, which can
        #   negatively reward what otherwise was a good trajectory containing a correct
        #   answer. We don't want "bad" offline data, so it's not what we do.
        # - Suppress the exception and just give super()'s reward, but again this could
        #   incorrectly reward what otherwise was a good trajectory.
        # - Don't suppress the exception, which leads to the trajectory failing, and
        #   removes it from the learnable pool. This is the only safe default behavior.
        evaluation = await self._evaluation_from_answer(self.state.session.answer)
        if evaluation_callback := self._evaluation_callback:
            await evaluation_callback(evaluation)
        return messages, reward + self._rewards[evaluation.value], done, truncated

    def export_frame(self) -> Frame:
        raise NotImplementedError("Didn't yet need to export a frame.")

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
            evaluation_from_answer=self._evaluation_from_answer,
            sources=self.sources,
            rewards=self._rewards,
            evaluation_callback=self._evaluation_callback,
            **env_model_kwargs,
        )
        copy_self.state = copy_state
        # Because we shallow copied the LiteLLMModels, we need to re-make the
        # tool functions within the tools
        copy_self.tools = copy_self.make_tools()
        return copy_self


ENV_NAME = "paperqa-local"
ENV_REGISTRY[ENV_NAME] = (
    GradablePaperQAEnvironment.__module__,
    GradablePaperQAEnvironment.__name__,
)


class LitQATaskDataset(
    TaskDataset[GradablePaperQAEnvironment], ComputeTrajectoryMetricsMixin, ABC
):
    """
    Abstract base class for a task dataset of LitQA v1 or v2 questions.

    This is an ABC because it's non-specific to a LitQA version.
    Examples include LitQA v1, v2, or a test stub version of LitQA.
    """

    def __init__(
        self,
        base_query: QueryRequest | dict | None = None,
        base_docs: Docs | dict | None = None,
        rewards: Mapping[str, float] = DEFAULT_REWARD_MAPPING,
        question_kwargs: Mapping[str, Any] | None = None,
        eval_model: LLMModel | str = DEFAULT_EVAL_MODEL_NAME,
        **env_kwargs,
    ):
        if base_query is None:
            base_query = QueryRequest()
        if isinstance(base_query, dict):
            base_query = QueryRequest(**base_query)
        self._base_query = base_query
        if base_docs is None:
            base_docs = Docs()
        if isinstance(base_docs, dict):
            base_docs = Docs(**base_docs)
        self._base_docs = base_docs
        self._rewards = rewards
        self._question_kwargs = question_kwargs
        self._eval_model = eval_model
        self._env_kwargs = env_kwargs

    def _make_gradable_environment(
        self,
        ideal: str,
        distractors: str | list[str],
        question: str,
        sources: str | list[str] | None = None,
    ) -> GradablePaperQAEnvironment:
        qa_prompt, evaluation_from_answer = LitQAEvaluation.from_question(
            ideal=ideal,
            distractors=distractors,
            question=question,
            eval_model=self._eval_model,
            **(self._question_kwargs or {}),
        )
        query = self._base_query.model_copy()
        query.query = qa_prompt
        return GradablePaperQAEnvironment(
            query=query,
            docs=self._base_docs.model_copy(),
            evaluation_from_answer=evaluation_from_answer,
            sources=sources,
            rewards=self._rewards,
            **self._env_kwargs,
        )

    def compute_trajectory_metrics(
        self, trajectories: "Sequence[Trajectory]"
    ) -> dict[str, list[float]]:
        total_paper_count: list[float] = []
        relevant_paper_count: list[float] = []
        evidence_count: list[float] = []
        for t in trajectories:
            split_certainties = [
                split_certainty
                for split_certainty in (
                    re.split(
                        pattern=Complete.CERTAINTY_SPLIT_REGEX_PATTERN,
                        string=obs.content,
                        maxsplit=1,
                    )
                    for obs in t.steps[-1].next_observation
                    if (
                        isinstance(obs, ToolResponseMessage)
                        and obs.name == Complete.TOOL_FN_NAME
                    )
                )
                # Filter for places where the regex split succeeded
                if len(split_certainty) >= 4  # noqa: PLR2004
            ]
            for i, metric_list in enumerate(
                (total_paper_count, relevant_paper_count, evidence_count),
                start=1,  # Regex extraction of status starts after has_successful_answer
            ):
                # NOTE: we use mean to not break if there's 2+ complete calls (which
                # we're prompted not to do). If it happens, they should all have the
                # same status, so the mean value should equal the individual values
                metric_list.append(
                    sum(int(sa[i]) for sa in split_certainties) / len(split_certainties)
                    if split_certainties  # Avoid div0 (when complete wasn't called)
                    else 0
                )
        return super().compute_trajectory_metrics(trajectories) | {
            "total_paper_count": total_paper_count,
            "relevant_paper_count": relevant_paper_count,
            "evidence_count": evidence_count,
            "correct": [
                int(t.steps[-1].reward == self._rewards["correct"])
                for t in trajectories
            ],
            "correct_unsure": [
                int(
                    t.steps[-1].reward
                    in {self._rewards["correct"], self._rewards["unsure"]}
                )
                for t in trajectories
            ],
        }


class LitQAv2TaskSplit(StrEnum):
    TRAIN = "train"
    EVAL = "eval"


class LitQAv2TaskDataset(LitQATaskDataset):
    """Task dataset of LitQA v2 questions."""

    def __init__(
        self,
        *args,
        labbench_dataset: str = DEFAULT_LABBENCH_HF_HUB_NAME,
        read_data_kwargs: Mapping[str, Any] | None = None,
        split: str | LitQAv2TaskSplit = LitQAv2TaskSplit.EVAL,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        train_df, eval_df = read_litqa_v2_from_hub(
            labbench_dataset, **(read_data_kwargs or {})
        )
        split = LitQAv2TaskSplit(split)
        if split == LitQAv2TaskSplit.TRAIN:
            self.data = train_df
        elif split == LitQAv2TaskSplit.EVAL:
            self.data = eval_df
        else:
            assert_never(split)

    def get_new_env_by_idx(self, idx: int) -> GradablePaperQAEnvironment:
        sources = []
        for s in self.data.iloc[idx].sources:
            try:
                (doi,) = (
                    s.split(substr, maxsplit=1)[1]
                    for substr in DocDetails.DOI_URL_FORMATS
                    if substr in s
                )
            except ValueError as exc:
                raise NotImplementedError(
                    f"Didn't handle DOI extraction from source {s!r}."
                ) from exc
            sources.append(doi)
        return self._make_gradable_environment(
            ideal=self.data.iloc[idx].ideal,
            distractors=self.data.iloc[idx].distractors,
            question=self.data.iloc[idx].question,
            sources=sources,
        )

    def __len__(self) -> int:
        return len(self.data)


TASK_DATASET_NAME = "litqa-v2"
TASK_DATASET_REGISTRY[TASK_DATASET_NAME] = (
    LitQAv2TaskDataset.__module__,
    LitQAv2TaskDataset.__name__,
)
