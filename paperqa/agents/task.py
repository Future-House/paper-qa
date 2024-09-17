__all__ = [
    "ENV_NAME",
    "TASK_DATASET_NAME",
    "GradablePaperQAEnvironment",
    "LitQATaskDataset",
    "LitQAv2TaskDataset",
    "LitQAv2TaskSplit",
]

from abc import ABC
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, assert_never

from aviary.env import ENV_REGISTRY, TASK_DATASET_REGISTRY, Frame, TaskDataset
from aviary.message import Message
from aviary.tools import ToolRequestMessage, ToolResponseMessage

try:
    from ldp.alg.callbacks import ComputeTrajectoryMetricsMixin
except ImportError:

    class ComputeTrajectoryMetricsMixin:  # type: ignore[no-redef]
        """Placeholder for when ldp isn't installed."""


from paperqa.docs import Docs
from paperqa.litqa import (
    DEFAULT_EVAL_MODEL_NAME,
    DEFAULT_LABBENCH_HF_HUB_NAME,
    DEFAULT_REWARD_DISTRIBUTION,
    LitQAEvaluation,
    read_litqa_v2_from_hub,
)
from paperqa.llms import EmbeddingModel, LiteLLMModel, LLMModel
from paperqa.types import Answer

from .env import POPULATE_FROM_SETTINGS, PaperQAEnvironment
from .models import QueryRequest
from .tools import GenerateAnswer

if TYPE_CHECKING:
    from ldp.data_structures import Trajectory


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
            Callable[[Answer | str], Awaitable[LitQAEvaluation]] | None
        ) = None,
        rewards: Sequence[float] = DEFAULT_REWARD_DISTRIBUTION,
        evaluation_callback: Callable[[LitQAEvaluation], Awaitable] | None = None,
        **env_kwargs,
    ):
        super().__init__(
            query, docs, llm_model, summary_llm_model, embedding_model, **env_kwargs
        )
        self._evaluation_from_answer = evaluation_from_answer
        self._evaluation_callback = evaluation_callback
        self._rewards = rewards

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[list[Message], float, bool, bool]:
        messages, reward, done, truncated = await super().step(action)
        if not done or not self._evaluation_from_answer:
            return messages, reward, done, truncated
        # Filter out non-answer messages (in case parallel tool calls)
        answer_tool_messages = [
            m
            for m in messages
            if isinstance(m, ToolResponseMessage)
            and m.name == GenerateAnswer.gen_answer.__name__
        ]
        if not answer_tool_messages:  # No answer, so no positive reward
            return messages, reward, done, truncated
        if len(answer_tool_messages) != 1:
            raise NotImplementedError(
                f"Expected just one answer message, got {messages}."
            )
        answer = GenerateAnswer.extract_answer_from_message(
            content=answer_tool_messages[0].content
        )
        if not answer:
            return messages, reward, done, truncated
        evaluation = await self._evaluation_from_answer(answer)
        if evaluation_callback := self._evaluation_callback:
            await evaluation_callback(evaluation)
        return messages, reward + self._rewards[evaluation.value], done, truncated

    def export_frame(self) -> Frame:
        raise NotImplementedError("Didn't yet need to export a frame.")


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
        base_query: QueryRequest | None = None,
        base_docs: Docs | None = None,
        rewards: Sequence[float] = DEFAULT_REWARD_DISTRIBUTION,
        eval_model: LLMModel | str = DEFAULT_EVAL_MODEL_NAME,
        **env_kwargs,
    ):
        self._base_query = base_query or QueryRequest()
        self._base_docs = base_docs or Docs()
        self._rewards = rewards
        self._env_kwargs = env_kwargs
        self._eval_model = eval_model

    def _make_gradable_environment(
        self,
        ideal: str,
        distractors: str | list[str],
        question: str,
        use_unsure: bool = True,
    ) -> GradablePaperQAEnvironment:
        qa_prompt, evaluation_from_answer = LitQAEvaluation.from_question(
            ideal=ideal,
            distractors=distractors,
            question=question,
            use_unsure=use_unsure,
            eval_model=self._eval_model,
        )
        query = self._base_query.model_copy()
        query.query = qa_prompt
        return GradablePaperQAEnvironment(
            query=query,
            docs=self._base_docs.model_copy(),
            evaluation_from_answer=evaluation_from_answer,
            rewards=self._rewards,
            **self._env_kwargs,
        )

    def compute_trajectory_metrics(
        self, trajectories: "Sequence[Trajectory]"
    ) -> dict[str, list[float]]:
        return super().compute_trajectory_metrics(trajectories) | {
            "correct": [
                int(traj.steps[-1].reward == self._rewards[0]) for traj in trajectories
            ],
            "correct_unsure": [
                int(traj.steps[-1].reward in {self._rewards[0], self._rewards[1]})
                for traj in trajectories
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
        split: str | LitQAv2TaskSplit = LitQAv2TaskSplit.EVAL,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        train_df, eval_df = read_litqa_v2_from_hub(labbench_dataset)
        split = LitQAv2TaskSplit(split)
        if split == LitQAv2TaskSplit.TRAIN:
            self.data = train_df
        elif split == LitQAv2TaskSplit.EVAL:
            self.data = eval_df
        else:
            assert_never(split)

    def get_new_env_by_idx(self, idx: int) -> GradablePaperQAEnvironment:
        return self._make_gradable_environment(
            ideal=self.data.iloc[idx].ideal,
            distractors=self.data.iloc[idx].distractors,
            question=self.data.iloc[idx].question,
        )

    def __len__(self) -> int:
        return len(self.data)


TASK_DATASET_NAME = "litqa-v2"
TASK_DATASET_REGISTRY[TASK_DATASET_NAME] = (
    LitQAv2TaskDataset.__module__,
    LitQAv2TaskDataset.__name__,
)
