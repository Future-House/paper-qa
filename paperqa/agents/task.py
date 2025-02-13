__all__ = [
    "ENV_NAME",
    "TASK_DATASET_NAME",
    "GradablePaperQAEnvironment",
    "LitQATaskDataset",
    "LitQAv2TaskDataset",
    "LitQAv2TaskSplit",
]

import logging
import random
import re
import sys
from abc import ABC
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic, Self, assert_never, cast

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar  # For TypeVar.default backport
from uuid import UUID

from aviary.core import (
    TASK_DATASET_REGISTRY,
    Environment,
    Frame,
    Messages,
    TaskDataset,
    ToolRequestMessage,
    ToolResponseMessage,
)
from aviary.env import ENV_REGISTRY
from aviary.utils import (
    DEFAULT_EVAL_MODEL_NAME,
    MultipleChoiceEvaluation,
    MultipleChoiceQuestion,
)
from llmclient import CommonLLMNames, EmbeddingModel, LiteLLMModel, LLMModel
from pydantic import BaseModel, model_validator

from paperqa._ldp_shims import (
    Callback,
    ComputeTrajectoryMetricsMixin,
    bulk_evaluate_consensus,
)
from paperqa.docs import Docs
from paperqa.litqa import (
    DEFAULT_AVIARY_PAPER_HF_HUB_NAME,
    DEFAULT_LABBENCH_HF_HUB_NAME,
    DEFAULT_REWARD_MAPPING,
    read_litqa_v2_from_hub,
)
from paperqa.prompts import lfrqa_prompt_template, lfrqa_system_prompt
from paperqa.settings import Settings
from paperqa.types import DocDetails, PQASession
from paperqa.utils import strip_citations

from .env import POPULATE_FROM_SETTINGS, PaperQAEnvironment
from .search import SearchIndex, maybe_get_manifest
from .tools import Complete, EnvironmentState

if TYPE_CHECKING:
    from ldp.agent import Agent
    from ldp.data_structures import Trajectory, Transition

logger = logging.getLogger(__name__)

TEvaluation = TypeVar("TEvaluation", default=MultipleChoiceEvaluation)


class GradablePaperQAEnvironment(PaperQAEnvironment, Generic[TEvaluation]):
    """Extended environment that can grade answers."""

    def __init__(
        self,
        query: str | MultipleChoiceQuestion,
        settings: Settings,
        docs: Docs,
        llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        summary_llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        embedding_model: EmbeddingModel | None = POPULATE_FROM_SETTINGS,
        session_id: UUID | None = None,
        sources: str | list[str] | None = None,
        rewards: Mapping[str, float] = DEFAULT_REWARD_MAPPING,
        evaluation_callback: Callable[[TEvaluation], Awaitable] | None = None,
        **env_kwargs,
    ):
        super().__init__(
            query,
            settings,
            docs,
            llm_model,
            summary_llm_model,
            embedding_model,
            session_id,
            **env_kwargs,
        )
        # Enables checking an Index has the right DOI(s)
        self.sources: list[str] | None = (
            [sources] if isinstance(sources, str) else sources
        )
        self._evaluation_callback = evaluation_callback
        self._rewards = rewards

    async def validate_sources(
        self, manifest_or_index: dict[str, dict[str, Any]] | SearchIndex | None = None
    ) -> None:
        """Validate the sources can be found in the input manifest or index."""
        if not self.sources:
            return
        if manifest_or_index is None:  # Let's try to load in the manifest
            manifest_or_index = await maybe_get_manifest(
                filename=await self._settings.agent.index.finalize_manifest_file()
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
            question = (
                self._query
                if isinstance(self._query, str)
                else self._query.question_prompt
            )
            raise ValueError(
                f"Sources {not_found} of {self.sources} not found in the {entity},"
                f" the corresponding query was {question!r}."
            )

    async def _evaluate_answer(self) -> TEvaluation:
        # If the ensuring evaluation fails (e.g. due to OpenAI being down), we can:
        # - Suppress the exception and declare the evaluation as incorrect, which can
        #   negatively reward what otherwise was a good trajectory containing a correct
        #   answer. We don't want "bad" offline data, so it's not what we do.
        # - Suppress the exception and just give super()'s reward, but again this could
        #   incorrectly reward what otherwise was a good trajectory.
        # - Don't suppress the exception, which leads to the trajectory failing, and
        #   removes it from the learnable pool. This is the only safe default behavior.
        evaluation, self.state.session.graded_answer = await cast(
            MultipleChoiceQuestion, self._query
        ).grade(self.state.session.answer)
        return evaluation  # type: ignore[return-value]

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        messages, reward, done, truncated = await super().step(action)
        if not done or not isinstance(self._query, MultipleChoiceQuestion):
            return messages, reward, done, truncated
        evaluation = await self._evaluate_answer()
        if evaluation_callback := self._evaluation_callback:
            await evaluation_callback(evaluation)

        return (
            messages,
            reward + self._rewards[cast(MultipleChoiceEvaluation, evaluation).value],
            done,
            truncated,
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


async def evaluate_consensus_sampling(
    data: Iterable[GradablePaperQAEnvironment | Frame],
    exclude_no_answer: bool = False,
    num_samples: int = 1,
    seed: int | None = None,
) -> tuple[dict[str, list[tuple[str, int]]], float]:
    """
    Create consensus groups based on question and evaluate the consensus for each.

    Args:
        data: Data to evaluate consensus upon, either gradable environments or frames.
        exclude_no_answer: Opt-in flag to filter out empty answers (due to the
            Environment/Frame not having a graded answer). Use of this flag does not
            affect the accuracy term of the return.
        num_samples: Passed through to evaluate_consensus.
        seed: Passed through to evaluate_consensus.

    Returns:
        Two-tuple of consensus list generated by collections.Counter.most_common (keys
            are question, values are list of (answer, vote count)) and the proportion of
            groups for which the consensus matches the ideal.
    """

    def extract_question(x: GradablePaperQAEnvironment | Frame) -> str:
        if isinstance(x, GradablePaperQAEnvironment):
            query: str | MultipleChoiceQuestion | dict[str, Any] = x._query
        else:
            query = x.info["query"]  # type: ignore[call-overload,index]
        if isinstance(query, str):
            return query
        if isinstance(query, MultipleChoiceQuestion):
            return query.question_prompt
        return query["question"]

    def extract_answer(x: GradablePaperQAEnvironment | Frame) -> str:
        ses: PQASession | dict[str, Any] = (
            x.state.session
            if isinstance(x.state, EnvironmentState)
            else cast(PQASession | dict[str, Any], x.state["session"])  # type: ignore[call-overload,index]
        )
        graded_answer = (
            ses.graded_answer if isinstance(ses, PQASession) else ses["graded_answer"]
        )
        # One can filter the below empty string injection via the exclude_no_answer arg
        return graded_answer or ""

    def extract_ideal(x: GradablePaperQAEnvironment | Frame) -> str:
        if isinstance(x, GradablePaperQAEnvironment):
            query: str | MultipleChoiceQuestion | dict[str, Any] = x._query
        else:
            query = x.info["query"]  # type: ignore[call-overload,index]
        if isinstance(query, str):
            raise ValueError(  # noqa: TRY004
                f"We require a {MultipleChoiceQuestion.__name__} variant to extract"
                " ideal answer, not a string."
            )
        if isinstance(query, MultipleChoiceQuestion):
            return query.ideal_answer
        return query["ideal_answer"]

    try:
        consensus, accuracy = await bulk_evaluate_consensus(
            data=data,
            grouping_fn=extract_question,
            extract_answer_fn=extract_answer,
            ideal_answer_fn=extract_ideal,
            num_samples=num_samples,
            seed=seed,
        )
    except TypeError:
        raise ImportError(
            "Evaluating consensus requires the 'ldp' extra for 'ldp'. Please:"
            " `pip install paper-qa[ldp]`."
        ) from None
    if exclude_no_answer:
        consensus = {
            q: [(a, c) for a, c in answers if a] for q, answers in consensus.items()
        }
    return consensus, accuracy


class StoreForConsensusSamplingCallback(Callback):
    """Store environments or frames for later consensus sampling."""

    def __init__(self):
        super().__init__()
        self.stored: list[GradablePaperQAEnvironment | Frame] = []

    async def after_transition(
        self,
        traj_id: str,  # noqa: ARG002
        agent: "Agent",  # noqa: ARG002
        env: Environment,
        transition: "Transition",
    ) -> None:
        if not isinstance(env, GradablePaperQAEnvironment):
            raise NotImplementedError(
                f"So far only handled {GradablePaperQAEnvironment} in this callback,"
                f" not {type(env)}."
            )
        if transition.done and not transition.failed:  # Only store once
            return
        self.stored.append(env.export_frame())

    async def evaluate_consensus_sampling(
        self, num_samples: int = 1, seed: int | None = None
    ) -> tuple[dict[str, list[tuple[str, int]]], float]:
        return await evaluate_consensus_sampling(
            data=self.stored, num_samples=num_samples, seed=seed
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
        settings: Settings | dict | None = None,
        base_docs: Docs | dict | None = None,
        rewards: Mapping[str, float] = DEFAULT_REWARD_MAPPING,
        question_kwargs: Mapping[str, Any] | None = None,
        eval_model: LLMModel | str = DEFAULT_EVAL_MODEL_NAME,
        **env_kwargs,
    ):
        if settings is None:
            settings = Settings()
        if isinstance(settings, dict):
            settings = Settings(**settings)
        self._settings = settings
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
        ideal_answer: str,
        distractors: str | list[str],
        question: str,
        sources: str | list[str] | None = None,
    ) -> GradablePaperQAEnvironment:
        mc_question = MultipleChoiceQuestion(
            question=question,
            options=(
                distractors
                if isinstance(distractors, list)
                else MultipleChoiceQuestion.split_options(distractors)
            ),
            ideal_answer=ideal_answer,
            **(self._question_kwargs or {}),
        )
        return GradablePaperQAEnvironment(
            query=mc_question,
            settings=self._settings,
            docs=self._base_docs.model_copy(),
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
    TEST = "test"

    def get_index(self) -> int:
        """
        Get the index of the train (0), eval (1), or test (2) split.

        NOTE: the value matches the index in read_litqa_v2_from_hub's returned splits.
        """
        if self == self.TRAIN:
            return 0
        if self == self.EVAL:
            return 1
        if self == self.TEST:
            return 2
        assert_never(self)  # type: ignore[arg-type]


class LitQAv2TaskDataset(LitQATaskDataset):
    """Task dataset of LitQA v2 questions."""

    def __init__(
        self,
        *args,
        train_eval_dataset: str = DEFAULT_LABBENCH_HF_HUB_NAME,
        test_dataset: str = DEFAULT_AVIARY_PAPER_HF_HUB_NAME,
        read_data_kwargs: Mapping[str, Any] | None = None,
        split: str | LitQAv2TaskSplit = LitQAv2TaskSplit.EVAL,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        split_dfs = read_litqa_v2_from_hub(
            train_eval_dataset, test_dataset, **(read_data_kwargs or {})
        )
        self.data = split_dfs[LitQAv2TaskSplit(split).get_index()]

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
            ideal_answer=self.data.iloc[idx].ideal,
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


class LFRQAPairwiseEvalEnv(GradablePaperQAEnvironment[dict]):
    """Environment to evaluate paperqa's vs human's answers on Long Form RAG QA questions."""

    def __init__(
        self,
        *args,
        qid: str | UUID,
        question: str,
        human_answer: str,
        gt_doc_ids: list[int],
        pairwise_eval_llm: LLMModel | str = CommonLLMNames.GPT_4O.value,
        **kwargs,
    ):
        kwargs["query"] = question
        kwargs["docs"] = Docs()
        super().__init__(*args, **kwargs)

        self.qid = qid
        self.question = question
        self.human_answer = human_answer
        self.gt_doc_ids = gt_doc_ids
        self.pairwise_eval_llm = pairwise_eval_llm

    def extract_best_answer_index(self, text: str) -> int:
        match = re.search(r"<rating>(\d+)</rating>", text)
        return int(match.group(1)) if match else 0

    async def _evaluate_answer(self) -> dict:
        """Pairwise evaluation of PaperQA vs Human answer."""
        paper_search_ids = [int(doc.docname) for doc in self.state.docs.docs.values()]

        pairwise_eval_llm = LiteLLMModel(name=self.pairwise_eval_llm)
        pqa_answer = strip_citations(self.state.session.answer)
        pqa_answer_index = 1 if random.random() < 0.5 else 2  # noqa: PLR2004
        data = {
            "question": self.question,
            "answer1": pqa_answer if pqa_answer_index == 1 else self.human_answer,
            "answer2": self.human_answer if pqa_answer_index == 1 else pqa_answer,
        }

        result = await pairwise_eval_llm.run_prompt(
            prompt=lfrqa_prompt_template,
            data=data,
            system_prompt=lfrqa_system_prompt,
        )

        best_answer_index = self.extract_best_answer_index(result.text)
        if best_answer_index == pqa_answer_index:
            winner, reward = "paperqa", self._rewards["win"]
        elif best_answer_index != 0:
            winner, reward = "human", self._rewards["lose"]
        else:
            winner, reward = "tie", self._rewards["tie"]

        return {
            "llm": self._settings.llm,
            "evaluator_llm": self.pairwise_eval_llm,
            "qid": self.qid,
            "question": self.question,
            "pqa_answer": pqa_answer,
            "human_answer": self.human_answer,
            "winner": winner,
            "paper_search_ids": paper_search_ids,
            "gt_doc_ids": self.gt_doc_ids,
            "pqa_answer_was_answer_1": pqa_answer_index == 1,
            "complete_evaluator_response": result.text,
            "reward": reward,
        }

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        messages, reward, done, truncated = await super().step(action)
        if not done:
            return messages, reward, done, truncated

        evaluation = await self._evaluate_answer()
        if evaluation_callback := self._evaluation_callback:
            await evaluation_callback(evaluation)

        return messages, evaluation["reward"], done, truncated


class LFRQAQuestion(BaseModel):
    qid: str | UUID
    question: str
    answer: str
    gt_doc_ids: list[int]

    @model_validator(mode="before")
    @classmethod
    def _validate_gt_doc_ids(cls, data: dict) -> dict:
        if data.get("gold_doc_ids") and not data.get("gt_doc_ids"):
            data["gt_doc_ids"] = data["gold_doc_ids"]
        if isinstance(data["gt_doc_ids"], str):
            data["gt_doc_ids"] = data["gt_doc_ids"].strip("[]").split(",")
            data["gt_doc_ids"] = [int(_id) for _id in data["gt_doc_ids"]]
        return data


class LFRQATaskDataset(TaskDataset[LFRQAPairwiseEvalEnv]):
    """Task dataset for custom evaluation of non-multiple choice questions."""

    def __init__(
        self,
        data: list[LFRQAQuestion],
        settings: Settings | dict | None = None,
        pairwise_eval_llm: LLMModel | str = CommonLLMNames.GPT_4O.value,
        evaluation_callback: Callable[[dict], Awaitable] | None = None,
    ):
        self.data = data
        self.pairwise_eval_llm = pairwise_eval_llm

        if settings is None:
            settings = Settings()
        if isinstance(settings, dict):
            settings = Settings(**settings)
        self._settings = settings
        self._rewards = {"win": 1, "tie": 0, "lose": -1}
        self._evaluation_callback = evaluation_callback

    def get_new_env_by_idx(self, idx: int) -> LFRQAPairwiseEvalEnv:
        """Create a new environment instance for the given index."""
        question = self.data[idx]

        return LFRQAPairwiseEvalEnv(
            qid=question.qid,
            question=question.question,
            human_answer=question.answer,
            gt_doc_ids=question.gt_doc_ids,
            pairwise_eval_llm=self.pairwise_eval_llm,
            settings=self._settings,
            rewards=self._rewards,
            evaluation_callback=self._evaluation_callback,
        )

    def __len__(self) -> int:
        return len(self.data)


# Register your custom dataset
CUSTOM_TASK_DATASET_NAME = "lfrqa"
TASK_DATASET_REGISTRY[CUSTOM_TASK_DATASET_NAME] = (
    LFRQATaskDataset.__module__,
    LFRQATaskDataset.__name__,
)
