import asyncio
from collections.abc import Iterable
from copy import deepcopy
from unittest.mock import patch

import pytest
from aviary.env import TASK_DATASET_REGISTRY, TaskConfig, TaskDataset
from ldp.agent import SimpleAgent
from ldp.alg.callbacks import Callback, MeanMetricsCallback, StoreTrajectoriesCallback
from ldp.alg.runners import Evaluator, EvaluatorConfig
from pytest_subtests import SubTests

from paperqa import Docs, QueryRequest, Settings
from paperqa.agents import get_directory_index
from paperqa.agents.env import PaperQAEnvironment
from paperqa.agents.task import (
    GradablePaperQAEnvironment,
    LitQATaskDataset,
    LitQAv2TaskDataset,
    LitQAv2TaskSplit,
)
from paperqa.agents.tools import GenerateAnswer
from paperqa.litqa import DEFAULT_REWARD_MAPPING, LitQAEvaluation


@pytest.fixture(name="base_query_request")
def fixture_base_query_request(agent_test_settings: Settings) -> QueryRequest:
    agent_test_settings.agent.index.manifest_file = "stub_manifest.csv"
    return QueryRequest(settings=agent_test_settings)


class StubLitQADataset(LitQATaskDataset):
    """Made up dataset of questions answerable from this repo's stub_data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: list[tuple[str, str | list[str], str, str]] = [
            (
                "Politician",
                ["Technologist", "Plumber"],
                "Who is Frederick Bates?",
                "bates.txt",
            ),
            (
                "Make molecular counterfactuals",
                [
                    "Generating images of cats",
                    "Simple explanations of internet searches",
                ],
                "How can you use XAI for chemical property prediction?",
                "paper.pdf",
            ),
            (
                "Maple Leaf",
                ["The Stars and Stripes", "The Blue and Yellow", "The Southern Cross"],
                "What is the national flag of Canada?",
                "flag_day.html",
            ),
        ]

    def get_new_env_by_idx(self, idx: int) -> GradablePaperQAEnvironment:
        return self._make_gradable_environment(
            ideal=self.data[idx][0],
            distractors=self.data[idx][1],
            question=self.data[idx][2],
            sources=self.data[idx][3],
        )

    def __len__(self) -> int:
        return len(self.data)


STUB_TASK_DATASET_NAME = "stub-litqa"
TASK_DATASET_REGISTRY[STUB_TASK_DATASET_NAME] = (
    StubLitQADataset.__module__,
    StubLitQADataset.__name__,
)


class StoreEnvCallback(Callback):
    """Test utility to store instantiated environments."""

    def __init__(self):
        super().__init__()
        # NOTE: using question-to-env because too lazy to implement __hash__
        # for this being a set of envs
        self.query_to_envs: dict[str, PaperQAEnvironment] = {}

    async def before_transition(
        self, traj_id, agent, env, agent_state, obs  # noqa: ARG002
    ) -> None:
        if not isinstance(env, PaperQAEnvironment):
            raise NotImplementedError(
                f"Didn't yet handle environment type {type(env).__name__}."
            )
        question = env._query.query
        if env not in self.query_to_envs:
            self.query_to_envs[question] = env


class TestTaskDataset:

    @pytest.mark.parametrize(
        ("split", "expected_length"),
        [(LitQAv2TaskSplit.TRAIN, 159), (LitQAv2TaskSplit.EVAL, 40)],
    )
    @pytest.mark.asyncio
    async def test___len__(
        self,
        split: str | LitQAv2TaskSplit,
        expected_length: int,
        base_query_request: QueryRequest,
    ) -> None:
        task_dataset = LitQAv2TaskDataset(base_query=base_query_request, split=split)
        assert len(task_dataset) == expected_length

        # Now let's check we could use the sources in a validation
        for i in range(len(task_dataset)):
            env = task_dataset.get_new_env_by_idx(i)
            assert env.sources, "Sources need to be accessible"
            assert isinstance(
                env.sources, Iterable
            ), "Sources need to be at least iterable"

    @pytest.mark.asyncio
    async def test_can_validate_stub_dataset_sources(
        self, base_query_request: QueryRequest
    ) -> None:
        ds = StubLitQADataset(base_query=base_query_request)
        await asyncio.gather(
            *(ds.get_new_env_by_idx(i).validate_sources() for i in range(len(ds)))
        )

    @pytest.mark.asyncio
    async def test_evaluation(
        self, subtests: SubTests, base_query_request: QueryRequest
    ) -> None:
        await get_directory_index(settings=base_query_request.settings)  # Build
        docs = Docs()
        raw_docs_deepcopy = deepcopy(docs)  # Preserve for later assertions
        # Why are we constructing a TaskConfig here using a serialized QueryRequest and
        # Docs? It's to confirm everything works as if hydrating from a YAML config file
        task_config = TaskConfig(
            name=STUB_TASK_DATASET_NAME,
            eval_kwargs={
                "base_docs": docs.model_dump(
                    exclude={
                        "id",
                        "docnames",
                        "texts_index",
                        "index_path",
                        "deleted_dockeys",
                    }
                ),
            },
        )
        # NOTE: set base_query after construction of the TaskConfig. because in
        # aviary 0.10 the TaskConfig Pydnatic model has types `BaseModel | JsonValue`,
        # which lead to base_query being cast into a BaseModel. This is probably a bug
        # in aviary, but for now let's just assign it after TaskConfig construction
        task_config.eval_kwargs["base_query"] = base_query_request.model_dump(
            exclude={"id", "docs_name"}
        )
        dataset = task_config.make_dataset(split="eval")  # noqa: FURB184
        assert isinstance(dataset, StubLitQADataset), "Test assertions depend on this"
        metrics_callback = MeanMetricsCallback(eval_dataset=dataset)
        store_env_callback = StoreEnvCallback()

        evaluator = Evaluator(
            config=EvaluatorConfig(batch_size=len(dataset.data), max_rollout_steps=10),
            agent=SimpleAgent(),
            dataset=dataset,
            callbacks=[metrics_callback, store_env_callback],
        )
        await evaluator.evaluate()

        assert (
            not base_query_request.query
        ), "Should not have mutated query in base request"
        assert not docs.docs, "Should not have mutated docs in base docs"
        assert (
            metrics_callback.eval_means["total_paper_count"] > 0
        ), "Expected some papers to help us answer questions"
        correct_percentage = metrics_callback.eval_means["correct"]
        assert metrics_callback.eval_means["reward"] > 0, "Expected some wins"
        correct_reward, incorrect_reward = (
            DEFAULT_REWARD_MAPPING[evaluation.value]
            for evaluation in (LitQAEvaluation.CORRECT, LitQAEvaluation.INCORRECT)
        )
        worst_case_reward_given_correct = (
            correct_reward * correct_percentage
            + incorrect_reward * (1 - correct_percentage)
        )
        assert (
            metrics_callback.eval_means["reward"] >= worst_case_reward_given_correct
        ), "Expected reward to be above worst case value"

        with subtests.test(msg="confirming-reset-works"):
            assert len(store_env_callback.query_to_envs) == len(dataset)
            for env in store_env_callback.query_to_envs.values():
                await env.reset()
                assert env.state.docs == raw_docs_deepcopy

        with subtests.test(msg="zero-shot"):
            # Confirm we can just directly call gen_answer
            base_query_request.settings.agent.tool_names = {
                GenerateAnswer.gen_answer.__name__
            }
            base_query_request.settings.answer.max_answer_attempts = 2
            base_query_request.settings.answer.get_evidence_if_no_contexts = False
            dataset = LitQAv2TaskDataset(base_query=base_query_request)
            dataset.data = dataset.data[:2]  # Save the world: just use two questions
            storage_callback = StoreTrajectoriesCallback()
            evaluator = Evaluator(
                config=EvaluatorConfig(batch_size=len(dataset), max_rollout_steps=4),
                agent=SimpleAgent(),
                dataset=dataset,
                callbacks=[storage_callback],
            )
            await evaluator.evaluate()
            for traj in storage_callback.eval_trajectories:
                assert not traj.failed
                assert traj.done
                for step in traj.steps:
                    assert all(
                        tc.function.name == GenerateAnswer.gen_answer.__name__
                        for tc in step.action.value.tool_calls
                    )

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_tool_failure(self, base_query_request: QueryRequest) -> None:
        docs = Docs()
        dataset = TaskDataset.from_name(
            STUB_TASK_DATASET_NAME, base_query=base_query_request, base_docs=docs
        )
        metrics_callback = MeanMetricsCallback(eval_dataset=dataset)

        evaluator = Evaluator(
            config=EvaluatorConfig(
                batch_size=1, num_eval_iterations=1, max_rollout_steps=2
            ),
            agent=SimpleAgent(),
            dataset=dataset,
            callbacks=[metrics_callback],
        )
        with patch(
            "paperqa.agents.search.SearchIndex",
            side_effect=Exception("Totally unexpected but retryable error."),
        ) as mock_SearchIndex:
            await evaluator.evaluate()  # Confirm this does not crash
        assert (
            metrics_callback.eval_means["truncation_rate"] == 1.0
        ), "Expected 100% truncations due to max_rollout_steps"
        mock_SearchIndex.assert_called(), "Expected failures to come from unit test"
        assert metrics_callback.eval_means["correct"] == 0.0
        assert metrics_callback.eval_means["correct_unsure"] == 0.0
