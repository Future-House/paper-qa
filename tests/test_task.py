from unittest.mock import patch

import pytest
from aviary.env import TASK_DATASET_REGISTRY, TaskConfig, TaskDataset
from ldp.agent import SimpleAgent
from ldp.alg.callbacks import MeanMetricsCallback
from ldp.alg.runners import Evaluator, EvaluatorConfig

from paperqa import Docs, QueryRequest, Settings
from paperqa.agents import get_directory_index
from paperqa.agents.task import (
    GradablePaperQAEnvironment,
    LitQATaskDataset,
    LitQAv2TaskDataset,
    LitQAv2TaskSplit,
)


@pytest.fixture(name="base_query_request")
def fixture_base_query_request(agent_test_settings: Settings) -> QueryRequest:
    return QueryRequest(settings=agent_test_settings)


class StubLitQADataset(LitQATaskDataset):
    """Made up dataset of questions answerable from this repo's stub_data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: list[tuple[str, str | list[str], str]] = [
            ("Politician", ["Technologist", "Plumber"], "Who is Frederick Bates?"),
            (
                "Make molecular counterfactuals",
                [
                    "Generating images of cats",
                    "Simple explanations of internet searches",
                ],
                "How can you use XAI for chemical property prediction?",
            ),
            (
                "Maple Leaf",
                ["The Stars and Stripes", "The Blue and Yellow", "The Southern Cross"],
                "What is the national flag of Canada?",
            ),
        ]

    def get_new_env_by_idx(self, idx: int) -> GradablePaperQAEnvironment:
        return self._make_gradable_environment(
            ideal=self.data[idx][0],
            distractors=self.data[idx][1],
            question=self.data[idx][2],
        )

    def __len__(self) -> int:
        return len(self.data)


STUB_TASK_DATASET_NAME = "stub-litqa"
TASK_DATASET_REGISTRY[STUB_TASK_DATASET_NAME] = (
    StubLitQADataset.__module__,
    StubLitQADataset.__name__,
)


class TestTaskDataset:
    @pytest.mark.parametrize(
        ("split", "expected_length"),
        [(LitQAv2TaskSplit.TRAIN, 159), (LitQAv2TaskSplit.EVAL, 40)],
    )
    def test___len__(
        self,
        split: str | LitQAv2TaskSplit,
        expected_length: int,
        base_query_request: QueryRequest,
    ) -> None:
        task_dataset = LitQAv2TaskDataset(base_query=base_query_request, split=split)
        assert len(task_dataset) == expected_length

    @pytest.mark.asyncio
    async def test_evaluation(self, base_query_request: QueryRequest) -> None:
        await get_directory_index(settings=base_query_request.settings)  # Build
        docs = Docs()
        # Why are we constructing a TaskConfig here using a serialized QueryRequest and
        # Docs? It's to confirm everything works as if hydrating from a YAML config file
        task_config = TaskConfig(
            name=STUB_TASK_DATASET_NAME,
            eval_kwargs={
                "base_query": base_query_request.model_dump(
                    exclude={"id", "settings", "docs_name"}
                ),
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
        dataset = task_config.make_dataset(split="eval")  # noqa: FURB184
        metrics_callback = MeanMetricsCallback(eval_dataset=dataset)

        evaluator = Evaluator(
            config=EvaluatorConfig(batch_size=3, max_rollout_steps=10),
            agent=SimpleAgent(),
            dataset=dataset,
            callbacks=[metrics_callback],
        )
        await evaluator.evaluate()

        assert (
            not base_query_request.query
        ), "Should not have mutated query in base request"
        assert not docs.docs, "Should not have mutated docs in base docs"
        assert (
            isinstance(metrics_callback.eval_means["total_paper_count"], float) > 0
        ), "Expected some papers to help us answer questions"
        assert (
            isinstance(metrics_callback.eval_means["reward"], float) > 0
        ), "Expected some wins"

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
