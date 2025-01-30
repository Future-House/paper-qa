import os
import asyncio
from aviary.env import TaskDataset
from ldp.agent import SimpleAgent
from ldp.alg.callbacks import MeanMetricsCallback
from ldp.alg.runners import Evaluator, EvaluatorConfig

from paperqa import Settings
from paperqa.agents.task import LFRQATaskDataset

async def evaluate() -> None:
    settings = Settings()
    settings.agent.index.paper_directory = (
        "/Users/joaquinpolonuer/Documents/rag-qa-arena/lfrqa/science_docs_for_paperqa"
    )
    settings.agent.index.index_directory = (
        "/Users/joaquinpolonuer/Documents/rag-qa-arena/lfrqa/science_docs_for_paperqa_index"
    )
    settings.agent.index.manifest_file = "manifest.csv"
    settings.agent.agent_llm = "gpt-4o"
    # settings.agent.search_count = 15

    settings.parsing.use_doc_details = False

    dataset = LFRQATaskDataset(data_path="/Users/joaquinpolonuer/Documents/rag-qa-arena/lfrqa/questions.csv", settings=settings)
    metrics_callback = MeanMetricsCallback(eval_dataset=dataset)

    evaluator = Evaluator(
        config=EvaluatorConfig(batch_size=3),
        agent=SimpleAgent(),
        dataset=dataset,
        callbacks=[metrics_callback],
    )
    await evaluator.evaluate()

    print(metrics_callback.eval_means)


if __name__ == "__main__":
    asyncio.run(evaluate())
