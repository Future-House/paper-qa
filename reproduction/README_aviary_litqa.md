# aviary.litqa

LitQA2 environment implemented with aviary,
allowing agents to perform question answering on the LitQA dataset.

[LitQA](https://github.com/Future-House/LitQA) (now legacy) is a dataset composed from 50 multiple-choice questions from recent literature.
It is designed to test the LLM's the ability to retrieve information outside of the pre-training corpus.
To ensure the questions are not in the pre-training corpus, the questions were collected
from scientific papers published after September 2021 -- cut-off date of GPT-4's training data.

LitQA2 is part of the [LAB-Bench dataset](https://arxiv.org/abs/2407.10362).
LitQA2 contains 248 multiple-choice questions from the literature and was created ensuring that the questions cannot be answered by recalling from the pre-training corpus only.
It considered scientific paper published within 36 months from the data of its publication.
Therefore, LitQA2 is considered a scientific RAG dataset.

## Installation

To install the LitQA environment, run:

```bash
pip install 'fhaviary[litqa]'
```

## Usage

In [`litqa/env.py`](litqa/env.py), you will find:

`GradablePaperQAEnvironment`: an environment that can grade answers given an evaluation function.

And in [`litqa/task.py`](litqa/task.py), you will find:

`LitQAv2TaskDataset`: a task dataset designed to pull LitQA v2 from Hugging Face,
and create one `GradablePaperQAEnvironment` per question

Here is an example of how to use them:

```python
import os

from ldp.agent import SimpleAgent
from ldp.alg import Evaluator, EvaluatorConfig, MeanMetricsCallback
from paperqa import Settings

from aviary.env import TaskDataset
from aviary.envs.litqa.task import TASK_DATASET_NAME


async def evaluate(folder_of_litqa_v2_papers: str | os.PathLike) -> None:
    settings = Settings(paper_directory=folder_of_litqa_v2_papers)
    dataset = TaskDataset.from_name(TASK_DATASET_NAME, settings=settings)
    metrics_callback = MeanMetricsCallback(eval_dataset=dataset)

    evaluator = Evaluator(
        config=EvaluatorConfig(batch_size=3),
        agent=SimpleAgent(),
        dataset=dataset,
        callbacks=[metrics_callback],
    )
    await evaluator.evaluate()

    print(metrics_callback.eval_means)
```

## References

[1] Lála et al. [PaperQA: Retrieval-Augmented Generative Agent for Scientific Research](https://arxiv.org/abs/2312.07559). ArXiv:2312.07559, 2023.

[2] Skarlinski et al. [Language agents achieve superhuman synthesis of scientific knowledge](https://arxiv.org/abs/2409.13740). ArXiv:2409.13740, 2024.

[3] Laurent et al. [LAB-Bench: Measuring Capabilities of Language Models for Biology Research](https://arxiv.org/abs/2407.10362). ArXiv:2407.10362, 2024.
