"""Utilities for working with LitQA v1 and v2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from aviary.utils import MultipleChoiceEvaluation

from paperqa._ldp_shims import discounted_returns

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_REWARD_MAPPING = {"correct": 1.0, "unsure": 0.1, "incorrect": -1.0}


def make_discounted_returns(
    evaluation: MultipleChoiceEvaluation,
    num_steps: int,
    rewards: Mapping[str, float] = DEFAULT_REWARD_MAPPING,
    discount: float = 1.0,
) -> list[float]:
    try:
        return discounted_returns(
            # paper-qa has no intermediary rewards
            [0] * (num_steps - 1) + [rewards[evaluation.value]],
            terminated=[False] * (num_steps - 1) + [True],
            discount=discount,
        )
    except TypeError as exc:
        raise ImportError(
            "Making discounted returns requires the 'ldp' extra for 'ldp'. Please:"
            " `pip install paper-qa[ldp]`."
        ) from exc


DEFAULT_LABBENCH_HF_HUB_NAME = "futurehouse/lab-bench"
# Test split from Aviary paper's section 4.3: https://doi.org/10.48550/arXiv.2412.21154
DEFAULT_AVIARY_PAPER_HF_HUB_NAME = "futurehouse/aviary-paper-data"


def read_litqa_v2_from_hub(
    train_eval_dataset: str = DEFAULT_LABBENCH_HF_HUB_NAME,
    test_dataset: str = DEFAULT_AVIARY_PAPER_HF_HUB_NAME,
    randomize: bool = True,
    seed: int | None = None,
    train_eval_split: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read LitQA v2 JSONL into train, eval, and test DataFrames.

    Args:
        train_eval_dataset: Hugging Face Hub dataset's name corresponding with train
            and eval splits.
        test_dataset: Hugging Face Hub dataset's name corresponding with a test split.
        randomize: Opt-out flag to shuffle the dataset after loading in by question.
        seed: Random seed to use for the shuffling.
        train_eval_split: Train/eval split fraction, default is 80% train 20% eval.

    Raises:
        DatasetNotFoundError: If any of the datasets are not found, or the
            user is unauthenticated.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Reading in LitQA2 requires the 'datasets' extra for 'datasets'. Please:"
            " `pip install paper-qa[datasets]`."
        ) from exc

    train_eval = load_dataset(train_eval_dataset, "LitQA2")["train"].to_pandas()
    test = load_dataset(test_dataset, "LitQA2")["test"].to_pandas()
    # Convert to list so it's not unexpectedly a numpy array
    train_eval["distractors"] = train_eval["distractors"].apply(list)
    test["distractors"] = test["distractors"].apply(list)
    # Let downstream usage in the TaskDataset's environment factories check for the
    # presence of other DataFrame columns
    if randomize:
        train_eval = train_eval.sample(frac=1, random_state=seed)
        test = test.sample(frac=1, random_state=seed)
    num_train = int(len(train_eval) * train_eval_split)
    return train_eval[:num_train], train_eval[num_train:], test
