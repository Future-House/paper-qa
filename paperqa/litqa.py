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


def read_litqa_v2_from_hub(
    labbench_dataset: str = DEFAULT_LABBENCH_HF_HUB_NAME,
    randomize: bool = True,
    seed: int | None = None,
    train_eval_split: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read LitQA v2 JSONL into train and eval DataFrames.

    Args:
        labbench_dataset: The Hugging Face Hub dataset's name corresponding with the
            LAB-Bench dataset.
        randomize: Opt-out flag to shuffle the dataset after loading in by question.
        seed: Random seed to use for the shuffling.
        train_eval_split: Train/eval split fraction, default is 80% train 20% eval.

    Raises:
        DatasetNotFoundError: If the LAB-Bench dataset is not found, or the
            user is unauthenticated.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Reading in LitQA2 requires the 'datasets' extra for 'datasets'. Please:"
            " `pip install paper-qa[datasets]`."
        ) from exc

    litqa_v2 = load_dataset(labbench_dataset, "LitQA2")["train"].to_pandas()
    litqa_v2["distractors"] = litqa_v2["distractors"].apply(list)
    if randomize:
        litqa_v2 = litqa_v2.sample(frac=1, random_state=seed)
    num_train = int(len(litqa_v2) * train_eval_split)
    return litqa_v2[:num_train], litqa_v2[num_train:]
