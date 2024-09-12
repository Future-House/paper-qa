"""Utilities for working with LitQA v1 and v2."""

from __future__ import annotations

import random
import re
from ast import literal_eval
from collections.abc import Awaitable, Callable, Sequence
from enum import IntEnum
from typing import TYPE_CHECKING

try:
    from ldp.alg.algorithms import discounted_returns
except ImportError:
    discounted_returns = None  # type: ignore[assignment]

from paperqa.llms import LiteLLMModel, LLMModel
from paperqa.prompts import EVAL_PROMPT_TEMPLATE, QA_PROMPT_TEMPLATE
from paperqa.settings import make_default_litellm_router_settings
from paperqa.types import Answer

if TYPE_CHECKING:
    import pandas as pd

# Special case for LitQA, when ideal == "null"
UNSURE_OPTION = "Insufficient information to answer this question"
_CAPITAL_A_INDEX = ord("A")


def make_mc_options(
    ideal: str,
    distractors: str | list[str],
    unsure_option: str | None = UNSURE_OPTION,
    seed: int | None = None,
) -> tuple[str, str, str | None]:
    """Return string of options (as letters) and correct answer."""
    if isinstance(distractors, str):
        try:
            split_distractors = literal_eval(distractors)
            if not isinstance(split_distractors, list):
                raise TypeError("Need split_distractors to be a list.")  # noqa: TRY301
        except (ValueError, SyntaxError, TypeError):
            split_distractors = [d.strip("'[ ]\"") for d in distractors.split(",")]
        options: list[str] = split_distractors
    else:
        options = distractors

    if ideal == "null":
        if not unsure_option:
            raise ValueError(
                'Dataset configured for "unsure" options via '
                'ideal="null", please specify "unsure_option".'
            )
        correct_answer = unsure_option
    else:
        # add the answer to the options, only if not null
        options.append(ideal)
        correct_answer = ideal

    if unsure_option:
        options.append(unsure_option)

    random.Random(seed).shuffle(options)
    return (
        "\n".join([f"{_CAPITAL_A_INDEX + i:c}) {o}" for i, o in enumerate(options)]),
        chr(_CAPITAL_A_INDEX + options.index(correct_answer)),
        chr(_CAPITAL_A_INDEX + options.index(unsure_option)) if unsure_option else None,
    )


DEFAULT_EVAL_MODEL_NAME = "gpt-4-turbo-2024-04-09"
DEFAULT_REWARD_DISTRIBUTION: tuple[float, float, float] = 1.0, 0.1, -1.0


class LitQAEvaluation(IntEnum):
    """Possible evaluation results for a LitQA question."""

    CORRECT = 0
    INCORRECT = 1
    UNSURE = 2

    def make_discounted_returns(
        self,
        num_steps: int,
        rewards: Sequence[float] = DEFAULT_REWARD_DISTRIBUTION,
        discount: float = 1.0,
    ) -> list[float]:
        try:
            return discounted_returns(
                [i * rewards[self.value] for i in range(num_steps, 0, -1)],
                terminated=[False] * (num_steps - 1) + [True],
                discount=discount,
            )
        except TypeError as exc:
            raise ImportError(
                "Making discounted returns requires the 'ldp' extra for 'ldp'. Please:"
                " `pip install paper-qa[ldp]`."
            ) from exc

    @classmethod
    def from_answer(
        cls, text: str, ideal_mc_answer: str, unsure_mc_answer: str | None = None
    ) -> LitQAEvaluation:
        """Compare text with a multiple choice answer or optionally an unsure answer."""

        def extract_answer(answer: str) -> str:
            # first capital letter, like A or A)
            s = re.search(r"([A-Z])\)?", answer, re.DOTALL)
            if s is not None:
                return s.group(1)
            return answer.split()[0][0].upper()

        result = extract_answer(text)
        evaluation_result = cls.INCORRECT
        if unsure_mc_answer and result[0].lower() == unsure_mc_answer[0].lower():
            evaluation_result = cls.UNSURE
        if result[0].lower() == ideal_mc_answer[0].lower():
            evaluation_result = cls.CORRECT
        return evaluation_result

    @classmethod
    def from_question(
        cls,
        ideal: str,
        distractors: str | list[str],
        question: str,
        use_unsure: bool = True,
        eval_model: LLMModel | str = DEFAULT_EVAL_MODEL_NAME,
        seed: int | None = None,
    ) -> tuple[str, Callable[[Answer | str], Awaitable[LitQAEvaluation]]]:
        """Create a LitQA question and an evaluation callback."""
        text, ideal_answer, unsure_answer = make_mc_options(
            ideal=ideal,
            distractors=distractors,
            seed=seed,
            **({} if use_unsure else {"unsure_option": None}),
        )
        qa_prompt = QA_PROMPT_TEMPLATE.format(question=question, options=text)

        if isinstance(eval_model, str):
            eval_model = LiteLLMModel(
                name=eval_model, config=make_default_litellm_router_settings(eval_model)
            )

        async def llm_from_answer(answer: Answer | str) -> LitQAEvaluation:
            if isinstance(answer, Answer):
                answer = answer.answer
            eval_chunk = await eval_model.achat(
                messages=[
                    {
                        "role": "user",
                        "content": EVAL_PROMPT_TEMPLATE.format(
                            qa_prompt=qa_prompt, qa_answer=answer
                        ),
                    }
                ],
            )
            if not isinstance(eval_chunk.text, str):
                raise NotImplementedError(
                    f"Expected evaluation chunk to be a string, not {eval_chunk.text}."
                )
            return cls.from_answer(
                text=eval_chunk.text,
                ideal_mc_answer=ideal_answer,
                unsure_mc_answer=unsure_answer,
            )

        return qa_prompt, llm_from_answer


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
