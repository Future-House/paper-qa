from typing import cast

import pytest
from aviary.utils import MultipleChoiceEvaluation, MultipleChoiceQuestion

from paperqa.litqa import make_discounted_returns, read_litqa_v2_from_hub


@pytest.mark.parametrize(
    ("evaluation", "expected_dreturns"),
    [
        (MultipleChoiceEvaluation.CORRECT, [0.25, 0.5, 1.0]),
        (MultipleChoiceEvaluation.INCORRECT, [-0.25, -0.5, -1.0]),
        (MultipleChoiceEvaluation.UNSURE, [0.025, 0.05, 0.1]),
    ],
)
def test_make_discounted_returns(
    evaluation: MultipleChoiceEvaluation, expected_dreturns: list[float]
) -> None:
    assert (
        make_discounted_returns(evaluation, num_steps=3, discount=0.5)
        == expected_dreturns
    )


def test_creating_litqa_questions() -> None:
    """Test making LitQA eval questions after downloading from Hugging Face Hub."""
    eval_split = read_litqa_v2_from_hub(seed=42)[1]
    assert len(eval_split) > 3
    assert [
        MultipleChoiceQuestion(
            question=cast(str, row.question),
            options=cast(list[str], row.distractors),
            ideal_answer=cast(str, row.ideal),
            shuffle_seed=42,
        ).question_prompt
        for row in eval_split[:3].itertuples()
    ] == [
        (
            "Q: Which of the following mutations in yeast Pbs2 increases its"
            " interaction with SH3?\n\nOptions:\nA) P97A\nB) N92S\nC) Insufficient"
            " information to answer this question\nD) K85W\nE) N92H\nF) I87W\nG) S83F"
        ),
        (
            "Q: What percentage of colorectal cancer-associated fibroblasts typically"
            " survive at 2 weeks if cultured with the platinum-based chemotherapy"
            " oxaliplatin?\n\nOptions:\nA) Insufficient information to answer this"
            " question\nB) 0%\nC) 50-80%\nD) 20-50%\nE) 1-20%\nF) 80-99%"
        ),
        (
            "Q: Which of the following genes shows the greatest difference in gene"
            " expression between homologous cell types in mouse and human"
            " brain?\n\nOptions:\nA) Htr1d\nB) Insufficient information to answer this"
            " question\nC) Htr6\nD) Htr5a\nE) Htr3a"
        ),
    ]
