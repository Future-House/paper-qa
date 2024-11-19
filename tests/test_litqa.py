from collections.abc import Sequence
from typing import cast

import pytest

from paperqa.litqa import LitQAEvaluation, read_litqa_v2_from_hub
from tests.conftest import VCR_DEFAULT_MATCH_ON


class TestLitQAEvaluation:
    @staticmethod
    def _assert_prompt_is_valid(
        qa_prompt: str, question: str, ideal: str, distractors: Sequence[str]
    ) -> None:
        for substr in (question, "Insufficient information", ideal, *distractors):
            assert qa_prompt.count(substr) == 1

    # Use for general purpose testing
    ZIP_CODE_QUESTION_IDEAL_DISTRACTORS = (
        "What is my office's zip code?",
        "94107",
        ["-8", "94106", "cheesecake"],
    )
    # Use to check we don't leak on the LLM's innate knowledge
    MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS = (
        "What is the meaning of life?",
        "42",
        ["-84", "11", "cheesecake"],
    )

    @pytest.mark.asyncio
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        (
            "question",
            "ideal",
            "distractors",
            "answer",
            "expected_eval",
            "expected_dreturns",
        ),
        [
            (
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 94107",
                LitQAEvaluation.CORRECT,
                [0.25, 0.5, 1.0],
            ),
            (
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 14004",
                LitQAEvaluation.INCORRECT,
                [-0.25, -0.5, -1.0],
            ),
            (
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 94106",
                LitQAEvaluation.INCORRECT,
                [-0.25, -0.5, -1.0],
            ),
            (
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "Insufficient information",
                LitQAEvaluation.UNSURE,
                [0.025, 0.05, 0.1],
            ),
            (
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 94106 or 94107",
                LitQAEvaluation.INCORRECT,
                [-0.25, -0.5, -1.0],
            ),
            (
                *MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS,
                "14",
                LitQAEvaluation.INCORRECT,
                [-0.25, -0.5, -1.0],
            ),
        ],
    )
    async def test_from_question(
        self,
        question: str,
        ideal: str,
        distractors: str | list[str],
        answer: str,
        expected_eval: LitQAEvaluation,
        expected_dreturns: list[float],
    ) -> None:
        """Tests that we can create a LitQA question and evaluate answers."""
        qa_prompt, eval_fn = LitQAEvaluation.from_question(
            ideal=ideal,
            distractors=distractors,
            question=question,
            seed=42,  # Seed for VCR cassette
        )
        self._assert_prompt_is_valid(qa_prompt, question, ideal, distractors)

        evaluation = await eval_fn(answer)
        assert evaluation == expected_eval
        assert evaluation.make_discounted_returns(3, discount=0.5) == expected_dreturns

    def test_consistent_mc_options(self) -> None:
        """Tests that creating multiple evaluations with the same seed results in the same prompt."""
        question, ideal, distractors = self.MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS

        qa_prompt_1, _ = LitQAEvaluation.from_question(
            ideal=ideal, distractors=distractors, question=question, seed=0
        )
        self._assert_prompt_is_valid(qa_prompt_1, question, ideal, distractors)

        qa_prompt_2, _ = LitQAEvaluation.from_question(
            ideal=ideal, distractors=distractors, question=question, seed=0
        )
        self._assert_prompt_is_valid(qa_prompt_1, question, ideal, distractors)
        assert qa_prompt_1 == qa_prompt_2

    def test_creating_litqa_questions(self) -> None:
        """Test making LitQA eval questions after downloading from Hugging Face Hub."""
        _, eval_split = read_litqa_v2_from_hub(seed=42)
        assert len(eval_split) > 3
        assert [
            LitQAEvaluation.from_question(
                ideal=cast(str, row.ideal),
                distractors=cast(list[str], row.distractors),
                question=cast(str, row.question),
                seed=42,
            )[0]
            for row in eval_split[:3].itertuples()
        ] == [
            (
                "Q: Which of the following mutations in yeast Pbs2 increases its"
                " interaction with SH3?\n\nOptions:\nA) S83F\nB) I87W\nC) N92H\nD) K85W\nE)"
                " Insufficient information to answer this question\nF) N92S\nG) P97A"
            ),
            (
                "Q: What percentage of colorectal cancer-associated fibroblasts typically"
                " survive at 2 weeks if cultured with the platinum-based chemotherapy"
                " oxaliplatin?\n\nOptions:\nA) 80-99%\nB) 1-20%\nC) 20-50%\nD) 50-80%\nE)"
                " 0%\nF) Insufficient information to answer this question"
            ),
            (
                "Q: Which of the following genes shows the greatest difference in gene"
                " expression between homologous cell types in mouse and human"
                " brain?\n\nOptions:\nA) Htr3a\nB) Htr5a\nC) Htr6\nD) Insufficient"
                " information to answer this question\nE) Htr1d"
            ),
        ]
