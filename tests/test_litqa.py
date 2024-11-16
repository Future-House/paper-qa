from typing import cast

import pytest

from paperqa.litqa import LitQAEvaluation, read_litqa_v2_from_hub
from tests.conftest import VCR_DEFAULT_MATCH_ON


class TestLitQAEvaluation:
    @staticmethod
    def _assert_prompt_is_valid(
        qa_prompt: str, question: str, ideal: str, distractors: list[str]
    ):
        for substr in (question, "Insufficient information", ideal, *distractors):
            assert qa_prompt.count(substr) == 1

    @pytest.mark.asyncio
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    async def test_from_question(self) -> None:
        """Tests that we can create a LitQA question and evaluate answers."""
        question = "What is my office's zip code?"
        ideal = "94107"
        distractors = ["-8", "94106", "cheesecake"]

        qa_prompt, eval_fn = LitQAEvaluation.from_question(
            ideal=ideal,
            distractors=distractors,
            question=question,
            seed=42,  # Seed for VCR cassette
        )
        self._assert_prompt_is_valid(qa_prompt, question, ideal, distractors)

        for answer, expected in (
            ("the answer is 94107", LitQAEvaluation.CORRECT),
            # NOTE: The below case fails this test, because the LM doesn't accept an answer not in the options.
            # See https://github.com/Future-House/paper-qa/issues/693
            # ("the answer is 14004", LitQAEvaluation.INCORRECT),
            ("the answer is 94106", LitQAEvaluation.INCORRECT),
            ("Insufficient information to answer", LitQAEvaluation.UNSURE),
        ):
            result = await eval_fn(answer)
            assert result == expected

    def test_consistent_mc_options(self) -> None:
        """Tests that creating multiple evaluations with the same seed results in the same prompt."""
        question = "What is the meaning of life?"
        ideal = "42"
        distractors = ["-84", "11", "cheesecake"]

        qa_prompt_1, _ = LitQAEvaluation.from_question(
            ideal=ideal, distractors=distractors, question=question, seed=0
        )
        self._assert_prompt_is_valid(qa_prompt_1, question, ideal, distractors)

        qa_prompt_2, _ = LitQAEvaluation.from_question(
            ideal=ideal, distractors=distractors, question=question, seed=0
        )
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
