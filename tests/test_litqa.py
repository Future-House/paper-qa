from collections.abc import Sequence
from typing import cast

import pytest

from paperqa.litqa import SEED_USING_QUESTION, LitQAEvaluation, read_litqa_v2_from_hub
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
    # The following two are used to check we don't leak on the LLM's innate knowledge
    MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS = (
        "What is the meaning of life?",
        "42",
        ["-84", "11", "cheesecake"],
    )
    # Source: https://github.com/Future-House/LAB-Bench/blob/43b2045c67a2da12c233689cf538f1ed5c42f590/LitQA2/litqa-v2-public.jsonl#L130
    LITQA2_QUESTION_IDEAL_DISTRACTORS = (
        (
            "What method was used to demonstrate that the enzyme PafA is stable after"
            " incubation with 4M urea for 14 days?"
        ),
        "circular dichroism",
        ["cryo EM", "x-ray crystallography", "NMR"],
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
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 94107",
                LitQAEvaluation.CORRECT,
                [0.25, 0.5, 1.0],
                id="matched-correct-option",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 14004",
                LitQAEvaluation.INCORRECT,
                [-0.25, -0.5, -1.0],
                id="didnt-match-and-no-llm-innate-knowledge",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 94106",
                LitQAEvaluation.INCORRECT,
                [-0.25, -0.5, -1.0],
                id="matched-incorrect-option",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "Insufficient information",
                LitQAEvaluation.UNSURE,
                [0.025, 0.05, 0.1],
                id="matched-unsure-option",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 94106 or 94107",
                LitQAEvaluation.INCORRECT,
                [-0.25, -0.5, -1.0],
                id="matched-several-options",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "",
                LitQAEvaluation.INCORRECT,
                [-0.25, -0.5, -1.0],
                id="empty-answer1",
            ),
            pytest.param(
                *MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS,
                "14",
                LitQAEvaluation.INCORRECT,
                [-0.25, -0.5, -1.0],
                id="didnt-match-and-llm-has-innate-knowledge",
            ),
            pytest.param(
                *MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS,
                "",
                LitQAEvaluation.INCORRECT,
                [-0.25, -0.5, -1.0],
                id="empty-answer2",
            ),
            pytest.param(
                *LITQA2_QUESTION_IDEAL_DISTRACTORS,
                "",
                LitQAEvaluation.INCORRECT,
                [-0.25, -0.5, -1.0],
                id="empty-answer3",
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

        qa_prompt_1a, _ = LitQAEvaluation.from_question(
            ideal=ideal, distractors=distractors, question=question, seed=0
        )
        self._assert_prompt_is_valid(qa_prompt_1a, question, ideal, distractors)

        qa_prompt_1b, _ = LitQAEvaluation.from_question(
            ideal=ideal, distractors=distractors, question=question, seed=0
        )
        self._assert_prompt_is_valid(qa_prompt_1b, question, ideal, distractors)
        assert qa_prompt_1a == qa_prompt_1b, "Same seeding should lead to same prompts"

        qa_prompt_2a, _ = LitQAEvaluation.from_question(
            ideal=ideal,
            distractors=distractors,
            question=question,
            seed=SEED_USING_QUESTION,
        )
        self._assert_prompt_is_valid(qa_prompt_2a, question, ideal, distractors)

        qa_prompt_2b, _ = LitQAEvaluation.from_question(
            ideal=ideal,
            distractors=distractors,
            question=question,
            seed=SEED_USING_QUESTION,
        )
        self._assert_prompt_is_valid(qa_prompt_2b, question, ideal, distractors)
        assert (
            qa_prompt_2a == qa_prompt_2b
        ), "Same seeding strategy should lead to same prompts"
        assert (
            qa_prompt_2a != qa_prompt_1a
        ), "Different seeding strategies should lead to different prompts"

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

    @pytest.mark.parametrize(
        ("evals", "accuracy_precision"),
        [
            (
                [
                    LitQAEvaluation.CORRECT,
                    LitQAEvaluation.CORRECT,
                    LitQAEvaluation.CORRECT,
                ],
                (1, 1),
            ),
            (["correct", "correct", "unsure"], (2 / 3, 1)),
            (
                [LitQAEvaluation.CORRECT, LitQAEvaluation.UNSURE, "incorrect"],
                (1 / 3, 1 / 2),
            ),
        ],
    )
    def test_calculate_accuracy_precision(
        self, evals: Sequence[LitQAEvaluation], accuracy_precision: tuple[float, float]
    ) -> None:
        assert LitQAEvaluation.calculate_accuracy_precision(evals) == accuracy_precision
