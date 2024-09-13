from typing import cast

from paperqa.litqa import LitQAEvaluation, read_litqa_v2_from_hub


def test_creating_litqa_questions() -> None:
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
