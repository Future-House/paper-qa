from collections.abc import Collection

import pytest
from llmclient import MultipleCompletionLLMModel as LLMModel

from paperqa.binning import BinningFunction, llm_categorize_then_bin, llm_tournament_bin


@pytest.mark.parametrize(
    ("texts", "expected_bins"),
    [
        pytest.param(
            {
                "I love playing basketball",
                "Soccer is a popular sport worldwide",
                "Tennis requires great skill",
                "Quantum physics explores the behavior of particles",
                "Biology is the study of living organisms",
                "Astronomy is the study of celestial objects",
                "Paris is a beautiful city",
                "I visited New York last summer",
                "Tokyo is famous for its cherry blossoms",
            },
            {
                frozenset(
                    {
                        "Astronomy is the study of celestial objects",
                        "Quantum physics explores the behavior of particles",
                        "Biology is the study of living organisms",
                    }
                ),
                frozenset(
                    {
                        "Tokyo is famous for its cherry blossoms",
                        "I visited New York last summer",
                        "Paris is a beautiful city",
                    }
                ),
                frozenset(
                    {
                        "Soccer is a popular sport worldwide",
                        "Tennis requires great skill",
                        "I love playing basketball",
                    }
                ),
            },
            id="sports-sciences-places",
        ),
        pytest.param(
            {
                "Who discovered gravity?",
                "Python is great for data analysis",
                "The cat sat on the mat",
            },
            {
                frozenset({"Who discovered gravity?"}),
                frozenset({"Python is great for data analysis"}),
                frozenset({"The cat sat on the mat"}),
            },
            id="separate-categories",
        ),
    ],
)
@pytest.mark.parametrize(
    "binning_function", [llm_categorize_then_bin, llm_tournament_bin]
)
@pytest.mark.asyncio
async def test_llm_categorize_then_bin(
    binning_function: BinningFunction,
    texts: Collection[str],
    expected_bins: set[Collection[str]],
) -> None:
    bins = await binning_function(LLMModel(name="gpt-4o"), texts)
    assert len(bins) == len(expected_bins)
    assert all(set(b) in expected_bins for b in bins)
