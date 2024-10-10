import pytest

from paperqa import LiteLLMModel
from tests.conftest import VCR_DEFAULT_MATCH_ON


class TestLiteLLMModel:
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.asyncio
    async def test_run_prompt(self) -> None:
        llm = LiteLLMModel(
            name="gpt-4o-mini",
            config={
                "model_list": [
                    {
                        "model_name": "gpt-4o-mini",
                        "litellm_params": {
                            "model": "gpt-4o-mini",
                            "temperature": 0,
                            "max_tokens": 56,
                        },
                    }
                ]
            },
        )

        outputs = []

        def accum(x) -> None:
            outputs.append(x)

        completion = await llm.run_prompt(
            prompt="The {animal} says",
            data={"animal": "duck"},
            skip_system=True,
            callbacks=[accum],
        )
        assert completion.seconds_to_first_token > 0
        assert completion.prompt_count > 0
        assert completion.completion_count > 0
        assert str(completion) == "".join(outputs)
        assert completion.cost > 0

        completion = await llm.run_prompt(
            prompt="The {animal} says",
            data={"animal": "duck"},
            skip_system=True,
        )
        assert completion.seconds_to_first_token == 0
        assert completion.seconds_to_last_token > 0
        assert completion.cost > 0

        # check with mixed callbacks
        async def ac(x) -> None:
            pass

        completion = await llm.run_prompt(
            prompt="The {animal} says",
            data={"animal": "duck"},
            skip_system=True,
            callbacks=[accum, ac],
        )
        assert completion.cost > 0
