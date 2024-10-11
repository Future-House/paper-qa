import pathlib
import pickle
from typing import Any
from unittest.mock import patch

import litellm
import pytest

from paperqa import LiteLLMModel
from paperqa.llms import Chunk
from tests.conftest import VCR_DEFAULT_MATCH_ON


class TestLiteLLMModel:
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
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
                id="with-router",
            ),
            pytest.param(
                {
                    "pass_through_router": True,
                    "router_kwargs": {"temperature": 0, "max_tokens": 56},
                },
                id="without-router",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_run_prompt(self, config: dict[str, Any]) -> None:
        llm = LiteLLMModel(name="gpt-4o-mini", config=config)

        outputs = []

        def accum(x) -> None:
            outputs.append(x)

        completion = await llm.run_prompt(
            prompt="The {animal} says",
            data={"animal": "duck"},
            skip_system=True,
            callbacks=[accum],
        )
        assert completion.model == "gpt-4o-mini"
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

    @pytest.mark.vcr
    @pytest.mark.parametrize(
        ("config", "bypassed_router"),
        [
            pytest.param(
                {
                    "model_list": [
                        {
                            "model_name": "gpt-4o-mini",
                            "litellm_params": {"model": "gpt-4o-mini", "max_tokens": 3},
                        }
                    ]
                },
                False,
                id="with-router",
            ),
            pytest.param(
                {"pass_through_router": True, "router_kwargs": {"max_tokens": 3}},
                True,
                id="without-router",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_max_token_truncation(
        self, config: dict[str, Any], bypassed_router: bool
    ) -> None:
        llm = LiteLLMModel(name="gpt-4o-mini", config=config)
        with patch(
            "litellm.Router.atext_completion",
            side_effect=litellm.Router.atext_completion,
            autospec=True,
        ) as mock_atext_completion:
            chunk = await llm.acomplete("Please tell me a story")  # type: ignore[call-arg]
        if bypassed_router:
            mock_atext_completion.assert_not_awaited()
        else:
            mock_atext_completion.assert_awaited_once()
        assert isinstance(chunk, Chunk)
        assert chunk.completion_tokens == 3
        assert chunk.text
        assert len(chunk.text) < 20

    def test_pickling(self, tmp_path: pathlib.Path) -> None:
        pickle_path = tmp_path / "llm_model.pickle"
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
        with pickle_path.open("wb") as f:
            pickle.dump(llm, f)
        with pickle_path.open("rb") as f:
            rehydrated_llm = pickle.load(f)
        assert llm.name == rehydrated_llm.name
        assert llm.config == rehydrated_llm.config
        assert llm.router.deployment_names == rehydrated_llm.router.deployment_names
