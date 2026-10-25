"""Regression checks for model defaults when callers supply partial router config."""

import litellm
import pytest
from ldp.agent import SimpleAgent
from lmi import LiteLLMModel

from paperqa.settings import Settings

pytestmark = pytest.mark.offline


def test_default_models_are_accepted_by_litellm() -> None:
    settings = Settings()
    models = (
        settings.get_llm(),
        settings.get_summary_llm(),
        settings.get_agent_llm(),
    )
    for model in models:
        params = model.config["model_list"][0]["litellm_params"]
        assert litellm.get_llm_provider(params["model"])[1] == "openai"

    agent_params = models[-1].config["model_list"][0]["litellm_params"]
    request_params = litellm.get_optional_params(
        model=agent_params["model"],
        reasoning_effort=agent_params["reasoning_effort"],
        allowed_openai_params=agent_params["allowed_openai_params"],
    )
    assert request_params["reasoning_effort"] == "none"


def test_partial_answer_config_keeps_sol_parameters() -> None:
    settings = Settings(
        llm_config={"rate_limit": {"openai/gpt-6.1-sol": "30000 per 1 minute"}}
    )

    config = settings.get_llm().config
    assert config["rate_limit"] == {"openai/gpt-6.1-sol": "30000 per 1 minute"}
    params = config["model_list"][0]["litellm_params"]
    assert params["model"] == "openai/gpt-6.1-sol"
    assert "temperature" not in params


def test_astra_defaults_omit_temperature() -> None:
    settings = Settings(llm="openai/gpt-6-astra", summary_llm="openai/gpt-6-astra")

    for model in (settings.get_llm(), settings.get_summary_llm()):
        params = model.config["model_list"][0]["litellm_params"]
        assert params["model"] == "openai/gpt-6-astra"
        assert "temperature" not in params
        assert "reasoning_effort" not in params


def test_explicit_model_list_is_preserved() -> None:
    model_list = [
        {
            "model_name": "openai/gpt-6.1-sol",
            "litellm_params": {"model": "openai/gpt-6.1-sol"},
        }
    ]
    settings = Settings(llm_config={"model_list": model_list})

    assert settings.get_llm().config["model_list"] == model_list


@pytest.mark.parametrize(
    ("agent_llm", "reasoning_effort"),
    [("openai/gpt-6.1-sol", None), ("openai/gpt-6-sol", "none")],
)
@pytest.mark.asyncio
async def test_ldp_agent_omits_default_temperature(
    agent_llm: str, reasoning_effort: str | None
) -> None:
    settings = Settings(agent={"agent_llm": agent_llm})

    agent = await settings.make_ldp_agent(SimpleAgent)
    assert agent is not None
    assert "temperature" not in agent.llm_model
    params = LiteLLMModel(config=agent.llm_model).config["model_list"][0][
        "litellm_params"
    ]
    assert params["model"] == agent_llm
    assert "temperature" not in params
    if reasoning_effort is None:
        assert "reasoning_effort" not in params
    else:
        assert params["reasoning_effort"] == reasoning_effort
        assert params["allowed_openai_params"] == ["reasoning_effort"]
