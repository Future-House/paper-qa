"""Regression checks for model defaults when callers supply partial router config."""

import pytest
from ldp.agent import SimpleAgent
from lmi import LiteLLMModel

from paperqa.settings import Settings

pytestmark = pytest.mark.offline


def test_partial_answer_config_keeps_sol_parameters() -> None:
    settings = Settings(
        llm_config={"rate_limit": {"gpt-6.1-sol": "30000 per 1 minute"}}
    )

    config = settings.get_llm().config
    assert config["rate_limit"] == {"gpt-6.1-sol": "30000 per 1 minute"}
    params = config["model_list"][0]["litellm_params"]
    assert params["model"] == "gpt-6.1-sol"
    assert "temperature" not in params


def test_astra_defaults_omit_temperature() -> None:
    settings = Settings(llm="gpt-6-astra", summary_llm="gpt-6-astra")

    for model in (settings.get_llm(), settings.get_summary_llm()):
        params = model.config["model_list"][0]["litellm_params"]
        assert params["model"] == "gpt-6-astra"
        assert "temperature" not in params
        assert "reasoning_effort" not in params


def test_explicit_model_list_is_preserved() -> None:
    model_list = [
        {"model_name": "gpt-6.1-sol", "litellm_params": {"model": "gpt-6.1-sol"}}
    ]
    settings = Settings(llm_config={"model_list": model_list})

    assert settings.get_llm().config["model_list"] == model_list


@pytest.mark.parametrize(
    ("agent_llm", "reasoning_effort"),
    [("gpt-6.1-sol", None), ("gpt-6-sol", "none")],
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
