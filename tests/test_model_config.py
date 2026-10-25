"""Regression checks for model defaults when callers supply partial router config."""

import pytest
from ldp.agent import SimpleAgent

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


def test_partial_summary_config_keeps_luna_reasoning_off() -> None:
    settings = Settings(
        summary_llm_config={"rate_limit": {"gpt-6-luna": "30000 per 1 minute"}}
    )

    params = settings.get_summary_llm().config["model_list"][0]["litellm_params"]
    assert params["model"] == "gpt-6-luna"
    assert params["reasoning_effort"] == "none"


def test_explicit_model_list_is_preserved() -> None:
    model_list = [
        {"model_name": "gpt-6.1-sol", "litellm_params": {"model": "gpt-6.1-sol"}}
    ]
    settings = Settings(llm_config={"model_list": model_list})

    assert settings.get_llm().config["model_list"] == model_list


@pytest.mark.asyncio
async def test_ldp_sol_config_omits_temperature() -> None:
    settings = Settings(agent={"agent_llm": "gpt-6.1-sol"})

    agent = await settings.make_ldp_agent(SimpleAgent)
    assert agent is not None
    assert "temperature" not in agent.llm_model
    params = agent.llm_model["model_list"][0]["litellm_params"]
    assert params["model"] == "gpt-6.1-sol"
    assert "temperature" not in params
