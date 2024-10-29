from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pytest_subtests import SubTests

from paperqa.settings import (
    AgentSettings,
    IndexSettings,
    PromptSettings,
    Settings,
    get_formatted_variables,
    get_settings,
)


def test_prompt_settings_validation() -> None:
    with pytest.raises(ValidationError):
        PromptSettings(summary="Invalid {variable}")

    valid_settings = PromptSettings(
        summary="{citation} {question} {summary_length} {text}"
    )
    assert valid_settings.summary == "{citation} {question} {summary_length} {text}"

    valid_pre_settings = PromptSettings(pre="{question}")
    assert valid_pre_settings.pre == "{question}"


def test_get_formatted_variables() -> None:
    template = "This is a test {variable} with {another_variable}"
    variables = get_formatted_variables(template)
    assert variables == {"variable", "another_variable"}


def test_get_settings_with_valid_config() -> None:
    settings = get_settings("fast")
    assert not settings.parsing.use_doc_details


def test_get_settings_missing_file() -> None:
    with (
        patch("importlib.resources.files", side_effect=FileNotFoundError),
        pytest.raises(FileNotFoundError),
    ):
        get_settings("missing_config")


def test_settings_default_instantiation() -> None:
    settings = Settings()
    assert "gpt-" in settings.llm
    assert settings.answer.evidence_k == 10


def test_index_naming(subtests: SubTests) -> None:
    with subtests.test(msg="no name"):
        settings = Settings()
        with pytest.raises(ValueError, match="specify a name"):
            settings.agent.index.get_named_index_directory()

    with subtests.test(msg="with name"):
        settings = Settings(agent=AgentSettings(index=IndexSettings(name="test")))
        assert settings.agent.index.get_named_index_directory().name == "test"


def test_router_kwargs_present_in_models() -> None:
    settings = Settings()
    assert settings.get_llm().config["router_kwargs"] is not None
    assert settings.get_summary_llm().config["router_kwargs"] is not None


def test_o1_requires_temp_equals_1() -> None:
    with pytest.raises(ValidationError):
        _ = Settings(llm="o1-thismodeldoesnotexist", temperature=0)
    _ = Settings(llm="o1-thismodeldoesnotexist", temperature=1)
