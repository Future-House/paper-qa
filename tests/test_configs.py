from unittest.mock import patch

import pytest
from pydantic import ValidationError

from paperqa.settings import (
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

    with pytest.raises(ValidationError):
        PromptSettings(pre="Invalid {var}")

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
