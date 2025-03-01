import os
import pathlib
import warnings
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pytest_subtests import SubTests

from paperqa.prompts import citation_prompt
from paperqa.settings import (
    AgentSettings,
    IndexSettings,
    MaybeSettings,
    PromptSettings,
    Settings,
    get_formatted_variables,
    get_settings,
)
from paperqa.types import Doc, DocDetails
from paperqa.utils import get_year


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


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("fast", id="name"),
        pytest.param({"parsing": {"use_doc_details": False}}, id="serialized"),
    ],
)
def test_get_settings_with_valid_config(value: MaybeSettings) -> None:
    settings = get_settings(value)
    assert not settings.parsing.use_doc_details


def test_get_settings_missing_file() -> None:
    with (
        patch("importlib.resources.files", side_effect=FileNotFoundError),
        pytest.raises(FileNotFoundError),
    ):
        get_settings("missing_config")


HOME_DIR = str(pathlib.Path.home())


def test_settings_default_instantiation(tmpdir, subtests: SubTests) -> None:
    default_settings = Settings()
    assert "gpt-" in default_settings.llm
    assert default_settings.answer.evidence_k == 10
    assert HOME_DIR in str(default_settings.agent.index.index_directory)
    assert ".pqa" in str(default_settings.agent.index.index_directory)

    with subtests.test(msg="alternate-pqa-home"):
        assert HOME_DIR not in str(tmpdir), "Later assertion requires this to pass"
        with patch.dict(os.environ, values={"PQA_HOME": str(tmpdir)}):
            alt_home_settings = Settings()
        assert (
            alt_home_settings.agent.index.index_directory
            != default_settings.agent.index.index_directory
        )
        assert HOME_DIR not in str(alt_home_settings.agent.index.index_directory)
        assert ".pqa" in str(alt_home_settings.agent.index.index_directory)


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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = Settings(llm="o1-thismodeldoesnotexist", temperature=0)
        assert "temperature must be set to 1" in str(w[-1].message)
        assert s.temperature == 1

    # Test that temperature=1 produces no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = Settings(llm="o1-thismodeldoesnotexist", temperature=1)
        assert not w


@pytest.mark.parametrize(
    ("doc_class", "doc_data", "filter_criteria", "expected_result"),
    [
        pytest.param(
            Doc,
            {
                "docname": "Test Paper",
                "citation": "Test Citation",
                "dockey": "key1",
            },
            {"docname": "Test Paper"},
            True,
            id="Doc-matching-docname",
        ),
        pytest.param(
            Doc,
            {
                "docname": "Test Paper",
                "citation": "Test Citation",
                "dockey": "key1",
            },
            {"docname": "Another Paper"},
            False,
            id="Doc-nonmatching-docname",
        ),
        pytest.param(
            DocDetails,
            {
                "title": "Test Paper",
                "authors": ["Alice", "Bob"],
                "year": 2020,
            },
            {"title": "Test Paper"},
            True,
            id="DocDetails-matching-title",
        ),
        pytest.param(
            DocDetails,
            {
                "title": "Test Paper",
                "authors": ["Alice", "Bob"],
                "year": 2020,
            },
            {"!year": 2020, "?foo": "bar"},
            False,
            id="DocDetails-inverted-matching-year",
        ),
        pytest.param(
            DocDetails,
            {
                "title": "Test Paper",
                "authors": ["Alice", "Bob"],
                "year": 2020,
            },
            {"year": 2020, "foo": "bar"},
            False,
            id="DocDetails-missing-param-fail",
        ),
        pytest.param(
            DocDetails,
            {
                "title": "Test Paper",
                "authors": ["Alice", "Bob"],
                "year": 2020,
            },
            {"?volume": "10", "!title": "Another Paper"},
            True,
            id="DocDetails-relaxed-missing-volume",
        ),
    ],
)
def test_matches_filter_criteria(doc_class, doc_data, filter_criteria, expected_result):
    doc = doc_class(**doc_data)
    assert doc.matches_filter_criteria(filter_criteria) == expected_result


def test_citation_prompt_current_year():
    expected_year_text = f"the current year is {get_year()}"

    assert expected_year_text in citation_prompt, (
        f"Citation prompt should contain '{expected_year_text}' but got:"
        f" {citation_prompt}"
    )
