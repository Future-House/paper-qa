import importlib.resources
import os
import pathlib
import re
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError
from pytest_subtests import SubTests

import paperqa.configs
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
from tests.conftest import TESTS_DIR


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
    # Also let's check our default settings work fine with round-trip JSON serialization
    serde_default_settings = Settings(**default_settings.model_dump(mode="json"))
    for setting in (default_settings, serde_default_settings):
        assert any(x in setting.llm for x in ("gpt-", "claude-"))
        assert setting.answer.evidence_k == 10
        assert HOME_DIR in str(setting.agent.index.index_directory)
        assert ".pqa" in str(setting.agent.index.index_directory)

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


@pytest.mark.parametrize(
    "model_name",
    [
        "o1",
        "o1-thismodeldoesnotexist",
        "gpt-5",
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
    ],
)
def test_models_requiring_temp_1(model_name: str) -> None:
    with pytest.warns(UserWarning, match="temperature") as record:  # noqa: PT031
        s = Settings(llm=model_name, temperature=0)
        (w,) = record.list
        assert "temperature must be set to 1" in str(w.message)
        assert s.temperature == 1

        Settings(llm=model_name, temperature=1)
        assert record.list == [w], "Expected no new warnings with correct temperature"


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


def test_validity_of_bundled_configs(subtests: SubTests) -> None:
    for config_file in [
        f
        for f in importlib.resources.files(paperqa.configs).iterdir()
        if f.name.endswith(".json")
    ]:
        config_name = config_file.name.removesuffix(".json")
        with subtests.test(msg=config_name):
            settings = get_settings(config_name)
            assert isinstance(settings, Settings)


def test_readme_settings_cheatsheet_accuracy(subtests: SubTests) -> None:
    readme_path = TESTS_DIR.parent / "README.md"

    # Extract the Markdown table in the Settings Cheatsheet section
    *_, after_header = readme_path.read_text().partition("## Settings Cheatsheet")
    cheatsheet_match = re.match(r"\s*\|.*?\n\|[-| ]+\n((?:\|.*\n)+)", after_header)
    assert cheatsheet_match, "Expected to find Settings Cheatsheet"
    setting_paths: list[str] = [
        row.split("|")[1].strip().strip("`")
        for row in cheatsheet_match.group(1).strip().split("\n")
    ]
    assert setting_paths, "No settings found in the cheatsheet table"
    assert len(setting_paths) == len(set(setting_paths)), "Expected unique settings"

    # Check settings in the cheatsheet are present in Settings
    settings = Settings()
    for setting_path in setting_paths:
        with subtests.test(msg=setting_path):
            # Navigate the dotted path (e.g., "answer.evidence_k" -> settings.answer.evidence_k)
            obj = settings
            for part in setting_path.split("."):
                assert hasattr(obj, part), (
                    f"Setting path '{setting_path}' does not exist: "
                    f"'{part}' not found on {type(obj).__name__}"
                )
                obj = getattr(obj, part)

    # Also check the reverse, that all settings are documented
    def get_leaf_field_paths(model: type[BaseModel], prefix: str = "") -> set[str]:
        paths: set[str] = set()
        for name, field_info in model.model_fields.items():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(field_info.annotation, type) and issubclass(
                field_info.annotation, BaseModel
            ):
                # Field is a nested model
                paths |= get_leaf_field_paths(field_info.annotation, path)
            else:
                paths.add(path)
        return paths

    undocumented = get_leaf_field_paths(Settings) - set(setting_paths)
    assert not undocumented, "Settings fields are missing from the cheatsheet"
