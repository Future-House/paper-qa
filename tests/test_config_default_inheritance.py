"""Guard against duplicating the default answer-source limit in bundled presets."""

import importlib.resources
import json

import pytest

import paperqa.configs
from paperqa.settings import Settings, get_settings


@pytest.mark.parametrize(
    "config_name",
    ["clinical_trials", "high_quality", "search_only_clinical_trials"],
)
def test_presets_inherit_default_answer_max_sources(config_name: str) -> None:
    config_path = importlib.resources.files(paperqa.configs) / f"{config_name}.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))

    assert "answer_max_sources" not in raw_config["answer"]
    assert (
        get_settings(config_name).answer.answer_max_sources
        == Settings().answer.answer_max_sources
    )
