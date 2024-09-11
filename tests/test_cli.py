import io
import os
import sys
from pathlib import Path

import pytest

from paperqa.settings import Settings
from paperqa.utils import pqa_directory

try:
    from paperqa.agents import ask, build_index, main, search_query
    from paperqa.agents.models import AnswerResponse
except ImportError:
    pytest.skip("agents module is not installed", allow_module_level=True)


def test_can_modify_settings() -> None:
    old_argv = sys.argv
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    try:
        sys.argv = "paperqa -s debug --llm=my-model-foo save unit_test".split()
        main()

        sys.stdout = captured_output
        assert Settings.from_name("unit_test").llm == "my-model-foo"

        sys.argv = "paperqa -s unit_test view".split()
        main()

        output = captured_output.getvalue().strip()
        assert "my-model-foo" in output
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        os.unlink(pqa_directory("settings") / "unit_test.json")


def test_cli_ask(agent_index_dir: Path, stub_data_dir: Path) -> None:
    settings = Settings.from_name("debug")
    settings.index_directory = agent_index_dir
    settings.paper_directory = stub_data_dir
    response = ask(
        "How can you use XAI for chemical property prediction?", settings=settings
    )
    assert response.answer.formatted_answer

    search_result = search_query(
        " ".join(response.answer.formatted_answer.split()[:5]),
        "answers",
        settings,
    )
    found_answer = search_result[0][0]
    assert isinstance(found_answer, AnswerResponse)
    assert found_answer.model_dump_json() == response.model_dump_json()


def test_cli_can_build_and_search_index(
    agent_index_dir: Path, stub_data_dir: Path
) -> None:
    settings = Settings.from_name("debug")
    settings.index_directory = agent_index_dir
    index_name = "test"
    build_index(index_name, stub_data_dir, settings)
    search_result = search_query("XAI", index_name, settings)
    assert search_result
