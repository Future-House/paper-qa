import os
import sys
import zlib
from pathlib import Path

import pytest
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt

from paperqa import Docs
from paperqa.settings import Settings
from paperqa.utils import pqa_directory

try:
    from paperqa.agents import ask, build_index, main, search_query
    from paperqa.agents.models import AnswerResponse
except ImportError:
    pytest.skip("agents module is not installed", allow_module_level=True)


def test_can_modify_settings(capsys, stub_data_dir: Path) -> None:
    rel_path_home_to_stub_data = Path("~") / stub_data_dir.relative_to(Path.home())

    # This test depends on the unit_test config not previously existing
    with pytest.raises(FileNotFoundError, match="unit_test"):
        Settings.from_name("unit_test")

    old_argv = sys.argv
    try:
        sys.argv = (
            "paperqa -s debug --llm=my-model-foo"
            f" --agent.index.paper_directory={rel_path_home_to_stub_data!s} save"
            " unit_test"
        ).split()
        main()

        captured = capsys.readouterr()
        assert not captured.err
        assert "Settings saved" in captured.out
        settings = Settings.from_name("unit_test")
        assert settings.llm == "my-model-foo"
        assert settings.agent.index.paper_directory == str(rel_path_home_to_stub_data)

        sys.argv = ["paperqa", "-s", "unit_test", "view"]
        main()

        captured = capsys.readouterr()
        assert not captured.err
        assert "my-model-foo" in captured.out
    finally:
        sys.argv = old_argv
        os.unlink(pqa_directory("settings") / "unit_test.json")


def test_cli_ask(agent_index_dir: Path, stub_data_dir: Path) -> None:
    settings = Settings.from_name("debug")
    settings.agent.index.index_directory = agent_index_dir
    settings.agent.index.paper_directory = stub_data_dir
    response = ask(
        "How can you use XAI for chemical property prediction?", settings=settings
    )
    assert response.session.formatted_answer

    search_result = search_query(
        " ".join(response.session.formatted_answer.split()),
        "answers",
        settings,
    )
    found_answer = search_result[0][0]
    assert isinstance(found_answer, AnswerResponse)
    assert found_answer.model_dump() == response.model_dump()


def test_cli_can_build_and_search_index(
    agent_index_dir: Path, stub_data_dir: Path
) -> None:
    rel_path_home_to_stub_data = Path("~") / stub_data_dir.relative_to(Path.home())
    settings = Settings.from_name("debug")
    settings.agent.index.paper_directory = rel_path_home_to_stub_data
    settings.agent.index.index_directory = agent_index_dir
    index_name = "test"
    for attempt in Retrying(
        stop=stop_after_attempt(3),
        # zlib.error: Error -5 while decompressing data: incomplete or truncated stream
        retry=retry_if_exception_type(zlib.error),
    ):
        with attempt:
            build_index(index_name, stub_data_dir, settings)
    result = search_query("XAI", index_name, settings)
    assert len(result) == 1
    assert isinstance(result[0][0], Docs)
    assert all(d.startswith("Wellawatte") for d in result[0][0].docnames)
    assert result[0][1] == "paper.pdf"
