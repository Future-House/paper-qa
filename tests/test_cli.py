import logging
import os
import sys
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt

from paperqa import Docs
from paperqa.agents import (
    ask,
    build_index,
    list_built_indexes,
    main,
    pqa_root,
    search_query,
)
from paperqa.agents.models import AnswerResponse
from paperqa.settings import Settings
from paperqa.utils import pqa_directory


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


def test_cli_where_prints_pqa_directory_and_indexes(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PQA_HOME", str(tmp_path))
    index_dir = tmp_path / ".pqa" / "indexes"
    (index_dir / "answers").mkdir(parents=True)
    (index_dir / "nanomaterials").mkdir()
    (index_dir / "notes.txt").write_text("not an index\n")

    old_argv = sys.argv
    try:
        sys.argv = ["paperqa", "where"]
        with caplog.at_level(logging.INFO, logger="paperqa.agents"):
            main()
    finally:
        sys.argv = old_argv

    text = "\n".join(caplog.messages)
    assert f"PQA directory: {tmp_path / '.pqa'}" in text
    assert f"Index directory: {index_dir}" in text
    assert "Indexes:" in text
    assert "  answers" in text
    assert "  nanomaterials" in text
    assert "notes.txt" not in text


def test_list_built_indexes_skips_files_and_missing_dirs(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert list_built_indexes(missing) == []

    (tmp_path / "alpha").mkdir()
    (tmp_path / "zeta").mkdir()
    (tmp_path / "readme.txt").write_text("skip\n")
    assert list_built_indexes(tmp_path) == ["alpha", "zeta"]


def test_pqa_root_expands_tilde_in_pqa_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    nested = tmp_path / "pqa-parent"
    nested.mkdir()
    monkeypatch.setenv("PQA_HOME", "~/pqa-parent")
    assert pqa_root() == nested / ".pqa"
    assert pqa_directory("indexes") == nested / ".pqa" / "indexes"


def test_list_built_indexes_returns_empty_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "indexes"
    directory.mkdir()

    def boom(self: Path) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "iterdir", boom)
    assert list_built_indexes(directory) == []


def test_cli_missing_command_prints_stable_help(capsys) -> None:
    old_argv = sys.argv
    try:
        sys.argv = ["paperqa"]
        main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert "view, ask, search, index, where" in captured.out


def test_cli_ask(agent_index_dir: Path, stub_data_dir: Path) -> None:
    settings = Settings.from_name("debug")
    settings.agent.index.index_directory = agent_index_dir
    settings.agent.index.paper_directory = stub_data_dir
    response = ask(
        "How can you use XAI for chemical property prediction?", settings=settings
    )
    assert isinstance(response, AnswerResponse)
    assert response.session.formatted_answer

    search_result = search_query(
        " ".join(response.session.formatted_answer.split()),
        "answers",
        settings,
    )
    assert isinstance(search_result, list)
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
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0][0], Docs)
    assert all(d.startswith("Wellawatte") for d in result[0][0].docnames)
    assert result[0][1] == "paper.pdf"


def test_settings_index_name_used_when_index_arg_is_default(
    stub_data_dir: Path,
) -> None:
    settings = Settings(
        agent={"index": {"paper_directory": stub_data_dir, "name": "my_named_index"}}
    )

    with patch("paperqa.agents.get_directory_index") as mock_get_directory_index:
        # When --index isn't provided, the default name of "default" will still
        # respect a custom-specified index name
        build_index("default", stub_data_dir, settings)

    assert settings.agent.index.name == "my_named_index"
    mock_get_directory_index.assert_awaited_once_with(settings=settings)
    passed_settings = mock_get_directory_index.call_args.kwargs["settings"]
    assert passed_settings.agent.index.name == "my_named_index"

    with patch("paperqa.agents.index_search", return_value=[]) as mock_index_search:
        # When --index isn't provided, the default name of "default" will still
        # respect a custom-specified index name
        search_query("XAI", "default", settings)

    mock_index_search.assert_awaited_once()
    assert mock_index_search.call_args.kwargs["index_name"] == "my_named_index"
