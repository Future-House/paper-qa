from pathlib import Path

import pytest

try:
    from paperqa.agents import ask, build_index, clear, search_query, set_setting, show
    from paperqa.agents.models import AnswerResponse
    from paperqa.agents.search import SearchIndex
except ImportError:
    pytest.skip("agents module is not installed", allow_module_level=True)


def test_cli_set(agent_index_dir: Path):  # noqa: ARG001
    set_setting("temperature", "0.5")
    assert show("temperature") == "0.5", "Temperature not properly set"

    with pytest.raises(ValueError) as excinfo:  # noqa: PT011
        set_setting("temperature", "abc")
    assert "temperature (with value abc) is not a valid setting." in str(excinfo.value)

    # ensure we can do nested settings
    set_setting("agent_tools.paper_directory", "my_directory")
    assert (
        show("agent_tools.paper_directory") == "my_directory"
    ), "Nested setting not properly set"

    # ensure we can set settings which fail specific validations
    # normally we'd get a failure for model mixing, but this is reserved for runtime
    set_setting("llm", "claude-3-5-sonnet-20240620")
    assert show("llm") == "claude-3-5-sonnet-20240620", "Setting not properly set"

    # test that we're able to collection structures like lists
    set_setting(
        "parsing_configuration.ordered_parser_preferences", '["paperqa_default"]'
    )
    assert show("parsing_configuration.ordered_parser_preferences") == [
        "paperqa_default"
    ], "List setting not properly set"


@pytest.mark.asyncio
async def test_cli_show(agent_index_dir: Path):

    # make empty index
    assert not show("indexes"), "No indexes should be present"

    # creates a new index/file
    si = SearchIndex(index_directory=agent_index_dir)
    await si.init_directory()

    set_setting("temperature", "0.5")
    set_setting("agent_tools.paper_directory", "my_directory")

    assert show("temperature") == "0.5", "Temperature not properly set"

    assert not show("fake_variable"), "Fake variable should not be set"

    assert show("all") == {
        "temperature": "0.5",
        "agent_tools": {"paper_directory": "my_directory"},
    }, "All settings not properly set"

    assert show("indexes") == ["pqa_index"], "Index not properly set"

    assert show("answers") == [], "Answers should be empty"


@pytest.mark.asyncio
async def test_cli_clear(agent_index_dir: Path):

    set_setting("temperature", "0.5")
    assert show("temperature") == "0.5", "Temperature not properly set"

    clear("temperature")
    assert show("temperature") is None, "Temperature not properly cleared"

    # set a nested variable
    set_setting("prompts.qa", "Answer my question!")
    assert show("prompts.qa") == "Answer my question!", "Prompt not properly set"
    clear("prompts.qa")
    assert show("prompts.qa") is None, "Prompt not properly cleared"

    # creates a new index/file
    si = SearchIndex(index_directory=agent_index_dir)
    await si.init_directory()

    clear("pqa_index", index=True)

    assert show("indexes") == [], "Index not properly cleared"


def test_cli_ask(agent_index_dir: Path):
    set_setting("consider_sources", "10")
    set_setting("max_sources", "2")
    set_setting("embedding", "sparse")
    set_setting("agent_tools.search_count", "1")
    answer = ask(
        "How can you use XAI for chemical property prediction?",
        directory=Path(__file__).parent,
        index_directory=agent_index_dir,
    )
    assert isinstance(answer, AnswerResponse), "Answer not properly returned"
    assert (
        "I cannot answer" not in answer.answer.answer
    ), "An answer should be generated."
    assert len(answer.answer.context) >= 1, "No contexts were found."
    answers = search_query(
        "How can you use XAI for chemical property prediction?",
        index_directory=agent_index_dir,
    )
    answer = answers[0][0]
    assert isinstance(answer, AnswerResponse), "Answer not properly returned"
    assert (
        "I cannot answer" not in answer.answer.answer
    ), "An answer should be generated."
    assert len(answer.answer.context) >= 1, "No contexts were found."
    assert len(show("answers")) == 1, "An answer should be returned"


def test_cli_index(agent_index_dir: Path, caplog):

    build_index(directory=Path(__file__).parent, index_directory=agent_index_dir)

    caplog.clear()

    ask(
        "How can you use XAI for chemical property prediction?",
        directory=Path(__file__).parent,
        index_directory=agent_index_dir,
    )

    # ensure we have no indexing logs after starting the search
    for record in caplog.records:
        if "Metadata not found" in record.msg:
            raise AssertionError(
                "Indexing logs should not be present after search starts"
            )

    caplog.clear()

    # running again should not trigger any indexing
    build_index(directory=Path(__file__).parent, index_directory=agent_index_dir)
    assert not caplog.records, "Indexing should not be triggered again"

    # now we want to change the settings
    set_setting("embedding", "sparse")

    # running again should now re-trigger an indexing
    build_index(directory=Path(__file__).parent, index_directory=agent_index_dir)

    assert caplog.records, "Indexing should be triggered again"
