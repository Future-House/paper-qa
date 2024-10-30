from __future__ import annotations

import importlib
import itertools
import json
import logging
import re
import shutil
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import ldp.agent
import pytest
from aviary.tools import ToolCall, ToolRequestMessage, ToolsAdapter, ToolSelector
from ldp.agent import MemoryAgent, SimpleAgent
from ldp.graph.memory import Memory, UIndexMemoryModel
from ldp.graph.ops import OpResult
from ldp.llms import EmbeddingModel, MultipleCompletionLLMModel
from pydantic import ValidationError
from pytest_subtests import SubTests
from tantivy import Index

from paperqa.agents import SearchIndex, agent_query
from paperqa.agents.env import settings_to_tools
from paperqa.agents.main import FAKE_AGENT_TYPE
from paperqa.agents.models import AgentStatus, AnswerResponse, QueryRequest
from paperqa.agents.search import (
    FAILED_DOCUMENT_ADD_ID,
    get_directory_index,
    maybe_get_manifest,
)
from paperqa.agents.task import GradablePaperQAEnvironment
from paperqa.agents.tools import (
    EnvironmentState,
    GatherEvidence,
    GenerateAnswer,
    PaperSearch,
    make_status,
)
from paperqa.docs import Docs
from paperqa.settings import AgentSettings, IndexSettings, Settings
from paperqa.types import Context, Doc, PQASession, Text
from paperqa.utils import extract_thought, get_year, md5sum


@pytest.mark.asyncio
async def test_get_directory_index(agent_test_settings: Settings) -> None:
    # Since agent_test_settings is used by other tests, we use a tempdir so we
    # can delete files without affecting concurrent tests
    with tempfile.TemporaryDirectory() as tempdir:
        shutil.copytree(
            agent_test_settings.agent.index.paper_directory, tempdir, dirs_exist_ok=True
        )
        paper_dir = agent_test_settings.agent.index.paper_directory = Path(tempdir)

        index_name = agent_test_settings.agent.index.name = (
            f"stub{uuid4()}"  # Unique across test invocations
        )
        index = await get_directory_index(settings=agent_test_settings)
        assert (
            index.index_name == index_name
        ), "Index name should match its specification"
        assert index.fields == [
            "file_location",
            "body",
            "title",
            "year",
        ], "Incorrect fields in index"
        assert not index.changed, "Expected index to not have changes at this point"
        # paper.pdf + empty.txt + flag_day.html + bates.txt + obama.txt,
        # but empty.txt fails to be added
        path_to_id = await index.index_files
        assert (
            sum(id_ != FAILED_DOCUMENT_ADD_ID for id_ in path_to_id.values()) == 4
        ), "Incorrect number of parsed index files"
        results = await index.query(query="who is Frederick Bates?")
        assert results[0].docs.keys() == {md5sum((paper_dir / "bates.txt").absolute())}

        # Check getting the same index name will not reprocess files
        with patch.object(Docs, "aadd") as mock_aadd:
            index = await get_directory_index(settings=agent_test_settings)
        assert len(await index.index_files) == len(path_to_id)
        mock_aadd.assert_not_awaited(), "Expected we didn't re-add files"

        # Now we actually remove (but not add!) a file from the paper directory,
        # and we still don't reprocess files
        (paper_dir / "obama.txt").unlink()
        with patch.object(
            Docs, "aadd", autospec=True, side_effect=Docs.aadd
        ) as mock_aadd:
            index = await get_directory_index(settings=agent_test_settings)
        assert len(await index.index_files) == len(path_to_id) - 1
        mock_aadd.assert_not_awaited(), "Expected we didn't re-add files"

        # Note let's delete files.zip, and confirm we can't load the index
        await (await index.file_index_filename).unlink()
        with pytest.raises(RuntimeError, match="please rebuild"):
            await get_directory_index(settings=agent_test_settings, build=False)


@pytest.mark.asyncio
async def test_resuming_crashed_index_build(agent_test_settings: Settings) -> None:
    index_settings = agent_test_settings.agent.index
    crash_threshold, index_settings.concurrency = 3, 2
    num_source_files = len(
        [
            x
            for x in cast(Path, index_settings.paper_directory).iterdir()
            if x.suffix != ".csv"
        ]
    )
    assert (
        num_source_files >= 5
    ), "Less source files than this test was designed to work with"
    call_count = 0
    original_docs_aadd = Docs.aadd

    async def crashing_aadd(*args, **kwargs) -> str | None:
        nonlocal call_count
        if call_count == crash_threshold:
            raise RuntimeError("Unexpected crash.")
        call_count += 1
        return await original_docs_aadd(*args, **kwargs)

    # 1. Try to build an index, and crash halfway through
    with (
        pytest.raises(ExceptionGroup, match="unhandled"),
        patch.object(
            Docs, "aadd", side_effect=crashing_aadd, autospec=True
        ) as mock_aadd,
    ):
        await get_directory_index(settings=agent_test_settings)
    mock_aadd.assert_awaited()

    # 2. Resume and complete building the index
    with patch.object(Docs, "aadd", autospec=True, side_effect=Docs.aadd) as mock_aadd:
        index = await get_directory_index(settings=agent_test_settings)
    assert (
        mock_aadd.await_count <= crash_threshold
    ), "Should have been able to resume build"
    assert len(await index.index_files) > crash_threshold


@pytest.mark.asyncio
async def test_getting_manifest(
    agent_test_settings: Settings, stub_data_dir: Path, caplog
) -> None:
    agent_test_settings.agent.index.manifest_file = "stub_manifest.csv"

    # Since stub_manifest.csv is used by other tests, we use a tempdir so we
    # can modify it without affecting concurrent tests
    with tempfile.TemporaryDirectory() as tempdir, caplog.at_level(logging.WARNING):
        shutil.copytree(stub_data_dir, tempdir, dirs_exist_ok=True)
        agent_test_settings.agent.index.paper_directory = tempdir
        manifest_filepath = (
            await agent_test_settings.agent.index.finalize_manifest_file()
        )
        assert manifest_filepath
        assert await maybe_get_manifest(manifest_filepath)
        assert not caplog.records

        # If a header line isn't present, our manifest extraction should fail
        original_manifest_lines = (await manifest_filepath.read_text()).splitlines()
        await manifest_filepath.write_text(data="\n".join(original_manifest_lines[1:]))
        await maybe_get_manifest(manifest_filepath)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR


EXPECTED_STUB_DATA_FILES = {
    "bates.txt",
    "empty.txt",
    "flag_day.html",
    "obama.txt",
    "paper.pdf",
}


@pytest.mark.asyncio
async def test_get_directory_index_w_manifest(agent_test_settings: Settings) -> None:
    # Set the paper_directory to be a relative path as starting point to confirm this
    # won't trip us up, and set the manifest file too
    abs_paper_dir = cast(Path, agent_test_settings.agent.index.paper_directory)
    agent_test_settings.agent.index.paper_directory = abs_paper_dir.relative_to(
        Path.cwd()
    )
    agent_test_settings.agent.index.manifest_file = "stub_manifest.csv"

    # Now set up both relative and absolute test settings
    relative_test_settings = agent_test_settings.model_copy(deep=True)
    absolute_test_settings = agent_test_settings.model_copy(deep=True)
    absolute_test_settings.agent.index.use_absolute_paper_directory = True
    assert (
        relative_test_settings != absolute_test_settings
    ), "We need to be able to differentiate between relative and absolute settings"
    del agent_test_settings

    relative_index = await get_directory_index(settings=relative_test_settings)
    assert (
        set((await relative_index.index_files).keys()) == EXPECTED_STUB_DATA_FILES
    ), "Incorrect index files, should be relative to share indexes across machines"
    absolute_index = await get_directory_index(settings=absolute_test_settings)
    assert set((await absolute_index.index_files).keys()) == {
        str(abs_paper_dir / f) for f in EXPECTED_STUB_DATA_FILES
    }, (
        "Incorrect index files, should be absolute to deny sharing indexes across"
        " machines"
    )
    for index in (relative_index, absolute_index):
        assert index.fields == [
            "file_location",
            "body",
            "title",
            "year",
        ], "Incorrect fields in index"

        results = await index.query(query="who is Frederick Bates?")
        top_result = next(iter(results[0].docs.values()))
        assert top_result.dockey == md5sum(abs_paper_dir / "bates.txt")
        # note: this title comes from the manifest, so we know it worked
        assert top_result.title == "Frederick Bates (Wikipedia article)"


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError", "httpx.RemoteProtocolError"])
@pytest.mark.parametrize("agent_type", [FAKE_AGENT_TYPE, ToolSelector, SimpleAgent])
@pytest.mark.asyncio
async def test_agent_types(
    agent_test_settings: Settings, agent_type: str | type
) -> None:
    question = "How can you use XAI for chemical property prediction?"

    # make sure agent_llm is different from default, so we can correctly track tokens
    # for agent
    agent_test_settings.agent.agent_llm = "gpt-4o-2024-08-06"
    agent_test_settings.llm = "gpt-4o-mini"
    agent_test_settings.summary_llm = "gpt-4o-mini"
    agent_test_settings.agent.agent_prompt += (
        "\n\nCall each tool once in appropriate order and"
        " accept the answer for now, as we're in debug mode."
    )
    request = QueryRequest(query=question, settings=agent_test_settings)
    with patch.object(
        Index, "open", side_effect=Index.open, autospec=True
    ) as mock_open:
        response = await agent_query(request, agent_type=agent_type)
    assert (
        mock_open.call_count <= 1
    ), "Expected one Index.open call, or possibly zero if multiprocessing tests"
    assert response.session.answer, "Answer not generated"
    assert response.session.answer != "I cannot answer", "Answer not generated"
    assert response.session.context, "No contexts were found"
    assert response.session.question == question
    agent_llm = request.settings.agent.agent_llm
    # TODO: once LDP can track tokens, we can remove this check
    if agent_type not in {FAKE_AGENT_TYPE, SimpleAgent}:
        assert (
            response.session.token_counts[agent_llm][0] > 1000
        ), "Expected many prompt tokens"
        assert (
            response.session.token_counts[agent_llm][1] > 50
        ), "Expected many completion tokens"
        assert response.session.cost > 0, "Expected nonzero cost"


@pytest.mark.asyncio
async def test_successful_memory_agent(agent_test_settings: Settings) -> None:
    tic = time.perf_counter()
    memory_id = "call_Wtmv95JbNcQ2nRQCZBoOfcJy"  # Stub value
    memory = Memory(
        query=(
            "Use the tools to answer the question: How can you use XAI for chemical"
            " property prediction?\n\nThe gen_answer tool output is visible to the"
            " user, so you do not need to restate the answer and can simply"
            " terminate if the answer looks sufficient. The current status of"
            " evidence/papers/cost is "
            f"{make_status(total_paper_count=0, relevant_paper_count=0, evidence_count=0, cost=0.0)}"  # Started 0
            "\n\nTool request message '' for tool calls: paper_search(query='XAI for"
            " chemical property prediction', min_year='2018', max_year='2024')"
            f" [id={memory_id}]\n\nTool response message '"
            f"{make_status(total_paper_count=2, relevant_paper_count=0, evidence_count=0, cost=0.0)}"  # Found 2
            f"' for tool call ID {memory_id} of tool 'paper_search'"
        ),
        input=(
            "Use the tools to answer the question: How can you use XAI for chemical"
            " property prediction?\n\nThe gen_answer tool output is visible to the"
            " user, so you do not need to restate the answer and can simply terminate"
            " if the answer looks sufficient. The current status of"
            " evidence/papers/cost is "
            f"{make_status(total_paper_count=0, relevant_paper_count=0, evidence_count=0, cost=0.0)}"
        ),
        output=(
            "Tool request message '' for tool calls: paper_search(query='XAI for"
            " chemical property prediction', min_year='2018', max_year='2024')"
            f" [id={memory_id}]"
        ),
        value=5.0,  # Stub value
        template="Input: {input}\n\nOutput: {output}\n\nDiscounted Reward: {value}",
    )
    memory_model = UIndexMemoryModel(
        embedding_model=EmbeddingModel.from_name("text-embedding-3-small")
    )
    await memory_model.add_memory(memory)
    serialized_memory_model = memory_model.model_dump(exclude_none=True)
    query = QueryRequest(
        query="How can you use XAI for chemical property prediction?",
        settings=agent_test_settings,
    )
    # NOTE: use Claude 3 for its <thinking> feature, testing regex replacement of it
    query.settings.agent.agent_llm = "claude-3-5-sonnet-20240620"
    query.settings.agent.agent_config = {
        "memories": serialized_memory_model.pop("memories"),
        "memory_model": serialized_memory_model,
    }

    thoughts: list[str] = []
    orig_llm_model_call = MultipleCompletionLLMModel.call

    async def on_agent_action(action: OpResult[ToolRequestMessage], *_) -> None:
        thoughts.append(extract_thought(content=action.value.content))

    async def llm_model_call(*args, **kwargs):
        # NOTE: "required" will not lead to thoughts being emitted, it has to be "auto"
        # https://docs.anthropic.com/en/docs/build-with-claude/tool-use#chain-of-thought
        kwargs.pop("tool_choice", MultipleCompletionLLMModel.TOOL_CHOICE_REQUIRED)
        return await orig_llm_model_call(*args, tool_choice="auto", **kwargs)  # type: ignore[misc]

    with patch.object(
        MultipleCompletionLLMModel, "call", side_effect=llm_model_call, autospec=True
    ):
        response = await agent_query(
            query,
            Docs(),
            agent_type=f"{ldp.agent.__name__}.{MemoryAgent.__name__}",
            on_agent_action_callback=on_agent_action,
        )
    assert response.status == AgentStatus.SUCCESS, "Agent did not succeed"
    assert (
        time.perf_counter() - tic <= query.settings.agent.timeout
    ), "Agent should not have timed out"
    assert all(thought and "<thinking>" not in thought for thought in thoughts)


@pytest.mark.parametrize("agent_type", [ToolSelector, SimpleAgent])
@pytest.mark.asyncio
async def test_timeout(agent_test_settings: Settings, agent_type: str | type) -> None:
    agent_test_settings.prompts.pre = None
    agent_test_settings.agent.timeout = 0.001
    agent_test_settings.llm = "gpt-4o-mini"
    agent_test_settings.agent.tool_names = {"gen_answer"}
    response = await agent_query(
        QueryRequest(
            query="Are COVID-19 vaccines effective?", settings=agent_test_settings
        ),
        agent_type=agent_type,
    )
    # ensure that GenerateAnswerTool was called
    assert response.status == AgentStatus.TRUNCATED, "Agent did not timeout"
    assert "I cannot answer" in response.session.answer


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
@pytest.mark.asyncio
async def test_propagate_options(agent_test_settings: Settings) -> None:
    llm_name = "gpt-4o-mini"
    default_llm_names = {
        cls.model_fields[name].default
        for name, cls in itertools.product(("llm", "summary_llm"), (Settings,))
    }
    assert (
        llm_name not in default_llm_names
    ), f"Assertions require not matching a default LLM name in {default_llm_names}."

    agent_test_settings.llm = llm_name
    agent_test_settings.answer.answer_max_sources = 5
    agent_test_settings.answer.evidence_k = 6
    agent_test_settings.answer.answer_length = "400 words"
    agent_test_settings.prompts.pre = None
    agent_test_settings.prompts.system = "End all responses with ###"
    agent_test_settings.answer.evidence_skip_summary = True
    agent_test_settings.answer.evidence_detailed_citations = False

    query = QueryRequest(
        query="What is is a self-explanatory model?", settings=agent_test_settings
    )
    response = await agent_query(query, agent_type=FAKE_AGENT_TYPE)
    assert response.status == AgentStatus.SUCCESS, "Agent did not succeed"
    result = response.session
    assert len(result.answer) > 200, "Answer did not return any results"
    assert "###" in result.answer, "Answer did not propagate system prompt"
    assert (
        len(result.contexts[0].context) == agent_test_settings.parsing.chunk_size
    ), "Summary was not skipped"


@pytest.mark.asyncio
async def test_gather_evidence_rejects_empty_docs(
    agent_test_settings: Settings,
) -> None:
    # Patch GenerateAnswerTool._arun so that if this tool is chosen first, we
    # don't give a 'cannot answer' response. A 'cannot answer' response can
    # lead to an unsure status, which will break this test's assertions. Since
    # this test is about a GatherEvidenceTool edge case, defeating
    # GenerateAnswerTool is fine
    original_doc = GenerateAnswer.gen_answer.__doc__
    with patch.object(
        GenerateAnswer,
        "gen_answer",
        return_value="Failed to answer question.",
        autospec=True,
    ) as mock_gen_answer:
        mock_gen_answer.__doc__ = original_doc
        agent_test_settings.agent = AgentSettings(
            tool_names={"gather_evidence", "gen_answer"},
            max_timesteps=3,
            search_count=agent_test_settings.agent.search_count,
            index=IndexSettings(
                paper_directory=agent_test_settings.agent.index.paper_directory,
                index_directory=agent_test_settings.agent.index.index_directory,
            ),
        )
        response = await agent_query(
            query=QueryRequest(
                query="Are COVID-19 vaccines effective?", settings=agent_test_settings
            ),
            docs=Docs(),
        )
    assert (
        response.status == AgentStatus.TRUNCATED
    ), "Agent should have hit its max timesteps"


@pytest.mark.parametrize("callback_type", [None, "async"])
@pytest.mark.flaky(reruns=3, only_rerun=["AssertionError", "EmptyDocsError"])
@pytest.mark.asyncio
async def test_agent_sharing_state(
    agent_test_settings: Settings, subtests: SubTests, callback_type: str
) -> None:
    agent_test_settings.agent.search_count = 3  # Keep low for speed
    agent_test_settings.answer.evidence_k = 2
    agent_test_settings.answer.answer_max_sources = 1
    llm_model = agent_test_settings.get_llm()
    summary_llm_model = agent_test_settings.get_summary_llm()
    embedding_model = agent_test_settings.get_embedding_model()

    callbacks = {}
    if callback_type == "async":
        gen_answer_initialized_callback = AsyncMock()
        gen_answer_completed_callback = AsyncMock()
        gather_evidence_initialized_callback = AsyncMock()
        gather_evidence_completed_callback = AsyncMock()

        callbacks = {
            "gen_answer_initialized": [gen_answer_initialized_callback],
            "gen_answer_completed": [gen_answer_completed_callback],
            "gather_evidence_initialized": [gather_evidence_initialized_callback],
            "gather_evidence_completed": [gather_evidence_completed_callback],
        }

    agent_test_settings.agent.callbacks = callbacks  # type: ignore[assignment]

    answer = PQASession(question="What is is a self-explanatory model?")
    query = QueryRequest(query=answer.question, settings=agent_test_settings)
    env_state = EnvironmentState(docs=Docs(), answer=answer)
    built_index = await get_directory_index(settings=agent_test_settings)
    assert await built_index.count, "Index build did not work"

    with subtests.test(msg=PaperSearch.__name__):
        search_tool = PaperSearch(
            settings=agent_test_settings, embedding_model=embedding_model
        )
        with (
            patch.object(
                SearchIndex, "save_index", wraps=SearchIndex.save_index, autospec=True
            ) as mock_save_index,
            patch.object(
                Index, "open", side_effect=Index.open, autospec=True
            ) as mock_open,
        ):
            await search_tool.paper_search(
                "XAI self explanatory model",
                min_year=None,
                max_year=None,
                state=env_state,
            )
            assert env_state.docs.docs, "Search did not add any papers"
            assert (
                mock_open.call_count <= 1
            ), "Expected one Index.open call, or possibly zero if multiprocessing tests"
            assert all(
                isinstance(d, Doc) for d in env_state.docs.docs.values()
            ), "Document type or DOI propagation failure"

            await search_tool.paper_search(
                "XAI for chemical property prediction",
                min_year=2018,
                max_year=2024,
                state=env_state,
            )
            assert (
                mock_open.call_count <= 1
            ), "Expected one Index.open call, or possibly zero if multiprocessing tests"
            mock_save_index.assert_not_awaited()

    with subtests.test(msg=GatherEvidence.__name__):
        assert not answer.contexts, "No contexts is required for a later assertion"

        gather_evidence_tool = GatherEvidence(
            settings=agent_test_settings,
            summary_llm_model=summary_llm_model,
            embedding_model=embedding_model,
        )
        await gather_evidence_tool.gather_evidence(answer.question, state=env_state)

        if callback_type == "async":
            gather_evidence_initialized_callback.assert_awaited_once_with(env_state)
            gather_evidence_completed_callback.assert_awaited_once_with(env_state)

        assert answer.contexts, "Evidence did not return any results"

    with subtests.test(msg=f"{GenerateAnswer.__name__} working"):
        generate_answer_tool = GenerateAnswer(
            settings=agent_test_settings,
            llm_model=llm_model,
            summary_llm_model=summary_llm_model,
            embedding_model=embedding_model,
        )
        result = await generate_answer_tool.gen_answer(answer.question, state=env_state)

        if callback_type == "async":
            gen_answer_initialized_callback.assert_awaited_once_with(env_state)
            gen_answer_completed_callback.assert_awaited_once_with(env_state)

        assert re.search(
            pattern=EnvironmentState.STATUS_SEARCH_REGEX_PATTERN, string=result
        )
        assert len(answer.answer) > 200, "Answer did not return any results"
        assert (
            GenerateAnswer.extract_answer_from_message(result) == answer.answer
        ), "Failed to regex extract answer from result"
        assert (
            len(answer.used_contexts) <= query.settings.answer.answer_max_sources
        ), "Answer has more sources than expected"


def test_settings_model_config() -> None:
    settings_name = "tier1_limits"
    settings = Settings.from_name(settings_name)
    assert (
        settings.embedding_config
    ), "Test assertions are only effective if there's something to configure"

    with Path(
        str(importlib.resources.files("paperqa.configs") / f"{settings_name}.json")
    ).open() as f:
        raw_settings = json.loads(f.read())

    llm_model = settings.get_llm()
    summary_llm_model = settings.get_summary_llm()
    embedding_model = settings.get_embedding_model()
    assert (
        llm_model.config["rate_limit"]["gpt-4o"]
        == raw_settings["llm_config"]["rate_limit"]["gpt-4o"]
    )
    assert (
        summary_llm_model.config["rate_limit"]["gpt-4o"]
        == raw_settings["summary_llm_config"]["rate_limit"]["gpt-4o"]
    )
    assert (
        embedding_model.config["rate_limit"]
        == raw_settings["embedding_config"]["rate_limit"]
    )


def test_tool_schema(agent_test_settings: Settings) -> None:
    """Check the tool schema passed to LLM providers."""
    tools = settings_to_tools(agent_test_settings)
    assert ToolsAdapter.dump_python(tools, exclude_none=True) == [
        {
            "type": "function",
            "info": {
                "name": "gather_evidence",
                "description": (
                    "Gather evidence from previous papers given a specific question"
                    " to increase evidence and relevant paper counts.\n\nA valuable"
                    " time to invoke this tool is right after another tool"
                    " increases paper count.\nFeel free to invoke this tool in"
                    " parallel with other tools, but do not call this tool in"
                    " parallel with itself.\nOnly invoke this tool when the paper"
                    " count is above zero, or this tool will be useless."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Specific question to gather evidence for.",
                            "title": "Question",
                        }
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "info": {
                "name": "paper_search",
                "description": (
                    "Search for papers to increase the paper count.\n\nRepeat"
                    " previous calls with the same query and years to continue a"
                    " search. Only repeat a maximum of twice.\nThis tool can be"
                    " called concurrently.\nThis tool"
                    " introduces novel papers, so invoke this tool when just"
                    " beginning or when unsatisfied with the current evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "A search query, which can be a specific phrase,"
                                " complete sentence, or general keywords, e.g."
                                " 'machine learning for immunology'. Also can be"
                                " given search operators."
                            ),
                            "title": "Query",
                        },
                        "min_year": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "description": (
                                "Filter for minimum publication year, or None for"
                                " no minimum year. The current year is"
                                f" {get_year()}."
                            ),
                            "title": "Min Year",
                        },
                        "max_year": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "description": (
                                "Filter for maximum publication year, or None for"
                                " no maximum year. The current year is"
                                f" {get_year()}."
                            ),
                            "title": "Max Year",
                        },
                    },
                    "required": ["query", "min_year", "max_year"],
                },
            },
        },
        {
            "type": "function",
            "info": {
                "name": "gen_answer",
                "description": (
                    "Ask a model to propose an answer using current"
                    " evidence.\n\nThe tool may fail, indicating that better or"
                    " different evidence should be found.\nAim for at least five"
                    " pieces of evidence from multiple sources before invoking this"
                    " tool.\nFeel free to invoke this tool in parallel with other"
                    " tools, but do not call this tool in parallel with itself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question to be answered.",
                            "title": "Question",
                        }
                    },
                    "required": ["question"],
                },
            },
        },
    ]


def test_query_request_docs_name_serialized() -> None:
    """Test that the query request has a docs_name property."""
    request = QueryRequest(query="Are COVID-19 vaccines effective?")
    request_data = json.loads(request.model_dump_json())
    assert "docs_name" in request_data
    assert request_data["docs_name"] is None
    request.set_docs_name("my_doc")
    request_data = json.loads(request.model_dump_json())
    assert request_data["docs_name"] == "my_doc"


def test_answers_are_striped() -> None:
    """Test that answers are striped."""
    session = PQASession(
        question="What is the meaning of life?",
        contexts=[
            Context(
                context="bla",
                text=Text(
                    name="text",
                    text="The meaning of life is 42.",
                    embedding=[43.3, 34.2],
                    doc=Doc(
                        docname="foo",
                        citation="bar",
                        dockey="baz",
                        embedding=[43.1, 65.2],
                    ),
                ),
                score=3,
            )
        ],
    )
    response = AnswerResponse(session=session, bibtex={}, status="success")

    assert response.session.contexts[0].text.embedding is None
    assert response.session.contexts[0].text.text == ""  # type: ignore[unreachable,unused-ignore]
    assert response.session.contexts[0].text.doc is not None
    assert response.session.contexts[0].text.doc.embedding is None
    # make sure it serializes
    response.model_dump_json()


@pytest.mark.parametrize(
    ("kwargs", "result"),
    [
        ({}, None),
        ({"tool_names": {GenerateAnswer.TOOL_FN_NAME}}, None),
        ({"tool_names": set()}, ValidationError),
        ({"tool_names": {PaperSearch.TOOL_FN_NAME}}, ValidationError),
    ],
)
def test_agent_prompt_collection_validations(
    kwargs: dict[str, Any], result: type[Exception] | None
) -> None:
    if result is None:
        AgentSettings(**kwargs)
    else:
        with pytest.raises(result):
            AgentSettings(**kwargs)


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
@pytest.mark.asyncio
async def test_deepcopy_env(agent_test_settings: Settings) -> None:
    await get_directory_index(settings=agent_test_settings)  # Trigger build

    question = "How can you use XAI for chemical property prediction?"
    env = GradablePaperQAEnvironment(
        query=QueryRequest(query=question, settings=agent_test_settings),
        docs=Docs(),
    )

    # 1. Rollout until after gather evidence
    await env.reset()
    for tool_call in (
        ToolCall.from_name(
            "paper_search",
            query="XAI for chemical property prediction",
            min_year=2018,
            max_year=2024,
        ),
        ToolCall.from_name("gather_evidence", question=question),
    ):
        await env.step(ToolRequestMessage(tool_calls=[tool_call]))

    # 2. Now we deepcopy the environment
    env_copy = deepcopy(env)
    assert env.state == env_copy.state

    # 3. Generate an answer for both, and confirm they are identical
    gen_answer_action = ToolRequestMessage(
        tool_calls=[ToolCall.from_name("gen_answer", question=question)]
    )
    _, _, done, _ = await env.step(gen_answer_action)
    assert done
    assert not env.state.session.could_not_answer
    assert env.state.session.used_contexts
    _, _, done, _ = await env_copy.step(gen_answer_action)
    assert done
    assert not env_copy.state.session.could_not_answer
    assert env_copy.state.session.used_contexts
    assert sorted(env.state.session.used_contexts) == sorted(
        env_copy.state.session.used_contexts
    )
