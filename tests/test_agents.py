from __future__ import annotations

import importlib
import itertools
import json
import logging
import re
import shutil
import tempfile
import time
import zlib
from functools import wraps
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import ldp.agent
import pytest
from aviary.core import Tool, ToolRequestMessage, ToolsAdapter, ToolSelector
from ldp.agent import MemoryAgent, SimpleAgent
from ldp.graph.memory import Memory, UIndexMemoryModel
from ldp.graph.ops import OpResult
from lmi import CommonLLMNames, EmbeddingModel, LiteLLMModel
from pytest_subtests import SubTests
from tantivy import Index
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt

from paperqa.agents import SearchIndex, agent_query
from paperqa.agents.env import (
    CLINICAL_STATUS_SEARCH_REGEX_PATTERN,
    clinical_trial_status,
    settings_to_tools,
)
from paperqa.agents.main import FAKE_AGENT_TYPE, run_agent
from paperqa.agents.models import AgentStatus, AnswerResponse
from paperqa.agents.search import (
    FAILED_DOCUMENT_ADD_ID,
    get_directory_index,
    maybe_get_manifest,
)
from paperqa.agents.tools import (
    ClinicalTrialsSearch,
    Complete,
    EnvironmentState,
    GatherEvidence,
    GenerateAnswer,
    PaperSearch,
    Reset,
    make_status,
)
from paperqa.docs import Docs
from paperqa.prompts import CANNOT_ANSWER_PHRASE, CONTEXT_INNER_PROMPT_NOT_DETAILED
from paperqa.settings import AgentSettings, IndexSettings, Settings
from paperqa.types import Context, Doc, PQASession, Text
from paperqa.utils import encode_id, extract_thought, get_year, md5sum


@pytest.mark.asyncio
async def test_get_directory_index(
    subtests: SubTests, agent_test_settings: Settings
) -> None:
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
        # bates.txt + empty.txt + flag_day.html + gravity_hill.md + obama.txt + paper.pdf,
        # but empty.txt fails to be added
        path_to_id = await index.index_files
        assert (
            sum(id_ != FAILED_DOCUMENT_ADD_ID for id_ in path_to_id.values()) == 5
        ), "Incorrect number of parsed index files"

        with subtests.test(msg="check-txt-query"):
            results = await index.query(query="who is Frederick Bates?", min_score=5)
            assert results
            target_doc_path = (paper_dir / "bates.txt").absolute()
            assert results[0].docs.keys() == {md5sum(target_doc_path)}, (
                f"Expected to find {target_doc_path.name!r}, got citations"
                f" {[d.formatted_citation for d in results[0].docs.values()]}."
            )

        with subtests.test(msg="check-md-query"):
            results = await index.query(query="what is a gravity hill?", min_score=5)
            assert results
            first_result = results[0]
            target_doc_path = (paper_dir / "gravity_hill.md").absolute()
            expected_ids = {
                md5sum(target_doc_path),  # What we actually expect
                encode_id(
                    "10.2307/j.ctt5vkfh7.11"  # Crossref may match this Gravity Hill poem, lol
                ),
            }
            for expected_id in expected_ids:
                if expected_id in set(first_result.docs.keys()):
                    break
            else:
                raise AssertionError(
                    f"Failed to match an ID in {expected_ids}, got citations"
                    f" {[d.formatted_citation for d in first_result.docs.values()]}."
                )
            assert all(
                x in first_result.docs[expected_id].formatted_citation
                for x in ("Wikipedia", "Gravity")
            )

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
            for x in cast("Path", index_settings.paper_directory).iterdir()
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
    for attempt in Retrying(
        stop=stop_after_attempt(3),
        # zlib.error: Error -5 while decompressing data: incomplete or truncated stream
        retry=retry_if_exception_type(zlib.error),
    ):
        with (
            attempt,
            patch.object(
                Docs, "aadd", autospec=True, side_effect=Docs.aadd
            ) as mock_aadd,
        ):
            index = await get_directory_index(settings=agent_test_settings)

    assert len(await index.index_files) == num_source_files
    assert (
        mock_aadd.await_count < num_source_files
    ), "Should not rebuild the whole index"


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
    "gravity_hill.md",
    "obama.txt",
    "paper.pdf",
}


@pytest.mark.asyncio
async def test_get_directory_index_w_manifest(agent_test_settings: Settings) -> None:
    # Set the paper_directory to be a relative path as starting point to confirm this
    # won't trip us up, and set the manifest file too
    abs_paper_dir = cast("Path", agent_test_settings.agent.index.paper_directory)
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

        # note: we get every possible field from the manifest constructed in maybe_get_manifest,
        # and then DocDetails construction sets the dockey to the doc_id.
        assert top_result.dockey == top_result.doc_id
        # note: this title comes from the manifest, so we know it worked
        assert top_result.title == "Frederick Bates (Wikipedia article)"

        assert "wikipedia article" in top_result.citation.lower(), (
            "Other tests check we can override citation,"
            " so here we check here it's actually populated"
        )


@pytest.mark.asyncio
async def test_get_directory_index_w_no_citations(
    agent_test_settings: Settings,
) -> None:
    agent_test_settings.agent.index.manifest_file = "stub_manifest_nocitation.csv"
    index = await get_directory_index(settings=agent_test_settings)

    results = await index.query(query="who is Frederick Bates?")
    top_result = next(iter(results[0].docs.values()))

    assert not top_result.citation


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError", "httpx.RemoteProtocolError"])
@pytest.mark.parametrize("agent_type", [FAKE_AGENT_TYPE, ToolSelector, SimpleAgent])
@pytest.mark.parametrize("llm_name", ["gpt-4o", "gemini/gemini-1.5-flash"])
@pytest.mark.asyncio
async def test_agent_types(
    agent_test_settings: Settings, agent_type: str | type, llm_name: str
) -> None:
    question = "How can you use XAI for chemical property prediction?"

    # make sure agent_llm is different from default, so we can correctly track tokens
    # for agent
    agent_test_settings.agent.agent_llm = llm_name
    agent_test_settings.llm = "gpt-4o-mini"
    agent_test_settings.summary_llm = "gpt-4o-mini"
    agent_test_settings.agent.agent_prompt += (
        "\n\nCall each tool once in appropriate order and"
        " accept the answer for now, as we're in debug mode."
    )
    with patch.object(
        Index, "open", side_effect=Index.open, autospec=True
    ) as mock_open:
        response = await agent_query(
            question, agent_test_settings, agent_type=agent_type
        )
    assert (
        mock_open.call_count <= 1
    ), "Expected one Index.open call, or possibly zero if multiprocessing tests"
    assert response.session.answer, "Answer not generated"
    assert response.session.answer != CANNOT_ANSWER_PHRASE, "Answer not generated"
    assert response.session.context, "No contexts were found"
    assert response.session.question == question
    agent_llm = agent_test_settings.agent.agent_llm
    # TODO: once LDP can track tokens, we can remove this check
    if agent_type not in {FAKE_AGENT_TYPE, SimpleAgent}:
        assert (
            response.session.token_counts[agent_llm][0] > 500
        ), "Expected many prompt tokens"
        assert (
            response.session.token_counts[agent_llm][1] > 30
        ), "Expected many completion tokens"
        assert response.session.cost > 0, "Expected nonzero cost"


@pytest.mark.asyncio
async def test_successful_memory_agent(agent_test_settings: Settings) -> None:
    tic = time.perf_counter()
    memory_id = "call_Wtmv95JbNcQ2nRQCZBoOfcJy"  # Stub value
    memory = Memory(
        query=(
            "Use the tools to answer the question: How can you use XAI for chemical"
            " property prediction?\n\nWhen the answer looks sufficient,"
            " you can terminate by calling the {complete_tool_name} tool."
            " If the answer does not look sufficient,"
            " and you have already tried to answer several times,"
            " you can terminate by calling the {complete_tool_name} tool."
            " The current status of evidence/papers/cost is "
            f"{make_status(total_paper_count=0, relevant_paper_count=0, evidence_count=0, cost=0.0)}"  # Started 0  # noqa: E501
            "\n\nTool request message '' for tool calls: paper_search(query='XAI for"
            " chemical property prediction', min_year='2018', max_year='2024')"
            f" [id={memory_id}]\n\nTool response message '"
            f"{make_status(total_paper_count=2, relevant_paper_count=0, evidence_count=0, cost=0.0)}"  # Found 2  # noqa: E501
            f"' for tool call ID {memory_id} of tool 'paper_search'"
        ),
        input=(
            "Use the tools to answer the question: How can you use XAI for chemical"
            " property prediction?\n\nWhen the answer looks sufficient,"
            " you can terminate by calling the {complete_tool_name} tool."
            " If the answer does not look sufficient,"
            " and you have already tried to answer several times,"
            " you can terminate by calling the {complete_tool_name} tool."
            " The current status of evidence/papers/cost is "
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
    query = "How can you use XAI for chemical property prediction?"
    # NOTE: use Claude 3 for its <thinking> feature, testing regex replacement of it
    agent_test_settings.agent.agent_llm = CommonLLMNames.CLAUDE_35_SONNET.value
    agent_test_settings.agent.agent_config = {
        "memories": serialized_memory_model.pop("memories"),
        "memory_model": serialized_memory_model,
    }

    thoughts: list[str] = []
    orig_llm_model_call = LiteLLMModel.call

    async def on_agent_action(  # noqa: RUF029
        action: OpResult[ToolRequestMessage], *_
    ) -> None:
        thoughts.append(extract_thought(content=action.value.content))

    async def llm_model_call(*args, **kwargs):
        # NOTE: "required" will not lead to thoughts being emitted, it has to be "auto"
        # https://docs.anthropic.com/en/docs/build-with-claude/tool-use#chain-of-thought
        args = args[:-1]  # removing last element (tool_choice) from args
        return await orig_llm_model_call(*args, tool_choice="auto", **kwargs)  # type: ignore[misc]

    with patch.object(LiteLLMModel, "call", side_effect=llm_model_call, autospec=True):
        response = await agent_query(
            query,
            agent_test_settings,
            Docs(),
            agent_type=f"{ldp.agent.__name__}.{MemoryAgent.__name__}",
            on_agent_action_callback=on_agent_action,
        )
    assert response.status == AgentStatus.SUCCESS, "Agent did not succeed"
    assert (
        time.perf_counter() - tic <= agent_test_settings.agent.timeout
    ), "Agent should not have timed out"
    assert all(thought and "<thinking>" not in thought for thought in thoughts)


@pytest.mark.parametrize("agent_type", [ToolSelector, SimpleAgent])
@pytest.mark.asyncio
async def test_timeout(agent_test_settings: Settings, agent_type: str | type) -> None:
    agent_test_settings.prompts.pre = None
    agent_test_settings.agent.timeout = 0.001
    agent_test_settings.llm = "gpt-4o-mini"
    agent_test_settings.agent.tool_names = {"gen_answer", "complete"}
    response = await agent_query(
        query="Are COVID-19 vaccines effective?",
        settings=agent_test_settings,
        agent_type=agent_type,
    )
    # ensure that GenerateAnswerTool was called
    assert response.status == AgentStatus.TRUNCATED, "Agent did not timeout"
    assert CANNOT_ANSWER_PHRASE in response.session.answer


@pytest.mark.flaky(reruns=5, only_rerun=["AssertionError"])
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
    agent_test_settings.prompts.context_inner = CONTEXT_INNER_PROMPT_NOT_DETAILED
    agent_test_settings.answer.evidence_skip_summary = True

    response = await agent_query(
        query="What is is a self-explanatory model?",
        settings=agent_test_settings,
        agent_type=FAKE_AGENT_TYPE,
    )
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

    @wraps(GenerateAnswer.gen_answer)
    async def gen_answer(self, state) -> str:  # noqa: ARG001, RUF029
        return f"{CANNOT_ANSWER_PHRASE}."

    # Patch GenerateAnswerTool.gen_answer so that if this tool is chosen first,
    # we keep running until we get truncated
    with (
        patch(
            "paperqa.agents.env.settings_to_tools",
            side_effect=[
                [
                    Tool.from_function(
                        GatherEvidence(
                            settings=agent_test_settings,
                            summary_llm_model=agent_test_settings.get_summary_llm(),
                            embedding_model=agent_test_settings.get_embedding_model(),
                        ).gather_evidence
                    ),
                    Tool.from_function(
                        GenerateAnswer(
                            settings=agent_test_settings,
                            llm_model=agent_test_settings.get_llm(),
                            summary_llm_model=agent_test_settings.get_summary_llm(),
                            embedding_model=agent_test_settings.get_embedding_model(),
                        ).gen_answer
                    ),
                ]
            ],
        ),
        patch.object(GenerateAnswer, "gen_answer", gen_answer),
    ):
        agent_test_settings.agent = AgentSettings(
            max_timesteps=3,
            search_count=agent_test_settings.agent.search_count,
            index=IndexSettings(
                paper_directory=agent_test_settings.agent.index.paper_directory,
                index_directory=agent_test_settings.agent.index.index_directory,
            ),
        )
        response = await agent_query(
            query="Are COVID-19 vaccines effective?",
            settings=agent_test_settings,
            docs=Docs(),
        )
    assert (
        response.status == AgentStatus.TRUNCATED
    ), "Agent should have hit its max timesteps"


@pytest.mark.parametrize("callback_type", [None, "async"])
@pytest.mark.flaky(reruns=3, only_rerun=["AssertionError", "EmptyDocsError"])
@pytest.mark.asyncio
async def test_agent_sharing_state(
    agent_test_settings: Settings, subtests: SubTests, callback_type: str | None
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

    agent_test_settings.agent.callbacks = callbacks

    session = PQASession(question="What is is a self-explanatory model?")
    env_state = EnvironmentState(docs=Docs(), session=session)
    built_index = await get_directory_index(settings=agent_test_settings)
    assert await built_index.count, "Index build did not work"

    with subtests.test(msg="Custom and default environment status"):
        assert re.search(
            pattern=EnvironmentState.STATUS_SEARCH_REGEX_PATTERN,
            string=env_state.status,
        ), "Default status not formatted correctly"
        # override the status function with a new one

        def new_status(state: EnvironmentState) -> str:
            return f"Custom status: paper count = {len(state.docs.docs)}"

        env_state.status_fn = new_status
        assert env_state.status == new_status(
            env_state
        ), "Custom status not set correctly."
        env_state.status_fn = None

    # run an initial complete tool to see that the answer object is populated by it
    # this simulates if no gen_answer tool was called
    with subtests.test(msg=Complete.__name__):
        complete_tool = Complete()
        await complete_tool.complete(state=env_state, has_successful_answer=False)
        assert (
            env_state.session.answer == Complete.NO_ANSWER_PHRASE
        ), "Complete did not succeed"
        # now we wipe the answer for further tests
        env_state.session.answer = ""

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
        assert not session.contexts, "No contexts is required for a later assertion"

        gather_evidence_tool = GatherEvidence(
            settings=agent_test_settings,
            summary_llm_model=summary_llm_model,
            embedding_model=embedding_model,
        )

        response = await gather_evidence_tool.gather_evidence(
            session.question, state=env_state
        )

        if callback_type == "async":
            gather_evidence_initialized_callback.assert_awaited_once_with(env_state)
            gather_evidence_completed_callback.assert_awaited_once_with(env_state)

        split = re.split(
            r"(\d+) pieces of evidence, (\d+) of which were relevant",
            response,
            maxsplit=1,
        )
        assert len(split) == 4, "Unexpected response shape"
        total_added_1, relevant_added_1 = int(split[1]), int(split[2])
        assert all(
            x >= 0 for x in (total_added_1, relevant_added_1)
        ), "Expected non-negative counts"
        assert len(env_state.get_relevant_contexts()) == relevant_added_1
        # ensure 1 piece of top evidence is returned
        assert "\n1." in response, "gather_evidence did not return any results"
        assert (
            "\n2." not in response
        ), "gather_evidence should return only 1 context, not 2"

        # now adjust to give the agent 2x pieces of evidence
        gather_evidence_tool.settings.agent.agent_evidence_n = 2
        response = await gather_evidence_tool.gather_evidence(
            session.question, state=env_state
        )

        split = re.split(
            r"(\d+) pieces of evidence, (\d+) of which were relevant",
            response,
            maxsplit=1,
        )
        assert len(split) == 4, "Unexpected response shape"
        total_added_2, relevant_added_2 = int(split[1]), int(split[2])
        assert all(
            x >= 0 for x in (total_added_2, relevant_added_2)
        ), "Expected non-negative counts"
        assert (
            len(env_state.get_relevant_contexts())
            == relevant_added_1 + relevant_added_2
        )
        # ensure both evidences are returned
        assert "\n1." in response, "gather_evidence did not return any results"
        assert "\n2." in response, "gather_evidence should return 2 contexts"

        assert session.contexts, "Evidence did not return any results"
        assert not session.answer, "Expected no answer yet"

    with subtests.test(msg=f"{GenerateAnswer.__name__} working"):
        generate_answer_tool = GenerateAnswer(
            settings=agent_test_settings,
            llm_model=llm_model,
            summary_llm_model=summary_llm_model,
            embedding_model=embedding_model,
        )
        result = await generate_answer_tool.gen_answer(state=env_state)

        if callback_type == "async":
            gen_answer_initialized_callback.assert_awaited_once_with(env_state)
            gen_answer_completed_callback.assert_awaited_once_with(env_state)

        assert re.search(
            pattern=EnvironmentState.STATUS_SEARCH_REGEX_PATTERN, string=result
        )
        assert len(session.answer) > 200, "Answer did not return any results"
        assert (
            GenerateAnswer.extract_answer_from_message(result) == session.answer
        ), "Failed to regex extract answer from result"
        assert (
            len(session.used_contexts) <= agent_test_settings.answer.answer_max_sources
        ), "Answer has more sources than expected"

    with subtests.test(msg=f"{Reset.__name__} working"):
        reset_tool = Reset()
        await reset_tool.reset(state=env_state)
        assert not session.context
        assert not session.contexts


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
                "name": "reset",
                "description": (
                    "Reset by clearing all current evidence from the system."
                    "\n\nThis tool is useful when repeatedly failing to answer because"
                    " the existing evidence may unsuitable for the question.\nIt does"
                    " not make sense to call this tool in parallel with other tools,"
                    " as its resetting all state.\n"
                    "Only invoke this tool when the current evidence is above"
                    " zero, or this tool will be useless."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "info": {
                "name": "gen_answer",
                "description": (
                    "Generate an answer using current evidence.\n\nThe tool may fail,"
                    " indicating that better or different evidence should be"
                    " found.\nAim for at least five pieces of evidence from multiple"
                    " sources before invoking this tool.\nFeel free to invoke this tool"
                    " in parallel with other tools, but do not call this tool in"
                    " parallel with itself."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
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
            "info": {
                "description": (
                    "Terminate using the last proposed answer.\n\nDo not invoke this"
                    " tool in parallel with other tools or itself."
                ),
                "name": "complete",
                "parameters": {
                    "properties": {
                        "has_successful_answer": {
                            "description": (
                                "Set True if an answer that addresses all parts of the"
                                " task has been generated, otherwise set False to"
                                " indicate unsureness."
                            ),
                            "title": "Has Successful Answer",
                            "type": "boolean",
                        }
                    },
                    "required": ["has_successful_answer"],
                    "type": "object",
                },
            },
            "type": "function",
        },
    ]


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
    response = AnswerResponse(session=session, bibtex={}, status=AgentStatus.SUCCESS)

    assert response.session.contexts[0].text.embedding is None
    assert not response.session.contexts[0].text.text
    assert response.session.contexts[0].text.doc is not None
    assert response.session.contexts[0].text.doc.embedding is None
    # make sure it serializes
    response.model_dump_json()


@pytest.mark.asyncio
async def test_clinical_tool_usage(agent_test_settings) -> None:
    agent_test_settings.llm = "gpt-4o"
    agent_test_settings.summary_llm = "gpt-4o"
    agent_test_settings.agent.tool_names = {
        "clinical_trials_search",
        "gather_evidence",
        "gen_answer",
        "complete",
    }
    docs = Docs()
    response = await run_agent(
        docs,
        query=(
            "What are the NCTIDs of clinical trials for depression that focus on health"
            " services research, are in phase 2, have no status type, and started in or"
            " after 2017?"
        ),
        settings=agent_test_settings,
    )
    # make sure the tool was used at least once
    assert any(
        ClinicalTrialsSearch.TOOL_FN_NAME in step
        for step in response.session.tool_history
    ), "ClinicalTrialsSearch was not used"
    # make sure some clinical trials are pulled in as contexts
    assert any(
        "ClinicalTrials.gov" in c.text.doc.citation for c in response.session.contexts
    ), "No clinical trials were put into contexts"


@pytest.mark.asyncio
async def test_search_pagination(agent_test_settings: Settings) -> None:
    """Test that pagination works correctly in SearchIndex.query()."""
    index = await get_directory_index(settings=agent_test_settings)

    page_size = 1

    page1_results = await index.query(query="test", top_n=page_size, offset=0)
    page2_results = await index.query(query="test", top_n=page_size, offset=page_size)
    page1and2_results = await index.query(query="test", top_n=2 * page_size, offset=0)

    assert (
        page1_results == page1and2_results[:page_size]
    ), "First page should match start of all results"
    assert (
        page2_results == page1and2_results[page_size : page_size * 2]
    ), "Second page should match second slice of all results"


@pytest.mark.asyncio
async def test_empty_index_without_index_rebuild(agent_test_settings: Settings):
    """Test that empty index and `rebuild_index=False` lead to a RuntimeError."""
    agent_test_settings.agent = AgentSettings(index=IndexSettings())  # empty index
    agent_test_settings.agent.rebuild_index = False
    with pytest.raises(RuntimeError, match=r"Index .* was empty, please rebuild it."):
        await agent_query(
            query="Are COVID-19 vaccines effective?",
            settings=agent_test_settings,
            agent_type=FAKE_AGENT_TYPE,
            force_index_rebuild=False,
        )


class TestClinicalTrialSearchTool:
    @pytest.mark.asyncio
    async def test_continuation(self) -> None:
        docs = Docs()
        state = EnvironmentState(
            docs=docs, session=PQASession(question=""), status_fn=clinical_trial_status
        )
        tool = ClinicalTrialsSearch(
            search_count=4,  # Keep low for speed
            settings=Settings(),
        )
        result = await tool.clinical_trials_search("Covid-19 vaccines", state)
        # 4 trials + the metadata context = 5
        assert len(state.docs.docs) == 5, "Search did not return enough trials"
        assert re.search(pattern=CLINICAL_STATUS_SEARCH_REGEX_PATTERN, string=result)
        match = re.search(r"Clinical Trial Count=(\d+)", result)
        assert match
        trial_count = int(match.group(1))
        assert trial_count == len(state.docs.docs)

        # Check continuation of the search
        result = await tool.clinical_trials_search("Covid-19 vaccines", state)
        assert len(state.docs.docs) > trial_count, "Search was unable to continue"


@pytest.mark.asyncio
async def test_index_build_concurrency(agent_test_settings: Settings) -> None:

    high_concurrency_settings = agent_test_settings.model_copy(deep=True)
    high_concurrency_settings.agent.index.name = "high_concurrency"
    high_concurrency_settings.agent.index.concurrency = 3
    high_concurrency_settings.agent.index.batch_size = 3
    with patch.object(
        SearchIndex, "save_index", side_effect=SearchIndex.save_index, autospec=True
    ) as mock_save_index:
        start_time = time.perf_counter()
        await get_directory_index(settings=high_concurrency_settings)
        high_concurrency_duration = time.perf_counter() - start_time
    high_batch_save_count = mock_save_index.call_count

    low_concurrency_settings = agent_test_settings.model_copy(deep=True)
    low_concurrency_settings.agent.index.name = "low_concurrency"
    low_concurrency_settings.agent.index.concurrency = 1
    low_concurrency_settings.agent.index.batch_size = 1
    with patch.object(
        SearchIndex, "save_index", side_effect=SearchIndex.save_index, autospec=True
    ) as mock_save_index:
        start_time = time.perf_counter()
        await get_directory_index(settings=low_concurrency_settings)
        low_concurrency_duration = time.perf_counter() - start_time
    low_batch_save_count = mock_save_index.call_count

    assert high_concurrency_duration * 1.1 < low_concurrency_duration, (
        "Expected high concurrency to be faster, but took"
        f" {high_concurrency_duration:.2f}s compared to {low_concurrency_duration:.2f}s"
    )
    assert high_batch_save_count < low_batch_save_count, (
        "Expected fewer save_index with high batch size, but got"
        f" {high_batch_save_count} vs {low_batch_save_count}"
    )
