from __future__ import annotations

import asyncio
import collections
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
from typing import ClassVar, cast
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import anyio
import ldp.agent
import litellm
import pytest
from aviary.core import (
    Environment,
    Message,
    Tool,
    ToolRequestMessage,
    ToolResponseMessage,
    ToolsAdapter,
    ToolSelector,
)
from aviary.envs.labbench import (
    GradablePaperQAEnvironment,
    ImageQAEnvironment,
    ImageQATaskDataset,
    LABBenchDatasets,
    TextQATaskDataset,
)
from aviary.utils import MultipleChoiceEvaluation, MultipleChoiceQuestion
from ldp.agent import Agent, MemoryAgent, SimpleAgent
from ldp.alg import (
    Evaluator,
    EvaluatorConfig,
    MeanMetricsCallback,
    StoreEnvironmentsCallback,
    StoreTrajectoriesCallback,
)
from ldp.graph.memory import Memory, UIndexMemoryModel
from ldp.graph.ops import OpResult
from lmi import CommonLLMNames, EmbeddingModel, LiteLLMModel
from lmi.utils import bytes_to_string, gather_with_concurrency
from paperqa_docling import parse_pdf_to_pages as docling_parse_pdf_to_pages
from paperqa_nemotron import parse_pdf_to_pages as nemotron_parse_pdf_to_pages
from paperqa_pymupdf import parse_pdf_to_pages as pymupdf_parse_pdf_to_pages
from pytest_subtests import SubTests
from rich.progress import track
from tantivy import Index
from tenacity import (
    Retrying,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)
from tqdm.asyncio import tqdm

from paperqa._ldp_shims import EnvsTaskDataset
from paperqa.agents import SearchIndex, agent_query
from paperqa.agents.env import (
    CLINICAL_STATUS_SEARCH_REGEX_PATTERN,
    PaperQAEnvironment,
    clinical_trial_status,
    settings_to_tools,
)
from paperqa.agents.main import FAKE_AGENT_TYPE, run_agent
from paperqa.agents.models import AgentStatus, AnswerResponse
from paperqa.agents.search import (
    FAILED_DOCUMENT_ADD_ID,
    fetch_kwargs_from_manifest,
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
from paperqa.clients.client_models import DOIOrTitleBasedProvider
from paperqa.docs import Docs
from paperqa.prompts import (
    CANNOT_ANSWER_PHRASE,
    CONTEXT_INNER_PROMPT_NOT_DETAILED,
    full_page_enrichment_prompt_template,
)
from paperqa.readers import PDFParserFn
from paperqa.settings import (
    AgentSettings,
    IndexSettings,
    MultimodalOptions,
    ParsingSettings,
    Settings,
)
from paperqa.types import Context, Doc, DocDetails, ParsedMedia, PQASession, Text
from paperqa.utils import compute_unique_doc_id, extract_thought, get_year, md5sum

logger = logging.getLogger(__name__)


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
        # bates.txt + empty.txt + flag_day.html + gravity_hill.md + influence.pdf
        # + obama.txt + paper.pdf + pasa.pdf + duplicate_media.pdf
        # + dummy.docx + dummy.pptx + dummy.xlsx,
        # but empty.txt fails to be added
        path_to_id = await index.index_files
        assert (
            sum(id_ != FAILED_DOCUMENT_ADD_ID for id_ in path_to_id.values()) == 12
        ), "Incorrect number of parsed index files"

        with subtests.test(msg="check-txt-query"):
            results = await index.query(query="who is Frederick Bates?", min_score=5)
            assert results
            target_doc_path = (paper_dir / "bates.txt").absolute()
            assert results[0].docs.keys() == {md5sum(target_doc_path)}, (
                f"Expected to find {target_doc_path.name!r}, got citations"
                f" {[d.formatted_citation for d in results[0].docs.values()]}."
            )

            # Check single quoted text in the query doesn't crash us
            results = await index.query(query="Who is 'Bates'")
            assert results

            # Check possessive in the query doesn't crash us
            results = await index.query(query="What is Bates' first name")
            assert results

        with subtests.test(msg="check-md-query"):
            results = await index.query(query="what is a gravity hill?", min_score=5)
            assert results
            first_result = results[0]
            assert len(first_result.docs) == 1, "Expected one result (gravity_hill.md)"
            target_doc_path = (paper_dir / "gravity_hill.md").absolute()
            expected_ids = {
                compute_unique_doc_id(
                    None,
                    md5sum(target_doc_path),  # What we actually expect
                ),
                compute_unique_doc_id(
                    "10.2307/j.ctt5vkfh7.11",  # Crossref may match this Gravity Hill poem, lol
                    next(iter(first_result.docs.values())).content_hash,
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
            assert (
                "gravity hill"
                in first_result.docs[expected_id].formatted_citation.lower()
            )

        # Check getting the same index name will not reprocess files
        with patch.object(Docs, "aadd") as mock_aadd:
            index = await get_directory_index(settings=agent_test_settings)
        assert len(await index.index_files) == len(path_to_id)
        mock_aadd.assert_not_awaited(), "Expected we didn't re-add files"

        # Now we actually remove (but not add!) a file from the paper directory,
        # and we still don't reprocess files
        (paper_dir / "obama.txt").unlink()
        with (
            patch.object(
                Docs, "aadd", autospec=True, side_effect=Docs.aadd
            ) as mock_aadd,
            patch.object(
                agent_test_settings.agent.index,
                "files_filter",
                lambda f: f.suffix in {".txt", ".pdf", ".md"},  # Also, exclude HTML
            ),
        ):
            index = await get_directory_index(settings=agent_test_settings)
        # Subtract 5 for the removed obama.txt,
        # dummy.docx, dummy_jap.docx, dummy.pptx, and dummy.xlsx files,
        # and another 1 for the filtered out flag_day.html
        assert len(await index.index_files) == len(path_to_id) - 5 - 1
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
            # Filter out .csv and .DS_Store files
            if x.suffix != ".csv" and agent_test_settings.agent.index.files_filter(x)
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
    "duplicate_media.pdf",
    "empty.txt",
    "flag_day.html",
    "gravity_hill.md",
    "obama.txt",
    "paper.pdf",
    "pasa.pdf",
    "dummy.docx",
    "dummy_jap.docx",
    "dummy.pptx",
    "dummy.xlsx",
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
    agent_test_settings: Settings,
    agent_type: str | type,
    llm_name: str,
    subtests: SubTests,
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

    with subtests.test("Test citation formatting"):
        citation_w_et_al = r"\b[\w\-]+\set\sal\.\s\([0-9]{4}\)"
        assert not re.search(
            citation_w_et_al, response.session.answer
        ), "Answer contains citation with et al. instead of citation key"

        missing_pages_regex = r"\b([a-zA-Z]+\d{4}[a-zA-Z]*\s+\d+-\d+)\b"
        assert not re.search(
            missing_pages_regex, response.session.answer
        ), "Answer contains citation with missing 'pages' keyword"


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
    agent_test_settings.agent.agent_llm = CommonLLMNames.CLAUDE_37_SONNET.value
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
    agent_test_settings.agent.timeout = 0.05  # Give time for Environment.reset()
    agent_test_settings.llm = "gpt-4o-mini"
    agent_test_settings.agent.tool_names = {"gen_answer", "complete"}
    orig_exec_tool_calls = PaperQAEnvironment.exec_tool_calls
    tool_responses: list[list[ToolResponseMessage]] = []

    async def spy_exec_tool_calls(*args, **kwargs) -> list[ToolResponseMessage]:
        responses = await orig_exec_tool_calls(*args, **kwargs)
        tool_responses.append(responses)
        return responses

    with patch.object(PaperQAEnvironment, "exec_tool_calls", spy_exec_tool_calls):
        response = await agent_query(
            query="Are COVID-19 vaccines effective?",
            settings=agent_test_settings,
            agent_type=agent_type,
        )
    # Ensure that GenerateAnswerTool was called in truncation's failover
    assert response.status == AgentStatus.TRUNCATED, "Agent did not timeout"
    assert CANNOT_ANSWER_PHRASE in response.session.answer
    (last_response,) = tool_responses[-1]
    assert (
        "no papers" in last_response.content
    ), "Expecting agent to been shown specifics on the failure"


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

    docs = Docs()
    response = await agent_query(
        query="What is a self-explanatory model?",
        settings=agent_test_settings,
        docs=docs,
        agent_type=FAKE_AGENT_TYPE,
    )
    assert response.status == AgentStatus.SUCCESS, "Agent did not succeed"
    result = response.session
    assert len(result.answer) > 200, "Answer did not return any results"
    assert "###" in result.answer, "Answer did not propagate system prompt"
    assert docs.docs, "Expected docs to have been added"
    assert all(isinstance(d, DocDetails) for d in docs.docs.values())
    assert all(
        d.file_location for d in docs.docs.values()  # type: ignore[union-attr]
    ), "Expected file location to be populated"
    assert len(result.contexts) >= 2, "Test expects a few contexts"
    # Subtract 2 to allow tolerance for chunks with leading/trailing whitespace
    num_contexts_sufficient_length = sum(
        len(c.context) >= agent_test_settings.parsing.reader_config["chunk_chars"] - 2
        for c in result.contexts
    )
    # Check most contexts have the expected length
    assert (
        num_contexts_sufficient_length >= len(result.contexts) - 1
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
                        ).gather_evidence,
                        concurrency_safe=GatherEvidence.CONCURRENCY_SAFE,
                    ),
                    Tool.from_function(
                        GenerateAnswer(
                            settings=agent_test_settings,
                            llm_model=agent_test_settings.get_llm(),
                            summary_llm_model=agent_test_settings.get_summary_llm(),
                            embedding_model=agent_test_settings.get_embedding_model(),
                        ).gen_answer,
                        concurrency_safe=GenerateAnswer.CONCURRENCY_SAFE,
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
    SAVE_API_COSTS_FILES_TO_EXCLUDE = {
        "pasa.pdf",
        *(f"dummy{x}" for x in (".docx", "_jap.docx", ".pptx", ".xlsx")),
    }

    def files_filter(f) -> bool:
        return (
            f.name not in SAVE_API_COSTS_FILES_TO_EXCLUDE
            and IndexSettings.model_fields["files_filter"].default(f)
        )

    agent_test_settings.agent.index.files_filter = files_filter
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

    session = PQASession(question="What is a self-explanatory model?")
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

        split = re.split(r"(\d+) pieces of evidence", response, maxsplit=1)
        assert len(split) == 3, "Unexpected response shape"
        total_added_1 = int(split[1])
        assert total_added_1 > 0, "Expected non-negative added evidence count"
        assert len(env_state.get_relevant_contexts()) == total_added_1
        assert (
            response.count("\n- ") == 1
        ), "Expected exactly one best evidence to be shown"

        # now adjust to give the agent 2x pieces of evidence
        gather_evidence_tool.settings.agent.agent_evidence_n = 2
        # also reset the question to ensure that contexts are
        # only returned to the agent for the new question
        new_question = "How does XAI relate to a self-explanatory model?"
        response = await gather_evidence_tool.gather_evidence(
            new_question, state=env_state
        )
        assert len({c.question for c in session.contexts}) == 2, "Expected 2 questions"
        # now we make sure this is only for the old question
        for context in session.contexts:
            if context.question != new_question:
                assert (
                    context.context[:50] not in response
                ), "gather_evidence should not return any contexts for the old question"
        assert (
            sum(
                (1 if (context.context[:30] in response) else 0)
                for context in session.contexts
                if context.question == new_question
            )
            == 2
        ), "gather_evidence should only return 2 contexts for the new question"
        split = re.split(r"(\d+) pieces of evidence", response, maxsplit=1)
        assert len(split) == 3, "Unexpected response shape"
        total_added_2 = int(split[1])
        assert total_added_2 > 0, "Expected non-negative added evidence count"
        assert len(env_state.get_relevant_contexts()) == total_added_1 + total_added_2
        assert (
            response.count("\n- ") == 2
        ), "Expected both evidences to be shown as best evidences"

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
                                " complete sentence,\nor general keywords, e.g."
                                " 'machine learning for immunology'. Also can be"
                                "\ngiven search operators."
                            ),
                            "title": "Query",
                        },
                        "min_year": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "string"},
                                {"type": "null"},
                            ],
                            "description": (
                                "Filter for minimum publication year, or None for"
                                " no minimum year.\nThe current year is"
                                f" {get_year()}."
                            ),
                            "title": "Min Year",
                        },
                        "max_year": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "string"},
                                {"type": "null"},
                            ],
                            "description": (
                                "Filter for maximum publication year, or None for"
                                " no maximum year.\nThe current year is"
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
                                "\ntask has been generated, otherwise set False to"
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
                question="foo",
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
                extra_field="extra_value",
            )
        ],
    )
    response = AnswerResponse(session=session, bibtex={}, status=AgentStatus.SUCCESS)

    assert response.session.contexts[0].text.embedding is None
    assert not response.session.contexts[0].text.text
    assert response.session.contexts[0].text.doc is not None
    assert response.session.contexts[0].text.doc.embedding is None
    assert response.session.contexts[0].extra_field is not None  # type: ignore[attr-defined]
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

    page1_results = await index.query(query="test", top_n=page_size)
    page2_results = await index.query(query="test", top_n=page_size, offset=page_size)
    page1and2_results = await index.query(query="test", top_n=2 * page_size)

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


@pytest.mark.timeout(60 * 7)  # Extended from global 5-min timeout
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


@pytest.mark.asyncio
async def test_env_from_name(subtests: SubTests) -> None:
    assert "paperqa" in Environment.available()

    with subtests.test(msg="only-task"):
        env = Environment.from_name(  # type: ignore[var-annotated]
            "paperqa", "How can you use XAI for chemical property prediction?"
        )
        assert isinstance(env, PaperQAEnvironment)
        with pytest.raises(ValueError, match="configured"):
            await env.get_id()

    with subtests.test(msg="env-kwargs"):
        env = Environment.from_name(
            "paperqa",
            query="How can you use XAI for chemical property prediction?",
            settings=Settings(),
            docs=Docs(),
        )
        assert isinstance(env, PaperQAEnvironment)


@pytest.mark.skip(reason="Manually run benchmark")
@pytest.mark.usefixtures("extended_llm_retrying")
@pytest.mark.asyncio
async def test_image_qa_zero_shot() -> None:
    settings = ImageQAEnvironment.make_base_settings(llm="claude-opus-4-5-20251101")
    llm = settings.get_llm()
    dataset = ImageQATaskDataset(dataset=LABBenchDatasets.FIG_QA, settings=settings)

    async def run_zero_shot(
        idx: int,
    ) -> tuple[MultipleChoiceEvaluation, ImageQAEnvironment]:
        env = dataset.get_new_env_by_idx(idx)
        await env.reset()  # Add image(s) to internal docs
        if len(env.state.docs.texts) != 1:
            raise NotImplementedError(
                f"Didn't handle {len(env.state.docs.texts)} texts being present."
            )
        text = env.state.docs.texts[0]
        response = await llm.call_single(
            messages=[
                Message.create_message(
                    text=env.state.session.question,
                    images=[m.to_image_url() for m in text.media],
                )
            ]
        )
        if not response.text:
            raise ValueError(f"Failed to answer question {env.state.session.question}.")
        env.state.session.raw_answer = response.text
        evaluation, env.state.session.graded_answer = await cast(
            MultipleChoiceQuestion, env._query
        ).grade(env.state.session.raw_answer)
        return evaluation, env

    results = await gather_with_concurrency(
        n=128, coros=(run_zero_shot(i) for i in range(len(dataset))), progress=True
    )
    accuracy, precision = MultipleChoiceEvaluation.calculate_accuracy_precision(
        [r[0] for r in results]
    )
    print(accuracy, precision)

    qid_to_record = {
        str(env.state.session.id): {
            "evaluation": evaluation.value,
            "raw_answer": env.state.session.raw_answer,
            "graded_answer": env.state.session.graded_answer,
        }
        for evaluation, env in results
    }
    with open("records-zero-shot.json", "w") as f:
        json.dump(qid_to_record, f, indent=2)


@pytest.mark.skip(reason="Manually run benchmark")
@pytest.mark.usefixtures("extended_llm_retrying")
@pytest.mark.asyncio
async def test_image_qa(tmp_path) -> None:
    settings = ImageQAEnvironment.make_base_settings(
        llm="gpt-4o-2024-05-13",  # Match LAB-Bench paper
        summary_llm="gpt-4o-2024-05-13",  # Match LAB-Bench paper
        agent=AgentSettings(
            agent_type="ldp.agent.SimpleAgent",
            index=IndexSettings(paper_directory=tmp_path),
            # TODO: add image support for paper_search
            tool_names={"gather_evidence", "gen_answer", "complete"},
            agent_evidence_n=3,  # Bumped up to collect several perspectives
        ),
    )
    settings.parsing.multimodal = False  # Disable multimodal
    dataset = ImageQATaskDataset(dataset=LABBenchDatasets.FIG_QA, settings=settings)
    t_cb = StoreTrajectoriesCallback()
    env_cb = StoreEnvironmentsCallback()
    m_cb = MeanMetricsCallback(eval_dataset=dataset, track_tool_usage=True)
    evaluator = Evaluator(
        config=EvaluatorConfig(
            batch_size=31,  # TableQA in 8 batches, FigQA in 6 batches
            max_rollout_steps=18,  # Match aviary paper's PaperQA setting
        ),
        agent=cast(Agent, await settings.make_ldp_agent(settings.agent.agent_type)),
        dataset=dataset,
        callbacks=[t_cb, env_cb, m_cb],
    )
    await evaluator.evaluate()
    print(m_cb.eval_means)

    qid_to_record = {
        str(env.state.session.id): {
            "reward": (
                next(t for t in t_cb.eval_trajectories if t.traj_id == traj_id)
                .steps[-1]
                .reward
            ),
            "formatted_answer": env.state.session.formatted_answer,
            "contexts": [
                next(
                    c.model_dump(
                        mode="json",
                        exclude={
                            "text": {"doc": True, "embedding": True, "media": True}
                        },
                    )
                    for c in env.state.session.contexts
                    if c.id == cid
                )
                for cid in env.state.session.used_contexts
            ],
        }
        for traj_id, env in env_cb.traj_id_to_envs.items()
    }
    with open("records-image-qa.json", "w") as f:
        json.dump(qid_to_record, f, indent=2)


@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(litellm.APIError)
    | retry_if_exception_type(litellm.InternalServerError)
    | retry_if_exception_type(litellm.Timeout),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def has_image(
    media: ParsedMedia, im_bytes: bytes
) -> tuple[bool, litellm.ModelResponse]:
    completion = await litellm.acompletion(
        "gpt-5",
        messages=[
            Message.create_message(
                text=(
                    "Attached are two images, read with different DPI settings."
                    " The second image may be blurry."
                    " Does the first image contain the second image?"
                    " In other words, is the first image a superset of the second image?"
                    " If so, please answer YES, otherwise NO."
                ),
                images=[
                    f"data:image/png;base64,{bytes_to_string(im_bytes)}",
                    media.to_image_url(),
                ],
            ).model_dump(mode="json")
        ],
    )
    if (
        len(completion.choices) == 1
        and completion.choices[0].finish_reason == "stop"
        and completion.choices[0].message.content
    ):
        return "yes" in completion.choices[0].message.content.lower(), completion
    return False, completion


async def analyze_env(
    env: GradablePaperQAEnvironment, im_bytes: bytes | None
) -> tuple[list[Text], list[Text], list[tuple[bool, litellm.ModelResponse]], str]:
    # If appropriate source was not in docs it indicates a search failure
    added_texts = sorted(
        (t for t in env.state.docs.texts if t.doc.doi and t.doc.doi in env.sources),
        key=lambda t: int(t.name.rsplit("pages ")[-1].split("-")[0]),
    )
    used_contexts = [
        c for c in env.state.session.contexts if c.id in env.state.session.used_contexts
    ]
    used_texts = [
        c.text
        for c in used_contexts
        if c.text.doc.doi and c.text.doc.doi in env.sources
    ]
    if im_bytes is not None:
        contains_media = await asyncio.gather(
            *(
                has_image(m, im_bytes)
                for t in used_texts
                for m in t.media
                if m.info["type"] != "table"
            )
        )
    else:
        contains_media = []
    if not added_texts:
        fr = "search didn't add doc"
    elif not used_texts:
        fr = "evidence didn't use doc"
    elif not contains_media:
        fr = "unknown"
    elif not any(x[0] for x in contains_media):
        fr = "evidence didn't use image"
    else:
        fr = "bad pdf parse"
    return added_texts, used_texts, contains_media, fr


def get_reader(value: str) -> PDFParserFn:
    if value == "pymupdf":
        return pymupdf_parse_pdf_to_pages
    if value == "docling":
        return docling_parse_pdf_to_pages
    if value == "nemotron":
        return nemotron_parse_pdf_to_pages
    raise NotImplementedError(f"Didn't yet implement reader {value!r}.")


class PlaceholderProvider(DOIOrTitleBasedProvider):
    """
    Provider to appease DocMetadataClient's requirement for a metadata provider.

    This is useful if you're working with a manifest, and have MVP metadata.
    """

    async def _query(self, query) -> DocDetails | None:  # noqa: ARG002
        return DocDetails()


class TestTextQATaskDataset:
    PAPERS_DIR: ClassVar[Path] = Path.home() / "Documents" / "Datasets"

    @pytest.mark.parametrize("reader", ["pymupdf", "docling"])
    @pytest.mark.asyncio
    async def test_figqa_reads(self, reader: str) -> None:
        settings = Settings(
            agent=AgentSettings(
                index=IndexSettings(
                    # NOTE: name doesn't matter since we're not saving an index
                    paper_directory=self.PAPERS_DIR / "FigQA",
                    manifest_file="manifest.csv",
                ),
            ),
            parsing=ParsingSettings(
                parse_pdf=get_reader(reader),
                multimodal=False,  # No need for enrichments since not running agent
            ),
        )
        dataset = TextQATaskDataset(
            settings=settings,
            dataset=LABBenchDatasets.FIG_QA,
            read_data_kwargs={"seed": 42},
        )
        paper_directory = anyio.Path(settings.agent.index.paper_directory)
        manifest = await maybe_get_manifest(
            filename=await settings.agent.index.finalize_manifest_file()
        )
        docs = Docs()
        doi_locks = collections.defaultdict(asyncio.Lock)

        async def run_one(idx: int) -> tuple[bool, GradablePaperQAEnvironment]:
            env_i = dataset.get_new_env_by_idx(idx)
            (doi,) = env_i.sources
            (details_ser,) = (v for v in manifest.values() if v.get("doi") == doi)
            try:
                details = fetch_kwargs_from_manifest(
                    file_location=details_ser["file_location"],
                    manifest=manifest,
                    manifest_fallback_location="placeholder",
                )
            except TypeError:
                details = {
                    "file_location": details_ser["file_location"],
                    "doi": details_ser["doi"],
                    "dockey": details_ser["dockey"],
                    "doc_id": details_ser["doc_id"],
                }
            # The lock and DOI already in Docs check prevents a race conditions where
            # concurrently we read the same PDF many times. The reason is aadd_texts
            # needs to finish for Docs.docs to be manipulated, so we end up having to
            # call read_doc on all DOIs first
            async with doi_locks[doi]:
                if not any(dd.doi == details["doi"] for dd in docs.docs.values()):
                    # Only add each DOI once as an optimization
                    await docs.aadd(
                        path=paper_directory / details["file_location"],
                        **{k: v for k, v in details.items() if k != "file_location"},
                        clients={PlaceholderProvider},
                        settings=settings,
                    )
            figure = cast(bytes, await dataset.get_images(env_i))
            unique_media = list(
                dict.fromkeys(
                    m
                    for t in docs.texts
                    if t.doc.doi == details["doi"]
                    for m in t.media
                    if m.info["type"] in {"picture", "drawing"}
                )
            )
            contains_media: list[tuple[bool, litellm.ModelResponse]] = []

            async def early_stopping_has_image(media, im_bytes):
                if sum(x[0] for x in contains_media) >= 2:
                    return None
                ret = await has_image(media, im_bytes)
                contains_media.append(ret)
                return ret

            await tqdm.gather(
                *(early_stopping_has_image(m, figure) for m in unique_media),
                desc=f"Images {details.get('doi')}",
            )
            return any(x[0] for x in contains_media), env_i

        results = await tqdm.gather(
            *(run_one(i) for i in range(len(dataset))), desc="Environments"
        )
        performance = sum(x[0] for x in results) / len(results)
        print(f"Reader {reader} read {performance * 100}% of images.")
        doi_to_results = collections.defaultdict(dict)
        for b, env in sorted(results, key=lambda x: x[1].sources[0]):
            doi_to_results[env.sources[0]][str(env._session_id)] = b
        for doi, inner_results in doi_to_results.items():  # Sort by question ID
            doi_to_results[doi] = dict(sorted(inner_results.items()))
        with open(f"results-{reader}-floor.json", "w") as f:
            json.dump(doi_to_results, f, indent=2)
        assert len(docs.docs) == 41, "Incorrect number of DOIs for FigQA"

    # Sampled subset of FigQA version 1 where:
    # 1. Docling can read at least one image in the PDF
    # 2. There's at least 50 questions (52, to be exact) answerable with these PDFs
    SAMPLED_SUBSET_OF_FIGQA1_DOIS: ClassVar[tuple[str, ...]] = (
        "10.1016/j.cell.2017.10.015",
        "10.1016/j.cell.2018.05.002",
        "10.1016/j.cell.2018.05.057",
        "10.1016/j.cell.2018.06.019",
        "10.1016/j.cell.2019.02.037",
        "10.1016/j.cub.2015.07.056",
        "10.1016/j.neuron.2015.07.030",
        "10.1016/j.neuron.2015.09.020",
        "10.1016/j.neuron.2020.01.032",
        "10.1038/s41586-018-0425-3",
        "10.1038/s41593-019-0566-1",
    )
    BAD_FIGQA1_QUESTION_IDS = (
        "8c66c9c2-f45f-4c4b-8101-173511090db9"  # L5 vs Scan Lens L5 tricks humans
    )
    # FigQA2 has more modern DOIs than our local folder of papers
    FIGQA2_DOI_BACKDATES = {
        "10.1016/j.cub.2025.06.060": "10.1101/2025.02.28.640764",
        "10.1016/j.cub.2025.08.023": "10.1101/2024.09.11.611976",
        "10.1016/j.jbc.2024.106794": "10.1101/2023.07.10.548331",
    }

    # @pytest.mark.parametrize("reader", ["docling"])
    @pytest.mark.parametrize("reader", ["nemotron"])
    @pytest.mark.asyncio
    async def test_figqa_retrieval(self, tmp_path, reader: str) -> None:
        multimodal = MultimodalOptions.ON_WITH_ENRICHMENT
        settings = Settings(
            llm="claude-sonnet-4-5-20250929",
            summary_llm="claude-sonnet-4-5-20250929",
            # summary_llm="gpt-4o-2024-05-13",  # Match LAB-Bench paper
            embedding="text-embedding-3-large",
            agent=AgentSettings(
                agent_llm="gpt-4.1-2025-04-14",
                agent_type="ldp.agent.SimpleAgent",
                index=IndexSettings(
                    name=f"figqa2-{reader}-{multimodal.value}",
                    paper_directory=self.PAPERS_DIR / "FigQA2",
                    manifest_file="manifest.csv",
                ),
                agent_evidence_n=3,  # Bumped up to collect several perspectives
            ),
            parsing=ParsingSettings(
                parse_pdf=get_reader(reader),
                reader_config={
                    "chunk_chars": 3000,
                    "overlap": 250,
                    "dpi": 300,
                    "api_params": {"temperature": 0.05},
                    # "full_page": True,
                },
                # enrichment_prompt=full_page_enrichment_prompt_template,
                multimodal=multimodal,
            ),
        )
        dataset = TextQATaskDataset(
            settings=settings,
            dataset=LABBenchDatasets.FIG_QA2,
            read_data_kwargs={"seed": 42},
        )
        paper_directory = anyio.Path(settings.agent.index.paper_directory)
        manifest = await maybe_get_manifest(
            filename=await settings.agent.index.finalize_manifest_file()
        )

        def make_individual_env(idx: int) -> GradablePaperQAEnvironment:
            env = dataset.get_new_env_by_idx(idx)
            (doi,) = env.sources
            doi = self.FIGQA2_DOI_BACKDATES.get(doi, doi)
            (details,) = (v for v in manifest.values() if v.get("doi") == doi.lower())
            new_paper_dir = tmp_path / str(env._session_id)
            new_paper_dir.mkdir()
            for filename in (
                details.get("file_location"),
                settings.agent.index.manifest_file,
            ):
                shutil.copyfile(
                    src=paper_directory / filename, dst=new_paper_dir / filename
                )
            settings_orig_set = settings.model_dump()
            settings_orig_set["agent"]["index"]["paper_directory"] = new_paper_dir
            settings_orig_set["agent"]["index"]["name"] += f"-{env._session_id}"
            # Work around exclusion of PDF parser
            settings_orig_set["parsing"]["parse_pdf"] = settings.parsing.parse_pdf
            return GradablePaperQAEnvironment(
                query=env._query,
                settings=Settings(**settings_orig_set),
                docs=env._docs,
                sources=env.sources,
                rewards=env._rewards,
                session_id=env._query.question_id,  # Expedite manual inspection
            )

        envs = [make_individual_env(i) for i in range(len(dataset))]
        # Filter out envs not in the sampled subset
        # envs = [e for e in envs if e.sources[0] in self.SAMPLED_SUBSET_OF_FIGQA1_DOIS]
        # envs = [envs[i] for i in range(5, 12)]
        # envs = [
        #     e
        #     for e in envs
        #     if str(e._query.question_id) not in self.BAD_FIGQA1_QUESTION_IDS
        # ]
        index_lock = asyncio.Lock()  # Unclear if tantivy is async-safe
        index_barrier = asyncio.Barrier(parties=len(envs))

        async def run_one(idx: int):
            env = envs[idx]
            async with index_lock:
                try:
                    await get_directory_index(settings=env._settings)
                except ExceptionGroup as exc_group:
                    if len(exc_group.exceptions) != 1 or not isinstance(
                        exc_group.exceptions[0], RuntimeError
                    ):
                        raise
                    # Delete the index before blowing us up, to ensure future runs don't
                    # see a completed (but non-functional) index
                    shutil.rmtree(env._settings.agent.index.get_named_index_directory())
                    raise
            new_dataset = EnvsTaskDataset(envs=env)
            t_cb = StoreTrajectoriesCallback()
            m_cb = MeanMetricsCallback(eval_dataset=new_dataset, track_tool_usage=True)
            evaluator = Evaluator(
                config=EvaluatorConfig(
                    max_rollout_steps=18,  # Match aviary paper's PaperQA setting
                ),
                agent=cast(
                    Agent, await settings.make_ldp_agent(settings.agent.agent_type)
                ),
                dataset=new_dataset,
                callbacks=[t_cb, m_cb],
            )
            async with index_barrier:
                await evaluator.evaluate()
            reward = t_cb.eval_trajectories[-1].steps[-1].reward
            m_cb.eval_means["correct"] = reward == env._rewards["correct"]
            m_cb.eval_means["correct_unsure"] = reward in {
                env._rewards["correct"],
                env._rewards["unsure"],
            }
            try:
                *x, fr = await analyze_env(
                    env, im_bytes=cast(bytes, await dataset.get_images(env))
                )
            except ValueError as exc:
                if "no images column" not in str(exc):
                    raise
                *x, fr = await analyze_env(env, im_bytes=None)
            return fr, m_cb.eval_means, env, x
            # return m_cb.eval_means, env

        all_results = await tqdm.gather(
            *(run_one(i) for i in range(len(envs))), desc="Environments"
        )
        _ = 0
        results = [x for x in all_results if x[0] != "skipped"]
        correct_envs = {
            str(env.state.session.id): fr
            for fr, means, env, _ in results
            if means["correct"]
        }
        correct_unsure_envs = {
            str(env.state.session.id): fr
            for fr, means, env, _ in results
            if not means["correct"] and means["correct_unsure"]
        }
        correct = len(correct_envs) / len(results)
        correct_unsure = (len(correct_envs) + len(correct_unsure_envs)) / len(results)
        failure_reasons = {
            str(env.state.session.id): fr
            for fr, means, env, _ in results
            if not means["correct"]
        }
        print(f"Results are {correct=}, {correct_unsure=}.")
        _ = 0

        qid_to_record = {
            str(env.state.session.id): {
                "reward": (
                    1 if means["correct"] else (0.1 if means["correct_unsure"] else -1)
                ),
                "formatted_answer": env.state.session.formatted_answer,
                "contexts": [
                    next(
                        c.model_dump(
                            mode="json",
                            exclude={
                                "text": {"doc": True, "embedding": True, "media": True}
                            },
                        )
                        for c in env.state.session.contexts
                        if c.id == cid
                    )
                    for cid in env.state.session.used_contexts
                ],
                "failure_reason": fr,
            }
            for fr, means, env, _ in results
        }
        with open(f"records-individual-mm-enrichment-{reader}.json", "w") as f:
            json.dump(qid_to_record, f, indent=2)

    # @pytest.mark.skip(reason="Manually run benchmark")
    # @pytest.mark.usefixtures("extended_llm_retrying")
    @pytest.mark.parametrize("reader", ["nemotron"])
    @pytest.mark.asyncio
    async def test_figqa_pdf_bucket(self, reader: str) -> None:
        multimodal = MultimodalOptions.ON_WITH_ENRICHMENT
        settings = Settings(
            llm="claude-sonnet-4-5-20250929",
            summary_llm="claude-sonnet-4-5-20250929",
            # summary_llm="gpt-4o-2024-05-13",  # Match LAB-Bench paper
            embedding="text-embedding-3-large",
            agent=AgentSettings(
                agent_llm="gpt-4.1-2025-04-14",
                agent_type="ldp.agent.SimpleAgent",
                index=IndexSettings(
                    name=f"figqa2-{reader}-{multimodal.value}",
                    paper_directory=self.PAPERS_DIR / "FigQA2",
                    manifest_file="manifest.csv",
                ),
                agent_evidence_n=3,  # Bumped up to collect several perspectives
            ),
            parsing=ParsingSettings(
                parse_pdf=get_reader(reader),
                reader_config={
                    "chunk_chars": 3000,
                    "overlap": 250,
                    "dpi": 300,
                    "api_params": {"temperature": 0.05},
                    # "full_page": True,
                },
                # enrichment_prompt=full_page_enrichment_prompt_template,
                multimodal=multimodal,
            ),
        )
        dataset = TextQATaskDataset(
            settings=settings,
            dataset=LABBenchDatasets.FIG_QA2,
            read_data_kwargs={"seed": 42},
        )
        # indices_in_subset = [
        #     i
        #     for i in range(len(dataset.data))
        #     if dataset.data.iloc[i].id not in self.BAD_FIGQA1_QUESTION_IDS
        #     # and dataset._dataset.get_sources(dataset.data.iloc[i])[0]
        #     # in self.SAMPLED_SUBSET_OF_DOIS
        # ]
        # dataset.data = dataset.data.iloc[indices_in_subset]

        await get_directory_index(settings=settings)  # Build the index up front
        env_cb = StoreEnvironmentsCallback()
        t_cb = StoreTrajectoriesCallback()
        m_cb = MeanMetricsCallback(eval_dataset=dataset, track_tool_usage=True)
        evaluator = Evaluator(
            config=EvaluatorConfig(
                batch_size=34,  # TableQA in 8 batches, FigQA in 6 batches, FigQA2/TableQA2 in 3 batches
                # batch_size=13,
                max_rollout_steps=18,  # Match aviary paper's PaperQA setting
            ),
            agent=cast(Agent, await settings.make_ldp_agent(settings.agent.agent_type)),
            dataset=dataset,
            callbacks=[env_cb, t_cb, m_cb],
        )
        await evaluator.evaluate()
        print(m_cb.eval_means)
        _ = 0

        qid_to_record = {
            str(env.state.session.id): {
                "reward": (
                    next(t for t in t_cb.eval_trajectories if t.traj_id == traj_id)
                    .steps[-1]
                    .reward
                ),
                "formatted_answer": env.state.session.formatted_answer,
                "contexts": [
                    next(
                        c.model_dump(
                            mode="json",
                            exclude={
                                "text": {"doc": True, "embedding": True, "media": True}
                            },
                        )
                        for c in env.state.session.contexts
                        if c.id == cid
                    )
                    for cid in env.state.session.used_contexts
                ],
            }
            for traj_id, env in env_cb.traj_id_to_envs.items()
        }
        with open(f"records-mm-enrichment-{reader}.json", "w") as f:
            json.dump(qid_to_record, f, indent=2)

        qid_to_reward = dict(
            sorted(
                {
                    str(env_cb.traj_id_to_envs[t.traj_id].state.session.id): (
                        t.steps[-1].reward
                    )
                    for t in t_cb.eval_trajectories
                }.items()
            )
        )
        with open("results-multimodal.json", "w") as f:
            json.dump(qid_to_reward, f)

        # Sort by env state session ID for reproducibility
        env_cb.traj_id_to_envs = dict(
            sorted(
                env_cb.traj_id_to_envs.items(),
                key=lambda item: item[1].state.session.id,
            )
        )

        t_env_failed = [
            (t, env_cb.traj_id_to_envs[t.traj_id])
            for t in t_cb.eval_trajectories
            if not t.failed and t.steps and t.steps[-1].reward < 0
        ]

        # If appropriate source was not in docs it indicates a search failure
        added_texts = [
            (
                env,
                sorted(
                    (
                        t
                        for t in env.state.docs.texts
                        if t.doc.doi and t.doc.doi in env.sources
                    ),
                    key=lambda t: int(t.name.rsplit("pages ")[-1].split("-")[0]),
                ),
            )
            for env in env_cb.traj_id_to_envs.values()
        ]
        used_contexts = [
            [
                c
                for c in env.state.session.contexts
                if c.id in env.state.session.used_contexts
            ]
            for env in env_cb.traj_id_to_envs.values()
        ]
        used_texts = [
            [
                c.text
                for c in contexts
                if c.text.doc.doi and c.text.doc.doi in env.sources
            ]
            for env, contexts in zip(
                env_cb.traj_id_to_envs.values(), used_contexts, strict=True
            )
        ]

        failure_reasons = {env.state.session.id: "" for _, env in t_env_failed}
        env_to_images = collections.defaultdict(list)
        for (env, ts1), ts2 in track(
            zip(added_texts, used_texts, strict=True),
            total=len(added_texts),
            description="Checking images in evidence",
        ):
            if not ts1:
                failure_reasons[env.state.session.id] = "search didn't add doc"
            elif not ts2:
                failure_reasons[env.state.session.id] = "evidence didn't use doc"
            else:
                figure = cast(bytes, await dataset.get_images(env))

                @retry(
                    stop=stop_after_attempt(3),
                    retry=retry_if_exception_type(litellm.APIError)
                    | retry_if_exception_type(litellm.InternalServerError)
                    | retry_if_exception_type(litellm.Timeout),
                    before_sleep=before_sleep_log(logger, logging.WARNING),
                )
                async def has_image(
                    media: ParsedMedia, env_id: UUID, im_bytes: bytes = figure
                ) -> bool:
                    completion = await litellm.acompletion(
                        "gpt-5",
                        messages=[
                            Message.create_message(
                                text=(
                                    "Attached are two images, read with different DPI settings."
                                    " The second image may be blurry."
                                    " Does the first image contain the second image?"
                                    " In other words, is the first image a superset of the second image?"
                                    " If so, please answer YES, otherwise NO."
                                ),
                                images=[
                                    f"data:image/png;base64,{bytes_to_string(im_bytes)}",
                                    media.to_image_url(),
                                ],
                            ).model_dump(mode="json")
                        ],
                    )
                    if (
                        len(completion.choices) == 1
                        and completion.choices[0].finish_reason == "stop"
                        and completion.choices[0].message.content
                    ):
                        env_to_images[env_id].append(
                            (media.to_id(), completion.choices[0].message.content)
                        )
                        return "yes" in completion.choices[0].message.content.lower()
                    return False

                contains_media = await asyncio.gather(
                    *(
                        has_image(m, env.state.session.id)
                        for t in ts2
                        for m in t.media
                        if m.info["type"] != "table"
                    )
                )
                if not any(contains_media):
                    failure_reasons[env.state.session.id] = "evidence didn't use image"

        no_doc_in_search_failure_frac = sum(
            r == "search didn't add doc" for r in failure_reasons.values()
        ) / len(failure_reasons)
        no_doc_in_evidence_failure_frac = sum(
            r == "evidence didn't use doc" for r in failure_reasons.values()
        ) / len(failure_reasons)
        no_image_in_evidence_failure_frac = sum(
            r == "evidence didn't use image" for r in failure_reasons.values()
        ) / len(failure_reasons)
        print(
            f"{no_doc_in_search_failure_frac=},"
            f" {no_doc_in_evidence_failure_frac=},"
            f" {no_image_in_evidence_failure_frac=}."
        )
        _ = 0

    @pytest.mark.skip(reason="Manually run benchmark")
    @pytest.mark.usefixtures("extended_llm_retrying")
    @pytest.mark.asyncio
    async def test_tableqa_pdf_bucket(self) -> None:
        settings = Settings(
            summary_llm="gpt-4o-2024-05-13",  # Match LAB-Bench paper
            agent=AgentSettings(
                agent_type="ldp.agent.SimpleAgent",
                index=IndexSettings(
                    # name="tableqa-docling",
                    # name="tableqa-pymupdf",
                    name="tableqa",
                    paper_directory=self.PAPERS_DIR / "TableQA",
                    manifest_file="manifest.csv",
                ),
                agent_evidence_n=3,  # Bumped up to collect several perspectives
            ),
            # parsing=ParsingSettings(parse_pdf=parse_pdf_to_pages),
            parsing=ParsingSettings(multimodal=False),
        )
        await get_directory_index(settings=settings)  # Build the index up front
        dataset = TextQATaskDataset(
            settings=settings,
            dataset=LABBenchDatasets.TABLE_QA,
            read_data_kwargs={"seed": 42},
        )
        env_cb = StoreEnvironmentsCallback()
        m_cb = MeanMetricsCallback(eval_dataset=dataset, track_tool_usage=True)
        evaluator = Evaluator(
            config=EvaluatorConfig(
                batch_size=34,  # TableQA in 8 batches, FigQA in 6 batches, FigQA2/TableQA2 in 3 batches
                max_rollout_steps=18,  # Match aviary paper's PaperQA setting
            ),
            agent=cast(Agent, await settings.make_ldp_agent(settings.agent.agent_type)),
            dataset=dataset,
            callbacks=[env_cb, m_cb],
        )
        await evaluator.evaluate()
        print(m_cb.eval_means)
        _ = 0
