from __future__ import annotations

import itertools
import json
import re
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import ldp.agent
import pytest
from aviary.tools import ToolRequestMessage, ToolsAdapter, ToolSelector
from ldp.agent import MemoryAgent, SimpleAgent
from ldp.graph.memory import Memory, UIndexMemoryModel
from ldp.graph.ops import OpResult
from ldp.llms import EmbeddingModel, MultipleCompletionLLMModel
from pydantic import ValidationError
from pytest_subtests import SubTests

from paperqa.agents import agent_query
from paperqa.agents.env import settings_to_tools
from paperqa.agents.models import AgentStatus, AnswerResponse, QueryRequest
from paperqa.agents.search import FAILED_DOCUMENT_ADD_ID, get_directory_index
from paperqa.agents.tools import (
    EnvironmentState,
    GatherEvidence,
    GenerateAnswer,
    PaperSearch,
    make_status,
)
from paperqa.docs import Docs
from paperqa.settings import AgentSettings, Settings
from paperqa.types import Answer, Context, Doc, Text
from paperqa.utils import extract_thought, get_year, md5sum


@pytest.mark.asyncio
async def test_get_directory_index(agent_test_settings: Settings) -> None:
    index = await get_directory_index(settings=agent_test_settings)
    assert index.fields == [
        "file_location",
        "body",
        "title",
        "year",
    ], "Incorrect fields in index"
    # paper.pdf + empty.txt + flag_day.html + bates.txt + obama.txt,
    # but empty.txt fails to be added
    path_to_id = await index.index_files
    assert (
        sum(id_ != FAILED_DOCUMENT_ADD_ID for id_ in path_to_id.values()) == 4
    ), "Incorrect number of parsed index files"
    results = await index.query(query="who is Frederick Bates?")
    paper_dir = cast(Path, agent_test_settings.paper_directory)
    assert results[0].docs.keys() == {md5sum((paper_dir / "bates.txt").absolute())}


@pytest.mark.asyncio
async def test_get_directory_index_w_manifest(
    agent_test_settings: Settings, reset_log_levels, caplog  # noqa: ARG001
) -> None:
    agent_test_settings.manifest_file = "stub_manifest.csv"
    index = await get_directory_index(settings=agent_test_settings)
    assert index.fields == [
        "file_location",
        "body",
        "title",
        "year",
    ], "Incorrect fields in index"
    # paper.pdf + empty.txt + flag_day.html + bates.txt + obama.txt
    assert len(await index.index_files) == 5, "Incorrect number of index files"
    results = await index.query(query="who is Frederick Bates?")
    top_result = next(iter(results[0].docs.values()))
    paper_dir = cast(Path, agent_test_settings.paper_directory)
    assert top_result.dockey == md5sum((paper_dir / "bates.txt").absolute())
    # note: this title comes from the manifest, so we know it worked
    assert top_result.title == "Frederick Bates (Wikipedia article)"


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError", "httpx.RemoteProtocolError"])
@pytest.mark.parametrize("agent_type", ["fake", ToolSelector, SimpleAgent])
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
    response = await agent_query(request, agent_type=agent_type)
    assert response.answer.answer, "Answer not generated"
    assert response.answer.answer != "I cannot answer", "Answer not generated"
    assert response.answer.context, "No contexts were found"
    assert response.answer.question == question
    agent_llm = request.settings.agent.agent_llm
    # TODO: once LDP can track tokens, we can remove this check
    if agent_type not in {"fake", SimpleAgent}:
        print(response.answer.token_counts)
        assert (
            response.answer.token_counts[agent_llm][0] > 1000
        ), "Expected many prompt tokens"
        assert (
            response.answer.token_counts[agent_llm][1] > 50
        ), "Expected many completion tokens"
        assert response.answer.cost > 0, "Expected nonzero cost"


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
    assert response.status == AgentStatus.TIMEOUT, "Agent did not timeout"
    assert "I cannot answer" in response.answer.answer


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
    response = await agent_query(query, agent_type="fake")
    assert response.status == AgentStatus.SUCCESS, "Agent did not succeed"
    result = response.answer
    assert len(result.answer) > 200, "Answer did not return any results"
    assert "###" in result.answer, "Answer did not propagate system prompt"
    assert (
        len(result.contexts[0].context) == agent_test_settings.parsing.chunk_size
    ), "Summary was not skipped"


@pytest.mark.asyncio
async def test_gather_evidence_rejects_empty_docs() -> None:
    # Patch GenerateAnswerTool._arun so that if this tool is chosen first, we
    # don't give a 'cannot answer' response. A 'cannot answer' response can
    # lead to an unsure status, which will break this test's assertions. Since
    # this test is about a GatherEvidenceTool edge case, defeating
    # GenerateAnswerTool is fine
    with patch.object(
        GenerateAnswer, "gen_answer", return_value="Failed to answer question."
    ):
        settings = Settings()
        settings.agent.tool_names = {"gather_evidence", "gen_answer"}
        response = await agent_query(
            query=QueryRequest(
                query="Are COVID-19 vaccines effective?", settings=settings
            ),
            docs=Docs(),
        )
    assert response.status == AgentStatus.FAIL, "Agent should have registered a failure"


@pytest.mark.flaky(reruns=3, only_rerun=["AssertionError", "EmptyDocsError"])
@pytest.mark.asyncio
async def test_agent_sharing_state(
    agent_test_settings: Settings, subtests: SubTests
) -> None:
    agent_test_settings.agent.search_count = 3  # Keep low for speed
    agent_test_settings.answer.evidence_k = 2
    agent_test_settings.answer.answer_max_sources = 1
    llm_model = agent_test_settings.get_llm()
    summary_llm_model = agent_test_settings.get_summary_llm()
    embedding_model = agent_test_settings.get_embedding_model()

    answer = Answer(question="What is is a self-explanatory model?")
    docs = Docs()
    query = QueryRequest(query=answer.question, settings=agent_test_settings)
    env_state = EnvironmentState(docs=docs, answer=answer)

    with subtests.test(msg=PaperSearch.__name__):
        search_tool = PaperSearch(
            settings=agent_test_settings, embedding_model=embedding_model
        )
        await search_tool.paper_search(
            "XAI self explanatory model", min_year=None, max_year=None, state=env_state
        )
        assert env_state.docs.docs, "Search did not save any papers"
        assert all(
            (isinstance(d, Doc) or issubclass(d, Doc))  # type: ignore[unreachable]
            for d in env_state.docs.docs.values()
        ), "Document type or DOI propagation failure"

    with subtests.test(msg=GatherEvidence.__name__):
        assert not answer.contexts, "No contexts is required for a later assertion"

        gather_evidence_tool = GatherEvidence(
            settings=agent_test_settings,
            summary_llm_model=summary_llm_model,
            embedding_model=embedding_model,
        )
        await gather_evidence_tool.gather_evidence(answer.question, state=env_state)
        assert answer.contexts, "Evidence did not return any results"

    with subtests.test(msg=f"{GenerateAnswer.__name__} working"):
        generate_answer_tool = GenerateAnswer(
            settings=agent_test_settings,
            llm_model=llm_model,
            summary_llm_model=summary_llm_model,
            embedding_model=embedding_model,
        )
        result = await generate_answer_tool.gen_answer(answer.question, state=env_state)
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
    answer = Answer(
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
    response = AnswerResponse(answer=answer, bibtex={}, status="success")

    assert response.answer.contexts[0].text.embedding is None
    assert response.answer.contexts[0].text.text == ""  # type: ignore[unreachable,unused-ignore]
    assert response.answer.contexts[0].text.doc is not None
    assert response.answer.contexts[0].text.doc.embedding is None
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
