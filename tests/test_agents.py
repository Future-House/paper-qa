from __future__ import annotations

import itertools
import json
import re
from typing import Any, cast
from unittest.mock import patch

import pytest
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from pytest_subtests import SubTests

from paperqa.agents import agent_query
from paperqa.agents.models import (
    AgentStatus,
    AnswerResponse,
    QueryRequest,
)
from paperqa.agents.search import get_directory_index
from paperqa.agents.tools import (
    GatherEvidenceTool,
    GenerateAnswerTool,
    PaperSearchTool,
    SharedToolState,
)
from paperqa.config import AgentSettings, Settings
from paperqa.docs import Docs
from paperqa.types import Answer, Context, Doc, Text
from paperqa.utils import get_year, md5sum


@pytest.mark.asyncio
async def test_get_directory_index(agent_test_settings):
    index = await get_directory_index(
        settings=agent_test_settings,
    )
    assert index.fields == [
        "file_location",
        "body",
        "title",
        "year",
    ], "Incorrect fields in index"
    # paper.pdf + flag_day.html + bates.txt + obama.txt
    assert len(await index.index_files) == 4, "Incorrect number of index files"
    results = await index.query(query="who is Frederick Bates?")
    assert results[0].docs.keys() == {
        md5sum((agent_test_settings.paper_directory / "bates.txt").absolute())
    }


@pytest.mark.asyncio
async def test_get_directory_index_w_manifest(
    agent_test_settings, reset_log_levels, caplog  # noqa: ARG001
):
    agent_test_settings.manifest_file = "stub_manifest.csv"
    index = await get_directory_index(settings=agent_test_settings)
    assert index.fields == [
        "file_location",
        "body",
        "title",
        "year",
    ], "Incorrect fields in index"
    # paper.pdf + flag_day.html + bates.txt + obama.txt
    assert len(await index.index_files) == 4, "Incorrect number of index files"
    results = await index.query(query="who is Frederick Bates?")
    top_result = next(iter(results[0].docs.values()))
    assert top_result.dockey == md5sum(
        (agent_test_settings.paper_directory / "bates.txt").absolute()
    )
    # note: this title comes from the manifest, so we know it worked
    assert top_result.title == "Frederick Bates (Wikipedia article)"


@pytest.mark.flaky(reruns=2, only_rerun=["AssertionError", "httpx.RemoteProtocolError"])
@pytest.mark.parametrize("agent_type", ["fake", "OpenAIFunctionsAgent"])
@pytest.mark.asyncio
async def test_agent_types(agent_test_settings, agent_type):

    question = "How can you use XAI for chemical property prediction?"

    request = QueryRequest(
        query=question,
        settings=agent_test_settings,
    )
    response = await agent_query(request, agent_type=agent_type)
    assert response.answer.answer, "Answer not generated"
    assert response.answer.answer != "I cannot answer", "Answer not generated"
    assert len(response.answer.context) >= 1, "No contexts were found"
    assert response.answer.question == question


@pytest.mark.asyncio
async def test_timeout(agent_test_settings):
    agent_test_settings.prompts.pre = None
    agent_test_settings.agent.timeout = 0.001
    agent_test_settings.llm = "gpt-4o-mini"
    agent_test_settings.agent.tool_names = {"gen_answer"}
    response = await agent_query(
        QueryRequest(
            query="Are COVID-19 vaccines effective?", settings=agent_test_settings
        )
    )
    # ensure that GenerateAnswerTool was called
    assert response.status == AgentStatus.TIMEOUT, "Agent did not timeout"
    assert "I cannot answer" in response.answer.answer


@pytest.mark.asyncio
async def test_propagate_options(agent_test_settings) -> None:
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
        query="What is is a self-explanatory model?",
        settings=agent_test_settings,
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
        GenerateAnswerTool, "_arun", return_value="Failed to answer question."
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


@pytest.mark.flaky(reruns=3, only_rerun=["AssertionError"])
@pytest.mark.asyncio
async def test_agent_sharing_state(agent_test_settings, subtests: SubTests) -> None:
    agent_test_settings.agent.search_count = 3  # Keep low for speed
    agent_test_settings.answer.evidence_k = 2
    agent_test_settings.answer.answer_max_sources = 1
    answer = Answer(question="What is is a self-explanatory model?")
    my_docs = Docs()
    query = QueryRequest(
        query=answer.question,
        settings=agent_test_settings,
    )
    tool_state = SharedToolState(
        docs=my_docs, answer=answer, settings=agent_test_settings
    )

    with subtests.test(msg=PaperSearchTool.__name__):
        tool = PaperSearchTool(
            shared_state=tool_state,
        )
        await tool.arun("XAI self explanatory model")
        assert tool_state.docs.docs, "Search did not save any papers"
        assert all(
            (isinstance(d, Doc) or issubclass(d, Doc))  # type: ignore[unreachable]
            for d in tool_state.docs.docs.values()
        ), "Document type or DOI propagation failure"

    with subtests.test(msg=GatherEvidenceTool.__name__):
        assert not answer.contexts, "No contexts is required for a later assertion"

        tool = GatherEvidenceTool(shared_state=tool_state, query=query)
        await tool.arun(answer.question)
        assert answer.contexts, "Evidence did not return any results"

    with subtests.test(msg=f"{GenerateAnswerTool.__name__} working"):
        tool = GenerateAnswerTool(shared_state=tool_state, query=query)
        result = await tool.arun(answer.question)
        assert re.search(
            pattern=SharedToolState.STATUS_SEARCH_REGEX_PATTERN, string=result
        )
        assert len(answer.answer) > 200, "Answer did not return any results"
        assert (
            GenerateAnswerTool.extract_answer_from_message(result) == answer.answer
        ), "Failed to regex extract answer from result"
        assert (
            len(answer.used_contexts) <= query.settings.answer.answer_max_sources
        ), "Answer has more sources than expected"


def test_functions() -> None:
    """Check the functions schema passed to OpenAI."""
    shared_tool_state = SharedToolState(
        answer=Answer(question="stub"),
        docs=Docs(),
        settings=Settings(),
    )
    agent = OpenAIFunctionsAgent.from_llm_and_tools(
        llm=ChatOpenAI(model="gpt-4-turbo-2024-04-09"),
        tools=[
            PaperSearchTool(shared_state=shared_tool_state),
            GatherEvidenceTool(shared_state=shared_tool_state, query=QueryRequest()),
            GenerateAnswerTool(shared_state=shared_tool_state, query=QueryRequest()),
        ],
    )
    assert cast(OpenAIFunctionsAgent, agent).functions == [
        {
            "name": "paper_search",
            "description": (
                "Search for papers to increase the paper count. You can call this a second "
                "time with an different search to gather more papers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "description": (
                            "A search query in this format: [query], [start year]-[end year]. "
                            "You may include years as the last word in the query, "
                            "e.g. 'machine learning 2020' or 'machine learning 2010-2020'. "
                            f"The current year is {get_year()}. "
                            "The query portion can be a specific phrase, complete sentence, "
                            "or general keywords, e.g. 'machine learning for immunology'."
                        ),
                        "type": "string",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "gather_evidence",
            "description": (
                "Gather evidence from previous papers given a specific question. "
                "This will increase evidence and relevant paper counts. "
                "Only invoke when paper count is above zero."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "description": "Specific question to gather evidence for.",
                        "type": "string",
                    }
                },
                "required": ["question"],
            },
        },
        {
            "name": "gen_answer",
            "description": (
                "Ask a model to propose an answer answer using current evidence. "
                "The tool may fail, "
                "indicating that better or different evidence should be found. "
                "Having more than one piece of evidence or relevant papers is best."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "description": "Question to be answered.",
                        "type": "string",
                    }
                },
                "required": ["question"],
            },
        },
    ]


def test_query_request_docs_name_serialized():
    """Test that the query request has a docs_name property."""
    request = QueryRequest(query="Are COVID-19 vaccines effective?")
    request_data = json.loads(request.model_dump_json())
    assert "docs_name" in request_data
    assert request_data["docs_name"] is None
    request.set_docs_name("my_doc")
    request_data = json.loads(request.model_dump_json())
    assert request_data["docs_name"] == "my_doc"


def test_answers_are_striped():
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
    response = AnswerResponse(answer=answer, usage={}, bibtex={}, status="success")

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
        ({"tool_names": {GenerateAnswerTool.__fields__["name"].default}}, None),
        ({"tool_names": set()}, ValidationError),
        ({"tool_names": {PaperSearchTool.__fields__["name"].default}}, ValidationError),
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
