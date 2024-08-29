from __future__ import annotations

import itertools
import json
import logging
import re
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import anyio
import pytest
from pydantic import ValidationError
from pytest_subtests import SubTests

from paperqa.docs import Docs
from paperqa.llms import LangchainLLMModel
from paperqa.types import Answer, Context, Doc, PromptCollection, Text
from paperqa.utils import get_year

try:
    from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
    from langchain_openai import ChatOpenAI
    from tenacity import Retrying, retry_if_exception_type, stop_after_attempt

    from paperqa.agents import agent_query
    from paperqa.agents.helpers import (
        compute_total_model_token_cost,
        update_doc_models,
    )
    from paperqa.agents.models import (
        AgentPromptCollection,
        AgentStatus,
        AnswerResponse,
        MismatchedModelsError,
        QueryRequest,
    )
    from paperqa.agents.prompts import STATIC_PROMPTS
    from paperqa.agents.search import get_directory_index
    from paperqa.agents.tools import (
        GatherEvidenceTool,
        GenerateAnswerTool,
        PaperSearchTool,
        SharedToolState,
    )
except ImportError:
    pytest.skip("agents module is not installed", allow_module_level=True)


PAPER_DIRECTORY = Path(__file__).parent


@pytest.mark.asyncio
async def test_get_directory_index(agent_index_dir):
    index = await get_directory_index(
        directory=anyio.Path(PAPER_DIRECTORY),
        index_name="pqa_index_0",
        index_directory=agent_index_dir,
    )
    assert index.fields == [
        "title",
        "file_location",
        "body",
        "year",
    ], "Incorrect fields in index"
    assert len(await index.index_files) == 4, "Incorrect number of index files"
    results = await index.query(query="who is Frederick Bates?")
    # added docs.keys come from md5 hash of the file location
    assert results[0].docs.keys() == {"dab5b86dea3bd4c7ffe05a9f33ae95f7"}


@pytest.mark.asyncio
async def test_get_directory_index_w_manifest(agent_index_dir):
    index = await get_directory_index(
        directory=anyio.Path(PAPER_DIRECTORY),
        index_name="pqa_index_0",
        index_directory=agent_index_dir,
        manifest_file=anyio.Path(PAPER_DIRECTORY) / "stub_manifest.csv",
    )
    assert index.fields == [
        "title",
        "file_location",
        "body",
        "year",
    ], "Incorrect fields in index"
    # 4 = example.txt + example2.txt + paper.pdf + example.html
    assert len(await index.index_files) == 4, "Incorrect number of index files"
    results = await index.query(query="who is Barack Obama?")
    top_result = next(iter(results[0].docs.values()))
    assert top_result.dockey == "af2c9acf6018e62398fc6efc4f0a04b4"
    # note: this title comes from the manifest, so we know it worked
    assert top_result.title == "Barack Obama (Wikipedia article)"


@pytest.mark.flaky(reruns=3, only_rerun=["AssertionError", "httpx.RemoteProtocolError"])
@pytest.mark.parametrize("agent_type", ["OpenAIFunctionsAgent", "fake"])
@pytest.mark.asyncio
async def test_agent_types(agent_index_dir, agent_type):

    question = "How can you use XAI for chemical property prediction?"

    request = QueryRequest(
        query=question,
        consider_sources=10,
        max_sources=2,
        embedding="sparse",
        agent_tools=AgentPromptCollection(
            search_count=2,
            paper_directory=PAPER_DIRECTORY,
            index_directory=agent_index_dir,
        ),
    )
    response = await agent_query(
        request, agent_type=agent_type, index_directory=agent_index_dir
    )
    assert response.answer.answer != "I cannot answer", "Answer not generated"
    assert len(response.answer.context) >= 1, "No contexts were found"
    assert response.answer.question == question


@pytest.mark.asyncio
async def test_timeout(agent_index_dir):
    response = await agent_query(
        QueryRequest(
            query="Are COVID-19 vaccines effective?",
            llm="gpt-4o-mini",
            prompts=PromptCollection(pre=None),
            # We just need one tool to test the timeout, gen_answer is not that fast
            agent_tools=AgentPromptCollection(
                timeout=0.001,
                tool_names={"gen_answer"},
                paper_directory=PAPER_DIRECTORY,
                index_directory=agent_index_dir,
            ),
        ),
        Docs(),
    )
    # ensure that GenerateAnswerTool was called
    assert response.status == AgentStatus.TIMEOUT, "Agent did not timeout"
    assert "I cannot answer" in response.answer.answer


@pytest.mark.asyncio
async def test_propagate_options(agent_index_dir) -> None:
    llm_name = "gpt-4o-mini"
    default_llm_names = {
        cls.model_fields[name].default  # type: ignore[attr-defined]
        for name, cls in itertools.product(("llm", "summary_llm"), (QueryRequest, Docs))
    }
    assert (
        llm_name not in default_llm_names
    ), f"Assertions require not matching a default LLM name in {default_llm_names}."
    query = QueryRequest(
        query="What is is a self-explanatory model?",
        summary_llm=llm_name,
        llm=llm_name,
        max_sources=5,
        consider_sources=6,
        length="400 words",
        prompts=PromptCollection(
            pre=None, system="End all responses with ###", skip_summary=True
        ),
        # NOTE: this is testing that if our prompt forgets template fields (e.g. status),
        # the code still runs, despite the presence of extra keyword arguments to format
        agent_tools=AgentPromptCollection(
            paper_directory=PAPER_DIRECTORY,
            index_directory=agent_index_dir,
            agent_prompt=(
                "Answer question: {question}. Search for papers, gather evidence, and"
                " answer. If you do not have enough evidence, you can search for more"
                " papers (preferred) or gather more evidence with a different phrase."
                " You may rephrase or break-up the question in those steps. Once you"
                " have five or more pieces of evidence from multiple sources, or you"
                " have tried a few times, call {gen_answer_tool_name} tool. The"
                " {gen_answer_tool_name} tool output is visible to the user, so you do"
                " not need to restate the answer and can simply terminate if the answer"
                " looks sufficient."
            ),
            tool_names={"paper_search", "gather_evidence", "gen_answer"},
        ),
    )
    for attempt in Retrying(
        stop=stop_after_attempt(3), retry=retry_if_exception_type(AssertionError)
    ):
        with attempt:
            docs = Docs(llm=llm_name, summary_llm=llm_name)
            docs.prompts = query.prompts  # this line happens in main
            response = await agent_query(query, docs, agent_type="fake")
            assert response.status == AgentStatus.SUCCESS, "Agent did not succeed"
    result = response.answer
    assert len(result.answer) > 200, "Answer did not return any results"
    assert result.answer_length == query.length, "Answer length did not propagate"
    assert "###" in result.answer, "Answer did not propagate system prompt"


@pytest.mark.flaky(reruns=3, only_rerun=["AssertionError"])
@pytest.mark.asyncio
async def test_mixing_langchain_clients(caplog, agent_index_dir) -> None:
    docs = Docs()
    query = QueryRequest(
        query="What is is a self-explanatory model?",
        max_sources=2,
        consider_sources=3,
        llm="gemini-1.5-flash",
        summary_llm="gemini-1.5-flash",
        agent_tools=AgentPromptCollection(
            paper_directory=PAPER_DIRECTORY, index_directory=agent_index_dir
        ),
    )
    update_doc_models(docs, query)
    with caplog.at_level(logging.WARNING):
        response = await agent_query(query, docs)
    assert response.status == AgentStatus.SUCCESS, "Agent did not succeed"
    assert not [
        msg for (*_, msg) in caplog.record_tuples if "error" in msg.lower()
    ], "Expected clean logs"


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
        response = await agent_query(
            query=QueryRequest(
                query="Are COVID-19 vaccines effective?",
                agent_tools=AgentPromptCollection(
                    tool_names={"gather_evidence", "gen_answer"}
                ),
            ),
            docs=Docs(),
        )
    assert response.status == AgentStatus.FAIL, "Agent should have registered a failure"


@pytest.mark.flaky(reruns=3, only_rerun=["AssertionError"])
@pytest.mark.asyncio
async def test_agent_sharing_state(
    fixture_stub_answer, subtests: SubTests, agent_index_dir
) -> None:
    tool_state = SharedToolState(docs=Docs(), answer=fixture_stub_answer)
    search_count = 3  # Keep low for speed
    query = QueryRequest(
        query=fixture_stub_answer.question,
        consider_sources=2,
        max_sources=1,
        agent_tools=AgentPromptCollection(
            search_count=search_count,
            index_directory=agent_index_dir,
            paper_directory=PAPER_DIRECTORY,
        ),
    )

    with subtests.test(msg=PaperSearchTool.__name__):
        tool = PaperSearchTool(
            shared_state=tool_state,
            search_count=search_count,
            index_directory=agent_index_dir,
            paper_directory=PAPER_DIRECTORY,
        )
        await tool.arun("XAI self explanatory model")
        assert tool_state.docs.docs, "Search did not save any papers"
        assert all(
            (isinstance(d, Doc) or issubclass(d, Doc))  # type: ignore[unreachable]
            for d in tool_state.docs.docs.values()
        ), "Document type or DOI propagation failure"

    with subtests.test(msg=GatherEvidenceTool.__name__):
        assert (
            not fixture_stub_answer.contexts
        ), "No contexts is required for a later assertion"

        tool = GatherEvidenceTool(shared_state=tool_state, query=query)
        await tool.arun(fixture_stub_answer.question)
        assert (
            len(fixture_stub_answer.dockey_filter) > 0
        ), "Filter did not preserve reference"
        assert fixture_stub_answer.contexts, "Evidence did not return any results"

    with subtests.test(msg=f"{GenerateAnswerTool.__name__} working"):
        tool = GenerateAnswerTool(shared_state=tool_state, query=query)
        result = await tool.arun(fixture_stub_answer.question)
        assert re.search(
            pattern=SharedToolState.STATUS_SEARCH_REGEX_PATTERN, string=result
        )
        assert (
            len(fixture_stub_answer.answer) > 200
        ), "Answer did not return any results"
        assert (
            GenerateAnswerTool.extract_answer_from_message(result)
            == fixture_stub_answer.answer
        ), "Failed to regex extract answer from result"
        assert (
            len(fixture_stub_answer.contexts) <= query.max_sources
        ), "Answer has more sources than expected"

    with subtests.test(msg=f"{GenerateAnswerTool.__name__} misconfigured query"):
        query.consider_sources = 1  # k
        query.max_sources = 5
        tool = GenerateAnswerTool(shared_state=tool_state, query=query)
        with pytest.raises(ValueError, match="k should be greater than max_sources"):
            await tool.arun("Are COVID-19 vaccines effective?")


def test_functions() -> None:
    """Check the functions schema passed to OpenAI."""
    shared_tool_state = SharedToolState(
        answer=Answer(question="stub"),
        docs=Docs(),
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


def test_instruct_model():
    docs = Docs(name="tmp")
    query = QueryRequest(
        summary_llm="gpt-3.5-turbo-instruct",
        query="Are COVID-19 vaccines effective?",
        llm="gpt-3.5-turbo-instruct",
    )
    update_doc_models(docs, query)
    docs.query("Are COVID-19 vaccines effective?")
    query.llm = "gpt-3.5-turbo-instruct"
    query.summary_llm = "gpt-3.5-turbo-instruct"
    update_doc_models(docs, query)
    docs.query("Are COVID-19 vaccines effective?")


def test_anthropic_model():
    docs = Docs(name="tmp")
    query = QueryRequest(
        summary_llm="claude-3-sonnet-20240229",
        query="Are COVID-19 vaccines effective?",
        llm="claude-3-sonnet-20240229",
    )
    update_doc_models(docs, query)
    answer = docs.query("Are COVID-19 vaccines effective?")

    # make sure we can compute cost with this model.
    compute_total_model_token_cost(answer.token_counts)


def test_embeddings_anthropic():
    docs = Docs(name="tmp")
    query = QueryRequest(
        summary_llm="claude-3-sonnet-20240229",
        query="Are COVID-19 vaccines effective?",
        llm="claude-3-sonnet-20240229",
        embedding="sparse",
    )
    update_doc_models(docs, query)
    _ = docs.query("Are COVID-19 vaccines effective?")

    query = QueryRequest(
        summary_llm="claude-3-sonnet-20240229",
        query="Are COVID-19 vaccines effective?",
        llm="claude-3-sonnet-20240229",
        embedding="hybrid-text-embedding-3-small",
    )
    update_doc_models(docs, query)
    _ = docs.query("Are COVID-19 vaccines effective?")


@pytest.mark.asyncio
async def test_gemini_model_construction(
    stub_paper_path: Path,
) -> None:
    docs = Docs(name="tmp")
    query = QueryRequest(
        summary_llm="gemini-1.5-pro",
        llm="gemini-1.5-pro",
        embedding="sparse",
    )
    update_doc_models(docs, query)
    assert isinstance(docs.llm_model, LangchainLLMModel)  # We use LangChain for Gemini
    assert docs.llm_model.name == "gemini-1.5-pro", "Gemini Model: model not created"
    assert "model" not in docs.llm_model.config, "model should not be in config"

    # now try using it
    await docs.aadd(stub_paper_path)
    answer = await docs.aget_evidence(
        Answer(question="Are COVID-19 vaccines effective?")
    )
    assert answer.contexts, "Gemini Model: no contexts returned"


def test_query_request_summary():
    """Test that we can set summary llm to none and it will skip summary."""
    request = QueryRequest(query="Are COVID-19 vaccines effective?", summary_llm="none")
    assert request.summary_llm == "gpt-4o-mini"
    assert request.prompts.skip_summary, "Summary should be skipped with none llm"


def test_query_request_preset_prompts():
    """Test that we can set the prompt using our preset prompts."""
    request = QueryRequest(
        query="Are COVID-19 vaccines effective?",
        prompts=PromptCollection(
            summary_json_system=r"{gene_name} {summary} {relevance_score}"
        ),
    )
    assert "gene_name" in request.prompts.summary_json_system


def test_query_request_docs_name_serialized():
    """Test that the query request has a docs_name property."""
    request = QueryRequest(query="Are COVID-19 vaccines effective?")
    request_data = json.loads(request.model_dump_json())
    assert "docs_name" in request_data
    assert request_data["docs_name"] is None
    request.set_docs_name("my_doc")
    request_data = json.loads(request.model_dump_json())
    assert request_data["docs_name"] == "my_doc"


def test_query_request_model_mismatch():
    with pytest.raises(
        MismatchedModelsError,
        match=(
            "Answer LLM and summary LLM types must match: "
            "<class 'paperqa.llms.AnthropicLLMModel'> != <class 'paperqa.llms.OpenAILLMModel'>"
        ),
    ):
        _ = QueryRequest(
            summary_llm="gpt-4o-mini",
            query="Are COVID-19 vaccines effective?",
            llm="claude-3-sonnet-20240229",
            embedding="sparse",
        )


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


def test_prompts_are_set():
    assert (
        STATIC_PROMPTS["json"].summary_json_system
        != PromptCollection().summary_json_system
    )


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
        AgentPromptCollection(**kwargs)
    else:
        with pytest.raises(result):
            AgentPromptCollection(**kwargs)
