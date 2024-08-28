from __future__ import annotations
import inspect
import logging
import os
import re
import sys
from typing import ClassVar
import anyio
from langchain_core.callbacks import BaseCallbackHandler

from langchain.tools import BaseTool
from paperqa import Answer, Docs
from ..utils import pqa_directory
from pydantic import BaseModel, ConfigDict, Field

# ruff: noqa: I001
from pydantic.v1 import (  # TODO: move to Pydantic v2 after LangChain moves to Pydantic v2, SEE: https://github.com/langchain-ai/langchain/issues/16306
    BaseModel as BaseModelV1,
    Field as FieldV1,
)

from .helpers import compute_total_model_token_cost, get_year
from .search import get_directory_index
from .models import ParsingConfiguration, QueryRequest, SimpleProfiler

logger = logging.getLogger(__name__)


async def status(docs: Docs, answer: Answer, relevant_score_cutoff: int = 5) -> str:
    """Create a string that provides a summary of the input doc/answer."""
    answer.cost = compute_total_model_token_cost(answer.token_counts)
    return (
        f"Status: Paper Count={len(docs.docs)}"
        f" | Relevant Papers={len({c.text.doc.dockey for c in answer.contexts if c.score > relevant_score_cutoff})}"
        f" | Current Evidence={len([c for c in answer.contexts if c.score > relevant_score_cutoff])}"
        f" | Current Cost=${answer.cost:.2f}"
    )


class SharedToolState(BaseModel):
    """Shared collection of variables for collection of tools. We use this to avoid
    the fact that pydantic treats dictionaries as values, instead of references.
    """  # noqa: D205

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    answer: Answer
    docs: Docs
    profiler: SimpleProfiler = Field(default_factory=SimpleProfiler)

    # SEE: https://regex101.com/r/RmuVdC/1
    STATUS_SEARCH_REGEX_PATTERN: ClassVar[str] = (
        r"Status: Paper Count=(\d+) \| Relevant Papers=(\d+) \| Current Evidence=(\d+)"
    )

    async def get_status(self) -> str:
        return await status(self.docs, self.answer)


def _time_tool(func):
    """Decorator to time the execution of a tool method.
    Assumes that the tool has a shared state.
    """  # noqa: D205

    async def wrapper(self, *args, **kwargs):
        async with self.shared_state.profiler.timer(self.name):
            return await func(self, *args, **kwargs)

    return wrapper


class PaperSearchTool(BaseTool):
    class InputSchema(
        BaseModelV1  # TODO: move to Pydantic v2 after LangChain moves to Pydantic v2, SEE: https://github.com/langchain-ai/langchain/issues/16306
    ):
        query: str = FieldV1(
            description=(
                "A search query in this format: [query], [start year]-[end year]. "
                "You may include years as the last word in the query, "
                "e.g. 'machine learning 2020' or 'machine learning 2010-2020'. "
                f"The current year is {get_year()}. "
                "The query portion can be a specific phrase, complete sentence, "
                "or general keywords, e.g. 'machine learning for immunology'."
            )
        )

    paper_directory: str | os.PathLike = "."
    index_directory: str | os.PathLike | None = None
    manifest_file: str | os.PathLike | None = None
    name: str = "paper_search"
    args_schema: type[BaseModelV1] | None = InputSchema
    description: str = (
        "Search for papers to increase the paper count. You can call this a second "
        "time with an different search to gather more papers."
    )

    shared_state: SharedToolState
    return_paper_metadata: bool = False
    # Second item being True means specify a year range in the search
    search_type: tuple[str, bool] = ("google", False)
    search_count: int = 8
    previous_searches: dict[str, int] = FieldV1(default_factory=dict)
    embedding: str = "text-embedding-3-small"
    parsing_configuration: ParsingConfiguration = FieldV1(
        default_factory=ParsingConfiguration
    )

    def _run(self, query: str) -> str:
        raise NotImplementedError

    @_time_tool
    async def _arun(self, query: str) -> str:
        """
        Run asynchronously, in-place mutating `self.shared_state.docs`.

        Args:
            query: Search keywords followed by optional year or year range
                (e.g. COVID-19 vaccines, 2022).

        Returns:
            String describing searched papers and the current status.
        """
        # get offset if we've done this search before (continuation of search)
        # or mark this search as new (so offset 0)
        logger.info(f"Starting paper search for '{query}'.")
        search_key = query
        if search_key in self.previous_searches:
            offset = self.previous_searches[search_key]
        else:
            offset = self.previous_searches[search_key] = 0

        # Preprocess inputs to make ScrapeRequest
        keywords = query.replace('"', "")  # Remove quotes
        year: str | None = None
        last_word = keywords.split(" ")[-1]
        if re.match(r"\d{4}(-\d{4})?", last_word):
            keywords = keywords[: -len(last_word)].removesuffix(",").strip()
            if self.search_type[1]:
                year = last_word
                if "-" not in year:
                    year = year + "-" + year  # Convert to date range (e.g. 2022-2022)
        index = await get_directory_index(
            directory=anyio.Path(self.paper_directory),
            index_name=QueryRequest.get_index_name(
                self.paper_directory, self.embedding, self.parsing_configuration
            ),
            index_directory=self.index_directory,
            manifest_file=(
                anyio.Path(self.manifest_file) if self.manifest_file else None
            ),
            embedding=self.embedding,
            chunk_chars=self.parsing_configuration.chunksize,
            overlap=self.parsing_configuration.overlap,
        )

        results = await index.query(
            keywords,
            top_n=self.search_count,
            offset=offset,
            field_subset=[f for f in index.fields if f != "year"],
        )

        logger.info(f"Search for '{keywords}' returned {len(results)} papers.")
        # combine all the resulting doc objects into one and update the state
        # there's only one doc per result, so we can just take the first one
        all_docs = []
        for r in results:
            this_doc = next(iter(r.docs.values()))
            all_docs.append(this_doc)
            await self.shared_state.docs.aadd_texts(texts=r.texts, doc=this_doc)

        status = await self.shared_state.get_status()

        logger.info(status)

        # mark how far we've searched so that continuation will start at the right place
        self.previous_searches[search_key] += self.search_count

        if self.return_paper_metadata:
            retrieved_papers = "\n".join([f"{x.title} ({x.year})" for x in all_docs])
            return f"Retrieved Papers:\n{retrieved_papers}\n\n{status}"
        return status


class EmptyDocsError(RuntimeError):
    """Error to throw when we needed docs to be present."""


class GatherEvidenceTool(BaseTool):
    class InputSchema(
        BaseModelV1  # TODO: move to Pydantic v2 after LangChain moves to Pydantic v2, SEE: https://github.com/langchain-ai/langchain/issues/16306
    ):
        question: str = FieldV1(description="Specific question to gather evidence for.")

    name: str = "gather_evidence"
    args_schema: type[BaseModelV1] | None = InputSchema
    description: str = (
        "Gather evidence from previous papers given a specific question. "
        "This will increase evidence and relevant paper counts. "
        "Only invoke when paper count is above zero."
    )

    shared_state: SharedToolState
    query: QueryRequest

    def _run(self, query: str) -> str:
        raise NotImplementedError

    @_time_tool
    async def _arun(self, question: str) -> str:
        if not self.shared_state.docs.docs:
            raise EmptyDocsError("Not gathering evidence due to having no papers.")

        logger.info(f"Gathering and ranking evidence for '{question}'.")

        # first we see if we'd like to filter any docs for relevance
        # at the citation level
        if len(self.shared_state.docs.docs) >= self.query.adoc_match_threshold:
            doc_keys_to_keep = await self.shared_state.docs.adoc_match(
                question,
                rerank=True,  # want to set it explicitly
                answer=self.shared_state.answer,
            )
        else:
            doc_keys_to_keep = set(self.shared_state.docs.docs.keys())

        self.shared_state.answer.dockey_filter = doc_keys_to_keep

        # swap out the question
        # TODO: evaluate how often does the agent changes the question
        old = self.shared_state.answer.question
        self.shared_state.answer.question = question

        # generator, so run it
        l0 = len(self.shared_state.answer.contexts)

        # set jit so that the index is rebuilt; helps if the texts have changed
        self.shared_state.docs.jit_texts_index = True
        # ensure length is set correctly
        self.shared_state.answer.summary_length = self.query.summary_length
        # TODO: refactor answer out of this...
        self.shared_state.answer = await self.shared_state.docs.aget_evidence(
            answer=self.shared_state.answer,
            max_sources=self.query.max_sources,
            k=self.query.consider_sources,
            detailed_citations=True,
        )
        l1 = len(self.shared_state.answer.contexts)
        self.shared_state.answer.question = old
        sorted_contexts = sorted(
            self.shared_state.answer.contexts, key=lambda x: x.score, reverse=True
        )
        best_evidence = ""
        if len(sorted_contexts) > 0:
            best_evidence = f" Best evidence:\n\n{sorted_contexts[0].context}"
        status = await self.shared_state.get_status()
        logger.info(status)
        return f"Added {l1 - l0} pieces of evidence.{best_evidence}\n\n" + status


class GenerateAnswerTool(BaseTool):
    class InputSchema(
        BaseModelV1  # TODO: move to Pydantic v2 after LangChain moves to Pydantic v2, SEE: https://github.com/langchain-ai/langchain/issues/16306
    ):
        question: str = FieldV1(description="Question to be answered.")

    name: str = "gen_answer"
    args_schema: type[BaseModelV1] | None = InputSchema
    description: str = (
        "Ask a model to propose an answer answer using current evidence. "
        "The tool may fail, "
        "indicating that better or different evidence should be found. "
        "Having more than one piece of evidence or relevant papers is best."
    )
    shared_state: SharedToolState
    wipe_context_on_answer_failure: bool = True
    query: QueryRequest

    FAILED_TO_ANSWER: ClassVar[str] = "Failed to answer question."

    @classmethod
    def did_not_fail_to_answer(cls, message: str) -> bool:
        return not message.startswith(cls.FAILED_TO_ANSWER)

    def _run(self, query: str) -> str:
        raise NotImplementedError

    @_time_tool
    async def _arun(self, question: str) -> str:
        logger.info(f"Generating answer for '{question}'.")
        # TODO: Should we allow the agent to change the question?
        # self.answer.question = query
        self.shared_state.answer.answer_length = self.query.length
        self.shared_state.answer = await self.shared_state.docs.aquery(
            self.query.query,
            k=self.query.consider_sources,
            max_sources=self.query.max_sources,
            answer=self.shared_state.answer,
        )

        if self.query.filter_extra_background:
            # filter out "(Extra Background Information)" from the answer
            self.shared_state.answer.answer = re.sub(
                r"\([Ee]xtra [Bb]ackground [Ii]nformation\)",
                "",
                self.shared_state.answer.answer,
            )

        if "cannot answer" in self.shared_state.answer.answer.lower():
            if self.wipe_context_on_answer_failure:
                self.shared_state.answer.contexts = []
                self.shared_state.answer.dockey_filter = None
                self.shared_state.answer.context = ""
            status = await self.shared_state.get_status()
            logger.info(status)
            return f"{self.FAILED_TO_ANSWER} | " + status
        status = await self.shared_state.get_status()
        logger.info(status)
        return f"{self.shared_state.answer.answer} | {status}"

    # NOTE: can match failure to answer or an actual answer
    ANSWER_SPLIT_REGEX_PATTERN: ClassVar[str] = (
        r" \| " + SharedToolState.STATUS_SEARCH_REGEX_PATTERN
    )

    @classmethod
    def extract_answer_from_message(cls, content: str) -> str:
        """Extract the answer from a message content."""
        answer, *rest = re.split(
            pattern=cls.ANSWER_SPLIT_REGEX_PATTERN, string=content, maxsplit=1
        )
        if len(rest) != 4 or not cls.did_not_fail_to_answer(answer):  # noqa: PLR2004
            return ""
        return answer


AVAILABLE_TOOL_NAME_TO_CLASS: dict[str, type[BaseTool]] = {
    cls.__fields__["name"].default: cls
    for _, cls in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda v: inspect.isclass(v)
        and issubclass(v, BaseTool)
        and v is not BaseTool,
    )
}


def query_to_tools(
    query: QueryRequest,
    state: SharedToolState,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> list[BaseTool]:
    if query.agent_tools.tool_names is None:
        tool_types: list[type[BaseTool]] = [
            PaperSearchTool,
            GatherEvidenceTool,
            GenerateAnswerTool,
        ]
    else:
        tool_types = [
            AVAILABLE_TOOL_NAME_TO_CLASS[name]
            for name in set(query.agent_tools.tool_names)
        ]
    tools: list[BaseTool] = []
    for tool_type in tool_types:
        if issubclass(tool_type, PaperSearchTool):
            tools.append(
                PaperSearchTool(
                    shared_state=state,
                    search_count=query.agent_tools.search_count,
                    embedding=query.embedding,
                    parsing_configuration=query.parsing_configuration,
                    paper_directory=query.agent_tools.paper_directory,
                    index_directory=query.agent_tools.index_directory
                    or pqa_directory("indexes"),
                    manifest_file=query.agent_tools.manifest_file,
                    callbacks=callbacks,
                )
            )
        elif issubclass(tool_type, GatherEvidenceTool):
            tools.append(
                GatherEvidenceTool(shared_state=state, query=query, callbacks=callbacks)
            )
        elif issubclass(tool_type, GenerateAnswerTool):
            tools.append(
                GenerateAnswerTool(
                    shared_state=state,
                    wipe_context_on_answer_failure=query.agent_tools.wipe_context_on_answer_failure,
                    query=query,
                    callbacks=callbacks,
                )
            )
        else:
            tools.append(tool_type(shared_state=state))
    return tools
