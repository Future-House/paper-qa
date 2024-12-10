"""Base classes for tools, implemented in a functional manner."""

import asyncio
import inspect
import logging
import re
import sys
from collections.abc import Callable
from itertools import chain
from typing import ClassVar, Self, cast

from aviary.core import ToolRequestMessage
from pydantic import BaseModel, ConfigDict, Field, computed_field

from paperqa.docs import Docs
from paperqa.llms import EmbeddingModel, LiteLLMModel
from paperqa.settings import Settings
from paperqa.types import DocDetails, Embeddable, PQASession

from .search import get_directory_index

logger = logging.getLogger(__name__)


def make_status(
    total_paper_count: int, relevant_paper_count: int, evidence_count: int, cost: float
) -> str:
    return (
        f"Status: Paper Count={total_paper_count}"
        f" | Relevant Papers={relevant_paper_count} | Current Evidence={evidence_count}"
        f" | Current Cost=${cost:.4f}"
    )


def default_status(state: "EnvironmentState") -> str:
    return make_status(
        total_paper_count=len(state.docs.docs),
        relevant_paper_count=len(
            {
                c.text.doc.dockey
                for c in state.session.contexts
                if c.score > state.RELEVANT_SCORE_CUTOFF
            }
        ),
        evidence_count=len(
            [c for c in state.session.contexts if c.score > state.RELEVANT_SCORE_CUTOFF]
        ),
        cost=state.session.cost,
    )


class EnvironmentState(BaseModel):
    """State here contains documents and answer being populated."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    docs: Docs
    session: PQASession = Field(..., alias="answer")
    status_fn: Callable[[Self], str] | None = Field(
        default=None,
        description=(
            "Function used to generate status,"
            " uses `paperqa.agents.tools.default_status` "
            "if not provided."
        ),
    )

    # SEE: https://regex101.com/r/RmuVdC/1
    STATUS_SEARCH_REGEX_PATTERN: ClassVar[str] = (
        r"Status: Paper Count=(\d+) \| Relevant Papers=(\d+) \| Current Evidence=(\d+)"
    )
    RELEVANT_SCORE_CUTOFF: ClassVar[int] = 5

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> str:
        if self.status_fn is not None:
            return self.status_fn(cast(Self, self))
        return default_status(self)

    def record_action(self, action: ToolRequestMessage) -> None:
        self.session.add_tokens(action)
        self.session.tool_history.append([tc.function.name for tc in action.tool_calls])

    def query_tool_history(self, tool_name: str) -> bool:
        """Return true if the tool is has been called in history."""
        return tool_name in set(chain.from_iterable(self.session.tool_history))


class NamedTool(BaseModel):
    """Base class to make looking up tools easier."""

    TOOL_FN_NAME: ClassVar[str] = (
        "# unpopulated"  # Comment symbol ensures no collisions
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class PaperSearch(NamedTool):
    TOOL_FN_NAME = "paper_search"

    settings: Settings
    embedding_model: EmbeddingModel
    previous_searches: dict[tuple[str, str | None], int] = Field(default_factory=dict)

    async def paper_search(
        self,
        query: str,
        min_year: int | None,
        max_year: int | None,
        state: EnvironmentState,
    ) -> str:
        """
        Search for papers to increase the paper count.

        Repeat previous calls with the same query and years to continue a search. Only repeat a maximum of twice.
        This tool can be called concurrently.
        This tool introduces novel papers, so invoke this tool when just beginning or when unsatisfied with the current evidence.

        Args:
            query: A search query, which can be a specific phrase, complete sentence,
                or general keywords, e.g. 'machine learning for immunology'. Also can be
                given search operators.
            min_year: Filter for minimum publication year, or None for no minimum year.
                The current year is {current_year}.
            max_year: Filter for maximum publication year, or None for no maximum year.
                The current year is {current_year}.
            state: Current state.

        Returns:
            String describing searched papers and the current status.
        """  # noqa: E501,W505
        # Convert to date range (e.g. 2022-2022) if date is present
        year = (
            f"{min_year if min_year else ''}-{max_year if max_year else ''}"  # noqa: FURB110
            if (min_year or max_year)
            else None
        )
        # get offset if we've done this search before (continuation of search)
        # or mark this search as new (so offset 0)
        search_key = query, year
        try:
            offset = self.previous_searches[search_key]
        except KeyError:
            offset = self.previous_searches[search_key] = 0

        logger.info(f"Starting paper search for {query!r}.")
        index = await get_directory_index(settings=self.settings, build=False)
        results: list[Docs] = await index.query(
            query,
            top_n=self.settings.agent.search_count,
            offset=offset,
            field_subset=[f for f in index.fields if f != "year"],
        )
        logger.info(
            f"{self.TOOL_FN_NAME} for query {query!r} and offset {offset} returned"
            f" {len(results)} papers."
        )

        # combine all the resulting doc objects into one and update the state
        all_doc_details: list[DocDetails] = []
        for r in results:
            # there's only one doc per result, so just take the first one
            this_doc_details = cast(DocDetails, next(iter(r.docs.values())))
            all_doc_details.append(this_doc_details)
            await state.docs.aadd_texts(
                texts=r.texts,
                doc=this_doc_details,
                settings=self.settings,
                embedding_model=self.embedding_model,
            )

        status = state.status
        logger.info(status)
        # mark how far we've searched so that continuation will start at the right place
        self.previous_searches[search_key] += self.settings.agent.search_count
        if self.settings.agent.return_paper_metadata:
            retrieved_papers = "\n".join(
                [f"{x.title} ({x.year})" for x in all_doc_details]
            )
            return f"Retrieved Papers:\n{retrieved_papers}\n\n{status}"
        return status


class EmptyDocsError(RuntimeError):
    """Error to throw when we needed docs to be present."""


class GatherEvidence(NamedTool):
    TOOL_FN_NAME = "gather_evidence"

    settings: Settings
    summary_llm_model: LiteLLMModel
    embedding_model: EmbeddingModel
    partitioning_fn: Callable[[Embeddable], int] | None = None

    async def gather_evidence(self, question: str, state: EnvironmentState) -> str:
        """
        Gather evidence from previous papers given a specific question to increase evidence and relevant paper counts.

        A valuable time to invoke this tool is right after another tool increases paper count.
        Feel free to invoke this tool in parallel with other tools, but do not call this tool in parallel with itself.
        Only invoke this tool when the paper count is above zero, or this tool will be useless.

        Args:
            question: Specific question to gather evidence for.
            state: Current state.

        Returns:
            String describing gathered evidence and the current status.
        """
        if not state.docs.docs:
            raise EmptyDocsError("Not gathering evidence due to having no papers.")

        if f"{self.TOOL_FN_NAME}_initialized" in self.settings.agent.callbacks:
            await asyncio.gather(
                *(
                    c(state)
                    for c in self.settings.agent.callbacks[
                        f"{self.TOOL_FN_NAME}_initialized"
                    ]
                )
            )

        logger.info(f"{self.TOOL_FN_NAME} starting for question {question!r}.")
        original_question = state.session.question
        try:
            # Swap out the question with the more specific question
            # TODO: remove this swap, as it prevents us from supporting parallel calls
            state.session.question = question
            l0 = len(state.session.contexts)

            # TODO: refactor answer out of this...
            state.session = await state.docs.aget_evidence(
                query=state.session,
                settings=self.settings,
                embedding_model=self.embedding_model,
                summary_llm_model=self.summary_llm_model,
                partitioning_fn=self.partitioning_fn,
                callbacks=self.settings.agent.callbacks.get(
                    f"{self.TOOL_FN_NAME}_aget_evidence"
                ),
            )
            l1 = len(state.session.contexts)
        finally:
            state.session.question = original_question

        status = state.status
        logger.info(status)
        sorted_contexts = sorted(
            state.session.contexts, key=lambda x: x.score, reverse=True
        )
        best_evidence = (
            f" Best evidence:\n\n{sorted_contexts[0].context}"
            if sorted_contexts
            else ""
        )

        if f"{self.TOOL_FN_NAME}_completed" in self.settings.agent.callbacks:
            await asyncio.gather(
                *(
                    callback(state)
                    for callback in self.settings.agent.callbacks[
                        f"{self.TOOL_FN_NAME}_completed"
                    ]
                )
            )

        return f"Added {l1 - l0} pieces of evidence.{best_evidence}\n\n" + status


class GenerateAnswer(NamedTool):
    TOOL_FN_NAME = "gen_answer"

    settings: Settings
    llm_model: LiteLLMModel
    summary_llm_model: LiteLLMModel
    embedding_model: EmbeddingModel
    partitioning_fn: Callable[[Embeddable], int] | None = None

    async def gen_answer(self, state: EnvironmentState) -> str:
        """
        Generate an answer using current evidence.

        The tool may fail, indicating that better or different evidence should be found.
        Aim for at least five pieces of evidence from multiple sources before invoking this tool.
        Feel free to invoke this tool in parallel with other tools, but do not call this tool in parallel with itself.

        Args:
            state: Current state.
        """
        logger.info(f"Generating answer for '{state.session.question}'.")

        if f"{self.TOOL_FN_NAME}_initialized" in self.settings.agent.callbacks:
            await asyncio.gather(
                *(
                    callback(state)
                    for callback in self.settings.agent.callbacks[
                        f"{self.TOOL_FN_NAME}_initialized"
                    ]
                )
            )

        state.session = await state.docs.aquery(
            query=state.session,
            settings=self.settings,
            llm_model=self.llm_model,
            summary_llm_model=self.summary_llm_model,
            embedding_model=self.embedding_model,
            partitioning_fn=self.partitioning_fn,
            callbacks=self.settings.agent.callbacks.get(
                f"{self.TOOL_FN_NAME}_aget_query"
            ),
        )

        answer = state.session.answer
        status = state.status
        logger.info(status)

        if f"{self.TOOL_FN_NAME}_completed" in self.settings.agent.callbacks:
            await asyncio.gather(
                *(
                    callback(state)
                    for callback in self.settings.agent.callbacks[
                        f"{self.TOOL_FN_NAME}_completed"
                    ]
                )
            )

        return f"{answer} | {status}"

    # Use to separate answer from status
    # NOTE: can match failure to answer or an actual answer
    ANSWER_SPLIT_REGEX_PATTERN: ClassVar[str] = (
        r" \| " + EnvironmentState.STATUS_SEARCH_REGEX_PATTERN
    )

    @classmethod
    def extract_answer_from_message(cls, content: str) -> str:
        """Extract the answer from a message content."""
        answer, *rest = re.split(
            pattern=cls.ANSWER_SPLIT_REGEX_PATTERN, string=content, maxsplit=1
        )
        return answer if len(rest) == 4 else ""  # noqa: PLR2004


class Reset(NamedTool):
    TOOL_FN_NAME = "reset"

    async def reset(self, state: EnvironmentState) -> None:
        """
        Reset by clearing all current evidence from the system.

        This tool is useful when repeatedly failing to answer because the existing evidence may unsuitable for the question.
        It does not make sense to call this tool in parallel with other tools, as its resetting all state.
        Only invoke this tool when the current evidence is above zero, or this tool will be useless.
        """  # noqa: E501,W505
        logger.info(f"Resetting '{state.session.question}'.")
        state.session.contexts = []
        state.session.context = ""


class Complete(NamedTool):
    TOOL_FN_NAME = "complete"

    # Use to separate certainty from status
    CERTAINTY_SPLIT_REGEX_PATTERN: ClassVar[str] = (
        r" \| " + EnvironmentState.STATUS_SEARCH_REGEX_PATTERN
    )

    NO_ANSWER_PHRASE: ClassVar[str] = "No answer generated."

    async def complete(
        self, has_successful_answer: bool, state: EnvironmentState
    ) -> str:
        """
        Terminate using the last proposed answer.

        Do not invoke this tool in parallel with other tools or itself.

        Args:
            has_successful_answer: Set True if an answer that addresses all parts of the
                task has been generated, otherwise set False to indicate unsureness.
            state: Current state.
        """
        # TODO: eliminate race condition here if agent calls 2+ times in parallel
        # with opposite has_successful_answer values
        state.session.has_successful_answer = has_successful_answer

        if not state.session.answer:
            state.session.answer = self.NO_ANSWER_PHRASE

        logger.info(
            f"Completing '{state.session.question}' as"
            f" '{'a success' if has_successful_answer else 'unsure'}'."
        )
        # Return answer and status to simplify postprocessing of tool response
        return f"{'Success' if has_successful_answer else 'Unsure'} | {state.status}"


AVAILABLE_TOOL_NAME_TO_CLASS: dict[str, type[NamedTool]] = {
    cls.TOOL_FN_NAME: cls
    for _, cls in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda v: inspect.isclass(v)
        and issubclass(v, NamedTool)
        and v is not NamedTool,
    )
}
