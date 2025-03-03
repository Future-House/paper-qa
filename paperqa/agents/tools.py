"""Base classes for tools, implemented in a functional manner."""

import asyncio
import inspect
import logging
import os
import re
import sys
from collections.abc import Callable
from itertools import chain
from typing import ClassVar, Self, cast

from aviary.core import ToolRequestMessage
from lmi import Embeddable, EmbeddingModel, LiteLLMModel
from pydantic import BaseModel, ConfigDict, Field, computed_field

from paperqa.docs import Docs
from paperqa.settings import Settings
from paperqa.sources.clinical_trials import add_clinical_trials_to_docs
from paperqa.types import Context, DocDetails, PQASession

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
    relevant_contexts = state.get_relevant_contexts()
    return make_status(
        total_paper_count=len(state.docs.docs),
        relevant_paper_count=len({c.text.doc.dockey for c in relevant_contexts}),
        evidence_count=len(relevant_contexts),
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
            return self.status_fn(cast("Self", self))
        return default_status(self)

    def get_relevant_contexts(self) -> list[Context]:
        return [
            c for c in self.session.contexts if c.score > self.RELEVANT_SCORE_CUTOFF
        ]

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
            this_doc_details = cast("DocDetails", next(iter(r.docs.values())))
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
        l1 = l0 = len(state.session.contexts)
        l1_relevant = l0_relevant = len(state.get_relevant_contexts())

        try:
            # Swap out the question with the more specific question
            # TODO: remove this swap, as it prevents us from supporting parallel calls
            state.session.question = question

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
            l1_relevant = len(state.get_relevant_contexts())
        finally:
            state.session.question = original_question

        status = state.status
        logger.info(status)
        sorted_contexts = sorted(
            state.session.contexts, key=lambda x: x.score, reverse=True
        )

        top_contexts = "\n".join(
            [
                f"{n + 1}. {sc.context}\n"
                for n, sc in enumerate(
                    sorted_contexts[: self.settings.agent.agent_evidence_n]
                )
            ]
        )

        best_evidence = f" Best evidence(s):\n\n{top_contexts}" if top_contexts else ""

        if f"{self.TOOL_FN_NAME}_completed" in self.settings.agent.callbacks:
            await asyncio.gather(
                *(
                    callback(state)
                    for callback in self.settings.agent.callbacks[
                        f"{self.TOOL_FN_NAME}_completed"
                    ]
                )
            )

        return (
            f"Added {l1 - l0} pieces of evidence, {l1_relevant - l0_relevant} of which"
            f" were relevant.{best_evidence}\n\n" + status
        )


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
            f" '{'certain' if has_successful_answer else 'unsure'}'."
        )
        # Return answer and status to simplify postprocessing of tool response
        return f"{'Certain' if has_successful_answer else 'Unsure'} | {state.status}"


class ClinicalTrialsSearch(NamedTool):
    TOOL_FN_NAME = "clinical_trials_search"

    model_config = ConfigDict(extra="forbid")

    search_count: int = 8
    previous_searches: dict[str, int] = Field(default_factory=dict)
    settings: Settings = Field(default_factory=Settings)

    # Gather evidence tool must be modified to understand the new evidence
    GATHER_EVIDENCE_TOOL_PROMPT_OVERRIDE: ClassVar[str] = (
        """Gather evidence from previous papers and clinical trials given a specific question.

        Will increase evidence, relevant paper counts, and relevant clinical trial counts.
        A valuable time to invoke this tool is right after another tool increases paper or clinical trials count.
        Feel free to invoke this tool in parallel with other tools, but do not call this tool in parallel with itself.
        Only invoke this tool when the paper count or clinical trial count is above zero, or this tool will be useless.

        Args:
            question: Specific question to gather evidence for.
            state: Current state.

        Returns:
            String describing gathered evidence and the current status.
        """
    )

    async def clinical_trials_search(self, query: str, state: EnvironmentState) -> str:
        r"""Search for clinical trials, with support for repeated calls and concurrent execution.

        Will add new clinical trials to the state, and return metadata about the number of trials found.

        Args:
            query: The search query string. Supports complex boolean expressions, field-specific
                searches, and query modifiers through operators. All configuration is done through
                operators in the query string.
                Query Syntax:
                    Basic Search:
                        Simple text automatically uses default EXPANSION[Relaxation] and COVERAGE[Contains]
                        >>> "heart attack"

                    Modified Search:
                        Use operators to modify search behavior:
                        >>> 'EXPANSION[None]COVERAGE[FullMatch]"exact phrase"'
                        >>> 'EXPANSION[Concept]heart attack'

                    Field Search:
                        Specify fields using AREA operator:
                        >>> 'AREA[InterventionName]aspirin'
                        >>> 'AREA[Phase]PHASE3'

                    Location Search:
                        Use SEARCH operator for compound location queries:
                        >>> 'cancer AND SEARCH[Location](AREA[LocationCity]Boston AND AREA[LocationState]Massachusetts)'

                    Complex Boolean:
                        Combine terms with AND, OR, NOT and parentheses:
                        >>> '(cancer OR tumor) AND NOT (EXPANSION[None]pediatric OR AREA[StdAge]CHILD)'

                    Date Ranges:
                        Use RANGE to specify date ranges with formats like "yyyy-MM" or "yyyy-MM-dd".
                        Note that MIN and MAX can be used for open-ended ranges:
                        >>> AREA[ResultsFirstPostDate]RANGE[2015-01-01, MAX]

                Operators:
                    EXPANSION[type]: Controls term expansion
                        - None: Exact match only, case and accent sensitive
                        - Term: Includes lexical variants (plurals, spellings)
                        - Concept: Includes UMLS synonyms
                        - Relaxation: Relaxes adjacency requirements (default)
                        - Lossy: Allows missing partial terms

                    COVERAGE[type]: Controls text matching
                        - FullMatch: Must match entire field
                        - StartsWith: Must match beginning of field
                        - EndsWith: Must match end of field
                        - Contains: Must match part of field (default)

                    AREA[field]: Specifies field to search
                        - See Field Reference for available fields

                    SEARCH[type]: Groups field searches
                        - Location: Groups location-related fields
                        - Study: Groups study-related fields

                Usage Notes:
                    - All search expressions are implicitly OR expressions
                    - Operator precedence (highest to lowest): terms/source operators, NOT/context operators, AND, OR
                    - Use quotes for exact phrase matching: "heart attack"
                    - Use parentheses for grouping: (heart OR cardiac) AND attack
                    - Use backslash to escape operators: \AND
                    - Default expansion is EXPANSION[Relaxation]
                    - Default coverage is COVERAGE[Contains]

                Field Reference:
                    High Priority Fields (weight >= 0.8):
                        - NCTId (1.0): Trial identifier
                        - Acronym (1.0): Study acronym
                        - BriefTitle (0.89): Short title
                        - OfficialTitle (0.85): Full official title
                        - Condition (0.81): Medical condition
                        - InterventionName (0.8): Primary intervention name
                        - OverallStatus: Trial status

                    Medium Priority Fields (0.5-0.79):
                        - InterventionOtherName (0.75): Alternative intervention names
                        - Phase (0.65): Trial phase
                        - StdAge (0.65): Standard age groups
                        - Keyword (0.6): Study keywords
                        - BriefSummary (0.6): Short description
                        - SecondaryOutcomeMeasure (0.5): Secondary outcomes

                    Low Priority Fields (< 0.5):
                        - DesignPrimaryPurpose (0.3): Primary purpose of study
                        - StudyType (0.3)
                        - Various descriptive, location, and administrative fields

                Supported Enums:
                    Phase:
                        - EARLY_PHASE1: Early Phase 1
                        - PHASE1: Phase 1
                        - PHASE2: Phase 2
                        - PHASE3: Phase 3
                        - PHASE4: Phase 4
                        - NA: Not Applicable

                    StandardAge:
                        - CHILD: Child
                        - ADULT: Adult
                        - OLDER_ADULT: Older Adult

                    Status:
                        - RECRUITING: Currently recruiting participants
                        - ACTIVE_NOT_RECRUITING: Active but not recruiting
                        - COMPLETED: Study completed
                        - ENROLLING_BY_INVITATION: Enrolling by invitation only
                        - NOT_YET_RECRUITING: Not yet recruiting
                        - SUSPENDED: Study suspended
                        - TERMINATED: Study terminated
                        - WITHDRAWN: Study withdrawn
                        - AVAILABLE: Available
                        - NO_LONGER_AVAILABLE: No longer available
                        - TEMPORARILY_NOT_AVAILABLE: Temporarily not available
                        - APPROVED_FOR_MARKETING: Approved for marketing
                        - WITHHELD: Withheld
                        - UNKNOWN: Unknown status

                    StudyType:
                        - INTERVENTIONAL: Interventional studies
                        - OBSERVATIONAL: Observational studies
                        - EXPANDED_ACCESS: Expanded access studies

                    PrimaryPurpose:
                        - TREATMENT: Treatment
                        - PREVENTION: Prevention
                        - DIAGNOSTIC: Diagnostic
                        - ECT: Educational/Counseling/Training
                        - SUPPORTIVE_CARE: Supportive Care
                        - SCREENING: Screening
                        - HEALTH_SERVICES_RESEARCH: Health Services Research
                        - BASIC_SCIENCE: Basic Science
                        - DEVICE_FEASIBILITY: Device Feasibility
                        - OTHER: Other

                    InterventionType:
                        - BEHAVIORAL: Behavioral interventions
                        - BIOLOGICAL: Biological interventions
                        - COMBINATION_PRODUCT: Combination product interventions
                        - DEVICE: Device interventions
                        - DIAGNOSTIC_TEST: Diagnostic test interventions
                        - DIETARY_SUPPLEMENT: Dietary supplement interventions
                        - DRUG: Drug interventions
                        - GENETIC: Genetic interventions
                        - PROCEDURE: Procedure interventions
                        - RADIATION: Radiation interventions
                        - OTHER: Other interventions

                    DesignAllocation:
                        - RANDOMIZED: Randomized allocation
                        - NON_RANDOMIZED: Non-randomized allocation
                        - NA: Not applicable

                    InterventionalAssignment:
                        - SINGLE_GROUP: Single group assignment
                        - PARALLEL: Parallel assignment
                        - CROSSOVER: Crossover assignment
                        - FACTORIAL: Factorial assignment
                        - SEQUENTIAL: Sequential assignment

                    ObservationalModel:
                        - COHORT: Cohort
                        - CASE_CONTROL: Case-Control
                        - CASE_ONLY: Case-Only
                        - CASE_CROSSOVER: Case-Crossover
                        - ECOLOGIC_OR_COMMUNITY: Ecologic or Community
                        - FAMILY_BASED: Family-Based
                        - DEFINED_POPULATION: Defined Population
                        - NATURAL_HISTORY: Natural History
                        - OTHER: Other

                    DesignMasking:
                        - NONE: None (Open Label)
                        - SINGLE: Single
                        - DOUBLE: Double
                        - TRIPLE: Triple
                        - QUADRUPLE: Quadruple

                    WhoMasked:
                        - PARTICIPANT: Participant
                        - CARE_PROVIDER: Care Provider
                        - INVESTIGATOR: Investigator
                        - OUTCOMES_ASSESSOR: Outcomes Assessor

            state: Current state

        Returns:
            String describing current status
        """
        # get offset if we've done this search before (continuation of search)
        # or mark this search as new (so offset 0)
        try:
            offset = self.previous_searches[query]
        except KeyError:
            offset = self.previous_searches[query] = 0

        total_result_count, new_result_count, error_message = (
            await add_clinical_trials_to_docs(
                query,
                state.docs,
                self.settings,
                limit=self.search_count,
                offset=offset,
            )
        )
        # mark how far we've searched so that continuation will start at the right place
        self.previous_searches[query] += self.search_count
        if error_message is None:
            return (
                f"Found clinical trial search results from search {offset} to"
                f" {offset + new_result_count} among {total_result_count} total"
                f" results. {state.status}"
            )
        return f"Error in clinical trial query syntax: {error_message}"


AVAILABLE_TOOL_NAME_TO_CLASS: dict[str, type[NamedTool]] = {
    cls.TOOL_FN_NAME: cls
    for _, cls in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda v: inspect.isclass(v)
        and issubclass(v, NamedTool)
        and v is not NamedTool,
    )
}


DEFAULT_TOOL_NAMES: list[str] = [
    name.strip()
    for name in os.environ.get("PAPERQA_DEFAULT_TOOL_NAMES", "").split(",")
    if name.strip()
] or [
    PaperSearch.TOOL_FN_NAME,
    GatherEvidence.TOOL_FN_NAME,
    GenerateAnswer.TOOL_FN_NAME,
    Reset.TOOL_FN_NAME,
    Complete.TOOL_FN_NAME,
]
