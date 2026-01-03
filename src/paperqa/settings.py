import asyncio
import importlib.resources
import logging
import os
import pathlib
import re
import warnings
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from enum import IntEnum, StrEnum
from itertools import starmap
from pydoc import locate
from typing import (
    Any,
    ClassVar,
    Protocol,
    Self,
    TypeAlias,
    assert_never,
    cast,
    runtime_checkable,
)

import anyio
import litellm
from aviary.core import Message, Tool, ToolSelector
from lmi import (
    CommonLLMNames,
    EmbeddingModel,
    LiteLLMModel,
    embedding_model_factory,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core.core_schema import SerializationInfo
from pydantic_settings import BaseSettings, CliSettingsSource, SettingsConfigDict

import paperqa.configs
from paperqa._ldp_shims import (
    HAS_LDP_INSTALLED,
    Agent,
    HTTPAgentClient,
    MemoryAgent,
    ReActAgent,
    SimpleAgent,
    SimpleAgentState,
    UIndexMemoryModel,
    _Memories,
    set_training_mode,
)
from paperqa.prompts import (
    CONTEXT_INNER_PROMPT,
    CONTEXT_OUTER_PROMPT,
    answer_iteration_prompt_template,
    citation_prompt,
    default_system_prompt,
    env_reset_prompt,
    env_system_prompt,
    individual_media_enrichment_prompt_template,
    qa_prompt,
    select_paper_prompt,
    structured_citation_prompt,
    summary_json_prompt,
    summary_json_system_prompt,
    summary_prompt,
)
from paperqa.readers import PDFParserFn
from paperqa.types import Context, ParsedMedia, ParsedText
from paperqa.utils import (
    get_stable_str,
    hexdigest,
    parse_enrichment_irrelevance,
    pqa_directory,
)

logger = logging.getLogger(__name__)

# TODO: move to actual EnvironmentState
# when we can do so without a circular import
_EnvironmentState: TypeAlias = Any


@runtime_checkable
class AsyncContextSerializer(Protocol):
    """Protocol for generating a context string from settings and context."""

    async def __call__(
        self,
        settings: "Settings",
        contexts: Sequence[Context],
        question: str,
        pre_str: str | None,
    ) -> str: ...


class AnswerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    evidence_k: int = Field(
        default=10, description="Number of evidence pieces to retrieve."
    )
    evidence_retrieval: bool = Field(
        default=True,
        description="Whether to use retrieval instead of processing all docs.",
    )
    # no validator because you can set the range in a prompt
    evidence_relevance_score_cutoff: int = Field(
        default=1,
        ge=0,
        description=(
            "Relevance score cutoff for evidence retrieval, default is 1, meaning"
            " only evidence with relevance score >= 1 will be used."
        ),
    )
    evidence_summary_length: str = Field(
        default="about 100 words", description="Length of evidence summary."
    )
    evidence_skip_summary: bool = Field(
        default=False, description="Whether to summarization."
    )
    evidence_text_only_fallback: bool = Field(
        default=False,
        description=(
            "Opt-in flag to allow creating contexts without media (just text),"
            " if the media is problematic for the LLM provider or network."
        ),
    )
    answer_max_sources: int = Field(
        default=5, description="Max number of sources to use for an answer."
    )
    max_answer_attempts: int | None = Field(
        default=None,
        description=(
            "Optional (exclusive) max number (default is no max) of attempts to"
            " generate an answer before declaring done (without a complete tool call). "
        ),
    )
    answer_length: str = Field(
        default="about 200 words, but can be longer",
        description="Length of final answer.",
    )
    max_concurrent_requests: int = Field(
        default=4, description="Max concurrent requests to LLMs."
    )
    answer_filter_extra_background: bool = Field(
        default=False,
        description="Whether to cite background information provided by model.",
    )
    get_evidence_if_no_contexts: bool = Field(
        default=True,
        description=(
            "Opt-out flag for allowing answer generation to lazily gather evidence if"
            " called before evidence was gathered."
        ),
    )
    group_contexts_by_question: bool = Field(
        default=False,
        description="Whether to group contexts by question when generating answers.",
    )
    skip_evidence_citation_strip: bool = Field(
        default=False,
        description="Whether to skip stripping citations from evidence.",
    )


class ChunkingOptions(StrEnum):
    SIMPLE_OVERLAP = "simple_overlap"


def get_default_pdf_parser() -> PDFParserFn:
    parse_pdf_to_pages: PDFParserFn

    try:
        from paperqa_pymupdf import parse_pdf_to_pages
    except ImportError:
        try:
            from paperqa_pypdf import parse_pdf_to_pages  # type: ignore[no-redef,unused-ignore]
        except ImportError as exc:
            raise ImportError(
                "To parse PDFs we need a parsing function. Please install either:"
                " (1) paper-qa-pypdf via `pip install paper-qa[pypdf]` or"
                " (2) paper-qa-pymupdf via `pip install paper-qa[pymupdf]`."
            ) from exc

    return parse_pdf_to_pages


def default_pdf_parser_configurator() -> None:
    try:
        from paperqa_pymupdf import setup_pymupdf_python_logging
    except ImportError:
        return

    setup_pymupdf_python_logging()


class MultimodalOptions(IntEnum):
    # Text-only PDF reads
    OFF = 0  # Falsey
    ON_WITH_ENRICHMENT = 1  # Default
    # Without image enrichment, multimodal will miss retrieval of certain images,
    # but it costs less money (no enrichment LLM calls)
    ON_WITHOUT_ENRICHMENT = 2

    @classmethod
    def from_value(cls, value: "bool | MultimodalOptions") -> "MultimodalOptions":
        if isinstance(value, bool):
            return cls(int(value))
        return value

    @property
    def should_parse_and_enrich_media(self) -> tuple[bool, bool]:
        """Get if the settings indicate to parse and also enrich media."""
        if self == MultimodalOptions.OFF:
            return False, False
        if self == MultimodalOptions.ON_WITHOUT_ENRICHMENT:
            return True, False
        if self == MultimodalOptions.ON_WITH_ENRICHMENT:
            return True, True
        assert_never(self)


class ParsingSettings(BaseModel):
    """Settings relevant for parsing and chunking documents."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    page_size_limit: int | None = Field(
        default=1_280_000,
        description=(
            "Optional limit on the number of characters to parse in one 'page', default"
            " is 1.28 million chars, 10X larger than a 128k tokens context limit"
            " (ignoring chars vs tokens difference)."
        ),
    )
    use_doc_details: bool = Field(
        default=True, description="Whether to try to get metadata details for a Doc."
    )
    reader_config: dict[str, Any] = Field(
        default_factory=lambda: {"chunk_chars": 5000, "overlap": 250},
        description="Optional keyword arguments for the document reader.",
        examples=[{"dpi": 300}],
    )
    multimodal: bool | MultimodalOptions = Field(
        default=MultimodalOptions.ON_WITH_ENRICHMENT,
        description=(
            "Controls on parsing images/tables (if applicable to a given document)."
            " Setting false or off will parse only text,"
            " setting true or 'on with enrichment' will parse media and use"
            " the enrichment LLM to generate descriptions of the media,"
            " setting 'on without enrichment' will parse media without enrichment."
        ),
    )
    citation_prompt: str = Field(
        default=citation_prompt,
        description="Prompt that tries to create citation from peeking one chunk.",
    )
    structured_citation_prompt: str = Field(
        default=structured_citation_prompt,
        description=(
            "Prompt that tries to creates a citation in JSON from peeking one chunk."
        ),
    )
    disable_doc_valid_check: bool = Field(
        default=False,
        description=(
            "Whether to disable checking if a document looks like text (was parsed"
            " correctly)."
        ),
    )
    defer_embedding: bool = Field(
        default=False,
        description=(
            "Whether to embed documents immediately as they are added, or defer until"
            " summarization."
        ),
    )
    parse_pdf: PDFParserFn = Field(
        default_factory=get_default_pdf_parser,
        description="Function to parse PDF, or a fully qualified name to import.",
        examples=["paperqa_docling.parse_pdf_to_pages"],
        exclude=True,  # NOTE: a custom serializer is used below, so it's not excluded
    )
    configure_pdf_parser: Callable[[], Any] = Field(
        default=default_pdf_parser_configurator,
        description=(
            "Callable to configure the PDF parser within parse_pdf,"
            " useful for behaviors such as enabling logging."
        ),
        exclude=True,
    )

    @field_validator("parse_pdf", mode="before")
    @classmethod
    def _resolve_parse_pdf(cls, v: str | PDFParserFn) -> PDFParserFn:
        """Resolve a fully qualified name to a parser function."""
        if isinstance(v, str):
            resolved = locate(v)
            if resolved is None:
                raise ValueError(f"Failed to locate PDF parser function {v!r}.")
            if not isinstance(resolved, PDFParserFn):
                raise TypeError(f"Value {v!r} is not a PDF parser function.")
            return resolved
        return v

    @model_serializer(mode="wrap")
    def _custom_serializer(
        self, serializer: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        data = serializer(self)
        # NOTE: due to parse_pdf's exclude=True flag, it's not yet in this data.
        # Let's now add it back if we can safely deserialize "over the network"
        if isinstance(self.parse_pdf, str):
            # Already JSON-compliant, so let's un-exclude
            data["parse_pdf"] = self.parse_pdf
        elif info.mode == "json":
            # If going to JSON, and we can get a FQN, do so for JSON compliance
            with suppress(ValueError):  # Suppress when not serialization-safe
                data["parse_pdf"] = get_stable_str(self.parse_pdf)
        return data

    doc_filters: Sequence[Mapping[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional filters to only allow documents that match this filter. This is a"
            " dictionary where the keys are the fields from DocDetails or Docs to"
            " filter on, and the values are the values to filter for. To invert filter,"
            " prefix the key with a '!'. If the key is not found, by default the Doc is"
            " rejected. To change this behavior, prefix the key with a '?' to allow the"
            " Doc to pass if the key is not found. For example, {'!title': 'bad title',"
            " '?year': '2022'} would only allow Docs with a title that is not 'bad"
            " title' and a year of 2022 or no year at all."
        ),
    )
    use_human_readable_clinical_trials: bool = Field(
        default=False,
        description="Parse clinical trial JSONs into human readable text.",
    )
    enrichment_llm: str = Field(
        # NOTE: from CapArena (https://arxiv.org/abs/2503.12329),
        # GPT-4o was the best image captioning model as of spring 2025
        # NOTE: claude-haiku-4-5-20251001 recurringly failed to describe display type
        # to be display type, so its captioning ability isn't good enough yet
        default=CommonLLMNames.GPT_4O.value,
        description="LLM for media enrichment (e.g. generating descriptions).",
    )
    enrichment_llm_config: dict | None = Field(
        default=None,
        description=(
            "Optional configuration for the enrichment_llm model. More specifically, it's"
            " a LiteLLM Router configuration to pass to LiteLLMModel, must have"
            " `model_list` key (corresponding to model_list inputs here:"
            " https://docs.litellm.ai/docs/routing), and can optionally include a"
            " router_kwargs key with router kwargs as values."
        ),
    )
    enrichment_page_radius: int = Field(
        default=1,  # Default is 1 because figures are usually +/- 1 page in LaTeX
        ge=-1,
        description=(
            "Page radius for context text in enrichment. "
            "-1 means all pages, 0 means current page only, "
            "1+ means a radius of pages around the current page."
        ),
    )
    enrichment_prompt: str = Field(
        default=individual_media_enrichment_prompt_template,
        description="Prompt template for enriching media.",
    )

    @property
    def should_parse_and_enrich_media(self) -> tuple[bool, bool]:
        """Get if the settings indicate to parse and also enrich media."""
        mm_enum = MultimodalOptions.from_value(self.multimodal)
        return mm_enum.should_parse_and_enrich_media


class _FormatDict(dict):  # noqa: FURB189
    """Mock a dictionary and store any missing items."""

    def __init__(self) -> None:
        self.key_set: set[str] = set()

    def __missing__(self, key: str) -> str:
        self.key_set.add(key)
        return key


def get_formatted_variables(s: str) -> set[str]:
    """Returns the set of variables implied by the format string."""
    format_dict = _FormatDict()
    s.format_map(format_dict)
    return format_dict.key_set


class PromptSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # citations are inserted with Context.id as follows,
    # these are translated to MLA parenthetical in-text citation styling
    # SEE: https://nwtc.libguides.com/citations/MLA#s-lg-box-707489
    EXAMPLE_CITATION: ClassVar[str] = "(pqac-0f650d59)"

    summary: str = summary_prompt
    qa: str = qa_prompt
    answer_iteration_prompt: str | None = Field(
        default=answer_iteration_prompt_template,
        description=(
            "Prompt to inject existing prior answers into the qa prompt to allow the model to iterate. "
            "If None, then no prior answers will be injected."
        ),
    )
    select: str = select_paper_prompt
    pre: str | None = Field(
        default=None,
        description=(
            "Opt-in pre-prompt (templated with just the original question) to append"
            " information before a qa prompt. For example:"
            " 'Summarize all scientific terms in the following question:\n{question}'."
            " This pre-prompt can enable injection of question-specific guidance later"
            " used by the qa prompt, without changing the qa prompt's template."
        ),
    )
    post: str | None = None
    system: str = default_system_prompt
    use_json: bool = True
    # Not thrilled about this model,
    # but need to split out the system/summary
    # to get JSON
    summary_json: str = summary_json_prompt
    summary_json_system: str = summary_json_system_prompt
    context_outer: str = Field(
        default=CONTEXT_OUTER_PROMPT,
        description="Prompt for how to format all contexts in generate answer.",
    )
    context_inner: str = Field(
        default=CONTEXT_INNER_PROMPT,
        description=(
            "Prompt for how to format a single context in generate answer. "
            "This should at least contain key and name."
        ),
    )

    @field_validator("summary")
    @classmethod
    def check_summary(cls, v: str) -> str:
        if not get_formatted_variables(v).issubset(
            get_formatted_variables(summary_prompt)
        ):
            raise ValueError(
                "Summary prompt can only have variables:"
                f" {get_formatted_variables(summary_prompt)}"
            )
        return v

    @field_validator("qa")
    @classmethod
    def check_qa(cls, v: str) -> str:
        if not get_formatted_variables(v).issubset(get_formatted_variables(qa_prompt)):
            raise ValueError(
                "QA prompt can only have variables:"
                f" {get_formatted_variables(qa_prompt)}"
            )
        return v

    @field_validator("select")
    @classmethod
    def check_select(cls, v: str) -> str:
        if not get_formatted_variables(v).issubset(
            get_formatted_variables(select_paper_prompt)
        ):
            raise ValueError(
                "Select prompt can only have variables:"
                f" {get_formatted_variables(select_paper_prompt)}"
            )
        return v

    @field_validator("post")
    @classmethod
    def check_post(cls, v: str | None) -> str | None:
        if v is not None:
            # kind of a hack to get list of attributes in answer
            from paperqa.types import PQASession

            attrs = set(PQASession.model_fields.keys())
            if not get_formatted_variables(v).issubset(attrs):
                raise ValueError(f"Post prompt must have input variables: {attrs}")
        return v

    @field_validator("context_outer")
    @classmethod
    def check_context_outer(cls, v: str) -> str:
        if not get_formatted_variables(v).issubset(
            get_formatted_variables(CONTEXT_OUTER_PROMPT)
        ):
            raise ValueError(
                "Context outer prompt can only have variables:"
                f" {get_formatted_variables(CONTEXT_OUTER_PROMPT)}"
            )
        return v

    @field_validator("context_inner")
    @classmethod
    def check_context_inner(cls, v: str) -> str:
        fvars = get_formatted_variables(v)
        if "name" not in fvars or "text" not in fvars:
            raise ValueError("Context inner prompt must have name and text")
        return v


class IndexSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        description=(
            "Optional name of the index. If unspecified, the name should be generated."
        ),
    )
    paper_directory: str | os.PathLike = Field(
        default=pathlib.Path.cwd(),
        description=(
            "Local directory which contains the papers to be indexed and searched."
        ),
    )
    manifest_file: str | os.PathLike | None = Field(
        default=None,
        description=(
            "Optional absolute path to a manifest CSV, or a relative path from the"
            " paper_directory to a manifest CSV. A manifest CSV contains columns which"
            " are attributes for a DocDetails object. Only 'file_location', 'doi', and"
            " 'title' will be used when indexing, others are discarded."
        ),
    )
    index_directory: str | os.PathLike = Field(
        default_factory=lambda: pqa_directory("indexes"),
        description=(
            "Directory to store the PQA built search index, configuration, and"
            " answer indexes."
        ),
    )
    use_absolute_paper_directory: bool = Field(
        default=False,
        description=(
            "Opt-in flag to convert the paper_directory to an absolute path. Setting"
            " this to True will make the index user-specific, defeating sharing."
        ),
    )
    recurse_subdirectories: bool = Field(
        default=True,
        description="Whether to recurse into subdirectories when indexing sources.",
    )
    concurrency: int = Field(
        default=5,  # low default for folks without S2/Crossref keys
        description="Number of concurrent filesystem reads for indexing",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of files to process before committing to the index.",
    )
    sync_with_paper_directory: bool = Field(
        default=True,
        description=(
            "Whether to sync the index with the paper directory when loading an index."
            " Setting to True will add or delete index files to match the source paper"
            " directory."
        ),
    )
    files_filter: Callable[[anyio.Path | pathlib.Path], bool] = Field(
        default=lambda f: (
            f.suffix
            # TODO: add images after embeddings are supported
            in {".txt", ".pdf", ".html", ".md", ".xlsx", ".docx", ".pptx"}
        ),
        exclude=True,
        description=(
            "Filter function to apply to files in the paper directory."
            " When the function returns True, the file will be indexed."
        ),
    )

    def get_named_index_directory(self) -> anyio.Path:
        """Get the directory where the index, when named, will be located.

        Raises:
            ValueError: If the index name was unset, because otherwise the name is
                autogenerated.
        """
        if self.name is None:
            raise ValueError(
                "Getting a named index directory requires an index name to have been"
                " specified, please specify a name."
            )
        return anyio.Path(self.index_directory) / self.name

    async def finalize_manifest_file(self) -> anyio.Path | None:
        manifest_file = anyio.Path(self.manifest_file) if self.manifest_file else None
        if manifest_file and not await manifest_file.exists():
            # If the manifest file was specified but doesn't exist,
            # perhaps it was specified as a relative path from the paper_directory
            manifest_file = anyio.Path(self.paper_directory) / manifest_file
        return manifest_file


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_llm: str = Field(
        default=CommonLLMNames.GPT_4O.value,
        description="LLM inside the agent making tool selections.",
    )

    agent_llm_config: dict | None = Field(
        default=None,
        description=(
            "Optional configuration for the agent_llm model. More specifically, it's"
            " a LiteLLM Router configuration to pass to LiteLLMModel, must have"
            " `model_list` key (corresponding to model_list inputs here:"
            " https://docs.litellm.ai/docs/routing), and can optionally include a"
            " router_kwargs key with router kwargs as values."
        ),
    )

    agent_type: str = Field(
        default="ToolSelector",
        description="Type of agent to use",
    )
    agent_config: dict[str, Any] | None = Field(
        default=None,
        description="Optional kwarg for AGENT constructor.",
    )
    agent_system_prompt: str | None = Field(
        default=env_system_prompt,
        description="Optional system prompt message to precede the below agent_prompt.",
    )
    agent_prompt: str = env_reset_prompt
    return_paper_metadata: bool = Field(
        default=False,
        description=(
            "Set True to have the search tool include paper title/year information as"
            " part of its return."
        ),
    )
    search_count: int = 8
    agent_evidence_n: int = Field(
        default=1,
        ge=1,
        description=(
            "Top n ranked evidences shown to the agent after the GatherEvidence tool."
        ),
    )
    timeout: float = Field(
        default=500.0,
        description=(
            "Matches LangChain AgentExecutor.max_execution_time (seconds), the timeout"
            " on agent execution."
        ),
    )

    tool_names: set[str] | Sequence[str] | None = Field(
        default=None,
        description=(
            "Optional override on the tools to provide the agent. Leaving as the"
            " default of None will use a minimal toolset of the paper search, gather"
            " evidence, collect cited papers from evidence, and gen answer. If passing"
            " tool names (non-default route), at least the gen answer tool must be"
            " supplied."
        ),
    )
    max_timesteps: int | None = Field(
        default=None,
        description="Optional upper limit on the number of environment steps.",
    )

    index: IndexSettings = Field(default_factory=IndexSettings)

    rebuild_index: bool = Field(
        default=True,
        description=(
            "Flag to rebuild the index at the start of agent runners, default is True"
            " for CLI users to ensure all source PDFs are pulled in."
        ),
    )

    callbacks: Mapping[str, Sequence[Callable[[_EnvironmentState], Any]]] = Field(
        default_factory=dict,
        description="""
            A mapping that associates callback names with lists of corresponding callable functions.
            Each callback list contains functions that will be called with an instance of `EnvironmentState`,
            representing the current state context.

            Accepted callback names:
            - 'gen_answer_initialized': Triggered when `GenerateAnswer.gen_answer`
                is initialized.

            - 'gen_answer_aget_query': LLM callbacks to execute in the prompt runner
                as part of `GenerateAnswer.gen_answer`.

            - 'gen_answer_completed': Triggered after `GenerateAnswer.gen_answer`
                successfully generates an answer.

            - 'gather_evidence_initialized': Triggered when `GatherEvidence.gather_evidence`
                is initialized.

            - 'gather_evidence_aget_evidence: LLM callbacks to execute in the prompt runner
                as part of `GatherEvidence.gather_evidence`.

            - 'gather_evidence_completed': Triggered after `GatherEvidence.gather_evidence`
                completes evidence gathering.
        """,
        exclude=True,
    )


def make_default_litellm_model_list_settings(
    llm: str, temperature: float = 0.0
) -> dict:
    """Settings matching "model_list" schema here: https://docs.litellm.ai/docs/routing."""
    return {
        "name": llm,
        "model_list": [
            {
                "model_name": llm,
                "litellm_params": {"model": llm, "temperature": temperature},
            }
        ],
    }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    llm: str = Field(
        default=CommonLLMNames.GPT_4O.value,
        description=(
            "LLM for general use including metadata inference (see Docs.aadd)"
            " and answer generation (see Docs.aquery and gen_answer tool)."
            " Should be 'best' LLM. Uses include:"
            " 1. Inferring citation information from documents when left unspecified,"
            " 2. Extracting title, DOI, and authors from citation information when left unspecified,"
            " 3. Optionally injecting pre-answer information (see PromptSettings.pre),"
            " 4. Generating an answer given evidence (see PromptSettings.qa),"
            " 5. Optionally injecting post-answer information (see PromptSettings.post),"
            " 6. If using the 'fake' agent, proposing search queries."
        ),
    )
    llm_config: dict | None = Field(
        default=None,
        description=(
            "Optional configuration for the llm model. More specifically, it's"
            " a LiteLLM Router configuration to pass to LiteLLMModel, must have"
            " `model_list` key (corresponding to model_list inputs here:"
            " https://docs.litellm.ai/docs/routing), and can optionally include a"
            " router_kwargs key with router kwargs as values."
        ),
    )
    summary_llm: str = Field(
        default=CommonLLMNames.GPT_4O.value,
        description=(
            "LLM for creating contextual summaries"
            " (see Docs.aget_evidence and gather_evidence tool)."
        ),
    )
    summary_llm_config: dict | None = Field(
        default=None,
        description=(
            "Optional configuration for the summary_llm model. More specifically, it's"
            " a LiteLLM Router configuration to pass to LiteLLMModel, must have"
            " `model_list` key (corresponding to model_list inputs here:"
            " https://docs.litellm.ai/docs/routing), and can optionally include a"
            " router_kwargs key with router kwargs as values."
        ),
    )
    embedding: str = Field(
        default="text-embedding-3-small",
        description="Embedding model for embedding text chunks when adding papers.",
    )
    embedding_config: dict | None = Field(
        default=None,
        description="Optional configuration for the embedding model.",
    )
    temperature: float = Field(default=0.0, description="Temperature for LLMs.")
    batch_size: int = Field(default=1, description="Batch size for calling LLMs.")
    texts_index_mmr_lambda: float = Field(
        default=1.0, description="Lambda for MMR in text index."
    )
    verbosity: int = Field(
        default=0,
        description=(
            "Integer verbosity level for logging (0-3). 3 = all LLM/Embeddings calls"
            " logged."
        ),
    )
    custom_context_serializer: AsyncContextSerializer | None = Field(
        default=None,
        description=(
            "Function to turn settings and contexts into an answer context str."
            " If not populated, the default context serializer will be used."
        ),
        exclude=True,
    )

    @model_validator(mode="after")
    def _update_temperature(self) -> Self:
        """Ensures temperature is 1 if the LLM requires it.

        o1 reasoning models only support temperature = 1.
        SEE: https://platform.openai.com/docs/guides/reasoning/quickstart
        """
        for model_prefix in ("o1", "gpt-5"):
            if self.llm.startswith(model_prefix) and self.temperature != 1:
                warnings.warn(
                    f"When dealing with OpenAI {model_prefix} models,"
                    " the temperature must be set to 1,"
                    f" so the specified temperature {self.temperature}"
                    " has been overridden to 1.",
                    category=UserWarning,
                    stacklevel=2,
                )
                self.temperature = 1
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def md5(self) -> str:
        return hexdigest(self.model_dump_json(exclude={"md5"}))

    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    parsing: ParsingSettings = Field(default_factory=ParsingSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    def get_index_name(self) -> str:
        """Get programmatically generated index name.

        This index is where parsings are stored based on parsing/embedding strategy.
        """
        if isinstance(self.agent.index.paper_directory, pathlib.Path):
            # Here we use an absolute path so that where the user locally
            # uses '.', two different folders will make different indexes
            first_segment: str = str(self.agent.index.paper_directory.absolute())
        else:
            first_segment = str(self.agent.index.paper_directory)
        segments = [
            first_segment,
            str(self.agent.index.use_absolute_paper_directory),
            self.embedding,
            get_stable_str(self.parsing.parse_pdf, for_hash=True),
            str(self.parsing.reader_config["chunk_chars"]),
            str(self.parsing.reader_config["overlap"]),
            str(self.parsing.reader_config.get("full_page", False)),
            str(self.parsing.multimodal),
        ]
        return f"pqa_index_{hexdigest('|'.join(segments))}"

    @classmethod
    def from_name(
        cls, config_name: str = "default", cli_source: CliSettingsSource | None = None
    ) -> "Settings":
        json_path: pathlib.Path | None = None

        # quick exit for default settings
        if config_name == "default":
            if not cli_source:
                raise NotImplementedError(
                    f"For config_name {config_name!r}, we require cli_source."
                )
            return Settings(_cli_settings_source=cli_source(args=True))

        user_config_path = pqa_directory("settings") / f"{config_name}.json"
        pkg_config_path = (
            # Use importlib.resources.files() which is recommended for Python 3.9+
            importlib.resources.files(paperqa.configs)
            / f"{config_name}.json"
        )
        if user_config_path.exists():
            # First, try to find the config file in the user's .config directory
            json_path = user_config_path
        else:
            # If not found, fall back to the package's default config
            try:
                if pkg_config_path.is_file():
                    json_path = cast("pathlib.Path", pkg_config_path)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"No configuration file {config_name!r} found at user config path"
                    f" {user_config_path} or bundled config path {pkg_config_path}."
                ) from e

        if json_path:
            # we do the ole switcheroo
            # json - validate to deserialize knowing the types
            # then dump it
            # going json.loads directly will not get types correct
            tmp = Settings.model_validate_json(json_path.read_text())
            return Settings(
                **(tmp.model_dump()),
                _cli_settings_source=cli_source(args=True) if cli_source else None,
            )
        raise FileNotFoundError(
            f"No configuration file {config_name!r} found at user config path"
            f" {user_config_path} or bundled config path {pkg_config_path}."
        )

    def get_llm(self) -> LiteLLMModel:
        return LiteLLMModel(
            name=self.llm,
            config=self.llm_config
            or make_default_litellm_model_list_settings(self.llm, self.temperature),
        )

    def get_summary_llm(self) -> LiteLLMModel:
        return LiteLLMModel(
            name=self.summary_llm,
            config=self.summary_llm_config
            or make_default_litellm_model_list_settings(
                self.summary_llm, self.temperature
            ),
        )

    def get_agent_llm(self) -> LiteLLMModel:
        return LiteLLMModel(
            name=self.agent.agent_llm,
            config=self.agent.agent_llm_config
            or make_default_litellm_model_list_settings(
                self.agent.agent_llm, self.temperature
            ),
        )

    def get_embedding_model(self) -> EmbeddingModel:
        return embedding_model_factory(self.embedding, **(self.embedding_config or {}))

    def get_enrichment_llm(self) -> LiteLLMModel:
        return LiteLLMModel(
            name=self.parsing.enrichment_llm,
            config=self.parsing.enrichment_llm_config
            or make_default_litellm_model_list_settings(
                self.parsing.enrichment_llm, self.temperature
            ),
        )

    def make_aviary_tool_selector(self, agent_type: str | type) -> ToolSelector | None:
        """Attempt to convert the input agent type to an aviary ToolSelector."""
        if agent_type is ToolSelector or (
            isinstance(agent_type, str)
            and (
                agent_type == ToolSelector.__name__
                or (
                    agent_type.startswith(
                        ToolSelector.__module__.split(".", maxsplit=1)[0]
                    )
                    and locate(agent_type) is ToolSelector
                )
            )
        ):
            return ToolSelector(
                model_name=self.agent.agent_llm,
                acompletion=self.get_agent_llm().get_router().acompletion,
                **(self.agent.agent_config or {}),
            )
        return None

    async def make_ldp_agent(
        self, agent_type: str | type
    ) -> "Agent[SimpleAgentState] | None":
        """Attempt to convert the input agent type to an ldp Agent."""
        if not isinstance(agent_type, str):  # Convert to fully qualified name
            agent_type = f"{agent_type.__module__}.{agent_type.__name__}"
        if not agent_type.startswith("ldp"):
            return None
        if not HAS_LDP_INSTALLED:
            raise ImportError(
                "ldp agents requires the 'ldp' extra for 'ldp'. Please:"
                " `pip install paper-qa[ldp]`."
            )

        # TODO: support general agents
        agent_cls = cast("type[Agent]", locate(agent_type))
        agent_settings = self.agent
        agent_llm, config = agent_settings.agent_llm, agent_settings.agent_config or {}
        if issubclass(agent_cls, ReActAgent | MemoryAgent):
            if (
                issubclass(agent_cls, MemoryAgent)
                and "memory_model" in config
                and "memories" in config
            ):
                if "embedding_model" in config["memory_model"]:
                    config["memory_model"]["embedding_model"] = (
                        EmbeddingModel.from_name(
                            embedding=config["memory_model"].pop("embedding_model")[
                                "name"
                            ]
                        )
                    )
                try:
                    config["memory_model"] = UIndexMemoryModel(**config["memory_model"])
                except ImportError as exc:
                    raise ImportError(
                        "Memory agents require the 'usearch' package,"
                        " which is part of the 'memory' extra."
                        " Please: `pip install paper-qa[memory]`."
                    ) from exc
                memories = _Memories.validate_python(config.pop("memories"))
                await asyncio.gather(
                    *(
                        config["memory_model"].add_memory(memory)
                        for memory in (
                            memories.values()
                            if isinstance(memories, dict)
                            else memories
                        )
                    )
                )
            return agent_cls(
                llm_model={"name": agent_llm, "temperature": self.temperature},
                **config,
            )
        if issubclass(agent_cls, SimpleAgent):
            return agent_cls(
                llm_model={"name": agent_llm, "temperature": self.temperature},
                sys_prompt=agent_settings.agent_system_prompt,
                **config,
            )
        if issubclass(agent_cls, HTTPAgentClient):
            set_training_mode(False)
            return HTTPAgentClient[SimpleAgentState](
                agent_state_type=SimpleAgentState, **config
            )
        raise NotImplementedError(f"Didn't yet handle agent type {agent_type}.")

    def make_media_enricher(self) -> Callable[[ParsedText], Awaitable[str]]:
        """Create an enricher function from settings."""

        async def enrich_media_with_llm(parsed_text: ParsedText) -> str:
            """Enrich media in parsed text with LLM-generated descriptions and then filter irrelevant media.

            Returns:
                A summary string of the enrichment.
            """
            if not isinstance(parsed_text.content, dict) or not any(
                isinstance(c, tuple) for c in parsed_text.content.values()
            ):
                raise ValueError(
                    "Media enrichment requires media to be in the parsed text."
                )
            text_content = cast(
                dict[str, str | tuple[str, list[ParsedMedia]]], parsed_text.content
            )

            # Collect all media with their page numbers
            # NOTE: we could deduplicate media here across pages,
            # but this introduces a bunch of complexity:
            # - Do we enrich using on text surrounding the earlier or later image?
            # - Or do we enrich using text surrounding all images,
            #   and risk confusing the LLM if the texts aren't similar?
            # Given these risks, we just enrich extra times
            media_to_enrich: list[tuple[str, ParsedMedia]] = [
                (page_num, media)
                for page_num, page_contents in text_content.items()
                if isinstance(page_contents, tuple)
                for media in page_contents[1]
                if not media.info.get("enriched_description")  # Don't clobber prior
            ]
            llm = self.get_enrichment_llm()
            radius = self.parsing.enrichment_page_radius

            async def enrich_single_media(
                page_num: int | str, media: ParsedMedia
            ) -> None:
                """Enrich a single media item with LLM-generated description."""
                if radius == -1:  # All pages
                    context_text: str = "\n\n".join(
                        (
                            (
                                pg_contents
                                if isinstance(pg_contents, str)
                                else pg_contents[0]
                            )
                            for _, pg_contents in sorted(
                                text_content.items(), key=lambda x: int(x[0])
                            )
                        )
                    )
                    radius_msg: str = "all pages"
                else:  # Specific page radius
                    page_texts: list[str] = []
                    for pg_int in range(
                        max(1, int(page_num) - radius),
                        min(len(text_content), int(page_num) + radius) + 1,
                    ):
                        # Use get so we're tolerant to missing pages here
                        page_content = text_content.get(str(pg_int))
                        if page_content:
                            page_texts.append(
                                page_content
                                if isinstance(page_content, str)
                                else page_content[0]
                            )
                    context_text = "\n\n".join(page_texts)
                    radius_msg = (
                        f"a radius of {'1 page' if radius == 1 else f'{radius} pages'}"
                    )

                prompt = self.parsing.enrichment_prompt.format(
                    context_text=(
                        f"Here is the co-located text from {radius_msg}:\n\n{context_text}\n\n"
                        if context_text
                        else ""
                    )
                )
                try:
                    result = await llm.call_single(
                        messages=[
                            Message.create_message(
                                text=prompt, images=[media.to_image_url()]
                            )
                        ]
                    )
                    if result.text:
                        (
                            media.info["is_irrelevant"],
                            media.info["enriched_description"],
                        ) = parse_enrichment_irrelevance(result.text)
                except (litellm.InternalServerError, litellm.BadRequestError) as exc:
                    # Handle image > 5-MB failure mode 1:
                    # > litellm.BadRequestError: AnthropicException -
                    # > {"type":"error","error":{"type":"invalid_request_error",
                    # > "message":"messages.0.content.0.image.source.base64: image exceeds 5 MB maximum: 6229564 bytes > 5242880 bytes"},  # noqa: E501, W505
                    # > "request_id":"req_abc123"}.
                    # and image > 5-MB failure mode 2:
                    # > litellm.InternalServerError: AnthropicException -
                    # > {"type":"error","error":{"type":"invalid_request_error",
                    # > "message":"messages.0.content.0.image.source.base64: image exceeds 5 MB maximum: 5690780 bytes > 5242880 bytes"},  # noqa: E501, W505
                    # > "request_id":"req_abc123"}.
                    # And the image being corrupt (but a reasonable size)
                    if (
                        isinstance(exc, litellm.InternalServerError)
                        and re.search(
                            r"image exceeds .+ maximum", str(exc), re.IGNORECASE
                        )
                    ) or isinstance(exc, litellm.BadRequestError):
                        logger.warning(
                            f"Skipping enrichment for media index {media.index}"
                            f" on page {page_num} with metadata {media.info} because"
                            f" it was rejected by the LLM provider for {llm.name!r}."
                            f" Full error message: {exc!r}"
                        )
                    else:
                        raise

            await asyncio.gather(*list(starmap(enrich_single_media, media_to_enrich)))

            # Filter out irrelevant media from parsed_text.content in-place,
            # while counting enrichments and filtration
            count_enriched = count_filtered = 0
            for page_num, page_contents in text_content.items():
                if isinstance(page_contents, tuple):
                    page_text, media_list = page_contents
                    count_enriched += sum(
                        bool(m.info.get("enriched_description")) for m in media_list
                    )
                    filtered_media = [
                        m for m in media_list if not m.info.get("is_irrelevant", False)
                    ]
                    count_filtered += len(media_list) - len(filtered_media)
                    # In-place update the content for the filtered media
                    text_content[page_num] = (page_text, filtered_media)

            return (
                f"enriched={count_enriched}|filtered={count_filtered}|radius={radius}"
            )

        return enrich_media_with_llm

    def adjust_tools_for_agent_llm(self, tools: list[Tool]) -> None:
        """In-place adjust tool attributes or schemae to match agent LLM-specifics."""
        # This was originally made for Gemini 1.5 Flash not supporting empty tool args
        # in February 2025 (https://github.com/BerriAI/litellm/issues/7634), but then
        # Gemini fixed this server-side by mid-April 2025,
        # so this method is now just available for use

    async def context_serializer(
        self, contexts: Sequence[Context], question: str, pre_str: str | None
    ) -> str:
        """Default function for sorting ranked contexts and inserting into a context string."""
        if self.custom_context_serializer:
            return await self.custom_context_serializer(
                settings=self, contexts=contexts, question=question, pre_str=pre_str
            )

        answer_config = self.answer
        prompt_config = self.prompts

        # sort by first score, then name
        filtered_contexts = sorted(
            contexts,
            key=lambda x: (-x.score, x.text.name),
        )[: answer_config.answer_max_sources]
        # remove any contexts with a score below the cutoff
        filtered_contexts = [
            c
            for c in filtered_contexts
            if c.score >= answer_config.evidence_relevance_score_cutoff
        ]

        context_inner_prompt = prompt_config.context_inner
        context_str_body = ""
        if answer_config.group_contexts_by_question:
            contexts_by_question: dict[str, list[Context]] = defaultdict(list)
            for c in filtered_contexts:
                # Fallback to the main session question if not available.
                # question attribute is optional, so if a user
                # sets contexts externally, it may not have a question.
                context_question = getattr(c, "question", question)
                contexts_by_question[context_question].append(c)

            context_sections = []
            for context_question, contexts_in_group in contexts_by_question.items():
                inner_strs = [
                    context_inner_prompt.format(
                        name=c.id,
                        text=c.context,
                        citation=c.text.doc.formatted_citation,
                        **(c.model_extra or {}),
                    )
                    for c in contexts_in_group
                ]
                # Create a section with a question heading
                section_header = (
                    f'Contexts related to the question: "{context_question}"'
                )
                section = f"{section_header}\n\n" + "\n\n".join(inner_strs)
                context_sections.append(section)
            context_str_body = "\n\n---\n\n".join(context_sections)
        else:
            inner_context_strs = [
                context_inner_prompt.format(
                    name=c.id,
                    text=c.context,
                    citation=c.text.doc.formatted_citation,
                    **(c.model_extra or {}),
                )
                for c in filtered_contexts
            ]
            context_str_body = "\n\n".join(inner_context_strs)

        if pre_str:
            context_str_body += f"\n\nExtra background information: {pre_str}"

        return prompt_config.context_outer.format(
            context_str=context_str_body,
            valid_keys=", ".join([c.id for c in filtered_contexts]),
        )


# Settings: already Settings
# dict[str, Any]: serialized Settings
# str: named Settings
# None: defaulted Settings
MaybeSettings = Settings | dict[str, Any] | str | None


def get_settings(config_or_name: MaybeSettings = None) -> Settings:
    if isinstance(config_or_name, Settings):
        return config_or_name
    if isinstance(config_or_name, dict):
        return Settings.model_validate(config_or_name)
    if config_or_name is None:
        return Settings()
    return Settings.from_name(config_name=config_or_name)
