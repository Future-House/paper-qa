import asyncio
import importlib.resources
import os
from enum import StrEnum
from pathlib import Path
from pydoc import locate
from typing import Any, ClassVar, assert_never, cast

from aviary.tools import ToolSelector
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    computed_field,
    field_validator,
)
from pydantic_settings import BaseSettings, CliSettingsSource, SettingsConfigDict

try:
    from ldp.agent import (
        Agent,
        HTTPAgentClient,
        MemoryAgent,
        ReActAgent,
        SimpleAgent,
        SimpleAgentState,
    )
    from ldp.graph.memory import Memory, UIndexMemoryModel
    from ldp.graph.op_utils import set_training_mode
    from ldp.llms import EmbeddingModel as LDPEmbeddingModel

    _Memories = TypeAdapter(dict[int, Memory] | list[Memory])  # type: ignore[var-annotated]

    HAS_LDP_INSTALLED = True
except ImportError:
    HAS_LDP_INSTALLED = False

from paperqa.llms import EmbeddingModel, LiteLLMModel, embedding_model_factory
from paperqa.prompts import (
    citation_prompt,
    default_system_prompt,
    qa_prompt,
    select_paper_prompt,
    structured_citation_prompt,
    summary_json_prompt,
    summary_json_system_prompt,
    summary_prompt,
)
from paperqa.utils import hexdigest, pqa_directory
from paperqa.version import __version__


class AnswerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_k: int = Field(
        default=10, description="Number of evidence pieces to retrieve"
    )
    evidence_detailed_citations: bool = Field(
        default=True, description="Whether to include detailed citations in summaries"
    )
    evidence_retrieval: bool = Field(
        default=True,
        description="Whether to use retrieval instead of processing all docs",
    )
    evidence_summary_length: str = Field(
        default="about 100 words", description="Length of evidence summary"
    )
    evidence_skip_summary: bool = Field(
        default=False, description="Whether to summarization"
    )
    answer_max_sources: int = Field(
        default=5, description="Max number of sources to use for an answer"
    )
    answer_length: str = Field(
        "about 200 words, but can be longer", description="Length of final answer"
    )
    max_concurrent_requests: int = Field(
        default=4, description="Max concurrent requests to LLMs"
    )
    answer_filter_extra_background: bool = Field(
        default=False,
        description="Whether to cite background information provided by model.",
    )


class ParsingOptions(StrEnum):
    PAPERQA_DEFAULT = "paperqa_default"

    def available_for_inference(self) -> list["ParsingOptions"]:
        return [self.PAPERQA_DEFAULT]  # type: ignore[list-item]


def _get_parse_type(opt: ParsingOptions, config: "ParsingSettings") -> str:
    if opt == ParsingOptions.PAPERQA_DEFAULT:
        return config.parser_version_string
    assert_never(opt)


class ChunkingOptions(StrEnum):
    SIMPLE_OVERLAP = "simple_overlap"

    @property
    def valid_parsings(self) -> list[ParsingOptions]:
        # Note that SIMPLE_OVERLAP must be valid for all by default
        # TODO: implement for future parsing options
        valid_parsing_dict: dict[str, list[ParsingOptions]] = {}
        return valid_parsing_dict.get(self.value, [])  # noqa: FURB184


class ParsingSettings(BaseModel):
    chunk_size: int = Field(default=3000, description="Number of characters per chunk")
    use_doc_details: bool = Field(
        default=True, description="Whether to try to get metadata details for a Doc"
    )
    overlap: int = Field(
        default=100, description="Number of characters to overlap chunks"
    )
    citation_prompt: str = Field(
        default=citation_prompt,
        description="Prompt that tries to create citation from peeking one page",
    )
    structured_citation_prompt: str = Field(
        default=structured_citation_prompt,
        description=(
            "Prompt that tries to creates a citation in JSON from peeking one page"
        ),
    )
    disable_doc_valid_check: bool = Field(
        default=False,
        description=(
            "Whether to disable checking if a document looks like text (was parsed"
            " correctly)"
        ),
    )
    chunking_algorithm: ChunkingOptions = ChunkingOptions.SIMPLE_OVERLAP
    model_config = ConfigDict(extra="forbid")

    def chunk_type(self, chunking_selection: ChunkingOptions | None = None) -> str:
        """Future chunking implementations (i.e. by section) will get an elif clause here."""
        if chunking_selection is None:
            chunking_selection = self.chunking_algorithm
        if chunking_selection == ChunkingOptions.SIMPLE_OVERLAP:
            return (
                f"{self.parser_version_string}|{chunking_selection.value}"
                f"|tokens={self.chunk_size}|overlap={self.overlap}"
            )
        assert_never(chunking_selection)

    @property
    def parser_version_string(self) -> str:
        return f"paperqa-{__version__}"

    def is_chunking_valid_for_parsing(self, parsing: str):
        # must map the parsings because they won't include versions by default
        return (
            self.chunking_algorithm == ChunkingOptions.SIMPLE_OVERLAP
            or parsing
            in {  # type: ignore[unreachable]
                _get_parse_type(p, self) for p in self.chunking_algorithm.valid_parsings
            }
        )


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
    model_config = ConfigDict(extra="forbid")

    summary: str = summary_prompt
    qa: str = qa_prompt
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
    use_json: bool = False
    # Not thrilled about this model,
    # but need to split out the system/summary
    # to get JSON
    summary_json: str = summary_json_prompt
    summary_json_system: str = summary_json_system_prompt
    EXAMPLE_CITATION: ClassVar[str] = "(Example2012Example pages 3-4)"

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

    @field_validator("pre")
    @classmethod
    def check_pre(cls, v: str | None) -> str | None:
        if v is not None and get_formatted_variables(v) != {"question"}:
            raise ValueError("Pre prompt must have input variables: question")
        return v

    @field_validator("post")
    @classmethod
    def check_post(cls, v: str | None) -> str | None:
        if v is not None:
            # kind of a hack to get list of attributes in answer
            from paperqa.types import Answer

            attrs = set(Answer.model_fields.keys())
            if not get_formatted_variables(v).issubset(attrs):
                raise ValueError(f"Post prompt must have input variables: {attrs}")
        return v


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_llm: str = Field(
        default="gpt-4o-2024-08-06",
        description="Model to use for agent",
    )

    agent_llm_config: dict | None = Field(
        default=None,
        description="Optional kwargs for LLM constructor",
    )

    agent_type: str = Field(
        default="fake",
        description="Type of agent to use",
    )
    agent_config: dict[str, Any] | None = Field(
        default=None,
        description="Optional kwarg for AGENT constructor",
    )

    agent_system_prompt: str | None = Field(
        # Matching https://github.com/langchain-ai/langchain/blob/langchain%3D%3D0.2.3/libs/langchain/langchain/agents/openai_functions_agent/base.py#L213-L215
        default="You are a helpful AI assistant.",
        description="Optional system prompt message to precede the below agent_prompt.",
    )

    # TODO: make this prompt more minimalist, instead improving tool descriptions so
    # how to use them together can be intuited, and exposing them for configuration
    agent_prompt: str = (
        "Use the tools to answer the question: {question}\n\nThe {gen_answer_tool_name}"
        " tool output is visible to the user, so you do not need to restate the answer"
        " and can simply terminate if the answer looks sufficient. The current status"
        " of evidence/papers/cost is {status}"
    )
    return_paper_metadata: bool = Field(
        default=False,
        description=(
            "Set True to have the search tool include paper title/year information as"
            " part of its return."
        ),
    )
    search_count: int = 8
    wipe_context_on_answer_failure: bool = True
    timeout: float = Field(
        default=500.0,
        description=(
            "Matches LangChain AgentExecutor.max_execution_time (seconds), the timeout"
            " on agent execution."
        ),
    )
    should_pre_search: bool = Field(
        default=False,
        description="If set to true, run the search tool before invoking agent.",
    )

    tool_names: set[str] | None = Field(
        default=None,
        description=(
            "Optional override on the tools to provide the agent. Leaving as the"
            " default of None will use a minimal toolset of the paper search, gather"
            " evidence, collect cited papers from evidence, and gen answer. If passing"
            " tool names (non-default route), at least the gen answer tool must be"
            " supplied."
        ),
    )

    index_concurrency: int = Field(
        default=30,
        description="Number of concurrent filesystem reads for indexing",
    )

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, v: set[str] | None) -> set[str] | None:
        if v is None:
            return None
        # imported here to avoid circular imports
        from paperqa.agents.tools import GenerateAnswer

        answer_tool_name = GenerateAnswer.TOOL_FN_NAME
        if answer_tool_name not in v:
            raise ValueError(
                f"If using an override, must contain at least the {answer_tool_name}."
            )
        return v


def make_default_litellm_router_settings(llm: str, temperature: float = 0.0) -> dict:
    """Settings matching "model_list" schema here: https://docs.litellm.ai/docs/routing."""
    return {
        "model_list": [
            {
                "model_name": llm,
                "litellm_params": {"model": llm, "temperature": temperature},
            }
        ]
    }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    llm: str = Field(
        default="gpt-4o-2024-08-06",
        description=(
            "Default LLM for most things, including answers. Should be 'best' LLM"
        ),
    )
    llm_config: dict | None = Field(
        default=None,
        description=(
            "LiteLLM Router configuration to pass to LiteLLMModel, must have"
            " `model_list` key (corresponding to model_list inputs here:"
            " https://docs.litellm.ai/docs/routing), and can optionally include a"
            " router_kwargs key with router kwargs as values."
        ),
    )
    summary_llm: str = Field(
        default="gpt-4o-2024-08-06",
        description="Default LLM for summaries and parsing citations",
    )
    summary_llm_config: dict | None = Field(
        default=None,
        description=(
            "LiteLLM Router configuration to pass to LiteLLMModel, must have"
            " `model_list` key (corresponding to model_list inputs here:"
            " https://docs.litellm.ai/docs/routing), and can optionally include a"
            " router_kwargs key with router kwargs as values."
        ),
    )
    embedding: str = Field(
        "text-embedding-3-small",
        description="Default embedding model for texts",
    )
    embedding_config: dict | None = Field(
        default=None,
        description="Extra kwargs to pass to embedding model",
    )
    temperature: float = Field(default=0.0, description="Temperature for LLMs")
    batch_size: int = Field(default=1, description="Batch size for calling LLMs")
    texts_index_mmr_lambda: float = Field(
        default=1.0, description="Lambda for MMR in text index"
    )
    index_absolute_directory: bool = Field(
        default=False,
        description="Whether to use the absolute directory for the PQA index",
    )
    index_directory: str | os.PathLike | None = Field(
        default=pqa_directory("indexes"),
        description=(
            "Directory to store the PQA generated search index, configuration, and"
            " answer indexes."
        ),
    )
    index_recursively: bool = Field(
        default=True,
        description="Whether to recurse into subdirectories when indexing sources.",
    )
    verbosity: int = Field(
        default=0,
        description=(
            "Integer verbosity level for logging (0-3). 3 = all LLM/Embeddings calls"
            " logged"
        ),
    )
    manifest_file: str | os.PathLike | None = Field(
        default=None,
        description=(
            "Optional manifest CSV, containing columns which are attributes for a"
            " DocDetails object. Only 'file_location','doi', and 'title' will be used"
            " when indexing."
        ),
    )
    paper_directory: str | os.PathLike = Field(
        default=Path.cwd(),
        description=(
            "Local directory which contains the papers to be indexed and searched."
        ),
    )

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
        # index name should use an absolute path
        # this way two different folders where the
        # user locally uses '.' will make different indexes
        paper_directory = self.paper_directory
        if isinstance(paper_directory, Path):
            paper_directory = str(paper_directory.absolute())

        index_fields = "|".join(
            [
                str(paper_directory),
                self.embedding,
                str(self.parsing.chunk_size),
                str(self.parsing.overlap),
                self.parsing.chunking_algorithm,
            ]
        )

        return f"pqa_index_{hexdigest(index_fields)}"

    @classmethod
    def from_name(
        cls, config_name: str = "default", cli_source: CliSettingsSource | None = None
    ) -> "Settings":
        json_path: Path | None = None

        # quick exit for default settings
        if config_name == "default":
            if not cli_source:
                raise NotImplementedError(
                    f"For config_name {config_name!r}, we require cli_source."
                )
            return Settings(_cli_settings_source=cli_source(args=True))

        # First, try to find the config file in the user's .config directory
        user_config_path = pqa_directory("settings") / f"{config_name}.json"

        if user_config_path.exists():
            json_path = user_config_path

        # If not found, fall back to the package's default config
        try:
            # Use importlib.resources.files() which is recommended for Python 3.9+
            pkg_config_path = (
                importlib.resources.files("paperqa.configs") / f"{config_name}.json"
            )
            if pkg_config_path.is_file():
                json_path = cast(Path, pkg_config_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No configuration file found for {config_name}"
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

        raise FileNotFoundError(f"No configuration file found for {config_name}")

    def get_llm(self) -> LiteLLMModel:
        return LiteLLMModel(
            name=self.llm,
            config=self.llm_config
            or make_default_litellm_router_settings(self.llm, self.temperature),
        )

    def get_summary_llm(self) -> LiteLLMModel:
        return LiteLLMModel(
            name=self.summary_llm,
            config=self.summary_llm_config
            or make_default_litellm_router_settings(self.summary_llm, self.temperature),
        )

    def get_agent_llm(self) -> LiteLLMModel:
        return LiteLLMModel(
            name=self.agent.agent_llm,
            config=self.agent.agent_llm_config
            or make_default_litellm_router_settings(
                self.agent.agent_llm, self.temperature
            ),
        )

    def get_embedding_model(self) -> EmbeddingModel:
        return embedding_model_factory(self.embedding, **(self.embedding_config or {}))

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
                acompletion=self.get_agent_llm().router.acompletion,
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
        agent_cls = cast(type[Agent], locate(agent_type))
        agent_settings = self.agent
        agent_llm, config = agent_settings.agent_llm, agent_settings.agent_config or {}
        if issubclass(agent_cls, ReActAgent | MemoryAgent):
            if (
                issubclass(agent_cls, MemoryAgent)
                and "memory_model" in config
                and "memories" in config
            ):
                if "embedding_model" in config["memory_model"]:
                    # Work around LDPEmbeddingModel not yet supporting deserialization
                    config["memory_model"]["embedding_model"] = (
                        LDPEmbeddingModel.from_name(
                            embedding=config["memory_model"].pop("embedding_model")[
                                "name"
                            ]
                        )
                    )
                config["memory_model"] = UIndexMemoryModel(**config["memory_model"])
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
                llm_model={"model": agent_llm, "temperature": self.temperature},
                **config,
            )
        if issubclass(agent_cls, SimpleAgent):
            return agent_cls(
                llm_model={"model": agent_llm, "temperature": self.temperature},
                sys_prompt=agent_settings.agent_system_prompt,
                **config,
            )
        if issubclass(agent_cls, HTTPAgentClient):
            set_training_mode(False)
            return HTTPAgentClient[SimpleAgentState](
                agent_state_type=SimpleAgentState, **config
            )
        raise NotImplementedError(f"Didn't yet handle agent type {agent_type}.")


MaybeSettings = Settings | str | None


def get_settings(config_or_name: MaybeSettings = None) -> Settings:
    if isinstance(config_or_name, Settings):
        return config_or_name
    if config_or_name is None:
        return Settings()
    return Settings.from_name(config_name=config_or_name)
