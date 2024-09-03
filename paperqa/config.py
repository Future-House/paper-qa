import importlib.resources
from pathlib import Path
from typing import ClassVar, cast

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic_settings import (
    BaseSettings,
)

from .paths import PAPERQA_DIR
from .prompts import (
    citation_prompt,
    default_system_prompt,
    qa_prompt,
    select_paper_prompt,
    structured_citation_prompt,
    summary_json_prompt,
    summary_json_system_prompt,
    summary_prompt,
)
from .types import Answer


class AnswerSettings(BaseModel):
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
    answer_max_sources: int = Field(
        default=5, description="Max number of sources to use for an answer"
    )
    answer_length: str = Field(
        "about 200 words, but can be longer", description="Length of final answer"
    )
    max_concurrent_requests: int = Field(
        default=4, description="Max concurrent requests to LLMs"
    )

    @field_validator("answer_max_sources")
    @classmethod
    def k_should_be_greater_than_max_sources(cls, v: int, info: ValidationInfo) -> int:
        if v > info.data["evidence_k"]:
            raise ValueError(
                "answer_max_sources should be less than or equal to doc_match_k"
            )
        return v


class ParsingSettings(BaseModel):
    chunk_chars: int = Field(default=3000, description="Number of characters per chunk")
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
        description="Prompt that tries to creates a citation in JSON from peeking one page",
    )
    disable_doc_valid_check: bool = Field(
        default=False,
        description="Whether to disable checking if a document looks like text (was parsed correctly)",
    )


# Mock a dictionary and store any missing items
class _FormatDict(dict):
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
    skip_summary: bool = False
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
        if not set(get_formatted_variables(v)).issubset(
            set(get_formatted_variables(summary_prompt))
        ):
            raise ValueError(
                f"Summary prompt can only have variables: {get_formatted_variables(summary_prompt)}"
            )
        return v

    @field_validator("qa")
    @classmethod
    def check_qa(cls, v: str) -> str:
        if not set(get_formatted_variables(v)).issubset(
            set(get_formatted_variables(qa_prompt))
        ):
            raise ValueError(
                f"QA prompt can only have variables: {get_formatted_variables(qa_prompt)}"
            )
        return v

    @field_validator("select")
    @classmethod
    def check_select(cls, v: str) -> str:
        if not set(get_formatted_variables(v)).issubset(
            set(get_formatted_variables(select_paper_prompt))
        ):
            raise ValueError(
                f"Select prompt can only have variables: {get_formatted_variables(select_paper_prompt)}"
            )
        return v

    @field_validator("pre")
    @classmethod
    def check_pre(cls, v: str | None) -> str | None:
        if v is not None and set(get_formatted_variables(v)) != {"question"}:
            raise ValueError("Pre prompt must have input variables: question")
        return v

    @field_validator("post")
    @classmethod
    def check_post(cls, v: str | None) -> str | None:
        if v is not None:
            # kind of a hack to get list of attributes in answer
            attrs = set(Answer.model_fields.keys())
            if not set(get_formatted_variables(v)).issubset(attrs):
                raise ValueError(f"Post prompt must have input variables: {attrs}")
        return v


class Settings(BaseSettings):
    llm: str = Field(
        default="openai/gpt-4o-2024-08-06",
        description="Default LLM for most things, including answers. Should be 'best' LLM",
    )
    summary_llm: str = Field(
        default="openai/gpt-4o-2024-08-06",
        description="Default LLM for summaries and parsing citations",
    )
    batch_size: int = Field(default=1, description="Batch size for calling LLMs")
    answer: AnswerSettings = AnswerSettings()
    parsing: ParsingSettings = ParsingSettings()
    prompts: PromptSettings = PromptSettings()

    @classmethod
    def from_name(cls, config_name: str) -> "Settings":
        json_path: Path | None = None

        # First, try to find the config file in the user's .config directory
        user_config_path = PAPERQA_DIR / f"{config_name}.json"

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
            return Settings.model_validate_json(json_path.read_text())

        raise FileNotFoundError(f"No configuration file found for {config_name}")


MaybeSettings = Settings | str | None


def get_settings(config_or_name: MaybeSettings = None) -> Settings:
    if isinstance(config_or_name, Settings):
        return config_or_name
    if config_or_name is None:
        return Settings()

    config_name = config_or_name
    return Settings.from_name(config_name)
