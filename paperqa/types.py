from __future__ import annotations

from typing import Any, Callable
from uuid import UUID, uuid4

import tiktoken
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from .prompts import (
    citation_prompt,
    default_system_prompt,
    qa_prompt,
    select_paper_prompt,
    summary_json_prompt,
    summary_json_system_prompt,
    summary_prompt,
)
from .utils import get_citenames
from .version import __version__ as pqa_version

# Just for clarity
DocKey = Any
CallbackFactory = Callable[[str], list[Callable[[str], None]] | None]


class LLMResult(BaseModel):
    """A class to hold the result of a LLM completion."""

    id: UUID = Field(default_factory=uuid4)
    answer_id: UUID | None = None
    name: str | None = None
    prompt: str | list[dict] | None = Field(
        default=None,
        description="Optional prompt (str) or list of serialized prompts (list[dict]).",
    )
    text: str = ""
    prompt_count: int = 0
    completion_count: int = 0
    model: str
    date: str
    seconds_to_first_token: float = Field(
        default=0.0, description="Delta time (sec) to first response token's arrival."
    )
    seconds_to_last_token: float = Field(
        default=0.0, description="Delta time (sec) to last response token's arrival."
    )

    def __str__(self):
        return self.text


class Embeddable(BaseModel):
    embedding: list[float] | None = Field(default=None, repr=False)


class Doc(Embeddable):
    docname: str
    citation: str
    dockey: DocKey

    def __hash__(self) -> int:
        return hash((self.docname, self.dockey))


class Text(Embeddable):
    text: str
    name: str
    doc: Doc


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


class PromptCollection(BaseModel):
    summary: str = summary_prompt
    qa: str = qa_prompt
    select: str = select_paper_prompt
    cite: str = citation_prompt
    pre: str | None = None
    post: str | None = None
    system: str = default_system_prompt
    skip_summary: bool = False
    json_summary: bool = False
    # Not thrilled about this model,
    # but need to split out the system/summary
    # to get JSON
    summary_json: str = summary_json_prompt
    summary_json_system: str = summary_json_system_prompt

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


class Context(BaseModel):
    """A class to hold the context of a question."""

    context: str
    text: Text
    score: int = 5
    model_config = ConfigDict(
        extra="allow",
    )


def __str__(self) -> str:  # noqa: N807
    """Return the context as a string."""
    return self.context


class Answer(BaseModel):
    """A class to hold the answer to a question."""

    id: UUID = Field(default_factory=uuid4)
    question: str
    answer: str = ""
    context: str = ""
    contexts: list[Context] = []
    references: str = ""
    formatted_answer: str = ""
    dockey_filter: set[DocKey] | None = None
    summary_length: str = "about 100 words"
    answer_length: str = "about 100 words"
    # just for convenience you can override this
    cost: float | None = None
    # Map model name to a two-item list of LLM prompt token counts
    # and LLM completion token counts
    token_counts: dict[str, list[int]] = Field(default_factory=dict)
    model_config = ConfigDict(extra="ignore")

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer

    @model_validator(mode="before")
    @classmethod
    def remove_computed(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.pop("used_contexts", None)
        return data

    @computed_field  # type: ignore[misc]
    @property
    def used_contexts(self) -> set[str]:
        """Return the used contexts."""
        return get_citenames(self.formatted_answer)

    def get_citation(self, name: str) -> str:
        """Return the formatted citation for the given docname."""
        try:
            doc: Doc = next(
                filter(lambda x: x.text.name == name, self.contexts)
            ).text.doc
        except StopIteration as exc:
            raise ValueError(f"Could not find docname {name} in contexts.") from exc
        return doc.citation

    def add_tokens(self, result: LLMResult):
        """Update the token counts for the given result."""
        if result.model not in self.token_counts:
            self.token_counts[result.model] = [
                result.prompt_count,
                result.completion_count,
            ]
        else:
            self.token_counts[result.model][0] += result.prompt_count
            self.token_counts[result.model][1] += result.completion_count

    def get_unique_docs_from_contexts(self, score_threshold: int = 0) -> set[Doc]:
        """Parse contexts for docs with scores above the input threshold."""
        return {
            c.text.doc
            for c in filter(lambda x: x.score >= score_threshold, self.contexts)
        }


class ChunkMetadata(BaseModel):
    """Metadata for chunking algorithm."""

    chunk_chars: int
    overlap: int
    chunk_type: str


class ParsedMetadata(BaseModel):
    """Metadata for parsed text."""

    parsing_libraries: list[str]
    total_parsed_text_length: int
    paperqa_version: str = pqa_version
    parse_type: str | None = None
    chunk_metadata: ChunkMetadata | None = None


class ParsedText(BaseModel):
    """Parsed text (pre-chunking)."""

    content: dict | str | list[str]
    metadata: ParsedMetadata

    def encode_content(self):
        # we tokenize using tiktoken so cuts are in reasonable places
        # See https://github.com/openai/tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        if isinstance(self.content, str):
            return enc.encode_ordinary(self.content)
        elif isinstance(self.content, list):  # noqa: RET505
            return [enc.encode_ordinary(c) for c in self.content]
        else:
            raise NotImplementedError(
                "Encoding only implemented for str and list[str] content."
            )
