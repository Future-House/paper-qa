from __future__ import annotations

from datetime import datetime
import logging
import re
from typing import Any, Callable, ClassVar, Collection
from uuid import UUID, uuid4

import aiohttp
from networkx import volume
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
from .utils import encode_id, get_citenames
from .version import __version__ as pqa_version

# Just for clarity
DocKey = Any
CallbackFactory = Callable[[str], list[Callable[[str], None]] | None]

logger = logging.getLogger(__name__)

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
    date: str = Field(default_factory=datetime.now().isoformat)
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
    details: PaperDetails

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

    model_config = ConfigDict(extra="allow")

    context: str = Field(description="Summary of the text with respect to a question.")
    text: Text
    score: int = 5


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


# We use these integer values
# as defined in https://jfp.csc.fi/en/web/haku/kayttoohje
# which is a recommended ranking system
SOURCE_QUALITY_MESSAGES = {
    0: "poor quality or predatory journal",
    1: "peer-reviewed journal",
    2: "domain leading peer-reviewed journal",
    3: "highest quality peer-reviewed journal",
}


class PaperDetails(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    citation: str | None = None
    key: str | None = None
    bibtex: str | None = None #TODO: may be able to build this via method
    authors: list[str] | None = None
    publication_date: datetime | None = None
    year: int | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    journal: str | None = None
    year: int | None = None
    url: str | None = Field(
        default=None,
        description=(
            "Optional URL to the paper, which can lead to a Semantic Scholar page,"
            " arXiv abstract, etc. As of version 0.67 on 5/10/2024, we don't use this"
            " URL anywhere in the source code."
        ),
    )
    title: str | None = None
    citation_count: int | None = None  # noqa: N815

    source_quality: int | None = Field(
        default=None,
        description="Quality of journal/venue of paper. "
        " We use None as a sentinel for unset values (like for determining hydration) "
        " So, we use -1 means unknown quality and None means it needs to be hydrated.",
    )

    doi: str | None = None
    paper_id: str | None = None  # noqa: N815
    other: dict[str, Any] = Field(
        default_factory=dict,
        description="Other metadata besides the above standardized fields.",
    )
    UNDEFINED_JOURNAL_QUALITY: ClassVar[int] = -1

    @field_validator("key")
    @classmethod
    def clean_key(cls, value: str) -> str:
        # Replace HTML tags with empty string
        return re.sub(pattern=r"<\/?\w{1,10}>", repl="", string=value)

    def __getitem__(self, item: str):
        """Allow for dictionary-like access, falling back on other."""
        try:
            return getattr(self, item)
        except AttributeError:
            return self.other[item]

    @classmethod
    async def get_source_quality(cls, bibtex: str) -> int:
        # extract journal name (or None)
        match = re.search(r"journal\s*=\s*{([^}]+)}", bibtex)
        journal_name = match.group(1) if match else None
        # TODO: implement JournalQualityDB
        # if journal_name:
        #     record = await JournalQualityDB.filter(
        #         journal=journal_name.casefold()
        #     ).first()
        #     if record:
        #         return record.quality
        #     logger.warning(
        #         f"Failed to find journal {journal_name!r} in JournalQualityDB."
        #     )
        return cls.UNDEFINED_JOURNAL_QUALITY

    @property
    def formatted_citation(self) -> str:
        if (
            self.citation is None
            or self.citation_count is None
            or self.source_quality is None
        ):
            raise ValueError(
                "Citation, citationCount, and sourceQuality are not set -- do you need to call `hydrate`?"
            )
        quality = (
            SOURCE_QUALITY_MESSAGES[self.source_quality]
            if self.source_quality >= 0
            else None
        )
        if quality is None:
            return f"{self.citation} This article has {self.citation_count} citations."
        return (
            f"{self.citation} This article has {self.citation_count} citations and is from a "
            f"{quality}."
        )

    @property
    def docname(self) -> str | None:
        """Alias for key."""
        return self.key

    OPTIONAL_HYDRATION_FIELDS: ClassVar[Collection[str]] = {"url"}

    def is_hydration_needed(  # pylint: disable=dangerous-default-value
        self, exclusion: Collection[str] = OPTIONAL_HYDRATION_FIELDS
    ) -> bool:
        """Determine if we have unfilled attributes."""
        return any(
            v is None for k, v in self.model_dump().items() if k not in exclusion
        )
    
    async def populate_paper_id(self) -> None:
        #TODO: should this be a hash of the doi?
        if self.paper_id is None:
            self.paper_id = encode_id(uuid4())

    async def metadata_from_crossref(self, **kwargs) -> dict[str, Any]:
        """Crossref metadata returned in the format expected by scraper parse funcs."""
        return await get_paper_details(doi=self.doi, title=self.title, **kwargs)

    # NOTE: export CROSSREF_MAILTO with a real email address for better experience
    async def _hydrate_from_crossref_metadata(
        self,
        force_overwrite: bool = False,
        session: aiohttp.ClientSession | None = None,
        other_keys_overrides: Collection[str] | None = None,
    ) -> PaperDetails:
        """Populate attributes in-place using Crossref then Google Scholar."""
        if not self.is_hydration_needed() and not force_overwrite:
            return self

        async def parse(session_: aiohttp.ClientSession) -> PaperDetails:
            cr_metadata = await self.metadata_from_crossref(session=session_)
            other_keys = set(cr_metadata.keys())
            for k, v in (
                # NOTE: shared rate limits between gs/crossref
                await parse_google_scholar_metadata(cr_metadata, session_)
            ).items():
                if not other_keys_overrides or k not in other_keys_overrides:
                    other_keys.discard(k)
                    setattr(self, k, v)
                elif k in cr_metadata:
                    other_keys.add(k)
            # Preserve remaining metadata (e.g. externalIds) for scraping
            self.other |= {k: cr_metadata[k] for k in other_keys}
            return self

        if session is not None:
            return await parse(session)
        async with ThrottledClientSession(
            headers=get_header(), rate_limit=RateLimits.CROSSREF.value
        ) as session:  # noqa: PLR1704
            return await parse(session)

    async def hydrate(
        self, no_db_cache: bool = False, **crossref_hydration_kwargs
    ) -> PaperDetails:
        if "force_overwrite" in crossref_hydration_kwargs:
            raise NotImplementedError("Didn't yet handle force overwrite in logic.")
        if not no_db_cache:
            with contextlib.suppress(
                AttributeError  # Suppress failures and fall back on Crossref
            ):
                await self._hydrate_from_db()
        await self._hydrate_from_crossref_metadata(**crossref_hydration_kwargs)
        if self.bibtex:
            self.source_quality = await PaperDetails.get_source_quality(self.bibtex)
        else:
            raise ValueError("Bibtex was not set from hydration")
        return self

    async def safe_hydrate(
        self, required: Collection[str] | None = None, **hydrate_kwargs
    ) -> PaperDetails:
        """
        Hydrate while suppressing common hydration exceptions.

        Use this over vanilla hydrate when tolerant to missing information.
        """
        try:
            return await self.hydrate(**hydrate_kwargs)
        except (RuntimeError, DOINotFoundError) as exc:
            if required and (
                missing := {field for field in required if getattr(self, field) is None}
            ):
                raise RuntimeError(
                    f"Missing required fields {missing} in safe hydration of paper"
                    f" details {self}."
                ) from exc
            # RuntimeError: Failed to follow citations link
            # DOINotFoundError: Failed to fall back on Google Scholar
            logger.warning(f"Safe failure to hydrate paper details {self}.")
            return self

    @classmethod
    def from_crossref(cls, metadata: dict[str, Any]) -> PaperDetails:
        """Convert a Crossref reference into PaperDetails."""
        doi: str | None = metadata.pop("DOI", None)
        details = cls(
            year=int(metadata.pop("year")) if "year" in metadata else None,
            title=metadata.pop("article-title", None),
            doi=doi.lower() if doi else None,
            other=metadata | {"source": "Crossref"},
        )
        if details.doi is details.title is None:
            raise ValueError(
                f"Paper details {details} has no DOI or title, so it's useless."
            )
        return details

    @classmethod
    def from_google_scholar(cls, metadata: dict[str, Any]) -> PaperDetails:
        """Convert a Google Scholar organic result into PaperDetails."""
        doi: str | None = find_doi(metadata.get("link", ""))
        return cls(
            title=metadata.pop("title"),
            citationCount=(
                metadata.get("inline_links", {}).get("cited_by", {}).get("total")
            ),
            doi=doi.lower() if doi else None,
            other=metadata | {"source": "Google Scholar"},
        )

    @classmethod
    def from_semantic_scholar(cls, metadata: dict[str, Any]) -> PaperDetails:
        """Convert a Semantic Scholar result into PaperDetails."""
        # TODO: combine with paperscraper parse_semantic_scholar_metadata for
        # key and citation
        bibtex = metadata.pop("bibtex", None) or (
            metadata.get("citationStyles") or {}
        ).get("bibtex")
        doi: str | None = (metadata.get("externalIds") or {}).get("DOI")
        return cls(
            bibtex=(clean_upbibtex(bibtex) if bibtex is not None else None),
            year=metadata.pop("year"),
            url=metadata.pop("url"),
            title=metadata.pop("title"),
            citationCount=metadata.pop("citationCount"),
            doi=doi.lower() if doi else None,
            paper_id=metadata.pop("paper_id"),
            other=metadata | {"source": "Semantic Scholar"},
        )

    def to_scrape_metadata(self) -> dict[str, Any]:
        """Convert to a format ingestible by scraper functions."""
        return self.other | self.model_dump(exclude={"other"})
    