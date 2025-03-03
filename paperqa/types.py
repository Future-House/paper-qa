from __future__ import annotations

import logging
import os
import re
import warnings
from collections.abc import Collection, Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any, ClassVar, cast
from uuid import UUID, uuid4

import tiktoken
from aviary.core import Message
from lmi import Embeddable, LLMResult
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.input.bibtex import Parser
from pybtex.scanner import PybtexSyntaxError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from paperqa.utils import (
    create_bibtex_key,
    encode_id,
    format_bibtex,
    get_citenames,
    maybe_get_date,
)
from paperqa.version import __version__ as pqa_version

# Just for clarity
# also in case one day we want to narrow
# the type
DocKey = Any
logger = logging.getLogger(__name__)


VAR_MATCH_LOOKUP: Collection[str] = {"1", "true"}
VAR_MISMATCH_LOOKUP: Collection[str] = {"0", "false"}
DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA: Collection[str] = {
    "key",
    "doc_id",
    "docname",
    "dockey",
    "citation",
}


class Doc(Embeddable):
    model_config = ConfigDict(extra="forbid")

    docname: str
    dockey: DocKey
    citation: str
    fields_to_overwrite_from_metadata: set[str] = Field(
        default_factory=lambda: set(DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA),
        description="fields from metadata to overwrite when upgrading to a DocDetails",
    )

    @model_validator(mode="before")
    @classmethod
    def remove_computed_fields(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in data.items() if k != "formatted_citation"}

    def __hash__(self) -> int:
        return hash((self.docname, self.dockey))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def formatted_citation(self) -> str:
        return self.citation

    def matches_filter_criteria(self, filter_criteria: Mapping[str, Any]) -> bool:
        """Returns True if the doc matches the filter criteria, False otherwise."""
        data_dict = self.model_dump()
        for key, value in filter_criteria.items():
            invert = key.startswith("!")
            relaxed = key.startswith("?")
            key = key.lstrip("!?")
            # we check if missing or sentinel/unset
            if relaxed and (key not in data_dict or data_dict[key] is None):
                continue
            if key not in data_dict:
                return False
            if invert and data_dict[key] == value:
                return False
            if not invert and data_dict[key] != value:
                return False
        return True


class Text(Embeddable):
    text: str
    name: str
    doc: Doc | DocDetails = Field(union_mode="left_to_right")

    def __hash__(self) -> int:
        return hash(self.text)


class Context(BaseModel):
    """A class to hold the context of a question."""

    model_config = ConfigDict(extra="allow")

    context: str = Field(description="Summary of the text with respect to a question.")
    text: Text
    score: int = 5

    def __str__(self) -> str:
        """Return the context as a string."""
        return self.context


class PQASession(BaseModel):
    """A class to hold session about researching/answering."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: UUID = Field(default_factory=uuid4)
    question: str
    answer: str = ""
    answer_reasoning: str | None = Field(
        default=None,
        description=(
            "Optional reasoning from the LLM. If the LLM does not support reasoning,"
            " it will be None."
        ),
    )
    has_successful_answer: bool | None = Field(
        default=None,
        description=(
            "True if the agent was sure of the answer, False if the agent was unsure of"
            " the answer, and None if the agent hasn't yet completed."
        ),
    )
    context: str = ""
    contexts: list[Context] = Field(default_factory=list)
    references: str = ""
    formatted_answer: str = Field(
        default="",
        description=(
            "Optional prettified answer that includes information like question and"
            " citations."
        ),
    )
    graded_answer: str | None = Field(
        default=None,
        description=(
            "Optional graded answer, used for things like multiple choice questions."
        ),
    )
    cost: float = 0.0
    # Map model name to a two-item list of LLM prompt token counts
    # and LLM completion token counts
    token_counts: dict[str, list[int]] = Field(default_factory=dict)
    config_md5: str | None = Field(
        default=None,
        frozen=True,
        description=(
            "MD5 hash of the settings used to generate the answer. Cannot change"
        ),
    )
    tool_history: list[list[str]] = Field(
        default_factory=list,
        description=(
            "History of tool names input to each Environment.step (regardless of being"
            " a typo or not), where the outer list is steps, and the inner list matches"
            " the order of tool calls at each step."
        ),
    )

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer

    @model_validator(mode="before")
    @classmethod
    def remove_computed(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.pop("used_contexts", None)
        return data

    @computed_field  # type: ignore[prop-decorator]
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

    def add_tokens(self, result: LLMResult | Message) -> None:
        """Update the token counts for the given LLM result or message."""
        if isinstance(result, Message):
            if not result.info or any(x not in result.info for x in ("model", "usage")):
                return
            result = LLMResult(
                model=result.info["model"],
                prompt_count=result.info["usage"][0],
                completion_count=result.info["usage"][1],
            )
        if result.model not in self.token_counts:
            self.token_counts[result.model] = [
                result.prompt_count,
                result.completion_count,
            ]
        else:
            self.token_counts[result.model][0] += result.prompt_count
            self.token_counts[result.model][1] += result.completion_count

        self.cost += result.cost

    def get_unique_docs_from_contexts(self, score_threshold: int = 0) -> set[Doc]:
        """Parse contexts for docs with scores above the input threshold."""
        return {
            c.text.doc
            for c in filter(lambda x: x.score >= score_threshold, self.contexts)
        }

    def filter_content_for_user(self) -> None:
        """Filter out extra items (inplace) that do not need to be returned to the user."""
        self.contexts = [
            Context(
                context=c.context,
                score=c.score,
                text=Text(
                    text="",
                    **c.text.model_dump(exclude={"text", "embedding", "doc"}),
                    doc=c.text.doc.model_dump(exclude={"embedding"}),
                ),
            )
            for c in self.contexts
        ]


# for backwards compatibility
class Answer(PQASession):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The 'Answer' class is deprecated and will be removed in future versions."
            " Use 'PQASession' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


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

    def reduce_content(self) -> str:
        """Reduce any content to a string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n\n".join(self.content)
        return "\n\n".join(self.content.values())


# We use these integer values
# as defined in https://jfp.csc.fi/en/web/haku/kayttoohje
# which is a recommended ranking system
SOURCE_QUALITY_MESSAGES = {
    0: "poor quality or predatory journal",
    1: "peer-reviewed journal",
    2: "domain leading peer-reviewed journal",
    3: "highest quality peer-reviewed journal",
}

CITATION_FALLBACK_DATA = {
    "authors": ["Unknown authors"],
    "author": "Unknown author(s)",
    "year": "Unknown year",
    "title": "Unknown title",
    "journal": "Unknown journal",
}

JOURNAL_EXPECTED_DOI_LENGTHS = {
    "BioRxiv": 25,
    "MedRxiv": 27,
}


class DocDetails(Doc):
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    # Sentinel to auto-populate a field within model_validator
    AUTOPOPULATE_VALUE: ClassVar[str] = ""

    docname: str = AUTOPOPULATE_VALUE
    dockey: DocKey = AUTOPOPULATE_VALUE
    citation: str = AUTOPOPULATE_VALUE
    key: str | None = None
    bibtex: str | None = Field(
        default=AUTOPOPULATE_VALUE,
        description="Autogenerated from other represented fields.",
    )
    authors: list[str] | None = None
    publication_date: datetime | None = None
    year: int | None = None
    volume: str | None = None
    issue: str | None = None  # TODO: in bibtex this may be "number"
    issn: str | None = None
    pages: str | None = None
    journal: str | None = None
    publisher: str | None = None
    url: str | None = Field(
        default=None,
        description=(
            "Optional URL to the paper, which can lead to a Semantic Scholar page,"
            " arXiv abstract, etc. As of version 0.67 on 5/10/2024, we don't use this"
            " URL anywhere in the source code."
        ),
    )
    title: str | None = None
    citation_count: int | None = None
    bibtex_type: str | None = None

    source_quality: int | None = Field(
        default=None,
        description=(
            "Quality of journal/venue of paper.  We use None as a sentinel for unset"
            " values (like for determining hydration)  So, we use -1 means unknown"
            " quality and None means it needs to be hydrated."
        ),
    )

    is_retracted: bool | None = Field(
        default=None, description="Flag for whether the paper is retracted."
    )
    doi: str | None = None
    doi_url: str | None = None
    doc_id: str | None = None
    file_location: str | os.PathLike | None = None
    license: str | None = Field(
        default=None,
        description=(
            "string indicating license. Should refer specifically to pdf_url (since"
            " that could be preprint). None means unknown/unset."
        ),
    )
    pdf_url: str | None = None
    other: dict[str, Any] = Field(
        default_factory=dict,
        description="Other metadata besides the above standardized fields.",
    )
    UNDEFINED_JOURNAL_QUALITY: ClassVar[int] = -1
    # NOTE: can use a regex starting from the pattern in https://regex101.com/r/lpF1up/1
    DOI_URL_FORMATS: ClassVar[Collection[str]] = {
        "https://doi.org/",
        "http://dx.doi.org/",
    }
    AUTHOR_NAMES_TO_REMOVE: ClassVar[Collection[str]] = {"et al", "et al."}

    @field_validator("key")
    @classmethod
    def clean_key(cls, value: str) -> str:
        # Replace HTML tags with empty string
        return re.sub(pattern=r"<\/?\w{1,10}>", repl="", string=value)

    @classmethod
    def lowercase_doi_and_populate_doc_id(cls, data: dict[str, Any]) -> dict[str, Any]:
        doi: str | list[str] | None = data.get("doi")
        if isinstance(doi, list):
            if len(doi) != 1:
                logger.warning(
                    f"Discarding list of DOIs {doi} due to it not having one value,"
                    f" full data was {data}."
                )
                doi = None
            else:
                doi = doi[0]
        if doi:
            for url_prefix_to_remove in cls.DOI_URL_FORMATS:
                if doi.startswith(url_prefix_to_remove):
                    doi = doi.replace(url_prefix_to_remove, "")
            data["doi"] = doi.lower()
            data["doc_id"] = encode_id(doi.lower())
        else:
            data["doc_id"] = encode_id(uuid4())

        if "dockey" in data.get(
            "fields_to_overwrite_from_metadata",
            DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA,
        ):
            data["dockey"] = data["doc_id"]

        return data

    @staticmethod
    def is_bibtex_complete(bibtex: str, fields: list[str] | None = None) -> bool:
        """Validate bibtex entries have certain fields."""
        if fields is None:
            fields = ["doi", "title"]
        return all(field + "=" in bibtex for field in fields)

    @staticmethod
    def merge_bibtex_entries(entry1: Entry, entry2: Entry) -> Entry:
        """Merge two bibtex entries into one, preferring entry2 fields."""
        merged_entry = Entry(entry1.type)

        for field, value in entry1.fields.items():
            merged_entry.fields[field] = value
        for field, value in entry2.fields.items():
            merged_entry.fields[field] = value

        return merged_entry

    @staticmethod
    def misc_string_cleaning(data: dict[str, Any]) -> dict[str, Any]:
        """Clean strings before the enter the validation process."""
        if pages := data.get("pages"):
            data["pages"] = pages.replace("--", "-").replace(" ", "")
        return data

    @staticmethod
    def inject_clean_doi_url_into_data(data: dict[str, Any]) -> dict[str, Any]:
        """Ensure doi_url is present in data (since non-default arguments are not included)."""
        doi_url, doi = data.get("doi_url"), data.get("doi")

        if doi and not doi_url:
            doi_url = "https://doi.org/" + doi

        # ensure the modern doi url is used
        if doi_url:
            data["doi_url"] = doi_url.replace(
                "http://dx.doi.org/", "https://doi.org/"
            ).lower()

        return data

    @staticmethod
    def add_preprint_journal_from_doi_if_missing(
        data: dict[str, Any],
    ) -> dict[str, Any]:
        if not data.get("journal"):
            doi = data.get("doi", "") or ""
            if "10.48550/" in doi or "ArXiv" in (
                (data.get("other", {}) or {}).get("externalIds", {}) or {}
            ):
                data["journal"] = "ArXiv"
            elif "10.26434/" in doi:
                data["journal"] = "ChemRxiv"
            elif (
                "10.1101/" in doi
                and len(data.get("doi", "")) == JOURNAL_EXPECTED_DOI_LENGTHS["BioRxiv"]
            ):
                data["journal"] = "BioRxiv"
            elif (
                "10.1101/" in doi
                and len(data.get("doi", "")) == JOURNAL_EXPECTED_DOI_LENGTHS["MedRxiv"]
            ):
                data["journal"] = "MedRxiv"
            elif "10.31224/" in doi:
                data["journal"] = "EngRxiv"
        return data

    @classmethod
    def remove_invalid_authors(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Capture and cull strange author names."""
        if authors := data.get("authors"):
            # On 10/29/2024 while indexing 19k PDFs, a provider (unclear which one)
            # returned an author of None. The vast majority of the time authors are str
            authors = cast("list[str | None]", authors)
            data["authors"] = [
                a for a in authors if a and a.lower() not in cls.AUTHOR_NAMES_TO_REMOVE
            ]

        return data

    @staticmethod
    def overwrite_docname_dockey_for_compatibility_w_doc(
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Overwrite fields from metadata if specified."""
        overwrite_fields = {"key": "docname", "doc_id": "dockey"}
        fields_to_overwrite = data.get(
            "fields_to_overwrite_from_metadata",
            DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA,
        )
        for field in overwrite_fields.keys() & fields_to_overwrite:
            if data.get(field):
                data[overwrite_fields[field]] = data[field]
        return data

    @classmethod
    def populate_bibtex_key_citation(  # noqa: PLR0912
        cls, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Add or modify bibtex, key, and citation fields.

        Missing values, 'unknown' keys, and incomplete bibtex entries are regenerated.

        When fields_to_overwrite_from_metadata:
            If bibtex is regenerated, the citation field is also regenerated.

            Otherwise we keep the citation field as is.

        """
        # we try to regenerate the key if unknowns are present, maybe they have been found
        if not data.get("key") or "unknown" in data["key"].lower():
            data["key"] = create_bibtex_key(
                data.get("authors") or CITATION_FALLBACK_DATA["authors"],  # type: ignore[arg-type]
                data.get("year") or CITATION_FALLBACK_DATA["year"],  # type: ignore[arg-type]
                data.get("title") or CITATION_FALLBACK_DATA["title"],  # type: ignore[arg-type]
            )
            if "docname" in data.get(
                "fields_to_overwrite_from_metadata",
                DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA,
            ):
                data["docname"] = data["key"]

        # even if we have a bibtex, it may not be complete, thus we need to add to it
        if not data.get("bibtex") or not cls.is_bibtex_complete(data["bibtex"]):
            existing_entry = None
            # if our bibtex already exists, but is incomplete, we add self_generated to metadata
            if data.get("bibtex"):
                if data.get("other"):
                    if (
                        "bibtex_source" in data["other"]
                        and "self_generated" not in data["other"]["bibtex_source"]
                    ):
                        data["other"]["bibtex_source"].append("self_generated")
                    else:
                        data["other"]["bibtex_source"] = ["self_generated"]
                else:
                    data["other"] = {"bibtex_source": ["self_generated"]}
                try:
                    existing_entry = next(
                        iter(Parser().parse_string(data["bibtex"]).entries.values())
                    )
                except PybtexSyntaxError:
                    logger.warning(f"Failed to parse bibtex for {data['bibtex']}.")
                    existing_entry = None

            entry_data = {
                "title": data.get("title") or CITATION_FALLBACK_DATA["title"],
                "year": (
                    CITATION_FALLBACK_DATA["year"]
                    if not data.get("year")
                    else str(data["year"])
                ),
                "journal": data.get("journal") or CITATION_FALLBACK_DATA["journal"],
                "volume": data.get("volume"),
                "pages": data.get("pages"),
                "month": (
                    None
                    if not (maybe_date := maybe_get_date(data.get("publication_date")))
                    else maybe_date.strftime("%b")
                ),
                "doi": data.get("doi"),
                "url": data.get("doi_url"),
                "publisher": data.get("publisher"),
                "issue": data.get("issue"),
                "issn": data.get("issn"),
            }
            entry_data = {k: v for k, v in entry_data.items() if v}
            try:
                new_entry = Entry(
                    data.get("bibtex_type", "article") or "article", fields=entry_data
                )
                if existing_entry:
                    new_entry = cls.merge_bibtex_entries(existing_entry, new_entry)
                # add in authors manually into the entry
                authors = [Person(a) for a in data.get("authors", ["Unknown authors"])]
                for a in authors:
                    new_entry.add_person(a, "author")
                data["bibtex"] = BibliographyData(
                    entries={data["key"]: new_entry}
                ).to_string("bibtex")
                # clear out the citation, since it will be regenerated
                if "citation" in data.get(
                    "fields_to_overwrite_from_metadata",
                    DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA,
                ):
                    data["citation"] = None
            except Exception:
                logger.warning(
                    "Failed to generate bibtex for"
                    f" {data.get('docname') or data.get('citation')}"
                )
        if data.get("citation") is None and data.get("bibtex") is not None:
            data["citation"] = format_bibtex(
                data["bibtex"], missing_replacements=CITATION_FALLBACK_DATA  # type: ignore[arg-type]
            )
        elif data.get("citation") is None:
            data["citation"] = data.get("title") or CITATION_FALLBACK_DATA["title"]
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_all_fields(cls, data: Mapping[str, Any]) -> dict[str, Any]:

        data = deepcopy(data)  # Avoid mutating input
        data = dict(data)
        if isinstance(data.get("fields_to_overwrite_from_metadata"), str):
            data["fields_to_overwrite_from_metadata"] = {
                s.strip()
                for s in data.get("fields_to_overwrite_from_metadata", "").split(",")
            }
        data = cls.lowercase_doi_and_populate_doc_id(data)
        data = cls.remove_invalid_authors(data)
        data = cls.misc_string_cleaning(data)
        data = cls.inject_clean_doi_url_into_data(data)
        data = cls.add_preprint_journal_from_doi_if_missing(data)
        data = cls.populate_bibtex_key_citation(data)
        return cls.overwrite_docname_dockey_for_compatibility_w_doc(data)

    def __getitem__(self, item: str):
        """Allow for dictionary-like access, falling back on other."""
        try:
            return getattr(self, item)
        except AttributeError:
            return self.other[item]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def formatted_citation(self) -> str:

        if self.is_retracted:
            base_message = "**RETRACTED ARTICLE**"
            retract_info = "Retrieved from http://retractiondatabase.org/."
            citation_message = (
                f"Citation: {self.citation}"
                if self.citation
                else f"Original DOI: {self.doi}"
            )
            return f"{base_message} {citation_message} {retract_info}"

        if self.citation_count is None or self.source_quality is None:
            logger.debug("citation_count and source_quality are not set.")
            return self.citation

        if self.source_quality_message:
            return (
                f"{self.citation} This article has {self.citation_count} citations and"
                f" is from a {self.source_quality_message}."
            )
        return f"{self.citation} This article has {self.citation_count} citations."

    @property
    def source_quality_message(self) -> str:
        return (
            SOURCE_QUALITY_MESSAGES[self.source_quality]
            if self.source_quality is not None
            and self.source_quality
            != DocDetails.UNDEFINED_JOURNAL_QUALITY  # note - zero is a valid value
            else ""
        )

    OPTIONAL_HYDRATION_FIELDS: ClassVar[Collection[str]] = {"url"}

    def is_hydration_needed(
        self,
        exclusion: Collection[str] = OPTIONAL_HYDRATION_FIELDS,
        inclusion: Collection[str] = (),
    ) -> bool:
        """Determine if we have unfilled attributes."""
        if inclusion:
            return any(
                v is None for k, v in self.model_dump().items() if k in inclusion
            )
        return any(
            v is None for k, v in self.model_dump().items() if k not in exclusion
        )

    def repopulate_doc_id_from_doi(self) -> None:
        # TODO: should this be a hash of the doi?
        if self.doi:
            self.doc_id = encode_id(self.doi)

    def __add__(self, other: DocDetails | int) -> DocDetails:  # noqa: PLR0912
        """Merge two DocDetails objects together."""
        # control for usage w. Python's sum() function
        if isinstance(other, int):
            return self

        # first see if one of the entries is newer, which we will prefer
        PREFER_OTHER = True
        if self.publication_date and other.publication_date:
            PREFER_OTHER = self.publication_date <= other.publication_date

        merged_data = {}
        for field in self.model_fields:
            self_value = getattr(self, field)
            other_value = getattr(other, field)

            if field == "other":
                # Merge 'other' dictionaries
                merged_data[field] = {**self.other, **other.other}
                # handle the bibtex / sources as special fields
                for field_to_combine in ("bibtex_source", "client_source"):
                    # Ensure the fields are lists before combining
                    if self.other.get(field_to_combine) and not isinstance(
                        self.other[field_to_combine], list
                    ):
                        self.other[field_to_combine] = [self.other[field_to_combine]]
                    if other.other.get(field_to_combine) and not isinstance(
                        other.other[field_to_combine], list
                    ):
                        other.other[field_to_combine] = [other.other[field_to_combine]]

                    if self.other.get(field_to_combine) and other.other.get(
                        field_to_combine
                    ):
                        # Note: these should always be lists
                        merged_data[field][field_to_combine] = (
                            self.other[field_to_combine] + other.other[field_to_combine]
                        )

            elif field == "authors" and self_value and other_value:
                # Combine authors lists, removing duplicates
                # Choose whichever author list is longer
                best_authors = (
                    self.authors
                    if (
                        sum(len(a) for a in (self.authors or []))
                        >= sum(len(a) for a in (other.authors or []))
                    )
                    else other.authors
                )
                merged_data[field] = best_authors or None  # type: ignore[assignment]

            elif field == "key" and self_value is not None and other_value is not None:
                # if we have multiple keys, we wipe them and allow regeneration
                merged_data[field] = None  # type: ignore[assignment]

            elif field in {"citation_count", "year", "publication_date"}:
                # get the latest data
                # this conditional is written in a way to handle if multiple doc objects
                # are provided, we'll use the highest value
                # if there's only one valid value, we'll use that regardless even if
                # that value is 0
                if self_value is None or other_value is None:
                    merged_data[field] = (
                        self_value
                        if self_value is not None  # Dance around 0
                        else other_value
                    )
                else:
                    merged_data[field] = max(self_value, other_value)

            else:
                # Prefer non-null values, default preference for 'other' object.
                # Note: if PREFER_OTHER = False then even if 'other' data exists
                # we will use 'self' data. This is to control for when we have
                # pre-prints / arXiv versions of papers that are not as up-to-date
                merged_data[field] = (
                    other_value
                    if (
                        (other_value is not None and other_value != []) and PREFER_OTHER
                    )
                    else self_value
                )

        # Recalculate doc_id if doi has changed
        if merged_data["doi"] != self.doi:
            merged_data["doc_id"] = (
                encode_id(merged_data["doi"].lower()) if merged_data["doi"] else None  # type: ignore[attr-defined,assignment]
            )

        # Create and return new DocDetails instance
        return DocDetails(**merged_data)

    def __radd__(self, other: DocDetails | int) -> DocDetails:
        # other == 0 captures the first call of sum()
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def __iadd__(self, other: DocDetails | int) -> DocDetails:  # noqa: PYI034
        # only includes int to align with __radd__ and __add__
        if isinstance(other, int):
            return self
        return self.__add__(other)
