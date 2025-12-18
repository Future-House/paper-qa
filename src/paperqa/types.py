from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import json
import logging
import os
import re
from collections.abc import Collection, Container, Hashable, Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from random import Random
from typing import Annotated, Any, ClassVar, Self, cast
from uuid import UUID, uuid4

import tiktoken
from aviary.core import Message
from lmi import Embeddable, LLMResult
from lmi.utils import bytes_to_string, encode_image_as_url, string_to_bytes
from pybtex.database import BibliographyData, Entry, InvalidNameString, Person
from pybtex.database.input.bibtex import Parser
from pybtex.scanner import PybtexSyntaxError
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

from paperqa.utils import (
    compute_unique_doc_id,
    create_bibtex_key,
    encode_id,
    format_bibtex,
    get_citation_ids,
    get_parenthetical_substrings,
    maybe_get_date,
    md5sum,
)
from paperqa.version import __version__ as pqa_version

# Just for clarity
# also in case one day we want to narrow
# the type
DocKey = Any
logger = logging.getLogger(__name__)


# These probably should be promoted to be on DocDetails
# but this will take a larger refactor.
VAR_MATCH_LOOKUP: Collection[str] = {"1", "true"}
VAR_MISMATCH_LOOKUP: Collection[str] = {"0", "false"}
DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA: Collection[str] = {
    "key",
    "doc_id",
    "docname",
    "dockey",
    "citation",
    "content_hash",  # Metadata providers won't give this
}
# Sentinel to autopopulate a field within model_validator
AUTOPOPULATE_VALUE = ""  # NOTE: this is falsy by design


class Doc(Embeddable):
    model_config = ConfigDict(extra="forbid")

    docname: str
    dockey: DocKey
    citation: str
    content_hash: str | None = Field(
        default=AUTOPOPULATE_VALUE,
        description=(
            "Optional hash of the document's contents (to reiterate, not a file path to"
            " the document, but the document's contents itself)."
        ),
    )
    # Sort the serialization to minimize the diff of serialized objects
    fields_to_overwrite_from_metadata: Annotated[set[str], PlainSerializer(sorted)] = (
        Field(
            default_factory=lambda: set(DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA),
            description="fields from metadata to overwrite when upgrading to a DocDetails",
        )
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

    FIELDS_TO_EXCLUDE_FROM_CSV: ClassVar[set[str]] = {
        "embedding",  # Don't store to allow for configuration of embedding models
    }
    CSV_FIELDS_UP_FRONT: ClassVar[Sequence[str]] = ()

    @classmethod
    def to_csv(cls, values: Iterable[Self], target_csv_path: str | os.PathLike) -> None:
        """Dump many instances into a CSV, for later use as a manifest."""
        headers = set(cls.model_fields) - cls.FIELDS_TO_EXCLUDE_FROM_CSV
        with open(target_csv_path, "w", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    *sorted(cls.CSV_FIELDS_UP_FRONT),  # Make easy reading
                    *sorted(headers - set(cls.CSV_FIELDS_UP_FRONT)),
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    v.model_dump(
                        exclude={"formatted_citation"} | cls.FIELDS_TO_EXCLUDE_FROM_CSV
                    )
                    for v in values
                ]
            )


class Text(Embeddable):
    """A text chunk ready for use in retrieval with a linked document."""

    # We allow extras so one can introduce chunk-specific metadata
    model_config = ConfigDict(extra="allow")

    text: str = Field(description="Processed text content of the chunk.")
    name: str = Field(
        description=(
            "Human-readable identifier for the chunk"
            " (e.g., 'Wiki2023 chunk 1', 'sentence1')."
        )
    )
    media: list[ParsedMedia] = Field(
        default_factory=list, description="Optional list of associated media."
    )
    doc: Doc | DocDetails = Field(
        union_mode="left_to_right",
        description="Source document this text chunk originates from.",
    )

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        # We ignore the embedding since the embedding can:
        # - Be lazily acquired, or not used (depending on settings)
        # - Get ditched when serializing a text for an HTTP request
        return (
            self.name == other.name
            and self.text == other.text
            and self.media == other.media
            and self.doc == other.doc
            and self.__pydantic_extra__ == other.__pydantic_extra__
        )

    def __hash__(self) -> int:
        if self.__pydantic_extra__:
            unhashable = [
                k
                for k, v in self.__pydantic_extra__.items()
                if not isinstance(v, Hashable)
            ]
            if unhashable:
                raise NotImplementedError(
                    f"Hashing a {type(self).__name__} with unhashable extras"
                    " is not yet supported."
                )
            # As Python dict equality (used in __eq__) is order independent,
            # let's go ahead and be order independent in __hash__ too for consistency
            extras = tuple(sorted(self.__pydantic_extra__.items()))
            return hash((self.name, self.text, tuple(self.media), extras))
        return hash((self.name, self.text, tuple(self.media)))

    async def get_embeddable_text(self, with_enrichment: bool = False) -> str:
        """Get the text to embed, which may be different from the actual text content.

        This method, despite currently not involving any awaits,
        is async so subclassers can have custom just-in-time enrichment logic
        or fetch enrichments from an external service.

        Args:
            with_enrichment: Opt-in flag to include media enrichment in the return.
                Media enrichment can improve placement in embedding space,
                without affecting the text used for quotation.

        Returns:
            Content to embed.
        """
        if not with_enrichment:
            return self.text
        # Media enrichment can improve placement in embedding space,
        # without affecting the text used for quotation
        enriched_media = (
            (
                f"Media {m.index} from page {m.info.get('page_num', 'unknown')!s}'s"
                f" enriched description:\n\n{m.info['enriched_description']!s}"
            )
            for m in self.media
            if m.info.get("enriched_description")
        )
        return "\n\n".join((self.text, *enriched_media))


class Context(BaseModel):
    """A class to hold the context of a question."""

    # We allow extras so one can extend the summary JSON prompt
    # to have the LLM answer with more conclusions such as alternate scores
    # or useful excerpts of text
    model_config = ConfigDict(extra="allow")

    # Value was chosen to be below a 0-10 scale, making the 'unset' nature obvious
    UNSET_RELEVANCE: ClassVar[int] = -1

    id: str = Field(
        default=AUTOPOPULATE_VALUE,
        description="Unique identifier for the context. Auto-generated if not provided.",
    )

    context: Annotated[str, StringConstraints(strip_whitespace=True)] = Field(
        description=(
            "Summary of the text with respect to a question."
            " Can be an empty string if a summary is not useful/irrelevant"
            " (which should be paired with a score of 0)."
        )
    )
    question: str | None = Field(
        default=None,
        description=(
            "Question that the context is summarizing for. "
            "Note this can differ from the user query."
        ),
    )
    text: Text
    score: int = Field(
        default=UNSET_RELEVANCE,
        description=(
            "Relevance score for this context to the question."
            " The range used here is 0-10, where 0 is 'irrelevant',"
            " 1 is barely relevant, and 10 is most relevant."
            " The default is -1 to have a 'sorting safe' default as sub-relevant."
        ),
    )

    CONTEXT_ENCODING_LENGTH: ClassVar[int] = 500  # chars
    ID_HASH_LENGTH: ClassVar[int] = 8  # chars
    # pqac stands for "paper qa context"
    REFERENCE_TEMPLATE: ClassVar[str] = "pqac-{id}"

    def __str__(self) -> str:
        """Return the context as a string."""
        return self.context

    def __hash__(self) -> int:
        extras = (
            tuple(
                sorted(
                    (k, v)
                    for k, v in self.__pydantic_extra__.items()
                    if isinstance(v, Hashable)  # Don't consider unhashable extras
                )
            )
            if self.__pydantic_extra__
            else ()
        )
        return hash(
            (self.id, self.question, self.context, self.text, self.score, extras)
        )

    @model_validator(mode="before")
    @classmethod
    def populate_id(cls, data: dict[str, Any]) -> dict[str, Any]:
        if not data.get("id"):  # NOTE: this includes missing or empty strings
            content = (data.get("question") or "") + data.get("context", "")[
                : cls.CONTEXT_ENCODING_LENGTH
            ]
            return data | {  # Avoid mutating input data
                "id": cls.REFERENCE_TEMPLATE.format(
                    id=encode_id(content or str(uuid4()), maxsize=cls.ID_HASH_LENGTH)
                )
            }
        return data


class PQASession(BaseModel):
    """A class to hold session about researching/answering."""

    # Keys in the other field to not remove when filtering for user display
    DOC_DETAILS_OTHERS_TO_KEEP: ClassVar[Collection[str]] = {
        "bibtex_source",
        "client_source",
    }

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: UUID = Field(default_factory=uuid4)
    question: str
    answer: str = ""
    raw_answer: str = Field(
        default="",
        description="Raw answer from the LLM, including context IDs.",
    )
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
        return {c.id for c in self.contexts if c.id in self.raw_answer}

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

        prompt_count = result.prompt_count or 0
        completion_count = result.completion_count or 0
        if result.model not in self.token_counts:
            self.token_counts[result.model] = [prompt_count, completion_count]
        else:
            self.token_counts[result.model][0] += prompt_count
            self.token_counts[result.model][1] += completion_count

        self.cost += result.cost

    def get_unique_docs_from_contexts(self, score_threshold: int = 0) -> set[Doc]:
        """Parse contexts for docs with scores above the input threshold."""
        return {
            c.text.doc
            for c in filter(lambda x: x.score >= score_threshold, self.contexts)
        }

    def filter_content_for_user(self) -> None:
        """
        In-place filter/drop items that are irrelevant to the user.

        This is mainly done to keep HTTP requests reasonably sized.
        """
        self.contexts = [
            Context(
                # Dump all fields from the original context (including extras),
                # but exclude 'text' so we can replace it below.
                **c.model_dump(exclude={"text"}),
                text=Text(
                    text="",
                    # Similar to the explanation in `map_fxn_summary`'s internals
                    # on why we drop embeddings, drop embeddings here too because
                    # embeddings aren't displayed to front end users
                    doc=c.text.doc.model_dump(exclude={"embedding"}),
                    # We drop media since images can be quite large
                    **c.text.model_dump(exclude={"text", "embedding", "doc", "media"}),
                ),
            )
            for c in self.contexts
        ]
        # Now we drop extras from other fields
        for c in self.contexts:
            if isinstance(c.text.doc, DocDetails):
                c.text.doc.other = {
                    k: v
                    for k, v in c.text.doc.other.items()
                    if k in self.DOC_DETAILS_OTHERS_TO_KEEP
                }

    def populate_formatted_answers_and_bib_from_raw_answer(
        self,
    ) -> None:
        """Format a raw answer for display, mutating the session in place."""
        formatted_without_references = self.raw_answer

        id_to_name_map = {c.id: c.text.name for c in self.contexts}
        name_to_citation_map = {
            c.text.name: c.text.doc.formatted_citation for c in self.contexts
        }
        name_bib = {}

        for parenthetical in get_parenthetical_substrings(formatted_without_references):
            # now we replace eligible parentheticals with the deduped names
            # while we preserve order and deduplicate
            deduped_names: dict[str, None] = dict.fromkeys(
                id_to_name_map[key]
                for key in get_citation_ids(parenthetical)
                if id_to_name_map.get(key)
            )
            if deduped_names:
                formatted_without_references = formatted_without_references.replace(
                    parenthetical,
                    f"({', '.join(deduped_names)})",
                )
                for deduped_name in deduped_names:
                    if (
                        deduped_name in name_to_citation_map
                        and deduped_name not in name_bib
                    ):
                        name_bib[deduped_name] = name_to_citation_map[deduped_name]

        bib = "\n\n".join(
            [f"{i + 1}. ({k}): {c}" for i, (k, c) in enumerate(name_bib.items())]
        )

        # strip out any leftover hallucinated citations
        included_keys = get_citation_ids(self.raw_answer)
        for hallucinated_key in set(included_keys) - set(id_to_name_map):
            formatted_without_references = formatted_without_references.replace(
                hallucinated_key, ""
            )

        formatted_with_references = (
            f"Question: {self.question}\n\n{formatted_without_references}"
        )

        if bib:
            formatted_with_references += f"\n\nReferences\n\n{bib}"

        self.answer = formatted_without_references
        self.formatted_answer = formatted_with_references
        self.references = bib


class ChunkMetadata(BaseModel):
    """Metadata for chunking algorithm."""

    size: int = Field(description="Chunk size (chars), or 0 for no chunking.")
    overlap: int = Field(description="Chunk overlap (chars), or 0 for no overlap.")
    name: str | None = Field(
        default=None,
        description=(
            "Optional string summarizing the chunking parameters, embodying a hash."
        ),
    )


class ParsedMetadata(BaseModel):
    """Metadata for parsed text."""

    parsing_libraries: list[str] = Field(
        description="Libraries used to generate the parsing."
    )
    paperqa_version: str = Field(
        default=pqa_version,
        description="PaperQA version that invoked the parsing_libraries.",
    )
    total_parsed_text_length: int = Field(
        ge=0, description="Length (chars) of the parsed text."
    )
    count_parsed_media: int = Field(default=0, ge=0)
    name: str | None = Field(
        default=None,
        description=(
            "Optional string summarizing the parsing parameters, embodying a hash."
        ),
    )
    chunk_metadata: ChunkMetadata | None = Field(
        default=None, description="Optional metadata from the chunking process."
    )


class ParsedMedia(BaseModel):
    """Raw image or table parsed from a document's page."""

    index: int = Field(
        description="Index of the image in a given page, or 0 if solely an image."
    )
    data: Annotated[
        bytes,
        PlainSerializer(bytes_to_string),
        BeforeValidator(lambda x: x if isinstance(x, bytes) else string_to_bytes(x)),
    ] = Field(description="Raw image, ideally directly savable to a PNG image.")
    text: str | None = Field(
        default=None,
        description=(
            "Optional associated text content (e.g. markdown export of a table)."
            " This should not be enriched/augmented text, but actual text."
        ),
    )
    info: dict[str, JsonValue | tuple[float, ...] | bytes] = Field(
        default_factory=dict,
        description=(
            "Optional image metadata. This may come from image definitions sourced from"
            " the PDF, attributes of custom pixel maps, or what the PDF reader"
            " considered the media to be (e.g. table or image). It may also include"
            " model-generated description(s) of the image."
        ),
    )

    def _get_info_hashable(self) -> Hashable:
        if info_hashable := self.info.get("info_hashable"):
            return cast(Hashable, info_hashable)
        # We know info_hashable_hash key isn't present, so no need to filter it
        return json.dumps(self.info, sort_keys=True)

    def __hash__(self) -> int:
        return hash((self.index, self.data, self.text, self._get_info_hashable()))

    def to_id(self) -> UUID:
        """Convert this media to a UUID4 suitable for a database ID."""
        # We only hash the image and text content, since we don't want
        # minor parsing details (e.g. inconsequentially-small decimal values
        # in bounding boxes) to change the resultant ID
        to_hash: bytes = (
            self.data if not self.text else self.data + self.text.encode("utf-8")
        )
        seed_hash = hashlib.sha256(to_hash).hexdigest()
        seed_uint32 = int(seed_hash, 16) % (2**32)

        # Convert uint32 to UUID4
        uuid_int = Random(seed_uint32).getrandbits(128)
        uuid_int &= ~(0xF << 76)  # Clear version bits
        uuid_int |= 0x4 << 76  # Then set version to 4
        uuid_int &= ~(0x3 << 62)  # Clear variant bits
        uuid_int |= 0x2 << 62  # Then set variant to 10 for RFC 4122
        return UUID(int=uuid_int)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ParsedMedia):
            return NotImplemented
        return (
            self.index == other.index
            and self.data == other.data
            and self.text == other.text
            and self._get_info_hashable() == other._get_info_hashable()
        )

    def to_image_url(self) -> str:
        """Convert the image data to an RFC 2397 data URL format."""
        image_type = cast(str, self.info.get("suffix", "png")).removeprefix(".")
        if image_type == "jpg":  # SEE: https://stackoverflow.com/a/54488403
            image_type = "jpeg"
        return encode_image_as_url(image_type, self.data)

    def save(self, path: str | os.PathLike) -> None:
        """Save the image to the input file path."""
        with Path(path).open("wb") as f:
            f.write(self.data)


class ParsedText(BaseModel):
    """All text from a document read, before chunking."""

    content: (
        dict[str, str] | str | list[str] | dict[str, tuple[str, list[ParsedMedia]]]
    ) = Field(
        description=(
            "All parsed but not further processed (e.g. not chunked) contents from a"
            " document. It may be structured, depending on the parser's implementation."
            " Thus it can take various shapes depending on the document type"
            " (e.g. PDF, HTML) and parser:"
            "\n- `dict[str, str]` (e.g. page number -> page text) for PDFs."
            "\n- `str` for text files."
            "\n- `list[str]` for line-by-line parsings."
            "\n- `dict[str, tuple[str, list[ParsedMedia]]]` (e.g. page number"
            " -> (page text, page images)) for PDFs."
        )
    )
    metadata: ParsedMetadata = Field(
        description="Metadata on the parsing process used."
    )

    def encode_content(
        self, enc: tiktoken.Encoding | str = "cl100k_base"
    ) -> list[int] | list[list[int]]:
        if isinstance(enc, str):
            enc = tiktoken.get_encoding(enc)
        if isinstance(self.content, str):
            return enc.encode_ordinary(self.content)
        if isinstance(self.content, list):
            return [enc.encode_ordinary(c) for c in self.content]
        raise NotImplementedError(
            "Encoding only implemented for str and list[str] content,"
            f" not {type(self.content)}."
        )

    def reduce_content(self) -> str:
        """Reduce any content to a string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n\n".join(self.content)
        return "\n\n".join(
            x[0] if not isinstance(x, str) else x for x in self.content.values()
        )


class BibTeXSource(StrEnum):
    """Possible BibTeX sources."""

    # This source is used when the BibTeX is incomplete or missing,
    # and means we generated the BibTeX ourselves
    SELF_GENERATED = "self_generated"
    CROSSREF = "crossref"
    SEMANTIC_SCHOLAR = "semantic_scholar"

    def update_other(self, other: dict[str, Any] | None = None) -> dict[str, Any]:
        """Update an 'other' data dictionary to include this BibTeX source."""
        if not other:
            return {"bibtex_source": [self.value]}
        if "bibtex_source" in other:
            if self.value not in other["bibtex_source"]:
                other["bibtex_source"].append(self.value)
        else:
            other["bibtex_source"] = [self.value]
        return other


# We use these integer values
# as defined in https://jfp.csc.fi/en/web/haku/kayttoohje
# which is a recommended ranking system
SOURCE_QUALITY_MESSAGES = {
    0: "poor quality or predatory journal",
    1: "peer-reviewed journal",
    2: "domain leading peer-reviewed journal",
    3: "highest quality peer-reviewed journal",
}

CITATION_FALLBACK_DATA: dict[str, str | list[str]] = {
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
    doc_id: str | None = Field(
        default=None,
        description=(
            "Unique ID for this document. A simple and robust way to acquire one is"
            " hashing the paper content's hash concatenate with the lowercased DOI."
        ),
    )
    file_location: str | os.PathLike | None = Field(
        default=None,
        description=(
            "Optional path to the stored paper (local file path"
            " or mountable cloud path), or a database ID for the storage location."
        ),
    )
    license: str | None = Field(
        default=None,
        description=(
            "string indicating license. Should refer specifically to pdf_url (since"
            " that could be preprint). None means unknown/unset."
        ),
    )
    pdf_url: str | None = Field(
        default=None, description="URL to the PDF of the paper, if known."
    )
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
    FIELDS_TO_EXCLUDE_FROM_CSV: ClassVar[set[str]] = {
        "bibtex",  # Let this be autogenerated, to avoid dealing with newlines
        "embedding",  # Don't store to allow for configuration of embedding models
    }
    CSV_FIELDS_UP_FRONT: ClassVar[Sequence[str]] = ("doi", "file_location")

    @field_validator("key")
    @classmethod
    def clean_key(cls, value: str) -> str:
        # Replace HTML tags with empty string
        return re.sub(pattern=r"<\/?\w{1,10}>", repl="", string=value)

    @field_validator("publication_date")
    @classmethod
    def add_tzinfo(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            return value.replace(tzinfo=UTC)  # Assume UTC if unspecified
        return value.astimezone(UTC)  # Convert to UTC

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
            if not data.get("doc_id"):  # keep user defined doc_ids
                data["doc_id"] = compute_unique_doc_id(doi, data.get("content_hash"))
        elif not data.get("doc_id"):  # keep user defined doc_ids
            data["doc_id"] = compute_unique_doc_id(doi, data.get("content_hash"))

        if "dockey" in data.get(
            "fields_to_overwrite_from_metadata",
            DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA,
        ) and ("dockey" not in data or not data["dockey"]):
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
            data["pages"] = pages.replace("--", "-").replace(" ", "").strip()
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
    def populate_bibtex_key_citation(cls, data: dict[str, Any]) -> dict[str, Any]:
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
                data["other"] = BibTeXSource.SELF_GENERATED.update_other(
                    data.get("other")
                )
                try:
                    existing_entry = next(
                        iter(Parser().parse_string(data["bibtex"]).entries.values())
                    )
                except (PybtexSyntaxError, InvalidNameString) as exc:
                    # InvalidNameString: names like "Kyriacos, Κυριάκος, Athanasiou, Αθανασίου"
                    logger.warning(
                        f"Failed to parse bibtex for DOI {data.get('doi')},"
                        f" title {data.get('title')!r}, and bibtex {data['bibtex']}."
                        f" Failure message: {exc!r}"
                    )
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
                # We consider the source self-generated because the 'key' gets
                # autogenerated above via `create_bibtex_key` if it wasn't present
                data["other"] = BibTeXSource.SELF_GENERATED.update_other(
                    data.get("other")
                )
                # clear out the citation, since it will be regenerated
                if "citation" in data.get(
                    "fields_to_overwrite_from_metadata",
                    DEFAULT_FIELDS_TO_OVERWRITE_FROM_METADATA,
                ):
                    data["citation"] = None
            except Exception:
                logger.debug(
                    "Failed to generate bibtex for"
                    f" {data.get('docname') or data.get('citation')}"
                )
        if data.get("citation") is None and data.get("bibtex") is not None:
            data["citation"] = format_bibtex(
                data["bibtex"], missing_replacements=CITATION_FALLBACK_DATA
            )
        elif data.get("citation") is None:
            data["citation"] = data.get("title") or CITATION_FALLBACK_DATA["title"]
        return data

    @classmethod
    def populate_content_hash(cls, data: dict[str, Any]) -> dict[str, Any]:
        if (  # Check for missing or autopopulate value, but preserve `None`
            data.get("content_hash", AUTOPOPULATE_VALUE) == AUTOPOPULATE_VALUE
        ):
            data["content_hash"] = None  # Assume we don't have it
            if data.get("file_location"):  # Try to update it
                with contextlib.suppress(FileNotFoundError):
                    data["content_hash"] = md5sum(data["file_location"])
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_all_fields(cls, data: Mapping[str, Any]) -> dict[str, Any]:

        data = deepcopy(data)  # Avoid mutating input
        data = dict(data)
        if "fields_to_overwrite_from_metadata" in data:
            raw_value = data["fields_to_overwrite_from_metadata"]
            if isinstance(raw_value, str):
                if (raw_value[0], raw_value[-1]) in {("[", "]"), ("{", "}")}:
                    # If string-ified set or list, remove brackets before split
                    raw_value = raw_value[1:-1]
                data["fields_to_overwrite_from_metadata"] = {
                    s.strip("\"' ") for s in raw_value.split(",")
                }
            if not isinstance(data["fields_to_overwrite_from_metadata"], Container):
                raise TypeError(
                    "fields_to_overwrite_from_metadata should be a container,"
                    f" not {type(data['fields_to_overwrite_from_metadata'])}."
                )
        for possibly_str_field in ("authors", "other"):
            if data.get(possibly_str_field) and isinstance(
                data[possibly_str_field], str
            ):
                data[possibly_str_field] = ast.literal_eval(data[possibly_str_field])
        data = cls.populate_content_hash(data)
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

    def make_filename(self, title_limit: int | None = 48) -> str:
        """
        Make a filesystem-safe filename that has the doc ID appended, but no extension.

        Args:
            title_limit: Character limit on the title.

        Returns:
            Filename that is filesystem safe (e.g. non-safe chars are replaced with dash).
        """
        if not self.title or not self.doc_id:
            raise ValueError("Unable to create filename without both title and doc_id.")
        # SEE: https://stackoverflow.com/a/71199182
        encoded_title = re.sub(
            r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", self.title[:title_limit]
        )
        # NOTE: we append the doc ID for a few reasons:
        # 1. Prevent collisions for identical titles
        #    SEE: https://stackoverflow.com/a/71761675
        # 2. Filenames shouldn't end in a period,
        #    so append the doc ID to circumvent that gotcha
        return "_".join((encoded_title, self.doc_id))

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

    def __add__(self, other: DocDetails | int) -> DocDetails:  # noqa: PLR0912
        """Merge two DocDetails objects together."""
        # control for usage w. Python's sum() function
        if isinstance(other, int):
            return self

        # first see if one of the entries is newer, which we will prefer
        PREFER_OTHER = True
        if self.publication_date and other.publication_date:
            PREFER_OTHER = self.publication_date <= other.publication_date

        merged_data: dict[str, Any] = {}
        # pylint: disable-next=not-an-iterable  # pylint bug: https://github.com/pylint-dev/pylint/issues/10144
        for field in type(self).model_fields:
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
                        sum(len(a) for a in self.authors or [])
                        >= sum(len(a) for a in other.authors or [])
                    )
                    else other.authors
                )
                merged_data[field] = best_authors or None

            elif field == "key" and self_value is not None and other_value is not None:
                # if we have multiple keys, we wipe them and allow regeneration
                merged_data[field] = None

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
            elif field == "content_hash" and (
                # Hashes are both present but differ
                self_value
                and other_value
                and self_value != other_value
            ):
                # If hashes are both present but differ,
                # we don't know which to pick, so just discard the value
                merged_data[field] = None

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

        if (
            merged_data["doi"] != self.doi
            or merged_data["content_hash"] != self.content_hash
        ):
            # Recalculate doc_id if doi or content hash has changed
            merged_data["doc_id"] = compute_unique_doc_id(
                merged_data["doi"], merged_data.get("content_hash")
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
